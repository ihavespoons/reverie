// Command reverie is the MCP server entrypoint for the reverie memory system.
// It provides subcommands for serving the MCP protocol over stdio and querying
// memory status.
package main

import (
	"context"
	"fmt"
	"log/slog"
	"os"
	"os/signal"
	"path/filepath"
	"syscall"

	"personal/reverie/internal/cluster"
	"personal/reverie/internal/config"
	"personal/reverie/internal/db"
	"personal/reverie/internal/decay"
	"personal/reverie/internal/embed"
	"personal/reverie/internal/manager"
	"personal/reverie/internal/mcpserver"
	"personal/reverie/internal/memory"
	"personal/reverie/internal/migrate"
)

const usage = `reverie - persistent memory system for coding agents

Usage:
  reverie serve       Start the MCP server over stdio (default)
  reverie import      Migrate Claude Code auto-memory files into reverie
  reverie status      Show memory counts, DB size, and configuration
  reverie view        List memories in a formatted table
  reverie forget <id> Delete a memory by ID (with confirmation)
  reverie reindex     Re-embed all memories (after switching models)
  reverie help        Show this help message

Import flags:
  --project-dir <path>  Import a single memory directory
  --all-projects        Import all projects (default)

View flags:
  --layer l2|l3         Memory layer to list (default: l2)
  --subtype TYPE        Filter by subtype (user, feedback, project, reference)
  --limit N             Max results (default: 50, max: 1000)

Forget flags:
  --force               Skip confirmation prompt

Reindex flags:
  --dry-run             Show counts without updating

Environment:
  VOYAGE_API_KEY       API key for Voyage AI embedding provider (voyage provider)
  OPENAI_API_KEY       API key for OpenAI-compatible embeddings (optional for Ollama/LM Studio)
  REVERIE_EMBED_URL    Override embedding base URL (e.g. http://localhost:11434/v1)
  REVERIE_EMBED_MODEL  Override embedding model name
  REVERIE_DB_PATH      Override database path
  REVERIE_CONFIG       Override config file path
  REVERIE_LOG_LEVEL    Set log level (debug, info, warn, error)
  REVERIE_DISABLED     Set to 1 to disable all tools (return stub responses)
`

func main() {
	cmd := "help"
	if len(os.Args) > 1 {
		cmd = os.Args[1]
	} else if len(os.Args) == 1 {
		// No subcommand: default to serve if stdin is not a terminal,
		// otherwise show usage. MCP clients pipe stdin, so this heuristic
		// lets "reverie" work as both a user command and an MCP spawn.
		cmd = "help"
	}

	switch cmd {
	case "serve":
		if err := runServe(); err != nil {
			fmt.Fprintf(os.Stderr, "reverie serve: %v\n", err)
			os.Exit(1)
		}
	case "import":
		if err := runImport(); err != nil {
			fmt.Fprintf(os.Stderr, "reverie import: %v\n", err)
			os.Exit(1)
		}
	case "status":
		if err := runStatus(); err != nil {
			fmt.Fprintf(os.Stderr, "reverie status: %v\n", err)
			os.Exit(1)
		}
	case "view":
		if err := runView(); err != nil {
			fmt.Fprintf(os.Stderr, "reverie view: %v\n", err)
			os.Exit(1)
		}
	case "forget":
		if err := runForget(); err != nil {
			fmt.Fprintf(os.Stderr, "reverie forget: %v\n", err)
			os.Exit(1)
		}
	case "reindex":
		if err := runReindex(); err != nil {
			fmt.Fprintf(os.Stderr, "reverie reindex: %v\n", err)
			os.Exit(1)
		}
	default:
		fmt.Fprint(os.Stderr, usage)
		if cmd != "help" && cmd != "-h" && cmd != "--help" {
			os.Exit(1)
		}
	}
}

func runServe() error {
	cfg, err := loadConfig()
	if err != nil {
		return err
	}

	logger := newLogger(cfg.Server.LogLevel)

	// Ensure the database directory exists.
	dbDir := filepath.Dir(cfg.Storage.DBPath)
	if err := os.MkdirAll(dbDir, 0o755); err != nil {
		return fmt.Errorf("create db directory %q: %w", dbDir, err)
	}

	database, err := db.Open(cfg.Storage.DBPath)
	if err != nil {
		return fmt.Errorf("open database: %w", err)
	}
	defer database.Close()

	store := memory.NewSQLiteStore(database)

	// Build the embedding provider chain based on configured provider.
	inner, err := buildEmbedder(cfg)
	if err != nil {
		return fmt.Errorf("build embedder: %w", err)
	}
	embedder := embed.NewCachedProvider(inner, database)

	// Build the decay and memory management components.
	dec := decay.NewDecayer(cfg.Decay.Temperature, cfg.Memory.RetentionThreshold)
	mgr := manager.NewMemoryManager(store, dec, cfg.Decay.UtilityAlpha, cfg.Decay.FrequencyBeta)

	// Build the cluster assigner for online nearest-centroid assignment.
	assigner := cluster.NewAssigner(store, cfg.Cluster.MinSimilarityForAssignment, cfg.Decay.ColdStartUtility, cfg.Decay.ColdStartFrequency)

	srv := mcpserver.NewServer(store, embedder, dec, mgr, assigner, cfg, logger)

	// Signal handling for graceful shutdown.
	ctx, cancel := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer cancel()

	logger.Info("reverie serve starting",
		"db_path", cfg.Storage.DBPath,
		"embedding_model", cfg.Embedding.Model,
		"disabled", cfg.Server.Disabled,
	)

	return srv.Run(ctx)
}

func runStatus() error {
	cfg, err := loadConfig()
	if err != nil {
		return err
	}

	// Check if the DB file exists.
	if _, err := os.Stat(cfg.Storage.DBPath); os.IsNotExist(err) {
		fmt.Fprintf(os.Stderr, "Database not found at %s\n", cfg.Storage.DBPath)
		fmt.Fprintf(os.Stderr, "Run 'reverie serve' first to initialize the database.\n")
		return nil
	}

	database, err := db.Open(cfg.Storage.DBPath)
	if err != nil {
		return fmt.Errorf("open database: %w", err)
	}
	defer database.Close()

	store := memory.NewSQLiteStore(database)

	ctx := context.Background()

	// Count all facts.
	allFacts, err := store.ListFacts(ctx, memory.ListFilter{Limit: 1000})
	if err != nil {
		return fmt.Errorf("list facts: %w", err)
	}

	// Count by subtype.
	subtypeCounts := make(map[string]int)
	for _, f := range allFacts {
		key := f.Subtype
		if key == "" {
			key = "(untyped)"
		}
		subtypeCounts[key]++
	}

	// Get DB file size.
	var dbSize int64
	if fi, err := os.Stat(cfg.Storage.DBPath); err == nil {
		dbSize = fi.Size()
	}

	fmt.Println("reverie status")
	fmt.Println("==============")
	fmt.Printf("Database: %s\n", cfg.Storage.DBPath)
	fmt.Printf("DB size:  %s\n", formatBytes(dbSize))
	fmt.Println()
	fmt.Printf("L2 semantic facts: %d\n", len(allFacts))
	for subtype, count := range subtypeCounts {
		fmt.Printf("  %-12s %d\n", subtype, count)
	}
	fmt.Println()
	fmt.Println("L1 clusters:  (Phase 3)")
	fmt.Println("L3 episodes:  (Phase 3)")
	fmt.Println()
	fmt.Printf("Embedding model: %s\n", cfg.Embedding.Model)
	fmt.Printf("Disabled:        %v\n", cfg.Server.Disabled)

	return nil
}

func runImport() error {
	cfg, err := loadConfig()
	if err != nil {
		return err
	}

	// Ensure the database directory exists.
	dbDir := filepath.Dir(cfg.Storage.DBPath)
	if err := os.MkdirAll(dbDir, 0o755); err != nil {
		return fmt.Errorf("create db directory %q: %w", dbDir, err)
	}

	database, err := db.Open(cfg.Storage.DBPath)
	if err != nil {
		return fmt.Errorf("open database: %w", err)
	}
	defer database.Close()

	store := memory.NewSQLiteStore(database)

	inner, err := buildEmbedder(cfg)
	if err != nil {
		return fmt.Errorf("build embedder: %w", err)
	}
	embedder := embed.NewCachedProvider(inner, database)

	assigner := cluster.NewAssigner(store, cfg.Cluster.MinSimilarityForAssignment, cfg.Decay.ColdStartUtility, cfg.Decay.ColdStartFrequency)

	logger := func(format string, args ...any) {
		fmt.Fprintf(os.Stderr, format, args...)
	}

	imp := migrate.NewImporter(store, embedder, assigner, logger)
	ctx := context.Background()

	// Parse flags: --project-dir <path> or --all-projects (default).
	var projectDir string
	args := os.Args[2:] // skip "reverie" and "import"
	for i := 0; i < len(args); i++ {
		switch args[i] {
		case "--project-dir":
			if i+1 >= len(args) {
				return fmt.Errorf("--project-dir requires a path argument")
			}
			i++
			projectDir = args[i]
		case "--all-projects":
			// Explicit default; nothing to set.
		default:
			return fmt.Errorf("unknown flag %q (expected --project-dir <path> or --all-projects)", args[i])
		}
	}

	var result *migrate.ImportResult
	if projectDir != "" {
		result, err = imp.ImportDir(ctx, projectDir)
	} else {
		home, homeErr := os.UserHomeDir()
		if homeErr != nil {
			return fmt.Errorf("determine home directory: %w", homeErr)
		}
		claudeDir := filepath.Join(home, ".claude", "projects")
		result, err = imp.ImportAllProjects(ctx, claudeDir)
	}
	if err != nil {
		return err
	}

	fmt.Fprintf(os.Stderr, "\nImport complete: %d files scanned, %d facts created, %d skipped\n",
		result.FilesScanned, result.FactsCreated, result.Skipped)
	if len(result.Errors) > 0 {
		fmt.Fprintf(os.Stderr, "Errors (%d):\n", len(result.Errors))
		for _, e := range result.Errors {
			fmt.Fprintf(os.Stderr, "  - %s\n", e)
		}
	}

	return nil
}

// buildEmbedder constructs the inner embedding provider based on cfg.Embedding.Provider.
// The caller is expected to wrap the result in a CachedProvider.
func buildEmbedder(cfg *config.Config) (embed.EmbeddingProvider, error) {
	switch cfg.Embedding.Provider {
	case "", "openai_compat":
		if cfg.Embedding.BaseURL == "" {
			return nil, fmt.Errorf("embedding.base_url is required for openai_compat provider")
		}
		if cfg.Embedding.Model == "" {
			return nil, fmt.Errorf("embedding.model is required")
		}
		return embed.NewOpenAICompatProvider(
			cfg.Embedding.BaseURL,
			cfg.Embedding.APIKey,
			cfg.Embedding.Model,
			cfg.Embedding.BatchSize,
			cfg.Embedding.Dimensions,
		), nil
	case "voyage":
		key := cfg.Embedding.APIKey
		if key == "" {
			key = os.Getenv("VOYAGE_API_KEY")
		}
		return embed.NewVoyageProvider(key, cfg.Embedding.Model, cfg.Embedding.BatchSize), nil
	default:
		return nil, fmt.Errorf("unknown embedding provider %q", cfg.Embedding.Provider)
	}
}

func loadConfig() (*config.Config, error) {
	configPath := os.Getenv("REVERIE_CONFIG")
	cfg, err := config.Load(configPath)
	if err != nil {
		// Don't fail on missing config; fall back to defaults.
		cfg = config.Defaults()
	}
	return cfg, nil
}

func newLogger(level string) *slog.Logger {
	var lvl slog.Level
	switch level {
	case "debug":
		lvl = slog.LevelDebug
	case "warn":
		lvl = slog.LevelWarn
	case "error":
		lvl = slog.LevelError
	default:
		lvl = slog.LevelInfo
	}
	return slog.New(slog.NewJSONHandler(os.Stderr, &slog.HandlerOptions{Level: lvl}))
}

func formatBytes(b int64) string {
	const unit = 1024
	if b < unit {
		return fmt.Sprintf("%d B", b)
	}
	div, exp := int64(unit), 0
	for n := b / unit; n >= unit; n /= unit {
		div *= unit
		exp++
	}
	return fmt.Sprintf("%.1f %cB", float64(b)/float64(div), "KMG"[exp])
}
