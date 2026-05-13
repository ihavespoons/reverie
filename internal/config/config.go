// Package config handles TOML-based configuration for reverie with
// environment variable overrides and sensible defaults.
package config

import (
	"fmt"
	"os"
	"path/filepath"
	"strconv"

	"github.com/BurntSushi/toml"
)

// Config is the top-level configuration for the reverie memory server.
type Config struct {
	Storage   Storage   `toml:"storage"`
	Memory    Memory    `toml:"memory"`
	Decay     Decay     `toml:"decay"`
	Embedding Embedding `toml:"embedding"`
	Cluster   Cluster   `toml:"cluster"`
	Server    Server    `toml:"server"`
	Session   Session   `toml:"session"`
}

// Storage configures the SQLite database location.
type Storage struct {
	DBPath string `toml:"db_path"`
}

// Memory configures working memory bounds and retrieval thresholds.
type Memory struct {
	SlidingWindowK      int     `toml:"sliding_window_k"`
	CacheBudgetMax      int     `toml:"cache_budget_max"`
	SimilarityThreshold float64 `toml:"similarity_threshold"`
	RetentionThreshold  float64 `toml:"retention_threshold"`
	ConflictThreshold   float64 `toml:"conflict_threshold"`

	// GraphDecayPerHop controls the per-hop decay in graph-aware recall
	// composite scoring: composite = sim * retention * (GraphDecayPerHop ^ hop).
	// Default 0.5. Range (0,1]. Zero or negative falls back to 0.5.
	GraphDecayPerHop float64 `toml:"graph_decay_per_hop"`

	// GraphMaxVisited bounds the distinct neighbor count during graph BFS to
	// prevent pathological blowup on dense graphs. Default 2000. Zero or
	// negative means unbounded (not recommended).
	GraphMaxVisited int `toml:"graph_max_visited"`

	// GraphMinRetentionForExpansion skips graph neighbors whose retention is
	// below this threshold during BFS -- decayed memories don't pollute the
	// candidate set and reduce global-cap pressure. Default 0.05. Zero or
	// negative means no pre-filter.
	GraphMinRetentionForExpansion float64 `toml:"graph_min_retention_for_expansion"`
}

// Decay configures the Ebbinghaus retention curve and utility/frequency learning rates.
type Decay struct {
	Temperature        float64 `toml:"temperature"`
	UtilityAlpha       float64 `toml:"utility_alpha"`
	FrequencyBeta      float64 `toml:"frequency_beta"`
	ColdStartUtility   float64 `toml:"cold_start_utility"`
	ColdStartFrequency float64 `toml:"cold_start_frequency"`
}

// Embedding configures the vector embedding provider.
//
// Provider selects the adapter:
//   - "openai_compat": OpenAI-compatible /v1/embeddings endpoint. Works with
//     Ollama, LM Studio, real OpenAI, and any other compatible service.
//     BaseURL and Dimensions must be set.
//   - "voyage": Voyage AI hosted API. APIKey (from VOYAGE_API_KEY) required;
//     BaseURL ignored.
type Embedding struct {
	Provider   string `toml:"provider"`
	BaseURL    string `toml:"base_url"` // openai_compat only
	Model      string `toml:"model"`
	BatchSize  int    `toml:"batch_size"`
	Dimensions int    `toml:"dimensions"` // model's embedding dim; advisory
	APIKey     string `toml:"-"`          // loaded from env only, never serialized
}

// Cluster configures L1 cluster assignment parameters.
type Cluster struct {
	MinSimilarityForAssignment float64 `toml:"min_similarity_for_assignment"`
	MaxClusters                int     `toml:"max_clusters"`
}

// Server configures the MCP server runtime behavior.
type Server struct {
	InactivityConsolidateSeconds int    `toml:"inactivity_consolidate_seconds"`
	RecallCacheTTLSeconds        int    `toml:"recall_cache_ttl_seconds"`
	LogLevel                     string `toml:"log_level"`
	Disabled                     bool   `toml:"disabled"`
}

// Session configures per-session working memory bounds. Added in Phase 6b.
// BufferBudgetMax caps the number of MemoryRef entries stored in a session's
// persisted buffer; appends that would exceed it evict the lowest-scored
// entry first.
type Session struct {
	BufferBudgetMax int `toml:"buffer_budget_max"`
}

// Defaults returns a fully-populated Config with sensible default values
// matching the plan's TOML example.
func Defaults() *Config {
	return &Config{
		Storage: Storage{
			DBPath: "~/.local/share/reverie/reverie.db",
		},
		Memory: Memory{
			SlidingWindowK:                20,
			CacheBudgetMax:                50,
			SimilarityThreshold:           0.70,
			RetentionThreshold:            0.30,
			ConflictThreshold:             0.92,
			GraphDecayPerHop:              0.5,
			GraphMaxVisited:               2000,
			GraphMinRetentionForExpansion: 0.05,
		},
		Decay: Decay{
			Temperature:        10.0,
			UtilityAlpha:       0.10,
			FrequencyBeta:      0.05,
			ColdStartUtility:   0.5,
			ColdStartFrequency: 0.5,
		},
		Embedding: Embedding{
			Provider:   "openai_compat",
			BaseURL:    "http://localhost:11434/v1", // Ollama default
			Model:      "nomic-embed-text",
			BatchSize:  32,
			Dimensions: 768,
		},
		Cluster: Cluster{
			MinSimilarityForAssignment: 0.60,
			MaxClusters:                100,
		},
		Server: Server{
			InactivityConsolidateSeconds: 300,
			RecallCacheTTLSeconds:        300,
			LogLevel:                     "info",
			Disabled:                     false,
		},
		Session: Session{
			BufferBudgetMax: 50,
		},
	}
}

// Load reads configuration from a TOML file at the given path, applies
// environment variable overrides, and returns the resulting Config.
//
// If path is empty, Load looks for the config file at
// $XDG_CONFIG_HOME/reverie/reverie.toml, falling back to
// ~/.config/reverie/reverie.toml.
//
// Environment overrides:
//   - VOYAGE_API_KEY       -> Embedding.APIKey (when provider is "voyage")
//   - OPENAI_API_KEY       -> Embedding.APIKey (when provider is "openai_compat"
//     and VOYAGE_API_KEY is unset); optional for Ollama
//     and LM Studio which ignore the bearer token
//   - REVERIE_EMBED_URL    -> Embedding.BaseURL (for openai_compat)
//   - REVERIE_EMBED_MODEL  -> Embedding.Model
//   - REVERIE_DB_PATH      -> Storage.DBPath
//   - REVERIE_LOG_LEVEL    -> Server.LogLevel
//   - REVERIE_DISABLED=1   -> Server.Disabled
func Load(path string) (*Config, error) {
	cfg := Defaults()

	if path == "" {
		path = defaultConfigPath()
	}

	// If the config file exists, decode it on top of defaults.
	if _, err := os.Stat(path); err == nil {
		if _, err := toml.DecodeFile(path, cfg); err != nil {
			return nil, fmt.Errorf("decode config %q: %w", path, err)
		}
	}

	// Apply environment variable overrides.
	applyEnvOverrides(cfg)

	// Expand ~ in db_path.
	cfg.Storage.DBPath = expandHome(cfg.Storage.DBPath)

	return cfg, nil
}

// defaultConfigPath returns the expected config file location, respecting
// XDG_CONFIG_HOME with a fallback to ~/.config.
func defaultConfigPath() string {
	configHome := os.Getenv("XDG_CONFIG_HOME")
	if configHome == "" {
		home, err := os.UserHomeDir()
		if err != nil {
			return "reverie.toml" // last-resort fallback
		}
		configHome = filepath.Join(home, ".config")
	}
	return filepath.Join(configHome, "reverie", "reverie.toml")
}

// applyEnvOverrides reads environment variables and applies them to the config.
func applyEnvOverrides(cfg *Config) {
	if v := os.Getenv("VOYAGE_API_KEY"); v != "" {
		cfg.Embedding.APIKey = v
	} else if v := os.Getenv("OPENAI_API_KEY"); v != "" {
		cfg.Embedding.APIKey = v
	}
	if v := os.Getenv("REVERIE_EMBED_URL"); v != "" {
		cfg.Embedding.BaseURL = v
	}
	if v := os.Getenv("REVERIE_EMBED_MODEL"); v != "" {
		cfg.Embedding.Model = v
	}
	if v := os.Getenv("REVERIE_DB_PATH"); v != "" {
		cfg.Storage.DBPath = v
	}
	if v := os.Getenv("REVERIE_LOG_LEVEL"); v != "" {
		cfg.Server.LogLevel = v
	}
	if v := os.Getenv("REVERIE_DISABLED"); v != "" {
		cfg.Server.Disabled = parseBool(v)
	}
}

// expandHome replaces a leading ~/ with the user's home directory.
func expandHome(path string) string {
	if len(path) < 2 || path[:2] != "~/" {
		return path
	}
	home, err := os.UserHomeDir()
	if err != nil {
		return path
	}
	return filepath.Join(home, path[2:])
}

// parseBool treats "1", "true", and "yes" (case-insensitive) as true.
func parseBool(s string) bool {
	b, err := strconv.ParseBool(s)
	if err != nil {
		return false
	}
	return b
}
