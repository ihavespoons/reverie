package main

import (
	"bufio"
	"context"
	"fmt"
	"os"
	"strconv"
	"strings"
	"time"

	"personal/reverie/internal/config"
	"personal/reverie/internal/db"
	"personal/reverie/internal/embed"
	"personal/reverie/internal/memory"
)

// parseArgs parses CLI flags from a slice of arguments into a map.
// Boolean flags (--force, --dry-run) get the value "true".
// Key-value flags (--limit 50, --layer l2) map key to value.
// The first non-flag argument is stored under "_positional".
func parseArgs(args []string) map[string]string {
	m := make(map[string]string)
	for i := 0; i < len(args); i++ {
		if strings.HasPrefix(args[i], "--") {
			key := strings.TrimPrefix(args[i], "--")
			if i+1 < len(args) && !strings.HasPrefix(args[i+1], "--") {
				m[key] = args[i+1]
				i++
			} else {
				m[key] = "true"
			}
		} else {
			if _, exists := m["_positional"]; !exists {
				m["_positional"] = args[i]
			}
		}
	}
	return m
}

// truncate shortens s to maxLen characters, appending "..." if truncated.
func truncate(s string, maxLen int) string {
	// Replace newlines with spaces for single-line display.
	s = strings.ReplaceAll(s, "\n", " ")
	s = strings.ReplaceAll(s, "\r", "")
	if len(s) > maxLen {
		return s[:maxLen-3] + "..."
	}
	return s
}

// openStoreReadOnly opens the database and returns a store for CLI read operations.
func openStoreReadOnly(cfg *config.Config) (memory.Store, func(), error) {
	if _, err := os.Stat(cfg.Storage.DBPath); os.IsNotExist(err) {
		return nil, nil, fmt.Errorf("database not found at %s\nRun 'reverie serve' first to initialize the database", cfg.Storage.DBPath)
	}
	database, err := db.Open(cfg.Storage.DBPath)
	if err != nil {
		return nil, nil, fmt.Errorf("open database: %w", err)
	}
	store := memory.NewSQLiteStore(database)
	cleanup := func() { database.Close() }
	return store, cleanup, nil
}

// runView implements the `reverie view` subcommand.
func runView() error {
	cfg, err := loadConfig()
	if err != nil {
		return err
	}

	store, cleanup, err := openStoreReadOnly(cfg)
	if err != nil {
		return err
	}
	defer cleanup()

	// Parse flags from os.Args[2:].
	args := parseArgs(os.Args[2:])

	layer := args["layer"]
	if layer == "" {
		layer = "l2"
	}
	if layer != "l2" && layer != "l3" {
		return fmt.Errorf("invalid layer %q: must be l2 or l3", layer)
	}

	limit := 50
	if v, ok := args["limit"]; ok {
		n, err := strconv.Atoi(v)
		if err != nil || n < 1 {
			return fmt.Errorf("invalid --limit %q: must be a positive integer", v)
		}
		if n > 1000 {
			n = 1000
		}
		limit = n
	}

	ctx := context.Background()

	if layer == "l2" {
		filter := memory.ListFilter{Limit: limit}
		if v, ok := args["subtype"]; ok {
			filter.Subtype = &v
		}

		facts, err := store.ListFacts(ctx, filter)
		if err != nil {
			return fmt.Errorf("list facts: %w", err)
		}

		fmt.Printf("%-10s%-10s%-12s%s\n", "ID", "TYPE", "CREATED", "CONTENT")
		for _, f := range facts {
			id := f.ID
			if len(id) > 8 {
				id = id[:8]
			}
			subtype := f.Subtype
			if subtype == "" {
				subtype = "-"
			}
			created := f.CreatedAt.Format("2006-01-02")
			content := truncate(f.Content, 80)
			fmt.Printf("%-10s%-10s%-12s%s\n", id, subtype, created, content)
		}
		fmt.Println("---")
		fmt.Printf("%d facts shown\n", len(facts))
	} else {
		// l3 episodes
		filter := memory.ListFilter{Limit: limit}
		episodes, err := store.ListEpisodes(ctx, filter)
		if err != nil {
			return fmt.Errorf("list episodes: %w", err)
		}

		fmt.Printf("%-10s%-12s%s\n", "ID", "CREATED", "SITUATION")
		for _, ep := range episodes {
			id := ep.ID
			if len(id) > 8 {
				id = id[:8]
			}
			created := ep.CreatedAt.Format("2006-01-02")
			situation := truncate(ep.Situation, 60)
			fmt.Printf("%-10s%-12s%s\n", id, created, situation)
		}
		fmt.Println("---")
		fmt.Printf("%d episodes shown\n", len(episodes))
	}

	return nil
}

// runForget implements the `reverie forget <id>` subcommand.
func runForget() error {
	cfg, err := loadConfig()
	if err != nil {
		return err
	}

	store, cleanup, err := openStoreReadOnly(cfg)
	if err != nil {
		return err
	}
	defer cleanup()

	// Parse flags from os.Args[2:].
	args := parseArgs(os.Args[2:])

	id := args["_positional"]
	if id == "" {
		return fmt.Errorf("usage: reverie forget <id> [--force]")
	}

	force := args["force"] == "true"

	ctx := context.Background()

	// Try as a fact first.
	fact, err := store.GetFact(ctx, id)
	if err != nil {
		return fmt.Errorf("get fact: %w", err)
	}
	if fact != nil {
		content := truncate(fact.Content, 80)
		fmt.Printf("Fact: %s\n", content)

		if !force {
			fmt.Print("Delete? [y/N] ")
			reader := bufio.NewReader(os.Stdin)
			line, err := reader.ReadString('\n')
			if err != nil {
				return fmt.Errorf("read confirmation: %w", err)
			}
			line = strings.TrimSpace(strings.ToLower(line))
			if line != "y" && line != "yes" {
				fmt.Println("Cancelled.")
				return nil
			}
		}

		if err := store.DeleteFact(ctx, id); err != nil {
			return fmt.Errorf("delete fact: %w", err)
		}
		fmt.Println("Deleted.")
		return nil
	}

	// Try as an episode.
	episode, err := store.GetEpisode(ctx, id)
	if err != nil {
		return fmt.Errorf("get episode: %w", err)
	}
	if episode != nil {
		content := truncate(episode.Situation, 80)
		fmt.Printf("Episode: %s\n", content)

		if !force {
			fmt.Print("Delete? [y/N] ")
			reader := bufio.NewReader(os.Stdin)
			line, err := reader.ReadString('\n')
			if err != nil {
				return fmt.Errorf("read confirmation: %w", err)
			}
			line = strings.TrimSpace(strings.ToLower(line))
			if line != "y" && line != "yes" {
				fmt.Println("Cancelled.")
				return nil
			}
		}

		if err := store.DeleteEpisode(ctx, id); err != nil {
			return fmt.Errorf("delete episode: %w", err)
		}
		fmt.Println("Deleted.")
		return nil
	}

	fmt.Fprintf(os.Stderr, "Not found: %s\n", id)
	os.Exit(1)
	return nil // unreachable, but satisfies the compiler
}

// runReindex implements the `reverie reindex [--dry-run]` subcommand.
func runReindex() error {
	cfg, err := loadConfig()
	if err != nil {
		return err
	}

	// Parse flags from os.Args[2:].
	args := parseArgs(os.Args[2:])
	dryRun := args["dry-run"] == "true"

	// Check DB exists.
	if _, err := os.Stat(cfg.Storage.DBPath); os.IsNotExist(err) {
		return fmt.Errorf("database not found at %s\nRun 'reverie serve' first to initialize the database", cfg.Storage.DBPath)
	}

	database, err := db.Open(cfg.Storage.DBPath)
	if err != nil {
		return fmt.Errorf("open database: %w", err)
	}
	defer database.Close()

	store := memory.NewSQLiteStore(database)

	// Build the embedder (same chain as runServe).
	inner, err := buildEmbedder(cfg)
	if err != nil {
		return fmt.Errorf("build embedder: %w", err)
	}
	embedder := embed.NewCachedProvider(inner, database)

	ctx := context.Background()

	// Load all facts and episodes.
	facts, err := store.ListFacts(ctx, memory.ListFilter{Limit: 100000})
	if err != nil {
		return fmt.Errorf("list facts: %w", err)
	}

	episodes, err := store.ListEpisodes(ctx, memory.ListFilter{Limit: 100000})
	if err != nil {
		return fmt.Errorf("list episodes: %w", err)
	}

	if dryRun {
		fmt.Printf("Would reindex %d facts and %d episodes\n", len(facts), len(episodes))
		return nil
	}

	start := time.Now()

	// Clear the embedding cache since cached embeddings are from the old model.
	if _, err := database.ExecContext(ctx, `DELETE FROM embedding_cache`); err != nil {
		return fmt.Errorf("clear embedding cache: %w", err)
	}

	// Reindex facts.
	for i, f := range facts {
		vecs, err := embedder.Embed(ctx, []string{f.Content})
		if err != nil {
			return fmt.Errorf("embed fact %s: %w", f.ID, err)
		}
		if err := store.UpdateFactEmbedding(ctx, f.ID, vecs[0]); err != nil {
			return fmt.Errorf("update fact embedding %s: %w", f.ID, err)
		}
		if (i+1)%10 == 0 || i+1 == len(facts) {
			fmt.Printf("reindexed %d/%d facts...\n", i+1, len(facts))
		}
	}

	// Reindex episodes.
	for i, ep := range episodes {
		text := ep.Situation + "\n" + ep.Action + "\n" + ep.Outcome + "\n" + ep.Preemptive
		vecs, err := embedder.Embed(ctx, []string{text})
		if err != nil {
			return fmt.Errorf("embed episode %s: %w", ep.ID, err)
		}
		if err := store.UpdateEpisodeEmbedding(ctx, ep.ID, vecs[0]); err != nil {
			return fmt.Errorf("update episode embedding %s: %w", ep.ID, err)
		}
		if (i+1)%10 == 0 || i+1 == len(episodes) {
			fmt.Printf("reindexed %d/%d episodes...\n", i+1, len(episodes))
		}
	}

	elapsed := time.Since(start)
	fmt.Printf("Reindex complete: %d facts, %d episodes reindexed in %.1fs\n",
		len(facts), len(episodes), elapsed.Seconds())

	return nil
}
