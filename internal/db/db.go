// Package db provides SQLite database initialization for reverie.
// It uses the pure-Go modernc.org/sqlite driver (no CGO) and embeds versioned
// migrations that are applied automatically on startup.
package db

import (
	"database/sql"
	_ "embed"
	"fmt"

	_ "modernc.org/sqlite"
)

// schemaSQL is the canonical initial schema snapshot. It is embedded as
// migration 1 (`initial_schema`) in the migrations slice. Future schema
// changes go in new migrations; this file is not edited after release.
//
//go:embed schema.sql
var schemaSQL string

// Open opens (or creates) a SQLite database at the given path, configures
// WAL mode, foreign keys, and busy timeout, then applies any pending schema
// migrations. The caller is responsible for closing the returned *sql.DB.
func Open(path string) (*sql.DB, error) {
	db, err := sql.Open("sqlite", path)
	if err != nil {
		return nil, fmt.Errorf("open sqlite %q: %w", path, err)
	}

	// Configure pragmas for performance and correctness.
	pragmas := []string{
		"PRAGMA journal_mode=WAL",
		"PRAGMA foreign_keys=ON",
		"PRAGMA busy_timeout=5000",
	}
	for _, p := range pragmas {
		if _, err := db.Exec(p); err != nil {
			db.Close()
			return nil, fmt.Errorf("exec %q: %w", p, err)
		}
	}

	// Apply pending schema migrations (creates schema_migrations bookkeeping
	// table on first run; no-op when the DB is fully migrated).
	if err := applyMigrations(db); err != nil {
		db.Close()
		return nil, fmt.Errorf("apply migrations: %w", err)
	}

	return db, nil
}
