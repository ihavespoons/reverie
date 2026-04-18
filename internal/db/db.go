// Package db provides SQLite database initialization for reverie.
// It uses the pure-Go modernc.org/sqlite driver (no CGO) and embeds the
// schema DDL for automatic table creation on startup.
package db

import (
	"database/sql"
	_ "embed"
	"fmt"

	_ "modernc.org/sqlite"
)

//go:embed schema.sql
var schemaSQL string

// Open opens (or creates) a SQLite database at the given path, configures
// WAL mode, foreign keys, and busy timeout, then applies the embedded schema.
// The caller is responsible for closing the returned *sql.DB.
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

	// Apply the full schema. CREATE TABLE IF NOT EXISTS makes this idempotent.
	if _, err := db.Exec(schemaSQL); err != nil {
		db.Close()
		return nil, fmt.Errorf("exec schema: %w", err)
	}

	return db, nil
}
