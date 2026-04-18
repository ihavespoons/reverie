package db

import (
	"database/sql"
	_ "embed"
	"fmt"
	"time"
)

// migration describes a single schema migration. Version must be unique and
// strictly monotonically increasing across the migrations slice.
type migration struct {
	Version int
	Name    string
	SQL     string
}

// migrations is the ordered list of schema migrations. New migrations MUST be
// appended at the end; never reorder or renumber existing entries.
//
// Migration 1 is the initial schema (identical to schema.sql). Every statement
// in it uses CREATE TABLE/INDEX IF NOT EXISTS, so running it against a legacy
// database that already has these tables (from pre-migration startup paths)
// is a safe no-op. Subsequent migrations then advance the schema forward.
var migrations = []migration{
	{
		Version: 1,
		Name:    "initial_schema",
		SQL:     schemaSQL,
	},
	{
		Version: 2,
		Name:    "add_tags_columns",
		SQL: `ALTER TABLE facts    ADD COLUMN tags TEXT DEFAULT '[]';
ALTER TABLE episodes ADD COLUMN tags TEXT DEFAULT '[]';`,
	},
}

// applyMigrations ensures the schema_migrations bookkeeping table exists and
// applies every migration with Version > max(applied). Each migration runs in
// its own transaction: on failure the transaction is rolled back and no row is
// recorded in schema_migrations, so the migration will be retried on the next
// Open call. Fully-migrated databases are a no-op past the bookkeeping query.
func applyMigrations(db *sql.DB) error {
	// Bookkeeping table. Uses IF NOT EXISTS so repeated calls are safe.
	_, err := db.Exec(
		`CREATE TABLE IF NOT EXISTS schema_migrations (
			version INTEGER PRIMARY KEY,
			name TEXT NOT NULL,
			applied_at TEXT NOT NULL
		)`,
	)
	if err != nil {
		return fmt.Errorf("create schema_migrations: %w", err)
	}

	// Determine the highest applied version. COALESCE protects against an
	// empty table where MAX() returns NULL.
	var current int
	if err := db.QueryRow(
		`SELECT COALESCE(MAX(version), 0) FROM schema_migrations`,
	).Scan(&current); err != nil {
		return fmt.Errorf("read schema_migrations: %w", err)
	}

	for _, m := range migrations {
		if m.Version <= current {
			continue
		}
		if err := applyMigration(db, m); err != nil {
			return fmt.Errorf("apply migration %d (%s): %w", m.Version, m.Name, err)
		}
	}
	return nil
}

// applyMigration runs a single migration inside a transaction. On any error
// the transaction is rolled back, leaving the database unchanged and the
// migration un-recorded.
func applyMigration(db *sql.DB, m migration) error {
	tx, err := db.Begin()
	if err != nil {
		return fmt.Errorf("begin: %w", err)
	}
	// Rollback is a no-op once Commit has succeeded; it's safe to defer
	// unconditionally so that any early return path tears the tx down.
	defer tx.Rollback() //nolint:errcheck

	if _, err := tx.Exec(m.SQL); err != nil {
		return fmt.Errorf("exec sql: %w", err)
	}
	if _, err := tx.Exec(
		`INSERT INTO schema_migrations (version, name, applied_at) VALUES (?, ?, ?)`,
		m.Version, m.Name, time.Now().UTC().Format(time.RFC3339),
	); err != nil {
		return fmt.Errorf("insert schema_migrations: %w", err)
	}

	if err := tx.Commit(); err != nil {
		return fmt.Errorf("commit: %w", err)
	}
	return nil
}
