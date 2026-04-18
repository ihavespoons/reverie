package db

import (
	"database/sql"
	"path/filepath"
	"strings"
	"testing"
)

// countRows returns the number of rows in the named table.
func countRows(t *testing.T, db *sql.DB, table string) int {
	t.Helper()
	var n int
	if err := db.QueryRow(`SELECT COUNT(*) FROM ` + table).Scan(&n); err != nil {
		t.Fatalf("count %s: %v", table, err)
	}
	return n
}

// appliedVersions reads schema_migrations ordered by version.
func appliedVersions(t *testing.T, db *sql.DB) []int {
	t.Helper()
	rows, err := db.Query(`SELECT version FROM schema_migrations ORDER BY version ASC`)
	if err != nil {
		t.Fatalf("query schema_migrations: %v", err)
	}
	defer rows.Close()
	var out []int
	for rows.Next() {
		var v int
		if err := rows.Scan(&v); err != nil {
			t.Fatalf("scan version: %v", err)
		}
		out = append(out, v)
	}
	if err := rows.Err(); err != nil {
		t.Fatalf("rows err: %v", err)
	}
	return out
}

func TestApplyMigrations_FreshDB(t *testing.T) {
	path := filepath.Join(t.TempDir(), "fresh.db")
	db, err := Open(path)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer db.Close()

	// schema_migrations must contain one row per embedded migration.
	got := appliedVersions(t, db)
	if len(got) != len(migrations) {
		t.Fatalf("schema_migrations rows = %d, want %d", len(got), len(migrations))
	}
	for i, m := range migrations {
		if got[i] != m.Version {
			t.Errorf("applied[%d] = %d, want %d", i, got[i], m.Version)
		}
	}

	// Initial-schema tables exist.
	for _, tbl := range []string{"clusters", "facts", "episodes", "fact_episode_links", "embedding_cache", "sessions"} {
		if countRows(t, db, tbl) < 0 {
			t.Errorf("table %q should exist", tbl)
		}
	}

	// Migration 2 added tags columns on facts + episodes.
	for _, tbl := range []string{"facts", "episodes"} {
		var name string
		row := db.QueryRow(
			`SELECT name FROM pragma_table_info(?) WHERE name = 'tags'`, tbl,
		)
		if err := row.Scan(&name); err != nil {
			t.Errorf("table %q missing tags column: %v", tbl, err)
		}
	}
}

func TestApplyMigrations_FullyMigratedNoOp(t *testing.T) {
	path := filepath.Join(t.TempDir(), "reopen.db")
	db, err := Open(path)
	if err != nil {
		t.Fatalf("first Open: %v", err)
	}
	first := appliedVersions(t, db)
	db.Close()

	// Reopen: applyMigrations should see all versions applied and do nothing.
	db2, err := Open(path)
	if err != nil {
		t.Fatalf("second Open: %v", err)
	}
	defer db2.Close()

	second := appliedVersions(t, db2)
	if len(first) != len(second) {
		t.Fatalf("row count changed on reopen: first=%d second=%d", len(first), len(second))
	}
	for i := range first {
		if first[i] != second[i] {
			t.Errorf("applied[%d] changed: first=%d second=%d", i, first[i], second[i])
		}
	}
}

func TestApplyMigrations_PartiallyMigrated(t *testing.T) {
	// Simulate a DB where only migration 1 has been applied, with none of
	// migration 2's tags columns present. Then run applyMigrations directly
	// and assert only migration 2+ runs.
	path := filepath.Join(t.TempDir(), "partial.db")
	raw, err := sql.Open("sqlite", path)
	if err != nil {
		t.Fatalf("sql.Open: %v", err)
	}
	defer raw.Close()

	// Hand-apply migration 1 + record it.
	if _, err := raw.Exec(schemaSQL); err != nil {
		t.Fatalf("exec initial schema: %v", err)
	}
	if _, err := raw.Exec(
		`CREATE TABLE IF NOT EXISTS schema_migrations (
			version INTEGER PRIMARY KEY,
			name TEXT NOT NULL,
			applied_at TEXT NOT NULL
		)`,
	); err != nil {
		t.Fatalf("create schema_migrations: %v", err)
	}
	if _, err := raw.Exec(
		`INSERT INTO schema_migrations (version, name, applied_at) VALUES (1, 'initial_schema', datetime('now'))`,
	); err != nil {
		t.Fatalf("insert migration 1: %v", err)
	}

	// Sanity: tags column is absent before migration 2 runs.
	var tagCount int
	if err := raw.QueryRow(
		`SELECT COUNT(*) FROM pragma_table_info('facts') WHERE name = 'tags'`,
	).Scan(&tagCount); err != nil {
		t.Fatalf("pragma facts: %v", err)
	}
	if tagCount != 0 {
		t.Fatalf("partial state broken: facts.tags already exists")
	}

	// Run the migration runner.
	if err := applyMigrations(raw); err != nil {
		t.Fatalf("applyMigrations: %v", err)
	}

	got := appliedVersions(t, raw)
	if len(got) != len(migrations) {
		t.Fatalf("schema_migrations rows = %d, want %d", len(got), len(migrations))
	}
	// Migration 2 is now present.
	if err := raw.QueryRow(
		`SELECT COUNT(*) FROM pragma_table_info('facts') WHERE name = 'tags'`,
	).Scan(&tagCount); err != nil {
		t.Fatalf("pragma facts after: %v", err)
	}
	if tagCount != 1 {
		t.Fatalf("migration 2 did not add facts.tags column")
	}
}

func TestApplyMigrations_LegacyDBWithInitialSchema(t *testing.T) {
	// Legacy scenario: database predates the migration framework and already
	// has every initial_schema table (from the old Exec(schemaSQL) code path).
	// schema_migrations does NOT exist yet. Running applyMigrations must NOT
	// fail on migration 1 (all IF NOT EXISTS) and must apply migration 2.
	path := filepath.Join(t.TempDir(), "legacy.db")
	raw, err := sql.Open("sqlite", path)
	if err != nil {
		t.Fatalf("sql.Open: %v", err)
	}
	defer raw.Close()

	if _, err := raw.Exec(schemaSQL); err != nil {
		t.Fatalf("exec initial schema: %v", err)
	}

	if err := applyMigrations(raw); err != nil {
		t.Fatalf("applyMigrations on legacy DB: %v", err)
	}
	got := appliedVersions(t, raw)
	if len(got) != len(migrations) {
		t.Fatalf("schema_migrations rows = %d, want %d", len(got), len(migrations))
	}
}

func TestApplyMigrations_FailureRollsBack(t *testing.T) {
	// Inject a deliberately-broken migration after the real ones. Keep the
	// real slice intact across the test via restore.
	saved := migrations
	t.Cleanup(func() { migrations = saved })

	next := saved[len(saved)-1].Version + 1
	migrations = append(append([]migration{}, saved...), migration{
		Version: next,
		Name:    "bad_migration",
		// Reference a nonexistent table so the ALTER errors at exec time.
		SQL: `ALTER TABLE does_not_exist ADD COLUMN foo TEXT;`,
	})

	path := filepath.Join(t.TempDir(), "rollback.db")
	_, err := Open(path)
	if err == nil {
		t.Fatal("Open should have failed due to bad migration")
	}
	if !strings.Contains(err.Error(), "bad_migration") {
		t.Errorf("error should mention bad_migration, got: %v", err)
	}

	// Reopen raw and confirm bad_migration is NOT recorded in
	// schema_migrations (tx rolled back). The previous migrations should be.
	raw, err := sql.Open("sqlite", path)
	if err != nil {
		t.Fatalf("sql.Open reopen: %v", err)
	}
	defer raw.Close()

	var rows int
	if err := raw.QueryRow(
		`SELECT COUNT(*) FROM schema_migrations WHERE version = ?`, next,
	).Scan(&rows); err != nil {
		t.Fatalf("query schema_migrations: %v", err)
	}
	if rows != 0 {
		t.Errorf("failed migration should not be recorded, got %d rows for version %d", rows, next)
	}
	// The real migrations should all have committed.
	var applied int
	if err := raw.QueryRow(
		`SELECT COUNT(*) FROM schema_migrations WHERE version <= ?`, saved[len(saved)-1].Version,
	).Scan(&applied); err != nil {
		t.Fatalf("count applied: %v", err)
	}
	if applied != len(saved) {
		t.Errorf("prior migrations should all be recorded; got %d want %d", applied, len(saved))
	}
}

// Guard against duplicate versions in the embedded migrations slice — a
// programmer error that would silently break version ordering.
func TestMigrationsVersionsUnique(t *testing.T) {
	seen := map[int]string{}
	for _, m := range migrations {
		if prev, ok := seen[m.Version]; ok {
			t.Fatalf("duplicate migration version %d (name=%s and %s)", m.Version, prev, m.Name)
		}
		seen[m.Version] = m.Name
	}
}

// Ensure migrations slice is strictly monotonically increasing by Version —
// the apply loop assumes this, and the order is load-bearing.
func TestMigrationsMonotonic(t *testing.T) {
	for i := 1; i < len(migrations); i++ {
		if migrations[i].Version <= migrations[i-1].Version {
			t.Errorf("migrations[%d].Version=%d should be > migrations[%d].Version=%d",
				i, migrations[i].Version, i-1, migrations[i-1].Version)
		}
	}
}

// Smoke check that the migration 2 payload is exactly the two ALTERs the
// spec calls for. Using contains rather than equality so formatting is
// flexible; the pair of ALTER TABLEs is the intent.
func TestMigration2AddsTagsColumns(t *testing.T) {
	var m2 migration
	for _, m := range migrations {
		if m.Version == 2 {
			m2 = m
			break
		}
	}
	if m2.Version != 2 {
		t.Fatal("migration 2 not found in migrations slice")
	}
	for _, want := range []string{
		`ALTER TABLE facts`,
		`ALTER TABLE episodes`,
		`tags TEXT DEFAULT '[]'`,
	} {
		if !strings.Contains(m2.SQL, want) {
			t.Errorf("migration 2 SQL missing %q: %s", want, m2.SQL)
		}
	}
	if m2.Name != "add_tags_columns" {
		t.Errorf("migration 2 name = %q, want add_tags_columns", m2.Name)
	}
}
