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

// TestMigration3_FreshDB covers the tables + seed row + trigger set created
// by migration 3 when run on an empty database. Checks structural presence
// rather than any specific trigger body.
func TestMigration3_FreshDB(t *testing.T) {
	path := filepath.Join(t.TempDir(), "mig3_fresh.db")
	db, err := Open(path)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer db.Close()

	// daily_stats exists (empty since no facts/episodes yet).
	if countRows(t, db, "daily_stats") != 0 {
		t.Errorf("daily_stats should be empty on fresh DB")
	}

	// decay_state has exactly one row, id=1, last_tick NULL.
	var rows int
	if err := db.QueryRow(`SELECT COUNT(*) FROM decay_state`).Scan(&rows); err != nil {
		t.Fatalf("count decay_state: %v", err)
	}
	if rows != 1 {
		t.Errorf("decay_state row count = %d, want 1", rows)
	}
	var id int
	var lastTick sql.NullString
	if err := db.QueryRow(`SELECT id, last_tick FROM decay_state WHERE id = 1`).Scan(&id, &lastTick); err != nil {
		t.Fatalf("select decay_state: %v", err)
	}
	if id != 1 {
		t.Errorf("decay_state id = %d, want 1", id)
	}
	if lastTick.Valid {
		t.Errorf("decay_state.last_tick should be NULL on fresh DB, got %q", lastTick.String)
	}

	// All six triggers are present in sqlite_master.
	wantTriggers := []string{
		"trg_facts_insert",
		"trg_facts_delete",
		"trg_facts_supersede",
		"trg_episodes_insert",
		"trg_episodes_delete",
	}
	for _, name := range wantTriggers {
		var found string
		err := db.QueryRow(
			`SELECT name FROM sqlite_master WHERE type='trigger' AND name=?`,
			name,
		).Scan(&found)
		if err != nil {
			t.Errorf("trigger %q missing: %v", name, err)
		}
	}
}

// TestMigration3_Backfill exercises the one-shot backfill path by applying
// the embedded migrations against a DB that already has facts + episodes
// sitting at known creation dates. Verifies that daily_stats receives the
// aggregated counts and that a date shared between facts and episodes ends
// up with both counters set (the ON CONFLICT case the comment warns about).
func TestMigration3_Backfill(t *testing.T) {
	// Apply migrations 1 and 2 manually, seed data, then append migration 3
	// and re-run applyMigrations. Using the raw driver bypasses Open's
	// PRAGMA setup; that's fine for a test focused on migration body semantics.
	saved := migrations
	t.Cleanup(func() { migrations = saved })

	// Keep only migrations 1 and 2 during the seed phase. Migration 3
	// (the backfill under test) and any later migrations are re-plugged
	// after we seed facts/episodes, so the backfill sees the seed data.
	var laterMigs []migration
	trimmed := make([]migration, 0, len(saved))
	for _, m := range saved {
		if m.Version >= 3 {
			laterMigs = append(laterMigs, m)
			continue
		}
		trimmed = append(trimmed, m)
	}
	migrations = trimmed

	path := filepath.Join(t.TempDir(), "mig3_backfill.db")
	raw, err := sql.Open("sqlite", path)
	if err != nil {
		t.Fatalf("sql.Open: %v", err)
	}
	defer raw.Close()
	if _, err := raw.Exec(`PRAGMA foreign_keys=ON`); err != nil {
		t.Fatalf("pragma fk: %v", err)
	}

	if err := applyMigrations(raw); err != nil {
		t.Fatalf("applyMigrations (up to 2): %v", err)
	}

	// Seed default cluster (FK target) and facts/episodes with explicit
	// created_at strings so the backfill grouping is deterministic.
	if _, err := raw.Exec(
		`INSERT INTO clusters (id, summary) VALUES ('default', 'default')`,
	); err != nil {
		t.Fatalf("seed cluster: %v", err)
	}

	const dayA = "2026-04-10"
	const dayB = "2026-04-11"
	// Two facts on dayA, one fact on dayB.
	factSeed := []struct{ id, created string }{
		{"f1", dayA + "T00:00:01Z"},
		{"f2", dayA + "T10:00:00Z"},
		{"f3", dayB + "T00:00:01Z"},
	}
	for _, f := range factSeed {
		if _, err := raw.Exec(
			`INSERT INTO facts (id, cluster_id, content, content_hash, created_at, accessed_at)
			 VALUES (?, 'default', ?, ?, ?, ?)`,
			f.id, f.id+" content", f.id+"-hash", f.created, f.created,
		); err != nil {
			t.Fatalf("seed fact %s: %v", f.id, err)
		}
	}

	// One episode on dayA (shared date with facts), one on dayB.
	epSeed := []struct{ id, created string }{
		{"e1", dayA + "T12:00:00Z"},
		{"e2", dayB + "T12:00:00Z"},
	}
	for _, e := range epSeed {
		if _, err := raw.Exec(
			`INSERT INTO episodes (id, cluster_id, situation, action, outcome, preemptive, content_hash, created_at, accessed_at)
			 VALUES (?, 'default', 'sit', 'act', 'out', 'pre', ?, ?, ?)`,
			e.id, e.id+"-hash", e.created, e.created,
		); err != nil {
			t.Fatalf("seed episode %s: %v", e.id, err)
		}
	}

	// Now plug migrations 3+ back in (in their original order) and apply.
	migrations = append(append([]migration{}, trimmed...), laterMigs...)
	if err := applyMigrations(raw); err != nil {
		t.Fatalf("applyMigrations (mig 3+): %v", err)
	}

	// Expect: dayA has facts_in=2, episodes_in=1; dayB has facts_in=1, episodes_in=1.
	rowsByDate := map[string]struct{ factsIn, epIn int }{}
	rows, err := raw.Query(`SELECT date, facts_in, episodes_in FROM daily_stats ORDER BY date`)
	if err != nil {
		t.Fatalf("query daily_stats: %v", err)
	}
	defer rows.Close()
	for rows.Next() {
		var d string
		var fin, epin int
		if err := rows.Scan(&d, &fin, &epin); err != nil {
			t.Fatalf("scan: %v", err)
		}
		rowsByDate[d] = struct{ factsIn, epIn int }{fin, epin}
	}

	gotA, okA := rowsByDate[dayA]
	if !okA {
		t.Fatalf("missing row for %s; got %+v", dayA, rowsByDate)
	}
	if gotA.factsIn != 2 || gotA.epIn != 1 {
		t.Errorf("%s: facts_in=%d episodes_in=%d, want 2/1", dayA, gotA.factsIn, gotA.epIn)
	}
	gotB, okB := rowsByDate[dayB]
	if !okB {
		t.Fatalf("missing row for %s; got %+v", dayB, rowsByDate)
	}
	if gotB.factsIn != 1 || gotB.epIn != 1 {
		t.Errorf("%s: facts_in=%d episodes_in=%d, want 1/1", dayB, gotB.factsIn, gotB.epIn)
	}
}

// TestMigration3_Idempotent verifies that re-running applyMigrations on a
// fully-migrated DB doesn't re-run migration 3's backfill. The
// schema_migrations bookkeeping row for version 3 gates the rerun; the test
// checks the gate holds by writing a counter via raw SQL and confirming it
// isn't clobbered.
func TestMigration3_Idempotent(t *testing.T) {
	path := filepath.Join(t.TempDir(), "mig3_idem.db")
	db, err := Open(path)
	if err != nil {
		t.Fatalf("first Open: %v", err)
	}
	// Write a known daily_stats row via raw SQL — simulating a day with
	// counts that didn't come from triggers.
	const probeDate = "2026-04-15"
	if _, err := db.Exec(
		`INSERT INTO daily_stats (date, facts_in, episodes_in) VALUES (?, 7, 11)`,
		probeDate,
	); err != nil {
		t.Fatalf("seed probe row: %v", err)
	}
	db.Close()

	// Reopen: applyMigrations should see version 3 already applied and skip
	// the backfill entirely (no INSERT OR IGNORE collision recomputation).
	db2, err := Open(path)
	if err != nil {
		t.Fatalf("second Open: %v", err)
	}
	defer db2.Close()

	// Version 3 is present (not repeated, not rolled back by a later
	// migration). The highest applied version advances as new migrations
	// land, so we check for 3's presence rather than leadership.
	got := appliedVersions(t, db2)
	hasMig3 := false
	for _, v := range got {
		if v == 3 {
			hasMig3 = true
			break
		}
	}
	if !hasMig3 {
		t.Errorf("migration 3 missing from applied versions: %v", got)
	}
	// Probe row unchanged.
	var factsIn, epIn int
	if err := db2.QueryRow(
		`SELECT facts_in, episodes_in FROM daily_stats WHERE date = ?`, probeDate,
	).Scan(&factsIn, &epIn); err != nil {
		t.Fatalf("probe reselect: %v", err)
	}
	if factsIn != 7 || epIn != 11 {
		t.Errorf("probe row changed: facts_in=%d episodes_in=%d, want 7/11", factsIn, epIn)
	}
}

// TestMigration4_FreshDB exercises the schema changes made by migration 4
// (session_metadata): the sessions table picks up project_hint / tags /
// created_at / closed_at columns and the updated_at index is created.
func TestMigration4_FreshDB(t *testing.T) {
	path := filepath.Join(t.TempDir(), "mig4_fresh.db")
	db, err := Open(path)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer db.Close()

	wantCols := []string{"project_hint", "tags", "created_at", "closed_at"}
	for _, col := range wantCols {
		var found string
		row := db.QueryRow(
			`SELECT name FROM pragma_table_info('sessions') WHERE name = ?`, col,
		)
		if err := row.Scan(&found); err != nil {
			t.Errorf("sessions.%s column missing: %v", col, err)
		}
	}

	// Index present.
	var idxName string
	if err := db.QueryRow(
		`SELECT name FROM sqlite_master WHERE type='index' AND name='idx_sessions_updated'`,
	).Scan(&idxName); err != nil {
		t.Errorf("idx_sessions_updated missing: %v", err)
	}
}

// TestMigration4_Idempotent verifies that reopening a fully-migrated DB
// does not re-run migration 4. We write-then-read a sessions row across
// the reopen boundary to prove the added columns persist unchanged.
func TestMigration4_Idempotent(t *testing.T) {
	path := filepath.Join(t.TempDir(), "mig4_idem.db")
	db, err := Open(path)
	if err != nil {
		t.Fatalf("first Open: %v", err)
	}
	// Write a session with specific project_hint/tags so the idempotency
	// check has something observable to compare.
	if _, err := db.Exec(
		`INSERT INTO sessions (id, turn_counter, working_memory, project_hint, tags, created_at, updated_at)
		 VALUES ('s1', 0, '{}', 'reverie', '["go","mcp"]', datetime('now'), datetime('now'))`,
	); err != nil {
		t.Fatalf("seed session: %v", err)
	}
	db.Close()

	// Reopen: schema_migrations already has version 4, so the migration
	// must be a no-op and the row must round-trip unchanged.
	db2, err := Open(path)
	if err != nil {
		t.Fatalf("second Open: %v", err)
	}
	defer db2.Close()

	var projectHint, tags string
	if err := db2.QueryRow(
		`SELECT project_hint, tags FROM sessions WHERE id = 's1'`,
	).Scan(&projectHint, &tags); err != nil {
		t.Fatalf("reselect session: %v", err)
	}
	if projectHint != "reverie" {
		t.Errorf("project_hint = %q, want %q", projectHint, "reverie")
	}
	if tags != `["go","mcp"]` {
		t.Errorf("tags = %q, want %q", tags, `["go","mcp"]`)
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
