package memory

import (
	"context"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"personal/reverie/internal/db"
)

func openTestDB(t *testing.T) *sqliteStore {
	t.Helper()
	dbPath := filepath.Join(t.TempDir(), "test.db")
	sqlDB, err := db.Open(dbPath)
	if err != nil {
		t.Fatalf("db.Open: %v", err)
	}
	t.Cleanup(func() { sqlDB.Close() })
	return NewSQLiteStore(sqlDB).(*sqliteStore)
}

func TestSQLiteInsertGetRoundTrip(t *testing.T) {
	s := openTestDB(t)
	ctx := context.Background()

	facts := testdataFacts()
	f := facts[0]

	id, err := s.InsertFact(ctx, f)
	if err != nil {
		t.Fatalf("InsertFact: %v", err)
	}
	if id == "" {
		t.Fatal("InsertFact returned empty id")
	}

	got, err := s.GetFact(ctx, id)
	if err != nil {
		t.Fatalf("GetFact: %v", err)
	}
	if got == nil {
		t.Fatal("GetFact returned nil")
	}

	if got.Content != f.Content {
		t.Errorf("Content = %q, want %q", got.Content, f.Content)
	}
	if got.Subtype != f.Subtype {
		t.Errorf("Subtype = %q, want %q", got.Subtype, f.Subtype)
	}
	if got.Source != f.Source {
		t.Errorf("Source = %q, want %q", got.Source, f.Source)
	}
	if got.Confidence != 1.0 {
		t.Errorf("Confidence = %v, want 1.0", got.Confidence)
	}

	// Embedding round-trip.
	if len(got.Embedding) != len(f.Embedding) {
		t.Fatalf("Embedding length = %d, want %d", len(got.Embedding), len(f.Embedding))
	}
	for i := range f.Embedding {
		if got.Embedding[i] != f.Embedding[i] {
			t.Errorf("Embedding[%d] = %v, want %v", i, got.Embedding[i], f.Embedding[i])
		}
	}
}

func TestSQLiteInsertIdempotency(t *testing.T) {
	s := openTestDB(t)
	ctx := context.Background()

	facts := testdataFacts()
	f := facts[0]

	id1, err := s.InsertFact(ctx, f)
	if err != nil {
		t.Fatalf("first InsertFact: %v", err)
	}

	id2, err := s.InsertFact(ctx, f)
	if err != nil {
		t.Fatalf("second InsertFact: %v", err)
	}

	if id1 != id2 {
		t.Errorf("idempotency broken: id1=%s, id2=%s", id1, id2)
	}
}

func TestSQLiteListFactsSubtypeFilter(t *testing.T) {
	s := openTestDB(t)
	ctx := context.Background()

	for _, f := range testdataFacts() {
		if _, err := s.InsertFact(ctx, f); err != nil {
			t.Fatalf("InsertFact: %v", err)
		}
	}

	subtype := "feedback"
	results, err := s.ListFacts(ctx, ListFilter{Subtype: &subtype})
	if err != nil {
		t.Fatalf("ListFacts: %v", err)
	}
	if len(results) != 1 {
		t.Fatalf("ListFacts(subtype=feedback) returned %d, want 1", len(results))
	}
	if results[0].Subtype != "feedback" {
		t.Errorf("Subtype = %q, want %q", results[0].Subtype, "feedback")
	}
}

func TestSQLiteDeleteFact(t *testing.T) {
	s := openTestDB(t)
	ctx := context.Background()

	facts := testdataFacts()
	id, err := s.InsertFact(ctx, facts[0])
	if err != nil {
		t.Fatalf("InsertFact: %v", err)
	}

	if err := s.DeleteFact(ctx, id); err != nil {
		t.Fatalf("DeleteFact: %v", err)
	}

	got, err := s.GetFact(ctx, id)
	if err != nil {
		t.Fatalf("GetFact after delete: %v", err)
	}
	if got != nil {
		t.Error("GetFact after delete returned non-nil")
	}

	// Deleting a non-existent id should not error.
	if err := s.DeleteFact(ctx, "nonexistent"); err != nil {
		t.Errorf("DeleteFact(nonexistent) = %v, want nil", err)
	}
}

func TestSQLiteGlobalSearchOrdering(t *testing.T) {
	s := openTestDB(t)
	ctx := context.Background()

	facts := testdataFacts()
	ids := make([]string, len(facts))
	for i, f := range facts {
		id, err := s.InsertFact(ctx, f)
		if err != nil {
			t.Fatalf("InsertFact[%d]: %v", i, err)
		}
		ids[i] = id
	}

	// Query vector along x axis: [1,0,0,0].
	// Expected cosine similarities:
	//   facts[0] embedding [1,0,0,0]: cosine = 1.0
	//   facts[2] embedding [0.6,0.8,0,0]: cosine = 0.6
	//   facts[1] embedding [0,1,0,0]: cosine = 0.0
	queryVec := []float32{1, 0, 0, 0}
	candidates, err := s.GlobalSearch(ctx, queryVec, 10)
	if err != nil {
		t.Fatalf("GlobalSearch: %v", err)
	}
	if len(candidates) != 3 {
		t.Fatalf("GlobalSearch returned %d candidates, want 3", len(candidates))
	}

	if candidates[0].Fact.ID != ids[0] {
		t.Errorf("rank 0: got id %s, want %s", candidates[0].Fact.ID, ids[0])
	}
	if candidates[1].Fact.ID != ids[2] {
		t.Errorf("rank 1: got id %s, want %s", candidates[1].Fact.ID, ids[2])
	}
	if candidates[2].Fact.ID != ids[1] {
		t.Errorf("rank 2: got id %s, want %s", candidates[2].Fact.ID, ids[1])
	}

	// Verify similarity values are descending.
	for i := 1; i < len(candidates); i++ {
		if candidates[i].Similarity > candidates[i-1].Similarity {
			t.Errorf("candidates not sorted: [%d].Similarity=%v > [%d].Similarity=%v",
				i, candidates[i].Similarity, i-1, candidates[i-1].Similarity)
		}
	}
}

func TestSQLiteGlobalSearchLimit(t *testing.T) {
	s := openTestDB(t)
	ctx := context.Background()

	for _, f := range testdataFacts() {
		if _, err := s.InsertFact(ctx, f); err != nil {
			t.Fatalf("InsertFact: %v", err)
		}
	}

	candidates, err := s.GlobalSearch(ctx, []float32{1, 0, 0, 0}, 2)
	if err != nil {
		t.Fatalf("GlobalSearch: %v", err)
	}
	if len(candidates) != 2 {
		t.Errorf("GlobalSearch(limit=2) returned %d, want 2", len(candidates))
	}
}

func TestSQLiteTouchAccessed(t *testing.T) {
	s := openTestDB(t)
	ctx := context.Background()

	facts := testdataFacts()
	id, err := s.InsertFact(ctx, facts[0])
	if err != nil {
		t.Fatalf("InsertFact: %v", err)
	}

	before, err := s.GetFact(ctx, id)
	if err != nil {
		t.Fatalf("GetFact before touch: %v", err)
	}

	// Small delay to ensure time difference.
	time.Sleep(10 * time.Millisecond)

	if err := s.TouchAccessed(ctx, []string{id}); err != nil {
		t.Fatalf("TouchAccessed: %v", err)
	}

	after, err := s.GetFact(ctx, id)
	if err != nil {
		t.Fatalf("GetFact after touch: %v", err)
	}

	if !after.AccessedAt.After(before.AccessedAt) {
		t.Errorf("AccessedAt not updated: before=%v, after=%v", before.AccessedAt, after.AccessedAt)
	}

	// TouchAccessed with empty slice should not error.
	if err := s.TouchAccessed(ctx, nil); err != nil {
		t.Errorf("TouchAccessed(nil) = %v, want nil", err)
	}
}

func TestSQLiteSupersededFiltering(t *testing.T) {
	s := openTestDB(t)
	ctx := context.Background()

	facts := testdataFacts()

	id1, err := s.InsertFact(ctx, facts[0])
	if err != nil {
		t.Fatalf("InsertFact[0]: %v", err)
	}

	// Insert a second fact that will supersede the first.
	f2 := facts[1]
	id2, err := s.InsertFact(ctx, f2)
	if err != nil {
		t.Fatalf("InsertFact[1]: %v", err)
	}

	// Mark first fact as superseded by second via raw SQL.
	_, err = s.db.ExecContext(ctx, `UPDATE facts SET superseded_by = ? WHERE id = ?`, id2, id1)
	if err != nil {
		t.Fatalf("raw UPDATE superseded_by: %v", err)
	}

	// ListFacts should exclude the superseded fact.
	results, err := s.ListFacts(ctx, ListFilter{})
	if err != nil {
		t.Fatalf("ListFacts: %v", err)
	}
	for _, r := range results {
		if r.ID == id1 {
			t.Error("ListFacts included superseded fact")
		}
	}
	if len(results) != 1 {
		t.Errorf("ListFacts returned %d results, want 1", len(results))
	}
	if results[0].ID != id2 {
		t.Errorf("ListFacts returned id=%s, want %s", results[0].ID, id2)
	}

	// GlobalSearch should also exclude the superseded fact.
	candidates, err := s.GlobalSearch(ctx, []float32{1, 0, 0, 0}, 10)
	if err != nil {
		t.Fatalf("GlobalSearch: %v", err)
	}
	for _, c := range candidates {
		if c.Fact.ID == id1 {
			t.Error("GlobalSearch included superseded fact")
		}
	}
}

func TestSQLiteGetFactNotFound(t *testing.T) {
	s := openTestDB(t)
	ctx := context.Background()

	got, err := s.GetFact(ctx, "nonexistent-id")
	if err != nil {
		t.Fatalf("GetFact(nonexistent) = %v, want nil error", err)
	}
	if got != nil {
		t.Error("GetFact(nonexistent) returned non-nil")
	}
}

func TestSQLiteListFactsDefaultSort(t *testing.T) {
	s := openTestDB(t)
	ctx := context.Background()

	// Insert facts in order; default sort is created_at DESC.
	for _, f := range testdataFacts() {
		if _, err := s.InsertFact(ctx, f); err != nil {
			t.Fatalf("InsertFact: %v", err)
		}
	}

	results, err := s.ListFacts(ctx, ListFilter{})
	if err != nil {
		t.Fatalf("ListFacts: %v", err)
	}
	if len(results) != 3 {
		t.Fatalf("ListFacts returned %d, want 3", len(results))
	}

	// Most recent first.
	for i := 1; i < len(results); i++ {
		if results[i].CreatedAt.After(results[i-1].CreatedAt) {
			t.Errorf("results not sorted by created_at DESC: [%d]=%v > [%d]=%v",
				i, results[i].CreatedAt, i-1, results[i-1].CreatedAt)
		}
	}
}

// --- Cluster method tests (Phase 2) ---

// insertTestClusterSQLite inserts a cluster directly via SQL for test setup.
func insertTestClusterSQLite(t *testing.T, s *sqliteStore, id string) {
	t.Helper()
	now := time.Now().UTC().Format(timeFormat)
	_, err := s.db.ExecContext(context.Background(),
		`INSERT INTO clusters (id, summary, utility, frequency, turns_since, last_access, created_at)
		 VALUES (?, ?, 0.0, 0.0, 0, ?, ?)`,
		id, id, now, now,
	)
	if err != nil {
		t.Fatalf("insertTestClusterSQLite(%q): %v", id, err)
	}
}

func TestSQLiteGetCluster_DefaultExists(t *testing.T) {
	s := openTestDB(t)
	ctx := context.Background()

	// InsertFact auto-creates the "default" cluster.
	facts := testdataFacts()
	if _, err := s.InsertFact(ctx, facts[0]); err != nil {
		t.Fatalf("InsertFact: %v", err)
	}

	c, err := s.GetCluster(ctx, "default")
	if err != nil {
		t.Fatalf("GetCluster: %v", err)
	}
	if c == nil {
		t.Fatal("GetCluster(\"default\") returned nil after InsertFact")
	}
	if c.ID != "default" {
		t.Errorf("cluster ID = %q, want %q", c.ID, "default")
	}
}

func TestSQLiteGetCluster_NotFound(t *testing.T) {
	s := openTestDB(t)
	ctx := context.Background()

	c, err := s.GetCluster(ctx, "nonexistent")
	if err != nil {
		t.Fatalf("GetCluster(\"nonexistent\") error = %v, want nil", err)
	}
	if c != nil {
		t.Errorf("GetCluster(\"nonexistent\") = %+v, want nil", c)
	}
}

func TestSQLiteListClusters_Empty(t *testing.T) {
	s := openTestDB(t)
	ctx := context.Background()

	// Fresh store: no facts inserted, so no clusters exist.
	// Design choice: ListClusters returns an empty slice on a fresh store
	// (the "default" cluster is only created lazily on the first InsertFact).
	clusters, err := s.ListClusters(ctx)
	if err != nil {
		t.Fatalf("ListClusters: %v", err)
	}
	if len(clusters) != 0 {
		t.Errorf("ListClusters on fresh store returned %d clusters, want 0", len(clusters))
	}
}

func TestSQLiteUpdateClusterState_Success(t *testing.T) {
	s := openTestDB(t)
	ctx := context.Background()

	// Insert a fact to create the default cluster.
	if _, err := s.InsertFact(ctx, testdataFacts()[0]); err != nil {
		t.Fatalf("InsertFact: %v", err)
	}

	// Update cluster state.
	if err := s.UpdateClusterState(ctx, "default", 0.75, 0.42, 5); err != nil {
		t.Fatalf("UpdateClusterState: %v", err)
	}

	c, err := s.GetCluster(ctx, "default")
	if err != nil {
		t.Fatalf("GetCluster: %v", err)
	}
	if c == nil {
		t.Fatal("GetCluster returned nil")
	}

	if c.Utility != 0.75 {
		t.Errorf("Utility = %v, want 0.75", c.Utility)
	}
	if c.Frequency != 0.42 {
		t.Errorf("Frequency = %v, want 0.42", c.Frequency)
	}
	if c.TurnsSince != 5 {
		t.Errorf("TurnsSince = %d, want 5", c.TurnsSince)
	}
}

func TestSQLiteUpdateClusterState_NotFound(t *testing.T) {
	s := openTestDB(t)
	ctx := context.Background()

	err := s.UpdateClusterState(ctx, "nonexistent", 0.5, 0.5, 0)
	if err == nil {
		t.Fatal("UpdateClusterState on nonexistent cluster should return error")
	}
}

func TestSQLiteTickAllClusters_Basic(t *testing.T) {
	s := openTestDB(t)
	ctx := context.Background()

	// Create two clusters directly.
	insertTestClusterSQLite(t, s, "A")
	insertTestClusterSQLite(t, s, "B")

	// Set initial turns_since for A=2, B=3 to make the test more interesting.
	if err := s.UpdateClusterState(ctx, "A", 0.5, 0.5, 2); err != nil {
		t.Fatalf("UpdateClusterState(A): %v", err)
	}
	if err := s.UpdateClusterState(ctx, "B", 0.5, 0.5, 3); err != nil {
		t.Fatalf("UpdateClusterState(B): %v", err)
	}

	// Tick with A accessed.
	if err := s.TickAllClusters(ctx, []string{"A"}); err != nil {
		t.Fatalf("TickAllClusters: %v", err)
	}

	a, err := s.GetCluster(ctx, "A")
	if err != nil {
		t.Fatalf("GetCluster(A): %v", err)
	}
	b, err := s.GetCluster(ctx, "B")
	if err != nil {
		t.Fatalf("GetCluster(B): %v", err)
	}

	// A was accessed: incremented from 2 to 3, then reset to 0.
	if a.TurnsSince != 0 {
		t.Errorf("A.TurnsSince = %d, want 0", a.TurnsSince)
	}
	// B was NOT accessed: incremented from 3 to 4.
	if b.TurnsSince != 4 {
		t.Errorf("B.TurnsSince = %d, want 4", b.TurnsSince)
	}
}

func TestSQLiteTickAllClusters_Empty(t *testing.T) {
	s := openTestDB(t)
	ctx := context.Background()

	// Create two clusters.
	insertTestClusterSQLite(t, s, "X")
	insertTestClusterSQLite(t, s, "Y")

	// Set initial state.
	if err := s.UpdateClusterState(ctx, "X", 0.5, 0.5, 1); err != nil {
		t.Fatalf("UpdateClusterState(X): %v", err)
	}
	if err := s.UpdateClusterState(ctx, "Y", 0.5, 0.5, 2); err != nil {
		t.Fatalf("UpdateClusterState(Y): %v", err)
	}

	// TickAllClusters with no accessed IDs: all clusters get incremented.
	if err := s.TickAllClusters(ctx, []string{}); err != nil {
		t.Fatalf("TickAllClusters: %v", err)
	}

	x, err := s.GetCluster(ctx, "X")
	if err != nil {
		t.Fatalf("GetCluster(X): %v", err)
	}
	y, err := s.GetCluster(ctx, "Y")
	if err != nil {
		t.Fatalf("GetCluster(Y): %v", err)
	}

	if x.TurnsSince != 2 {
		t.Errorf("X.TurnsSince = %d, want 2", x.TurnsSince)
	}
	if y.TurnsSince != 3 {
		t.Errorf("Y.TurnsSince = %d, want 3", y.TurnsSince)
	}
}

// --- Phase 3: Episode tests ---

func TestSQLiteInsertEpisode_RoundTrip(t *testing.T) {
	s := openTestDB(t)
	ctx := context.Background()

	episodes := testdataEpisodes()
	ep := episodes[0]
	// Pre-create a fact to link.
	factID, err := s.InsertFact(ctx, testdataFacts()[0])
	if err != nil {
		t.Fatalf("InsertFact: %v", err)
	}
	ep.LinkedFactIDs = []string{factID}

	id, err := s.InsertEpisode(ctx, ep)
	if err != nil {
		t.Fatalf("InsertEpisode: %v", err)
	}
	if id == "" {
		t.Fatal("InsertEpisode returned empty id")
	}

	got, err := s.GetEpisode(ctx, id)
	if err != nil {
		t.Fatalf("GetEpisode: %v", err)
	}
	if got == nil {
		t.Fatal("GetEpisode returned nil")
	}

	if got.Situation != ep.Situation {
		t.Errorf("Situation = %q, want %q", got.Situation, ep.Situation)
	}
	if got.Action != ep.Action {
		t.Errorf("Action = %q, want %q", got.Action, ep.Action)
	}
	if got.Outcome != ep.Outcome {
		t.Errorf("Outcome = %q, want %q", got.Outcome, ep.Outcome)
	}
	if got.Preemptive != ep.Preemptive {
		t.Errorf("Preemptive = %q, want %q", got.Preemptive, ep.Preemptive)
	}

	// Embedding round-trip.
	if len(got.Embedding) != len(ep.Embedding) {
		t.Fatalf("Embedding length = %d, want %d", len(got.Embedding), len(ep.Embedding))
	}
	for i := range ep.Embedding {
		if got.Embedding[i] != ep.Embedding[i] {
			t.Errorf("Embedding[%d] = %v, want %v", i, got.Embedding[i], ep.Embedding[i])
		}
	}

	// Linked fact IDs.
	if len(got.LinkedFactIDs) != 1 {
		t.Fatalf("LinkedFactIDs length = %d, want 1", len(got.LinkedFactIDs))
	}
	if got.LinkedFactIDs[0] != factID {
		t.Errorf("LinkedFactIDs[0] = %q, want %q", got.LinkedFactIDs[0], factID)
	}
}

func TestSQLiteListEpisodes_OrderAndFilter(t *testing.T) {
	s := openTestDB(t)
	ctx := context.Background()

	episodes := testdataEpisodes()
	for _, ep := range episodes {
		if _, err := s.InsertEpisode(ctx, ep); err != nil {
			t.Fatalf("InsertEpisode: %v", err)
		}
	}

	results, err := s.ListEpisodes(ctx, ListFilter{Sort: "created"})
	if err != nil {
		t.Fatalf("ListEpisodes: %v", err)
	}
	if len(results) != 3 {
		t.Fatalf("ListEpisodes returned %d, want 3", len(results))
	}

	// Most recent first (created_at DESC).
	for i := 1; i < len(results); i++ {
		if results[i].CreatedAt.After(results[i-1].CreatedAt) {
			t.Errorf("results not sorted by created_at DESC: [%d]=%v > [%d]=%v",
				i, results[i].CreatedAt, i-1, results[i-1].CreatedAt)
		}
	}
}

func TestSQLiteDeleteEpisode(t *testing.T) {
	s := openTestDB(t)
	ctx := context.Background()

	// Insert a fact and an episode, link them.
	factID, err := s.InsertFact(ctx, testdataFacts()[0])
	if err != nil {
		t.Fatalf("InsertFact: %v", err)
	}

	ep := testdataEpisodes()[0]
	epID, err := s.InsertEpisode(ctx, ep)
	if err != nil {
		t.Fatalf("InsertEpisode: %v", err)
	}

	if _, err := s.LinkFactEpisode(ctx, factID, epID, "evidence"); err != nil {
		t.Fatalf("LinkFactEpisode: %v", err)
	}

	// Delete episode.
	if err := s.DeleteEpisode(ctx, epID); err != nil {
		t.Fatalf("DeleteEpisode: %v", err)
	}

	got, err := s.GetEpisode(ctx, epID)
	if err != nil {
		t.Fatalf("GetEpisode after delete: %v", err)
	}
	if got != nil {
		t.Error("GetEpisode after delete returned non-nil")
	}

	// Links should be cascade-deleted.
	links, err := s.GetFactLinks(ctx, factID)
	if err != nil {
		t.Fatalf("GetFactLinks after episode delete: %v", err)
	}
	if len(links) != 0 {
		t.Errorf("expected 0 links after episode delete, got %d", len(links))
	}

	// Deleting a non-existent id should not error.
	if err := s.DeleteEpisode(ctx, "nonexistent"); err != nil {
		t.Errorf("DeleteEpisode(nonexistent) = %v, want nil", err)
	}
}

// --- Phase 3: Cross-link tests ---

func TestSQLiteLinkFactEpisode_Insert(t *testing.T) {
	s := openTestDB(t)
	ctx := context.Background()

	factID, err := s.InsertFact(ctx, testdataFacts()[0])
	if err != nil {
		t.Fatalf("InsertFact: %v", err)
	}
	epID, err := s.InsertEpisode(ctx, testdataEpisodes()[0])
	if err != nil {
		t.Fatalf("InsertEpisode: %v", err)
	}

	if _, err := s.LinkFactEpisode(ctx, factID, epID, "evidence"); err != nil {
		t.Fatalf("LinkFactEpisode: %v", err)
	}

	// Verify via GetFactLinks.
	epLinks, err := s.GetFactLinks(ctx, factID)
	if err != nil {
		t.Fatalf("GetFactLinks: %v", err)
	}
	if len(epLinks) != 1 {
		t.Fatalf("GetFactLinks returned %d links, want 1", len(epLinks))
	}
	if epLinks[0].EpisodeID != epID {
		t.Errorf("EpisodeID = %q, want %q", epLinks[0].EpisodeID, epID)
	}
	if epLinks[0].LinkType != "evidence" {
		t.Errorf("LinkType = %q, want %q", epLinks[0].LinkType, "evidence")
	}
	if epLinks[0].Episode == nil {
		t.Fatal("Episode not eager-loaded")
	}
	if epLinks[0].Episode.ID != epID {
		t.Errorf("eager-loaded Episode.ID = %q, want %q", epLinks[0].Episode.ID, epID)
	}

	// Verify via GetEpisodeLinks.
	fLinks, err := s.GetEpisodeLinks(ctx, epID)
	if err != nil {
		t.Fatalf("GetEpisodeLinks: %v", err)
	}
	if len(fLinks) != 1 {
		t.Fatalf("GetEpisodeLinks returned %d links, want 1", len(fLinks))
	}
	if fLinks[0].FactID != factID {
		t.Errorf("FactID = %q, want %q", fLinks[0].FactID, factID)
	}
	if fLinks[0].Fact == nil {
		t.Fatal("Fact not eager-loaded")
	}
	if fLinks[0].Fact.ID != factID {
		t.Errorf("eager-loaded Fact.ID = %q, want %q", fLinks[0].Fact.ID, factID)
	}
}

func TestSQLiteLinkFactEpisode_CascadeOnFactDelete(t *testing.T) {
	s := openTestDB(t)
	ctx := context.Background()

	factID, err := s.InsertFact(ctx, testdataFacts()[0])
	if err != nil {
		t.Fatalf("InsertFact: %v", err)
	}
	epID, err := s.InsertEpisode(ctx, testdataEpisodes()[0])
	if err != nil {
		t.Fatalf("InsertEpisode: %v", err)
	}
	if _, err := s.LinkFactEpisode(ctx, factID, epID, "evidence"); err != nil {
		t.Fatalf("LinkFactEpisode: %v", err)
	}

	// Delete fact — link should cascade.
	if err := s.DeleteFact(ctx, factID); err != nil {
		t.Fatalf("DeleteFact: %v", err)
	}

	links, err := s.GetEpisodeLinks(ctx, epID)
	if err != nil {
		t.Fatalf("GetEpisodeLinks: %v", err)
	}
	if len(links) != 0 {
		t.Errorf("expected 0 links after fact delete, got %d", len(links))
	}
}

func TestSQLiteLinkFactEpisode_CascadeOnEpisodeDelete(t *testing.T) {
	s := openTestDB(t)
	ctx := context.Background()

	factID, err := s.InsertFact(ctx, testdataFacts()[0])
	if err != nil {
		t.Fatalf("InsertFact: %v", err)
	}
	epID, err := s.InsertEpisode(ctx, testdataEpisodes()[0])
	if err != nil {
		t.Fatalf("InsertEpisode: %v", err)
	}
	if _, err := s.LinkFactEpisode(ctx, factID, epID, "evidence"); err != nil {
		t.Fatalf("LinkFactEpisode: %v", err)
	}

	// Delete episode — link should cascade.
	if err := s.DeleteEpisode(ctx, epID); err != nil {
		t.Fatalf("DeleteEpisode: %v", err)
	}

	links, err := s.GetFactLinks(ctx, factID)
	if err != nil {
		t.Fatalf("GetFactLinks: %v", err)
	}
	if len(links) != 0 {
		t.Errorf("expected 0 links after episode delete, got %d", len(links))
	}
}

// --- Phase 4A: LinkFactEpisode / UnlinkFactEpisode idempotency ---

func TestSQLiteLinkFactEpisode_CreatedFlag(t *testing.T) {
	s := openTestDB(t)
	ctx := context.Background()

	factID, err := s.InsertFact(ctx, testdataFacts()[0])
	if err != nil {
		t.Fatalf("InsertFact: %v", err)
	}
	epID, err := s.InsertEpisode(ctx, testdataEpisodes()[0])
	if err != nil {
		t.Fatalf("InsertEpisode: %v", err)
	}

	created, err := s.LinkFactEpisode(ctx, factID, epID, "evidence")
	if err != nil {
		t.Fatalf("LinkFactEpisode: %v", err)
	}
	if !created {
		t.Error("first LinkFactEpisode: created = false, want true")
	}

	created, err = s.LinkFactEpisode(ctx, factID, epID, "evidence")
	if err != nil {
		t.Fatalf("LinkFactEpisode repeat: %v", err)
	}
	if created {
		t.Error("repeat LinkFactEpisode: created = true, want false")
	}
}

func TestSQLiteUnlinkFactEpisode(t *testing.T) {
	s := openTestDB(t)
	ctx := context.Background()

	factID, err := s.InsertFact(ctx, testdataFacts()[0])
	if err != nil {
		t.Fatalf("InsertFact: %v", err)
	}
	epID, err := s.InsertEpisode(ctx, testdataEpisodes()[0])
	if err != nil {
		t.Fatalf("InsertEpisode: %v", err)
	}

	// Unlink with no existing link is a no-op.
	deleted, err := s.UnlinkFactEpisode(ctx, factID, epID)
	if err != nil {
		t.Fatalf("UnlinkFactEpisode (absent): %v", err)
	}
	if deleted {
		t.Error("UnlinkFactEpisode (absent): deleted = true, want false")
	}

	// Create the link, then unlink it.
	if _, err := s.LinkFactEpisode(ctx, factID, epID, "evidence"); err != nil {
		t.Fatalf("LinkFactEpisode: %v", err)
	}
	deleted, err = s.UnlinkFactEpisode(ctx, factID, epID)
	if err != nil {
		t.Fatalf("UnlinkFactEpisode: %v", err)
	}
	if !deleted {
		t.Error("UnlinkFactEpisode: deleted = false, want true")
	}

	// Verify link is gone.
	links, err := s.GetFactLinks(ctx, factID)
	if err != nil {
		t.Fatalf("GetFactLinks after unlink: %v", err)
	}
	if len(links) != 0 {
		t.Errorf("expected 0 links after unlink, got %d", len(links))
	}

	// Second unlink is a no-op.
	deleted, err = s.UnlinkFactEpisode(ctx, factID, epID)
	if err != nil {
		t.Fatalf("UnlinkFactEpisode repeat: %v", err)
	}
	if deleted {
		t.Error("UnlinkFactEpisode repeat: deleted = true, want false")
	}
}

// --- Phase 3: GlobalSearch polymorphism ---

func TestSQLiteGlobalSearch_MixesFactsAndEpisodes(t *testing.T) {
	s := openTestDB(t)
	ctx := context.Background()

	// Insert 2 facts.
	facts := testdataFacts()[:2]
	for _, f := range facts {
		if _, err := s.InsertFact(ctx, f); err != nil {
			t.Fatalf("InsertFact: %v", err)
		}
	}

	// Insert 2 episodes.
	episodes := testdataEpisodes()[:2]
	for _, ep := range episodes {
		if _, err := s.InsertEpisode(ctx, ep); err != nil {
			t.Fatalf("InsertEpisode: %v", err)
		}
	}

	// Query vector along x axis: [1,0,0,0].
	queryVec := []float32{1, 0, 0, 0}
	candidates, err := s.GlobalSearch(ctx, queryVec, 10)
	if err != nil {
		t.Fatalf("GlobalSearch: %v", err)
	}

	// Should have 4 candidates total: 2 facts + 2 episodes.
	if len(candidates) != 4 {
		t.Fatalf("GlobalSearch returned %d candidates, want 4", len(candidates))
	}

	// Verify mix: at least one fact and one episode.
	hasFact, hasEpisode := false, false
	for _, c := range candidates {
		if c.Fact != nil {
			hasFact = true
		}
		if c.Episode != nil {
			hasEpisode = true
		}
	}
	if !hasFact {
		t.Error("expected at least one fact candidate")
	}
	if !hasEpisode {
		t.Error("expected at least one episode candidate")
	}

	// Verify sorted by similarity descending.
	for i := 1; i < len(candidates); i++ {
		if candidates[i].Similarity > candidates[i-1].Similarity {
			t.Errorf("candidates not sorted: [%d].Similarity=%v > [%d].Similarity=%v",
				i, candidates[i].Similarity, i-1, candidates[i-1].Similarity)
		}
	}

	// Verify Candidate helpers work.
	top := candidates[0]
	if top.ID() == "" {
		t.Error("Candidate.ID() returned empty string")
	}
	if top.Content() == "" {
		t.Error("Candidate.Content() returned empty string")
	}
	if top.Layer() == "" {
		t.Error("Candidate.Layer() returned empty string")
	}
	if top.ClusterID() == "" {
		t.Error("Candidate.ClusterID() returned empty string")
	}
}

// --- Phase 3: Cluster centroid tests ---

func TestSQLiteCreateCluster_Basic(t *testing.T) {
	s := openTestDB(t)
	ctx := context.Background()

	c := ClusterNode{
		ID:        "test-cluster",
		Summary:   "A test cluster",
		Domain:    "testing",
		MetaInstr: "Be thorough",
		ItemCount: 5,
		Centroid:  []float32{0.5, 0.5, 0, 0},
		Utility:   0.75,
		Frequency: 0.42,
	}

	if err := s.CreateCluster(ctx, c); err != nil {
		t.Fatalf("CreateCluster: %v", err)
	}

	got, err := s.GetCluster(ctx, "test-cluster")
	if err != nil {
		t.Fatalf("GetCluster: %v", err)
	}
	if got == nil {
		t.Fatal("GetCluster returned nil")
	}
	if got.ID != c.ID {
		t.Errorf("ID = %q, want %q", got.ID, c.ID)
	}
	if got.Summary != c.Summary {
		t.Errorf("Summary = %q, want %q", got.Summary, c.Summary)
	}
	if got.Domain != c.Domain {
		t.Errorf("Domain = %q, want %q", got.Domain, c.Domain)
	}
	if got.MetaInstr != c.MetaInstr {
		t.Errorf("MetaInstr = %q, want %q", got.MetaInstr, c.MetaInstr)
	}
	if got.ItemCount != c.ItemCount {
		t.Errorf("ItemCount = %d, want %d", got.ItemCount, c.ItemCount)
	}
	if got.Utility != c.Utility {
		t.Errorf("Utility = %v, want %v", got.Utility, c.Utility)
	}
	if got.Frequency != c.Frequency {
		t.Errorf("Frequency = %v, want %v", got.Frequency, c.Frequency)
	}

	// Centroid round-trip.
	if len(got.Centroid) != len(c.Centroid) {
		t.Fatalf("Centroid length = %d, want %d", len(got.Centroid), len(c.Centroid))
	}
	for i := range c.Centroid {
		if got.Centroid[i] != c.Centroid[i] {
			t.Errorf("Centroid[%d] = %v, want %v", i, got.Centroid[i], c.Centroid[i])
		}
	}
}

func TestSQLiteUpdateClusterCentroid(t *testing.T) {
	s := openTestDB(t)
	ctx := context.Background()

	// Create a cluster first.
	c := ClusterNode{
		ID:        "centroid-test",
		Summary:   "centroid test",
		ItemCount: 1,
		Centroid:  []float32{1, 0, 0, 0},
	}
	if err := s.CreateCluster(ctx, c); err != nil {
		t.Fatalf("CreateCluster: %v", err)
	}

	// Update centroid and item count.
	newCentroid := []float32{0.5, 0.5, 0, 0}
	if err := s.UpdateClusterCentroid(ctx, "centroid-test", newCentroid, 3); err != nil {
		t.Fatalf("UpdateClusterCentroid: %v", err)
	}

	got, err := s.GetCluster(ctx, "centroid-test")
	if err != nil {
		t.Fatalf("GetCluster: %v", err)
	}
	if got == nil {
		t.Fatal("GetCluster returned nil")
	}
	if got.ItemCount != 3 {
		t.Errorf("ItemCount = %d, want 3", got.ItemCount)
	}
	if len(got.Centroid) != len(newCentroid) {
		t.Fatalf("Centroid length = %d, want %d", len(got.Centroid), len(newCentroid))
	}
	for i := range newCentroid {
		if got.Centroid[i] != newCentroid[i] {
			t.Errorf("Centroid[%d] = %v, want %v", i, got.Centroid[i], newCentroid[i])
		}
	}
}

// --- Phase 3: Temporal conflict resolution tests ---

func TestSQLiteFindSimilarFacts_Match(t *testing.T) {
	s := openTestDB(t)
	ctx := context.Background()

	// Insert two facts with same subtype and similar embeddings.
	f1 := Fact{
		Content:   "Go uses goroutines for concurrency.",
		Subtype:   "project",
		Source:    "inferred",
		Embedding: []float32{0.9, 0.1, 0, 0},
	}
	f2 := Fact{
		Content:   "Go leverages goroutines for concurrent work.",
		Subtype:   "project",
		Source:    "inferred",
		Embedding: []float32{0.85, 0.15, 0, 0},
	}

	if _, err := s.InsertFact(ctx, f1); err != nil {
		t.Fatalf("InsertFact[0]: %v", err)
	}
	if _, err := s.InsertFact(ctx, f2); err != nil {
		t.Fatalf("InsertFact[1]: %v", err)
	}

	// Query with a similar vector at a low threshold.
	queryVec := []float32{0.9, 0.1, 0, 0}
	results, err := s.FindSimilarFacts(ctx, "project", queryVec, 0.95, 10)
	if err != nil {
		t.Fatalf("FindSimilarFacts: %v", err)
	}

	// Both should match since their embeddings are very similar to query.
	if len(results) == 0 {
		t.Fatal("FindSimilarFacts returned 0 results, expected at least 1")
	}

	// At least one should have high similarity.
	if results[0].Similarity < 0.95 {
		t.Errorf("top result similarity = %v, want >= 0.95", results[0].Similarity)
	}
}

func TestSQLiteFindSimilarFacts_NoCrossSubtype(t *testing.T) {
	s := openTestDB(t)
	ctx := context.Background()

	// Insert two facts with DIFFERENT subtypes but similar embeddings.
	f1 := Fact{
		Content:   "User prefers vim.",
		Subtype:   "user",
		Source:    "conversation",
		Embedding: []float32{0.9, 0.1, 0, 0},
	}
	f2 := Fact{
		Content:   "Project uses vim config.",
		Subtype:   "project",
		Source:    "inferred",
		Embedding: []float32{0.9, 0.1, 0, 0}, // identical embedding
	}

	if _, err := s.InsertFact(ctx, f1); err != nil {
		t.Fatalf("InsertFact[0]: %v", err)
	}
	if _, err := s.InsertFact(ctx, f2); err != nil {
		t.Fatalf("InsertFact[1]: %v", err)
	}

	// Search for "user" subtype only.
	queryVec := []float32{0.9, 0.1, 0, 0}
	results, err := s.FindSimilarFacts(ctx, "user", queryVec, 0.90, 10)
	if err != nil {
		t.Fatalf("FindSimilarFacts: %v", err)
	}

	// Should only find the "user" subtype fact.
	for _, c := range results {
		if c.Fact.Subtype != "user" {
			t.Errorf("FindSimilarFacts returned subtype=%q, expected only 'user'", c.Fact.Subtype)
		}
	}
	if len(results) != 1 {
		t.Errorf("expected 1 result, got %d", len(results))
	}
}

func TestSQLiteFindSimilarFacts_ExcludesSuperseded(t *testing.T) {
	s := openTestDB(t)
	ctx := context.Background()

	f1 := Fact{
		Content:   "Old fact about Go.",
		Subtype:   "project",
		Source:    "inferred",
		Embedding: []float32{0.9, 0.1, 0, 0},
	}
	f2 := Fact{
		Content:   "New fact about Go.",
		Subtype:   "project",
		Source:    "inferred",
		Embedding: []float32{0.85, 0.15, 0, 0},
	}

	id1, err := s.InsertFact(ctx, f1)
	if err != nil {
		t.Fatalf("InsertFact[0]: %v", err)
	}
	id2, err := s.InsertFact(ctx, f2)
	if err != nil {
		t.Fatalf("InsertFact[1]: %v", err)
	}

	// Supersede f1 with f2.
	if err := s.SupersedeFact(ctx, id1, id2); err != nil {
		t.Fatalf("SupersedeFact: %v", err)
	}

	queryVec := []float32{0.9, 0.1, 0, 0}
	results, err := s.FindSimilarFacts(ctx, "project", queryVec, 0.80, 10)
	if err != nil {
		t.Fatalf("FindSimilarFacts: %v", err)
	}

	for _, c := range results {
		if c.Fact.ID == id1 {
			t.Error("FindSimilarFacts included superseded fact")
		}
	}
}

func TestSQLiteSupersedeFact(t *testing.T) {
	s := openTestDB(t)
	ctx := context.Background()

	facts := testdataFacts()
	id1, err := s.InsertFact(ctx, facts[0])
	if err != nil {
		t.Fatalf("InsertFact[0]: %v", err)
	}
	id2, err := s.InsertFact(ctx, facts[1])
	if err != nil {
		t.Fatalf("InsertFact[1]: %v", err)
	}

	// Supersede first with second.
	if err := s.SupersedeFact(ctx, id1, id2); err != nil {
		t.Fatalf("SupersedeFact: %v", err)
	}

	// Verify via direct read.
	f, err := s.GetFact(ctx, id1)
	if err != nil {
		t.Fatalf("GetFact: %v", err)
	}
	if f == nil {
		t.Fatal("GetFact returned nil")
	}
	if f.SupersededBy == nil || *f.SupersededBy != id2 {
		t.Errorf("SupersededBy = %v, want %q", f.SupersededBy, id2)
	}

	// ListFacts should exclude it.
	results, err := s.ListFacts(ctx, ListFilter{})
	if err != nil {
		t.Fatalf("ListFacts: %v", err)
	}
	for _, r := range results {
		if r.ID == id1 {
			t.Error("ListFacts included superseded fact")
		}
	}

	// SupersedeFact on nonexistent should error.
	if err := s.SupersedeFact(ctx, "nonexistent", id2); err == nil {
		t.Error("SupersedeFact on nonexistent should return error")
	}
}

// --- Phase 5: UpdateFactEmbedding / UpdateEpisodeEmbedding tests ---

func TestSQLiteUpdateFactEmbedding(t *testing.T) {
	s := openTestDB(t)
	ctx := context.Background()

	facts := testdataFacts()
	id, err := s.InsertFact(ctx, facts[0])
	if err != nil {
		t.Fatalf("InsertFact: %v", err)
	}

	// Verify original embedding.
	before, err := s.GetFact(ctx, id)
	if err != nil {
		t.Fatalf("GetFact before: %v", err)
	}
	if before.Embedding[0] != 1 || before.Embedding[1] != 0 {
		t.Fatalf("unexpected initial embedding: %v", before.Embedding)
	}

	// Update to a new embedding.
	newEmb := []float32{0, 0, 1, 0}
	if err := s.UpdateFactEmbedding(ctx, id, newEmb); err != nil {
		t.Fatalf("UpdateFactEmbedding: %v", err)
	}

	after, err := s.GetFact(ctx, id)
	if err != nil {
		t.Fatalf("GetFact after: %v", err)
	}
	if len(after.Embedding) != len(newEmb) {
		t.Fatalf("Embedding length = %d, want %d", len(after.Embedding), len(newEmb))
	}
	for i := range newEmb {
		if after.Embedding[i] != newEmb[i] {
			t.Errorf("Embedding[%d] = %v, want %v", i, after.Embedding[i], newEmb[i])
		}
	}
}

func TestSQLiteUpdateFactEmbedding_NotFound(t *testing.T) {
	s := openTestDB(t)
	ctx := context.Background()

	err := s.UpdateFactEmbedding(ctx, "nonexistent", []float32{1, 0, 0, 0})
	if err == nil {
		t.Fatal("UpdateFactEmbedding on nonexistent fact should return error")
	}
}

func TestSQLiteUpdateEpisodeEmbedding(t *testing.T) {
	s := openTestDB(t)
	ctx := context.Background()

	episodes := testdataEpisodes()
	id, err := s.InsertEpisode(ctx, episodes[0])
	if err != nil {
		t.Fatalf("InsertEpisode: %v", err)
	}

	// Verify original embedding.
	before, err := s.GetEpisode(ctx, id)
	if err != nil {
		t.Fatalf("GetEpisode before: %v", err)
	}
	if before.Embedding[0] != 0.7 {
		t.Fatalf("unexpected initial embedding: %v", before.Embedding)
	}

	// Update to a new embedding.
	newEmb := []float32{0, 0, 0, 1}
	if err := s.UpdateEpisodeEmbedding(ctx, id, newEmb); err != nil {
		t.Fatalf("UpdateEpisodeEmbedding: %v", err)
	}

	after, err := s.GetEpisode(ctx, id)
	if err != nil {
		t.Fatalf("GetEpisode after: %v", err)
	}
	if len(after.Embedding) != len(newEmb) {
		t.Fatalf("Embedding length = %d, want %d", len(after.Embedding), len(newEmb))
	}
	for i := range newEmb {
		if after.Embedding[i] != newEmb[i] {
			t.Errorf("Embedding[%d] = %v, want %v", i, after.Embedding[i], newEmb[i])
		}
	}
}

func TestSQLiteUpdateEpisodeEmbedding_NotFound(t *testing.T) {
	s := openTestDB(t)
	ctx := context.Background()

	err := s.UpdateEpisodeEmbedding(ctx, "nonexistent", []float32{1, 0, 0, 0})
	if err == nil {
		t.Fatal("UpdateEpisodeEmbedding on nonexistent episode should return error")
	}
}

// --- Phase 4: UpdateClusterMeta tests ---

func TestSQLiteUpdateClusterMeta_Basic(t *testing.T) {
	s := openTestDB(t)
	ctx := context.Background()

	// Create a cluster.
	c := ClusterNode{
		ID:        "meta-test",
		Summary:   "old summary",
		Domain:    "old domain",
		MetaInstr: "old meta",
		ItemCount: 3,
		Centroid:  []float32{0.5, 0.5, 0, 0},
	}
	if err := s.CreateCluster(ctx, c); err != nil {
		t.Fatalf("CreateCluster: %v", err)
	}

	// Update all three meta fields.
	if err := s.UpdateClusterMeta(ctx, "meta-test", "new summary", "new domain", "new meta"); err != nil {
		t.Fatalf("UpdateClusterMeta: %v", err)
	}

	got, err := s.GetCluster(ctx, "meta-test")
	if err != nil {
		t.Fatalf("GetCluster: %v", err)
	}
	if got == nil {
		t.Fatal("GetCluster returned nil")
	}
	if got.Summary != "new summary" {
		t.Errorf("Summary = %q, want %q", got.Summary, "new summary")
	}
	if got.Domain != "new domain" {
		t.Errorf("Domain = %q, want %q", got.Domain, "new domain")
	}
	if got.MetaInstr != "new meta" {
		t.Errorf("MetaInstr = %q, want %q", got.MetaInstr, "new meta")
	}

	// Verify other fields are untouched.
	if got.ItemCount != 3 {
		t.Errorf("ItemCount = %d, want 3 (should be unchanged)", got.ItemCount)
	}
}

func TestSQLiteUpdateClusterMeta_NotFound(t *testing.T) {
	s := openTestDB(t)
	ctx := context.Background()

	err := s.UpdateClusterMeta(ctx, "nonexistent", "summary", "domain", "meta")
	if err == nil {
		t.Fatal("UpdateClusterMeta on nonexistent cluster should return error")
	}
}

// --- GetFactSupersedes tests (Phase 1C) ---

func TestSQLiteGetFactSupersedes_Empty(t *testing.T) {
	s := openTestDB(t)
	ctx := context.Background()

	facts := testdataFacts()
	id, err := s.InsertFact(ctx, facts[0])
	if err != nil {
		t.Fatalf("InsertFact: %v", err)
	}

	got, err := s.GetFactSupersedes(ctx, id)
	if err != nil {
		t.Fatalf("GetFactSupersedes: %v", err)
	}
	if got == nil {
		t.Fatal("GetFactSupersedes returned nil; want empty slice")
	}
	if len(got) != 0 {
		t.Errorf("GetFactSupersedes returned %v, want empty slice", got)
	}
}

func TestSQLiteGetFactSupersedes_WithPredecessors(t *testing.T) {
	s := openTestDB(t)
	ctx := context.Background()

	f1 := Fact{
		Content:   "alpha fact",
		Subtype:   "project",
		Source:    "inferred",
		Embedding: []float32{1, 0, 0, 0},
	}
	f2 := Fact{
		Content:   "beta fact",
		Subtype:   "project",
		Source:    "inferred",
		Embedding: []float32{0, 1, 0, 0},
	}
	f3 := Fact{
		Content:   "gamma fact (new winner)",
		Subtype:   "project",
		Source:    "inferred",
		Embedding: []float32{0, 0, 1, 0},
	}

	id1, err := s.InsertFact(ctx, f1)
	if err != nil {
		t.Fatalf("InsertFact[1]: %v", err)
	}
	id2, err := s.InsertFact(ctx, f2)
	if err != nil {
		t.Fatalf("InsertFact[2]: %v", err)
	}
	id3, err := s.InsertFact(ctx, f3)
	if err != nil {
		t.Fatalf("InsertFact[3]: %v", err)
	}

	if err := s.SupersedeFact(ctx, id1, id3); err != nil {
		t.Fatalf("SupersedeFact(id1): %v", err)
	}
	if err := s.SupersedeFact(ctx, id2, id3); err != nil {
		t.Fatalf("SupersedeFact(id2): %v", err)
	}

	got, err := s.GetFactSupersedes(ctx, id3)
	if err != nil {
		t.Fatalf("GetFactSupersedes: %v", err)
	}
	if len(got) != 2 {
		t.Fatalf("got %d predecessors, want 2: %v", len(got), got)
	}
	has := map[string]bool{got[0]: true, got[1]: true}
	if !has[id1] || !has[id2] {
		t.Errorf("predecessors = %v, want [%s %s]", got, id1, id2)
	}

	// A fact that supersedes nothing returns an empty slice.
	got1, err := s.GetFactSupersedes(ctx, id1)
	if err != nil {
		t.Fatalf("GetFactSupersedes(id1): %v", err)
	}
	if len(got1) != 0 {
		t.Errorf("id1 predecessors = %v, want empty", got1)
	}
}

func TestSQLiteGetFactSupersedes_UnknownID(t *testing.T) {
	s := openTestDB(t)
	ctx := context.Background()

	got, err := s.GetFactSupersedes(ctx, "nonexistent")
	if err != nil {
		t.Fatalf("GetFactSupersedes: %v", err)
	}
	if len(got) != 0 {
		t.Errorf("unknown id returned %v, want empty slice", got)
	}
}

// --- ListFactsByCluster / ListEpisodesByCluster / Count*ByCluster (Phase 1E) ---

func TestSQLiteClusterMembership_PaginatedAndOrdered(t *testing.T) {
	s := openTestDB(t)
	ctx := context.Background()

	clusterID := "cluster-1e"
	if err := s.CreateCluster(ctx, ClusterNode{
		ID:         clusterID,
		Summary:    "test",
		CreatedAt:  time.Now().UTC(),
		LastAccess: time.Now().UTC(),
	}); err != nil {
		t.Fatalf("CreateCluster: %v", err)
	}

	base := time.Now().UTC().Add(-time.Hour)
	for i := 0; i < 5; i++ {
		_, err := s.InsertFact(ctx, Fact{
			ClusterID: clusterID,
			Content:   "f" + string(rune('a'+i)),
			Subtype:   "project",
			Source:    "inferred",
			Embedding: []float32{1, 0, 0, 0},
			CreatedAt: base.Add(time.Duration(i) * time.Second),
		})
		if err != nil {
			t.Fatalf("InsertFact %d: %v", i, err)
		}
	}
	for i := 0; i < 3; i++ {
		_, err := s.InsertEpisode(ctx, Episode{
			ClusterID: clusterID,
			Situation: "s" + string(rune('a'+i)),
			Embedding: []float32{1, 0, 0, 0},
			CreatedAt: base.Add(time.Duration(10+i) * time.Second),
		})
		if err != nil {
			t.Fatalf("InsertEpisode %d: %v", i, err)
		}
	}

	// Counts.
	fc, err := s.CountFactsByCluster(ctx, clusterID)
	if err != nil || fc != 5 {
		t.Fatalf("CountFactsByCluster = %d (err %v), want 5", fc, err)
	}
	ec, err := s.CountEpisodesByCluster(ctx, clusterID)
	if err != nil || ec != 3 {
		t.Fatalf("CountEpisodesByCluster = %d (err %v), want 3", ec, err)
	}

	// Paginated list across two pages — order stable, no gaps.
	p1, err := s.ListFactsByCluster(ctx, clusterID, 3, 0)
	if err != nil {
		t.Fatalf("ListFactsByCluster p1: %v", err)
	}
	p2, err := s.ListFactsByCluster(ctx, clusterID, 3, 3)
	if err != nil {
		t.Fatalf("ListFactsByCluster p2: %v", err)
	}
	if len(p1) != 3 || len(p2) != 2 {
		t.Fatalf("pages = %d+%d, want 3+2", len(p1), len(p2))
	}
	// Ascending ordering.
	all := append(append([]Fact{}, p1...), p2...)
	for i := 1; i < len(all); i++ {
		if all[i].CreatedAt.Before(all[i-1].CreatedAt) {
			t.Errorf("facts not ordered ASC: %v", all)
		}
	}
}

func TestSQLiteClusterMembership_ExcludesSuperseded(t *testing.T) {
	s := openTestDB(t)
	ctx := context.Background()

	clusterID := "cluster-super"
	if err := s.CreateCluster(ctx, ClusterNode{
		ID: clusterID, Summary: "t", CreatedAt: time.Now().UTC(), LastAccess: time.Now().UTC(),
	}); err != nil {
		t.Fatalf("CreateCluster: %v", err)
	}

	oldID, err := s.InsertFact(ctx, Fact{
		ClusterID: clusterID, Content: "old", Subtype: "project", Source: "inferred",
		Embedding: []float32{1, 0, 0, 0},
	})
	if err != nil {
		t.Fatalf("InsertFact old: %v", err)
	}
	newID, err := s.InsertFact(ctx, Fact{
		ClusterID: clusterID, Content: "new", Subtype: "project", Source: "inferred",
		Embedding: []float32{0, 1, 0, 0},
	})
	if err != nil {
		t.Fatalf("InsertFact new: %v", err)
	}
	if err := s.SupersedeFact(ctx, oldID, newID); err != nil {
		t.Fatalf("SupersedeFact: %v", err)
	}

	got, err := s.ListFactsByCluster(ctx, clusterID, 10, 0)
	if err != nil {
		t.Fatalf("ListFactsByCluster: %v", err)
	}
	if len(got) != 1 || got[0].ID != newID {
		t.Errorf("got %d facts (IDs=%v), want [%s]", len(got), idsOf(got), newID)
	}
	fc, _ := s.CountFactsByCluster(ctx, clusterID)
	if fc != 1 {
		t.Errorf("CountFactsByCluster = %d, want 1", fc)
	}
}

func TestSQLiteClusterMembership_UnknownCluster(t *testing.T) {
	s := openTestDB(t)
	ctx := context.Background()

	if got, err := s.ListFactsByCluster(ctx, "missing", 10, 0); err != nil || len(got) != 0 {
		t.Errorf("ListFactsByCluster missing = (%v, %v), want (empty, nil)", got, err)
	}
	if got, err := s.ListEpisodesByCluster(ctx, "missing", 10, 0); err != nil || len(got) != 0 {
		t.Errorf("ListEpisodesByCluster missing = (%v, %v), want (empty, nil)", got, err)
	}
	if c, err := s.CountFactsByCluster(ctx, "missing"); err != nil || c != 0 {
		t.Errorf("CountFactsByCluster missing = (%d, %v), want (0, nil)", c, err)
	}
	if c, err := s.CountEpisodesByCluster(ctx, "missing"); err != nil || c != 0 {
		t.Errorf("CountEpisodesByCluster missing = (%d, %v), want (0, nil)", c, err)
	}
}

func idsOf(fs []Fact) []string {
	out := make([]string, len(fs))
	for i, f := range fs {
		out[i] = f.ID
	}
	return out
}

// --- Phase 2A: SetMemoryCluster + DeleteCluster ---

func TestSQLiteSetMemoryCluster_Fact(t *testing.T) {
	s := openTestDB(t)
	ctx := context.Background()

	if err := s.CreateCluster(ctx, ClusterNode{ID: "target"}); err != nil {
		t.Fatalf("CreateCluster target: %v", err)
	}
	if err := s.CreateCluster(ctx, ClusterNode{ID: "source"}); err != nil {
		t.Fatalf("CreateCluster source: %v", err)
	}

	f := testdataFacts()[0]
	f.ClusterID = "source"
	id, err := s.InsertFact(ctx, f)
	if err != nil {
		t.Fatalf("InsertFact: %v", err)
	}

	before, err := s.GetFact(ctx, id)
	if err != nil {
		t.Fatalf("GetFact before: %v", err)
	}
	time.Sleep(1100 * time.Millisecond) // RFC3339 has 1-second granularity

	if err := s.SetMemoryCluster(ctx, id, "target"); err != nil {
		t.Fatalf("SetMemoryCluster: %v", err)
	}
	after, err := s.GetFact(ctx, id)
	if err != nil {
		t.Fatalf("GetFact after: %v", err)
	}
	if after.ClusterID != "target" {
		t.Errorf("ClusterID = %q, want target", after.ClusterID)
	}
	if !after.AccessedAt.After(before.AccessedAt) {
		t.Errorf("AccessedAt not bumped: before=%v after=%v", before.AccessedAt, after.AccessedAt)
	}
}

func TestSQLiteSetMemoryCluster_Episode(t *testing.T) {
	s := openTestDB(t)
	ctx := context.Background()

	if err := s.CreateCluster(ctx, ClusterNode{ID: "target"}); err != nil {
		t.Fatalf("CreateCluster target: %v", err)
	}
	if err := s.CreateCluster(ctx, ClusterNode{ID: "source"}); err != nil {
		t.Fatalf("CreateCluster source: %v", err)
	}

	ep := testdataEpisodes()[0]
	ep.ClusterID = "source"
	id, err := s.InsertEpisode(ctx, ep)
	if err != nil {
		t.Fatalf("InsertEpisode: %v", err)
	}

	before, err := s.GetEpisode(ctx, id)
	if err != nil {
		t.Fatalf("GetEpisode before: %v", err)
	}
	time.Sleep(1100 * time.Millisecond)

	if err := s.SetMemoryCluster(ctx, id, "target"); err != nil {
		t.Fatalf("SetMemoryCluster: %v", err)
	}
	after, err := s.GetEpisode(ctx, id)
	if err != nil {
		t.Fatalf("GetEpisode after: %v", err)
	}
	if after.ClusterID != "target" {
		t.Errorf("ClusterID = %q, want target", after.ClusterID)
	}
	if !after.AccessedAt.After(before.AccessedAt) {
		t.Errorf("AccessedAt not bumped: before=%v after=%v", before.AccessedAt, after.AccessedAt)
	}
}

func TestSQLiteSetMemoryCluster_NotFound(t *testing.T) {
	s := openTestDB(t)
	ctx := context.Background()
	err := s.SetMemoryCluster(ctx, "ghost", "anywhere")
	if err == nil {
		t.Fatal("expected error for missing memory")
	}
	if !strings.Contains(err.Error(), "memory not found: ghost") {
		t.Errorf("error = %q, want it to contain 'memory not found: ghost'", err)
	}
}

func TestSQLiteDeleteCluster_Empty(t *testing.T) {
	s := openTestDB(t)
	ctx := context.Background()

	if err := s.CreateCluster(ctx, ClusterNode{ID: "orphan"}); err != nil {
		t.Fatalf("CreateCluster: %v", err)
	}
	if err := s.DeleteCluster(ctx, "orphan"); err != nil {
		t.Fatalf("DeleteCluster: %v", err)
	}
	got, err := s.GetCluster(ctx, "orphan")
	if err != nil {
		t.Fatalf("GetCluster: %v", err)
	}
	if got != nil {
		t.Errorf("expected cluster to be gone, got %+v", got)
	}
}

func TestSQLiteDeleteCluster_Idempotent(t *testing.T) {
	s := openTestDB(t)
	ctx := context.Background()
	if err := s.DeleteCluster(ctx, "never-existed"); err != nil {
		t.Errorf("DeleteCluster missing = %v, want nil", err)
	}
}

func TestSQLiteDeleteCluster_RefusesNonEmpty(t *testing.T) {
	s := openTestDB(t)
	ctx := context.Background()

	if err := s.CreateCluster(ctx, ClusterNode{ID: "has-members"}); err != nil {
		t.Fatalf("CreateCluster: %v", err)
	}
	f := testdataFacts()[0]
	f.ClusterID = "has-members"
	if _, err := s.InsertFact(ctx, f); err != nil {
		t.Fatalf("InsertFact: %v", err)
	}

	err := s.DeleteCluster(ctx, "has-members")
	if err == nil {
		t.Fatal("expected error deleting non-empty cluster")
	}
	if !strings.Contains(err.Error(), "cluster not empty") {
		t.Errorf("error = %q, want it to contain 'cluster not empty'", err)
	}
}

// --- Phase 2A: RecomputeCentroid ---

func TestSQLite_RecomputeCentroid_HappyPath(t *testing.T) {
	s := openTestDB(t)
	ctx := context.Background()

	if err := s.CreateCluster(ctx, ClusterNode{
		ID:        "c1",
		Centroid:  []float32{9, 9, 9, 9},
		ItemCount: 99,
	}); err != nil {
		t.Fatalf("CreateCluster: %v", err)
	}
	f1 := Fact{Content: "a", Subtype: "project", ClusterID: "c1", Embedding: []float32{1, 0, 0, 0}}
	f2 := Fact{Content: "b", Subtype: "project", ClusterID: "c1", Embedding: []float32{0, 1, 0, 0}}
	ep := Episode{Situation: "sit", Action: "act", Outcome: "out", Preemptive: "p", ClusterID: "c1", Embedding: []float32{0, 0, 1, 0}}
	if _, err := s.InsertFact(ctx, f1); err != nil {
		t.Fatalf("InsertFact f1: %v", err)
	}
	if _, err := s.InsertFact(ctx, f2); err != nil {
		t.Fatalf("InsertFact f2: %v", err)
	}
	if _, err := s.InsertEpisode(ctx, ep); err != nil {
		t.Fatalf("InsertEpisode: %v", err)
	}

	if err := RecomputeCentroid(ctx, s, "c1"); err != nil {
		t.Fatalf("RecomputeCentroid: %v", err)
	}
	got, err := s.GetCluster(ctx, "c1")
	if err != nil {
		t.Fatalf("GetCluster: %v", err)
	}
	want := []float32{1.0 / 3, 1.0 / 3, 1.0 / 3, 0}
	if len(got.Centroid) != len(want) {
		t.Fatalf("centroid len = %d, want %d", len(got.Centroid), len(want))
	}
	const eps = 1e-6
	for i := range want {
		diff := float64(got.Centroid[i] - want[i])
		if diff < -eps || diff > eps {
			t.Errorf("centroid[%d] = %v, want %v", i, got.Centroid[i], want[i])
		}
	}
	if got.ItemCount != 3 {
		t.Errorf("ItemCount = %d, want 3", got.ItemCount)
	}
}

func TestSQLite_RecomputeCentroid_EmptyCluster(t *testing.T) {
	s := openTestDB(t)
	ctx := context.Background()
	if err := s.CreateCluster(ctx, ClusterNode{ID: "empty"}); err != nil {
		t.Fatalf("CreateCluster: %v", err)
	}
	err := RecomputeCentroid(ctx, s, "empty")
	if err != ErrEmptyCluster {
		t.Fatalf("RecomputeCentroid empty = %v, want ErrEmptyCluster", err)
	}
}

// --- Phase 2B: MoveAllClusterMembers ---

func TestSQLiteMoveAllClusterMembers_FactsAndEpisodes(t *testing.T) {
	s := openTestDB(t)
	ctx := context.Background()

	if err := s.CreateCluster(ctx, ClusterNode{ID: "src"}); err != nil {
		t.Fatalf("CreateCluster src: %v", err)
	}
	if err := s.CreateCluster(ctx, ClusterNode{ID: "dst"}); err != nil {
		t.Fatalf("CreateCluster dst: %v", err)
	}

	// Seed: 2 facts + 1 episode in src, 1 fact in dst (to prove we don't touch it).
	f1 := testdataFacts()[0]
	f1.ClusterID = "src"
	f1ID, err := s.InsertFact(ctx, f1)
	if err != nil {
		t.Fatalf("InsertFact f1: %v", err)
	}
	f2 := testdataFacts()[1]
	f2.ClusterID = "src"
	f2ID, err := s.InsertFact(ctx, f2)
	if err != nil {
		t.Fatalf("InsertFact f2: %v", err)
	}
	ep := testdataEpisodes()[0]
	ep.ClusterID = "src"
	epID, err := s.InsertEpisode(ctx, ep)
	if err != nil {
		t.Fatalf("InsertEpisode: %v", err)
	}
	f3 := testdataFacts()[2]
	f3.ClusterID = "dst"
	f3ID, err := s.InsertFact(ctx, f3)
	if err != nil {
		t.Fatalf("InsertFact f3: %v", err)
	}

	time.Sleep(1100 * time.Millisecond) // RFC3339 1-second granularity

	moved, err := s.MoveAllClusterMembers(ctx, "src", "dst")
	if err != nil {
		t.Fatalf("MoveAllClusterMembers: %v", err)
	}
	if moved != 3 {
		t.Errorf("moved = %d, want 3", moved)
	}

	// Facts moved.
	got1, _ := s.GetFact(ctx, f1ID)
	if got1.ClusterID != "dst" {
		t.Errorf("f1.ClusterID = %q, want dst", got1.ClusterID)
	}
	if !got1.AccessedAt.After(f1.AccessedAt) {
		t.Errorf("f1.AccessedAt not bumped")
	}
	got2, _ := s.GetFact(ctx, f2ID)
	if got2.ClusterID != "dst" {
		t.Errorf("f2.ClusterID = %q, want dst", got2.ClusterID)
	}
	// Episode moved.
	gotEp, _ := s.GetEpisode(ctx, epID)
	if gotEp.ClusterID != "dst" {
		t.Errorf("ep.ClusterID = %q, want dst", gotEp.ClusterID)
	}
	// Pre-existing dst fact untouched.
	got3, _ := s.GetFact(ctx, f3ID)
	if got3.ClusterID != "dst" {
		t.Errorf("f3.ClusterID = %q, want dst (unchanged)", got3.ClusterID)
	}

	// Counts: src empty, dst holds 4.
	n, err := s.CountFactsByCluster(ctx, "src")
	if err != nil || n != 0 {
		t.Errorf("CountFactsByCluster src = %d, %v; want 0, nil", n, err)
	}
	n, err = s.CountEpisodesByCluster(ctx, "src")
	if err != nil || n != 0 {
		t.Errorf("CountEpisodesByCluster src = %d, %v; want 0, nil", n, err)
	}
	nf, _ := s.CountFactsByCluster(ctx, "dst")
	ne, _ := s.CountEpisodesByCluster(ctx, "dst")
	if nf+ne != 4 {
		t.Errorf("dst total = %d, want 4", nf+ne)
	}
}

func TestSQLiteMoveAllClusterMembers_EmptySource(t *testing.T) {
	s := openTestDB(t)
	ctx := context.Background()

	if err := s.CreateCluster(ctx, ClusterNode{ID: "src"}); err != nil {
		t.Fatalf("CreateCluster src: %v", err)
	}
	if err := s.CreateCluster(ctx, ClusterNode{ID: "dst"}); err != nil {
		t.Fatalf("CreateCluster dst: %v", err)
	}

	moved, err := s.MoveAllClusterMembers(ctx, "src", "dst")
	if err != nil {
		t.Fatalf("MoveAllClusterMembers: %v", err)
	}
	if moved != 0 {
		t.Errorf("moved = %d, want 0", moved)
	}
}

func TestSQLiteMoveAllClusterMembers_IgnoresSuperseded(t *testing.T) {
	s := openTestDB(t)
	ctx := context.Background()

	if err := s.CreateCluster(ctx, ClusterNode{ID: "src"}); err != nil {
		t.Fatalf("CreateCluster src: %v", err)
	}
	if err := s.CreateCluster(ctx, ClusterNode{ID: "dst"}); err != nil {
		t.Fatalf("CreateCluster dst: %v", err)
	}

	f1 := testdataFacts()[0]
	f1.ClusterID = "src"
	f1ID, err := s.InsertFact(ctx, f1)
	if err != nil {
		t.Fatalf("InsertFact f1: %v", err)
	}
	f2 := testdataFacts()[1]
	f2.ClusterID = "src"
	f2ID, err := s.InsertFact(ctx, f2)
	if err != nil {
		t.Fatalf("InsertFact f2: %v", err)
	}
	// Mark f1 as superseded — it should NOT be moved (non-superseded only).
	if err := s.SupersedeFact(ctx, f1ID, f2ID); err != nil {
		t.Fatalf("SupersedeFact: %v", err)
	}

	moved, err := s.MoveAllClusterMembers(ctx, "src", "dst")
	if err != nil {
		t.Fatalf("MoveAllClusterMembers: %v", err)
	}
	if moved != 1 {
		t.Errorf("moved = %d, want 1 (f1 is superseded)", moved)
	}
	// f1 still in src; f2 moved.
	got1, _ := s.GetFact(ctx, f1ID)
	if got1.ClusterID != "src" {
		t.Errorf("f1 (superseded) moved: ClusterID = %q, want src", got1.ClusterID)
	}
	got2, _ := s.GetFact(ctx, f2ID)
	if got2.ClusterID != "dst" {
		t.Errorf("f2.ClusterID = %q, want dst", got2.ClusterID)
	}
}

// --- Phase 2D: UpdateFactContent / UpdateEpisodeContent / ReplaceEpisodeLinks ---

func TestSQLiteUpdateFactContent_HappyPath(t *testing.T) {
	s := openTestDB(t)
	ctx := context.Background()

	f := testdataFacts()[0]
	f.Tags = []string{"alpha", "beta"}
	id, err := s.InsertFact(ctx, f)
	if err != nil {
		t.Fatalf("InsertFact: %v", err)
	}
	before, err := s.GetFact(ctx, id)
	if err != nil {
		t.Fatalf("GetFact before: %v", err)
	}
	time.Sleep(1100 * time.Millisecond) // RFC3339 1-second granularity

	newTags := []string{"gamma"}
	if err := s.UpdateFactContent(ctx, id, "new content", "newhash", []float32{0, 0, 1, 0}, &newTags); err != nil {
		t.Fatalf("UpdateFactContent: %v", err)
	}

	after, err := s.GetFact(ctx, id)
	if err != nil {
		t.Fatalf("GetFact after: %v", err)
	}
	if after.Content != "new content" {
		t.Errorf("Content = %q, want 'new content'", after.Content)
	}
	if after.ContentHash != "newhash" {
		t.Errorf("ContentHash = %q, want newhash", after.ContentHash)
	}
	if len(after.Embedding) != 4 || after.Embedding[2] != 1 {
		t.Errorf("Embedding = %v, want [0,0,1,0]", after.Embedding)
	}
	if len(after.Tags) != 1 || after.Tags[0] != "gamma" {
		t.Errorf("Tags = %v, want [gamma]", after.Tags)
	}
	if !after.AccessedAt.After(before.AccessedAt) {
		t.Errorf("AccessedAt not bumped: before=%v after=%v", before.AccessedAt, after.AccessedAt)
	}
	// Preserved fields.
	if !after.CreatedAt.Equal(before.CreatedAt) {
		t.Errorf("CreatedAt changed: before=%v after=%v", before.CreatedAt, after.CreatedAt)
	}
	if after.ClusterID != before.ClusterID {
		t.Errorf("ClusterID changed: before=%q after=%q", before.ClusterID, after.ClusterID)
	}
	if !after.ValidFrom.Equal(before.ValidFrom) {
		t.Errorf("ValidFrom changed: before=%v after=%v", before.ValidFrom, after.ValidFrom)
	}
}

func TestSQLiteUpdateFactContent_TagsNilPreserves(t *testing.T) {
	s := openTestDB(t)
	ctx := context.Background()

	f := testdataFacts()[0]
	f.Tags = []string{"keep", "me"}
	id, err := s.InsertFact(ctx, f)
	if err != nil {
		t.Fatalf("InsertFact: %v", err)
	}

	if err := s.UpdateFactContent(ctx, id, "updated", "h", []float32{1, 0, 0, 0}, nil); err != nil {
		t.Fatalf("UpdateFactContent: %v", err)
	}
	got, err := s.GetFact(ctx, id)
	if err != nil {
		t.Fatalf("GetFact: %v", err)
	}
	if len(got.Tags) != 2 || got.Tags[0] != "keep" || got.Tags[1] != "me" {
		t.Errorf("Tags = %v, want [keep me]", got.Tags)
	}
}

func TestSQLiteUpdateFactContent_TagsEmptyClears(t *testing.T) {
	s := openTestDB(t)
	ctx := context.Background()

	f := testdataFacts()[0]
	f.Tags = []string{"will", "go"}
	id, err := s.InsertFact(ctx, f)
	if err != nil {
		t.Fatalf("InsertFact: %v", err)
	}

	empty := []string{}
	if err := s.UpdateFactContent(ctx, id, "updated", "h", []float32{1, 0, 0, 0}, &empty); err != nil {
		t.Fatalf("UpdateFactContent: %v", err)
	}
	got, err := s.GetFact(ctx, id)
	if err != nil {
		t.Fatalf("GetFact: %v", err)
	}
	if len(got.Tags) != 0 {
		t.Errorf("Tags = %v, want empty", got.Tags)
	}
}

func TestSQLiteUpdateFactContent_NotFound(t *testing.T) {
	s := openTestDB(t)
	ctx := context.Background()
	err := s.UpdateFactContent(ctx, "ghost", "x", "y", []float32{1}, nil)
	if err == nil {
		t.Fatal("expected error for missing fact")
	}
	if !strings.Contains(err.Error(), "fact \"ghost\" not found") {
		t.Errorf("err = %q, want it to mention missing fact", err)
	}
}

func TestSQLiteUpdateEpisodeContent_HappyPath(t *testing.T) {
	s := openTestDB(t)
	ctx := context.Background()

	ep := testdataEpisodes()[0]
	ep.Tags = []string{"orig"}
	id, err := s.InsertEpisode(ctx, ep)
	if err != nil {
		t.Fatalf("InsertEpisode: %v", err)
	}
	before, err := s.GetEpisode(ctx, id)
	if err != nil {
		t.Fatalf("GetEpisode before: %v", err)
	}
	time.Sleep(1100 * time.Millisecond)

	updated := Episode{
		Situation:   "newSit",
		Action:      "newAct",
		Outcome:     "newOut",
		Preemptive:  "newPre",
		Embedding:   []float32{0, 0, 1, 0},
		ContentHash: "newhash",
		Tags:        []string{"fresh"},
	}
	if err := s.UpdateEpisodeContent(ctx, id, updated); err != nil {
		t.Fatalf("UpdateEpisodeContent: %v", err)
	}

	after, err := s.GetEpisode(ctx, id)
	if err != nil {
		t.Fatalf("GetEpisode after: %v", err)
	}
	if after.Situation != "newSit" || after.Action != "newAct" || after.Outcome != "newOut" || after.Preemptive != "newPre" {
		t.Errorf("episode fields = %+v, want the new values", after)
	}
	if after.ContentHash != "newhash" {
		t.Errorf("ContentHash = %q, want newhash", after.ContentHash)
	}
	if len(after.Tags) != 1 || after.Tags[0] != "fresh" {
		t.Errorf("Tags = %v, want [fresh]", after.Tags)
	}
	if !after.AccessedAt.After(before.AccessedAt) {
		t.Errorf("AccessedAt not bumped: before=%v after=%v", before.AccessedAt, after.AccessedAt)
	}
	if !after.CreatedAt.Equal(before.CreatedAt) {
		t.Errorf("CreatedAt changed: before=%v after=%v", before.CreatedAt, after.CreatedAt)
	}
}

func TestSQLiteUpdateEpisodeContent_NotFound(t *testing.T) {
	s := openTestDB(t)
	ctx := context.Background()
	err := s.UpdateEpisodeContent(ctx, "ghost", Episode{Embedding: []float32{1}})
	if err == nil {
		t.Fatal("expected error for missing episode")
	}
	if !strings.Contains(err.Error(), "episode \"ghost\" not found") {
		t.Errorf("err = %q, want it to mention missing episode", err)
	}
}

func TestSQLiteReplaceEpisodeLinks(t *testing.T) {
	s := openTestDB(t)
	ctx := context.Background()

	// Insert two facts and one episode linked to both.
	f1 := testdataFacts()[0]
	f1.Subtype = "project"
	f2 := testdataFacts()[1]
	f2.Subtype = "project"
	f2.Content = "another fact"

	id1, err := s.InsertFact(ctx, f1)
	if err != nil {
		t.Fatalf("InsertFact f1: %v", err)
	}
	id2, err := s.InsertFact(ctx, f2)
	if err != nil {
		t.Fatalf("InsertFact f2: %v", err)
	}

	ep := testdataEpisodes()[0]
	ep.LinkedFactIDs = []string{id1}
	epID, err := s.InsertEpisode(ctx, ep)
	if err != nil {
		t.Fatalf("InsertEpisode: %v", err)
	}

	// Replace: link to id2 instead of id1.
	if err := s.ReplaceEpisodeLinks(ctx, epID, []string{id2}); err != nil {
		t.Fatalf("ReplaceEpisodeLinks: %v", err)
	}

	got, err := s.GetEpisode(ctx, epID)
	if err != nil {
		t.Fatalf("GetEpisode: %v", err)
	}
	if len(got.LinkedFactIDs) != 1 || got.LinkedFactIDs[0] != id2 {
		t.Errorf("links = %v, want [%s]", got.LinkedFactIDs, id2)
	}

	// Clearing via empty slice.
	if err := s.ReplaceEpisodeLinks(ctx, epID, []string{}); err != nil {
		t.Fatalf("ReplaceEpisodeLinks clear: %v", err)
	}
	got, err = s.GetEpisode(ctx, epID)
	if err != nil {
		t.Fatalf("GetEpisode after clear: %v", err)
	}
	if len(got.LinkedFactIDs) != 0 {
		t.Errorf("links after clear = %v, want empty", got.LinkedFactIDs)
	}

	// nil slice also clears (documented in interface).
	if err := s.ReplaceEpisodeLinks(ctx, epID, []string{id1, id2}); err != nil {
		t.Fatalf("ReplaceEpisodeLinks repopulate: %v", err)
	}
	if err := s.ReplaceEpisodeLinks(ctx, epID, nil); err != nil {
		t.Fatalf("ReplaceEpisodeLinks nil: %v", err)
	}
	got, err = s.GetEpisode(ctx, epID)
	if err != nil {
		t.Fatalf("GetEpisode after nil: %v", err)
	}
	if len(got.LinkedFactIDs) != 0 {
		t.Errorf("links after nil replace = %v, want empty", got.LinkedFactIDs)
	}
}

// --- Phase 4B: ClearFactSuperseded ---

func TestSQLiteClearFactSuperseded_HappyPath(t *testing.T) {
	s := openTestDB(t)
	ctx := context.Background()

	facts := testdataFacts()
	id1, err := s.InsertFact(ctx, facts[0])
	if err != nil {
		t.Fatalf("InsertFact[0]: %v", err)
	}
	id2, err := s.InsertFact(ctx, facts[1])
	if err != nil {
		t.Fatalf("InsertFact[1]: %v", err)
	}
	if err := s.SupersedeFact(ctx, id1, id2); err != nil {
		t.Fatalf("SupersedeFact: %v", err)
	}

	prev, err := s.ClearFactSuperseded(ctx, id1)
	if err != nil {
		t.Fatalf("ClearFactSuperseded: %v", err)
	}
	if prev != id2 {
		t.Errorf("previouslySupersededBy = %q, want %q", prev, id2)
	}

	f, err := s.GetFact(ctx, id1)
	if err != nil {
		t.Fatalf("GetFact: %v", err)
	}
	if f == nil {
		t.Fatal("GetFact returned nil after unsupersede")
	}
	if f.SupersededBy != nil {
		t.Errorf("SupersededBy = %v, want nil", f.SupersededBy)
	}

	listed, err := s.ListFacts(ctx, ListFilter{Limit: 100})
	if err != nil {
		t.Fatalf("ListFacts: %v", err)
	}
	var found bool
	for _, r := range listed {
		if r.ID == id1 {
			found = true
			break
		}
	}
	if !found {
		t.Error("ListFacts did not include revived fact")
	}
}

func TestSQLiteClearFactSuperseded_NotSuperseded(t *testing.T) {
	s := openTestDB(t)
	ctx := context.Background()

	id, err := s.InsertFact(ctx, testdataFacts()[0])
	if err != nil {
		t.Fatalf("InsertFact: %v", err)
	}

	_, err = s.ClearFactSuperseded(ctx, id)
	if err == nil {
		t.Fatal("expected error on active fact")
	}
	if !strings.Contains(err.Error(), "not superseded") {
		t.Errorf("error = %q, want to contain %q", err.Error(), "not superseded")
	}
}

func TestSQLiteClearFactSuperseded_NonexistentID(t *testing.T) {
	s := openTestDB(t)
	ctx := context.Background()

	_, err := s.ClearFactSuperseded(ctx, "does-not-exist")
	if err == nil {
		t.Fatal("expected error on nonexistent id")
	}
	if !strings.Contains(err.Error(), "not found") {
		t.Errorf("error = %q, want to contain %q", err.Error(), "not found")
	}
}

// --- Phase 5C: daily_stats triggers + ListDailyStats ---

// todayUTC returns the YYYY-MM-DD string for the current UTC date, matching
// what the SQLite date('now') call inside the triggers produces.
func todayUTC() string {
	return time.Now().UTC().Format("2006-01-02")
}

// readDailyStatsRow returns the counters for the given date. Missing rows
// surface as zeroes — matching the zero-fill semantics the resource handler
// uses downstream.
func readDailyStatsRow(t *testing.T, s *sqliteStore, date string) (factsIn, factsOut, episodesIn, episodesOut, supersedes int) {
	t.Helper()
	err := s.db.QueryRow(
		`SELECT facts_in, facts_out, episodes_in, episodes_out, supersedes
		 FROM daily_stats WHERE date = ?`, date,
	).Scan(&factsIn, &factsOut, &episodesIn, &episodesOut, &supersedes)
	if err != nil {
		return 0, 0, 0, 0, 0
	}
	return
}

func TestSQLiteDailyStats_FactInsertIncrementsFactsIn(t *testing.T) {
	s := openTestDB(t)
	ctx := context.Background()

	if _, err := s.InsertFact(ctx, testdataFacts()[0]); err != nil {
		t.Fatalf("InsertFact: %v", err)
	}
	fin, _, _, _, _ := readDailyStatsRow(t, s, todayUTC())
	if fin != 1 {
		t.Errorf("facts_in = %d, want 1", fin)
	}
}

func TestSQLiteDailyStats_FactDeleteIncrementsFactsOut(t *testing.T) {
	s := openTestDB(t)
	ctx := context.Background()

	id, err := s.InsertFact(ctx, testdataFacts()[0])
	if err != nil {
		t.Fatalf("InsertFact: %v", err)
	}
	if err := s.DeleteFact(ctx, id); err != nil {
		t.Fatalf("DeleteFact: %v", err)
	}
	_, fout, _, _, _ := readDailyStatsRow(t, s, todayUTC())
	if fout != 1 {
		t.Errorf("facts_out = %d, want 1", fout)
	}
}

func TestSQLiteDailyStats_SupersedeIncrementsSupersedes(t *testing.T) {
	s := openTestDB(t)
	ctx := context.Background()

	facts := testdataFacts()
	// Insert two facts; supersede the first by the second directly via raw
	// UPDATE so the trigger sees the exact NULL→non-NULL transition it's
	// guarded on. (Going through a similarity-driven path would make the
	// test depend on assigner behavior.)
	id1, err := s.InsertFact(ctx, facts[0])
	if err != nil {
		t.Fatalf("InsertFact 0: %v", err)
	}
	id2, err := s.InsertFact(ctx, facts[1])
	if err != nil {
		t.Fatalf("InsertFact 1: %v", err)
	}
	if _, err := s.db.ExecContext(ctx,
		`UPDATE facts SET superseded_by = ? WHERE id = ?`, id2, id1,
	); err != nil {
		t.Fatalf("UPDATE supersede: %v", err)
	}
	_, _, _, _, sup := readDailyStatsRow(t, s, todayUTC())
	if sup != 1 {
		t.Errorf("supersedes = %d, want 1", sup)
	}
}

func TestSQLiteDailyStats_EpisodeInsertIncrementsEpisodesIn(t *testing.T) {
	s := openTestDB(t)
	ctx := context.Background()

	if _, err := s.InsertEpisode(ctx, testdataEpisodes()[0]); err != nil {
		t.Fatalf("InsertEpisode: %v", err)
	}
	_, _, ein, _, _ := readDailyStatsRow(t, s, todayUTC())
	if ein != 1 {
		t.Errorf("episodes_in = %d, want 1", ein)
	}
}

func TestSQLiteDailyStats_EpisodeDeleteIncrementsEpisodesOut(t *testing.T) {
	s := openTestDB(t)
	ctx := context.Background()

	id, err := s.InsertEpisode(ctx, testdataEpisodes()[0])
	if err != nil {
		t.Fatalf("InsertEpisode: %v", err)
	}
	if err := s.DeleteEpisode(ctx, id); err != nil {
		t.Fatalf("DeleteEpisode: %v", err)
	}
	_, _, _, eout, _ := readDailyStatsRow(t, s, todayUTC())
	if eout != 1 {
		t.Errorf("episodes_out = %d, want 1", eout)
	}
}

func TestSQLiteListDailyStats_RangeQuery(t *testing.T) {
	s := openTestDB(t)
	ctx := context.Background()

	// Seed three distinct dates by inserting one fact per day and then
	// rewriting the daily_stats row via raw SQL (so we control exact dates).
	// Going through the trigger and then mutating the row afterwards keeps
	// the schema consistent.
	dates := []string{"2026-03-01", "2026-03-05", "2026-03-10"}
	for i, d := range dates {
		if _, err := s.db.ExecContext(ctx,
			`INSERT INTO daily_stats (date, facts_in) VALUES (?, ?)`, d, i+1,
		); err != nil {
			t.Fatalf("seed row %s: %v", d, err)
		}
	}

	got, err := s.ListDailyStats(ctx, "2026-03-01", "2026-03-07")
	if err != nil {
		t.Fatalf("ListDailyStats: %v", err)
	}
	if len(got) != 2 {
		t.Fatalf("len(got) = %d, want 2 (inclusive 03-01..03-07)", len(got))
	}
	// Ascending order.
	if got[0].Date != "2026-03-01" || got[1].Date != "2026-03-05" {
		t.Errorf("dates = [%s %s], want [2026-03-01 2026-03-05]", got[0].Date, got[1].Date)
	}
	if got[0].FactsIn != 1 || got[1].FactsIn != 2 {
		t.Errorf("facts_in = [%d %d], want [1 2]", got[0].FactsIn, got[1].FactsIn)
	}

	// Full range captures everything.
	all, err := s.ListDailyStats(ctx, "2026-03-01", "2026-03-31")
	if err != nil {
		t.Fatalf("ListDailyStats full: %v", err)
	}
	if len(all) != 3 {
		t.Errorf("full range len = %d, want 3", len(all))
	}

	// Empty range returns empty slice (not nil).
	none, err := s.ListDailyStats(ctx, "2025-01-01", "2025-01-05")
	if err != nil {
		t.Fatalf("ListDailyStats empty: %v", err)
	}
	if none == nil {
		t.Error("empty range returned nil; want empty slice")
	}
	if len(none) != 0 {
		t.Errorf("empty range len = %d, want 0", len(none))
	}
}

// --- LastTick (Phase 5A) ---

func TestSQLiteGetLastTick_FreshStoreReturnsZero(t *testing.T) {
	s := openTestDB(t)
	ctx := context.Background()

	got, err := s.GetLastTick(ctx)
	if err != nil {
		t.Fatalf("GetLastTick: %v", err)
	}
	if !got.IsZero() {
		t.Errorf("fresh GetLastTick = %v, want zero value", got)
	}
}

func TestSQLiteSetLastTick_RoundTrip(t *testing.T) {
	s := openTestDB(t)
	ctx := context.Background()

	now := time.Now().UTC().Truncate(time.Second)
	if err := s.SetLastTick(ctx, now); err != nil {
		t.Fatalf("SetLastTick: %v", err)
	}
	got, err := s.GetLastTick(ctx)
	if err != nil {
		t.Fatalf("GetLastTick after SetLastTick: %v", err)
	}
	// RFC3339 format loses sub-second precision; compare against the
	// truncated value we wrote.
	if !got.Equal(now) {
		t.Errorf("GetLastTick = %v, want %v", got, now)
	}
}

// --- SupersedeLongestChain (Phase 5A) ---

func TestSQLiteSupersedeLongestChain_EmptyDB(t *testing.T) {
	s := openTestDB(t)
	got, err := s.SupersedeLongestChain(context.Background())
	if err != nil {
		t.Fatalf("SupersedeLongestChain: %v", err)
	}
	if got != 0 {
		t.Errorf("SupersedeLongestChain on empty DB = %d, want 0", got)
	}
}

func TestSQLiteSupersedeLongestChain_ThreeFactChain(t *testing.T) {
	s := openTestDB(t)
	ctx := context.Background()

	// Insert three distinct facts bypassing any similarity-based supersede so
	// we control the chain shape precisely. Use the raw UPDATE to create the
	// edges A->B->C (A.superseded_by=B, B.superseded_by=C).
	facts := testdataFacts()
	idA, err := s.InsertFact(ctx, facts[0])
	if err != nil {
		t.Fatalf("insert A: %v", err)
	}
	idB, err := s.InsertFact(ctx, facts[1])
	if err != nil {
		t.Fatalf("insert B: %v", err)
	}
	idC, err := s.InsertFact(ctx, facts[2])
	if err != nil {
		t.Fatalf("insert C: %v", err)
	}
	if _, err := s.db.ExecContext(ctx,
		`UPDATE facts SET superseded_by = ? WHERE id = ?`, idB, idA,
	); err != nil {
		t.Fatalf("set A.superseded_by=B: %v", err)
	}
	if _, err := s.db.ExecContext(ctx,
		`UPDATE facts SET superseded_by = ? WHERE id = ?`, idC, idB,
	); err != nil {
		t.Fatalf("set B.superseded_by=C: %v", err)
	}

	got, err := s.SupersedeLongestChain(ctx)
	if err != nil {
		t.Fatalf("SupersedeLongestChain: %v", err)
	}
	if got != 3 {
		t.Errorf("SupersedeLongestChain for A->B->C = %d, want 3", got)
	}
}

// --- CountSupersededFacts (Phase 5A) ---

func TestSQLiteCountSupersededFacts(t *testing.T) {
	s := openTestDB(t)
	ctx := context.Background()

	// Start with zero.
	got, err := s.CountSupersededFacts(ctx)
	if err != nil {
		t.Fatalf("CountSupersededFacts empty: %v", err)
	}
	if got != 0 {
		t.Errorf("empty DB = %d, want 0", got)
	}

	facts := testdataFacts()
	id1, _ := s.InsertFact(ctx, facts[0])
	id2, _ := s.InsertFact(ctx, facts[1])
	if _, err := s.InsertFact(ctx, facts[2]); err != nil {
		t.Fatalf("insert: %v", err)
	}
	// Supersede f1 by f2 via direct UPDATE.
	if _, err := s.db.ExecContext(ctx,
		`UPDATE facts SET superseded_by = ? WHERE id = ?`, id2, id1,
	); err != nil {
		t.Fatalf("update: %v", err)
	}
	got, err = s.CountSupersededFacts(ctx)
	if err != nil {
		t.Fatalf("CountSupersededFacts: %v", err)
	}
	if got != 1 {
		t.Errorf("after one supersede = %d, want 1", got)
	}
}
