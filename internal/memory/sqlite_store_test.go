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

	if err := s.LinkFactEpisode(ctx, factID, epID, "evidence"); err != nil {
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

	if err := s.LinkFactEpisode(ctx, factID, epID, "evidence"); err != nil {
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
	if err := s.LinkFactEpisode(ctx, factID, epID, "evidence"); err != nil {
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
	if err := s.LinkFactEpisode(ctx, factID, epID, "evidence"); err != nil {
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
