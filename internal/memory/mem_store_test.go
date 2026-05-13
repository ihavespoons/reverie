package memory

import (
	"context"
	"strings"
	"testing"
	"time"
)

func TestMemInsertGetRoundTrip(t *testing.T) {
	s := NewMemStore()
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

func TestMemInsertIdempotency(t *testing.T) {
	s := NewMemStore()
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

func TestMemListFactsSubtypeFilter(t *testing.T) {
	s := NewMemStore()
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

func TestMemDeleteFact(t *testing.T) {
	s := NewMemStore()
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

func TestMemGlobalSearchOrdering(t *testing.T) {
	s := NewMemStore()
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

func TestMemTouchAccessed(t *testing.T) {
	s := NewMemStore()
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

func TestMemSupersededFiltering(t *testing.T) {
	s := NewMemStore().(*memStore)
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

	// Directly mark the first fact as superseded.
	s.mu.Lock()
	f := s.facts[id1]
	f.SupersededBy = &id2
	s.facts[id1] = f
	s.mu.Unlock()

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

func TestMemGetFactNotFound(t *testing.T) {
	s := NewMemStore()
	ctx := context.Background()

	got, err := s.GetFact(ctx, "nonexistent-id")
	if err != nil {
		t.Fatalf("GetFact(nonexistent) = %v, want nil error", err)
	}
	if got != nil {
		t.Error("GetFact(nonexistent) returned non-nil")
	}
}

func TestMemListFactsDefaultSort(t *testing.T) {
	s := NewMemStore()
	ctx := context.Background()

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

	// Most recent first (created_at DESC).
	for i := 1; i < len(results); i++ {
		if results[i].CreatedAt.After(results[i-1].CreatedAt) {
			t.Errorf("results not sorted by created_at DESC: [%d]=%v > [%d]=%v",
				i, results[i].CreatedAt, i-1, results[i-1].CreatedAt)
		}
	}
}

// --- Cluster method tests (Phase 2) ---

// insertTestClusterMem inserts a cluster directly into the memStore for test setup.
func insertTestClusterMem(t *testing.T, s *memStore, id string) {
	t.Helper()
	s.mu.Lock()
	defer s.mu.Unlock()
	now := time.Now().UTC()
	s.clusters[id] = ClusterNode{
		ID:         id,
		Summary:    id,
		LastAccess: now,
		CreatedAt:  now,
	}
}

func TestMemGetCluster_DefaultExists(t *testing.T) {
	s := NewMemStore()
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

func TestMemGetCluster_NotFound(t *testing.T) {
	s := NewMemStore()
	ctx := context.Background()

	c, err := s.GetCluster(ctx, "nonexistent")
	if err != nil {
		t.Fatalf("GetCluster(\"nonexistent\") error = %v, want nil", err)
	}
	if c != nil {
		t.Errorf("GetCluster(\"nonexistent\") = %+v, want nil", c)
	}
}

func TestMemListClusters_Empty(t *testing.T) {
	s := NewMemStore()
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

func TestMemUpdateClusterState_Success(t *testing.T) {
	s := NewMemStore()
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

func TestMemUpdateClusterState_NotFound(t *testing.T) {
	s := NewMemStore()
	ctx := context.Background()

	err := s.UpdateClusterState(ctx, "nonexistent", 0.5, 0.5, 0)
	if err == nil {
		t.Fatal("UpdateClusterState on nonexistent cluster should return error")
	}
}

func TestMemTickAllClusters_Basic(t *testing.T) {
	ms := NewMemStore().(*memStore)
	ctx := context.Background()

	// Create two clusters directly.
	insertTestClusterMem(t, ms, "A")
	insertTestClusterMem(t, ms, "B")

	// Set initial turns_since for A=2, B=3.
	if err := ms.UpdateClusterState(ctx, "A", 0.5, 0.5, 2); err != nil {
		t.Fatalf("UpdateClusterState(A): %v", err)
	}
	if err := ms.UpdateClusterState(ctx, "B", 0.5, 0.5, 3); err != nil {
		t.Fatalf("UpdateClusterState(B): %v", err)
	}

	// Tick with A accessed.
	if err := ms.TickAllClusters(ctx, []string{"A"}); err != nil {
		t.Fatalf("TickAllClusters: %v", err)
	}

	a, err := ms.GetCluster(ctx, "A")
	if err != nil {
		t.Fatalf("GetCluster(A): %v", err)
	}
	b, err := ms.GetCluster(ctx, "B")
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

func TestMemTickAllClusters_Empty(t *testing.T) {
	ms := NewMemStore().(*memStore)
	ctx := context.Background()

	// Create two clusters.
	insertTestClusterMem(t, ms, "X")
	insertTestClusterMem(t, ms, "Y")

	// Set initial state.
	if err := ms.UpdateClusterState(ctx, "X", 0.5, 0.5, 1); err != nil {
		t.Fatalf("UpdateClusterState(X): %v", err)
	}
	if err := ms.UpdateClusterState(ctx, "Y", 0.5, 0.5, 2); err != nil {
		t.Fatalf("UpdateClusterState(Y): %v", err)
	}

	// TickAllClusters with no accessed IDs: all clusters get incremented.
	if err := ms.TickAllClusters(ctx, []string{}); err != nil {
		t.Fatalf("TickAllClusters: %v", err)
	}

	x, err := ms.GetCluster(ctx, "X")
	if err != nil {
		t.Fatalf("GetCluster(X): %v", err)
	}
	y, err := ms.GetCluster(ctx, "Y")
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

func TestMemInsertEpisode_RoundTrip(t *testing.T) {
	s := NewMemStore()
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

func TestMemListEpisodes_OrderAndFilter(t *testing.T) {
	s := NewMemStore()
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

func TestMemDeleteEpisode(t *testing.T) {
	s := NewMemStore()
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

	// Add an evidence edge so we can confirm cascade.
	if _, err := s.AddEdge(ctx, Edge{SrcID: factID, DstID: epID, EdgeType: "evidence", Weight: 1.0}); err != nil {
		t.Fatalf("AddEdge: %v", err)
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

	// Evidence edges should be cascade-deleted.
	edges, err := s.ListEdges(ctx, factID, 1)
	if err != nil {
		t.Fatalf("ListEdges after episode delete: %v", err)
	}
	if len(edges) != 0 {
		t.Errorf("expected 0 edges after episode delete, got %d", len(edges))
	}

	// Deleting a non-existent id should not error.
	if err := s.DeleteEpisode(ctx, "nonexistent"); err != nil {
		t.Errorf("DeleteEpisode(nonexistent) = %v, want nil", err)
	}
}

// --- Phase 3: GlobalSearch polymorphism ---

func TestMemGlobalSearch_MixesFactsAndEpisodes(t *testing.T) {
	s := NewMemStore()
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

func TestMemCreateCluster_Basic(t *testing.T) {
	s := NewMemStore()
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

	// Duplicate create should error.
	err = s.CreateCluster(ctx, c)
	if err == nil {
		t.Error("expected error for duplicate cluster create")
	}
}

func TestMemUpdateClusterCentroid(t *testing.T) {
	s := NewMemStore()
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

	// Update nonexistent should error.
	err = s.UpdateClusterCentroid(ctx, "nonexistent", newCentroid, 1)
	if err == nil {
		t.Error("expected error for nonexistent cluster")
	}
}

// --- Phase 3: Temporal conflict resolution tests ---

func TestMemFindSimilarFacts_Match(t *testing.T) {
	s := NewMemStore()
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

	// Query with a similar vector.
	queryVec := []float32{0.9, 0.1, 0, 0}
	results, err := s.FindSimilarFacts(ctx, "project", queryVec, 0.95, 10)
	if err != nil {
		t.Fatalf("FindSimilarFacts: %v", err)
	}

	if len(results) == 0 {
		t.Fatal("FindSimilarFacts returned 0 results, expected at least 1")
	}

	if results[0].Similarity < 0.95 {
		t.Errorf("top result similarity = %v, want >= 0.95", results[0].Similarity)
	}
}

func TestMemFindSimilarFacts_NoCrossSubtype(t *testing.T) {
	s := NewMemStore()
	ctx := context.Background()

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
		Embedding: []float32{0.9, 0.1, 0, 0},
	}

	if _, err := s.InsertFact(ctx, f1); err != nil {
		t.Fatalf("InsertFact[0]: %v", err)
	}
	if _, err := s.InsertFact(ctx, f2); err != nil {
		t.Fatalf("InsertFact[1]: %v", err)
	}

	queryVec := []float32{0.9, 0.1, 0, 0}
	results, err := s.FindSimilarFacts(ctx, "user", queryVec, 0.90, 10)
	if err != nil {
		t.Fatalf("FindSimilarFacts: %v", err)
	}

	for _, c := range results {
		if c.Fact.Subtype != "user" {
			t.Errorf("FindSimilarFacts returned subtype=%q, expected only 'user'", c.Fact.Subtype)
		}
	}
	if len(results) != 1 {
		t.Errorf("expected 1 result, got %d", len(results))
	}
}

func TestMemFindSimilarFacts_ExcludesSuperseded(t *testing.T) {
	s := NewMemStore()
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

func TestMemSupersedeFact(t *testing.T) {
	s := NewMemStore()
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

func TestMemUpdateFactEmbedding(t *testing.T) {
	s := NewMemStore()
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

func TestMemUpdateFactEmbedding_NotFound(t *testing.T) {
	s := NewMemStore()
	ctx := context.Background()

	err := s.UpdateFactEmbedding(ctx, "nonexistent", []float32{1, 0, 0, 0})
	if err == nil {
		t.Fatal("UpdateFactEmbedding on nonexistent fact should return error")
	}
}

func TestMemUpdateEpisodeEmbedding(t *testing.T) {
	s := NewMemStore()
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

func TestMemUpdateEpisodeEmbedding_NotFound(t *testing.T) {
	s := NewMemStore()
	ctx := context.Background()

	err := s.UpdateEpisodeEmbedding(ctx, "nonexistent", []float32{1, 0, 0, 0})
	if err == nil {
		t.Fatal("UpdateEpisodeEmbedding on nonexistent episode should return error")
	}
}

// --- Phase 4: UpdateClusterMeta tests ---

func TestMemUpdateClusterMeta_Basic(t *testing.T) {
	s := NewMemStore()
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

func TestMemUpdateClusterMeta_NotFound(t *testing.T) {
	s := NewMemStore()
	ctx := context.Background()

	err := s.UpdateClusterMeta(ctx, "nonexistent", "summary", "domain", "meta")
	if err == nil {
		t.Fatal("UpdateClusterMeta on nonexistent cluster should return error")
	}
}

// --- GetFactSupersedes tests (Phase 1C) ---

func TestMemGetFactSupersedes_Empty(t *testing.T) {
	s := NewMemStore()
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

func TestMemGetFactSupersedes_WithPredecessors(t *testing.T) {
	s := NewMemStore()
	ctx := context.Background()

	// Insert three distinct facts so none are deduped by content-hash.
	f1 := Fact{
		Content:   "alpha",
		Subtype:   "project",
		Source:    "inferred",
		Embedding: []float32{1, 0, 0, 0},
	}
	f2 := Fact{
		Content:   "beta",
		Subtype:   "project",
		Source:    "inferred",
		Embedding: []float32{0, 1, 0, 0},
	}
	f3 := Fact{
		Content:   "gamma (new winner)",
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

	// id1 and id2 both superseded by id3.
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

	// id1 by itself supersedes nothing.
	got1, err := s.GetFactSupersedes(ctx, id1)
	if err != nil {
		t.Fatalf("GetFactSupersedes(id1): %v", err)
	}
	if len(got1) != 0 {
		t.Errorf("id1 predecessors = %v, want empty", got1)
	}
}

func TestMemGetFactSupersedes_UnknownID(t *testing.T) {
	s := NewMemStore()
	ctx := context.Background()

	got, err := s.GetFactSupersedes(ctx, "nonexistent")
	if err != nil {
		t.Fatalf("GetFactSupersedes: %v", err)
	}
	if len(got) != 0 {
		t.Errorf("unknown id returned %v, want empty slice", got)
	}
}

// --- Cluster membership (Phase 1E) ---

func TestMemClusterMembership_PaginatedAndOrdered(t *testing.T) {
	s := NewMemStore()
	ctx := context.Background()

	clusterID := "cluster-1e-mem"
	if err := s.CreateCluster(ctx, ClusterNode{
		ID: clusterID, Summary: "t", CreatedAt: time.Now().UTC(), LastAccess: time.Now().UTC(),
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

	fc, err := s.CountFactsByCluster(ctx, clusterID)
	if err != nil || fc != 5 {
		t.Fatalf("CountFactsByCluster = %d (err %v), want 5", fc, err)
	}
	ec, err := s.CountEpisodesByCluster(ctx, clusterID)
	if err != nil || ec != 3 {
		t.Fatalf("CountEpisodesByCluster = %d (err %v), want 3", ec, err)
	}

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
	all := append(append([]Fact{}, p1...), p2...)
	for i := 1; i < len(all); i++ {
		if all[i].CreatedAt.Before(all[i-1].CreatedAt) {
			t.Errorf("facts not ordered ASC: %v", all)
		}
	}
}

func TestMemClusterMembership_ExcludesSuperseded(t *testing.T) {
	s := NewMemStore()
	ctx := context.Background()

	clusterID := "cluster-super-mem"
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
		t.Errorf("got %d facts, want [%s]", len(got), newID)
	}
	fc, _ := s.CountFactsByCluster(ctx, clusterID)
	if fc != 1 {
		t.Errorf("CountFactsByCluster = %d, want 1", fc)
	}
}

func TestMemClusterMembership_UnknownCluster(t *testing.T) {
	s := NewMemStore()
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

// --- Phase 2A: SetMemoryCluster + DeleteCluster ---

func TestMemSetMemoryCluster_Fact(t *testing.T) {
	s := NewMemStore()
	ctx := context.Background()

	// Pre-create target cluster.
	if err := s.CreateCluster(ctx, ClusterNode{ID: "target", Summary: "target"}); err != nil {
		t.Fatalf("CreateCluster: %v", err)
	}

	f := testdataFacts()[0]
	f.ClusterID = "source"
	id, err := s.InsertFact(ctx, f)
	if err != nil {
		t.Fatalf("InsertFact: %v", err)
	}

	// Grab the accessed_at before the call so we can verify it advances.
	before, err := s.GetFact(ctx, id)
	if err != nil {
		t.Fatalf("GetFact(before): %v", err)
	}
	// Ensure a measurable gap.
	time.Sleep(2 * time.Millisecond)

	if err := s.SetMemoryCluster(ctx, id, "target"); err != nil {
		t.Fatalf("SetMemoryCluster: %v", err)
	}

	after, err := s.GetFact(ctx, id)
	if err != nil {
		t.Fatalf("GetFact(after): %v", err)
	}
	if after.ClusterID != "target" {
		t.Errorf("ClusterID = %q, want %q", after.ClusterID, "target")
	}
	if !after.AccessedAt.After(before.AccessedAt) {
		t.Errorf("AccessedAt not bumped: before=%v after=%v", before.AccessedAt, after.AccessedAt)
	}
}

func TestMemSetMemoryCluster_Episode(t *testing.T) {
	s := NewMemStore()
	ctx := context.Background()

	if err := s.CreateCluster(ctx, ClusterNode{ID: "target", Summary: "target"}); err != nil {
		t.Fatalf("CreateCluster: %v", err)
	}

	ep := testdataEpisodes()[0]
	ep.ClusterID = "source"
	id, err := s.InsertEpisode(ctx, ep)
	if err != nil {
		t.Fatalf("InsertEpisode: %v", err)
	}

	before, err := s.GetEpisode(ctx, id)
	if err != nil {
		t.Fatalf("GetEpisode(before): %v", err)
	}
	time.Sleep(2 * time.Millisecond)

	if err := s.SetMemoryCluster(ctx, id, "target"); err != nil {
		t.Fatalf("SetMemoryCluster: %v", err)
	}

	after, err := s.GetEpisode(ctx, id)
	if err != nil {
		t.Fatalf("GetEpisode(after): %v", err)
	}
	if after.ClusterID != "target" {
		t.Errorf("ClusterID = %q, want %q", after.ClusterID, "target")
	}
	if !after.AccessedAt.After(before.AccessedAt) {
		t.Errorf("AccessedAt not bumped: before=%v after=%v", before.AccessedAt, after.AccessedAt)
	}
}

func TestMemSetMemoryCluster_NotFound(t *testing.T) {
	s := NewMemStore()
	ctx := context.Background()

	err := s.SetMemoryCluster(ctx, "ghost", "anywhere")
	if err == nil {
		t.Fatal("expected error for missing memory")
	}
	if !strings.Contains(err.Error(), "memory not found: ghost") {
		t.Errorf("error = %q, want it to contain 'memory not found: ghost'", err)
	}
}

func TestMemDeleteCluster_Empty(t *testing.T) {
	s := NewMemStore()
	ctx := context.Background()

	if err := s.CreateCluster(ctx, ClusterNode{ID: "orphan", Summary: "to delete"}); err != nil {
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

func TestMemDeleteCluster_Idempotent(t *testing.T) {
	s := NewMemStore()
	ctx := context.Background()
	if err := s.DeleteCluster(ctx, "never-existed"); err != nil {
		t.Errorf("DeleteCluster missing = %v, want nil", err)
	}
}

func TestMemDeleteCluster_RefusesNonEmpty(t *testing.T) {
	s := NewMemStore()
	ctx := context.Background()

	// Insert a fact — InsertFact auto-creates the cluster row.
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

func TestMem_RecomputeCentroid_HappyPath(t *testing.T) {
	s := NewMemStore()
	ctx := context.Background()

	// Pre-create the cluster with an arbitrary starting centroid and item count.
	if err := s.CreateCluster(ctx, ClusterNode{
		ID:        "c1",
		Summary:   "c1",
		Centroid:  []float32{9, 9, 9, 9},
		ItemCount: 42,
	}); err != nil {
		t.Fatalf("CreateCluster: %v", err)
	}

	// Two facts and one episode with known embeddings.
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
	if got == nil {
		t.Fatal("cluster missing")
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

func TestMem_RecomputeCentroid_EmptyCluster(t *testing.T) {
	s := NewMemStore()
	ctx := context.Background()

	if err := s.CreateCluster(ctx, ClusterNode{ID: "empty", Summary: "empty"}); err != nil {
		t.Fatalf("CreateCluster: %v", err)
	}

	err := RecomputeCentroid(ctx, s, "empty")
	if err != ErrEmptyCluster {
		t.Fatalf("RecomputeCentroid empty = %v, want ErrEmptyCluster", err)
	}
}

func TestMem_RecomputeCentroid_SkipsMembersWithoutEmbedding(t *testing.T) {
	s := NewMemStore()
	ctx := context.Background()

	if err := s.CreateCluster(ctx, ClusterNode{ID: "mixed", Summary: "mixed"}); err != nil {
		t.Fatalf("CreateCluster: %v", err)
	}

	// One with embedding, one without.
	f1 := Fact{Content: "with", Subtype: "project", ClusterID: "mixed", Embedding: []float32{1, 0}}
	f2 := Fact{Content: "without", Subtype: "project", ClusterID: "mixed"} // nil embedding
	if _, err := s.InsertFact(ctx, f1); err != nil {
		t.Fatalf("InsertFact f1: %v", err)
	}
	if _, err := s.InsertFact(ctx, f2); err != nil {
		t.Fatalf("InsertFact f2: %v", err)
	}

	if err := RecomputeCentroid(ctx, s, "mixed"); err != nil {
		t.Fatalf("RecomputeCentroid: %v", err)
	}
	got, err := s.GetCluster(ctx, "mixed")
	if err != nil {
		t.Fatalf("GetCluster: %v", err)
	}
	// The centroid equals the lone embedded vector.
	if len(got.Centroid) != 2 || got.Centroid[0] != 1 || got.Centroid[1] != 0 {
		t.Errorf("centroid = %v, want [1 0]", got.Centroid)
	}
	// ItemCount is total membership (not just embedded).
	if got.ItemCount != 2 {
		t.Errorf("ItemCount = %d, want 2", got.ItemCount)
	}
}

func TestMem_RecomputeCentroid_AllWithoutEmbedding(t *testing.T) {
	s := NewMemStore()
	ctx := context.Background()

	if err := s.CreateCluster(ctx, ClusterNode{ID: "no-vecs", Summary: "no-vecs"}); err != nil {
		t.Fatalf("CreateCluster: %v", err)
	}
	f := Fact{Content: "x", Subtype: "project", ClusterID: "no-vecs"} // nil embedding
	if _, err := s.InsertFact(ctx, f); err != nil {
		t.Fatalf("InsertFact: %v", err)
	}

	err := RecomputeCentroid(ctx, s, "no-vecs")
	if err != ErrEmptyCluster {
		t.Fatalf("RecomputeCentroid all-nil = %v, want ErrEmptyCluster", err)
	}
}

// --- Phase 2B: MoveAllClusterMembers ---

func TestMemMoveAllClusterMembers_FactsAndEpisodes(t *testing.T) {
	s := NewMemStore()
	ctx := context.Background()

	if err := s.CreateCluster(ctx, ClusterNode{ID: "dst", Summary: "dst"}); err != nil {
		t.Fatalf("CreateCluster dst: %v", err)
	}

	// 2 facts + 1 episode in src; 1 fact in dst.
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

	time.Sleep(2 * time.Millisecond)

	moved, err := s.MoveAllClusterMembers(ctx, "src", "dst")
	if err != nil {
		t.Fatalf("MoveAllClusterMembers: %v", err)
	}
	if moved != 3 {
		t.Errorf("moved = %d, want 3", moved)
	}

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
	gotEp, _ := s.GetEpisode(ctx, epID)
	if gotEp.ClusterID != "dst" {
		t.Errorf("ep.ClusterID = %q, want dst", gotEp.ClusterID)
	}
	got3, _ := s.GetFact(ctx, f3ID)
	if got3.ClusterID != "dst" {
		t.Errorf("f3.ClusterID = %q, want dst (unchanged)", got3.ClusterID)
	}

	nf, _ := s.CountFactsByCluster(ctx, "src")
	ne, _ := s.CountEpisodesByCluster(ctx, "src")
	if nf+ne != 0 {
		t.Errorf("src residual = %d, want 0", nf+ne)
	}
}

func TestMemMoveAllClusterMembers_EmptySource(t *testing.T) {
	s := NewMemStore()
	ctx := context.Background()

	if err := s.CreateCluster(ctx, ClusterNode{ID: "src", Summary: "src"}); err != nil {
		t.Fatalf("CreateCluster src: %v", err)
	}
	if err := s.CreateCluster(ctx, ClusterNode{ID: "dst", Summary: "dst"}); err != nil {
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

func TestMemMoveAllClusterMembers_IgnoresSuperseded(t *testing.T) {
	s := NewMemStore()
	ctx := context.Background()

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
	if err := s.SupersedeFact(ctx, f1ID, f2ID); err != nil {
		t.Fatalf("SupersedeFact: %v", err)
	}
	if err := s.CreateCluster(ctx, ClusterNode{ID: "dst", Summary: "dst"}); err != nil {
		t.Fatalf("CreateCluster dst: %v", err)
	}

	moved, err := s.MoveAllClusterMembers(ctx, "src", "dst")
	if err != nil {
		t.Fatalf("MoveAllClusterMembers: %v", err)
	}
	if moved != 1 {
		t.Errorf("moved = %d, want 1 (f1 is superseded)", moved)
	}
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

func TestMemUpdateFactContent_HappyPath(t *testing.T) {
	s := NewMemStore()
	ctx := context.Background()

	f := testdataFacts()[0]
	f.Tags = []string{"a", "b"}
	id, err := s.InsertFact(ctx, f)
	if err != nil {
		t.Fatalf("InsertFact: %v", err)
	}
	before, err := s.GetFact(ctx, id)
	if err != nil {
		t.Fatalf("GetFact before: %v", err)
	}
	time.Sleep(2 * time.Millisecond)

	newTags := []string{"c"}
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
	if len(after.Tags) != 1 || after.Tags[0] != "c" {
		t.Errorf("Tags = %v, want [c]", after.Tags)
	}
	if !after.AccessedAt.After(before.AccessedAt) {
		t.Errorf("AccessedAt not bumped: before=%v after=%v", before.AccessedAt, after.AccessedAt)
	}
	if !after.CreatedAt.Equal(before.CreatedAt) {
		t.Errorf("CreatedAt changed: before=%v after=%v", before.CreatedAt, after.CreatedAt)
	}
	if after.ClusterID != before.ClusterID {
		t.Errorf("ClusterID changed: before=%q after=%q", before.ClusterID, after.ClusterID)
	}
}

func TestMemUpdateFactContent_TagsNilPreserves(t *testing.T) {
	s := NewMemStore()
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

func TestMemUpdateFactContent_TagsEmptyClears(t *testing.T) {
	s := NewMemStore()
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

func TestMemUpdateFactContent_NotFound(t *testing.T) {
	s := NewMemStore()
	ctx := context.Background()
	err := s.UpdateFactContent(ctx, "ghost", "x", "y", []float32{1}, nil)
	if err == nil {
		t.Fatal("expected error for missing fact")
	}
	if !strings.Contains(err.Error(), "fact \"ghost\" not found") {
		t.Errorf("err = %q, want it to mention missing fact", err)
	}
}

func TestMemUpdateEpisodeContent_HappyPath(t *testing.T) {
	s := NewMemStore()
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
	time.Sleep(2 * time.Millisecond)

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

func TestMemUpdateEpisodeContent_NotFound(t *testing.T) {
	s := NewMemStore()
	ctx := context.Background()
	err := s.UpdateEpisodeContent(ctx, "ghost", Episode{Embedding: []float32{1}})
	if err == nil {
		t.Fatal("expected error for missing episode")
	}
	if !strings.Contains(err.Error(), "episode \"ghost\" not found") {
		t.Errorf("err = %q, want it to mention missing episode", err)
	}
}

func TestMemReplaceEpisodeLinks(t *testing.T) {
	s := NewMemStore()
	ctx := context.Background()

	f1 := testdataFacts()[0]
	f2 := testdataFacts()[1]
	f2.Content = "different"
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

	if err := s.ReplaceEpisodeLinks(ctx, epID, []string{}); err != nil {
		t.Fatalf("ReplaceEpisodeLinks clear: %v", err)
	}
	got, err = s.GetEpisode(ctx, epID)
	if err != nil {
		t.Fatalf("GetEpisode: %v", err)
	}
	if len(got.LinkedFactIDs) != 0 {
		t.Errorf("links after clear = %v, want empty", got.LinkedFactIDs)
	}

	if err := s.ReplaceEpisodeLinks(ctx, epID, []string{id1, id2}); err != nil {
		t.Fatalf("ReplaceEpisodeLinks repopulate: %v", err)
	}
	if err := s.ReplaceEpisodeLinks(ctx, epID, nil); err != nil {
		t.Fatalf("ReplaceEpisodeLinks nil: %v", err)
	}
	got, err = s.GetEpisode(ctx, epID)
	if err != nil {
		t.Fatalf("GetEpisode: %v", err)
	}
	if len(got.LinkedFactIDs) != 0 {
		t.Errorf("links after nil replace = %v, want empty", got.LinkedFactIDs)
	}
}

// --- Phase 4B: ClearFactSuperseded ---

func TestMemClearFactSuperseded_HappyPath(t *testing.T) {
	s := NewMemStore()
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

	// Verify the flag is cleared.
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

	// And the fact is now visible in ListFacts.
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

func TestMemClearFactSuperseded_NotSuperseded(t *testing.T) {
	s := NewMemStore()
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

func TestMemClearFactSuperseded_NonexistentID(t *testing.T) {
	s := NewMemStore()
	ctx := context.Background()

	_, err := s.ClearFactSuperseded(ctx, "does-not-exist")
	if err == nil {
		t.Fatal("expected error on nonexistent id")
	}
	if !strings.Contains(err.Error(), "not found") {
		t.Errorf("error = %q, want to contain %q", err.Error(), "not found")
	}
}

// --- LastTick (Phase 5A) ---

func TestMemGetLastTick_FreshStoreReturnsZero(t *testing.T) {
	s := NewMemStore()
	got, err := s.GetLastTick(context.Background())
	if err != nil {
		t.Fatalf("GetLastTick: %v", err)
	}
	if !got.IsZero() {
		t.Errorf("fresh GetLastTick = %v, want zero value", got)
	}
}

func TestMemSetLastTick_RoundTrip(t *testing.T) {
	s := NewMemStore()
	ctx := context.Background()

	now := time.Now().UTC()
	if err := s.SetLastTick(ctx, now); err != nil {
		t.Fatalf("SetLastTick: %v", err)
	}
	got, err := s.GetLastTick(ctx)
	if err != nil {
		t.Fatalf("GetLastTick: %v", err)
	}
	if got.Sub(now).Abs() > time.Second {
		t.Errorf("GetLastTick = %v, want within 1s of %v", got, now)
	}
}

// --- SupersedeLongestChain (Phase 5A) ---

func TestMemSupersedeLongestChain_EmptyStore(t *testing.T) {
	s := NewMemStore()
	got, err := s.SupersedeLongestChain(context.Background())
	if err != nil {
		t.Fatalf("SupersedeLongestChain: %v", err)
	}
	if got != 0 {
		t.Errorf("SupersedeLongestChain on empty store = %d, want 0", got)
	}
}

func TestMemSupersedeLongestChain_ThreeFactChain(t *testing.T) {
	s := NewMemStore()
	ctx := context.Background()

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
	// A->B->C
	if err := s.SupersedeFact(ctx, idA, idB); err != nil {
		t.Fatalf("supersede A by B: %v", err)
	}
	if err := s.SupersedeFact(ctx, idB, idC); err != nil {
		t.Fatalf("supersede B by C: %v", err)
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

func TestMemCountSupersededFacts(t *testing.T) {
	s := NewMemStore()
	ctx := context.Background()

	got, err := s.CountSupersededFacts(ctx)
	if err != nil {
		t.Fatalf("CountSupersededFacts empty: %v", err)
	}
	if got != 0 {
		t.Errorf("empty = %d, want 0", got)
	}

	facts := testdataFacts()
	id1, _ := s.InsertFact(ctx, facts[0])
	id2, _ := s.InsertFact(ctx, facts[1])
	if _, err := s.InsertFact(ctx, facts[2]); err != nil {
		t.Fatalf("insert: %v", err)
	}
	if err := s.SupersedeFact(ctx, id1, id2); err != nil {
		t.Fatalf("supersede: %v", err)
	}
	got, err = s.CountSupersededFacts(ctx)
	if err != nil {
		t.Fatalf("CountSupersededFacts: %v", err)
	}
	if got != 1 {
		t.Errorf("after one supersede = %d, want 1", got)
	}
}

// --- Phase 7 knowledge graph: KG method wrappers ---

func memFactory(t *testing.T) Store { t.Helper(); return NewMemStore() }

func TestMemAddEdge_Idempotent(t *testing.T)      { kgRunAddEdgeIdempotent(t, memFactory) }
func TestMemRemoveEdge(t *testing.T)              { kgRunRemoveEdge(t, memFactory) }
func TestMemListEdges_TwoHops(t *testing.T)       { kgRunListEdgesTwoHops(t, memFactory) }
func TestMemUpsertEntity_ExactDedup(t *testing.T) { kgRunUpsertEntityExactDedup(t, memFactory) }
func TestMemUpsertEntity_SimilarityDedup(t *testing.T) {
	kgRunUpsertEntitySimilarityDedup(t, memFactory)
}
func TestMemUpsertEntity_DifferentTypeNotDedup(t *testing.T) {
	kgRunUpsertEntityDifferentType(t, memFactory)
}
func TestMemTickAllEntities_TwoTicksWithoutAccess(t *testing.T) {
	kgRunTickAllEntitiesNoAccess(t, memFactory)
}
func TestMemTickAllEntities_AccessedReset(t *testing.T) {
	kgRunTickAllEntitiesAccessedReset(t, memFactory)
}
func TestMemAddEntityMentions_Idempotent(t *testing.T) {
	kgRunAddEntityMentionsIdempotent(t, memFactory)
}
func TestMemListMemoriesByEntity(t *testing.T)      { kgRunListMemoriesByEntity(t, memFactory) }
func TestMemListEntityNeighbors_Hops1(t *testing.T) { kgRunListEntityNeighborsHops1(t, memFactory) }
func TestMemCascadeOnDeleteFact(t *testing.T)       { kgRunCascadeOnDeleteFact(t, memFactory) }
func TestMemListEntitiesByMemoryIDs(t *testing.T)   { kgRunListEntitiesByMemoryIDs(t, memFactory) }
func TestMemCountEntitiesAndEdges(t *testing.T)     { kgRunCountEntitiesAndEdges(t, memFactory) }

// --- Phase 7C: ExpandViaGraph ---

func TestMemExpandViaGraph_DirectEdge(t *testing.T) { kgRunExpandViaGraph_DirectEdge(t, memFactory) }
func TestMemExpandViaGraph_EntityMention(t *testing.T) {
	kgRunExpandViaGraph_EntityMention(t, memFactory)
}
func TestMemExpandViaGraph_TwoHopEdgeChain(t *testing.T) {
	kgRunExpandViaGraph_TwoHopEdgeChain(t, memFactory)
}
func TestMemExpandViaGraph_HubEntityNoCap(t *testing.T) {
	kgRunExpandViaGraph_HubEntityNoCap(t, memFactory)
}
func TestMemExpandViaGraph_GlobalCapHonored(t *testing.T) {
	kgRunExpandViaGraph_GlobalCapHonored(t, memFactory)
}
func TestMemExpandViaGraph_RetentionPrefilter(t *testing.T) {
	kgRunExpandViaGraph_RetentionPrefilter(t, memFactory)
}
func TestMemExpandViaGraph_MultiSeedReturnsAllPairs(t *testing.T) {
	kgRunExpandViaGraph_MultiSeedReturnsAllPairs(t, memFactory)
}
func TestMemExpandViaGraph_SameSeedShortestPath(t *testing.T) {
	kgRunExpandViaGraph_SameSeedShortestPath(t, memFactory)
}
