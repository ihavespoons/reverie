package memory

import (
	"context"
	"sort"
	"testing"
)

// kgTestFactory constructs a fresh Store. Both backends provide one.
type kgTestFactory func(t *testing.T) Store

// kgRunAddEdgeIdempotent covers the AddEdge happy path and idempotency.
func kgRunAddEdgeIdempotent(t *testing.T, newStore kgTestFactory) {
	s := newStore(t)
	ctx := context.Background()

	created, err := s.AddEdge(ctx, Edge{SrcID: "a", DstID: "b", EdgeType: "refines", Weight: 1.0})
	if err != nil {
		t.Fatalf("AddEdge: %v", err)
	}
	if !created {
		t.Error("first AddEdge: created = false, want true")
	}

	created, err = s.AddEdge(ctx, Edge{SrcID: "a", DstID: "b", EdgeType: "refines", Weight: 5.0})
	if err != nil {
		t.Fatalf("AddEdge repeat: %v", err)
	}
	if created {
		t.Error("repeat AddEdge: created = true, want false")
	}

	// Default weight: omitting Weight should land 1.0 in the row.
	created, err = s.AddEdge(ctx, Edge{SrcID: "a", DstID: "c", EdgeType: "evidence"})
	if err != nil || !created {
		t.Fatalf("AddEdge default weight: created=%v err=%v", created, err)
	}
	edges, err := s.ListEdges(ctx, "a", 1)
	if err != nil {
		t.Fatalf("ListEdges: %v", err)
	}
	var found bool
	for _, e := range edges {
		if e.Edge.DstID == "c" && e.Edge.EdgeType == "evidence" {
			if e.Edge.Weight != 1.0 {
				t.Errorf("default Weight = %v, want 1.0", e.Edge.Weight)
			}
			found = true
		}
	}
	if !found {
		t.Error("default-weight edge not found in ListEdges results")
	}
}

func kgRunRemoveEdge(t *testing.T, newStore kgTestFactory) {
	s := newStore(t)
	ctx := context.Background()

	// Missing edge -> deleted=false, no error.
	deleted, err := s.RemoveEdge(ctx, "x", "y", "refines")
	if err != nil {
		t.Fatalf("RemoveEdge missing: %v", err)
	}
	if deleted {
		t.Error("RemoveEdge missing: deleted = true, want false")
	}

	// Insert then remove.
	if _, err := s.AddEdge(ctx, Edge{SrcID: "x", DstID: "y", EdgeType: "refines"}); err != nil {
		t.Fatalf("AddEdge: %v", err)
	}
	deleted, err = s.RemoveEdge(ctx, "x", "y", "refines")
	if err != nil {
		t.Fatalf("RemoveEdge present: %v", err)
	}
	if !deleted {
		t.Error("RemoveEdge present: deleted = false, want true")
	}
}

func kgRunListEdgesTwoHops(t *testing.T, newStore kgTestFactory) {
	s := newStore(t)
	ctx := context.Background()
	// Build a -> b -> c chain.
	if _, err := s.AddEdge(ctx, Edge{SrcID: "a", DstID: "b", EdgeType: "refines"}); err != nil {
		t.Fatalf("AddEdge a->b: %v", err)
	}
	if _, err := s.AddEdge(ctx, Edge{SrcID: "b", DstID: "c", EdgeType: "refines"}); err != nil {
		t.Fatalf("AddEdge b->c: %v", err)
	}

	// hops=1: only a->b.
	hop1, err := s.ListEdges(ctx, "a", 1)
	if err != nil {
		t.Fatalf("ListEdges hops=1: %v", err)
	}
	if len(hop1) != 1 {
		t.Errorf("hops=1 returned %d edges, want 1", len(hop1))
	}
	if len(hop1) > 0 && hop1[0].Distance != 1 {
		t.Errorf("hops=1 distance = %d, want 1", hop1[0].Distance)
	}

	// hops=2: both edges, with correct distances.
	hop2, err := s.ListEdges(ctx, "a", 2)
	if err != nil {
		t.Fatalf("ListEdges hops=2: %v", err)
	}
	if len(hop2) != 2 {
		t.Fatalf("hops=2 returned %d edges, want 2", len(hop2))
	}
	distByDst := map[string]int{}
	for _, e := range hop2 {
		// "other" endpoint at this depth.
		if e.Distance == 1 {
			distByDst[e.Edge.DstID] = 1
		} else {
			// at depth 2 the new endpoint is c.
			distByDst[e.Edge.DstID] = e.Distance
		}
	}
	if distByDst["b"] != 1 {
		t.Errorf("distance to b = %d, want 1", distByDst["b"])
	}
	if distByDst["c"] != 2 {
		t.Errorf("distance to c = %d, want 2", distByDst["c"])
	}

	// Empty seed: no edges.
	empty, err := s.ListEdges(ctx, "nonexistent", 2)
	if err != nil {
		t.Fatalf("ListEdges empty: %v", err)
	}
	if len(empty) != 0 {
		t.Errorf("ListEdges on isolated id returned %d edges, want 0", len(empty))
	}
}

func kgRunUpsertEntityExactDedup(t *testing.T, newStore kgTestFactory) {
	s := newStore(t)
	ctx := context.Background()

	id1, created1, sim1, err := s.UpsertEntity(ctx, "config.go", "file", []float32{1, 0, 0})
	if err != nil {
		t.Fatalf("first UpsertEntity: %v", err)
	}
	if !created1 || sim1 {
		t.Errorf("first upsert: created=%v matchedBySim=%v, want true/false", created1, sim1)
	}

	id2, created2, sim2, err := s.UpsertEntity(ctx, "config.go", "file", []float32{0, 1, 0})
	if err != nil {
		t.Fatalf("repeat UpsertEntity: %v", err)
	}
	if created2 || sim2 {
		t.Errorf("repeat upsert: created=%v matchedBySim=%v, want false/false", created2, sim2)
	}
	if id1 != id2 {
		t.Errorf("repeat upsert returned different id: %q vs %q", id1, id2)
	}
}

func kgRunUpsertEntitySimilarityDedup(t *testing.T, newStore kgTestFactory) {
	s := newStore(t)
	ctx := context.Background()

	// Insert with a known unit vector.
	id1, _, _, err := s.UpsertEntity(ctx, "configloader", "file", []float32{1, 0, 0, 0})
	if err != nil {
		t.Fatalf("first UpsertEntity: %v", err)
	}

	// Different name, almost-parallel vector (cos ~= 0.96, well above 0.55).
	id2, created, sim, err := s.UpsertEntity(ctx, "config_loader", "file", []float32{0.96, 0.28, 0, 0})
	if err != nil {
		t.Fatalf("second UpsertEntity: %v", err)
	}
	if created {
		t.Error("similarity dedup: created = true, want false (should reuse)")
	}
	if !sim {
		t.Error("similarity dedup: matchedBySimilarity = false, want true")
	}
	if id1 != id2 {
		t.Errorf("similarity dedup returned different id: %q vs %q", id1, id2)
	}
}

func kgRunUpsertEntityDifferentType(t *testing.T, newStore kgTestFactory) {
	s := newStore(t)
	ctx := context.Background()
	// Same name, different entity_type, identical embedding -> two distinct ids.
	id1, _, _, err := s.UpsertEntity(ctx, "auth", "file", []float32{1, 0, 0})
	if err != nil {
		t.Fatalf("first UpsertEntity: %v", err)
	}
	id2, created2, sim2, err := s.UpsertEntity(ctx, "auth", "concept", []float32{1, 0, 0})
	if err != nil {
		t.Fatalf("second UpsertEntity: %v", err)
	}
	if !created2 {
		t.Error("cross-type upsert: created = false, want true")
	}
	if sim2 {
		t.Error("cross-type upsert: matchedBySim = true, want false (similarity is intra-type only)")
	}
	if id1 == id2 {
		t.Error("cross-type upsert collapsed onto same id; types should partition the namespace")
	}
}

func kgRunTickAllEntitiesNoAccess(t *testing.T, newStore kgTestFactory) {
	s := newStore(t)
	ctx := context.Background()

	id, _, _, err := s.UpsertEntity(ctx, "foo", "file", []float32{1, 0, 0})
	if err != nil {
		t.Fatalf("UpsertEntity: %v", err)
	}
	if err := s.TickAllEntities(ctx, nil); err != nil {
		t.Fatalf("Tick 1: %v", err)
	}
	if err := s.TickAllEntities(ctx, nil); err != nil {
		t.Fatalf("Tick 2: %v", err)
	}
	ent, err := s.GetEntity(ctx, id)
	if err != nil {
		t.Fatalf("GetEntity: %v", err)
	}
	if ent.TurnsSince != 2 {
		t.Errorf("turns_since = %d, want 2", ent.TurnsSince)
	}
	if ent.Retention >= 1.0 {
		t.Errorf("retention = %v, want < 1.0 after two ticks without access", ent.Retention)
	}
	if ent.Retention <= 0 {
		t.Errorf("retention = %v, want > 0", ent.Retention)
	}
}

func kgRunTickAllEntitiesAccessedReset(t *testing.T, newStore kgTestFactory) {
	s := newStore(t)
	ctx := context.Background()

	id, _, _, err := s.UpsertEntity(ctx, "bar", "file", []float32{1, 0, 0})
	if err != nil {
		t.Fatalf("UpsertEntity: %v", err)
	}

	// One tick without access -> turns_since=1.
	if err := s.TickAllEntities(ctx, nil); err != nil {
		t.Fatalf("Tick: %v", err)
	}
	// Now tick with the entity in accessedIDs -> resets.
	if err := s.TickAllEntities(ctx, []string{id}); err != nil {
		t.Fatalf("Tick accessed: %v", err)
	}
	ent, err := s.GetEntity(ctx, id)
	if err != nil {
		t.Fatalf("GetEntity: %v", err)
	}
	if ent.TurnsSince != 0 {
		t.Errorf("turns_since = %d, want 0 after access", ent.TurnsSince)
	}
	if ent.Retention != 1.0 {
		t.Errorf("retention = %v, want 1.0 after access", ent.Retention)
	}
	if ent.LastAccess.IsZero() {
		t.Error("last_access not bumped on access")
	}
}

func kgRunAddEntityMentionsIdempotent(t *testing.T, newStore kgTestFactory) {
	s := newStore(t)
	ctx := context.Background()

	factID, err := s.InsertFact(ctx, testdataFacts()[0])
	if err != nil {
		t.Fatalf("InsertFact: %v", err)
	}
	eid, _, _, err := s.UpsertEntity(ctx, "foo", "file", []float32{1, 0, 0})
	if err != nil {
		t.Fatalf("UpsertEntity: %v", err)
	}

	n, err := s.AddEntityMentions(ctx, factID, []string{eid}, "subject")
	if err != nil {
		t.Fatalf("AddEntityMentions: %v", err)
	}
	if n != 1 {
		t.Errorf("first AddEntityMentions inserted = %d, want 1", n)
	}

	n, err = s.AddEntityMentions(ctx, factID, []string{eid}, "subject")
	if err != nil {
		t.Fatalf("AddEntityMentions repeat: %v", err)
	}
	if n != 0 {
		t.Errorf("repeat AddEntityMentions inserted = %d, want 0", n)
	}
}

func kgRunListMemoriesByEntity(t *testing.T, newStore kgTestFactory) {
	s := newStore(t)
	ctx := context.Background()

	factID, err := s.InsertFact(ctx, testdataFacts()[0])
	if err != nil {
		t.Fatalf("InsertFact: %v", err)
	}
	epID, err := s.InsertEpisode(ctx, testdataEpisodes()[0])
	if err != nil {
		t.Fatalf("InsertEpisode: %v", err)
	}
	eid, _, _, err := s.UpsertEntity(ctx, "auth", "file", []float32{1, 0, 0})
	if err != nil {
		t.Fatalf("UpsertEntity: %v", err)
	}
	if _, err := s.AddEntityMentions(ctx, factID, []string{eid}, ""); err != nil {
		t.Fatalf("AddEntityMentions fact: %v", err)
	}
	if _, err := s.AddEntityMentions(ctx, epID, []string{eid}, ""); err != nil {
		t.Fatalf("AddEntityMentions episode: %v", err)
	}

	refs, err := s.ListMemoriesByEntity(ctx, eid, 25)
	if err != nil {
		t.Fatalf("ListMemoriesByEntity: %v", err)
	}
	if len(refs) != 2 {
		t.Fatalf("ListMemoriesByEntity returned %d, want 2", len(refs))
	}
	gotLayers := map[MemoryType]bool{}
	for _, r := range refs {
		gotLayers[r.Layer] = true
		if r.Content == "" {
			t.Errorf("ref %s: content empty", r.ID)
		}
		if len(r.Content) > 120 {
			t.Errorf("ref %s: content not truncated (len=%d)", r.ID, len(r.Content))
		}
	}
	if !gotLayers[TypeL2Semantic] || !gotLayers[TypeL3Episodic] {
		t.Errorf("layers seen = %v, want both l2 and l3", gotLayers)
	}
}

func kgRunListEntityNeighborsHops1(t *testing.T, newStore kgTestFactory) {
	s := newStore(t)
	ctx := context.Background()

	factID, err := s.InsertFact(ctx, testdataFacts()[0])
	if err != nil {
		t.Fatalf("InsertFact: %v", err)
	}
	// Two entities, one mention, one entity-entity edge.
	a, _, _, err := s.UpsertEntity(ctx, "alpha", "file", []float32{1, 0, 0})
	if err != nil {
		t.Fatalf("UpsertEntity a: %v", err)
	}
	b, _, _, err := s.UpsertEntity(ctx, "beta", "file", []float32{0, 1, 0})
	if err != nil {
		t.Fatalf("UpsertEntity b: %v", err)
	}
	if _, err := s.AddEntityMentions(ctx, factID, []string{a}, ""); err != nil {
		t.Fatalf("AddEntityMentions: %v", err)
	}
	if _, err := s.AddEdge(ctx, Edge{SrcID: a, DstID: b, EdgeType: "references"}); err != nil {
		t.Fatalf("AddEdge a->b: %v", err)
	}

	memories, entities, err := s.ListEntityNeighbors(ctx, a, 1)
	if err != nil {
		t.Fatalf("ListEntityNeighbors: %v", err)
	}
	if len(memories) != 1 {
		t.Errorf("memories = %d, want 1", len(memories))
	} else {
		if memories[0].ID != factID {
			t.Errorf("memory id = %q, want %q", memories[0].ID, factID)
		}
		if memories[0].Distance != 1 {
			t.Errorf("memory distance = %d, want 1", memories[0].Distance)
		}
	}
	if len(entities) != 1 {
		t.Errorf("entities = %d, want 1", len(entities))
	} else {
		if entities[0].ID != b {
			t.Errorf("neighbor entity id = %q, want %q", entities[0].ID, b)
		}
		if entities[0].Name != "beta" {
			t.Errorf("neighbor entity name = %q, want %q", entities[0].Name, "beta")
		}
	}
}

func kgRunListEntitiesByMemoryIDs(t *testing.T, newStore kgTestFactory) {
	s := newStore(t)
	ctx := context.Background()

	// Empty input → empty (non-nil) slice, no error.
	got, err := s.ListEntitiesByMemoryIDs(ctx, nil)
	if err != nil {
		t.Fatalf("ListEntitiesByMemoryIDs(nil): %v", err)
	}
	if got == nil {
		t.Error("nil input: returned nil slice, want non-nil empty slice")
	}
	if len(got) != 0 {
		t.Errorf("nil input: len=%d, want 0", len(got))
	}

	// Two facts, three entities, mixed mentions:
	//   F1 → {E1, E2}
	//   F2 → {E2, E3}    (E2 must dedupe in the result)
	f1, err := s.InsertFact(ctx, testdataFacts()[0])
	if err != nil {
		t.Fatalf("InsertFact f1: %v", err)
	}
	f2, err := s.InsertFact(ctx, testdataFacts()[1])
	if err != nil {
		t.Fatalf("InsertFact f2: %v", err)
	}
	e1, _, _, err := s.UpsertEntity(ctx, "e1", "file", []float32{1, 0, 0})
	if err != nil {
		t.Fatalf("UpsertEntity e1: %v", err)
	}
	e2, _, _, err := s.UpsertEntity(ctx, "e2", "file", []float32{0, 1, 0})
	if err != nil {
		t.Fatalf("UpsertEntity e2: %v", err)
	}
	e3, _, _, err := s.UpsertEntity(ctx, "e3", "file", []float32{0, 0, 1})
	if err != nil {
		t.Fatalf("UpsertEntity e3: %v", err)
	}
	if _, err := s.AddEntityMentions(ctx, f1, []string{e1, e2}, ""); err != nil {
		t.Fatalf("AddEntityMentions f1: %v", err)
	}
	if _, err := s.AddEntityMentions(ctx, f2, []string{e2, e3}, ""); err != nil {
		t.Fatalf("AddEntityMentions f2: %v", err)
	}

	got, err = s.ListEntitiesByMemoryIDs(ctx, []string{f1, f2})
	if err != nil {
		t.Fatalf("ListEntitiesByMemoryIDs: %v", err)
	}
	if len(got) != 3 {
		t.Errorf("len=%d, want 3 (E1, E2 deduped, E3): got=%v", len(got), got)
	}
	gotSet := map[string]bool{}
	for _, id := range got {
		gotSet[id] = true
	}
	for _, want := range []string{e1, e2, e3} {
		if !gotSet[want] {
			t.Errorf("missing entity id %q in result %v", want, got)
		}
	}
}

func kgRunCountEntitiesAndEdges(t *testing.T, newStore kgTestFactory) {
	s := newStore(t)
	ctx := context.Background()

	// Empty.
	if n, err := s.CountEntities(ctx); err != nil || n != 0 {
		t.Errorf("CountEntities empty: n=%d err=%v, want 0/nil", n, err)
	}
	if n, err := s.CountEdges(ctx); err != nil || n != 0 {
		t.Errorf("CountEdges empty: n=%d err=%v, want 0/nil", n, err)
	}

	// Seed.
	if _, _, _, err := s.UpsertEntity(ctx, "a", "file", []float32{1, 0, 0}); err != nil {
		t.Fatalf("UpsertEntity a: %v", err)
	}
	if _, _, _, err := s.UpsertEntity(ctx, "b", "file", []float32{0, 1, 0}); err != nil {
		t.Fatalf("UpsertEntity b: %v", err)
	}
	if _, err := s.AddEdge(ctx, Edge{SrcID: "x", DstID: "y", EdgeType: "refines"}); err != nil {
		t.Fatalf("AddEdge: %v", err)
	}

	if n, err := s.CountEntities(ctx); err != nil || n != 2 {
		t.Errorf("CountEntities seeded: n=%d err=%v, want 2/nil", n, err)
	}
	if n, err := s.CountEdges(ctx); err != nil || n != 1 {
		t.Errorf("CountEdges seeded: n=%d err=%v, want 1/nil", n, err)
	}
}

func kgRunCascadeOnDeleteFact(t *testing.T, newStore kgTestFactory) {
	s := newStore(t)
	ctx := context.Background()

	factID, err := s.InsertFact(ctx, testdataFacts()[0])
	if err != nil {
		t.Fatalf("InsertFact: %v", err)
	}
	eid, _, _, err := s.UpsertEntity(ctx, "auth", "file", []float32{1, 0, 0})
	if err != nil {
		t.Fatalf("UpsertEntity: %v", err)
	}
	if _, err := s.AddEntityMentions(ctx, factID, []string{eid}, ""); err != nil {
		t.Fatalf("AddEntityMentions: %v", err)
	}
	// Edge involving the fact (fact -> entity).
	if _, err := s.AddEdge(ctx, Edge{SrcID: factID, DstID: eid, EdgeType: "depends_on"}); err != nil {
		t.Fatalf("AddEdge: %v", err)
	}

	// Pre-delete sanity.
	mems, err := s.ListMemoriesByEntity(ctx, eid, 10)
	if err != nil {
		t.Fatalf("pre ListMemoriesByEntity: %v", err)
	}
	if len(mems) != 1 {
		t.Fatalf("pre-delete mentions = %d, want 1", len(mems))
	}

	if err := s.DeleteFact(ctx, factID); err != nil {
		t.Fatalf("DeleteFact: %v", err)
	}

	// Mention should be gone.
	mems, err = s.ListMemoriesByEntity(ctx, eid, 10)
	if err != nil {
		t.Fatalf("post ListMemoriesByEntity: %v", err)
	}
	if len(mems) != 0 {
		t.Errorf("post-delete mentions = %d, want 0", len(mems))
	}
	// Edge from the fact should be gone.
	edges, err := s.ListEdges(ctx, factID, 1)
	if err != nil {
		t.Fatalf("ListEdges: %v", err)
	}
	if len(edges) != 0 {
		t.Errorf("post-delete edges = %d, want 0", len(edges))
	}
	// Entity itself must survive.
	ent, err := s.GetEntity(ctx, eid)
	if err != nil {
		t.Fatalf("GetEntity: %v", err)
	}
	if ent.ID == "" {
		t.Error("entity was cascade-deleted; should survive")
	}
}

// --- Phase 7C: ExpandViaGraph test helpers ---

// kgInsertFactInCluster wires up the default cluster on a fresh store
// and inserts a fact with a derived content+embedding so each call
// produces a unique row.
func kgInsertFactInCluster(t *testing.T, s Store, content string) string {
	t.Helper()
	f := testdataFacts()[0]
	f.ID = ""
	f.Content = content
	f.ContentHash = ""
	// Embedding is irrelevant for these tests but must be non-empty.
	f.Embedding = []float32{1, 0, 0, 0}
	id, err := s.InsertFact(context.Background(), f)
	if err != nil {
		t.Fatalf("InsertFact %q: %v", content, err)
	}
	return id
}

// kgFindHits filters hits to those matching a (neighbor, seed) pair.
// Returns the matched hits in ascending Distance order.
func kgFindHits(hits []GraphHit, neighbor, seed string) []GraphHit {
	var out []GraphHit
	for _, h := range hits {
		if h.NeighborID == neighbor && (seed == "" || h.SeedID == seed) {
			out = append(out, h)
		}
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Distance < out[j].Distance })
	return out
}

// kgRunExpandViaGraph_DirectEdge: M1 edge-> M2; expand from M1 hops=1
// returns M2 at distance 1.
func kgRunExpandViaGraph_DirectEdge(t *testing.T, newStore kgTestFactory) {
	s := newStore(t)
	ctx := context.Background()
	m1 := kgInsertFactInCluster(t, s, "m1 content")
	m2 := kgInsertFactInCluster(t, s, "m2 content")
	if _, err := s.AddEdge(ctx, Edge{SrcID: m1, DstID: m2, EdgeType: "refines"}); err != nil {
		t.Fatalf("AddEdge: %v", err)
	}
	hits, err := s.ExpandViaGraph(ctx, []string{m1}, 1, 0, 0)
	if err != nil {
		t.Fatalf("ExpandViaGraph: %v", err)
	}
	got := kgFindHits(hits, m2, m1)
	if len(got) != 1 {
		t.Fatalf("expected 1 hit (m2 from m1), got %d, hits=%v", len(got), hits)
	}
	if got[0].Distance != 1 {
		t.Errorf("distance = %d, want 1", got[0].Distance)
	}
	if got[0].NeighborLayer != string(TypeL2Semantic) {
		t.Errorf("layer = %q, want %q", got[0].NeighborLayer, string(TypeL2Semantic))
	}
}

// kgRunExpandViaGraph_EntityMention: M1 mentions E, M2 mentions E;
// expand from M1 hops=2 returns M2 at distance 2.
func kgRunExpandViaGraph_EntityMention(t *testing.T, newStore kgTestFactory) {
	s := newStore(t)
	ctx := context.Background()
	m1 := kgInsertFactInCluster(t, s, "m1 entity")
	m2 := kgInsertFactInCluster(t, s, "m2 entity")
	eid, _, _, err := s.UpsertEntity(ctx, "shared", "concept", []float32{1, 0, 0})
	if err != nil {
		t.Fatalf("UpsertEntity: %v", err)
	}
	if _, err := s.AddEntityMentions(ctx, m1, []string{eid}, ""); err != nil {
		t.Fatalf("AddEntityMentions m1: %v", err)
	}
	if _, err := s.AddEntityMentions(ctx, m2, []string{eid}, ""); err != nil {
		t.Fatalf("AddEntityMentions m2: %v", err)
	}
	hits, err := s.ExpandViaGraph(ctx, []string{m1}, 2, 0, 0)
	if err != nil {
		t.Fatalf("ExpandViaGraph: %v", err)
	}
	got := kgFindHits(hits, m2, m1)
	if len(got) != 1 {
		t.Fatalf("expected 1 hit (m2 from m1 via entity), got %d", len(got))
	}
	if got[0].Distance != 2 {
		t.Errorf("distance = %d, want 2", got[0].Distance)
	}
	// hops=1 must NOT return m2 (entity hop costs 2).
	hits1, err := s.ExpandViaGraph(ctx, []string{m1}, 1, 0, 0)
	if err != nil {
		t.Fatalf("ExpandViaGraph hops=1: %v", err)
	}
	if len(kgFindHits(hits1, m2, m1)) != 0 {
		t.Errorf("hops=1 returned m2 via entity; expected exclusion")
	}
}

// kgRunExpandViaGraph_TwoHopEdgeChain: M1->M2->M3 edges; expand from
// M1 hops=2 returns both with correct distances.
func kgRunExpandViaGraph_TwoHopEdgeChain(t *testing.T, newStore kgTestFactory) {
	s := newStore(t)
	ctx := context.Background()
	m1 := kgInsertFactInCluster(t, s, "m1 chain")
	m2 := kgInsertFactInCluster(t, s, "m2 chain")
	m3 := kgInsertFactInCluster(t, s, "m3 chain")
	if _, err := s.AddEdge(ctx, Edge{SrcID: m1, DstID: m2, EdgeType: "refines"}); err != nil {
		t.Fatalf("AddEdge m1->m2: %v", err)
	}
	if _, err := s.AddEdge(ctx, Edge{SrcID: m2, DstID: m3, EdgeType: "refines"}); err != nil {
		t.Fatalf("AddEdge m2->m3: %v", err)
	}
	hits, err := s.ExpandViaGraph(ctx, []string{m1}, 2, 0, 0)
	if err != nil {
		t.Fatalf("ExpandViaGraph: %v", err)
	}
	g2 := kgFindHits(hits, m2, m1)
	g3 := kgFindHits(hits, m3, m1)
	if len(g2) != 1 || g2[0].Distance != 1 {
		t.Errorf("m2 from m1: got=%v, want one at distance=1", g2)
	}
	if len(g3) != 1 || g3[0].Distance != 2 {
		t.Errorf("m3 from m1: got=%v, want one at distance=2", g3)
	}
}

// kgRunExpandViaGraph_HubEntityNoCap: an entity mentioned by many
// memories expands without per-seed truncation -- ExpandViaGraph has
// no per-seed cap.
func kgRunExpandViaGraph_HubEntityNoCap(t *testing.T, newStore kgTestFactory) {
	s := newStore(t)
	ctx := context.Background()
	eid, _, _, err := s.UpsertEntity(ctx, "hub", "concept", []float32{1, 0, 0})
	if err != nil {
		t.Fatalf("UpsertEntity: %v", err)
	}
	const N = 12
	ids := make([]string, N)
	for i := 0; i < N; i++ {
		mid := kgInsertFactInCluster(t, s, "hub-mem-"+itoa(i))
		ids[i] = mid
		if _, err := s.AddEntityMentions(ctx, mid, []string{eid}, ""); err != nil {
			t.Fatalf("AddEntityMentions[%d]: %v", i, err)
		}
	}
	// Use a large cap so no truncation occurs.
	hits, err := s.ExpandViaGraph(ctx, []string{ids[0]}, 2, 0, 1000)
	if err != nil {
		t.Fatalf("ExpandViaGraph: %v", err)
	}
	// Should find all N-1 other memories at distance 2.
	uniq := map[string]struct{}{}
	for _, h := range hits {
		if h.SeedID == ids[0] {
			uniq[h.NeighborID] = struct{}{}
		}
	}
	if len(uniq) != N-1 {
		t.Errorf("hub expansion returned %d unique neighbors, want %d", len(uniq), N-1)
	}
}

// kgRunExpandViaGraph_GlobalCapHonored: small maxVisited yields exactly
// that many distinct visited memories (plus the seed already-visited).
func kgRunExpandViaGraph_GlobalCapHonored(t *testing.T, newStore kgTestFactory) {
	s := newStore(t)
	ctx := context.Background()
	const N = 10
	ids := make([]string, N)
	for i := 0; i < N; i++ {
		ids[i] = kgInsertFactInCluster(t, s, "cap-mem-"+itoa(i))
	}
	// star from ids[0] to all others
	for i := 1; i < N; i++ {
		if _, err := s.AddEdge(ctx, Edge{SrcID: ids[0], DstID: ids[i], EdgeType: "refines"}); err != nil {
			t.Fatalf("AddEdge: %v", err)
		}
	}
	// maxVisited=4 -> seed(1) + 3 new = 4 total distinct memories.
	// So at most 3 neighbor hits should be recorded.
	hits, err := s.ExpandViaGraph(ctx, []string{ids[0]}, 2, 0, 4)
	if err != nil {
		t.Fatalf("ExpandViaGraph: %v", err)
	}
	uniq := map[string]struct{}{}
	for _, h := range hits {
		uniq[h.NeighborID] = struct{}{}
	}
	if len(uniq) > 3 {
		t.Errorf("global cap honored: got %d unique neighbors, want <= 3 (cap=4 incl. seed)", len(uniq))
	}
	if len(uniq) == 0 {
		t.Error("cap=4 should still allow some expansion; got zero")
	}
}

// kgRunExpandViaGraph_RetentionPrefilter: a low-retention cluster
// short-circuits BFS through its members.
func kgRunExpandViaGraph_RetentionPrefilter(t *testing.T, newStore kgTestFactory) {
	s := newStore(t)
	ctx := context.Background()

	// Build two clusters with vastly different decay profiles.
	freshCluster := ClusterNode{ID: "fresh-rp", Summary: "fresh", Utility: 1.0, Frequency: 1.0, TurnsSince: 0}
	if err := s.CreateCluster(ctx, freshCluster); err != nil {
		t.Fatalf("CreateCluster fresh: %v", err)
	}
	decayedCluster := ClusterNode{ID: "decayed-rp", Summary: "decayed", Utility: 0.0, Frequency: 0.0, TurnsSince: 100000}
	if err := s.CreateCluster(ctx, decayedCluster); err != nil {
		t.Fatalf("CreateCluster decayed: %v", err)
	}

	mkFact := func(content, cluster string) string {
		f := testdataFacts()[0]
		f.ID = ""
		f.Content = content
		f.ContentHash = ""
		f.ClusterID = cluster
		f.Embedding = []float32{1, 0, 0, 0}
		id, err := s.InsertFact(ctx, f)
		if err != nil {
			t.Fatalf("InsertFact: %v", err)
		}
		return id
	}

	m1 := mkFact("rp m1", "fresh-rp")   // seed: high retention
	m2 := mkFact("rp m2", "decayed-rp") // low retention -> pruned
	m3 := mkFact("rp m3", "fresh-rp")   // beyond m2; unreachable since m2 pruned

	if _, err := s.AddEdge(ctx, Edge{SrcID: m1, DstID: m2, EdgeType: "refines"}); err != nil {
		t.Fatalf("AddEdge m1->m2: %v", err)
	}
	if _, err := s.AddEdge(ctx, Edge{SrcID: m2, DstID: m3, EdgeType: "refines"}); err != nil {
		t.Fatalf("AddEdge m2->m3: %v", err)
	}

	hits, err := s.ExpandViaGraph(ctx, []string{m1}, 2, 0.5, 0)
	if err != nil {
		t.Fatalf("ExpandViaGraph: %v", err)
	}
	// m2 should be filtered (low retention) -> m3 also unreachable.
	if len(kgFindHits(hits, m2, m1)) != 0 {
		t.Errorf("m2 (decayed) appeared in hits; expected pre-filter exclusion")
	}
	if len(kgFindHits(hits, m3, m1)) != 0 {
		t.Errorf("m3 reached past pruned m2; expected zero hits")
	}
}

// kgRunExpandViaGraph_MultiSeedReturnsAllPairs: a neighbor reachable
// from multiple seeds yields one GraphHit per seed.
func kgRunExpandViaGraph_MultiSeedReturnsAllPairs(t *testing.T, newStore kgTestFactory) {
	s := newStore(t)
	ctx := context.Background()
	s1 := kgInsertFactInCluster(t, s, "ms s1")
	s2 := kgInsertFactInCluster(t, s, "ms s2")
	mx := kgInsertFactInCluster(t, s, "ms mx")

	// s1 -> mx (direct edge, dist=1)
	if _, err := s.AddEdge(ctx, Edge{SrcID: s1, DstID: mx, EdgeType: "refines"}); err != nil {
		t.Fatalf("AddEdge s1->mx: %v", err)
	}
	// s2 mentions entity E; mx mentions E -> 2 hop.
	eid, _, _, err := s.UpsertEntity(ctx, "ms-ent", "concept", []float32{1, 0, 0})
	if err != nil {
		t.Fatalf("UpsertEntity: %v", err)
	}
	if _, err := s.AddEntityMentions(ctx, s2, []string{eid}, ""); err != nil {
		t.Fatalf("AddEntityMentions s2: %v", err)
	}
	if _, err := s.AddEntityMentions(ctx, mx, []string{eid}, ""); err != nil {
		t.Fatalf("AddEntityMentions mx: %v", err)
	}

	hits, err := s.ExpandViaGraph(ctx, []string{s1, s2}, 2, 0, 0)
	if err != nil {
		t.Fatalf("ExpandViaGraph: %v", err)
	}
	g1 := kgFindHits(hits, mx, s1)
	g2 := kgFindHits(hits, mx, s2)
	if len(g1) != 1 || g1[0].Distance != 1 {
		t.Errorf("from s1: %v want one at distance=1", g1)
	}
	if len(g2) != 1 || g2[0].Distance != 2 {
		t.Errorf("from s2: %v want one at distance=2", g2)
	}
}

// kgRunExpandViaGraph_SameSeedShortestPath: within a single seed, if
// the same neighbor is reachable both via edge (dist=1) and entity
// (dist=2), only the dist=1 row is recorded.
func kgRunExpandViaGraph_SameSeedShortestPath(t *testing.T, newStore kgTestFactory) {
	s := newStore(t)
	ctx := context.Background()
	src := kgInsertFactInCluster(t, s, "ss src")
	dst := kgInsertFactInCluster(t, s, "ss dst")
	if _, err := s.AddEdge(ctx, Edge{SrcID: src, DstID: dst, EdgeType: "refines"}); err != nil {
		t.Fatalf("AddEdge: %v", err)
	}
	// Also share an entity so the entity path can produce a dist=2 hit.
	eid, _, _, err := s.UpsertEntity(ctx, "ss-ent", "concept", []float32{1, 0, 0})
	if err != nil {
		t.Fatalf("UpsertEntity: %v", err)
	}
	if _, err := s.AddEntityMentions(ctx, src, []string{eid}, ""); err != nil {
		t.Fatalf("AddEntityMentions src: %v", err)
	}
	if _, err := s.AddEntityMentions(ctx, dst, []string{eid}, ""); err != nil {
		t.Fatalf("AddEntityMentions dst: %v", err)
	}

	hits, err := s.ExpandViaGraph(ctx, []string{src}, 2, 0, 0)
	if err != nil {
		t.Fatalf("ExpandViaGraph: %v", err)
	}
	got := kgFindHits(hits, dst, src)
	if len(got) != 1 {
		t.Fatalf("same seed shortest path: got %d hits, want 1; hits=%v", len(got), got)
	}
	if got[0].Distance != 1 {
		t.Errorf("distance = %d, want 1 (shortest path wins)", got[0].Distance)
	}
}

// itoa is a tiny inline conversion to avoid pulling strconv into this
// test file just for index suffixes.
func itoa(i int) string {
	if i == 0 {
		return "0"
	}
	var digits []byte
	for i > 0 {
		digits = append([]byte{byte('0' + i%10)}, digits...)
		i /= 10
	}
	return string(digits)
}
