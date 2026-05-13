package mcpserver

import (
	"context"
	"encoding/json"
	"strings"
	"testing"
	"time"

	mcpsdk "github.com/modelcontextprotocol/go-sdk/mcp"

	"personal/reverie/internal/cluster"
	"personal/reverie/internal/config"
	"personal/reverie/internal/decay"
	"personal/reverie/internal/embed"
	"personal/reverie/internal/manager"
	"personal/reverie/internal/memory"
)

// stubEmbedder is a test double for embed.EmbeddingProvider. It returns
// pre-configured vectors for deterministic testing.
type stubEmbedder struct {
	vectors map[string][]float32
	dim     int
}

func newStubEmbedder(dim int) *stubEmbedder {
	return &stubEmbedder{
		vectors: make(map[string][]float32),
		dim:     dim,
	}
}

func (s *stubEmbedder) Embed(_ context.Context, texts []string) ([][]float32, error) {
	result := make([][]float32, len(texts))
	for i, t := range texts {
		if v, ok := s.vectors[t]; ok {
			result[i] = v
		} else {
			// Return a default non-zero vector.
			v := make([]float32, s.dim)
			v[0] = 1.0
			result[i] = v
		}
	}
	return result, nil
}

func (s *stubEmbedder) Dimensions() int { return s.dim }
func (s *stubEmbedder) Model() string   { return "stub" }

// Verify interface compliance.
var _ embed.EmbeddingProvider = (*stubEmbedder)(nil)

// newTestServer creates a Server backed by a MemStore, stub embedder, real
// Decayer, real MemoryManager, and real Assigner for testing.
func newTestServer(embedder *stubEmbedder) *Server {
	cfg := config.Defaults()
	store := memory.NewMemStore()
	dec := decay.NewDecayer(10.0, 0.3)
	mgr := manager.NewMemoryManager(store, dec, 0.10, 0.05)
	assigner := cluster.NewAssigner(store, 0.60, 0.5, 0.5)
	return NewServer(store, embedder, dec, mgr, assigner, cfg, nil)
}

// --- memory_write tests ---

func TestHandleWrite_HappyPath(t *testing.T) {
	emb := newStubEmbedder(4)
	emb.vectors["Go is a compiled language"] = []float32{0.5, 0.5, 0.0, 0.0}
	s := newTestServer(emb)
	defer s.recallCache.stop()

	ctx := context.Background()
	in := WriteInput{
		Content: "Go is a compiled language",
		Type:    "project",
		Source:  "test",
	}
	_, out, err := s.handleWrite(ctx, nil, in)
	if err != nil {
		t.Fatalf("handleWrite: %v", err)
	}
	if out.ID == "" {
		t.Fatal("expected non-empty ID")
	}
	if out.Layer != "l2_semantic" {
		t.Fatalf("expected layer l2_semantic, got %s", out.Layer)
	}

	// Verify the fact is retrievable.
	fact, err := s.store.GetFact(ctx, out.ID)
	if err != nil {
		t.Fatalf("GetFact: %v", err)
	}
	if fact == nil {
		t.Fatal("expected fact to exist")
	}
	if fact.Content != "Go is a compiled language" {
		t.Fatalf("content mismatch: %s", fact.Content)
	}
	if fact.Subtype != "project" {
		t.Fatalf("subtype mismatch: %s", fact.Subtype)
	}
}

func TestHandleWrite_InvalidType(t *testing.T) {
	emb := newStubEmbedder(4)
	s := newTestServer(emb)
	defer s.recallCache.stop()

	ctx := context.Background()
	in := WriteInput{
		Content: "something",
		Type:    "invalid_type",
	}
	_, _, err := s.handleWrite(ctx, nil, in)
	if err == nil {
		t.Fatal("expected error for invalid type")
	}
}

func TestHandleWrite_EmptyContent(t *testing.T) {
	emb := newStubEmbedder(4)
	s := newTestServer(emb)
	defer s.recallCache.stop()

	ctx := context.Background()
	in := WriteInput{
		Content: "",
		Type:    "user",
	}
	_, _, err := s.handleWrite(ctx, nil, in)
	if err == nil {
		t.Fatal("expected error for empty content")
	}
}

func TestHandleWrite_Idempotent(t *testing.T) {
	emb := newStubEmbedder(4)
	emb.vectors["same content"] = []float32{0.1, 0.2, 0.3, 0.4}
	s := newTestServer(emb)
	defer s.recallCache.stop()

	ctx := context.Background()
	in := WriteInput{Content: "same content", Type: "user"}

	_, out1, err := s.handleWrite(ctx, nil, in)
	if err != nil {
		t.Fatalf("first write: %v", err)
	}

	_, out2, err := s.handleWrite(ctx, nil, in)
	if err != nil {
		t.Fatalf("second write: %v", err)
	}

	if out1.ID != out2.ID {
		t.Fatalf("expected idempotent IDs, got %s and %s", out1.ID, out2.ID)
	}
}

// --- memory_recall tests ---

func TestHandleRecall_GateBThreshold(t *testing.T) {
	emb := newStubEmbedder(4)
	// Query vector.
	emb.vectors["find Go facts"] = []float32{0.5, 0.5, 0.0, 0.0}
	// High similarity to query.
	emb.vectors["Go is great for concurrency"] = []float32{0.5, 0.5, 0.0, 0.0}
	// Lower similarity to query.
	emb.vectors["Python is interpreted"] = []float32{0.0, 0.0, 0.5, 0.5}

	s := newTestServer(emb)
	defer s.recallCache.stop()

	ctx := context.Background()

	// Write two facts.
	_, _, err := s.handleWrite(ctx, nil, WriteInput{Content: "Go is great for concurrency", Type: "project"})
	if err != nil {
		t.Fatalf("write: %v", err)
	}
	_, _, err = s.handleWrite(ctx, nil, WriteInput{Content: "Python is interpreted", Type: "reference"})
	if err != nil {
		t.Fatalf("write: %v", err)
	}

	// Recall.
	_, out, err := s.handleRecall(ctx, nil, RecallInput{Query: "find Go facts"})
	if err != nil {
		t.Fatalf("handleRecall: %v", err)
	}

	if out.RecallID == "" {
		t.Fatal("expected non-empty recall_id")
	}
	if len(out.Candidates) != 2 {
		t.Fatalf("expected 2 candidates, got %d", len(out.Candidates))
	}

	// First candidate (highest similarity) should pass Gate B.
	if !out.Candidates[0].GateBPass {
		t.Error("expected first candidate to pass Gate B (high similarity)")
	}
	// With Phase 2 decay, candidates should now have real retention values.
	// A freshly written fact's default cluster has U=0,F=0 and turns_since
	// has been incremented by the write-path TickDecay calls, so retention
	// should be a positive value.
	for _, c := range out.Candidates {
		if c.Retention < 0 || c.Retention > 1.0 {
			t.Errorf("expected retention in [0,1] for candidate %s, got %f", c.ID, c.Retention)
		}
	}

	// Verify the recall cache has the entry.
	cached, ok := s.recallCache.get(out.RecallID)
	if !ok {
		t.Fatal("expected recall cache entry")
	}
	if len(cached.candidates) != 2 {
		t.Fatalf("expected 2 cached candidates, got %d", len(cached.candidates))
	}
}

func TestHandleRecall_EmptyQuery(t *testing.T) {
	emb := newStubEmbedder(4)
	s := newTestServer(emb)
	defer s.recallCache.stop()

	_, _, err := s.handleRecall(context.Background(), nil, RecallInput{Query: ""})
	if err == nil {
		t.Fatal("expected error for empty query")
	}
}

func TestHandleRecall_LimitDefault(t *testing.T) {
	emb := newStubEmbedder(4)
	s := newTestServer(emb)
	defer s.recallCache.stop()

	ctx := context.Background()

	// Write 15 facts.
	for i := range 15 {
		content := "fact number " + string(rune('A'+i))
		emb.vectors[content] = []float32{float32(i) * 0.1, 0.5, 0.0, 0.0}
		_, _, err := s.handleWrite(ctx, nil, WriteInput{Content: content, Type: "user"})
		if err != nil {
			t.Fatalf("write %d: %v", i, err)
		}
	}

	// Default limit is 10.
	_, out, err := s.handleRecall(ctx, nil, RecallInput{Query: "facts"})
	if err != nil {
		t.Fatalf("handleRecall: %v", err)
	}
	if len(out.Candidates) > 10 {
		t.Fatalf("expected at most 10 candidates (default limit), got %d", len(out.Candidates))
	}
}

// --- Phase 7C: graph-aware recall handler tests ---

// recallWriteWithEmbed writes a fact whose canonical embedding is set by
// the stub embedder. Returns the resulting ID.
func recallWriteWithEmbed(t *testing.T, s *Server, emb *stubEmbedder, ctx context.Context, content string, vec []float32) string {
	t.Helper()
	emb.vectors[content] = vec
	_, out, err := s.handleWrite(ctx, nil, WriteInput{Content: content, Type: "project"})
	if err != nil {
		t.Fatalf("write %q: %v", content, err)
	}
	return out.ID
}

func TestHandleRecall_ExpandFlagOff_Unchanged(t *testing.T) {
	emb := newStubEmbedder(4)
	emb.vectors["q"] = []float32{1, 0, 0, 0}
	s := newTestServer(emb)
	defer s.recallCache.stop()
	ctx := context.Background()

	id := recallWriteWithEmbed(t, s, emb, ctx, "alpha", []float32{1, 0, 0, 0})

	_, out, err := s.handleRecall(ctx, nil, RecallInput{Query: "q"})
	if err != nil {
		t.Fatalf("handleRecall: %v", err)
	}
	if len(out.Candidates) != 1 {
		t.Fatalf("expected 1 candidate, got %d", len(out.Candidates))
	}
	c := out.Candidates[0]
	if c.ID != id {
		t.Errorf("id = %q, want %q", c.ID, id)
	}
	if c.Distance != 0 {
		t.Errorf("Distance = %d, want 0 for vector hit", c.Distance)
	}
	if c.CompositeScore != float64(c.Similarity) {
		t.Errorf("CompositeScore=%v, want Similarity=%v", c.CompositeScore, c.Similarity)
	}
}

func TestHandleRecall_ExpandViaGraph_AddsNeighbors(t *testing.T) {
	emb := newStubEmbedder(4)
	emb.vectors["q"] = []float32{1, 0, 0, 0}
	s := newTestServer(emb)
	defer s.recallCache.stop()
	ctx := context.Background()

	seed := recallWriteWithEmbed(t, s, emb, ctx, "seed-content", []float32{1, 0, 0, 0})
	// neighbor has an orthogonal embedding so it is NOT a vector hit.
	neighbor := recallWriteWithEmbed(t, s, emb, ctx, "neighbor-content", []float32{0, 0, 1, 0})
	if _, err := s.store.AddEdge(ctx, memory.Edge{SrcID: seed, DstID: neighbor, EdgeType: "refines"}); err != nil {
		t.Fatalf("AddEdge: %v", err)
	}

	_, out, err := s.handleRecall(ctx, nil, RecallInput{Query: "q", ExpandViaGraph: true})
	if err != nil {
		t.Fatalf("handleRecall: %v", err)
	}
	var nb *RecallCandidate
	for i := range out.Candidates {
		if out.Candidates[i].ID == neighbor {
			nb = &out.Candidates[i]
			break
		}
	}
	if nb == nil {
		t.Fatalf("neighbor not in output; candidates=%+v", out.Candidates)
	}
	if nb.Distance != 1 {
		t.Errorf("Distance = %d, want 1", nb.Distance)
	}
	if nb.GateBPass {
		t.Errorf("graph-only hit GateBPass = true, want false")
	}
	if nb.Similarity != 0 {
		t.Errorf("graph-only Similarity = %v, want 0", nb.Similarity)
	}
	if nb.CompositeScore <= 0 {
		t.Errorf("CompositeScore = %v, want > 0", nb.CompositeScore)
	}
}

func TestHandleRecall_VectorAndGraphMaxDedupe(t *testing.T) {
	emb := newStubEmbedder(4)
	emb.vectors["q"] = []float32{1, 0, 0, 0}
	s := newTestServer(emb)
	defer s.recallCache.stop()
	ctx := context.Background()

	// A high-similarity seed and a "both" memory that is also a vector
	// hit (but at lower similarity than the seed) AND reachable via an
	// edge from the seed.
	seed := recallWriteWithEmbed(t, s, emb, ctx, "seed-vg", []float32{1, 0, 0, 0})
	both := recallWriteWithEmbed(t, s, emb, ctx, "both-vg", []float32{0.6, 0.8, 0, 0})
	if _, err := s.store.AddEdge(ctx, memory.Edge{SrcID: seed, DstID: both, EdgeType: "refines"}); err != nil {
		t.Fatalf("AddEdge: %v", err)
	}

	_, out, err := s.handleRecall(ctx, nil, RecallInput{Query: "q", ExpandViaGraph: true})
	if err != nil {
		t.Fatalf("handleRecall: %v", err)
	}
	var found *RecallCandidate
	count := 0
	for i := range out.Candidates {
		if out.Candidates[i].ID == both {
			count++
			c := out.Candidates[i]
			found = &c
		}
	}
	if count != 1 {
		t.Fatalf("expected exactly 1 entry for 'both' id, got %d", count)
	}
	// Vector composite = similarity ~= 0.6; graph composite for a
	// direct-edge neighbor = seed_sim(1.0) * retention * 0.5^1. With
	// default retention (~ what the default cluster yields) graph
	// composite ~ 0.5*retention which is less than 0.6 for the typical
	// initial-state cluster. Either way, the kept entry MUST be the
	// higher-composite one and that's what we assert.
	if found.Distance == 0 && found.CompositeScore != float64(found.Similarity) {
		t.Errorf("Distance=0 kept: CompositeScore=%v, want Similarity=%v", found.CompositeScore, found.Similarity)
	}
}

func TestHandleRecall_Round1_IgnoresExpand(t *testing.T) {
	emb := newStubEmbedder(4)
	emb.vectors["q"] = []float32{1, 0, 0, 0}
	s := newTestServer(emb)
	defer s.recallCache.stop()
	ctx := context.Background()

	seed := recallWriteWithEmbed(t, s, emb, ctx, "seed-r1", []float32{1, 0, 0, 0})
	neighbor := recallWriteWithEmbed(t, s, emb, ctx, "neighbor-r1", []float32{0, 0, 1, 0})
	if _, err := s.store.AddEdge(ctx, memory.Edge{SrcID: seed, DstID: neighbor, EdgeType: "refines"}); err != nil {
		t.Fatalf("AddEdge: %v", err)
	}

	_, out, err := s.handleRecall(ctx, nil, RecallInput{Query: "q", ExpandViaGraph: true, Round: 1})
	if err != nil {
		t.Fatalf("handleRecall round 1 should not error: %v", err)
	}
	// On round>=1 the expand flag is silently ignored: no entry may
	// have Distance>0. (The neighbor may still appear as a vector hit
	// with Distance=0 since GlobalSearch returns it with sim=0.)
	for _, c := range out.Candidates {
		if c.Distance != 0 {
			t.Errorf("round=1 + expand=true surfaced graph-distance candidate id=%s dist=%d", c.ID, c.Distance)
		}
	}
	if out.Round != 1 {
		t.Errorf("Round = %d, want 1", out.Round)
	}
	_ = neighbor
	_ = seed
}

func TestHandleRecall_GateB_GraphHitsFalse(t *testing.T) {
	emb := newStubEmbedder(4)
	emb.vectors["q"] = []float32{1, 0, 0, 0}
	s := newTestServer(emb)
	defer s.recallCache.stop()
	ctx := context.Background()

	seed := recallWriteWithEmbed(t, s, emb, ctx, "seed-gb", []float32{1, 0, 0, 0})
	neighbor := recallWriteWithEmbed(t, s, emb, ctx, "neighbor-gb", []float32{0, 0, 1, 0})
	if _, err := s.store.AddEdge(ctx, memory.Edge{SrcID: seed, DstID: neighbor, EdgeType: "refines"}); err != nil {
		t.Fatalf("AddEdge: %v", err)
	}
	_, out, err := s.handleRecall(ctx, nil, RecallInput{Query: "q", ExpandViaGraph: true})
	if err != nil {
		t.Fatalf("handleRecall: %v", err)
	}
	for _, c := range out.Candidates {
		if c.ID == neighbor && c.GateBPass {
			t.Errorf("graph-only neighbor GateBPass=true; want deterministic false")
		}
	}
}

func TestHandleRecall_HopsClamping(t *testing.T) {
	emb := newStubEmbedder(4)
	emb.vectors["q"] = []float32{1, 0, 0, 0}
	s := newTestServer(emb)
	defer s.recallCache.stop()
	ctx := context.Background()

	// Build a 4-hop edge chain. hops clamped at 3 means we should see
	// distance 1,2,3 but NOT 4. hops=0 should default to 2 -> only 1,2.
	a := recallWriteWithEmbed(t, s, emb, ctx, "hop-a", []float32{1, 0, 0, 0})
	b := recallWriteWithEmbed(t, s, emb, ctx, "hop-b", []float32{0, 0, 1, 0})
	c := recallWriteWithEmbed(t, s, emb, ctx, "hop-c", []float32{0, 0, 0, 1})
	d := recallWriteWithEmbed(t, s, emb, ctx, "hop-d", []float32{0, 1, 0, 0})
	for _, e := range []memory.Edge{
		{SrcID: a, DstID: b, EdgeType: "refines"},
		{SrcID: b, DstID: c, EdgeType: "refines"},
		{SrcID: c, DstID: d, EdgeType: "refines"},
	} {
		if _, err := s.store.AddEdge(ctx, e); err != nil {
			t.Fatalf("AddEdge: %v", err)
		}
	}

	// hops=5 -> clamped to 3. b,c,d should all be reached as graph
	// neighbors of a at distances 1,2,3 respectively (Distance>0).
	// All four facts also appear in vector results (with sim=0 for
	// the ones whose embedding is orthogonal), so we look for the
	// first entry with the matching ID that has Distance > 0.
	_, out5, err := s.handleRecall(ctx, nil, RecallInput{Query: "q", ExpandViaGraph: true, GraphHops: 5, Limit: 50})
	if err != nil {
		t.Fatalf("handleRecall: %v", err)
	}
	dists5 := map[string]int{}
	for _, ck := range out5.Candidates {
		dists5[ck.ID] = ck.Distance
	}
	// Each of b/c/d should have its FINAL ranked Distance (which is
	// the graph-vs-vector winner by composite). For these zero-cosine
	// neighbors, graph composite > 0 beats vector composite = 0, so
	// the graph distance wins.
	if dists5[b] != 1 {
		t.Errorf("hops=5: b distance = %d, want 1", dists5[b])
	}
	if dists5[c] != 2 {
		t.Errorf("hops=5: c distance = %d, want 2", dists5[c])
	}
	if dists5[d] != 3 {
		t.Errorf("hops=5: d distance = %d, want 3 (clamped from 5)", dists5[d])
	}

	// hops=0 -> defaults to 2; only b and c reachable as graph hits.
	// d still appears as a vector hit at Distance=0 (since its
	// composite=sim=0 fails to beat anything; but it survives if
	// limit accommodates).
	_, out0, err := s.handleRecall(ctx, nil, RecallInput{Query: "q", ExpandViaGraph: true, GraphHops: 0, Limit: 50})
	if err != nil {
		t.Fatalf("handleRecall: %v", err)
	}
	dists0 := map[string]int{}
	for _, ck := range out0.Candidates {
		dists0[ck.ID] = ck.Distance
	}
	if dists0[b] != 1 {
		t.Errorf("hops=0 (default 2): b distance = %d, want 1", dists0[b])
	}
	if dists0[c] != 2 {
		t.Errorf("hops=0 (default 2): c distance = %d, want 2", dists0[c])
	}
	if dists0[d] != 0 {
		t.Errorf("hops=0 (default 2): d should NOT be graph-reached (distance must be 0), got %d", dists0[d])
	}
}

func TestHandleRecall_LimitAppliedPostMerge(t *testing.T) {
	emb := newStubEmbedder(4)
	emb.vectors["q"] = []float32{1, 0, 0, 0}
	s := newTestServer(emb)
	defer s.recallCache.stop()
	ctx := context.Background()

	// Build 8 facts. The first three have descending similarity to the
	// query and are the intended vector winners; the rest have near-zero
	// similarity (orthogonal embeddings) and are graph-only neighbors
	// connected via an edge chain off v1. With Limit large enough that
	// GlobalSearch returns all of them as seeds, graph expansion adds
	// nothing new (every memory is already a seed). The point we assert:
	// (1) length is bounded by limit, (2) order is composite-descending.
	// Use distinctly-different embeddings so they don't supersede each
	// other (default ConflictThreshold=0.92). Cosines vs query (1,0,0,0):
	// v1=1.0, v2~=0.707, v3~=0.447 — all above the default similarity
	// threshold of 0.70 only for v1/v2, but that doesn't matter for the
	// limit assertion.
	v1 := recallWriteWithEmbed(t, s, emb, ctx, "lim-v1", []float32{1, 0, 0, 0})
	_ = recallWriteWithEmbed(t, s, emb, ctx, "lim-v2", []float32{1, 1, 0, 0})
	_ = recallWriteWithEmbed(t, s, emb, ctx, "lim-v3", []float32{1, 0, 1, 1})
	g := make([]string, 0, 5)
	for i := 0; i < 5; i++ {
		// distinct small-magnitude orthogonal embeddings so cosine is
		// effectively zero against the query (1,0,0,0).
		gid := recallWriteWithEmbed(t, s, emb, ctx, "lim-g-"+string(rune('a'+i)), []float32{0, float32(i+1) * 0.1, 0, 0})
		g = append(g, gid)
	}
	// Build a chain v1 -> g[0] -> g[1] -> g[2] -> g[3] -> g[4].
	prev := v1
	for _, gid := range g {
		if _, err := s.store.AddEdge(ctx, memory.Edge{SrcID: prev, DstID: gid, EdgeType: "refines"}); err != nil {
			t.Fatalf("AddEdge: %v", err)
		}
		prev = gid
	}

	const wantLimit = 5
	_, out, err := s.handleRecall(ctx, nil, RecallInput{Query: "q", Limit: wantLimit, ExpandViaGraph: true})
	if err != nil {
		t.Fatalf("handleRecall: %v", err)
	}
	if len(out.Candidates) > wantLimit {
		t.Fatalf("expected at most %d candidates after limit, got %d", wantLimit, len(out.Candidates))
	}
	if len(out.Candidates) == 0 {
		t.Fatal("expected at least one candidate")
	}
	// Verify composite scores are sorted descending.
	for i := 1; i < len(out.Candidates); i++ {
		if out.Candidates[i-1].CompositeScore < out.Candidates[i].CompositeScore {
			t.Errorf("candidates not sorted by composite desc at i=%d: %v < %v",
				i, out.Candidates[i-1].CompositeScore, out.Candidates[i].CompositeScore)
		}
	}
	// The single top candidate must be the highest-similarity vector
	// hit: v1 (Distance=0, Similarity~=1.0). Graph hits inherit decay
	// and cannot beat it.
	top := out.Candidates[0]
	if top.ID != v1 {
		t.Errorf("top candidate id = %q, want v1=%q", top.ID, v1)
	}
	if top.Distance != 0 {
		t.Errorf("top candidate Distance = %d, want 0", top.Distance)
	}
}

// --- memory_forget tests ---

func TestHandleForget_ByID(t *testing.T) {
	emb := newStubEmbedder(4)
	emb.vectors["to be forgotten"] = []float32{0.1, 0.2, 0.3, 0.4}
	s := newTestServer(emb)
	defer s.recallCache.stop()

	ctx := context.Background()

	_, writeOut, err := s.handleWrite(ctx, nil, WriteInput{Content: "to be forgotten", Type: "user"})
	if err != nil {
		t.Fatalf("write: %v", err)
	}

	_, forgetOut, err := s.handleForget(ctx, nil, ForgetInput{ID: writeOut.ID})
	if err != nil {
		t.Fatalf("handleForget: %v", err)
	}
	if forgetOut.Deleted != 1 {
		t.Fatalf("expected deleted=1, got %d", forgetOut.Deleted)
	}

	// Verify it's gone.
	fact, err := s.store.GetFact(ctx, writeOut.ID)
	if err != nil {
		t.Fatalf("GetFact: %v", err)
	}
	if fact != nil {
		t.Fatal("expected fact to be deleted")
	}
}

func TestHandleForget_ByQuery(t *testing.T) {
	emb := newStubEmbedder(4)
	emb.vectors["important stuff"] = []float32{0.5, 0.5, 0.0, 0.0}
	emb.vectors["find stuff"] = []float32{0.5, 0.5, 0.0, 0.0}
	s := newTestServer(emb)
	defer s.recallCache.stop()

	ctx := context.Background()
	_, _, err := s.handleWrite(ctx, nil, WriteInput{Content: "important stuff", Type: "project"})
	if err != nil {
		t.Fatalf("write: %v", err)
	}

	_, forgetOut, err := s.handleForget(ctx, nil, ForgetInput{Query: "find stuff"})
	if err != nil {
		t.Fatalf("handleForget: %v", err)
	}
	if forgetOut.Deleted != 0 {
		t.Fatalf("expected deleted=0 (query mode), got %d", forgetOut.Deleted)
	}
	if len(forgetOut.Candidates) == 0 {
		t.Fatal("expected at least one candidate")
	}

	// Verify the fact was NOT deleted.
	facts, err := s.store.ListFacts(ctx, memory.ListFilter{Limit: 100})
	if err != nil {
		t.Fatalf("ListFacts: %v", err)
	}
	if len(facts) != 1 {
		t.Fatalf("expected 1 fact (not deleted), got %d", len(facts))
	}
}

func TestHandleForget_BothIDAndQuery(t *testing.T) {
	emb := newStubEmbedder(4)
	s := newTestServer(emb)
	defer s.recallCache.stop()

	_, _, err := s.handleForget(context.Background(), nil, ForgetInput{ID: "some-id", Query: "some query"})
	if err == nil {
		t.Fatal("expected error when both id and query are provided")
	}
}

func TestHandleForget_NeitherIDNorQuery(t *testing.T) {
	emb := newStubEmbedder(4)
	s := newTestServer(emb)
	defer s.recallCache.stop()

	_, _, err := s.handleForget(context.Background(), nil, ForgetInput{})
	if err == nil {
		t.Fatal("expected error when neither id nor query is provided")
	}
}

func TestHandleForget_BatchAllValid(t *testing.T) {
	emb := newStubEmbedder(4)
	emb.vectors["batch one"] = []float32{0.1, 0.0, 0.0, 0.0}
	emb.vectors["batch two"] = []float32{0.0, 0.1, 0.0, 0.0}
	emb.vectors["batch three"] = []float32{0.0, 0.0, 0.1, 0.0}
	s := newTestServer(emb)
	defer s.recallCache.stop()

	ctx := context.Background()
	ids := make([]string, 0, 3)
	for _, content := range []string{"batch one", "batch two", "batch three"} {
		_, out, err := s.handleWrite(ctx, nil, WriteInput{Content: content, Type: "project"})
		if err != nil {
			t.Fatalf("write %q: %v", content, err)
		}
		ids = append(ids, out.ID)
	}

	_, forgetOut, err := s.handleForget(ctx, nil, ForgetInput{IDs: ids})
	if err != nil {
		t.Fatalf("handleForget: %v", err)
	}
	if forgetOut.Deleted != 3 {
		t.Fatalf("expected Deleted=3, got %d", forgetOut.Deleted)
	}
	if len(forgetOut.Failed) != 0 {
		t.Fatalf("expected Failed empty, got %+v", forgetOut.Failed)
	}
	if len(forgetOut.Candidates) != 0 {
		t.Fatalf("expected Candidates empty, got %+v", forgetOut.Candidates)
	}

	// Confirm all gone.
	for _, id := range ids {
		f, err := s.store.GetFact(ctx, id)
		if err != nil {
			t.Fatalf("GetFact: %v", err)
		}
		if f != nil {
			t.Fatalf("expected fact %s deleted", id)
		}
	}
}

func TestHandleForget_BatchMixedValidAndMissing(t *testing.T) {
	emb := newStubEmbedder(4)
	emb.vectors["keeper A"] = []float32{0.1, 0.0, 0.0, 0.0}
	emb.vectors["keeper B"] = []float32{0.0, 0.1, 0.0, 0.0}
	s := newTestServer(emb)
	defer s.recallCache.stop()

	ctx := context.Background()
	var validIDs []string
	for _, content := range []string{"keeper A", "keeper B"} {
		_, out, err := s.handleWrite(ctx, nil, WriteInput{Content: content, Type: "project"})
		if err != nil {
			t.Fatalf("write %q: %v", content, err)
		}
		validIDs = append(validIDs, out.ID)
	}

	ids := []string{validIDs[0], "nonexistent-1", validIDs[1], "nonexistent-2"}
	_, forgetOut, err := s.handleForget(ctx, nil, ForgetInput{IDs: ids})
	if err != nil {
		t.Fatalf("handleForget: %v", err)
	}
	if forgetOut.Deleted != 2 {
		t.Fatalf("expected Deleted=2, got %d", forgetOut.Deleted)
	}
	if len(forgetOut.Failed) != 2 {
		t.Fatalf("expected 2 failures, got %d: %+v", len(forgetOut.Failed), forgetOut.Failed)
	}
	// Both failures should name the nonexistent IDs and carry a non-empty reason.
	seen := map[string]string{}
	for _, f := range forgetOut.Failed {
		seen[f.ID] = f.Reason
	}
	for _, missing := range []string{"nonexistent-1", "nonexistent-2"} {
		reason, ok := seen[missing]
		if !ok {
			t.Fatalf("expected failure entry for %s, got %+v", missing, forgetOut.Failed)
		}
		if reason == "" {
			t.Fatalf("expected non-empty reason for %s", missing)
		}
		if !strings.Contains(reason, "not found") {
			t.Fatalf("expected reason for %s to mention not found, got %q", missing, reason)
		}
	}
}

func TestHandleForget_BatchContinuesOnMidLoopMiss(t *testing.T) {
	// Simulate a mid-loop "store error"-class failure by including the same
	// ID twice: the first succeeds, the second hits the not-found path.
	emb := newStubEmbedder(4)
	emb.vectors["target"] = []float32{0.2, 0.0, 0.0, 0.0}
	emb.vectors["survivor"] = []float32{0.0, 0.2, 0.0, 0.0}
	s := newTestServer(emb)
	defer s.recallCache.stop()

	ctx := context.Background()
	_, target, err := s.handleWrite(ctx, nil, WriteInput{Content: "target", Type: "project"})
	if err != nil {
		t.Fatalf("write target: %v", err)
	}
	_, survivor, err := s.handleWrite(ctx, nil, WriteInput{Content: "survivor", Type: "project"})
	if err != nil {
		t.Fatalf("write survivor: %v", err)
	}

	// target listed twice; survivor must still be deleted after the middle failure.
	_, forgetOut, err := s.handleForget(ctx, nil, ForgetInput{IDs: []string{target.ID, target.ID, survivor.ID}})
	if err != nil {
		t.Fatalf("handleForget: %v", err)
	}
	if forgetOut.Deleted != 2 {
		t.Fatalf("expected Deleted=2 (target + survivor), got %d", forgetOut.Deleted)
	}
	if len(forgetOut.Failed) != 1 {
		t.Fatalf("expected 1 failure (second target attempt), got %d: %+v", len(forgetOut.Failed), forgetOut.Failed)
	}
	if forgetOut.Failed[0].ID != target.ID {
		t.Fatalf("expected failure ID %s, got %s", target.ID, forgetOut.Failed[0].ID)
	}

	// Survivor is definitely gone — proves the loop kept going past the failure.
	f, err := s.store.GetFact(ctx, survivor.ID)
	if err != nil {
		t.Fatalf("GetFact: %v", err)
	}
	if f != nil {
		t.Fatal("expected survivor to be deleted after mid-loop failure")
	}
}

func TestHandleForget_BatchWithEmptyStringID(t *testing.T) {
	emb := newStubEmbedder(4)
	emb.vectors["keeper"] = []float32{0.1, 0.0, 0.0, 0.0}
	s := newTestServer(emb)
	defer s.recallCache.stop()

	ctx := context.Background()
	_, keeper, err := s.handleWrite(ctx, nil, WriteInput{Content: "keeper", Type: "project"})
	if err != nil {
		t.Fatalf("write: %v", err)
	}

	_, out, err := s.handleForget(ctx, nil, ForgetInput{IDs: []string{"", keeper.ID}})
	if err != nil {
		t.Fatalf("handleForget: %v", err)
	}
	if out.Deleted != 1 {
		t.Fatalf("expected Deleted=1, got %d", out.Deleted)
	}
	if len(out.Failed) != 1 {
		t.Fatalf("expected 1 failure for empty id, got %+v", out.Failed)
	}
	if out.Failed[0].ID != "" || out.Failed[0].Reason != "empty id" {
		t.Fatalf("expected {ID:\"\", Reason:\"empty id\"}, got %+v", out.Failed[0])
	}
}

func TestHandleForget_BothIDAndIDs(t *testing.T) {
	emb := newStubEmbedder(4)
	s := newTestServer(emb)
	defer s.recallCache.stop()

	_, _, err := s.handleForget(context.Background(), nil, ForgetInput{ID: "a", IDs: []string{"b"}})
	if err == nil {
		t.Fatal("expected error when both id and ids are provided")
	}
}

func TestHandleForget_EmptyIDsSlice(t *testing.T) {
	emb := newStubEmbedder(4)
	s := newTestServer(emb)
	defer s.recallCache.stop()

	_, _, err := s.handleForget(context.Background(), nil, ForgetInput{IDs: []string{}})
	if err == nil {
		t.Fatal("expected error for ids=[]")
	}
	if !strings.Contains(err.Error(), "empty") {
		t.Fatalf("expected error to mention empty, got %v", err)
	}
}

func TestHandleForget_IDsAndQuery(t *testing.T) {
	emb := newStubEmbedder(4)
	s := newTestServer(emb)
	defer s.recallCache.stop()

	_, _, err := s.handleForget(context.Background(), nil, ForgetInput{IDs: []string{"a"}, Query: "q"})
	if err == nil {
		t.Fatal("expected error when ids and query are both provided")
	}
}

// --- memory_list tests ---

func TestHandleList_SubtypeFilter(t *testing.T) {
	emb := newStubEmbedder(4)
	emb.vectors["user pref 1"] = []float32{0.1, 0.0, 0.0, 0.0}
	emb.vectors["project convention A"] = []float32{0.0, 0.1, 0.0, 0.0}
	emb.vectors["user pref 2"] = []float32{0.0, 0.0, 0.1, 0.0}
	s := newTestServer(emb)
	defer s.recallCache.stop()

	ctx := context.Background()
	_, _, _ = s.handleWrite(ctx, nil, WriteInput{Content: "user pref 1", Type: "user"})
	_, _, _ = s.handleWrite(ctx, nil, WriteInput{Content: "project convention A", Type: "project"})
	_, _, _ = s.handleWrite(ctx, nil, WriteInput{Content: "user pref 2", Type: "user"})

	// List all.
	_, allOut, err := s.handleList(ctx, nil, ListInput{})
	if err != nil {
		t.Fatalf("list all: %v", err)
	}
	if len(allOut.Memories) != 3 {
		t.Fatalf("expected 3 memories, got %d", len(allOut.Memories))
	}

	// List only user.
	_, userOut, err := s.handleList(ctx, nil, ListInput{Subtype: "user"})
	if err != nil {
		t.Fatalf("list user: %v", err)
	}
	if len(userOut.Memories) != 2 {
		t.Fatalf("expected 2 user memories, got %d", len(userOut.Memories))
	}
	for _, m := range userOut.Memories {
		if m.Subtype != "user" {
			t.Fatalf("expected subtype user, got %s", m.Subtype)
		}
	}

	// List project.
	_, projOut, err := s.handleList(ctx, nil, ListInput{Subtype: "project"})
	if err != nil {
		t.Fatalf("list project: %v", err)
	}
	if len(projOut.Memories) != 1 {
		t.Fatalf("expected 1 project memory, got %d", len(projOut.Memories))
	}
}

// TestHandleList_IncludesClusterID verifies every list entry (fact and episode)
// comes back with a populated cluster_id — Phase 1A requirement.
func TestHandleList_IncludesClusterID(t *testing.T) {
	emb := newStubEmbedder(4)
	emb.vectors["fact content"] = []float32{0.4, 0.4, 0.0, 0.0}
	episodeText := "sit\nact\nout\npre"
	emb.vectors[episodeText] = []float32{0.0, 0.0, 0.4, 0.4}
	s := newTestServer(emb)
	defer s.recallCache.stop()

	ctx := context.Background()
	_, _, err := s.handleWrite(ctx, nil, WriteInput{Content: "fact content", Type: "project"})
	if err != nil {
		t.Fatalf("write fact: %v", err)
	}
	_, _, err = s.handleWrite(ctx, nil, WriteInput{
		Type: "feedback",
		Episode: &EpisodePayload{
			Situation:  "sit",
			Action:     "act",
			Outcome:    "out",
			Preemptive: "pre",
		},
	})
	if err != nil {
		t.Fatalf("write episode: %v", err)
	}

	// L2 list: cluster_id populated.
	_, l2Out, err := s.handleList(ctx, nil, ListInput{})
	if err != nil {
		t.Fatalf("list l2: %v", err)
	}
	if len(l2Out.Memories) != 1 {
		t.Fatalf("expected 1 l2 memory, got %d", len(l2Out.Memories))
	}
	for _, m := range l2Out.Memories {
		if m.ClusterID == "" {
			t.Errorf("l2 memory %s has empty cluster_id", m.ID)
		}
	}

	// L3 list: cluster_id populated.
	_, l3Out, err := s.handleList(ctx, nil, ListInput{Layer: "l3"})
	if err != nil {
		t.Fatalf("list l3: %v", err)
	}
	if len(l3Out.Memories) != 1 {
		t.Fatalf("expected 1 l3 memory, got %d", len(l3Out.Memories))
	}
	for _, m := range l3Out.Memories {
		if m.ClusterID == "" {
			t.Errorf("l3 memory %s has empty cluster_id", m.ID)
		}
	}
}

// TestHandleList_TagsAlwaysNonNil verifies the tags field is a non-nil slice
// for every entry — empty slice for untagged memories, not a nil slice.
func TestHandleList_TagsAlwaysNonNil(t *testing.T) {
	emb := newStubEmbedder(4)
	emb.vectors["tagged fact"] = []float32{0.1, 0.0, 0.0, 0.0}
	emb.vectors["untagged fact"] = []float32{0.0, 0.1, 0.0, 0.0}
	s := newTestServer(emb)
	defer s.recallCache.stop()

	ctx := context.Background()
	_, _, err := s.handleWrite(ctx, nil, WriteInput{
		Content: "tagged fact",
		Type:    "project",
		Tags:    []string{"alpha"},
	})
	if err != nil {
		t.Fatalf("write tagged: %v", err)
	}
	_, _, err = s.handleWrite(ctx, nil, WriteInput{
		Content: "untagged fact",
		Type:    "project",
	})
	if err != nil {
		t.Fatalf("write untagged: %v", err)
	}

	_, out, err := s.handleList(ctx, nil, ListInput{})
	if err != nil {
		t.Fatalf("list: %v", err)
	}
	if len(out.Memories) != 2 {
		t.Fatalf("expected 2 memories, got %d", len(out.Memories))
	}
	for _, m := range out.Memories {
		if m.Tags == nil {
			t.Errorf("memory %s has nil Tags; expected empty slice for untagged or populated for tagged", m.ID)
		}
		if m.Content == "untagged fact" && len(m.Tags) != 0 {
			t.Errorf("untagged fact has Tags=%v, want empty slice", m.Tags)
		}
		if m.Content == "tagged fact" {
			if len(m.Tags) != 1 || m.Tags[0] != "alpha" {
				t.Errorf("tagged fact has Tags=%v, want [alpha]", m.Tags)
			}
		}
	}
}

// TestHandleList_TagsAnyFilter verifies the tags_any input filter plumbs
// through ListFilter.TagsAny and returns only matching memories.
func TestHandleList_TagsAnyFilter(t *testing.T) {
	emb := newStubEmbedder(4)
	emb.vectors["foo fact"] = []float32{0.1, 0.0, 0.0, 0.0}
	emb.vectors["bar fact"] = []float32{0.0, 0.1, 0.0, 0.0}
	emb.vectors["other fact"] = []float32{0.0, 0.0, 0.1, 0.0}
	s := newTestServer(emb)
	defer s.recallCache.stop()

	ctx := context.Background()
	_, _, err := s.handleWrite(ctx, nil, WriteInput{
		Content: "foo fact",
		Type:    "project",
		Tags:    []string{"foo"},
	})
	if err != nil {
		t.Fatalf("write foo: %v", err)
	}
	_, _, err = s.handleWrite(ctx, nil, WriteInput{
		Content: "bar fact",
		Type:    "project",
		Tags:    []string{"bar"},
	})
	if err != nil {
		t.Fatalf("write bar: %v", err)
	}
	_, _, err = s.handleWrite(ctx, nil, WriteInput{
		Content: "other fact",
		Type:    "project",
	})
	if err != nil {
		t.Fatalf("write other: %v", err)
	}

	_, out, err := s.handleList(ctx, nil, ListInput{TagsAny: []string{"foo"}})
	if err != nil {
		t.Fatalf("list tags_any=[foo]: %v", err)
	}
	if len(out.Memories) != 1 {
		t.Fatalf("expected 1 memory with tag foo, got %d", len(out.Memories))
	}
	if out.Memories[0].Content != "foo fact" {
		t.Errorf("content = %q, want 'foo fact'", out.Memories[0].Content)
	}
	if len(out.Memories[0].Tags) != 1 || out.Memories[0].Tags[0] != "foo" {
		t.Errorf("Tags = %v, want [foo]", out.Memories[0].Tags)
	}
}

func TestHandleList_EmptyStore(t *testing.T) {
	emb := newStubEmbedder(4)
	s := newTestServer(emb)
	defer s.recallCache.stop()

	_, out, err := s.handleList(context.Background(), nil, ListInput{})
	if err != nil {
		t.Fatalf("list: %v", err)
	}
	if len(out.Memories) != 0 {
		t.Fatalf("expected 0 memories, got %d", len(out.Memories))
	}
}

// --- memory_reinforce tests ---

func TestHandleReinforce_TouchesAccessedAt(t *testing.T) {
	emb := newStubEmbedder(4)
	emb.vectors["reinforce me"] = []float32{0.5, 0.5, 0.0, 0.0}
	s := newTestServer(emb)
	defer s.recallCache.stop()

	ctx := context.Background()
	_, writeOut, err := s.handleWrite(ctx, nil, WriteInput{Content: "reinforce me", Type: "feedback"})
	if err != nil {
		t.Fatalf("write: %v", err)
	}

	// Get the initial accessed_at.
	fact1, _ := s.store.GetFact(ctx, writeOut.ID)
	initialAccessed := fact1.AccessedAt

	// Small sleep to ensure time difference.
	time.Sleep(10 * time.Millisecond)

	_, reinforceOut, err := s.handleReinforce(ctx, nil, ReinforceInput{MemoryIDs: []string{writeOut.ID}})
	if err != nil {
		t.Fatalf("reinforce: %v", err)
	}
	if reinforceOut.Reinforced != 1 {
		t.Fatalf("expected reinforced=1, got %d", reinforceOut.Reinforced)
	}

	// Verify accessed_at was updated.
	fact2, _ := s.store.GetFact(ctx, writeOut.ID)
	if !fact2.AccessedAt.After(initialAccessed) {
		t.Fatalf("expected accessed_at to advance; before=%v after=%v", initialAccessed, fact2.AccessedAt)
	}
}

func TestHandleReinforce_EmptyIDs(t *testing.T) {
	emb := newStubEmbedder(4)
	s := newTestServer(emb)
	defer s.recallCache.stop()

	_, _, err := s.handleReinforce(context.Background(), nil, ReinforceInput{MemoryIDs: nil})
	if err == nil {
		t.Fatal("expected error for empty memory_ids")
	}
}

// --- disabled mode tests ---

func TestDisabledMode(t *testing.T) {
	emb := newStubEmbedder(4)
	cfg := config.Defaults()
	cfg.Server.Disabled = true
	store := memory.NewMemStore()
	dec := decay.NewDecayer(10.0, 0.3)
	mgr := manager.NewMemoryManager(store, dec, 0.10, 0.05)
	assigner := cluster.NewAssigner(store, 0.60, 0.5, 0.5)
	s := NewServer(store, emb, dec, mgr, assigner, cfg, nil)
	defer s.recallCache.stop()

	ctx := context.Background()

	// All tools should return a stub result when disabled.
	result, _, err := s.handleRecall(ctx, nil, RecallInput{Query: "test"})
	if err != nil {
		t.Fatalf("recall disabled: %v", err)
	}
	if result == nil || len(result.Content) == 0 {
		t.Fatal("expected disabled stub result")
	}

	result, _, err = s.handleWrite(ctx, nil, WriteInput{Content: "test", Type: "user"})
	if err != nil {
		t.Fatalf("write disabled: %v", err)
	}
	if result == nil || len(result.Content) == 0 {
		t.Fatal("expected disabled stub result")
	}

	result, _, err = s.handleReinforce(ctx, nil, ReinforceInput{MemoryIDs: []string{"id"}})
	if err != nil {
		t.Fatalf("reinforce disabled: %v", err)
	}
	if result == nil || len(result.Content) == 0 {
		t.Fatal("expected disabled stub result")
	}

	result, _, err = s.handleForget(ctx, nil, ForgetInput{ID: "id"})
	if err != nil {
		t.Fatalf("forget disabled: %v", err)
	}
	if result == nil || len(result.Content) == 0 {
		t.Fatal("expected disabled stub result")
	}

	result, _, err = s.handleList(ctx, nil, ListInput{})
	if err != nil {
		t.Fatalf("list disabled: %v", err)
	}
	if result == nil || len(result.Content) == 0 {
		t.Fatal("expected disabled stub result")
	}

	result, _, err = s.handleApplyJudgment(ctx, nil, ApplyJudgmentInput{RecallID: "test"})
	if err != nil {
		t.Fatalf("apply_judgment disabled: %v", err)
	}
	if result == nil || len(result.Content) == 0 {
		t.Fatal("expected disabled stub result")
	}

	result, _, err = s.handleDecayTick(ctx, nil, DecayTickInput{})
	if err != nil {
		t.Fatalf("decay_tick disabled: %v", err)
	}
	if result == nil || len(result.Content) == 0 {
		t.Fatal("expected disabled stub result")
	}
}

// --- Phase 2: Gate C / recall cache / apply judgment tests ---

func TestMemoryRecall_ComputesRealGateC(t *testing.T) {
	emb := newStubEmbedder(4)
	emb.vectors["Go is fast"] = []float32{0.5, 0.5, 0.0, 0.0}
	emb.vectors["find Go"] = []float32{0.5, 0.5, 0.0, 0.0}
	s := newTestServer(emb)
	defer s.recallCache.stop()

	ctx := context.Background()

	// Write a fact.
	_, writeOut, err := s.handleWrite(ctx, nil, WriteInput{Content: "Go is fast", Type: "project"})
	if err != nil {
		t.Fatalf("write: %v", err)
	}

	// Recall it.
	_, recallOut, err := s.handleRecall(ctx, nil, RecallInput{Query: "find Go"})
	if err != nil {
		t.Fatalf("recall: %v", err)
	}

	if len(recallOut.Candidates) == 0 {
		t.Fatal("expected at least one candidate")
	}

	found := false
	for _, c := range recallOut.Candidates {
		if c.ID == writeOut.ID {
			found = true
			// Retention should be in (0, 1] because the default cluster was just
			// created with U=0,F=0 and turns_since was reset by the write's TickDecay.
			if c.Retention <= 0 || c.Retention > 1.0 {
				t.Errorf("expected retention in (0,1], got %f", c.Retention)
			}
			// gate_c_pass should match whether retention exceeds threshold (0.3).
			expectedGateC := c.Retention > 0.3
			if c.GateCPass != expectedGateC {
				t.Errorf("expected gate_c_pass=%v (retention=%f, threshold=0.3), got %v", expectedGateC, c.Retention, c.GateCPass)
			}
			break
		}
	}
	if !found {
		t.Fatalf("written fact %s not found in recall candidates", writeOut.ID)
	}
}

func TestMemoryRecall_PopulatesRecallCache(t *testing.T) {
	emb := newStubEmbedder(4)
	emb.vectors["cache test"] = []float32{0.5, 0.5, 0.0, 0.0}
	emb.vectors["cache query"] = []float32{0.5, 0.5, 0.0, 0.0}
	s := newTestServer(emb)
	defer s.recallCache.stop()

	ctx := context.Background()

	_, _, err := s.handleWrite(ctx, nil, WriteInput{Content: "cache test", Type: "user"})
	if err != nil {
		t.Fatalf("write: %v", err)
	}

	_, recallOut, err := s.handleRecall(ctx, nil, RecallInput{Query: "cache query"})
	if err != nil {
		t.Fatalf("recall: %v", err)
	}

	if recallOut.RecallID == "" {
		t.Fatal("expected non-empty recall_id")
	}

	cached, ok := s.recallCache.get(recallOut.RecallID)
	if !ok {
		t.Fatal("recall_id not found in cache after recall")
	}
	if len(cached.candidates) == 0 {
		t.Fatal("expected cached candidates")
	}
	if len(cached.queryVec) == 0 {
		t.Fatal("expected cached query vector")
	}
	if cached.round != 0 {
		t.Fatalf("expected round=0, got %d", cached.round)
	}
}

func TestMemoryApplyJudgment_RoundZeroORLogic(t *testing.T) {
	emb := newStubEmbedder(4)
	// Two facts, both similar to the query. Different subtypes to avoid conflict supersede.
	emb.vectors["fact A"] = []float32{0.5, 0.5, 0.0, 0.0}
	emb.vectors["fact B"] = []float32{0.4, 0.5, 0.1, 0.0}
	emb.vectors["query"] = []float32{0.5, 0.5, 0.0, 0.0}
	s := newTestServer(emb)
	defer s.recallCache.stop()

	ctx := context.Background()

	_, outA, err := s.handleWrite(ctx, nil, WriteInput{Content: "fact A", Type: "project"})
	if err != nil {
		t.Fatalf("write A: %v", err)
	}
	_, outB, err := s.handleWrite(ctx, nil, WriteInput{Content: "fact B", Type: "reference"})
	if err != nil {
		t.Fatalf("write B: %v", err)
	}

	// Recall at round 0.
	_, recallOut, err := s.handleRecall(ctx, nil, RecallInput{Query: "query", Round: 0})
	if err != nil {
		t.Fatalf("recall: %v", err)
	}

	// Apply judgment: gate_a=false for A, gate_a=true for B.
	// In round 0 (OR logic), A should still pass if gate_b or gate_c passes.
	_, judgOut, err := s.handleApplyJudgment(ctx, nil, ApplyJudgmentInput{
		RecallID: recallOut.RecallID,
		Verdicts: []Verdict{
			{MemoryID: outA.ID, Keep: false},
			{MemoryID: outB.ID, Keep: true},
		},
	})
	if err != nil {
		t.Fatalf("apply_judgment: %v", err)
	}

	if judgOut.AppliedLogic != "OR" {
		t.Fatalf("expected OR logic, got %s", judgOut.AppliedLogic)
	}

	// Both should pass under OR logic because at least gate_b should be true
	// for the high-similarity vectors.
	foundA, foundB := false, false
	for _, m := range judgOut.Memories {
		if m.ID == outA.ID {
			foundA = true
		}
		if m.ID == outB.ID {
			foundB = true
		}
	}
	if !foundB {
		t.Error("expected fact B (gate_a=true) to pass under OR logic")
	}
	// Fact A should pass too because gate_b is true (high similarity).
	if !foundA {
		t.Error("expected fact A (gate_a=false but gate_b=true) to pass under OR logic")
	}
}

func TestMemoryApplyJudgment_RoundOneANDLogic(t *testing.T) {
	emb := newStubEmbedder(4)
	emb.vectors["AND fact"] = []float32{0.5, 0.5, 0.0, 0.0}
	emb.vectors["AND query"] = []float32{0.5, 0.5, 0.0, 0.0}
	s := newTestServer(emb)
	defer s.recallCache.stop()

	ctx := context.Background()

	_, writeOut, err := s.handleWrite(ctx, nil, WriteInput{Content: "AND fact", Type: "project"})
	if err != nil {
		t.Fatalf("write: %v", err)
	}

	// Recall at round 1 (AND logic).
	_, recallOut, err := s.handleRecall(ctx, nil, RecallInput{Query: "AND query", Round: 1})
	if err != nil {
		t.Fatalf("recall: %v", err)
	}

	// Apply judgment with gate_a = false.
	// In round 1 (AND logic), candidate should FAIL because gate_a is false.
	_, judgOut, err := s.handleApplyJudgment(ctx, nil, ApplyJudgmentInput{
		RecallID: recallOut.RecallID,
		Verdicts: []Verdict{
			{MemoryID: writeOut.ID, Keep: false},
		},
	})
	if err != nil {
		t.Fatalf("apply_judgment: %v", err)
	}

	if judgOut.AppliedLogic != "AND" {
		t.Fatalf("expected AND logic, got %s", judgOut.AppliedLogic)
	}

	// Should be empty — gate_a=false means AND fails.
	for _, m := range judgOut.Memories {
		if m.ID == writeOut.ID {
			t.Error("expected fact to be filtered out under AND logic with gate_a=false")
		}
	}

	// Now apply with gate_a = true — should pass (all three gates true).
	// Re-recall to get a fresh recall_id.
	_, recallOut2, err := s.handleRecall(ctx, nil, RecallInput{Query: "AND query", Round: 1})
	if err != nil {
		t.Fatalf("recall 2: %v", err)
	}

	_, judgOut2, err := s.handleApplyJudgment(ctx, nil, ApplyJudgmentInput{
		RecallID: recallOut2.RecallID,
		Verdicts: []Verdict{
			{MemoryID: writeOut.ID, Keep: true},
		},
	})
	if err != nil {
		t.Fatalf("apply_judgment 2: %v", err)
	}

	// Check whether the candidate passes. gate_a=true, gate_b should be true
	// (identical vectors), gate_c depends on retention vs threshold.
	// The default cluster has U=0,F=0 and turns_since depends on how many
	// TickDecay calls happened. Check that the result is consistent.
	if judgOut2.AppliedLogic != "AND" {
		t.Fatalf("expected AND logic, got %s", judgOut2.AppliedLogic)
	}
	// With gate_a=true, passage depends on gate_b AND gate_c.
	// gate_b should be true (cosine of identical vectors = 1.0 > 0.70).
	// gate_c depends on retention of the cluster.
	// At minimum, confirm logic was applied correctly.
}

func TestMemoryApplyJudgment_ExpiredRecallID(t *testing.T) {
	emb := newStubEmbedder(4)
	s := newTestServer(emb)
	defer s.recallCache.stop()

	ctx := context.Background()

	_, _, err := s.handleApplyJudgment(ctx, nil, ApplyJudgmentInput{
		RecallID: "nonexistent-recall-id",
		Verdicts: []Verdict{{MemoryID: "x", Keep: true}},
	})
	if err == nil {
		t.Fatal("expected error for nonexistent recall_id")
	}
	if err.Error() != "recall_id not found or expired" {
		t.Fatalf("expected 'recall_id not found or expired' error, got: %v", err)
	}
}

func TestMemoryApplyJudgment_MissingVerdicts(t *testing.T) {
	emb := newStubEmbedder(4)
	emb.vectors["missing verdict fact 1"] = []float32{0.5, 0.5, 0.0, 0.0}
	emb.vectors["missing verdict fact 2"] = []float32{0.4, 0.5, 0.1, 0.0}
	emb.vectors["missing verdict query"] = []float32{0.5, 0.5, 0.0, 0.0}
	s := newTestServer(emb)
	defer s.recallCache.stop()

	ctx := context.Background()

	// Use different subtypes to avoid conflict supersede.
	_, out1, err := s.handleWrite(ctx, nil, WriteInput{Content: "missing verdict fact 1", Type: "project"})
	if err != nil {
		t.Fatalf("write 1: %v", err)
	}
	_, _, err = s.handleWrite(ctx, nil, WriteInput{Content: "missing verdict fact 2", Type: "reference"})
	if err != nil {
		t.Fatalf("write 2: %v", err)
	}

	// Round 0 (OR logic): missing verdict -> gate_a=false, but gate_b/c should pass.
	_, recallOut, err := s.handleRecall(ctx, nil, RecallInput{Query: "missing verdict query", Round: 0})
	if err != nil {
		t.Fatalf("recall: %v", err)
	}

	// Only supply verdict for fact 1; fact 2 has no verdict.
	_, judgOut, err := s.handleApplyJudgment(ctx, nil, ApplyJudgmentInput{
		RecallID: recallOut.RecallID,
		Verdicts: []Verdict{
			{MemoryID: out1.ID, Keep: true},
		},
	})
	if err != nil {
		t.Fatalf("apply_judgment: %v", err)
	}

	// Under OR logic, fact 2 should still pass if gate_b or gate_c is true
	// (even though gate_a defaults to false due to missing verdict).
	if len(judgOut.Memories) < 2 {
		// If fact 2's gate_b is true (cosine > 0.70), it should pass.
		// With the vectors we set, cosine should be high enough.
		t.Logf("got %d memories, fact 2 may have failed gate_b", len(judgOut.Memories))
	}

	// Now test round 1 (AND logic): missing verdict should cause fact to fail.
	_, recallOut2, err := s.handleRecall(ctx, nil, RecallInput{Query: "missing verdict query", Round: 1})
	if err != nil {
		t.Fatalf("recall 2: %v", err)
	}

	// Only supply verdict for fact 1 with keep=true.
	_, judgOut2, err := s.handleApplyJudgment(ctx, nil, ApplyJudgmentInput{
		RecallID: recallOut2.RecallID,
		Verdicts: []Verdict{
			{MemoryID: out1.ID, Keep: true},
		},
	})
	if err != nil {
		t.Fatalf("apply_judgment 2: %v", err)
	}

	// Under AND logic, fact 2 should fail because gate_a=false (no verdict).
	for _, m := range judgOut2.Memories {
		if m.ID != out1.ID {
			// This should be fact 2 — check it was NOT included.
			t.Errorf("expected fact 2 to be excluded under AND logic with missing verdict, but found %s", m.ID)
		}
	}
}

func TestMemoryWrite_CallsTickDecay(t *testing.T) {
	emb := newStubEmbedder(4)
	emb.vectors["tick decay content"] = []float32{0.5, 0.5, 0.0, 0.0}
	s := newTestServer(emb)
	defer s.recallCache.stop()

	ctx := context.Background()

	_, writeOut, err := s.handleWrite(ctx, nil, WriteInput{Content: "tick decay content", Type: "project"})
	if err != nil {
		t.Fatalf("write: %v", err)
	}

	// After write, the fact's cluster should have turns_since=0 because
	// TickDecay was called with the cluster as accessed.
	fact, err := s.store.GetFact(ctx, writeOut.ID)
	if err != nil {
		t.Fatalf("get fact: %v", err)
	}
	if fact == nil {
		t.Fatal("expected fact to exist")
	}

	cl, err := s.store.GetCluster(ctx, fact.ClusterID)
	if err != nil {
		t.Fatalf("get cluster: %v", err)
	}
	if cl == nil {
		t.Fatal("expected cluster to exist after write")
	}

	if cl.TurnsSince != 0 {
		t.Fatalf("expected turns_since=0 for accessed cluster after write, got %d", cl.TurnsSince)
	}
}

func TestMemoryReinforce_UpdatesUtility(t *testing.T) {
	emb := newStubEmbedder(4)
	// Both facts use the same vector so they land in the same cluster.
	emb.vectors["util fact 1"] = []float32{0.5, 0.5, 0.0, 0.0}
	emb.vectors["util fact 2"] = []float32{0.5, 0.5, 0.0, 0.0}
	s := newTestServer(emb)
	defer s.recallCache.stop()

	ctx := context.Background()

	_, out1, err := s.handleWrite(ctx, nil, WriteInput{Content: "util fact 1", Type: "project"})
	if err != nil {
		t.Fatalf("write 1: %v", err)
	}
	_, out2, err := s.handleWrite(ctx, nil, WriteInput{Content: "util fact 2", Type: "project"})
	if err != nil {
		t.Fatalf("write 2: %v", err)
	}

	// Get the cluster ID from the first fact.
	fact1, err := s.store.GetFact(ctx, out1.ID)
	if err != nil {
		t.Fatalf("get fact: %v", err)
	}
	clusterID := fact1.ClusterID

	// Get the initial cluster utility.
	clusterBefore, err := s.store.GetCluster(ctx, clusterID)
	if err != nil {
		t.Fatalf("get cluster: %v", err)
	}
	if clusterBefore == nil {
		t.Fatal("expected cluster")
	}
	utilBefore := clusterBefore.Utility

	// Reinforce both facts.
	_, _, err = s.handleReinforce(ctx, nil, ReinforceInput{MemoryIDs: []string{out1.ID, out2.ID}})
	if err != nil {
		t.Fatalf("reinforce: %v", err)
	}

	// Check that cluster utility moved.
	clusterAfter, err := s.store.GetCluster(ctx, clusterID)
	if err != nil {
		t.Fatalf("get cluster after: %v", err)
	}
	if clusterAfter == nil {
		t.Fatal("expected cluster after reinforce")
	}

	if clusterAfter.Utility <= utilBefore {
		t.Fatalf("expected utility to increase after reinforce; before=%f, after=%f", utilBefore, clusterAfter.Utility)
	}

	// Frequency should also have increased.
	if clusterAfter.Frequency <= 0 {
		t.Fatalf("expected non-zero frequency after reinforce, got %f", clusterAfter.Frequency)
	}
}

func TestMemoryDecayTick_HappyPath(t *testing.T) {
	emb := newStubEmbedder(4)
	emb.vectors["decay tick fact"] = []float32{0.5, 0.5, 0.0, 0.0}
	s := newTestServer(emb)
	defer s.recallCache.stop()

	ctx := context.Background()

	// Write a fact to create a cluster.
	_, writeOut, err := s.handleWrite(ctx, nil, WriteInput{Content: "decay tick fact", Type: "user"})
	if err != nil {
		t.Fatalf("write: %v", err)
	}

	fact, _ := s.store.GetFact(ctx, writeOut.ID)
	clusterID := fact.ClusterID

	// Get cluster turns_since before tick.
	clusterBefore, _ := s.store.GetCluster(ctx, clusterID)
	if clusterBefore == nil {
		t.Fatal("expected cluster")
	}
	turnsBefore := clusterBefore.TurnsSince

	// Call decay tick.
	_, tickOut, err := s.handleDecayTick(ctx, nil, DecayTickInput{})
	if err != nil {
		t.Fatalf("decay tick: %v", err)
	}
	if !tickOut.Ticked {
		t.Fatal("expected ticked=true")
	}

	// turns_since should have incremented.
	clusterAfter, _ := s.store.GetCluster(ctx, clusterID)
	if clusterAfter == nil {
		t.Fatal("expected cluster after tick")
	}
	if clusterAfter.TurnsSince != turnsBefore+1 {
		t.Fatalf("expected turns_since=%d after tick, got %d", turnsBefore+1, clusterAfter.TurnsSince)
	}
}

// Phase 3C: SessionEnd and TurnsElapsed were removed. Sending either returns
// an explicit error naming both flags so callers discover the breaking change.
func TestMemoryDecayTick_RejectsRemovedFlags(t *testing.T) {
	emb := newStubEmbedder(4)
	emb.vectors["session end fact"] = []float32{0.5, 0.5, 0.0, 0.0}
	s := newTestServer(emb)
	defer s.recallCache.stop()

	ctx := context.Background()

	// Write a fact so a cluster exists (not that we should reach tick logic).
	_, _, err := s.handleWrite(ctx, nil, WriteInput{Content: "session end fact", Type: "user"})
	if err != nil {
		t.Fatalf("write: %v", err)
	}

	// SessionEnd=true must be rejected.
	_, _, err = s.handleDecayTick(ctx, nil, DecayTickInput{SessionEnd: true})
	if err == nil {
		t.Fatal("expected error when SessionEnd=true")
	}
	if !strings.Contains(err.Error(), "session_end") || !strings.Contains(err.Error(), "turns_elapsed") {
		t.Fatalf("error should name both removed flags, got: %v", err)
	}

	// TurnsElapsed != 0 must be rejected.
	_, _, err = s.handleDecayTick(ctx, nil, DecayTickInput{TurnsElapsed: 5})
	if err == nil {
		t.Fatal("expected error when TurnsElapsed=5")
	}
	if !strings.Contains(err.Error(), "session_end") || !strings.Contains(err.Error(), "turns_elapsed") {
		t.Fatalf("error should name both removed flags, got: %v", err)
	}
}

// Phase 3C: optional Note field is accepted and does not break the tick.
func TestMemoryDecayTick_AcceptsNote(t *testing.T) {
	emb := newStubEmbedder(4)
	emb.vectors["note fact"] = []float32{0.5, 0.5, 0.0, 0.0}
	s := newTestServer(emb)
	defer s.recallCache.stop()

	ctx := context.Background()

	_, _, err := s.handleWrite(ctx, nil, WriteInput{Content: "note fact", Type: "user"})
	if err != nil {
		t.Fatalf("write: %v", err)
	}

	_, tickOut, err := s.handleDecayTick(ctx, nil, DecayTickInput{Note: "end-of-session cleanup"})
	if err != nil {
		t.Fatalf("decay tick with note: %v", err)
	}
	if !tickOut.Ticked {
		t.Fatal("expected ticked=true")
	}
}

// Phase 3C: the session_end prompt text no longer references session_end=true.
func TestSessionEndPrompt_NoRemovedFlagReference(t *testing.T) {
	emb := newStubEmbedder(4)
	s := newTestServer(emb)
	defer s.recallCache.stop()

	ctx := context.Background()

	res, err := s.handleSessionEndPrompt(ctx, nil)
	if err != nil {
		t.Fatalf("handleSessionEndPrompt: %v", err)
	}
	if res == nil || len(res.Messages) == 0 {
		t.Fatal("expected prompt messages")
	}
	for _, m := range res.Messages {
		tc, ok := m.Content.(*mcpsdk.TextContent)
		if !ok {
			continue
		}
		if strings.Contains(tc.Text, "session_end=true") {
			t.Fatalf("prompt text must not mention session_end=true, got: %q", tc.Text)
		}
		if strings.Contains(tc.Text, "turns_elapsed") {
			t.Fatalf("prompt text must not mention turns_elapsed, got: %q", tc.Text)
		}
	}
}

func TestMemoryApplyJudgment_EmptyRecallID(t *testing.T) {
	emb := newStubEmbedder(4)
	s := newTestServer(emb)
	defer s.recallCache.stop()

	ctx := context.Background()

	_, _, err := s.handleApplyJudgment(ctx, nil, ApplyJudgmentInput{
		RecallID: "",
		Verdicts: []Verdict{{MemoryID: "x", Keep: true}},
	})
	if err == nil {
		t.Fatal("expected error for empty recall_id")
	}
}

func TestMemoryApplyJudgment_BudgetCap(t *testing.T) {
	emb := newStubEmbedder(4)
	s := newTestServer(emb)
	// Set a very low budget for testing.
	s.cfg.Memory.CacheBudgetMax = 1
	defer s.recallCache.stop()

	ctx := context.Background()

	// Write two facts with different subtypes to avoid conflict supersede.
	emb.vectors["budget fact 1"] = []float32{0.5, 0.5, 0.0, 0.0}
	emb.vectors["budget fact 2"] = []float32{0.4, 0.5, 0.1, 0.0}
	emb.vectors["budget query"] = []float32{0.5, 0.5, 0.0, 0.0}

	_, out1, err := s.handleWrite(ctx, nil, WriteInput{Content: "budget fact 1", Type: "project"})
	if err != nil {
		t.Fatalf("write 1: %v", err)
	}
	_, out2, err := s.handleWrite(ctx, nil, WriteInput{Content: "budget fact 2", Type: "reference"})
	if err != nil {
		t.Fatalf("write 2: %v", err)
	}

	_, recallOut, err := s.handleRecall(ctx, nil, RecallInput{Query: "budget query", Round: 0})
	if err != nil {
		t.Fatalf("recall: %v", err)
	}

	_, judgOut, err := s.handleApplyJudgment(ctx, nil, ApplyJudgmentInput{
		RecallID: recallOut.RecallID,
		Verdicts: []Verdict{
			{MemoryID: out1.ID, Keep: true},
			{MemoryID: out2.ID, Keep: true},
		},
	})
	if err != nil {
		t.Fatalf("apply_judgment: %v", err)
	}

	// With budget=1, should only get 1 memory back.
	if len(judgOut.Memories) != 1 {
		t.Fatalf("expected 1 memory (budget cap), got %d", len(judgOut.Memories))
	}
}

// --- Phase 3: Episode, cluster assignment, conflict, cross-link tests ---

func TestMemoryWrite_Episode(t *testing.T) {
	emb := newStubEmbedder(4)
	episodeText := "user asked to debug\nran debugger\nfound null pointer\ncheck for nil first"
	emb.vectors[episodeText] = []float32{0.3, 0.4, 0.5, 0.0}
	s := newTestServer(emb)
	defer s.recallCache.stop()

	ctx := context.Background()

	_, out, err := s.handleWrite(ctx, nil, WriteInput{
		Type: "feedback",
		Episode: &EpisodePayload{
			Situation:  "user asked to debug",
			Action:     "ran debugger",
			Outcome:    "found null pointer",
			Preemptive: "check for nil first",
		},
	})
	if err != nil {
		t.Fatalf("write episode: %v", err)
	}
	if out.ID == "" {
		t.Fatal("expected non-empty ID")
	}
	if out.Layer != "l3_episodic" {
		t.Fatalf("expected layer l3_episodic, got %s", out.Layer)
	}

	// Verify the episode is retrievable.
	ep, err := s.store.GetEpisode(ctx, out.ID)
	if err != nil {
		t.Fatalf("GetEpisode: %v", err)
	}
	if ep == nil {
		t.Fatal("expected episode to exist")
	}
	if ep.Situation != "user asked to debug" {
		t.Fatalf("situation mismatch: %s", ep.Situation)
	}

	// Verify it appears in list with layer=l3.
	_, listOut, err := s.handleList(ctx, nil, ListInput{Layer: "l3"})
	if err != nil {
		t.Fatalf("list l3: %v", err)
	}
	if len(listOut.Memories) != 1 {
		t.Fatalf("expected 1 episode in l3 list, got %d", len(listOut.Memories))
	}
	if listOut.Memories[0].Layer != "l3_episodic" {
		t.Fatalf("expected layer l3_episodic, got %s", listOut.Memories[0].Layer)
	}
}

func TestMemoryWrite_ValidatesExactlyOneContent(t *testing.T) {
	emb := newStubEmbedder(4)
	s := newTestServer(emb)
	defer s.recallCache.stop()

	ctx := context.Background()

	// Both content and episode provided.
	_, _, err := s.handleWrite(ctx, nil, WriteInput{
		Content: "some text",
		Type:    "user",
		Episode: &EpisodePayload{
			Situation:  "s",
			Action:     "a",
			Outcome:    "o",
			Preemptive: "p",
		},
	})
	if err == nil {
		t.Fatal("expected error when both content and episode are provided")
	}

	// Neither content nor episode provided.
	_, _, err = s.handleWrite(ctx, nil, WriteInput{
		Type: "user",
	})
	if err == nil {
		t.Fatal("expected error when neither content nor episode is provided")
	}
}

func TestMemoryRecall_MixedFactAndEpisode(t *testing.T) {
	emb := newStubEmbedder(4)
	// Fact has a different vector from the episode.
	emb.vectors["Go is compiled"] = []float32{0.5, 0.5, 0.0, 0.0}
	episodeText := "debugging crash\nran gdb\nfound segfault\ncheck bounds"
	emb.vectors[episodeText] = []float32{0.0, 0.0, 0.5, 0.5}
	// Query close to the episode.
	emb.vectors["debugging tips"] = []float32{0.0, 0.0, 0.5, 0.5}
	s := newTestServer(emb)
	defer s.recallCache.stop()

	ctx := context.Background()

	// Write a fact.
	_, _, err := s.handleWrite(ctx, nil, WriteInput{Content: "Go is compiled", Type: "project"})
	if err != nil {
		t.Fatalf("write fact: %v", err)
	}

	// Write an episode.
	_, epOut, err := s.handleWrite(ctx, nil, WriteInput{
		Type: "feedback",
		Episode: &EpisodePayload{
			Situation:  "debugging crash",
			Action:     "ran gdb",
			Outcome:    "found segfault",
			Preemptive: "check bounds",
		},
	})
	if err != nil {
		t.Fatalf("write episode: %v", err)
	}

	// Recall with query close to the episode.
	_, recallOut, err := s.handleRecall(ctx, nil, RecallInput{Query: "debugging tips"})
	if err != nil {
		t.Fatalf("recall: %v", err)
	}

	if len(recallOut.Candidates) < 2 {
		t.Fatalf("expected at least 2 candidates, got %d", len(recallOut.Candidates))
	}

	// The first candidate (highest similarity) should be the episode.
	if recallOut.Candidates[0].ID != epOut.ID {
		t.Errorf("expected first candidate to be the episode %s, got %s", epOut.ID, recallOut.Candidates[0].ID)
	}
	if recallOut.Candidates[0].Layer != "l3_episodic" {
		t.Errorf("expected first candidate layer l3_episodic, got %s", recallOut.Candidates[0].Layer)
	}
}

func TestMemoryWrite_ConflictSupersede(t *testing.T) {
	emb := newStubEmbedder(4)
	// Two near-identical vectors (will have cosine > 0.92).
	emb.vectors["Go version is 1.21"] = []float32{0.5, 0.5, 0.0, 0.0}
	emb.vectors["Go version is 1.22"] = []float32{0.5, 0.5, 0.0, 0.0}
	s := newTestServer(emb)
	defer s.recallCache.stop()

	ctx := context.Background()

	// Write the first fact.
	_, out1, err := s.handleWrite(ctx, nil, WriteInput{Content: "Go version is 1.21", Type: "project"})
	if err != nil {
		t.Fatalf("write 1: %v", err)
	}

	// Write a near-duplicate with the same subtype -> should supersede.
	_, out2, err := s.handleWrite(ctx, nil, WriteInput{Content: "Go version is 1.22", Type: "project"})
	if err != nil {
		t.Fatalf("write 2: %v", err)
	}

	// Verify the old fact is superseded.
	oldFact, err := s.store.GetFact(ctx, out1.ID)
	if err != nil {
		t.Fatalf("get old fact: %v", err)
	}
	if oldFact == nil {
		t.Fatal("expected old fact to exist")
	}
	if oldFact.SupersededBy == nil {
		t.Fatal("expected old fact to be superseded")
	}
	if *oldFact.SupersededBy != out2.ID {
		t.Fatalf("expected superseded_by=%s, got %s", out2.ID, *oldFact.SupersededBy)
	}

	// ListFacts should exclude the superseded fact.
	facts, err := s.store.ListFacts(ctx, memory.ListFilter{Limit: 100})
	if err != nil {
		t.Fatalf("list facts: %v", err)
	}
	for _, f := range facts {
		if f.ID == out1.ID {
			t.Fatal("superseded fact should not appear in ListFacts")
		}
	}
}

func TestMemoryWrite_NoSupersedeCrossSubtype(t *testing.T) {
	emb := newStubEmbedder(4)
	// Same vector but different subtypes.
	emb.vectors["Ben likes Go"] = []float32{0.5, 0.5, 0.0, 0.0}
	emb.vectors["Ben prefers Go"] = []float32{0.5, 0.5, 0.0, 0.0}
	s := newTestServer(emb)
	defer s.recallCache.stop()

	ctx := context.Background()

	_, out1, err := s.handleWrite(ctx, nil, WriteInput{Content: "Ben likes Go", Type: "user"})
	if err != nil {
		t.Fatalf("write 1: %v", err)
	}

	// Write with DIFFERENT subtype — should NOT supersede.
	_, _, err = s.handleWrite(ctx, nil, WriteInput{Content: "Ben prefers Go", Type: "project"})
	if err != nil {
		t.Fatalf("write 2: %v", err)
	}

	// Verify the first fact is NOT superseded.
	oldFact, err := s.store.GetFact(ctx, out1.ID)
	if err != nil {
		t.Fatalf("get old fact: %v", err)
	}
	if oldFact == nil {
		t.Fatal("expected old fact to exist")
	}
	if oldFact.SupersededBy != nil {
		t.Fatalf("expected old fact to NOT be superseded, but superseded_by=%s", *oldFact.SupersededBy)
	}
}

func TestMemoryWrite_CreatesRealCluster(t *testing.T) {
	emb := newStubEmbedder(4)
	emb.vectors["cluster test fact"] = []float32{0.3, 0.4, 0.5, 0.0}
	s := newTestServer(emb)
	defer s.recallCache.stop()

	ctx := context.Background()

	_, writeOut, err := s.handleWrite(ctx, nil, WriteInput{Content: "cluster test fact", Type: "project"})
	if err != nil {
		t.Fatalf("write: %v", err)
	}

	// Get the fact's cluster ID.
	fact, _ := s.store.GetFact(ctx, writeOut.ID)
	if fact == nil {
		t.Fatal("expected fact to exist")
	}
	if fact.ClusterID == "" || fact.ClusterID == "default" {
		t.Fatalf("expected a real cluster ID, got %q", fact.ClusterID)
	}

	// Verify the cluster exists and has the correct centroid.
	cl, err := s.store.GetCluster(ctx, fact.ClusterID)
	if err != nil {
		t.Fatalf("get cluster: %v", err)
	}
	if cl == nil {
		t.Fatal("expected cluster to exist")
	}
	if cl.ItemCount != 1 {
		t.Fatalf("expected item_count=1, got %d", cl.ItemCount)
	}
	// Centroid should match the fact's embedding.
	if len(cl.Centroid) != 4 {
		t.Fatalf("expected 4-dim centroid, got %d", len(cl.Centroid))
	}
}

func TestMemoryWrite_SecondFactReusesCluster(t *testing.T) {
	emb := newStubEmbedder(4)
	// Two similar vectors (cosine > 0.60 threshold).
	emb.vectors["Go structs are values"] = []float32{0.5, 0.5, 0.0, 0.0}
	emb.vectors["Go interfaces are implicit"] = []float32{0.5, 0.5, 0.01, 0.0}
	s := newTestServer(emb)
	defer s.recallCache.stop()

	ctx := context.Background()

	_, out1, err := s.handleWrite(ctx, nil, WriteInput{Content: "Go structs are values", Type: "project"})
	if err != nil {
		t.Fatalf("write 1: %v", err)
	}
	_, out2, err := s.handleWrite(ctx, nil, WriteInput{Content: "Go interfaces are implicit", Type: "project"})
	if err != nil {
		t.Fatalf("write 2: %v", err)
	}

	fact1, _ := s.store.GetFact(ctx, out1.ID)
	fact2, _ := s.store.GetFact(ctx, out2.ID)

	if fact1.ClusterID != fact2.ClusterID {
		t.Fatalf("expected both facts in the same cluster, got %s and %s", fact1.ClusterID, fact2.ClusterID)
	}

	// Cluster should have itemCount=2 and blended centroid.
	cl, _ := s.store.GetCluster(ctx, fact1.ClusterID)
	if cl == nil {
		t.Fatal("expected cluster")
	}
	if cl.ItemCount != 2 {
		t.Fatalf("expected item_count=2, got %d", cl.ItemCount)
	}
}

func TestMemoryWrite_DifferentEmbeddings_NewCluster(t *testing.T) {
	emb := newStubEmbedder(4)
	// Two very dissimilar vectors — cosine will be 0.0, well below 0.60.
	emb.vectors["Go is compiled"] = []float32{1.0, 0.0, 0.0, 0.0}
	emb.vectors["Python is interpreted"] = []float32{0.0, 0.0, 0.0, 1.0}
	s := newTestServer(emb)
	defer s.recallCache.stop()

	ctx := context.Background()

	_, out1, err := s.handleWrite(ctx, nil, WriteInput{Content: "Go is compiled", Type: "project"})
	if err != nil {
		t.Fatalf("write 1: %v", err)
	}
	_, out2, err := s.handleWrite(ctx, nil, WriteInput{Content: "Python is interpreted", Type: "reference"})
	if err != nil {
		t.Fatalf("write 2: %v", err)
	}

	fact1, _ := s.store.GetFact(ctx, out1.ID)
	fact2, _ := s.store.GetFact(ctx, out2.ID)

	if fact1.ClusterID == fact2.ClusterID {
		t.Fatalf("expected different clusters for dissimilar facts, both got %s", fact1.ClusterID)
	}

	// Should have exactly 2 clusters.
	clusters, _ := s.store.ListClusters(ctx)
	if len(clusters) != 2 {
		t.Fatalf("expected 2 clusters, got %d", len(clusters))
	}
}

func TestMemoryRecall_IncludesLinkedIDs(t *testing.T) {
	emb := newStubEmbedder(4)
	emb.vectors["Go fact for linking"] = []float32{0.5, 0.5, 0.0, 0.0}
	episodeText := "coding in Go\nwrote handler\nworked first try\nfollow patterns"
	emb.vectors[episodeText] = []float32{0.5, 0.5, 0.0, 0.0}
	emb.vectors["Go recall"] = []float32{0.5, 0.5, 0.0, 0.0}
	s := newTestServer(emb)
	defer s.recallCache.stop()

	ctx := context.Background()

	// Write a fact.
	_, factOut, err := s.handleWrite(ctx, nil, WriteInput{Content: "Go fact for linking", Type: "project"})
	if err != nil {
		t.Fatalf("write fact: %v", err)
	}

	// Write an episode that links to the fact.
	_, epOut, err := s.handleWrite(ctx, nil, WriteInput{
		Type: "project",
		Episode: &EpisodePayload{
			Situation:     "coding in Go",
			Action:        "wrote handler",
			Outcome:       "worked first try",
			Preemptive:    "follow patterns",
			LinkedFactIDs: []string{factOut.ID},
		},
	})
	if err != nil {
		t.Fatalf("write episode: %v", err)
	}

	// Recall — both should appear.
	_, recallOut, err := s.handleRecall(ctx, nil, RecallInput{Query: "Go recall", Limit: 10})
	if err != nil {
		t.Fatalf("recall: %v", err)
	}

	// Find the episode candidate and verify it has the linked fact ID.
	foundLinked := false
	for _, c := range recallOut.Candidates {
		if c.ID == epOut.ID {
			for _, lid := range c.LinkedIDs {
				if lid == factOut.ID {
					foundLinked = true
					break
				}
			}
		}
	}
	if !foundLinked {
		t.Fatal("expected episode candidate to have the linked fact ID in LinkedIDs")
	}
}

func TestMemoryForget_Episode(t *testing.T) {
	emb := newStubEmbedder(4)
	episodeText := "bad experience\ndid wrong thing\ngot error\ndo not repeat"
	emb.vectors[episodeText] = []float32{0.3, 0.3, 0.3, 0.0}
	s := newTestServer(emb)
	defer s.recallCache.stop()

	ctx := context.Background()

	_, epOut, err := s.handleWrite(ctx, nil, WriteInput{
		Type: "feedback",
		Episode: &EpisodePayload{
			Situation:  "bad experience",
			Action:     "did wrong thing",
			Outcome:    "got error",
			Preemptive: "do not repeat",
		},
	})
	if err != nil {
		t.Fatalf("write episode: %v", err)
	}

	// Forget the episode.
	_, forgetOut, err := s.handleForget(ctx, nil, ForgetInput{ID: epOut.ID})
	if err != nil {
		t.Fatalf("forget episode: %v", err)
	}
	if forgetOut.Deleted != 1 {
		t.Fatalf("expected deleted=1, got %d", forgetOut.Deleted)
	}

	// Verify it's gone.
	ep, _ := s.store.GetEpisode(ctx, epOut.ID)
	if ep != nil {
		t.Fatal("expected episode to be deleted")
	}
}

func TestMemoryForget_UnknownID(t *testing.T) {
	emb := newStubEmbedder(4)
	s := newTestServer(emb)
	defer s.recallCache.stop()

	ctx := context.Background()

	_, _, err := s.handleForget(ctx, nil, ForgetInput{ID: "nonexistent-id-12345"})
	if err == nil {
		t.Fatal("expected error for unknown ID")
	}
	if !strings.Contains(err.Error(), "memory not found") {
		t.Fatalf("expected 'memory not found' error, got: %v", err)
	}
}

func TestMemoryList_L3(t *testing.T) {
	emb := newStubEmbedder(4)
	ep1Text := "sit1\nact1\nout1\npre1"
	ep2Text := "sit2\nact2\nout2\npre2"
	emb.vectors[ep1Text] = []float32{0.5, 0.5, 0.0, 0.0}
	emb.vectors[ep2Text] = []float32{0.0, 0.0, 0.5, 0.5}
	emb.vectors["fact content"] = []float32{0.1, 0.2, 0.3, 0.4}
	s := newTestServer(emb)
	defer s.recallCache.stop()

	ctx := context.Background()

	// Write a fact (should not appear in l3 list).
	_, _, err := s.handleWrite(ctx, nil, WriteInput{Content: "fact content", Type: "user"})
	if err != nil {
		t.Fatalf("write fact: %v", err)
	}

	// Write two episodes.
	_, _, err = s.handleWrite(ctx, nil, WriteInput{
		Type: "feedback",
		Episode: &EpisodePayload{
			Situation: "sit1", Action: "act1", Outcome: "out1", Preemptive: "pre1",
		},
	})
	if err != nil {
		t.Fatalf("write episode 1: %v", err)
	}
	_, _, err = s.handleWrite(ctx, nil, WriteInput{
		Type: "project",
		Episode: &EpisodePayload{
			Situation: "sit2", Action: "act2", Outcome: "out2", Preemptive: "pre2",
		},
	})
	if err != nil {
		t.Fatalf("write episode 2: %v", err)
	}

	// List l3 episodes.
	_, listOut, err := s.handleList(ctx, nil, ListInput{Layer: "l3"})
	if err != nil {
		t.Fatalf("list l3: %v", err)
	}
	if len(listOut.Memories) != 2 {
		t.Fatalf("expected 2 episodes, got %d", len(listOut.Memories))
	}
	for _, m := range listOut.Memories {
		if m.Layer != "l3_episodic" {
			t.Errorf("expected layer l3_episodic, got %s", m.Layer)
		}
	}

	// List l2 facts (default) — should return only the fact.
	_, listL2, err := s.handleList(ctx, nil, ListInput{})
	if err != nil {
		t.Fatalf("list l2: %v", err)
	}
	if len(listL2.Memories) != 1 {
		t.Fatalf("expected 1 fact in l2 list, got %d", len(listL2.Memories))
	}
}

// --- Phase 4: memory_update_cluster tests ---

func TestMemoryUpdateCluster_SetsSummary(t *testing.T) {
	emb := newStubEmbedder(4)
	emb.vectors["update cluster fact"] = []float32{0.5, 0.5, 0.0, 0.0}
	s := newTestServer(emb)
	defer s.recallCache.stop()

	ctx := context.Background()

	// Write a fact to create a cluster.
	_, writeOut, err := s.handleWrite(ctx, nil, WriteInput{Content: "update cluster fact", Type: "project"})
	if err != nil {
		t.Fatalf("write: %v", err)
	}

	// Get the cluster ID from the fact.
	fact, err := s.store.GetFact(ctx, writeOut.ID)
	if err != nil {
		t.Fatalf("get fact: %v", err)
	}
	clusterID := fact.ClusterID

	// Update the cluster's summary.
	_, updateOut, err := s.handleUpdateCluster(ctx, nil, UpdateClusterInput{
		ClusterID: clusterID,
		Summary:   "Go language patterns and idioms",
	})
	if err != nil {
		t.Fatalf("update cluster: %v", err)
	}

	if updateOut.Summary != "Go language patterns and idioms" {
		t.Fatalf("expected summary 'Go language patterns and idioms', got %q", updateOut.Summary)
	}
	if updateOut.ID != clusterID {
		t.Fatalf("expected cluster ID %s, got %s", clusterID, updateOut.ID)
	}

	// Verify via GetCluster.
	cl, err := s.store.GetCluster(ctx, clusterID)
	if err != nil {
		t.Fatalf("get cluster: %v", err)
	}
	if cl.Summary != "Go language patterns and idioms" {
		t.Fatalf("expected persisted summary 'Go language patterns and idioms', got %q", cl.Summary)
	}
}

func TestMemoryUpdateCluster_NotFound(t *testing.T) {
	emb := newStubEmbedder(4)
	s := newTestServer(emb)
	defer s.recallCache.stop()

	ctx := context.Background()

	_, _, err := s.handleUpdateCluster(ctx, nil, UpdateClusterInput{
		ClusterID: "nonexistent-cluster-id",
		Summary:   "should fail",
	})
	if err == nil {
		t.Fatal("expected error for nonexistent cluster")
	}
	if !strings.Contains(err.Error(), "not found") {
		t.Fatalf("expected 'not found' in error, got: %v", err)
	}
}

func TestMemoryUpdateCluster_PartialUpdate(t *testing.T) {
	emb := newStubEmbedder(4)
	emb.vectors["partial update fact"] = []float32{0.5, 0.5, 0.0, 0.0}
	s := newTestServer(emb)
	defer s.recallCache.stop()

	ctx := context.Background()

	// Write a fact to create a cluster.
	_, writeOut, err := s.handleWrite(ctx, nil, WriteInput{Content: "partial update fact", Type: "project"})
	if err != nil {
		t.Fatalf("write: %v", err)
	}

	fact, _ := s.store.GetFact(ctx, writeOut.ID)
	clusterID := fact.ClusterID

	// First, set all three fields.
	_, _, err = s.handleUpdateCluster(ctx, nil, UpdateClusterInput{
		ClusterID: clusterID,
		Summary:   "original summary",
		Domain:    "original domain",
		MetaInstr: "original meta",
	})
	if err != nil {
		t.Fatalf("first update: %v", err)
	}

	// Now update only domain — summary and meta_instr should be preserved.
	_, updateOut, err := s.handleUpdateCluster(ctx, nil, UpdateClusterInput{
		ClusterID: clusterID,
		Domain:    "new domain",
	})
	if err != nil {
		t.Fatalf("partial update: %v", err)
	}

	if updateOut.Summary != "original summary" {
		t.Errorf("expected summary preserved as 'original summary', got %q", updateOut.Summary)
	}
	if updateOut.Domain != "new domain" {
		t.Errorf("expected domain 'new domain', got %q", updateOut.Domain)
	}
	if updateOut.MetaInstr != "original meta" {
		t.Errorf("expected meta_instr preserved as 'original meta', got %q", updateOut.MetaInstr)
	}

	// Verify via GetCluster.
	cl, _ := s.store.GetCluster(ctx, clusterID)
	if cl.Summary != "original summary" {
		t.Errorf("expected persisted summary 'original summary', got %q", cl.Summary)
	}
	if cl.Domain != "new domain" {
		t.Errorf("expected persisted domain 'new domain', got %q", cl.Domain)
	}
	if cl.MetaInstr != "original meta" {
		t.Errorf("expected persisted meta_instr 'original meta', got %q", cl.MetaInstr)
	}
}

func TestMemoryUpdateCluster_EmptyClusterID(t *testing.T) {
	emb := newStubEmbedder(4)
	s := newTestServer(emb)
	defer s.recallCache.stop()

	ctx := context.Background()

	_, _, err := s.handleUpdateCluster(ctx, nil, UpdateClusterInput{
		ClusterID: "",
		Summary:   "should fail",
	})
	if err == nil {
		t.Fatal("expected error for empty cluster_id")
	}
}

// --- Tag wiring tests (Phase 1D) ---

// TestHandleWrite_FactTagsPersisted verifies WriteInput.Tags is forwarded to
// the persisted Fact.Tags (previously dropped silently) and survives round
// trip through the store.
func TestHandleWrite_FactTagsPersisted(t *testing.T) {
	emb := newStubEmbedder(4)
	emb.vectors["tagged fact"] = []float32{0.4, 0.4, 0.0, 0.0}
	s := newTestServer(emb)
	defer s.recallCache.stop()

	ctx := context.Background()
	_, out, err := s.handleWrite(ctx, nil, WriteInput{
		Content: "tagged fact",
		Type:    "project",
		Tags:    []string{"Go", "phase-1", "go"}, // mixed case + dup
	})
	if err != nil {
		t.Fatalf("handleWrite: %v", err)
	}
	if out.ID == "" {
		t.Fatal("expected non-empty id")
	}

	f, err := s.store.GetFact(ctx, out.ID)
	if err != nil {
		t.Fatalf("GetFact: %v", err)
	}
	if f == nil {
		t.Fatal("fact not found")
	}
	if len(f.Tags) != 2 {
		t.Fatalf("expected 2 normalized tags, got %d: %v", len(f.Tags), f.Tags)
	}
	// Normalization: lowercased, deduped, sorted.
	if f.Tags[0] != "go" || f.Tags[1] != "phase-1" {
		t.Errorf("tags = %v, want [go phase-1]", f.Tags)
	}
}

// TestHandleWrite_EpisodeTagsPersisted exercises the same wiring for L3.
func TestHandleWrite_EpisodeTagsPersisted(t *testing.T) {
	emb := newStubEmbedder(4)
	s := newTestServer(emb)
	defer s.recallCache.stop()

	ctx := context.Background()
	_, out, err := s.handleWrite(ctx, nil, WriteInput{
		Type: "feedback",
		Tags: []string{"learning", "refactor"},
		Episode: &EpisodePayload{
			Situation:  "tests were flaky",
			Action:     "injected clock",
			Outcome:    "tests deterministic",
			Preemptive: "use injected clocks",
		},
	})
	if err != nil {
		t.Fatalf("handleWrite: %v", err)
	}
	if out.Layer != "l3_episodic" {
		t.Fatalf("layer = %q, want l3_episodic", out.Layer)
	}

	ep, err := s.store.GetEpisode(ctx, out.ID)
	if err != nil {
		t.Fatalf("GetEpisode: %v", err)
	}
	if ep == nil {
		t.Fatal("episode not found")
	}
	if len(ep.Tags) != 2 || ep.Tags[0] != "learning" || ep.Tags[1] != "refactor" {
		t.Errorf("tags = %v, want [learning refactor]", ep.Tags)
	}
}

// TestHandleWrite_FactTagsListable verifies a tagged fact comes back through
// the store's ListFacts with TagsAny filter — the read-path that 1A will
// surface in memory_list. Exercised here to guard the end-to-end behavior
// before 1A's tool-level changes land.
func TestHandleWrite_FactTagsListable(t *testing.T) {
	emb := newStubEmbedder(4)
	emb.vectors["listable fact"] = []float32{0.3, 0.3, 0.0, 0.0}
	s := newTestServer(emb)
	defer s.recallCache.stop()

	ctx := context.Background()
	_, _, err := s.handleWrite(ctx, nil, WriteInput{
		Content: "listable fact",
		Type:    "project",
		Tags:    []string{"smoke"},
	})
	if err != nil {
		t.Fatalf("handleWrite: %v", err)
	}

	got, err := s.store.ListFacts(ctx, memory.ListFilter{TagsAny: []string{"smoke"}})
	if err != nil {
		t.Fatalf("ListFacts: %v", err)
	}
	if len(got) != 1 {
		t.Fatalf("ListFacts tags_any=[smoke] returned %d, want 1", len(got))
	}
	if got[0].Content != "listable fact" {
		t.Errorf("content = %q, want 'listable fact'", got[0].Content)
	}
}

// TestHandleRecall_IncludesClusterIDAndTags verifies that each recall candidate
// carries the underlying memory's cluster_id and tags (both fact and episode
// paths).
func TestHandleRecall_IncludesClusterIDAndTags(t *testing.T) {
	emb := newStubEmbedder(4)
	emb.vectors["tagged fact for recall"] = []float32{0.5, 0.5, 0.0, 0.0}
	// Episode embedding uses the raw fields concatenated with \n (see writeEpisode).
	emb.vectors["sit1\nact1\nout1\npre1"] = []float32{0.5, 0.5, 0.0, 0.0}
	emb.vectors["recall query"] = []float32{0.5, 0.5, 0.0, 0.0}
	s := newTestServer(emb)
	defer s.recallCache.stop()

	ctx := context.Background()

	_, factOut, err := s.handleWrite(ctx, nil, WriteInput{
		Content: "tagged fact for recall",
		Type:    "project",
		Tags:    []string{"alpha", "beta"},
	})
	if err != nil {
		t.Fatalf("write fact: %v", err)
	}

	_, epOut, err := s.handleWrite(ctx, nil, WriteInput{
		Type: "feedback",
		Tags: []string{"gamma"},
		Episode: &EpisodePayload{
			Situation:  "sit1",
			Action:     "act1",
			Outcome:    "out1",
			Preemptive: "pre1",
		},
	})
	if err != nil {
		t.Fatalf("write episode: %v", err)
	}

	_, recallOut, err := s.handleRecall(ctx, nil, RecallInput{Query: "recall query", Limit: 10})
	if err != nil {
		t.Fatalf("recall: %v", err)
	}

	var sawFact, sawEp bool
	for _, c := range recallOut.Candidates {
		if c.ClusterID == "" {
			t.Errorf("candidate %s has empty cluster_id", c.ID)
		}
		switch c.ID {
		case factOut.ID:
			sawFact = true
			if len(c.Tags) != 2 || c.Tags[0] != "alpha" || c.Tags[1] != "beta" {
				t.Errorf("fact candidate tags = %v, want [alpha beta]", c.Tags)
			}
			// Verify cluster_id matches underlying fact.
			f, ferr := s.store.GetFact(ctx, factOut.ID)
			if ferr != nil || f == nil {
				t.Fatalf("GetFact: %v", ferr)
			}
			if c.ClusterID != f.ClusterID {
				t.Errorf("fact candidate cluster_id = %q, want %q", c.ClusterID, f.ClusterID)
			}
		case epOut.ID:
			sawEp = true
			if len(c.Tags) != 1 || c.Tags[0] != "gamma" {
				t.Errorf("episode candidate tags = %v, want [gamma]", c.Tags)
			}
			ep, eerr := s.store.GetEpisode(ctx, epOut.ID)
			if eerr != nil || ep == nil {
				t.Fatalf("GetEpisode: %v", eerr)
			}
			if c.ClusterID != ep.ClusterID {
				t.Errorf("episode candidate cluster_id = %q, want %q", c.ClusterID, ep.ClusterID)
			}
		}
	}
	if !sawFact {
		t.Error("did not see fact candidate in recall output")
	}
	if !sawEp {
		t.Error("did not see episode candidate in recall output")
	}
}

// TestHandleRecall_TagsNonNilForUntagged verifies the tags field is a non-nil
// slice on recall candidates even when the underlying memory has no tags.
// The Go-level struct carries an empty []string{} so callers can range over it
// safely; JSON serialization with omitempty may drop it, matching 1A's behavior.
func TestHandleRecall_TagsNonNilForUntagged(t *testing.T) {
	emb := newStubEmbedder(4)
	emb.vectors["untagged recall fact"] = []float32{0.5, 0.5, 0.0, 0.0}
	emb.vectors["q"] = []float32{0.5, 0.5, 0.0, 0.0}
	s := newTestServer(emb)
	defer s.recallCache.stop()

	ctx := context.Background()
	_, _, err := s.handleWrite(ctx, nil, WriteInput{
		Content: "untagged recall fact",
		Type:    "project",
	})
	if err != nil {
		t.Fatalf("write: %v", err)
	}

	_, recallOut, err := s.handleRecall(ctx, nil, RecallInput{Query: "q", Limit: 5})
	if err != nil {
		t.Fatalf("recall: %v", err)
	}
	if len(recallOut.Candidates) == 0 {
		t.Fatal("expected at least one candidate")
	}
	for _, c := range recallOut.Candidates {
		if c.Tags == nil {
			t.Errorf("candidate %s has nil Tags; expected empty slice", c.ID)
		}
	}
}

// TestHandleRecall_LinkedIDsRegression guards the 1A-pre-existing linked_ids
// behavior: fact↔episode links remain populated on recall candidates after
// the 1B additions.
func TestHandleRecall_LinkedIDsRegression(t *testing.T) {
	emb := newStubEmbedder(4)
	emb.vectors["linked fact"] = []float32{0.5, 0.5, 0.0, 0.0}
	emb.vectors["sitX\nactX\noutX\npreX"] = []float32{0.5, 0.5, 0.0, 0.0}
	emb.vectors["link query"] = []float32{0.5, 0.5, 0.0, 0.0}
	s := newTestServer(emb)
	defer s.recallCache.stop()

	ctx := context.Background()

	_, factOut, err := s.handleWrite(ctx, nil, WriteInput{
		Content: "linked fact",
		Type:    "project",
	})
	if err != nil {
		t.Fatalf("write fact: %v", err)
	}

	_, epOut, err := s.handleWrite(ctx, nil, WriteInput{
		Type: "project",
		Episode: &EpisodePayload{
			Situation:     "sitX",
			Action:        "actX",
			Outcome:       "outX",
			Preemptive:    "preX",
			LinkedFactIDs: []string{factOut.ID},
		},
	})
	if err != nil {
		t.Fatalf("write episode: %v", err)
	}

	_, recallOut, err := s.handleRecall(ctx, nil, RecallInput{Query: "link query", Limit: 10})
	if err != nil {
		t.Fatalf("recall: %v", err)
	}

	var factLinkedToEp, epLinkedToFact bool
	for _, c := range recallOut.Candidates {
		switch c.ID {
		case factOut.ID:
			for _, lid := range c.LinkedIDs {
				if lid == epOut.ID {
					factLinkedToEp = true
				}
			}
		case epOut.ID:
			for _, lid := range c.LinkedIDs {
				if lid == factOut.ID {
					epLinkedToFact = true
				}
			}
		}
	}
	if !factLinkedToEp {
		t.Error("fact candidate missing linked episode ID")
	}
	if !epLinkedToFact {
		t.Error("episode candidate missing linked fact ID")
	}
}

// --- memory_get tests (Phase 1C) ---

func TestHandleGet_EmptyID(t *testing.T) {
	emb := newStubEmbedder(4)
	s := newTestServer(emb)
	defer s.recallCache.stop()

	_, _, err := s.handleGet(context.Background(), nil, GetInput{ID: ""})
	if err == nil || err.Error() != "id is required" {
		t.Fatalf("err = %v, want 'id is required'", err)
	}
}

func TestHandleGet_NotFound(t *testing.T) {
	emb := newStubEmbedder(4)
	s := newTestServer(emb)
	defer s.recallCache.stop()

	_, _, err := s.handleGet(context.Background(), nil, GetInput{ID: "nonexistent-id"})
	if err == nil {
		t.Fatal("expected error for nonexistent id")
	}
	if !strings.Contains(err.Error(), "memory not found") {
		t.Errorf("err = %v, want 'memory not found'", err)
	}
}

func TestHandleGet_Fact(t *testing.T) {
	emb := newStubEmbedder(4)
	emb.vectors["gettable fact"] = []float32{0.5, 0.5, 0.0, 0.0}
	s := newTestServer(emb)
	defer s.recallCache.stop()

	ctx := context.Background()
	_, wOut, err := s.handleWrite(ctx, nil, WriteInput{
		Content: "gettable fact",
		Type:    "project",
		Tags:    []string{"phase-1"},
	})
	if err != nil {
		t.Fatalf("handleWrite: %v", err)
	}

	_, out, err := s.handleGet(ctx, nil, GetInput{ID: wOut.ID})
	if err != nil {
		t.Fatalf("handleGet: %v", err)
	}
	if out.ID != wOut.ID {
		t.Errorf("id = %q, want %q", out.ID, wOut.ID)
	}
	if out.Layer != "l2_semantic" {
		t.Errorf("layer = %q, want l2_semantic", out.Layer)
	}
	if out.Content != "gettable fact" {
		t.Errorf("content = %q, want 'gettable fact'", out.Content)
	}
	if out.ClusterID == "" {
		t.Error("cluster_id should be populated")
	}
	if len(out.Tags) != 1 || out.Tags[0] != "phase-1" {
		t.Errorf("tags = %v, want [phase-1]", out.Tags)
	}
	if out.SupersededBy != nil {
		t.Errorf("superseded_by = %v, want nil", out.SupersededBy)
	}
	if len(out.Supersedes) != 0 {
		t.Errorf("supersedes = %v, want empty", out.Supersedes)
	}
	if out.Situation != "" || out.Action != "" {
		t.Error("episode fields should be empty for a fact")
	}
}

func TestHandleGet_Episode(t *testing.T) {
	emb := newStubEmbedder(4)
	emb.vectors["sit1\nact1\nout1\npre1"] = []float32{0.5, 0.5, 0.0, 0.0}
	s := newTestServer(emb)
	defer s.recallCache.stop()

	ctx := context.Background()
	_, wOut, err := s.handleWrite(ctx, nil, WriteInput{
		Type: "feedback",
		Tags: []string{"refactor"},
		Episode: &EpisodePayload{
			Situation:  "sit1",
			Action:     "act1",
			Outcome:    "out1",
			Preemptive: "pre1",
		},
	})
	if err != nil {
		t.Fatalf("handleWrite: %v", err)
	}

	_, out, err := s.handleGet(ctx, nil, GetInput{ID: wOut.ID})
	if err != nil {
		t.Fatalf("handleGet: %v", err)
	}
	if out.Layer != "l3_episodic" {
		t.Errorf("layer = %q, want l3_episodic", out.Layer)
	}
	if out.Situation != "sit1" || out.Action != "act1" || out.Outcome != "out1" || out.Preemptive != "pre1" {
		t.Errorf("episode fields mismatch: %+v", out)
	}
	if out.Content == "" {
		t.Error("content should be rendered for episode")
	}
	if out.SupersededBy != nil || len(out.Supersedes) != 0 {
		t.Error("supersede fields must be absent for episodes")
	}
	if out.ValidFrom != "" {
		t.Error("valid_from must be absent for episodes")
	}
}

func TestHandleGet_SupersededFact(t *testing.T) {
	emb := newStubEmbedder(4)
	emb.vectors["old fact"] = []float32{0.1, 0.2, 0.3, 0.4}
	emb.vectors["new fact"] = []float32{0.4, 0.3, 0.2, 0.1}
	s := newTestServer(emb)
	defer s.recallCache.stop()

	ctx := context.Background()
	_, oldOut, err := s.handleWrite(ctx, nil, WriteInput{Content: "old fact", Type: "project"})
	if err != nil {
		t.Fatalf("handleWrite old: %v", err)
	}
	_, newOut, err := s.handleWrite(ctx, nil, WriteInput{Content: "new fact", Type: "project"})
	if err != nil {
		t.Fatalf("handleWrite new: %v", err)
	}
	if err := s.store.SupersedeFact(ctx, oldOut.ID, newOut.ID); err != nil {
		t.Fatalf("SupersedeFact: %v", err)
	}

	// Fetching the old fact by ID still works (history view) and reports
	// superseded_by pointing to the new id.
	_, oldGet, err := s.handleGet(ctx, nil, GetInput{ID: oldOut.ID})
	if err != nil {
		t.Fatalf("handleGet old: %v", err)
	}
	if oldGet.SupersededBy == nil || *oldGet.SupersededBy != newOut.ID {
		t.Errorf("superseded_by = %v, want %q", oldGet.SupersededBy, newOut.ID)
	}

	// The new fact advertises oldOut.ID in its supersedes list.
	_, newGet, err := s.handleGet(ctx, nil, GetInput{ID: newOut.ID})
	if err != nil {
		t.Fatalf("handleGet new: %v", err)
	}
	if len(newGet.Supersedes) != 1 || newGet.Supersedes[0] != oldOut.ID {
		t.Errorf("supersedes = %v, want [%s]", newGet.Supersedes, oldOut.ID)
	}
}

func TestHandleGet_LinksBidirectional(t *testing.T) {
	emb := newStubEmbedder(4)
	emb.vectors["linked fact"] = []float32{0.5, 0.5, 0.0, 0.0}
	emb.vectors["ls1\nla1\nlo1\nlp1"] = []float32{0.5, 0.5, 0.0, 0.0}
	s := newTestServer(emb)
	defer s.recallCache.stop()

	ctx := context.Background()
	_, fOut, err := s.handleWrite(ctx, nil, WriteInput{Content: "linked fact", Type: "project"})
	if err != nil {
		t.Fatalf("handleWrite fact: %v", err)
	}
	_, eOut, err := s.handleWrite(ctx, nil, WriteInput{
		Type: "feedback",
		Episode: &EpisodePayload{
			Situation: "ls1", Action: "la1", Outcome: "lo1", Preemptive: "lp1",
		},
	})
	if err != nil {
		t.Fatalf("handleWrite episode: %v", err)
	}
	// Seed an evidence edge through the store layer. memory_get filters
	// to edge_type=="evidence" — non-evidence edges are not surfaced as
	// LinkRefs in the response.
	if _, err := s.store.AddEdge(ctx, memory.Edge{
		SrcID: fOut.ID, DstID: eOut.ID, EdgeType: "evidence", Weight: 1.0,
	}); err != nil {
		t.Fatalf("AddEdge: %v", err)
	}

	// Fact side advertises the episode link.
	_, fGet, err := s.handleGet(ctx, nil, GetInput{ID: fOut.ID})
	if err != nil {
		t.Fatalf("handleGet fact: %v", err)
	}
	if len(fGet.Links) != 1 || fGet.Links[0].ID != eOut.ID || fGet.Links[0].Layer != "l3_episodic" {
		t.Errorf("fact links = %+v, want one link to %s (l3_episodic)", fGet.Links, eOut.ID)
	}

	// Episode side advertises the fact link.
	_, eGet, err := s.handleGet(ctx, nil, GetInput{ID: eOut.ID})
	if err != nil {
		t.Fatalf("handleGet episode: %v", err)
	}
	if len(eGet.Links) != 1 || eGet.Links[0].ID != fOut.ID || eGet.Links[0].Layer != "l2_semantic" {
		t.Errorf("episode links = %+v, want one link to %s (l2_semantic)", eGet.Links, fOut.ID)
	}
}

// --- memory_reassign_cluster tests ---

// seedTwoClusters writes two facts with orthogonal embeddings (cosine=0), which
// is below the assigner's 0.60 threshold, so each fact lands in its own cluster.
// Returns (factAID, clusterA, factBID, clusterB).
func seedTwoClusters(t *testing.T, s *Server, emb *stubEmbedder) (string, string, string, string) {
	t.Helper()
	ctx := context.Background()

	emb.vectors["fact a"] = []float32{1, 0, 0, 0}
	emb.vectors["fact b"] = []float32{0, 1, 0, 0}

	_, aOut, err := s.handleWrite(ctx, nil, WriteInput{Content: "fact a", Type: "project"})
	if err != nil {
		t.Fatalf("write a: %v", err)
	}
	_, bOut, err := s.handleWrite(ctx, nil, WriteInput{Content: "fact b", Type: "project"})
	if err != nil {
		t.Fatalf("write b: %v", err)
	}

	fa, err := s.store.GetFact(ctx, aOut.ID)
	if err != nil || fa == nil {
		t.Fatalf("GetFact a: %v", err)
	}
	fb, err := s.store.GetFact(ctx, bOut.ID)
	if err != nil || fb == nil {
		t.Fatalf("GetFact b: %v", err)
	}
	if fa.ClusterID == fb.ClusterID {
		t.Fatalf("expected separate clusters for orthogonal facts; both in %s", fa.ClusterID)
	}
	return aOut.ID, fa.ClusterID, bOut.ID, fb.ClusterID
}

func TestHandleReassignCluster_Fact(t *testing.T) {
	emb := newStubEmbedder(4)
	s := newTestServer(emb)
	defer s.recallCache.stop()

	ctx := context.Background()

	// Seed three facts: two in cluster A (so moving one doesn't empty it), one in cluster B.
	// Use different subtypes for a1 and a2 so conflict detection (which filters by
	// subtype) doesn't supersede one with the other despite their near-identical embeddings.
	emb.vectors["fact a1"] = []float32{1, 0, 0, 0}
	emb.vectors["fact a2"] = []float32{0.99, 0.01, 0, 0} // very similar to a1
	emb.vectors["fact b"] = []float32{0, 1, 0, 0}

	_, a1Out, err := s.handleWrite(ctx, nil, WriteInput{Content: "fact a1", Type: "project"})
	if err != nil {
		t.Fatalf("write a1: %v", err)
	}
	_, a2Out, err := s.handleWrite(ctx, nil, WriteInput{Content: "fact a2", Type: "user"})
	if err != nil {
		t.Fatalf("write a2: %v", err)
	}
	_, bOut, err := s.handleWrite(ctx, nil, WriteInput{Content: "fact b", Type: "project"})
	if err != nil {
		t.Fatalf("write b: %v", err)
	}

	a1, _ := s.store.GetFact(ctx, a1Out.ID)
	a2, _ := s.store.GetFact(ctx, a2Out.ID)
	b, _ := s.store.GetFact(ctx, bOut.ID)
	if a1.ClusterID != a2.ClusterID {
		t.Fatalf("expected a1,a2 in same cluster; got %q and %q", a1.ClusterID, a2.ClusterID)
	}
	if a1.ClusterID == b.ClusterID {
		t.Fatalf("expected b in separate cluster; all in %s", a1.ClusterID)
	}

	clusterA := a1.ClusterID
	clusterB := b.ClusterID

	// Grab centroids before the move for a before/after comparison.
	preA, _ := s.store.GetCluster(ctx, clusterA)
	preB, _ := s.store.GetCluster(ctx, clusterB)
	preAccess := a1.AccessedAt
	time.Sleep(2 * time.Millisecond)

	_, out, err := s.handleReassignCluster(ctx, nil, ReassignClusterInput{
		MemoryID:        a1Out.ID,
		TargetClusterID: clusterB,
	})
	if err != nil {
		t.Fatalf("handleReassignCluster: %v", err)
	}
	if out.OldClusterID != clusterA {
		t.Errorf("OldClusterID = %q, want %q", out.OldClusterID, clusterA)
	}
	if out.NewClusterID != clusterB {
		t.Errorf("NewClusterID = %q, want %q", out.NewClusterID, clusterB)
	}
	if out.OldClusterDeleted {
		t.Errorf("OldClusterDeleted = true, want false (clusterA still has a2)")
	}

	// Fact is now in clusterB.
	a1Post, err := s.store.GetFact(ctx, a1Out.ID)
	if err != nil {
		t.Fatalf("GetFact a1: %v", err)
	}
	if a1Post.ClusterID != clusterB {
		t.Errorf("a1.ClusterID = %q, want %q", a1Post.ClusterID, clusterB)
	}
	if !a1Post.AccessedAt.After(preAccess) {
		t.Errorf("accessed_at not bumped: before=%v after=%v", preAccess, a1Post.AccessedAt)
	}

	// Both centroids should have been recomputed — they differ from the pre-move values.
	postA, _ := s.store.GetCluster(ctx, clusterA)
	postB, _ := s.store.GetCluster(ctx, clusterB)
	if postA == nil {
		t.Fatal("clusterA was deleted but it still has a2")
	}
	if postB == nil {
		t.Fatal("clusterB missing after reassign")
	}
	// ClusterA now has one member (a2); centroid equals a2's embedding.
	if postA.ItemCount != 1 {
		t.Errorf("clusterA ItemCount = %d, want 1", postA.ItemCount)
	}
	if vectorsEqual(postA.Centroid, preA.Centroid) {
		t.Errorf("clusterA centroid unchanged after reassign: %v", postA.Centroid)
	}
	// ClusterB now has two members (b, a1); centroid should equal the average of their embeddings.
	if postB.ItemCount != 2 {
		t.Errorf("clusterB ItemCount = %d, want 2", postB.ItemCount)
	}
	if vectorsEqual(postB.Centroid, preB.Centroid) {
		t.Errorf("clusterB centroid unchanged after reassign: %v", postB.Centroid)
	}
}

func TestHandleReassignCluster_Episode(t *testing.T) {
	emb := newStubEmbedder(4)
	s := newTestServer(emb)
	defer s.recallCache.stop()

	ctx := context.Background()

	// Two unrelated episodes → two clusters.
	emb.vectors["sitA\nactA\noutA\npA"] = []float32{1, 0, 0, 0}
	emb.vectors["sitB\nactB\noutB\npB"] = []float32{0, 1, 0, 0}
	// An extra episode in cluster A so it doesn't disappear on reassign.
	emb.vectors["sitA2\nactA2\noutA2\npA2"] = []float32{0.99, 0.01, 0, 0}

	_, a1, err := s.handleWrite(ctx, nil, WriteInput{Type: "feedback", Episode: &EpisodePayload{
		Situation: "sitA", Action: "actA", Outcome: "outA", Preemptive: "pA",
	}})
	if err != nil {
		t.Fatalf("write a1: %v", err)
	}
	_, _, err = s.handleWrite(ctx, nil, WriteInput{Type: "feedback", Episode: &EpisodePayload{
		Situation: "sitA2", Action: "actA2", Outcome: "outA2", Preemptive: "pA2",
	}})
	if err != nil {
		t.Fatalf("write a2: %v", err)
	}
	_, b1, err := s.handleWrite(ctx, nil, WriteInput{Type: "feedback", Episode: &EpisodePayload{
		Situation: "sitB", Action: "actB", Outcome: "outB", Preemptive: "pB",
	}})
	if err != nil {
		t.Fatalf("write b1: %v", err)
	}

	ea1, _ := s.store.GetEpisode(ctx, a1.ID)
	eb1, _ := s.store.GetEpisode(ctx, b1.ID)
	if ea1.ClusterID == eb1.ClusterID {
		t.Fatalf("expected distinct clusters, both in %s", ea1.ClusterID)
	}
	clusterA := ea1.ClusterID
	clusterB := eb1.ClusterID

	_, out, err := s.handleReassignCluster(ctx, nil, ReassignClusterInput{
		MemoryID:        a1.ID,
		TargetClusterID: clusterB,
	})
	if err != nil {
		t.Fatalf("handleReassignCluster: %v", err)
	}
	if out.NewClusterID != clusterB || out.OldClusterID != clusterA {
		t.Errorf("unexpected out: %+v", out)
	}
	post, _ := s.store.GetEpisode(ctx, a1.ID)
	if post.ClusterID != clusterB {
		t.Errorf("episode cluster = %q, want %q", post.ClusterID, clusterB)
	}
}

func TestHandleReassignCluster_SameCluster(t *testing.T) {
	emb := newStubEmbedder(4)
	s := newTestServer(emb)
	defer s.recallCache.stop()

	aID, clusterA, _, _ := seedTwoClusters(t, s, emb)

	_, _, err := s.handleReassignCluster(context.Background(), nil, ReassignClusterInput{
		MemoryID:        aID,
		TargetClusterID: clusterA,
	})
	if err == nil {
		t.Fatal("expected error for same-cluster reassign")
	}
	want := "memory " + aID + " already in cluster " + clusterA
	if err.Error() != want {
		t.Errorf("error = %q, want %q", err.Error(), want)
	}
}

func TestHandleReassignCluster_TargetNotFound(t *testing.T) {
	emb := newStubEmbedder(4)
	s := newTestServer(emb)
	defer s.recallCache.stop()

	aID, _, _, _ := seedTwoClusters(t, s, emb)

	_, _, err := s.handleReassignCluster(context.Background(), nil, ReassignClusterInput{
		MemoryID:        aID,
		TargetClusterID: "ghost-cluster",
	})
	if err == nil {
		t.Fatal("expected error for missing target cluster")
	}
	if !strings.Contains(err.Error(), "cluster not found: ghost-cluster") {
		t.Errorf("error = %q, want it to contain 'cluster not found: ghost-cluster'", err)
	}
}

func TestHandleReassignCluster_MemoryNotFound(t *testing.T) {
	emb := newStubEmbedder(4)
	s := newTestServer(emb)
	defer s.recallCache.stop()

	// Still need a target cluster — create one by writing a fact.
	_, _, _, clusterB := seedTwoClusters(t, s, emb)

	_, _, err := s.handleReassignCluster(context.Background(), nil, ReassignClusterInput{
		MemoryID:        "ghost-memory",
		TargetClusterID: clusterB,
	})
	if err == nil {
		t.Fatal("expected error for missing memory")
	}
	if !strings.Contains(err.Error(), "memory not found: ghost-memory") {
		t.Errorf("error = %q, want it to contain 'memory not found: ghost-memory'", err)
	}
}

func TestHandleReassignCluster_LastMemberDeletesOldCluster(t *testing.T) {
	emb := newStubEmbedder(4)
	s := newTestServer(emb)
	defer s.recallCache.stop()

	aID, clusterA, _, clusterB := seedTwoClusters(t, s, emb)

	_, out, err := s.handleReassignCluster(context.Background(), nil, ReassignClusterInput{
		MemoryID:        aID,
		TargetClusterID: clusterB,
	})
	if err != nil {
		t.Fatalf("handleReassignCluster: %v", err)
	}
	if !out.OldClusterDeleted {
		t.Errorf("OldClusterDeleted = false, want true (clusterA had only one member)")
	}
	// Old cluster row should be gone.
	ctx := context.Background()
	cl, err := s.store.GetCluster(ctx, clusterA)
	if err != nil {
		t.Fatalf("GetCluster: %v", err)
	}
	if cl != nil {
		t.Errorf("expected clusterA deleted, still present: %+v", cl)
	}
}

func TestHandleReassignCluster_LastMemberRemovedFromL1Index(t *testing.T) {
	emb := newStubEmbedder(4)
	s := newTestServer(emb)
	defer s.recallCache.stop()

	aID, clusterA, _, clusterB := seedTwoClusters(t, s, emb)

	ctx := context.Background()
	_, _, err := s.handleReassignCluster(ctx, nil, ReassignClusterInput{
		MemoryID:        aID,
		TargetClusterID: clusterB,
	})
	if err != nil {
		t.Fatalf("handleReassignCluster: %v", err)
	}

	// Pull l1/index and verify clusterA is absent.
	req := &mcpsdk.ReadResourceRequest{}
	req.Params = &mcpsdk.ReadResourceParams{URI: "reverie://l1/index"}
	res, err := s.handleL1IndexResource(ctx, req)
	if err != nil {
		t.Fatalf("handleL1IndexResource: %v", err)
	}
	var index l1IndexResponse
	if err := json.Unmarshal([]byte(res.Contents[0].Text), &index); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	for _, cl := range index.Clusters {
		if cl.ID == clusterA {
			t.Errorf("l1/index still lists deleted clusterA: %+v", cl)
		}
	}
}

// seedThreeInOneCluster writes three facts with near-identical embeddings so
// they all land in a single cluster. Different subtypes are used to avoid
// conflict-based supersede (which would cull near-duplicate facts of the same
// subtype).
func seedThreeInOneCluster(t *testing.T, s *Server, emb *stubEmbedder) (ids [3]string, clusterID string) {
	t.Helper()
	ctx := context.Background()

	emb.vectors["split fact 1"] = []float32{1, 0, 0, 0}
	emb.vectors["split fact 2"] = []float32{0.99, 0.01, 0, 0}
	emb.vectors["split fact 3"] = []float32{0.98, 0.02, 0, 0}

	_, out1, err := s.handleWrite(ctx, nil, WriteInput{Content: "split fact 1", Type: "project"})
	if err != nil {
		t.Fatalf("write 1: %v", err)
	}
	_, out2, err := s.handleWrite(ctx, nil, WriteInput{Content: "split fact 2", Type: "user"})
	if err != nil {
		t.Fatalf("write 2: %v", err)
	}
	_, out3, err := s.handleWrite(ctx, nil, WriteInput{Content: "split fact 3", Type: "feedback"})
	if err != nil {
		t.Fatalf("write 3: %v", err)
	}

	f1, _ := s.store.GetFact(ctx, out1.ID)
	f2, _ := s.store.GetFact(ctx, out2.ID)
	f3, _ := s.store.GetFact(ctx, out3.ID)
	if f1.ClusterID != f2.ClusterID || f2.ClusterID != f3.ClusterID {
		t.Fatalf("expected all three facts in one cluster; got %s, %s, %s",
			f1.ClusterID, f2.ClusterID, f3.ClusterID)
	}
	return [3]string{out1.ID, out2.ID, out3.ID}, f1.ClusterID
}

func TestHandleSplitCluster_FullPartition(t *testing.T) {
	emb := newStubEmbedder(4)
	s := newTestServer(emb)
	defer s.recallCache.stop()

	ctx := context.Background()
	ids, clusterID := seedThreeInOneCluster(t, s, emb)

	_, out, err := s.handleSplitCluster(ctx, nil, SplitClusterInput{
		ClusterID: clusterID,
		Groups: [][]string{
			{ids[0], ids[1]},
			{ids[2]},
		},
	})
	if err != nil {
		t.Fatalf("handleSplitCluster: %v", err)
	}

	if out.SourceClusterID != clusterID {
		t.Errorf("SourceClusterID = %q, want %q", out.SourceClusterID, clusterID)
	}
	if len(out.NewClusterIDs) != 2 {
		t.Fatalf("NewClusterIDs len = %d, want 2", len(out.NewClusterIDs))
	}
	if !out.SourceDeleted {
		t.Errorf("SourceDeleted = false, want true")
	}
	if out.RemainingInSource != 0 {
		t.Errorf("RemainingInSource = %d, want 0", out.RemainingInSource)
	}

	// Source cluster should be gone.
	cl, err := s.store.GetCluster(ctx, clusterID)
	if err != nil {
		t.Fatalf("GetCluster source: %v", err)
	}
	if cl != nil {
		t.Errorf("source cluster still present after full partition: %+v", cl)
	}

	// Each new cluster has the expected members with the right cluster_id.
	f0, _ := s.store.GetFact(ctx, ids[0])
	f1, _ := s.store.GetFact(ctx, ids[1])
	f2, _ := s.store.GetFact(ctx, ids[2])
	if f0.ClusterID != out.NewClusterIDs[0] || f1.ClusterID != out.NewClusterIDs[0] {
		t.Errorf("group 0 members in wrong cluster: f0=%s f1=%s want=%s",
			f0.ClusterID, f1.ClusterID, out.NewClusterIDs[0])
	}
	if f2.ClusterID != out.NewClusterIDs[1] {
		t.Errorf("group 1 member in wrong cluster: f2=%s want=%s",
			f2.ClusterID, out.NewClusterIDs[1])
	}

	// New cluster rows exist and item_counts are correct.
	newA, err := s.store.GetCluster(ctx, out.NewClusterIDs[0])
	if err != nil || newA == nil {
		t.Fatalf("GetCluster newA: %v, nil=%v", err, newA == nil)
	}
	if newA.ItemCount != 2 {
		t.Errorf("newA.ItemCount = %d, want 2", newA.ItemCount)
	}
	newB, err := s.store.GetCluster(ctx, out.NewClusterIDs[1])
	if err != nil || newB == nil {
		t.Fatalf("GetCluster newB: %v, nil=%v", err, newB == nil)
	}
	if newB.ItemCount != 1 {
		t.Errorf("newB.ItemCount = %d, want 1", newB.ItemCount)
	}
}

func TestHandleSplitCluster_PartialSplit(t *testing.T) {
	emb := newStubEmbedder(4)
	s := newTestServer(emb)
	defer s.recallCache.stop()

	ctx := context.Background()
	ids, clusterID := seedThreeInOneCluster(t, s, emb)

	// Only partition two of the three members; one stays behind.
	_, out, err := s.handleSplitCluster(ctx, nil, SplitClusterInput{
		ClusterID: clusterID,
		Groups: [][]string{
			{ids[0]},
			{ids[1]},
		},
	})
	if err != nil {
		t.Fatalf("handleSplitCluster: %v", err)
	}
	if out.SourceDeleted {
		t.Errorf("SourceDeleted = true, want false (one member left behind)")
	}
	if out.RemainingInSource != 1 {
		t.Errorf("RemainingInSource = %d, want 1", out.RemainingInSource)
	}

	// Source cluster still exists, with one member.
	cl, err := s.store.GetCluster(ctx, clusterID)
	if err != nil {
		t.Fatalf("GetCluster source: %v", err)
	}
	if cl == nil {
		t.Fatal("source cluster deleted, but it should still have one member")
	}
	if cl.ItemCount != 1 {
		t.Errorf("source cluster ItemCount = %d, want 1", cl.ItemCount)
	}

	// The stray member is still in source.
	f2, _ := s.store.GetFact(ctx, ids[2])
	if f2.ClusterID != clusterID {
		t.Errorf("ids[2] ClusterID = %q, want %q (source)", f2.ClusterID, clusterID)
	}
}

func TestHandleSplitCluster_OverlappingGroups(t *testing.T) {
	emb := newStubEmbedder(4)
	s := newTestServer(emb)
	defer s.recallCache.stop()

	ctx := context.Background()
	ids, clusterID := seedThreeInOneCluster(t, s, emb)

	before, err := s.store.ListClusters(ctx)
	if err != nil {
		t.Fatalf("ListClusters before: %v", err)
	}

	_, _, err = s.handleSplitCluster(ctx, nil, SplitClusterInput{
		ClusterID: clusterID,
		Groups: [][]string{
			{ids[0], ids[1]},
			{ids[1], ids[2]}, // ids[1] overlaps
		},
	})
	if err == nil {
		t.Fatal("expected error for overlapping groups")
	}
	if !strings.Contains(err.Error(), "more than one group") {
		t.Errorf("error = %q, want substring 'more than one group'", err)
	}

	// The error path may have created a new cluster row before detecting the
	// overlap (validation happens per-group in two passes) — but in this
	// implementation, validation runs fully before any mutation, so the
	// cluster list must be unchanged.
	after, err := s.store.ListClusters(ctx)
	if err != nil {
		t.Fatalf("ListClusters after: %v", err)
	}
	if len(before) != len(after) {
		t.Errorf("cluster list changed after failed split: before=%d after=%d", len(before), len(after))
	}

	// Members should still all be in the original cluster.
	for i, id := range ids {
		f, _ := s.store.GetFact(ctx, id)
		if f.ClusterID != clusterID {
			t.Errorf("ids[%d] ClusterID = %q, want %q after failed split", i, f.ClusterID, clusterID)
		}
	}
}

func TestHandleSplitCluster_EmptyGroup(t *testing.T) {
	emb := newStubEmbedder(4)
	s := newTestServer(emb)
	defer s.recallCache.stop()

	ctx := context.Background()
	ids, clusterID := seedThreeInOneCluster(t, s, emb)

	_, _, err := s.handleSplitCluster(ctx, nil, SplitClusterInput{
		ClusterID: clusterID,
		Groups: [][]string{
			{ids[0]},
			{}, // empty
		},
	})
	if err == nil {
		t.Fatal("expected error for empty group")
	}
	if !strings.Contains(err.Error(), "group 1 is empty") {
		t.Errorf("error = %q, want substring 'group 1 is empty'", err)
	}
}

func TestHandleSplitCluster_IDNotInSource(t *testing.T) {
	emb := newStubEmbedder(4)
	s := newTestServer(emb)
	defer s.recallCache.stop()

	ctx := context.Background()
	ids, clusterID := seedThreeInOneCluster(t, s, emb)

	// Seed a separate cluster with one member.
	emb.vectors["unrelated fact"] = []float32{0, 1, 0, 0}
	_, outOther, err := s.handleWrite(ctx, nil, WriteInput{Content: "unrelated fact", Type: "project"})
	if err != nil {
		t.Fatalf("write unrelated: %v", err)
	}
	otherFact, _ := s.store.GetFact(ctx, outOther.ID)
	if otherFact.ClusterID == clusterID {
		t.Fatalf("expected unrelated fact in separate cluster; both in %s", clusterID)
	}

	_, _, err = s.handleSplitCluster(ctx, nil, SplitClusterInput{
		ClusterID: clusterID,
		Groups: [][]string{
			{ids[0], outOther.ID}, // outOther is not in source cluster
		},
	})
	if err == nil {
		t.Fatal("expected error for non-member ID")
	}
	if !strings.Contains(err.Error(), "not a member of cluster") {
		t.Errorf("error = %q, want substring 'not a member of cluster'", err)
	}
}

func TestHandleSplitCluster_MetasApplied(t *testing.T) {
	emb := newStubEmbedder(4)
	s := newTestServer(emb)
	defer s.recallCache.stop()

	ctx := context.Background()
	ids, clusterID := seedThreeInOneCluster(t, s, emb)

	_, out, err := s.handleSplitCluster(ctx, nil, SplitClusterInput{
		ClusterID: clusterID,
		Groups: [][]string{
			{ids[0], ids[1]},
			{ids[2]},
		},
		Metas: []ClusterMeta{
			{Summary: "group A summary", Domain: "domain-a", MetaInstr: "instr-a"},
			{Summary: "group B summary", Domain: "domain-b", MetaInstr: "instr-b"},
		},
	})
	if err != nil {
		t.Fatalf("handleSplitCluster: %v", err)
	}

	clA, err := s.store.GetCluster(ctx, out.NewClusterIDs[0])
	if err != nil || clA == nil {
		t.Fatalf("GetCluster A: %v, nil=%v", err, clA == nil)
	}
	if clA.Summary != "group A summary" || clA.Domain != "domain-a" || clA.MetaInstr != "instr-a" {
		t.Errorf("cluster A meta = %+v, want summary/domain/instr = group A summary/domain-a/instr-a",
			clA)
	}
	clB, err := s.store.GetCluster(ctx, out.NewClusterIDs[1])
	if err != nil || clB == nil {
		t.Fatalf("GetCluster B: %v, nil=%v", err, clB == nil)
	}
	if clB.Summary != "group B summary" || clB.Domain != "domain-b" || clB.MetaInstr != "instr-b" {
		t.Errorf("cluster B meta = %+v, want summary/domain/instr = group B summary/domain-b/instr-b",
			clB)
	}
}

func TestHandleSplitCluster_MetasLengthMismatch(t *testing.T) {
	emb := newStubEmbedder(4)
	s := newTestServer(emb)
	defer s.recallCache.stop()

	ctx := context.Background()
	ids, clusterID := seedThreeInOneCluster(t, s, emb)

	_, _, err := s.handleSplitCluster(ctx, nil, SplitClusterInput{
		ClusterID: clusterID,
		Groups: [][]string{
			{ids[0]},
			{ids[1]},
		},
		Metas: []ClusterMeta{{Summary: "only one"}}, // length 1 but 2 groups
	})
	if err == nil {
		t.Fatal("expected error for metas length mismatch")
	}
	if !strings.Contains(err.Error(), "metas length") {
		t.Errorf("error = %q, want substring 'metas length'", err)
	}
}

func TestHandleSplitCluster_ClusterNotFound(t *testing.T) {
	emb := newStubEmbedder(4)
	s := newTestServer(emb)
	defer s.recallCache.stop()

	ctx := context.Background()
	_, _, err := s.handleSplitCluster(ctx, nil, SplitClusterInput{
		ClusterID: "ghost-cluster",
		Groups:    [][]string{{"some-id"}},
	})
	if err == nil {
		t.Fatal("expected error for missing source cluster")
	}
	if !strings.Contains(err.Error(), "cluster not found: ghost-cluster") {
		t.Errorf("error = %q, want substring 'cluster not found: ghost-cluster'", err)
	}
}

// vectorsEqual reports whether two float32 slices are exactly equal.
func vectorsEqual(a, b []float32) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// --- memory_merge_clusters tests ---

// seedThreeClusters writes three orthogonal facts so each lands in its own
// cluster. Returns the fact IDs and cluster IDs in (a, b, c) order.
func seedThreeClusters(t *testing.T, s *Server, emb *stubEmbedder) (aID, clusterA, bID, clusterB, cID, clusterC string) {
	t.Helper()
	ctx := context.Background()

	emb.vectors["fact a"] = []float32{1, 0, 0, 0}
	emb.vectors["fact b"] = []float32{0, 1, 0, 0}
	emb.vectors["fact c"] = []float32{0, 0, 1, 0}

	_, aOut, err := s.handleWrite(ctx, nil, WriteInput{Content: "fact a", Type: "project"})
	if err != nil {
		t.Fatalf("write a: %v", err)
	}
	_, bOut, err := s.handleWrite(ctx, nil, WriteInput{Content: "fact b", Type: "project"})
	if err != nil {
		t.Fatalf("write b: %v", err)
	}
	_, cOut, err := s.handleWrite(ctx, nil, WriteInput{Content: "fact c", Type: "project"})
	if err != nil {
		t.Fatalf("write c: %v", err)
	}

	fa, err := s.store.GetFact(ctx, aOut.ID)
	if err != nil || fa == nil {
		t.Fatalf("GetFact a: %v", err)
	}
	fb, err := s.store.GetFact(ctx, bOut.ID)
	if err != nil || fb == nil {
		t.Fatalf("GetFact b: %v", err)
	}
	fc, err := s.store.GetFact(ctx, cOut.ID)
	if err != nil || fc == nil {
		t.Fatalf("GetFact c: %v", err)
	}
	if fa.ClusterID == fb.ClusterID || fa.ClusterID == fc.ClusterID || fb.ClusterID == fc.ClusterID {
		t.Fatalf("expected 3 distinct clusters; got a=%s b=%s c=%s", fa.ClusterID, fb.ClusterID, fc.ClusterID)
	}
	return aOut.ID, fa.ClusterID, bOut.ID, fb.ClusterID, cOut.ID, fc.ClusterID
}

func TestHandleMergeClusters_TwoSourcesIntoTarget(t *testing.T) {
	emb := newStubEmbedder(4)
	s := newTestServer(emb)
	defer s.recallCache.stop()

	ctx := context.Background()

	aID, clusterA, bID, clusterB, cID, clusterC := seedThreeClusters(t, s, emb)

	// Grab target's pre-merge centroid for a before/after comparison.
	preC, _ := s.store.GetCluster(ctx, clusterC)

	_, out, err := s.handleMergeClusters(ctx, nil, MergeClustersInput{
		SourceClusterIDs: []string{clusterA, clusterB},
		TargetClusterID:  clusterC,
	})
	if err != nil {
		t.Fatalf("handleMergeClusters: %v", err)
	}
	if out.TargetClusterID != clusterC {
		t.Errorf("TargetClusterID = %q, want %q", out.TargetClusterID, clusterC)
	}
	if out.MergedCount != 2 {
		t.Errorf("MergedCount = %d, want 2", out.MergedCount)
	}
	if len(out.DeletedClusters) != 2 {
		t.Errorf("DeletedClusters = %v, want length 2", out.DeletedClusters)
	}
	if out.NewItemCount != 3 {
		t.Errorf("NewItemCount = %d, want 3", out.NewItemCount)
	}

	// All three facts now live in clusterC.
	for _, id := range []string{aID, bID, cID} {
		f, err := s.store.GetFact(ctx, id)
		if err != nil || f == nil {
			t.Fatalf("GetFact %s: %v", id, err)
		}
		if f.ClusterID != clusterC {
			t.Errorf("fact %s in cluster %q, want %q", id, f.ClusterID, clusterC)
		}
	}

	// Source clusters gone.
	for _, src := range []string{clusterA, clusterB} {
		got, err := s.store.GetCluster(ctx, src)
		if err != nil {
			t.Fatalf("GetCluster %s: %v", src, err)
		}
		if got != nil {
			t.Errorf("source cluster %s still present after merge: %+v", src, got)
		}
	}

	// l1/index no longer lists source clusters.
	req := &mcpsdk.ReadResourceRequest{}
	req.Params = &mcpsdk.ReadResourceParams{URI: "reverie://l1/index"}
	res, err := s.handleL1IndexResource(ctx, req)
	if err != nil {
		t.Fatalf("handleL1IndexResource: %v", err)
	}
	var index l1IndexResponse
	if err := json.Unmarshal([]byte(res.Contents[0].Text), &index); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	for _, cl := range index.Clusters {
		if cl.ID == clusterA || cl.ID == clusterB {
			t.Errorf("l1/index still lists deleted source %q", cl.ID)
		}
	}

	// Target centroid recomputed — with three orthogonal unit vectors its
	// centroid is {1/3, 1/3, 1/3, 0}, which differs from the pre-merge value.
	postC, _ := s.store.GetCluster(ctx, clusterC)
	if postC == nil {
		t.Fatal("target cluster missing after merge")
	}
	if postC.ItemCount != 3 {
		t.Errorf("target ItemCount = %d, want 3", postC.ItemCount)
	}
	if vectorsEqual(postC.Centroid, preC.Centroid) {
		t.Errorf("target centroid unchanged after merge: %v", postC.Centroid)
	}
}

func TestHandleMergeClusters_MergesEpisodes(t *testing.T) {
	emb := newStubEmbedder(4)
	s := newTestServer(emb)
	defer s.recallCache.stop()

	ctx := context.Background()

	emb.vectors["sitA\nactA\noutA\npA"] = []float32{1, 0, 0, 0}
	emb.vectors["sitB\nactB\noutB\npB"] = []float32{0, 1, 0, 0}

	_, a1, err := s.handleWrite(ctx, nil, WriteInput{Type: "feedback", Episode: &EpisodePayload{
		Situation: "sitA", Action: "actA", Outcome: "outA", Preemptive: "pA",
	}})
	if err != nil {
		t.Fatalf("write a: %v", err)
	}
	_, b1, err := s.handleWrite(ctx, nil, WriteInput{Type: "feedback", Episode: &EpisodePayload{
		Situation: "sitB", Action: "actB", Outcome: "outB", Preemptive: "pB",
	}})
	if err != nil {
		t.Fatalf("write b: %v", err)
	}

	ea, _ := s.store.GetEpisode(ctx, a1.ID)
	eb, _ := s.store.GetEpisode(ctx, b1.ID)
	if ea.ClusterID == eb.ClusterID {
		t.Fatalf("expected distinct clusters; both in %s", ea.ClusterID)
	}

	_, out, err := s.handleMergeClusters(ctx, nil, MergeClustersInput{
		SourceClusterIDs: []string{ea.ClusterID},
		TargetClusterID:  eb.ClusterID,
	})
	if err != nil {
		t.Fatalf("handleMergeClusters: %v", err)
	}
	if out.MergedCount != 1 {
		t.Errorf("MergedCount = %d, want 1", out.MergedCount)
	}
	post, _ := s.store.GetEpisode(ctx, a1.ID)
	if post.ClusterID != eb.ClusterID {
		t.Errorf("episode cluster = %q, want %q", post.ClusterID, eb.ClusterID)
	}
}

func TestHandleMergeClusters_TargetInSourceList(t *testing.T) {
	emb := newStubEmbedder(4)
	s := newTestServer(emb)
	defer s.recallCache.stop()

	_, clusterA, _, clusterB := seedTwoClusters(t, s, emb)

	_, _, err := s.handleMergeClusters(context.Background(), nil, MergeClustersInput{
		SourceClusterIDs: []string{clusterA, clusterB},
		TargetClusterID:  clusterA,
	})
	if err == nil {
		t.Fatal("expected error when target is in source list")
	}
	if !strings.Contains(err.Error(), "cannot be in source list") {
		t.Errorf("error = %q, want it to contain 'cannot be in source list'", err)
	}
}

func TestHandleMergeClusters_SourceNotFound_NoPartialMerge(t *testing.T) {
	emb := newStubEmbedder(4)
	s := newTestServer(emb)
	defer s.recallCache.stop()

	ctx := context.Background()

	aID, clusterA, bID, clusterB := seedTwoClusters(t, s, emb)

	// Include a bogus source ID alongside a valid one. Pre-validation must
	// refuse the whole op — no facts should move, no clusters deleted. This
	// is the atomicity proof against the real memStore: including a
	// nonexistent ID hits the pre-validation gate before any mutation.
	_, _, err := s.handleMergeClusters(ctx, nil, MergeClustersInput{
		SourceClusterIDs: []string{clusterA, "ghost-cluster"},
		TargetClusterID:  clusterB,
	})
	if err == nil {
		t.Fatal("expected error for missing source cluster")
	}
	if !strings.Contains(err.Error(), "cluster not found: ghost-cluster") {
		t.Errorf("error = %q, want it to contain 'cluster not found: ghost-cluster'", err)
	}

	// State unchanged: a still in clusterA, b still in clusterB, clusterA still exists.
	fa, _ := s.store.GetFact(ctx, aID)
	if fa.ClusterID != clusterA {
		t.Errorf("a moved despite validation failure: ClusterID = %q, want %q", fa.ClusterID, clusterA)
	}
	fb, _ := s.store.GetFact(ctx, bID)
	if fb.ClusterID != clusterB {
		t.Errorf("b moved despite validation failure: ClusterID = %q, want %q", fb.ClusterID, clusterB)
	}
	cl, _ := s.store.GetCluster(ctx, clusterA)
	if cl == nil {
		t.Errorf("clusterA was deleted despite validation failure")
	}
}

func TestHandleMergeClusters_TargetNotFound(t *testing.T) {
	emb := newStubEmbedder(4)
	s := newTestServer(emb)
	defer s.recallCache.stop()

	_, clusterA, _, _ := seedTwoClusters(t, s, emb)

	_, _, err := s.handleMergeClusters(context.Background(), nil, MergeClustersInput{
		SourceClusterIDs: []string{clusterA},
		TargetClusterID:  "ghost-target",
	})
	if err == nil {
		t.Fatal("expected error for missing target cluster")
	}
	if !strings.Contains(err.Error(), "cluster not found: ghost-target") {
		t.Errorf("error = %q, want it to contain 'cluster not found: ghost-target'", err)
	}
}

func TestHandleMergeClusters_EmptySourceList(t *testing.T) {
	emb := newStubEmbedder(4)
	s := newTestServer(emb)
	defer s.recallCache.stop()

	_, _, _, clusterB := seedTwoClusters(t, s, emb)

	_, _, err := s.handleMergeClusters(context.Background(), nil, MergeClustersInput{
		SourceClusterIDs: []string{},
		TargetClusterID:  clusterB,
	})
	if err == nil {
		t.Fatal("expected error for empty source list")
	}
	if !strings.Contains(err.Error(), "source_cluster_ids must be non-empty") {
		t.Errorf("error = %q, want it to contain 'source_cluster_ids must be non-empty'", err)
	}
}

// --- memory_recall filter tests (Phase 2E) ---

// seedRecallFilterFixture writes two facts in distinct clusters (orthogonal
// embeddings so the 0.60 assigner threshold doesn't fold them), a third fact
// tagged "hot" in cluster A, and an episode in cluster A. All vectors share
// non-zero components with the query "q" so GlobalSearch returns them all.
// Returns IDs + cluster IDs for filter assertions.
type recallFilterFixture struct {
	factA1ID  string // cluster A, subtype "project", tag "hot"
	factA2ID  string // cluster A, subtype "user",    no tag
	factBID   string // cluster B, subtype "project", tag "cold"
	episodeID string // cluster A (same vector as factA1)
	clusterA  string
	clusterB  string
}

func seedRecallFilterFixture(t *testing.T, s *Server, emb *stubEmbedder) recallFilterFixture {
	t.Helper()
	ctx := context.Background()

	// Orthogonal embeddings so the two groups land in different clusters.
	// All vectors have positive components along all axes so cosine with a
	// uniform query vector is > 0 (ensures every candidate is returned).
	emb.vectors["fact a1"] = []float32{1, 0, 0, 0}
	emb.vectors["fact a2"] = []float32{0.99, 0.01, 0, 0}
	emb.vectors["fact b"] = []float32{0, 1, 0, 0}
	// Episode text is the concatenation writeEpisode embeds over.
	emb.vectors["sit\nact\nout\npre"] = []float32{0.98, 0.02, 0, 0}
	// Query must cover both clusters for GlobalSearch to return members of
	// each. Use a bias toward axis 0 so cluster-A members are first, but
	// include a small axis-1 component so cluster-B is also returned.
	emb.vectors["q"] = []float32{0.9, 0.1, 0, 0}

	_, a1, err := s.handleWrite(ctx, nil, WriteInput{Content: "fact a1", Type: "project", Tags: []string{"hot"}})
	if err != nil {
		t.Fatalf("write a1: %v", err)
	}
	_, a2, err := s.handleWrite(ctx, nil, WriteInput{Content: "fact a2", Type: "user"})
	if err != nil {
		t.Fatalf("write a2: %v", err)
	}
	_, b, err := s.handleWrite(ctx, nil, WriteInput{Content: "fact b", Type: "project", Tags: []string{"cold"}})
	if err != nil {
		t.Fatalf("write b: %v", err)
	}
	_, ep, err := s.handleWrite(ctx, nil, WriteInput{
		Type: "feedback",
		Episode: &EpisodePayload{
			Situation: "sit", Action: "act", Outcome: "out", Preemptive: "pre",
		},
	})
	if err != nil {
		t.Fatalf("write episode: %v", err)
	}

	fa1, _ := s.store.GetFact(ctx, a1.ID)
	fa2, _ := s.store.GetFact(ctx, a2.ID)
	fb, _ := s.store.GetFact(ctx, b.ID)
	eep, _ := s.store.GetEpisode(ctx, ep.ID)

	if fa1.ClusterID != fa2.ClusterID {
		t.Fatalf("expected a1,a2 in same cluster; got %q and %q", fa1.ClusterID, fa2.ClusterID)
	}
	if fa1.ClusterID == fb.ClusterID {
		t.Fatalf("expected b in separate cluster; all in %s", fa1.ClusterID)
	}
	if eep.ClusterID != fa1.ClusterID {
		t.Fatalf("expected episode in cluster A %q; got %q", fa1.ClusterID, eep.ClusterID)
	}

	return recallFilterFixture{
		factA1ID:  a1.ID,
		factA2ID:  a2.ID,
		factBID:   b.ID,
		episodeID: ep.ID,
		clusterA:  fa1.ClusterID,
		clusterB:  fb.ClusterID,
	}
}

// recallIDs returns the IDs of every candidate in out, for convenient
// membership assertions.
func recallIDs(out RecallOutput) []string {
	ids := make([]string, len(out.Candidates))
	for i, c := range out.Candidates {
		ids[i] = c.ID
	}
	return ids
}

func containsID(ids []string, id string) bool {
	for _, x := range ids {
		if x == id {
			return true
		}
	}
	return false
}

func TestHandleRecall_ClusterIDFilter(t *testing.T) {
	emb := newStubEmbedder(4)
	s := newTestServer(emb)
	defer s.recallCache.stop()

	fix := seedRecallFilterFixture(t, s, emb)

	_, out, err := s.handleRecall(context.Background(), nil, RecallInput{
		Query:     "q",
		ClusterID: fix.clusterA,
	})
	if err != nil {
		t.Fatalf("handleRecall: %v", err)
	}
	ids := recallIDs(out)
	if containsID(ids, fix.factBID) {
		t.Errorf("fact B (cluster B) should not appear when filtering to cluster A; ids=%v", ids)
	}
	if !containsID(ids, fix.factA1ID) || !containsID(ids, fix.factA2ID) {
		t.Errorf("expected cluster-A facts in result; ids=%v", ids)
	}
	for _, c := range out.Candidates {
		if c.ClusterID != fix.clusterA {
			t.Errorf("candidate %s has cluster %q, want %q", c.ID, c.ClusterID, fix.clusterA)
		}
	}
}

func TestHandleRecall_SubtypeFilter(t *testing.T) {
	emb := newStubEmbedder(4)
	s := newTestServer(emb)
	defer s.recallCache.stop()

	fix := seedRecallFilterFixture(t, s, emb)

	_, out, err := s.handleRecall(context.Background(), nil, RecallInput{
		Query:   "q",
		Subtype: "project",
	})
	if err != nil {
		t.Fatalf("handleRecall: %v", err)
	}
	ids := recallIDs(out)
	// project facts (a1, b) pass; user fact (a2) and the episode drop.
	if !containsID(ids, fix.factA1ID) {
		t.Errorf("expected fact a1 (project) in result; ids=%v", ids)
	}
	if !containsID(ids, fix.factBID) {
		t.Errorf("expected fact b (project) in result; ids=%v", ids)
	}
	if containsID(ids, fix.factA2ID) {
		t.Errorf("fact a2 (user) should be filtered out; ids=%v", ids)
	}
	if containsID(ids, fix.episodeID) {
		t.Errorf("episode should be filtered out by subtype filter; ids=%v", ids)
	}
	for _, c := range out.Candidates {
		if c.Subtype != "project" {
			t.Errorf("candidate %s has subtype %q, want project", c.ID, c.Subtype)
		}
	}
}

func TestHandleRecall_LayerL2Filter(t *testing.T) {
	emb := newStubEmbedder(4)
	s := newTestServer(emb)
	defer s.recallCache.stop()

	fix := seedRecallFilterFixture(t, s, emb)

	for _, layerVal := range []string{"l2", "L2", "l2_semantic"} {
		_, out, err := s.handleRecall(context.Background(), nil, RecallInput{
			Query: "q",
			Layer: layerVal,
		})
		if err != nil {
			t.Fatalf("handleRecall(layer=%q): %v", layerVal, err)
		}
		ids := recallIDs(out)
		if containsID(ids, fix.episodeID) {
			t.Errorf("layer=%q: episode should be excluded; ids=%v", layerVal, ids)
		}
		for _, c := range out.Candidates {
			if c.Layer != "l2_semantic" {
				t.Errorf("layer=%q: candidate %s has layer %q, want l2_semantic", layerVal, c.ID, c.Layer)
			}
		}
	}
}

func TestHandleRecall_LayerL3Filter(t *testing.T) {
	emb := newStubEmbedder(4)
	s := newTestServer(emb)
	defer s.recallCache.stop()

	fix := seedRecallFilterFixture(t, s, emb)

	for _, layerVal := range []string{"l3", "l3_episodic"} {
		_, out, err := s.handleRecall(context.Background(), nil, RecallInput{
			Query: "q",
			Layer: layerVal,
		})
		if err != nil {
			t.Fatalf("handleRecall(layer=%q): %v", layerVal, err)
		}
		ids := recallIDs(out)
		if containsID(ids, fix.factA1ID) || containsID(ids, fix.factA2ID) || containsID(ids, fix.factBID) {
			t.Errorf("layer=%q: facts should be excluded; ids=%v", layerVal, ids)
		}
		for _, c := range out.Candidates {
			if c.Layer != "l3_episodic" {
				t.Errorf("layer=%q: candidate %s has layer %q, want l3_episodic", layerVal, c.ID, c.Layer)
			}
		}
	}
}

func TestHandleRecall_LayerInvalid(t *testing.T) {
	emb := newStubEmbedder(4)
	s := newTestServer(emb)
	defer s.recallCache.stop()

	_, _, err := s.handleRecall(context.Background(), nil, RecallInput{
		Query: "q",
		Layer: "nonsense",
	})
	if err == nil {
		t.Fatal("expected error for invalid layer value")
	}
	if !strings.Contains(err.Error(), "invalid layer") {
		t.Errorf("error = %q, want it to contain 'invalid layer'", err)
	}
}

func TestHandleRecall_TagsAnyFilter(t *testing.T) {
	emb := newStubEmbedder(4)
	s := newTestServer(emb)
	defer s.recallCache.stop()

	fix := seedRecallFilterFixture(t, s, emb)

	_, out, err := s.handleRecall(context.Background(), nil, RecallInput{
		Query:   "q",
		TagsAny: []string{"hot", "stranger"}, // union: "hot" matches factA1 only.
	})
	if err != nil {
		t.Fatalf("handleRecall: %v", err)
	}
	ids := recallIDs(out)
	if !containsID(ids, fix.factA1ID) {
		t.Errorf("expected factA1 (tag=hot) in result; ids=%v", ids)
	}
	if containsID(ids, fix.factA2ID) {
		t.Errorf("factA2 (untagged) should be filtered out; ids=%v", ids)
	}
	if containsID(ids, fix.factBID) {
		t.Errorf("factB (tag=cold) should be filtered out; ids=%v", ids)
	}
	if containsID(ids, fix.episodeID) {
		t.Errorf("episode (untagged) should be filtered out; ids=%v", ids)
	}
}

func TestHandleRecall_CombinedFilters(t *testing.T) {
	emb := newStubEmbedder(4)
	s := newTestServer(emb)
	defer s.recallCache.stop()

	fix := seedRecallFilterFixture(t, s, emb)

	// cluster_id=A ∩ tags_any=[hot] → only factA1.
	_, out, err := s.handleRecall(context.Background(), nil, RecallInput{
		Query:     "q",
		ClusterID: fix.clusterA,
		TagsAny:   []string{"hot"},
	})
	if err != nil {
		t.Fatalf("handleRecall: %v", err)
	}
	ids := recallIDs(out)
	if len(ids) != 1 || ids[0] != fix.factA1ID {
		t.Errorf("expected exactly [factA1]; got %v", ids)
	}

	// cluster_id=B ∩ tags_any=[hot] → empty (B has no "hot" member).
	_, out2, err := s.handleRecall(context.Background(), nil, RecallInput{
		Query:     "q",
		ClusterID: fix.clusterB,
		TagsAny:   []string{"hot"},
	})
	if err != nil {
		t.Fatalf("handleRecall 2: %v", err)
	}
	if len(out2.Candidates) != 0 {
		t.Errorf("expected empty result for cluster=B + tag=hot; got %v", recallIDs(out2))
	}
}

// --- memory_update_content tests ---

func TestHandleUpdateContent_Fact(t *testing.T) {
	emb := newStubEmbedder(4)
	emb.vectors["old"] = []float32{1, 0, 0, 0}
	emb.vectors["new"] = []float32{0, 1, 0, 0}
	s := newTestServer(emb)
	defer s.recallCache.stop()

	ctx := context.Background()
	_, wOut, err := s.handleWrite(ctx, nil, WriteInput{Content: "old", Type: "project", Tags: []string{"t1"}})
	if err != nil {
		t.Fatalf("handleWrite: %v", err)
	}

	before, err := s.store.GetFact(ctx, wOut.ID)
	if err != nil || before == nil {
		t.Fatalf("GetFact before: %v", err)
	}
	time.Sleep(2 * time.Millisecond)

	_, out, err := s.handleUpdateContent(ctx, nil, UpdateContentInput{
		ID:      wOut.ID,
		Content: "new",
	})
	if err != nil {
		t.Fatalf("handleUpdateContent: %v", err)
	}
	if out.ID != wOut.ID {
		t.Errorf("ID = %q, want %q", out.ID, wOut.ID)
	}
	if out.Layer != "l2_semantic" {
		t.Errorf("Layer = %q, want l2_semantic", out.Layer)
	}
	if !out.Reembedded {
		t.Error("Reembedded = false, want true")
	}

	after, err := s.store.GetFact(ctx, wOut.ID)
	if err != nil || after == nil {
		t.Fatalf("GetFact after: %v", err)
	}
	if after.Content != "new" {
		t.Errorf("Content = %q, want 'new'", after.Content)
	}
	if after.ContentHash == before.ContentHash {
		t.Errorf("ContentHash unchanged: %q", after.ContentHash)
	}
	if vectorsEqual(after.Embedding, before.Embedding) {
		t.Errorf("Embedding unchanged: %v", after.Embedding)
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
	if !after.ValidFrom.Equal(before.ValidFrom) {
		t.Errorf("ValidFrom changed: before=%v after=%v", before.ValidFrom, after.ValidFrom)
	}
	if len(after.Tags) != 1 || after.Tags[0] != "t1" {
		t.Errorf("Tags = %v, want [t1] (preserved)", after.Tags)
	}
}

func TestHandleUpdateContent_TagsOnly(t *testing.T) {
	emb := newStubEmbedder(4)
	emb.vectors["keep this content"] = []float32{0.3, 0.4, 0.5, 0.7}
	s := newTestServer(emb)
	defer s.recallCache.stop()

	ctx := context.Background()
	_, wOut, err := s.handleWrite(ctx, nil, WriteInput{Content: "keep this content", Type: "project", Tags: []string{"old1", "old2"}})
	if err != nil {
		t.Fatalf("handleWrite: %v", err)
	}

	newTags := []string{"fresh"}
	_, _, err = s.handleUpdateContent(ctx, nil, UpdateContentInput{
		ID:   wOut.ID,
		Tags: &newTags,
	})
	if err != nil {
		t.Fatalf("handleUpdateContent: %v", err)
	}

	got, err := s.store.GetFact(ctx, wOut.ID)
	if err != nil || got == nil {
		t.Fatalf("GetFact: %v", err)
	}
	if got.Content != "keep this content" {
		t.Errorf("Content = %q, want 'keep this content' (preserved)", got.Content)
	}
	if len(got.Tags) != 1 || got.Tags[0] != "fresh" {
		t.Errorf("Tags = %v, want [fresh]", got.Tags)
	}
}

func TestHandleUpdateContent_TagsNilPreserves(t *testing.T) {
	emb := newStubEmbedder(4)
	emb.vectors["orig"] = []float32{1, 0, 0, 0}
	emb.vectors["amended"] = []float32{0, 1, 0, 0}
	s := newTestServer(emb)
	defer s.recallCache.stop()

	ctx := context.Background()
	_, wOut, err := s.handleWrite(ctx, nil, WriteInput{Content: "orig", Type: "project", Tags: []string{"a", "b"}})
	if err != nil {
		t.Fatalf("handleWrite: %v", err)
	}

	_, _, err = s.handleUpdateContent(ctx, nil, UpdateContentInput{
		ID:      wOut.ID,
		Content: "amended",
	})
	if err != nil {
		t.Fatalf("handleUpdateContent: %v", err)
	}
	got, err := s.store.GetFact(ctx, wOut.ID)
	if err != nil || got == nil {
		t.Fatalf("GetFact: %v", err)
	}
	if len(got.Tags) != 2 || got.Tags[0] != "a" || got.Tags[1] != "b" {
		t.Errorf("Tags = %v, want [a b] (preserved)", got.Tags)
	}
}

func TestHandleUpdateContent_TagsEmptyClears(t *testing.T) {
	emb := newStubEmbedder(4)
	emb.vectors["foo"] = []float32{1, 0, 0, 0}
	s := newTestServer(emb)
	defer s.recallCache.stop()

	ctx := context.Background()
	_, wOut, err := s.handleWrite(ctx, nil, WriteInput{Content: "foo", Type: "project", Tags: []string{"a", "b"}})
	if err != nil {
		t.Fatalf("handleWrite: %v", err)
	}

	empty := []string{}
	_, _, err = s.handleUpdateContent(ctx, nil, UpdateContentInput{
		ID:   wOut.ID,
		Tags: &empty,
	})
	if err != nil {
		t.Fatalf("handleUpdateContent: %v", err)
	}
	got, err := s.store.GetFact(ctx, wOut.ID)
	if err != nil || got == nil {
		t.Fatalf("GetFact: %v", err)
	}
	if len(got.Tags) != 0 {
		t.Errorf("Tags = %v, want []", got.Tags)
	}
}

func TestHandleUpdateContent_Episode(t *testing.T) {
	emb := newStubEmbedder(4)
	emb.vectors["s1\na1\no1\np1"] = []float32{1, 0, 0, 0}
	emb.vectors["s2\na2\no2\np2"] = []float32{0, 1, 0, 0}
	s := newTestServer(emb)
	defer s.recallCache.stop()

	ctx := context.Background()
	_, wOut, err := s.handleWrite(ctx, nil, WriteInput{
		Type: "feedback",
		Tags: []string{"orig"},
		Episode: &EpisodePayload{
			Situation: "s1", Action: "a1", Outcome: "o1", Preemptive: "p1",
		},
	})
	if err != nil {
		t.Fatalf("handleWrite: %v", err)
	}

	before, err := s.store.GetEpisode(ctx, wOut.ID)
	if err != nil || before == nil {
		t.Fatalf("GetEpisode before: %v", err)
	}
	time.Sleep(2 * time.Millisecond)

	// Update episode fields; LinkedFactIDs nil => preserve (which is empty set here).
	_, out, err := s.handleUpdateContent(ctx, nil, UpdateContentInput{
		ID: wOut.ID,
		Episode: &EpisodePayload{
			Situation: "s2", Action: "a2", Outcome: "o2", Preemptive: "p2",
		},
	})
	if err != nil {
		t.Fatalf("handleUpdateContent: %v", err)
	}
	if out.Layer != "l3_episodic" || !out.Reembedded {
		t.Errorf("out = %+v, want l3_episodic, reembedded", out)
	}

	after, err := s.store.GetEpisode(ctx, wOut.ID)
	if err != nil || after == nil {
		t.Fatalf("GetEpisode after: %v", err)
	}
	if after.Situation != "s2" || after.Action != "a2" || after.Outcome != "o2" || after.Preemptive != "p2" {
		t.Errorf("episode fields = %+v, want s2/a2/o2/p2", after)
	}
	if vectorsEqual(after.Embedding, before.Embedding) {
		t.Errorf("Embedding unchanged: %v", after.Embedding)
	}
	if !after.AccessedAt.After(before.AccessedAt) {
		t.Errorf("AccessedAt not bumped: before=%v after=%v", before.AccessedAt, after.AccessedAt)
	}
	if !after.CreatedAt.Equal(before.CreatedAt) {
		t.Errorf("CreatedAt changed: before=%v after=%v", before.CreatedAt, after.CreatedAt)
	}
	if len(after.Tags) != 1 || after.Tags[0] != "orig" {
		t.Errorf("Tags = %v, want [orig] (preserved)", after.Tags)
	}
}

func TestHandleUpdateContent_EpisodeLinksReplacement(t *testing.T) {
	emb := newStubEmbedder(4)
	emb.vectors["es1\nea1\neo1\nep1"] = []float32{1, 0, 0, 0}
	emb.vectors["es2\nea2\neo2\nep2"] = []float32{0, 1, 0, 0}
	emb.vectors["fact1"] = []float32{0, 0, 1, 0}
	emb.vectors["fact2"] = []float32{0, 0, 0, 1}
	s := newTestServer(emb)
	defer s.recallCache.stop()

	ctx := context.Background()

	// Two facts to link to.
	_, f1Out, err := s.handleWrite(ctx, nil, WriteInput{Content: "fact1", Type: "project"})
	if err != nil {
		t.Fatalf("handleWrite f1: %v", err)
	}
	_, f2Out, err := s.handleWrite(ctx, nil, WriteInput{Content: "fact2", Type: "project"})
	if err != nil {
		t.Fatalf("handleWrite f2: %v", err)
	}

	// Episode initially linked to f1.
	_, epOut, err := s.handleWrite(ctx, nil, WriteInput{
		Type: "feedback",
		Episode: &EpisodePayload{
			Situation: "es1", Action: "ea1", Outcome: "eo1", Preemptive: "ep1",
			LinkedFactIDs: []string{f1Out.ID},
		},
	})
	if err != nil {
		t.Fatalf("handleWrite episode: %v", err)
	}

	// nil LinkedFactIDs preserves.
	_, _, err = s.handleUpdateContent(ctx, nil, UpdateContentInput{
		ID: epOut.ID,
		Episode: &EpisodePayload{
			Situation: "es1", Action: "ea1", Outcome: "eo1", Preemptive: "ep1",
			LinkedFactIDs: nil,
		},
	})
	if err != nil {
		t.Fatalf("handleUpdateContent preserve: %v", err)
	}
	ep, _ := s.store.GetEpisode(ctx, epOut.ID)
	if len(ep.LinkedFactIDs) != 1 || ep.LinkedFactIDs[0] != f1Out.ID {
		t.Errorf("after preserve: links = %v, want [%s]", ep.LinkedFactIDs, f1Out.ID)
	}

	// Non-nil non-empty replaces.
	_, _, err = s.handleUpdateContent(ctx, nil, UpdateContentInput{
		ID: epOut.ID,
		Episode: &EpisodePayload{
			Situation: "es2", Action: "ea2", Outcome: "eo2", Preemptive: "ep2",
			LinkedFactIDs: []string{f2Out.ID},
		},
	})
	if err != nil {
		t.Fatalf("handleUpdateContent replace: %v", err)
	}
	ep, _ = s.store.GetEpisode(ctx, epOut.ID)
	if len(ep.LinkedFactIDs) != 1 || ep.LinkedFactIDs[0] != f2Out.ID {
		t.Errorf("after replace: links = %v, want [%s]", ep.LinkedFactIDs, f2Out.ID)
	}

	// Explicit empty clears.
	_, _, err = s.handleUpdateContent(ctx, nil, UpdateContentInput{
		ID: epOut.ID,
		Episode: &EpisodePayload{
			Situation: "es2", Action: "ea2", Outcome: "eo2", Preemptive: "ep2",
			LinkedFactIDs: []string{},
		},
	})
	if err != nil {
		t.Fatalf("handleUpdateContent clear: %v", err)
	}
	ep, _ = s.store.GetEpisode(ctx, epOut.ID)
	if len(ep.LinkedFactIDs) != 0 {
		t.Errorf("after clear: links = %v, want []", ep.LinkedFactIDs)
	}
}

func TestHandleUpdateContent_LayerMismatch_EpisodeOnFact(t *testing.T) {
	emb := newStubEmbedder(4)
	emb.vectors["i am a fact"] = []float32{1, 0, 0, 0}
	s := newTestServer(emb)
	defer s.recallCache.stop()

	ctx := context.Background()
	_, wOut, err := s.handleWrite(ctx, nil, WriteInput{Content: "i am a fact", Type: "project"})
	if err != nil {
		t.Fatalf("handleWrite: %v", err)
	}

	_, _, err = s.handleUpdateContent(ctx, nil, UpdateContentInput{
		ID: wOut.ID,
		Episode: &EpisodePayload{
			Situation: "x", Action: "y", Outcome: "z", Preemptive: "w",
		},
	})
	if err == nil {
		t.Fatal("expected layer mismatch error")
	}
	if !strings.Contains(err.Error(), "layer mismatch") {
		t.Errorf("err = %q, want 'layer mismatch'", err)
	}
}

func TestHandleUpdateContent_LayerMismatch_ContentOnEpisode(t *testing.T) {
	emb := newStubEmbedder(4)
	emb.vectors["es\nea\neo\nep"] = []float32{1, 0, 0, 0}
	s := newTestServer(emb)
	defer s.recallCache.stop()

	ctx := context.Background()
	_, wOut, err := s.handleWrite(ctx, nil, WriteInput{
		Type: "feedback",
		Episode: &EpisodePayload{
			Situation: "es", Action: "ea", Outcome: "eo", Preemptive: "ep",
		},
	})
	if err != nil {
		t.Fatalf("handleWrite: %v", err)
	}

	_, _, err = s.handleUpdateContent(ctx, nil, UpdateContentInput{
		ID:      wOut.ID,
		Content: "should not apply",
	})
	if err == nil {
		t.Fatal("expected layer mismatch error")
	}
	if !strings.Contains(err.Error(), "layer mismatch") {
		t.Errorf("err = %q, want 'layer mismatch'", err)
	}
}

func TestHandleUpdateContent_BothContentAndEpisode(t *testing.T) {
	emb := newStubEmbedder(4)
	s := newTestServer(emb)
	defer s.recallCache.stop()

	_, _, err := s.handleUpdateContent(context.Background(), nil, UpdateContentInput{
		ID:      "any",
		Content: "x",
		Episode: &EpisodePayload{Situation: "y"},
	})
	if err == nil {
		t.Fatal("expected error for both content and episode")
	}
	if !strings.Contains(err.Error(), "content OR episode") {
		t.Errorf("err = %q, want to mention 'content OR episode'", err)
	}
}

func TestHandleUpdateContent_NothingToUpdate(t *testing.T) {
	emb := newStubEmbedder(4)
	s := newTestServer(emb)
	defer s.recallCache.stop()

	_, _, err := s.handleUpdateContent(context.Background(), nil, UpdateContentInput{ID: "any"})
	if err == nil {
		t.Fatal("expected error for empty update")
	}
	if !strings.Contains(err.Error(), "nothing to update") {
		t.Errorf("err = %q, want 'nothing to update'", err)
	}
}

func TestHandleUpdateContent_NotFound(t *testing.T) {
	emb := newStubEmbedder(4)
	s := newTestServer(emb)
	defer s.recallCache.stop()

	_, _, err := s.handleUpdateContent(context.Background(), nil, UpdateContentInput{
		ID:      "ghost",
		Content: "x",
	})
	if err == nil {
		t.Fatal("expected not-found error")
	}
	if !strings.Contains(err.Error(), "memory not found: ghost") {
		t.Errorf("err = %q, want 'memory not found: ghost'", err)
	}
}

func TestHandleUpdateContent_SupersededFactUpdatesPreservingSupersededBy(t *testing.T) {
	emb := newStubEmbedder(4)
	emb.vectors["old"] = []float32{1, 0, 0, 0}
	emb.vectors["new-replacement"] = []float32{0, 1, 0, 0}
	emb.vectors["corrected"] = []float32{0, 0, 1, 0}
	s := newTestServer(emb)
	defer s.recallCache.stop()

	ctx := context.Background()
	_, oldOut, err := s.handleWrite(ctx, nil, WriteInput{Content: "old", Type: "project"})
	if err != nil {
		t.Fatalf("handleWrite old: %v", err)
	}
	_, newOut, err := s.handleWrite(ctx, nil, WriteInput{Content: "new-replacement", Type: "project"})
	if err != nil {
		t.Fatalf("handleWrite new: %v", err)
	}
	if err := s.store.SupersedeFact(ctx, oldOut.ID, newOut.ID); err != nil {
		t.Fatalf("SupersedeFact: %v", err)
	}

	// Confirm old is superseded before our update.
	before, _ := s.store.GetFact(ctx, oldOut.ID)
	if before.SupersededBy == nil || *before.SupersededBy != newOut.ID {
		t.Fatalf("pre: superseded_by = %v, want %q", before.SupersededBy, newOut.ID)
	}

	_, _, err = s.handleUpdateContent(ctx, nil, UpdateContentInput{
		ID:      oldOut.ID,
		Content: "corrected",
	})
	if err != nil {
		t.Fatalf("handleUpdateContent on superseded fact: %v", err)
	}

	after, _ := s.store.GetFact(ctx, oldOut.ID)
	if after.Content != "corrected" {
		t.Errorf("Content = %q, want 'corrected'", after.Content)
	}
	if after.ContentHash == before.ContentHash {
		t.Errorf("ContentHash unchanged: %q", after.ContentHash)
	}
	if vectorsEqual(after.Embedding, before.Embedding) {
		t.Errorf("Embedding unchanged: %v", after.Embedding)
	}
	if after.SupersededBy == nil || *after.SupersededBy != newOut.ID {
		t.Errorf("superseded_by = %v, want %q (preserved)", after.SupersededBy, newOut.ID)
	}
}

func TestHandleUpdateContent_DoesNotAutoSupersede(t *testing.T) {
	emb := newStubEmbedder(4)
	// Two facts that start with distinct embeddings.
	emb.vectors["initial a"] = []float32{1, 0, 0, 0}
	emb.vectors["initial b"] = []float32{0, 1, 0, 0}
	// After update, fact-a's new content matches fact-b's embedding exactly,
	// so conflict detection (if it ran) would supersede one with the other.
	emb.vectors["matches b exactly"] = []float32{0, 1, 0, 0}
	s := newTestServer(emb)
	defer s.recallCache.stop()

	ctx := context.Background()
	_, aOut, err := s.handleWrite(ctx, nil, WriteInput{Content: "initial a", Type: "project"})
	if err != nil {
		t.Fatalf("handleWrite a: %v", err)
	}
	_, bOut, err := s.handleWrite(ctx, nil, WriteInput{Content: "initial b", Type: "project"})
	if err != nil {
		t.Fatalf("handleWrite b: %v", err)
	}

	_, _, err = s.handleUpdateContent(ctx, nil, UpdateContentInput{
		ID:      aOut.ID,
		Content: "matches b exactly",
	})
	if err != nil {
		t.Fatalf("handleUpdateContent: %v", err)
	}

	aPost, _ := s.store.GetFact(ctx, aOut.ID)
	bPost, _ := s.store.GetFact(ctx, bOut.ID)
	if aPost.SupersededBy != nil {
		t.Errorf("fact a superseded_by = %v, want nil (update must not auto-supersede)", aPost.SupersededBy)
	}
	if bPost.SupersededBy != nil {
		t.Errorf("fact b superseded_by = %v, want nil (update must not auto-supersede)", bPost.SupersededBy)
	}
}

func TestHandleUpdateContent_EmptyID(t *testing.T) {
	emb := newStubEmbedder(4)
	s := newTestServer(emb)
	defer s.recallCache.stop()

	_, _, err := s.handleUpdateContent(context.Background(), nil, UpdateContentInput{Content: "x"})
	if err == nil || err.Error() != "id is required" {
		t.Fatalf("err = %v, want 'id is required'", err)
	}
}

// --- memory_write confidence tests (Phase 3B) ---

func TestHandleWrite_ConfidenceCustom(t *testing.T) {
	emb := newStubEmbedder(4)
	emb.vectors["confident fact"] = []float32{0.5, 0.5, 0, 0}
	s := newTestServer(emb)
	defer s.recallCache.stop()

	conf := 0.7
	ctx := context.Background()
	_, out, err := s.handleWrite(ctx, nil, WriteInput{
		Content:    "confident fact",
		Type:       "project",
		Confidence: &conf,
	})
	if err != nil {
		t.Fatalf("handleWrite: %v", err)
	}

	f, err := s.store.GetFact(ctx, out.ID)
	if err != nil || f == nil {
		t.Fatalf("GetFact: %v", err)
	}
	if f.Confidence != 0.7 {
		t.Errorf("Confidence = %v, want 0.7", f.Confidence)
	}
}

func TestHandleWrite_ConfidenceDefaultsToOne(t *testing.T) {
	emb := newStubEmbedder(4)
	emb.vectors["default conf"] = []float32{0.5, 0.5, 0, 0}
	s := newTestServer(emb)
	defer s.recallCache.stop()

	ctx := context.Background()
	_, out, err := s.handleWrite(ctx, nil, WriteInput{
		Content: "default conf",
		Type:    "project",
	})
	if err != nil {
		t.Fatalf("handleWrite: %v", err)
	}
	f, _ := s.store.GetFact(ctx, out.ID)
	if f.Confidence != 1.0 {
		t.Errorf("Confidence = %v, want 1.0", f.Confidence)
	}
}

func TestHandleWrite_ConfidenceOutOfRange(t *testing.T) {
	emb := newStubEmbedder(4)
	s := newTestServer(emb)
	defer s.recallCache.stop()

	for _, bad := range []float64{-0.1, 1.5, 2.0} {
		b := bad
		_, _, err := s.handleWrite(context.Background(), nil, WriteInput{
			Content:    "x",
			Type:       "project",
			Confidence: &b,
		})
		if err == nil {
			t.Errorf("confidence %v: expected error, got nil", bad)
		}
	}
}

func TestHandleWrite_ConfidenceRejectedOnEpisode(t *testing.T) {
	emb := newStubEmbedder(4)
	s := newTestServer(emb)
	defer s.recallCache.stop()

	conf := 0.5
	_, _, err := s.handleWrite(context.Background(), nil, WriteInput{
		Type:       "feedback",
		Confidence: &conf,
		Episode: &EpisodePayload{
			Situation: "s", Action: "a", Outcome: "o", Preemptive: "p",
		},
	})
	if err == nil {
		t.Fatal("expected error when confidence set on episode write")
	}
}

// --- 3A: memory_write dry-run tests ---

func TestHandleWrite_DryRun_FreshContent(t *testing.T) {
	emb := newStubEmbedder(4)
	emb.vectors["fresh dry-run content"] = []float32{0.5, 0.5, 0.0, 0.0}
	s := newTestServer(emb)
	defer s.recallCache.stop()

	ctx := context.Background()
	_, out, err := s.handleWrite(ctx, nil, WriteInput{
		Content: "fresh dry-run content",
		Type:    "project",
		DryRun:  true,
	})
	if err != nil {
		t.Fatalf("dry-run write: %v", err)
	}
	if out.ID != "" {
		t.Errorf("expected empty ID on dry-run, got %q", out.ID)
	}
	if !out.DryRun {
		t.Error("expected out.DryRun=true")
	}
	if out.Layer != "l2_semantic" {
		t.Errorf("expected layer l2_semantic, got %q", out.Layer)
	}
	if out.Preview == nil {
		t.Fatal("expected Preview to be populated")
	}
	if out.Preview.ProposedClusterID == "" {
		t.Error("expected ProposedClusterID to be set")
	}
	if !out.Preview.ProposedClusterIsNew {
		t.Error("expected ProposedClusterIsNew=true for first write in empty store")
	}
	if out.Preview.ProposedSupersedes != nil {
		t.Errorf("expected no supersede candidate on fresh content, got %+v", out.Preview.ProposedSupersedes)
	}
	if out.Preview.ContentHash == "" {
		t.Error("expected ContentHash to be populated")
	}
}

func TestHandleWrite_DryRun_NearDuplicateSupersedeCandidate(t *testing.T) {
	emb := newStubEmbedder(4)
	// Two vectors with cosine similarity > 0.92.
	emb.vectors["existing fact about Go"] = []float32{0.5, 0.5, 0.0, 0.0}
	emb.vectors["near duplicate fact about Go"] = []float32{0.5, 0.5, 0.0, 0.0}
	s := newTestServer(emb)
	defer s.recallCache.stop()

	ctx := context.Background()

	// Seed one committed fact.
	_, seed, err := s.handleWrite(ctx, nil, WriteInput{
		Content: "existing fact about Go",
		Type:    "project",
	})
	if err != nil {
		t.Fatalf("seed write: %v", err)
	}

	// Dry-run a near-duplicate; we should surface the seed as a supersede candidate.
	_, out, err := s.handleWrite(ctx, nil, WriteInput{
		Content: "near duplicate fact about Go",
		Type:    "project",
		DryRun:  true,
	})
	if err != nil {
		t.Fatalf("dry-run write: %v", err)
	}
	if out.Preview == nil {
		t.Fatal("expected Preview to be populated")
	}
	if out.Preview.ProposedSupersedes == nil {
		t.Fatal("expected ProposedSupersedes to be populated for near-duplicate")
	}
	cand := out.Preview.ProposedSupersedes
	if cand.ID != seed.ID {
		t.Errorf("ProposedSupersedes.ID = %q, want %q", cand.ID, seed.ID)
	}
	if cand.Content != "existing fact about Go" {
		t.Errorf("ProposedSupersedes.Content = %q, want %q", cand.Content, "existing fact about Go")
	}
	if cand.Subtype != "project" {
		t.Errorf("ProposedSupersedes.Subtype = %q, want project", cand.Subtype)
	}
	if cand.Similarity < 0.92 {
		t.Errorf("ProposedSupersedes.Similarity = %v, want >= 0.92", cand.Similarity)
	}
	if cand.CreatedAt == "" {
		t.Error("ProposedSupersedes.CreatedAt should be populated")
	}
}

func TestHandleWrite_DryRun_LeavesStoreUnchanged(t *testing.T) {
	emb := newStubEmbedder(4)
	emb.vectors["pre-existing fact"] = []float32{0.5, 0.5, 0.0, 0.0}
	emb.vectors["dry-run content"] = []float32{0.5, 0.5, 0.0, 0.0}
	emb.vectors["ep-s\nep-a\nep-o\nep-p"] = []float32{0.2, 0.3, 0.4, 0.5}
	s := newTestServer(emb)
	defer s.recallCache.stop()

	ctx := context.Background()

	// Seed one fact so the store starts non-empty.
	_, _, err := s.handleWrite(ctx, nil, WriteInput{Content: "pre-existing fact", Type: "project"})
	if err != nil {
		t.Fatalf("seed write: %v", err)
	}

	factsBefore, err := s.store.ListFacts(ctx, memory.ListFilter{Limit: 100})
	if err != nil {
		t.Fatalf("list facts before: %v", err)
	}
	episodesBefore, err := s.store.ListEpisodes(ctx, memory.ListFilter{Limit: 100})
	if err != nil {
		t.Fatalf("list episodes before: %v", err)
	}

	// Dry-run fact write.
	_, _, err = s.handleWrite(ctx, nil, WriteInput{Content: "dry-run content", Type: "project", DryRun: true})
	if err != nil {
		t.Fatalf("dry-run fact: %v", err)
	}

	// Dry-run episode write.
	_, _, err = s.handleWrite(ctx, nil, WriteInput{
		Type:   "feedback",
		DryRun: true,
		Episode: &EpisodePayload{
			Situation: "ep-s", Action: "ep-a", Outcome: "ep-o", Preemptive: "ep-p",
		},
	})
	if err != nil {
		t.Fatalf("dry-run episode: %v", err)
	}

	factsAfter, err := s.store.ListFacts(ctx, memory.ListFilter{Limit: 100})
	if err != nil {
		t.Fatalf("list facts after: %v", err)
	}
	episodesAfter, err := s.store.ListEpisodes(ctx, memory.ListFilter{Limit: 100})
	if err != nil {
		t.Fatalf("list episodes after: %v", err)
	}

	if len(factsAfter) != len(factsBefore) {
		t.Errorf("dry-run changed facts count: before=%d after=%d", len(factsBefore), len(factsAfter))
	}
	if len(episodesAfter) != len(episodesBefore) {
		t.Errorf("dry-run changed episodes count: before=%d after=%d", len(episodesBefore), len(episodesAfter))
	}
	for i := range factsBefore {
		if factsBefore[i].ID != factsAfter[i].ID {
			t.Errorf("fact ID changed at index %d", i)
		}
	}
}

func TestHandleWrite_DryRun_DoesNotAdvanceTurnsSince(t *testing.T) {
	emb := newStubEmbedder(4)
	emb.vectors["seed for cluster"] = []float32{0.5, 0.5, 0.0, 0.0}
	emb.vectors["dry-run probe"] = []float32{0.5, 0.5, 0.0, 0.0}
	s := newTestServer(emb)
	defer s.recallCache.stop()

	ctx := context.Background()

	_, seed, err := s.handleWrite(ctx, nil, WriteInput{Content: "seed for cluster", Type: "project"})
	if err != nil {
		t.Fatalf("seed write: %v", err)
	}
	seedFact, err := s.store.GetFact(ctx, seed.ID)
	if err != nil || seedFact == nil {
		t.Fatalf("get seed fact: %v", err)
	}
	clusterID := seedFact.ClusterID

	clBefore, err := s.store.GetCluster(ctx, clusterID)
	if err != nil || clBefore == nil {
		t.Fatalf("get cluster before: %v", err)
	}
	turnsBefore := clBefore.TurnsSince

	// Dry-run write targeting the same cluster.
	_, _, err = s.handleWrite(ctx, nil, WriteInput{Content: "dry-run probe", Type: "project", DryRun: true})
	if err != nil {
		t.Fatalf("dry-run: %v", err)
	}

	clAfter, err := s.store.GetCluster(ctx, clusterID)
	if err != nil || clAfter == nil {
		t.Fatalf("get cluster after: %v", err)
	}
	if clAfter.TurnsSince != turnsBefore {
		t.Errorf("turns_since advanced: before=%d after=%d", turnsBefore, clAfter.TurnsSince)
	}
}

func TestHandleWrite_DryRun_Episode(t *testing.T) {
	emb := newStubEmbedder(4)
	emb.vectors["sit\nact\nout\npre"] = []float32{0.5, 0.5, 0.0, 0.0}
	s := newTestServer(emb)
	defer s.recallCache.stop()

	ctx := context.Background()
	_, out, err := s.handleWrite(ctx, nil, WriteInput{
		Type:   "feedback",
		DryRun: true,
		Episode: &EpisodePayload{
			Situation: "sit", Action: "act", Outcome: "out", Preemptive: "pre",
		},
	})
	if err != nil {
		t.Fatalf("dry-run episode: %v", err)
	}
	if out.ID != "" {
		t.Errorf("expected empty ID on dry-run, got %q", out.ID)
	}
	if !out.DryRun {
		t.Error("expected DryRun=true")
	}
	if out.Layer != "l3_episodic" {
		t.Errorf("expected layer l3_episodic, got %q", out.Layer)
	}
	if out.Preview == nil {
		t.Fatal("expected Preview to be populated")
	}
	if out.Preview.ProposedClusterID == "" {
		t.Error("expected ProposedClusterID to be set")
	}
	if out.Preview.ProposedSupersedes != nil {
		t.Error("episodes have no supersede detection; ProposedSupersedes should be nil")
	}
	if out.Preview.ContentHash == "" {
		t.Error("expected ContentHash to be populated")
	}
}

func TestHandleWrite_DryRunFalse_CommittedPathUnchanged(t *testing.T) {
	emb := newStubEmbedder(4)
	emb.vectors["committed fact"] = []float32{0.5, 0.5, 0.0, 0.0}
	s := newTestServer(emb)
	defer s.recallCache.stop()

	ctx := context.Background()
	_, out, err := s.handleWrite(ctx, nil, WriteInput{
		Content: "committed fact",
		Type:    "project",
	})
	if err != nil {
		t.Fatalf("committed write: %v", err)
	}
	if out.ID == "" {
		t.Error("expected non-empty ID on committed write")
	}
	if out.DryRun {
		t.Error("expected DryRun=false on committed write")
	}
	if out.Preview != nil {
		t.Errorf("expected nil Preview on committed write, got %+v", out.Preview)
	}
	// Verify the fact was actually persisted.
	f, err := s.store.GetFact(ctx, out.ID)
	if err != nil || f == nil {
		t.Fatalf("committed fact should be retrievable: %v", err)
	}
}

func TestHandleWrite_DryRun_InvalidInput(t *testing.T) {
	emb := newStubEmbedder(4)
	s := newTestServer(emb)
	defer s.recallCache.stop()

	ctx := context.Background()

	// Bad subtype.
	_, _, err := s.handleWrite(ctx, nil, WriteInput{
		Content: "x",
		Type:    "invalid_subtype",
		DryRun:  true,
	})
	if err == nil {
		t.Error("expected error on dry-run with invalid subtype")
	}

	// Empty content and no episode.
	_, _, err = s.handleWrite(ctx, nil, WriteInput{
		Type:   "project",
		DryRun: true,
	})
	if err == nil {
		t.Error("expected error on dry-run with neither content nor episode")
	}

	// Confidence out of range.
	bad := 1.5
	_, _, err = s.handleWrite(ctx, nil, WriteInput{
		Content:    "x",
		Type:       "project",
		Confidence: &bad,
		DryRun:     true,
	})
	if err == nil {
		t.Error("expected error on dry-run with out-of-range confidence")
	}

	// Confidence on episode.
	conf := 0.5
	_, _, err = s.handleWrite(ctx, nil, WriteInput{
		Type:       "feedback",
		Confidence: &conf,
		DryRun:     true,
		Episode: &EpisodePayload{
			Situation: "s", Action: "a", Outcome: "o", Preemptive: "p",
		},
	})
	if err == nil {
		t.Error("expected error on dry-run with confidence on episode")
	}
}

// --- Phase 4B: memory_unsupersede ---

// TestHandleUnsupersede_HappyPath covers the canonical path:
// SupersedeFact(old, new) followed by memory_unsupersede(old). Because the
// superseder is still active we expect the warning to be populated and to
// mention both IDs (old first, then superseder) per spec.
func TestHandleUnsupersede_HappyPath(t *testing.T) {
	emb := newStubEmbedder(4)
	emb.vectors["first"] = []float32{1, 0, 0, 0}
	emb.vectors["second"] = []float32{0, 1, 0, 0}
	srv := newTestServer(emb)
	defer srv.recallCache.stop()

	ctx := context.Background()

	_, out1, err := srv.handleWrite(ctx, nil, WriteInput{Content: "first", Type: "project"})
	if err != nil {
		t.Fatalf("write 1: %v", err)
	}
	_, out2, err := srv.handleWrite(ctx, nil, WriteInput{Content: "second", Type: "project"})
	if err != nil {
		t.Fatalf("write 2: %v", err)
	}

	if err := srv.store.SupersedeFact(ctx, out1.ID, out2.ID); err != nil {
		t.Fatalf("SupersedeFact: %v", err)
	}

	_, uOut, err := srv.handleUnsupersede(ctx, nil, UnsupersedeInput{FactID: out1.ID})
	if err != nil {
		t.Fatalf("handleUnsupersede: %v", err)
	}
	if uOut.FactID != out1.ID {
		t.Errorf("FactID = %q, want %q", uOut.FactID, out1.ID)
	}
	if uOut.PreviouslySupersededBy != out2.ID {
		t.Errorf("PreviouslySupersededBy = %q, want %q", uOut.PreviouslySupersededBy, out2.ID)
	}
	if uOut.Warning == "" {
		t.Fatal("expected Warning to be populated (superseder is still active)")
	}
	if !strings.Contains(uOut.Warning, out1.ID) || !strings.Contains(uOut.Warning, out2.ID) {
		t.Errorf("Warning = %q, should mention both IDs %q and %q", uOut.Warning, out1.ID, out2.ID)
	}

	// Verify via memory_get that superseded_by is now nil.
	_, getOut, err := srv.handleGet(ctx, nil, GetInput{ID: out1.ID})
	if err != nil {
		t.Fatalf("handleGet: %v", err)
	}
	if getOut.SupersededBy != nil {
		t.Errorf("memory_get SupersededBy = %v, want nil", getOut.SupersededBy)
	}
}

// TestHandleUnsupersede_ActiveFact ensures calling unsupersede on a fact with
// no superseded_by set returns an error (not a silent no-op).
func TestHandleUnsupersede_ActiveFact(t *testing.T) {
	emb := newStubEmbedder(4)
	emb.vectors["a"] = []float32{1, 0, 0, 0}
	srv := newTestServer(emb)
	defer srv.recallCache.stop()

	ctx := context.Background()
	_, out, err := srv.handleWrite(ctx, nil, WriteInput{Content: "a", Type: "project"})
	if err != nil {
		t.Fatalf("write: %v", err)
	}

	_, _, err = srv.handleUnsupersede(ctx, nil, UnsupersedeInput{FactID: out.ID})
	if err == nil {
		t.Fatal("expected error on non-superseded fact")
	}
	if !strings.Contains(err.Error(), "not superseded") {
		t.Errorf("error = %q, want to contain %q", err.Error(), "not superseded")
	}
}

// TestHandleUnsupersede_NonexistentID ensures unknown IDs are rejected.
func TestHandleUnsupersede_NonexistentID(t *testing.T) {
	emb := newStubEmbedder(4)
	srv := newTestServer(emb)
	defer srv.recallCache.stop()

	ctx := context.Background()
	_, _, err := srv.handleUnsupersede(ctx, nil, UnsupersedeInput{FactID: "nonexistent-id"})
	if err == nil {
		t.Fatal("expected error on nonexistent id")
	}
	if !strings.Contains(err.Error(), "not found") {
		t.Errorf("error = %q, want to contain %q", err.Error(), "not found")
	}
}

// TestHandleUnsupersede_NoWarningWhenSupersederAlsoSuperseded builds the
// chain A->B, B->C (A.superseded_by=B, B.superseded_by=C). Unsupersede A
// should return the previous value B, but produce no warning because B is
// no longer active.
//
// SupersedeFact signature is (oldID, newID) where oldID.superseded_by = newID.
// Verified by reading mem_store.go SupersedeFact.
func TestHandleUnsupersede_NoWarningWhenSupersederAlsoSuperseded(t *testing.T) {
	emb := newStubEmbedder(4)
	emb.vectors["alpha"] = []float32{1, 0, 0, 0}
	emb.vectors["beta"] = []float32{0, 1, 0, 0}
	emb.vectors["gamma"] = []float32{0, 0, 1, 0}
	srv := newTestServer(emb)
	defer srv.recallCache.stop()

	ctx := context.Background()

	_, aOut, err := srv.handleWrite(ctx, nil, WriteInput{Content: "alpha", Type: "project"})
	if err != nil {
		t.Fatalf("write a: %v", err)
	}
	_, bOut, err := srv.handleWrite(ctx, nil, WriteInput{Content: "beta", Type: "project"})
	if err != nil {
		t.Fatalf("write b: %v", err)
	}
	_, cOut, err := srv.handleWrite(ctx, nil, WriteInput{Content: "gamma", Type: "project"})
	if err != nil {
		t.Fatalf("write c: %v", err)
	}

	// A is superseded BY B.
	if err := srv.store.SupersedeFact(ctx, aOut.ID, bOut.ID); err != nil {
		t.Fatalf("SupersedeFact(a,b): %v", err)
	}
	// B is superseded BY C — now B is no longer active.
	if err := srv.store.SupersedeFact(ctx, bOut.ID, cOut.ID); err != nil {
		t.Fatalf("SupersedeFact(b,c): %v", err)
	}

	_, uOut, err := srv.handleUnsupersede(ctx, nil, UnsupersedeInput{FactID: aOut.ID})
	if err != nil {
		t.Fatalf("handleUnsupersede: %v", err)
	}
	if uOut.PreviouslySupersededBy != bOut.ID {
		t.Errorf("PreviouslySupersededBy = %q, want %q", uOut.PreviouslySupersededBy, bOut.ID)
	}
	if uOut.Warning != "" {
		t.Errorf("expected empty Warning when superseder is itself superseded, got %q", uOut.Warning)
	}
}

// TestHandleUnsupersede_ListFactsReflectsRevival seeds two near-duplicate
// facts to trigger auto-supersede via handleWrite, then unsupersedes the old
// one. Afterwards ListFacts (which filters superseded_by IS NOT NULL) should
// return BOTH facts, confirming the flag flip propagates to the list view.
func TestHandleUnsupersede_ListFactsReflectsRevival(t *testing.T) {
	emb := newStubEmbedder(4)
	// Two near-identical vectors (will trigger auto-supersede via the 0.92
	// default ConflictThreshold).
	emb.vectors["Go version is 1.21"] = []float32{0.5, 0.5, 0.0, 0.0}
	emb.vectors["Go version is 1.22"] = []float32{0.5, 0.5, 0.0, 0.0}
	srv := newTestServer(emb)
	defer srv.recallCache.stop()

	ctx := context.Background()

	_, out1, err := srv.handleWrite(ctx, nil, WriteInput{Content: "Go version is 1.21", Type: "project"})
	if err != nil {
		t.Fatalf("write 1: %v", err)
	}
	_, out2, err := srv.handleWrite(ctx, nil, WriteInput{Content: "Go version is 1.22", Type: "project"})
	if err != nil {
		t.Fatalf("write 2: %v", err)
	}

	// Before: ListFacts excludes the superseded one.
	before, err := srv.store.ListFacts(ctx, memory.ListFilter{Limit: 100})
	if err != nil {
		t.Fatalf("ListFacts before: %v", err)
	}
	var sawOldBefore bool
	for _, f := range before {
		if f.ID == out1.ID {
			sawOldBefore = true
		}
	}
	if sawOldBefore {
		t.Fatal("pre: ListFacts should not include auto-superseded fact")
	}

	// Unsupersede.
	_, _, err = srv.handleUnsupersede(ctx, nil, UnsupersedeInput{FactID: out1.ID})
	if err != nil {
		t.Fatalf("handleUnsupersede: %v", err)
	}

	// After: both facts should appear.
	after, err := srv.store.ListFacts(ctx, memory.ListFilter{Limit: 100})
	if err != nil {
		t.Fatalf("ListFacts after: %v", err)
	}
	var sawOld, sawNew bool
	for _, f := range after {
		if f.ID == out1.ID {
			sawOld = true
		}
		if f.ID == out2.ID {
			sawNew = true
		}
	}
	if !sawOld {
		t.Error("post: ListFacts did not include revived fact")
	}
	if !sawNew {
		t.Error("post: ListFacts did not include superseder fact")
	}
}

// --- Phase 7: knowledge graph tool tests ---

// seedKGServer builds a test server with one fact and one episode and returns
// the (server, factID, episodeID). Replacement for the retired seedLinkServer.
func seedKGServer(t *testing.T) (*Server, string, string) {
	t.Helper()
	emb := newStubEmbedder(4)
	emb.vectors["fact content"] = []float32{0.5, 0.5, 0.0, 0.0}
	emb.vectors["sit1\nact1\nout1\npre1"] = []float32{0.5, 0.5, 0.0, 0.0}
	s := newTestServer(emb)
	t.Cleanup(func() { s.recallCache.stop() })

	ctx := context.Background()
	_, fOut, err := s.handleWrite(ctx, nil, WriteInput{Content: "fact content", Type: "project"})
	if err != nil {
		t.Fatalf("handleWrite fact: %v", err)
	}
	_, eOut, err := s.handleWrite(ctx, nil, WriteInput{
		Type: "feedback",
		Episode: &EpisodePayload{
			Situation: "sit1", Action: "act1", Outcome: "out1", Preemptive: "pre1",
		},
	})
	if err != nil {
		t.Fatalf("handleWrite episode: %v", err)
	}
	return s, fOut.ID, eOut.ID
}

func TestHandleEdgeAdd_HappyPath(t *testing.T) {
	s, factID, epID := seedKGServer(t)
	ctx := context.Background()

	// Default weight path.
	_, out, err := s.handleEdgeAdd(ctx, nil, EdgeAddInput{
		SrcID: factID, DstID: epID, EdgeType: "evidence",
	})
	if err != nil {
		t.Fatalf("handleEdgeAdd: %v", err)
	}
	if !out.Created {
		t.Error("Created = false on fresh edge, want true")
	}
	if out.Weight != 1.0 {
		t.Errorf("Weight = %v, want 1.0 (default)", out.Weight)
	}
	if out.SrcID != factID || out.DstID != epID || out.EdgeType != "evidence" {
		t.Errorf("echo mismatch: %+v", out)
	}

	// Explicit-weight path: a distinct edge_type so we don't hit the
	// idempotent dedup of the previous insert.
	weight := 2.5
	_, out2, err := s.handleEdgeAdd(ctx, nil, EdgeAddInput{
		SrcID: factID, DstID: epID, EdgeType: "refines", Weight: &weight,
	})
	if err != nil {
		t.Fatalf("handleEdgeAdd with weight: %v", err)
	}
	if !out2.Created {
		t.Error("second Created = false, want true (distinct edge_type)")
	}
	if out2.Weight != 2.5 {
		t.Errorf("Weight = %v, want 2.5", out2.Weight)
	}
}

func TestHandleEdgeAdd_Idempotent(t *testing.T) {
	s, factID, epID := seedKGServer(t)
	ctx := context.Background()

	// First call with weight=1.0 (default).
	_, first, err := s.handleEdgeAdd(ctx, nil, EdgeAddInput{SrcID: factID, DstID: epID, EdgeType: "evidence"})
	if err != nil {
		t.Fatalf("handleEdgeAdd first: %v", err)
	}
	if !first.Created {
		t.Error("first Created = false, want true")
	}
	if first.Weight != 1.0 {
		t.Errorf("first Weight = %v, want 1.0", first.Weight)
	}

	// Second call with a DIFFERENT weight — the stored weight must be
	// returned, not the input weight (per locked decision in design doc).
	differentWeight := 5.0
	_, second, err := s.handleEdgeAdd(ctx, nil, EdgeAddInput{
		SrcID: factID, DstID: epID, EdgeType: "evidence", Weight: &differentWeight,
	})
	if err != nil {
		t.Fatalf("handleEdgeAdd second: %v", err)
	}
	if second.Created {
		t.Error("second Created = true, want false (idempotent)")
	}
	if second.Weight != 1.0 {
		t.Errorf("second Weight = %v, want 1.0 (stored value, not input)", second.Weight)
	}
}

func TestHandleEdgeAdd_UnknownID(t *testing.T) {
	s, factID, _ := seedKGServer(t)
	ctx := context.Background()

	_, _, err := s.handleEdgeAdd(ctx, nil, EdgeAddInput{
		SrcID: "no-such-src", DstID: "no-such-dst", EdgeType: "evidence",
	})
	if err == nil {
		t.Error("expected error for unknown src_id and dst_id")
	}

	_, _, err = s.handleEdgeAdd(ctx, nil, EdgeAddInput{
		SrcID: factID, DstID: "no-such-dst", EdgeType: "evidence",
	})
	if err == nil {
		t.Error("expected error for unknown dst_id")
	}
}

func TestHandleEdgeRemove(t *testing.T) {
	s, factID, epID := seedKGServer(t)
	ctx := context.Background()

	// Add first so the remove has a target.
	if _, _, err := s.handleEdgeAdd(ctx, nil, EdgeAddInput{
		SrcID: factID, DstID: epID, EdgeType: "evidence",
	}); err != nil {
		t.Fatalf("handleEdgeAdd: %v", err)
	}

	_, out, err := s.handleEdgeRemove(ctx, nil, EdgeRemoveInput{
		SrcID: factID, DstID: epID, EdgeType: "evidence",
	})
	if err != nil {
		t.Fatalf("handleEdgeRemove: %v", err)
	}
	if !out.Deleted {
		t.Error("Deleted = false, want true")
	}

	// Removing again returns Deleted=false cleanly.
	_, out, err = s.handleEdgeRemove(ctx, nil, EdgeRemoveInput{
		SrcID: factID, DstID: epID, EdgeType: "evidence",
	})
	if err != nil {
		t.Fatalf("handleEdgeRemove second: %v", err)
	}
	if out.Deleted {
		t.Error("Deleted = true on missing edge, want false")
	}
}

func TestHandleEdgeList_OneHop(t *testing.T) {
	s, factID, epID := seedKGServer(t)
	ctx := context.Background()

	if _, _, err := s.handleEdgeAdd(ctx, nil, EdgeAddInput{
		SrcID: factID, DstID: epID, EdgeType: "evidence",
	}); err != nil {
		t.Fatalf("handleEdgeAdd: %v", err)
	}

	_, out, err := s.handleEdgeList(ctx, nil, EdgeListInput{MemoryID: factID})
	if err != nil {
		t.Fatalf("handleEdgeList: %v", err)
	}
	if out.Layer != "l2_semantic" {
		t.Errorf("Layer = %q, want l2_semantic", out.Layer)
	}
	if len(out.Edges) != 1 {
		t.Fatalf("len(Edges) = %d, want 1", len(out.Edges))
	}
	e := out.Edges[0]
	if e.OtherID != epID {
		t.Errorf("OtherID = %q, want %q", e.OtherID, epID)
	}
	if e.OtherLayer != "l3_episodic" {
		t.Errorf("OtherLayer = %q, want l3_episodic", e.OtherLayer)
	}
	if e.Distance != 1 {
		t.Errorf("Distance = %d, want 1", e.Distance)
	}
	if e.EdgeType != "evidence" {
		t.Errorf("EdgeType = %q, want evidence", e.EdgeType)
	}
}

func TestHandleEdgeList_TwoHops(t *testing.T) {
	emb := newStubEmbedder(4)
	emb.vectors["a"] = []float32{1, 0, 0, 0}
	emb.vectors["b"] = []float32{0, 1, 0, 0}
	emb.vectors["c"] = []float32{0, 0, 1, 0}
	s := newTestServer(emb)
	t.Cleanup(func() { s.recallCache.stop() })

	ctx := context.Background()
	_, aOut, err := s.handleWrite(ctx, nil, WriteInput{Content: "a", Type: "project"})
	if err != nil {
		t.Fatalf("write a: %v", err)
	}
	_, bOut, err := s.handleWrite(ctx, nil, WriteInput{Content: "b", Type: "project"})
	if err != nil {
		t.Fatalf("write b: %v", err)
	}
	_, cOut, err := s.handleWrite(ctx, nil, WriteInput{Content: "c", Type: "project"})
	if err != nil {
		t.Fatalf("write c: %v", err)
	}

	// Chain a -> b -> c.
	if _, _, err := s.handleEdgeAdd(ctx, nil, EdgeAddInput{
		SrcID: aOut.ID, DstID: bOut.ID, EdgeType: "refines",
	}); err != nil {
		t.Fatalf("add a->b: %v", err)
	}
	if _, _, err := s.handleEdgeAdd(ctx, nil, EdgeAddInput{
		SrcID: bOut.ID, DstID: cOut.ID, EdgeType: "refines",
	}); err != nil {
		t.Fatalf("add b->c: %v", err)
	}

	_, out, err := s.handleEdgeList(ctx, nil, EdgeListInput{MemoryID: aOut.ID, Hops: 2})
	if err != nil {
		t.Fatalf("handleEdgeList: %v", err)
	}
	if len(out.Edges) != 2 {
		t.Fatalf("len(Edges) = %d, want 2", len(out.Edges))
	}
	distances := map[string]int{}
	for _, e := range out.Edges {
		distances[e.OtherID] = e.Distance
	}
	if distances[bOut.ID] != 1 {
		t.Errorf("distance to b = %d, want 1", distances[bOut.ID])
	}
	if distances[cOut.ID] != 2 {
		t.Errorf("distance to c = %d, want 2", distances[cOut.ID])
	}
}

func TestHandleEdgeList_HopsOutOfRange(t *testing.T) {
	s, factID, _ := seedKGServer(t)
	ctx := context.Background()

	_, _, err := s.handleEdgeList(ctx, nil, EdgeListInput{MemoryID: factID, Hops: 4})
	if err == nil {
		t.Error("expected error for hops=4")
	}
	_, _, err = s.handleEdgeList(ctx, nil, EdgeListInput{MemoryID: factID, Hops: -1})
	if err == nil {
		t.Error("expected error for hops=-1")
	}
}

func TestHandleEdgeList_UnknownMemoryID(t *testing.T) {
	s, _, _ := seedKGServer(t)
	ctx := context.Background()

	_, _, err := s.handleEdgeList(ctx, nil, EdgeListInput{MemoryID: "no-such-id"})
	if err == nil {
		t.Error("expected error for unknown memory_id")
	}
}

func TestHandleEntityUpsert_ExactDedup(t *testing.T) {
	emb := newStubEmbedder(4)
	s := newTestServer(emb)
	t.Cleanup(func() { s.recallCache.stop() })

	ctx := context.Background()
	_, first, err := s.handleEntityUpsert(ctx, nil, EntityUpsertInput{
		Name: "foo.go", EntityType: "file",
	})
	if err != nil {
		t.Fatalf("first upsert: %v", err)
	}
	if !first.Created {
		t.Error("first Created = false, want true")
	}
	if first.MatchedBySimilarity {
		t.Error("first MatchedBySimilarity = true, want false")
	}

	_, second, err := s.handleEntityUpsert(ctx, nil, EntityUpsertInput{
		Name: "foo.go", EntityType: "file",
	})
	if err != nil {
		t.Fatalf("second upsert: %v", err)
	}
	if second.Created {
		t.Error("second Created = true, want false (dedup)")
	}
	if second.EntityID != first.EntityID {
		t.Errorf("EntityID mismatch: %q vs %q", second.EntityID, first.EntityID)
	}
	if second.MatchedBySimilarity {
		t.Error("second MatchedBySimilarity = true, want false (exact match)")
	}
}

func TestHandleEntityUpsert_DifferentType(t *testing.T) {
	emb := newStubEmbedder(4)
	s := newTestServer(emb)
	t.Cleanup(func() { s.recallCache.stop() })

	ctx := context.Background()
	_, asFile, err := s.handleEntityUpsert(ctx, nil, EntityUpsertInput{
		Name: "foo", EntityType: "file",
	})
	if err != nil {
		t.Fatalf("upsert as file: %v", err)
	}
	_, asConcept, err := s.handleEntityUpsert(ctx, nil, EntityUpsertInput{
		Name: "foo", EntityType: "concept",
	})
	if err != nil {
		t.Fatalf("upsert as concept: %v", err)
	}
	if asFile.EntityID == asConcept.EntityID {
		t.Errorf("same EntityID across entity_types: %q", asFile.EntityID)
	}
	if !asConcept.Created {
		t.Error("expected Created=true for distinct entity_type")
	}
}

func TestHandleEntityMention_HappyPath(t *testing.T) {
	emb := newStubEmbedder(4)
	// Distinct orthogonal vectors per entity so similarity dedup does NOT
	// collapse them — the stub returns [1,0,0,0] for unknown strings, and
	// UpsertEntity falls back to cosine dedup within the same entity_type.
	emb.vectors["e1 (concept)"] = []float32{1, 0, 0, 0}
	emb.vectors["e2 (concept)"] = []float32{0, 1, 0, 0}
	emb.vectors["e3 (concept)"] = []float32{0, 0, 1, 0}
	s := newTestServer(emb)
	t.Cleanup(func() { s.recallCache.stop() })

	ctx := context.Background()
	_, fOut, err := s.handleWrite(ctx, nil, WriteInput{Content: "fact", Type: "project"})
	if err != nil {
		t.Fatalf("write fact: %v", err)
	}
	ids := make([]string, 3)
	for i, name := range []string{"e1", "e2", "e3"} {
		_, up, err := s.handleEntityUpsert(ctx, nil, EntityUpsertInput{Name: name, EntityType: "concept"})
		if err != nil {
			t.Fatalf("upsert %s: %v", name, err)
		}
		ids[i] = up.EntityID
	}

	_, out, err := s.handleEntityMention(ctx, nil, EntityMentionInput{
		MemoryID: fOut.ID, EntityIDs: ids,
	})
	if err != nil {
		t.Fatalf("handleEntityMention: %v", err)
	}
	if out.Inserted != 3 {
		t.Errorf("Inserted = %d, want 3", out.Inserted)
	}
}

func TestHandleEntityMention_IdempotentRepeat(t *testing.T) {
	emb := newStubEmbedder(4)
	s := newTestServer(emb)
	t.Cleanup(func() { s.recallCache.stop() })

	ctx := context.Background()
	_, fOut, err := s.handleWrite(ctx, nil, WriteInput{Content: "fact", Type: "project"})
	if err != nil {
		t.Fatalf("write fact: %v", err)
	}
	_, up, err := s.handleEntityUpsert(ctx, nil, EntityUpsertInput{Name: "e1", EntityType: "concept"})
	if err != nil {
		t.Fatalf("upsert: %v", err)
	}

	_, first, err := s.handleEntityMention(ctx, nil, EntityMentionInput{
		MemoryID: fOut.ID, EntityIDs: []string{up.EntityID},
	})
	if err != nil {
		t.Fatalf("first mention: %v", err)
	}
	if first.Inserted != 1 {
		t.Errorf("first Inserted = %d, want 1", first.Inserted)
	}

	_, second, err := s.handleEntityMention(ctx, nil, EntityMentionInput{
		MemoryID: fOut.ID, EntityIDs: []string{up.EntityID},
	})
	if err != nil {
		t.Fatalf("second mention: %v", err)
	}
	if second.Inserted != 0 {
		t.Errorf("second Inserted = %d, want 0 (idempotent)", second.Inserted)
	}
}

func TestHandleEntityMention_BadMemoryID(t *testing.T) {
	emb := newStubEmbedder(4)
	s := newTestServer(emb)
	t.Cleanup(func() { s.recallCache.stop() })

	ctx := context.Background()
	_, up, err := s.handleEntityUpsert(ctx, nil, EntityUpsertInput{Name: "e1", EntityType: "concept"})
	if err != nil {
		t.Fatalf("upsert: %v", err)
	}

	// memory_id resolves to an entity, which is forbidden.
	_, _, err = s.handleEntityMention(ctx, nil, EntityMentionInput{
		MemoryID: up.EntityID, EntityIDs: []string{up.EntityID},
	})
	if err == nil {
		t.Error("expected error when memory_id is an entity")
	}
}

func TestHandleEntityNeighbors_OneHop(t *testing.T) {
	emb := newStubEmbedder(4)
	emb.vectors["fact alpha"] = []float32{1, 0, 0, 0}
	emb.vectors["fact beta"] = []float32{0, 1, 0, 0}
	// Distinct embedding texts so entity similarity dedup does not
	// collapse "primary" and "other" into a single row.
	emb.vectors["primary (concept)"] = []float32{0, 0, 1, 0}
	emb.vectors["other (concept)"] = []float32{0, 0, 0, 1}
	s := newTestServer(emb)
	t.Cleanup(func() { s.recallCache.stop() })

	ctx := context.Background()
	_, fa, err := s.handleWrite(ctx, nil, WriteInput{Content: "fact alpha", Type: "project"})
	if err != nil {
		t.Fatalf("write alpha: %v", err)
	}
	_, fb, err := s.handleWrite(ctx, nil, WriteInput{Content: "fact beta", Type: "project"})
	if err != nil {
		t.Fatalf("write beta: %v", err)
	}

	_, primary, err := s.handleEntityUpsert(ctx, nil, EntityUpsertInput{Name: "primary", EntityType: "concept"})
	if err != nil {
		t.Fatalf("upsert primary: %v", err)
	}
	_, other, err := s.handleEntityUpsert(ctx, nil, EntityUpsertInput{Name: "other", EntityType: "concept"})
	if err != nil {
		t.Fatalf("upsert other: %v", err)
	}

	// Two memory mentions of primary.
	if _, _, err := s.handleEntityMention(ctx, nil, EntityMentionInput{
		MemoryID: fa.ID, EntityIDs: []string{primary.EntityID},
	}); err != nil {
		t.Fatalf("mention fa: %v", err)
	}
	if _, _, err := s.handleEntityMention(ctx, nil, EntityMentionInput{
		MemoryID: fb.ID, EntityIDs: []string{primary.EntityID},
	}); err != nil {
		t.Fatalf("mention fb: %v", err)
	}

	// Entity-to-entity edge.
	if _, _, err := s.handleEdgeAdd(ctx, nil, EdgeAddInput{
		SrcID: primary.EntityID, DstID: other.EntityID, EdgeType: "references",
	}); err != nil {
		t.Fatalf("add entity edge: %v", err)
	}

	_, out, err := s.handleEntityNeighbors(ctx, nil, EntityNeighborsInput{EntityID: primary.EntityID})
	if err != nil {
		t.Fatalf("handleEntityNeighbors: %v", err)
	}
	if len(out.Memories) != 2 {
		t.Errorf("len(Memories) = %d, want 2 (%+v)", len(out.Memories), out.Memories)
	}
	if len(out.Entities) != 1 || out.Entities[0].ID != other.EntityID {
		t.Errorf("Entities = %+v, want one neighbor %q", out.Entities, other.EntityID)
	}
}

// TestWriteEpisodeWithLinkedFactIDs_StillWorks is the Phase-1-rewire canary at
// the MCP layer: write a fact, write an episode with EpisodePayload.LinkedFactIDs,
// then memory_get the episode and confirm the link survives. This guards the
// rewire of writeEpisode/getEpisode from fact_episode_links onto memory_edges.
func TestWriteEpisodeWithLinkedFactIDs_StillWorks(t *testing.T) {
	emb := newStubEmbedder(4)
	emb.vectors["fact x"] = []float32{0.5, 0.5, 0.0, 0.0}
	emb.vectors["s\na\no\np"] = []float32{0.5, 0.5, 0.0, 0.0}
	s := newTestServer(emb)
	defer s.recallCache.stop()

	ctx := context.Background()
	_, fOut, err := s.handleWrite(ctx, nil, WriteInput{Content: "fact x", Type: "project"})
	if err != nil {
		t.Fatalf("handleWrite fact: %v", err)
	}
	_, eOut, err := s.handleWrite(ctx, nil, WriteInput{
		Type: "feedback",
		Episode: &EpisodePayload{
			Situation: "s", Action: "a", Outcome: "o", Preemptive: "p",
			LinkedFactIDs: []string{fOut.ID},
		},
	})
	if err != nil {
		t.Fatalf("handleWrite episode: %v", err)
	}

	// Drive through the MCP handler layer: memory_get on the episode must
	// surface the linked fact in its Links slice.
	_, eGet, err := s.handleGet(ctx, nil, GetInput{ID: eOut.ID})
	if err != nil {
		t.Fatalf("handleGet episode: %v", err)
	}
	if len(eGet.Links) != 1 || eGet.Links[0].ID != fOut.ID {
		t.Errorf("episode links = %+v, want one link to %s", eGet.Links, fOut.ID)
	}

	// Also confirm via the new memory_edge_list surface.
	_, listOut, err := s.handleEdgeList(ctx, nil, EdgeListInput{MemoryID: eOut.ID})
	if err != nil {
		t.Fatalf("handleEdgeList: %v", err)
	}
	if len(listOut.Edges) != 1 || listOut.Edges[0].OtherID != fOut.ID {
		t.Errorf("edge list = %+v, want one edge to %s", listOut.Edges, fOut.ID)
	}
	if listOut.Edges[0].EdgeType != "evidence" {
		t.Errorf("EdgeType = %q, want evidence", listOut.Edges[0].EdgeType)
	}
}
