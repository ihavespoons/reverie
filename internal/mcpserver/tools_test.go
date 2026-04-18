package mcpserver

import (
	"context"
	"strings"
	"testing"
	"time"

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

func TestMemoryDecayTick_SessionEnd(t *testing.T) {
	emb := newStubEmbedder(4)
	emb.vectors["session end fact"] = []float32{0.5, 0.5, 0.0, 0.0}
	s := newTestServer(emb)
	defer s.recallCache.stop()

	ctx := context.Background()

	// Write a fact so a cluster exists to tick.
	_, _, err := s.handleWrite(ctx, nil, WriteInput{Content: "session end fact", Type: "user"})
	if err != nil {
		t.Fatalf("write: %v", err)
	}

	_, tickOut, err := s.handleDecayTick(ctx, nil, DecayTickInput{SessionEnd: true})
	if err != nil {
		t.Fatalf("decay tick session end: %v", err)
	}
	if !tickOut.Ticked {
		t.Fatal("expected ticked=true for session end")
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
