package mcpserver

import (
	"context"
	"strings"
	"testing"

	"personal/reverie/internal/cluster"
	"personal/reverie/internal/config"
	"personal/reverie/internal/decay"
	"personal/reverie/internal/manager"
	"personal/reverie/internal/memory"
)

// newSessionTestServer builds a server with a pre-populated session. The
// returned session_id is ready to be passed through to any handler.
func newSessionTestServer(t *testing.T, embedder *stubEmbedder, budgetMax int) (*Server, string) {
	t.Helper()
	cfg := config.Defaults()
	if budgetMax > 0 {
		cfg.Session.BufferBudgetMax = budgetMax
	}
	store := memory.NewMemStore()
	dec := decay.NewDecayer(10.0, 0.3)
	mgr := manager.NewMemoryManager(store, dec, 0.10, 0.05)
	assigner := cluster.NewAssigner(store, 0.60, 0.5, 0.5)
	s := NewServer(store, embedder, dec, mgr, assigner, cfg, nil)
	t.Cleanup(s.recallCache.stop)

	sessID := "test-session"
	if err := store.CreateSession(context.Background(), memory.Session{ID: sessID}); err != nil {
		t.Fatalf("seed session: %v", err)
	}
	return s, sessID
}

func getSession(t *testing.T, s *Server, id string) *memory.Session {
	t.Helper()
	sess, err := s.store.GetSession(context.Background(), id)
	if err != nil {
		t.Fatalf("GetSession: %v", err)
	}
	if sess == nil {
		t.Fatalf("session %q not found", id)
	}
	return sess
}

// TestRecall_WithSessionID_BufferUpdates verifies that a session-scoped
// recall populates the buffer with the returned candidates. Uses known
// similarity vectors so the test is deterministic.
func TestRecall_WithSessionID_BufferUpdates(t *testing.T) {
	emb := newStubEmbedder(4)
	emb.vectors["find it"] = []float32{0.5, 0.5, 0.0, 0.0}
	emb.vectors["alpha"] = []float32{0.5, 0.5, 0.0, 0.0}
	emb.vectors["beta"] = []float32{0.4, 0.5, 0.1, 0.0}
	s, sessID := newSessionTestServer(t, emb, 0)

	ctx := context.Background()
	_, _, err := s.handleWrite(ctx, nil, WriteInput{Content: "alpha", Type: "project"})
	if err != nil {
		t.Fatalf("write alpha: %v", err)
	}
	_, _, err = s.handleWrite(ctx, nil, WriteInput{Content: "beta", Type: "reference"})
	if err != nil {
		t.Fatalf("write beta: %v", err)
	}

	_, recallOut, err := s.handleRecall(ctx, nil, RecallInput{Query: "find it", SessionID: sessID})
	if err != nil {
		t.Fatalf("handleRecall: %v", err)
	}
	if len(recallOut.Candidates) == 0 {
		t.Fatal("expected at least one recall candidate")
	}

	sess := getSession(t, s, sessID)
	if len(sess.WorkingMem.Buffer) != len(recallOut.Candidates) {
		t.Fatalf("buffer len = %d, want %d (one per candidate)",
			len(sess.WorkingMem.Buffer), len(recallOut.Candidates))
	}
	// Every returned candidate should appear in the buffer.
	gotIDs := map[string]bool{}
	for _, r := range sess.WorkingMem.Buffer {
		gotIDs[r.ID] = true
	}
	for _, c := range recallOut.Candidates {
		if !gotIDs[c.ID] {
			t.Errorf("candidate %s missing from session buffer", c.ID)
		}
	}
}

// TestRecall_WithoutSessionID_NoBufferTouch is the regression guard: without
// session_id, sessions must NOT be created or mutated. We seed a session
// and verify its buffer stays untouched after a recall that omits session_id.
func TestRecall_WithoutSessionID_NoBufferTouch(t *testing.T) {
	emb := newStubEmbedder(4)
	emb.vectors["find"] = []float32{0.5, 0.5, 0.0, 0.0}
	emb.vectors["alpha"] = []float32{0.5, 0.5, 0.0, 0.0}
	s, sessID := newSessionTestServer(t, emb, 0)

	ctx := context.Background()
	_, _, err := s.handleWrite(ctx, nil, WriteInput{Content: "alpha", Type: "project"})
	if err != nil {
		t.Fatalf("write: %v", err)
	}

	// Recall WITHOUT session_id.
	_, _, err = s.handleRecall(ctx, nil, RecallInput{Query: "find"})
	if err != nil {
		t.Fatalf("handleRecall: %v", err)
	}

	sess := getSession(t, s, sessID)
	if len(sess.WorkingMem.Buffer) != 0 {
		t.Errorf("session buffer should be untouched; got %d entries", len(sess.WorkingMem.Buffer))
	}
}

func TestRecall_UnknownSessionID_Errors(t *testing.T) {
	emb := newStubEmbedder(4)
	emb.vectors["q"] = []float32{0.5, 0.5, 0.0, 0.0}
	s, _ := newSessionTestServer(t, emb, 0)

	_, _, err := s.handleRecall(context.Background(), nil, RecallInput{
		Query: "q", SessionID: "not-a-real-session",
	})
	if err == nil {
		t.Fatal("expected error for unknown session_id")
	}
	if !strings.Contains(err.Error(), "session not found") {
		t.Errorf("error should mention 'session not found': %v", err)
	}
}

func TestRecall_ClosedSessionID_Errors(t *testing.T) {
	emb := newStubEmbedder(4)
	emb.vectors["q"] = []float32{0.5, 0.5, 0.0, 0.0}
	s, sessID := newSessionTestServer(t, emb, 0)

	if err := s.store.CloseSession(context.Background(), sessID); err != nil {
		t.Fatalf("CloseSession: %v", err)
	}
	_, _, err := s.handleRecall(context.Background(), nil, RecallInput{
		Query: "q", SessionID: sessID,
	})
	if err == nil {
		t.Fatal("expected error for closed session_id")
	}
	if !strings.Contains(err.Error(), "session closed") {
		t.Errorf("error should mention 'session closed': %v", err)
	}
}

// TestApplyJudgment_WithSessionID_BufferShrinks seeds a recall (which fills
// the buffer), then applies judgments that keep only a subset and verifies
// the buffer shrinks to exactly those IDs.
func TestApplyJudgment_WithSessionID_BufferShrinks(t *testing.T) {
	emb := newStubEmbedder(4)
	emb.vectors["q"] = []float32{0.5, 0.5, 0.0, 0.0}
	emb.vectors["keep"] = []float32{0.5, 0.5, 0.0, 0.0}
	emb.vectors["drop"] = []float32{0.5, 0.5, 0.0, 0.0}
	s, sessID := newSessionTestServer(t, emb, 0)

	ctx := context.Background()
	_, keepOut, err := s.handleWrite(ctx, nil, WriteInput{Content: "keep", Type: "project"})
	if err != nil {
		t.Fatalf("write keep: %v", err)
	}
	_, dropOut, err := s.handleWrite(ctx, nil, WriteInput{Content: "drop", Type: "reference"})
	if err != nil {
		t.Fatalf("write drop: %v", err)
	}

	_, recallOut, err := s.handleRecall(ctx, nil, RecallInput{Query: "q", SessionID: sessID})
	if err != nil {
		t.Fatalf("recall: %v", err)
	}
	// Sanity: both are in the buffer after recall.
	sess := getSession(t, s, sessID)
	if len(sess.WorkingMem.Buffer) < 2 {
		t.Fatalf("buffer len = %d, want >= 2 pre-judgment", len(sess.WorkingMem.Buffer))
	}

	// Round 0 OR logic: both would naturally pass (high similarity). Use
	// round 1 (AND) so gate_a actually gates the result.
	_, recallOut, err = s.handleRecall(ctx, nil, RecallInput{Query: "q", Round: 1, SessionID: sessID})
	if err != nil {
		t.Fatalf("recall round 1: %v", err)
	}

	_, judgOut, err := s.handleApplyJudgment(ctx, nil, ApplyJudgmentInput{
		RecallID:  recallOut.RecallID,
		SessionID: sessID,
		Verdicts: []Verdict{
			{MemoryID: keepOut.ID, Keep: true},
			{MemoryID: dropOut.ID, Keep: false},
		},
	})
	if err != nil {
		t.Fatalf("apply_judgment: %v", err)
	}

	// Only keepOut should remain in memories.
	if len(judgOut.Memories) != 1 || judgOut.Memories[0].ID != keepOut.ID {
		t.Fatalf("apply_judgment kept = %+v, want only %s", judgOut.Memories, keepOut.ID)
	}

	sess = getSession(t, s, sessID)
	if len(sess.WorkingMem.Buffer) != 1 {
		t.Fatalf("post-judgment buffer len = %d, want 1", len(sess.WorkingMem.Buffer))
	}
	if sess.WorkingMem.Buffer[0].ID != keepOut.ID {
		t.Errorf("post-judgment buffer[0].ID = %s, want %s", sess.WorkingMem.Buffer[0].ID, keepOut.ID)
	}
}

func TestReinforce_WithSessionID_BufferScored(t *testing.T) {
	emb := newStubEmbedder(4)
	emb.vectors["q"] = []float32{0.5, 0.5, 0.0, 0.0}
	emb.vectors["r"] = []float32{0.5, 0.5, 0.0, 0.0}
	s, sessID := newSessionTestServer(t, emb, 0)

	ctx := context.Background()
	_, writeOut, err := s.handleWrite(ctx, nil, WriteInput{Content: "r", Type: "project"})
	if err != nil {
		t.Fatalf("write: %v", err)
	}
	_, _, err = s.handleRecall(ctx, nil, RecallInput{Query: "q", SessionID: sessID})
	if err != nil {
		t.Fatalf("recall: %v", err)
	}

	// Capture the current score for writeOut.ID.
	sess := getSession(t, s, sessID)
	var before float64
	for _, r := range sess.WorkingMem.Buffer {
		if r.ID == writeOut.ID {
			before = r.Score
		}
	}

	_, _, err = s.handleReinforce(ctx, nil, ReinforceInput{
		MemoryIDs: []string{writeOut.ID},
		SessionID: sessID,
	})
	if err != nil {
		t.Fatalf("reinforce: %v", err)
	}

	sess = getSession(t, s, sessID)
	var after float64
	found := false
	for _, r := range sess.WorkingMem.Buffer {
		if r.ID == writeOut.ID {
			after = r.Score
			found = true
		}
	}
	if !found {
		t.Fatal("reinforced id missing from buffer")
	}
	if after <= before && before < 1.0 {
		t.Errorf("reinforce score did not advance: before=%v after=%v", before, after)
	}
}

// TestWrite_WithSessionID_AppendsScoreOne verifies the write-path contract:
// new memory lands in the buffer with score 1.0.
func TestWrite_WithSessionID_AppendsScoreOne(t *testing.T) {
	emb := newStubEmbedder(4)
	emb.vectors["brand new"] = []float32{0.5, 0.5, 0.0, 0.0}
	s, sessID := newSessionTestServer(t, emb, 0)

	ctx := context.Background()
	_, writeOut, err := s.handleWrite(ctx, nil, WriteInput{
		Content: "brand new", Type: "user", SessionID: sessID,
	})
	if err != nil {
		t.Fatalf("write: %v", err)
	}

	sess := getSession(t, s, sessID)
	if len(sess.WorkingMem.Buffer) != 1 {
		t.Fatalf("buffer len = %d, want 1", len(sess.WorkingMem.Buffer))
	}
	ref := sess.WorkingMem.Buffer[0]
	if ref.ID != writeOut.ID {
		t.Errorf("buffer[0].ID = %s, want %s", ref.ID, writeOut.ID)
	}
	if ref.Score != 1.0 {
		t.Errorf("write buffer entry Score = %v, want 1.0", ref.Score)
	}
}

func TestWrite_WithSessionID_EpisodeAppendsScoreOne(t *testing.T) {
	emb := newStubEmbedder(4)
	emb.vectors["s\na\no\np"] = []float32{0.5, 0.5, 0.0, 0.0}
	s, sessID := newSessionTestServer(t, emb, 0)

	ctx := context.Background()
	_, writeOut, err := s.handleWrite(ctx, nil, WriteInput{
		Type: "feedback",
		Episode: &EpisodePayload{
			Situation: "s", Action: "a", Outcome: "o", Preemptive: "p",
		},
		SessionID: sessID,
	})
	if err != nil {
		t.Fatalf("write episode: %v", err)
	}

	sess := getSession(t, s, sessID)
	if len(sess.WorkingMem.Buffer) != 1 {
		t.Fatalf("buffer len = %d, want 1", len(sess.WorkingMem.Buffer))
	}
	if sess.WorkingMem.Buffer[0].ID != writeOut.ID {
		t.Errorf("buffer[0].ID = %s, want %s", sess.WorkingMem.Buffer[0].ID, writeOut.ID)
	}
	if sess.WorkingMem.Buffer[0].Score != 1.0 {
		t.Errorf("episode buffer entry Score = %v, want 1.0", sess.WorkingMem.Buffer[0].Score)
	}
	if sess.WorkingMem.Buffer[0].Layer != memory.TypeL3Episodic {
		t.Errorf("Layer = %q, want l3_episodic", sess.WorkingMem.Buffer[0].Layer)
	}
}

// TestRecall_BudgetRespected seeds a small budget and fires enough recalls
// to overflow — the buffer should stay capped at BufferBudgetMax. Uses
// distinct subtypes per fact so the write-side conflict detection (which
// groups by subtype) doesn't silently supersede them.
func TestRecall_BudgetRespected(t *testing.T) {
	emb := newStubEmbedder(4)
	// Query matches the first axis; each fact is roughly orthogonal so
	// they land in distinct cluster candidates. All five pass gate_b
	// (similarity > threshold) for the same query vector because their
	// first component is still dominant.
	emb.vectors["query"] = []float32{1, 0, 0, 0}
	type seed struct {
		content string
		vec     []float32
		subtype string
	}
	seeds := []seed{
		{"fact_a", []float32{1.0, 0.0, 0.0, 0.0}, "user"},
		{"fact_b", []float32{0.95, 0.05, 0.0, 0.0}, "feedback"},
		{"fact_c", []float32{0.9, 0.1, 0.0, 0.0}, "project"},
		{"fact_d", []float32{0.85, 0.15, 0.0, 0.0}, "reference"},
		{"fact_e", []float32{0.8, 0.2, 0.0, 0.0}, "user"},
	}
	for _, sd := range seeds {
		emb.vectors[sd.content] = sd.vec
	}

	s, sessID := newSessionTestServer(t, emb, 3)

	ctx := context.Background()
	// Nudge the conflict threshold high enough that the same-subtype seeds
	// (fact_a and fact_e) won't supersede each other.
	s.cfg.Memory.ConflictThreshold = 0.999
	for _, sd := range seeds {
		if _, _, err := s.handleWrite(ctx, nil, WriteInput{Content: sd.content, Type: sd.subtype}); err != nil {
			t.Fatalf("write %s: %v", sd.content, err)
		}
	}

	// Five back-to-back recalls with the same query. The appended entries
	// should dedup on ID, so buffer approaches the unique candidate count,
	// capped at BufferBudgetMax=3.
	for i := 0; i < 5; i++ {
		if _, _, err := s.handleRecall(ctx, nil, RecallInput{
			Query: "query", Limit: 5, SessionID: sessID,
		}); err != nil {
			t.Fatalf("recall %d: %v", i, err)
		}
	}

	sess := getSession(t, s, sessID)
	if len(sess.WorkingMem.Buffer) != 3 {
		t.Errorf("buffer len = %d, want 3 (capped at BufferBudgetMax)", len(sess.WorkingMem.Buffer))
	}
	if sess.WorkingMem.BudgetUsed != 3 {
		t.Errorf("BudgetUsed = %d, want 3", sess.WorkingMem.BudgetUsed)
	}
}

// Snapshot failure is not exercised here directly: it requires a test-only
// hook (e.g., an injected store that errors on UpdateSessionBuffer) which
// would add a fair bit of scaffolding for one log-and-continue path. The
// contract is asserted by applySessionMutation's comment + the behavior
// that the parent op's return value is unaffected by a snapshot failure.
// If this becomes load-bearing a later phase can add the hook and cover it.
