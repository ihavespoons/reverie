package mcpserver

import (
	"context"
	"strings"
	"testing"

	"github.com/diffsec/reverie/internal/cluster"
	"github.com/diffsec/reverie/internal/config"
	"github.com/diffsec/reverie/internal/decay"
	"github.com/diffsec/reverie/internal/manager"
	"github.com/diffsec/reverie/internal/memory"
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

// --- Phase 6c: session_init / snapshot / restore / end ---

// newFreshSessionTestServer builds a bare server (no pre-seeded session) so
// session_init tests can exercise the create path. Mirrors newSessionTestServer
// with the CreateSession call elided.
func newFreshSessionTestServer(t *testing.T, embedder *stubEmbedder, budgetMax int) *Server {
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
	return s
}

func TestSessionInit_NewSession(t *testing.T) {
	emb := newStubEmbedder(4)
	s := newFreshSessionTestServer(t, emb, 0)

	_, out, err := s.handleSessionInit(context.Background(), nil, SessionInitInput{
		SessionID:   "sess-new",
		ProjectHint: "reverie",
		Tags:        []string{"Phase6", "session"},
	})
	if err != nil {
		t.Fatalf("session_init: %v", err)
	}
	if !out.Created {
		t.Errorf("Created = false, want true")
	}
	if out.SessionID != "sess-new" {
		t.Errorf("SessionID = %q, want sess-new", out.SessionID)
	}
	if len(out.Buffer) != 0 {
		t.Errorf("Buffer len = %d, want 0", len(out.Buffer))
	}
	if out.CreatedAt == "" {
		t.Errorf("CreatedAt should be populated")
	}
	if out.ClosedAt != nil {
		t.Errorf("ClosedAt should be nil on fresh create, got %q", *out.ClosedAt)
	}
	// Tags should be normalized (lowercase, sorted).
	if len(out.Tags) != 2 || out.Tags[0] != "phase6" || out.Tags[1] != "session" {
		t.Errorf("Tags = %v, want [phase6 session]", out.Tags)
	}
	if out.ProjectHint != "reverie" {
		t.Errorf("ProjectHint = %q, want reverie", out.ProjectHint)
	}

	// Sanity: row exists in the store.
	sess := getSession(t, s, "sess-new")
	if sess == nil {
		t.Fatal("session row missing from store")
	}
}

func TestSessionInit_EmptyID(t *testing.T) {
	s := newFreshSessionTestServer(t, newStubEmbedder(4), 0)
	_, _, err := s.handleSessionInit(context.Background(), nil, SessionInitInput{SessionID: ""})
	if err == nil {
		t.Fatal("expected error for empty session_id")
	}
}

func TestSessionInit_ResumeExisting(t *testing.T) {
	emb := newStubEmbedder(4)
	emb.vectors["brand new"] = []float32{0.5, 0.5, 0.0, 0.0}
	s, sessID := newSessionTestServer(t, emb, 0)

	// Seed a buffer via a session-scoped write so resume has something to return.
	ctx := context.Background()
	_, writeOut, err := s.handleWrite(ctx, nil, WriteInput{
		Content: "brand new", Type: "user", SessionID: sessID,
	})
	if err != nil {
		t.Fatalf("seed write: %v", err)
	}

	_, out, err := s.handleSessionInit(ctx, nil, SessionInitInput{SessionID: sessID})
	if err != nil {
		t.Fatalf("session_init resume: %v", err)
	}
	if out.Created {
		t.Errorf("Created = true, want false on resume")
	}
	if len(out.Buffer) != 1 {
		t.Fatalf("Buffer len = %d, want 1", len(out.Buffer))
	}
	if out.Buffer[0].ID != writeOut.ID {
		t.Errorf("Buffer[0].ID = %q, want %q", out.Buffer[0].ID, writeOut.ID)
	}
	if out.ClosedAt != nil {
		t.Errorf("ClosedAt should be nil on active resume")
	}
}

func TestSessionInit_ResumeClosedSession_Errors(t *testing.T) {
	s, sessID := newSessionTestServer(t, newStubEmbedder(4), 0)
	if err := s.store.CloseSession(context.Background(), sessID); err != nil {
		t.Fatalf("CloseSession: %v", err)
	}

	_, _, err := s.handleSessionInit(context.Background(), nil, SessionInitInput{SessionID: sessID})
	if err == nil {
		t.Fatal("expected error resuming closed session")
	}
	if !strings.Contains(err.Error(), "session closed") {
		t.Errorf("error should mention session closed: %v", err)
	}
}

func TestSessionInit_ResumeReplacesProjectHint(t *testing.T) {
	s, sessID := newSessionTestServer(t, newStubEmbedder(4), 0)
	// Seed initial meta.
	if err := s.store.UpdateSessionMeta(context.Background(), sessID, "original", []string{"keep"}); err != nil {
		t.Fatalf("seed meta: %v", err)
	}

	_, out, err := s.handleSessionInit(context.Background(), nil, SessionInitInput{
		SessionID:   sessID,
		ProjectHint: "replaced",
	})
	if err != nil {
		t.Fatalf("session_init: %v", err)
	}
	if out.ProjectHint != "replaced" {
		t.Errorf("ProjectHint = %q, want 'replaced'", out.ProjectHint)
	}
	// Tags should be untouched because in.Tags is nil.
	if len(out.Tags) != 1 || out.Tags[0] != "keep" {
		t.Errorf("Tags = %v, want [keep]", out.Tags)
	}
}

func TestSessionInit_ResumeMergesTags(t *testing.T) {
	s, sessID := newSessionTestServer(t, newStubEmbedder(4), 0)
	if err := s.store.UpdateSessionMeta(context.Background(), sessID, "", []string{"one", "two"}); err != nil {
		t.Fatalf("seed meta: %v", err)
	}

	_, out, err := s.handleSessionInit(context.Background(), nil, SessionInitInput{
		SessionID: sessID,
		// Include a duplicate "One" (different case) + a new tag. Merge
		// should dedup case-insensitively via the store normalizer and add "three".
		Tags: []string{"One", "three"},
	})
	if err != nil {
		t.Fatalf("session_init: %v", err)
	}

	// Expect union {one,two,three} — normalized order is alphabetical.
	want := []string{"one", "three", "two"}
	if len(out.Tags) != len(want) {
		t.Fatalf("Tags len = %d, want %d (got %v)", len(out.Tags), len(want), out.Tags)
	}
	for i, w := range want {
		if out.Tags[i] != w {
			t.Errorf("Tags[%d] = %q, want %q", i, out.Tags[i], w)
		}
	}
}

func TestSessionSnapshot_Active(t *testing.T) {
	s, sessID := newSessionTestServer(t, newStubEmbedder(4), 0)

	_, out, err := s.handleSessionSnapshot(context.Background(), nil, SessionSnapshotInput{SessionID: sessID})
	if err != nil {
		t.Fatalf("session_snapshot: %v", err)
	}
	if !out.Persisted {
		t.Errorf("Persisted = false, want true")
	}
	if out.UpdatedAt == "" {
		t.Errorf("UpdatedAt should be populated")
	}
}

func TestSessionSnapshot_UnknownSession_Errors(t *testing.T) {
	s := newFreshSessionTestServer(t, newStubEmbedder(4), 0)
	_, _, err := s.handleSessionSnapshot(context.Background(), nil, SessionSnapshotInput{SessionID: "nope"})
	if err == nil {
		t.Fatal("expected error for unknown session")
	}
	if !strings.Contains(err.Error(), "session not found") {
		t.Errorf("error should mention 'session not found': %v", err)
	}
}

func TestSessionSnapshot_ClosedSession_Errors(t *testing.T) {
	s, sessID := newSessionTestServer(t, newStubEmbedder(4), 0)
	if err := s.store.CloseSession(context.Background(), sessID); err != nil {
		t.Fatalf("close: %v", err)
	}
	_, _, err := s.handleSessionSnapshot(context.Background(), nil, SessionSnapshotInput{SessionID: sessID})
	if err == nil {
		t.Fatal("expected error for closed session")
	}
	if !strings.Contains(err.Error(), "session closed") {
		t.Errorf("error should mention 'session closed': %v", err)
	}
}

func TestSessionRestore_ShapeIncludesClosedAt(t *testing.T) {
	s, sessID := newSessionTestServer(t, newStubEmbedder(4), 0)
	if err := s.store.UpdateSessionMeta(context.Background(), sessID, "proj", []string{"alpha"}); err != nil {
		t.Fatalf("seed meta: %v", err)
	}
	if err := s.store.CloseSession(context.Background(), sessID); err != nil {
		t.Fatalf("close: %v", err)
	}

	_, out, err := s.handleSessionRestore(context.Background(), nil, SessionRestoreInput{SessionID: sessID})
	if err != nil {
		t.Fatalf("session_restore: %v", err)
	}
	if out.ClosedAt == nil {
		t.Fatal("ClosedAt should be set for closed session")
	}
	if out.UpdatedAt == "" {
		t.Errorf("UpdatedAt should be populated")
	}
	if out.ProjectHint != "proj" {
		t.Errorf("ProjectHint = %q, want proj", out.ProjectHint)
	}
	if len(out.Tags) != 1 || out.Tags[0] != "alpha" {
		t.Errorf("Tags = %v, want [alpha]", out.Tags)
	}
}

func TestSessionRestore_UnknownSession_Errors(t *testing.T) {
	s := newFreshSessionTestServer(t, newStubEmbedder(4), 0)
	_, _, err := s.handleSessionRestore(context.Background(), nil, SessionRestoreInput{SessionID: "nope"})
	if err == nil {
		t.Fatal("expected error for unknown session")
	}
	if !strings.Contains(err.Error(), "session not found") {
		t.Errorf("error should mention 'session not found': %v", err)
	}
}

// TestSessionEnd_TickScopedToBufferClusters seeds a session with buffer
// entries pointing to TWO distinct clusters, verifies session_end ticks
// exactly those two (resetting turns_since) and bumps an untouched cluster.
func TestSessionEnd_TickScopedToBufferClusters(t *testing.T) {
	emb := newStubEmbedder(4)
	// Three orthogonal seeds so each lands in its own cluster (threshold=0.60
	// inside the assigner). Subtype distinct so no supersede.
	emb.vectors["seed-a"] = []float32{1, 0, 0, 0}
	emb.vectors["seed-b"] = []float32{0, 1, 0, 0}
	emb.vectors["seed-c"] = []float32{0, 0, 1, 0}
	s, sessID := newSessionTestServer(t, emb, 0)

	ctx := context.Background()
	_, wa, err := s.handleWrite(ctx, nil, WriteInput{Content: "seed-a", Type: "user"})
	if err != nil {
		t.Fatalf("write a: %v", err)
	}
	_, wb, err := s.handleWrite(ctx, nil, WriteInput{Content: "seed-b", Type: "project"})
	if err != nil {
		t.Fatalf("write b: %v", err)
	}
	// seed-c is NOT added to the session buffer; its cluster should bump.
	_, _, err = s.handleWrite(ctx, nil, WriteInput{Content: "seed-c", Type: "reference"})
	if err != nil {
		t.Fatalf("write c: %v", err)
	}

	factA, err := s.store.GetFact(ctx, wa.ID)
	if err != nil || factA == nil {
		t.Fatalf("getFact a: %v", err)
	}
	factB, err := s.store.GetFact(ctx, wb.ID)
	if err != nil || factB == nil {
		t.Fatalf("getFact b: %v", err)
	}
	if factA.ClusterID == factB.ClusterID {
		t.Fatalf("seeds share a cluster (%s); test needs two distinct clusters", factA.ClusterID)
	}

	// Inject the buffer directly — fastest path to "two buffer entries, two
	// clusters" without replaying multiple recalls.
	if err := s.store.UpdateSessionBuffer(ctx, sessID, memory.WorkingMemory{
		Buffer: []memory.MemoryRef{
			{ID: wa.ID, Layer: memory.TypeL2Semantic, Score: 0.9, Content: "seed-a"},
			{ID: wb.ID, Layer: memory.TypeL2Semantic, Score: 0.8, Content: "seed-b"},
		},
		BudgetUsed: 2,
		BudgetMax:  50,
	}); err != nil {
		t.Fatalf("inject buffer: %v", err)
	}

	// Snapshot pre-tick turns_since for each cluster. After the writes above
	// each cluster was touched by TickDecay([clusterID]) — turns_since values
	// are cluster-specific, so we just record what's there and verify
	// direction-of-change post-session_end.
	preClusters, err := s.store.ListClusters(ctx)
	if err != nil {
		t.Fatalf("list clusters pre: %v", err)
	}
	preTurns := map[string]int{}
	for _, c := range preClusters {
		preTurns[c.ID] = c.TurnsSince
	}

	_, endOut, err := s.handleSessionEnd(ctx, nil, SessionEndInput{SessionID: sessID})
	if err != nil {
		t.Fatalf("session_end: %v", err)
	}
	if endOut.ClustersTicked != 2 {
		t.Errorf("ClustersTicked = %d, want 2", endOut.ClustersTicked)
	}

	postClusters, err := s.store.ListClusters(ctx)
	if err != nil {
		t.Fatalf("list clusters post: %v", err)
	}
	touched := map[string]bool{factA.ClusterID: true, factB.ClusterID: true}
	for _, c := range postClusters {
		if touched[c.ID] {
			if c.TurnsSince != 0 {
				t.Errorf("touched cluster %s turns_since = %d, want 0", c.ID, c.TurnsSince)
			}
		} else {
			if c.TurnsSince != preTurns[c.ID]+1 {
				t.Errorf("untouched cluster %s turns_since = %d, want %d (pre=%d)", c.ID, c.TurnsSince, preTurns[c.ID]+1, preTurns[c.ID])
			}
		}
	}

	// Session must be closed.
	sess := getSession(t, s, sessID)
	if sess.ClosedAt == nil {
		t.Error("session ClosedAt should be set after session_end")
	}
}

func TestSessionEnd_WithEpisodePayload(t *testing.T) {
	emb := newStubEmbedder(4)
	emb.vectors["buffered"] = []float32{1, 0, 0, 0}
	emb.vectors["s\na\no\np"] = []float32{0, 1, 0, 0}
	s, sessID := newSessionTestServer(t, emb, 0)

	ctx := context.Background()
	_, writeOut, err := s.handleWrite(ctx, nil, WriteInput{
		Content: "buffered", Type: "project", SessionID: sessID,
	})
	if err != nil {
		t.Fatalf("seed write: %v", err)
	}

	_, endOut, err := s.handleSessionEnd(ctx, nil, SessionEndInput{
		SessionID: sessID,
		Episode: &EpisodePayload{
			Situation: "s", Action: "a", Outcome: "o", Preemptive: "p",
		},
	})
	if err != nil {
		t.Fatalf("session_end: %v", err)
	}
	if endOut.EpisodeID == "" {
		t.Fatal("EpisodeID should be populated")
	}

	// Verify the episode was written with the session tag and auto-link.
	ep, err := s.store.GetEpisode(ctx, endOut.EpisodeID)
	if err != nil || ep == nil {
		t.Fatalf("GetEpisode: %v", err)
	}
	expectTag := "session:" + sessID
	hasSessionTag := false
	for _, tag := range ep.Tags {
		if tag == expectTag {
			hasSessionTag = true
			break
		}
	}
	if !hasSessionTag {
		t.Errorf("episode tags %v should include %q", ep.Tags, expectTag)
	}

	// Auto-link: the buffered fact should be linked to the episode via an
	// evidence edge in memory_edges (the Phase-7 successor of
	// fact_episode_links). The episode is the destination (dst_id) of the
	// fact->episode evidence edge written by ReplaceEpisodeLinks.
	edges, err := s.store.ListEdges(ctx, ep.ID, 1)
	if err != nil {
		t.Fatalf("ListEdges: %v", err)
	}
	foundLink := false
	for _, e := range edges {
		if e.Edge.EdgeType != "evidence" {
			continue
		}
		if e.Edge.SrcID == writeOut.ID && e.Edge.DstID == ep.ID {
			foundLink = true
			break
		}
	}
	if !foundLink {
		t.Errorf("episode should have evidence edge from buffered fact %s; got %+v", writeOut.ID, edges)
	}
}

func TestSessionEnd_WithoutEpisode(t *testing.T) {
	s, sessID := newSessionTestServer(t, newStubEmbedder(4), 0)

	_, endOut, err := s.handleSessionEnd(context.Background(), nil, SessionEndInput{SessionID: sessID})
	if err != nil {
		t.Fatalf("session_end: %v", err)
	}
	if endOut.EpisodeID != "" {
		t.Errorf("EpisodeID = %q, want empty string", endOut.EpisodeID)
	}
	// Session must still close.
	sess := getSession(t, s, sessID)
	if sess.ClosedAt == nil {
		t.Error("session should be closed even without an episode")
	}
}

func TestSessionEnd_AlreadyClosed_Errors(t *testing.T) {
	s, sessID := newSessionTestServer(t, newStubEmbedder(4), 0)
	if err := s.store.CloseSession(context.Background(), sessID); err != nil {
		t.Fatalf("close: %v", err)
	}
	_, _, err := s.handleSessionEnd(context.Background(), nil, SessionEndInput{SessionID: sessID})
	if err == nil {
		t.Fatal("expected error for already-closed session")
	}
	if !strings.Contains(err.Error(), "already closed") {
		t.Errorf("error should mention 'already closed': %v", err)
	}
}

// TestSessionEnd_SkipsDeletedMemoryIDs injects a buffer with one live fact
// and one dangling reference to a deleted fact. session_end must gracefully
// skip the dead ref — ClustersTicked counts only the live membership.
func TestSessionEnd_SkipsDeletedMemoryIDs(t *testing.T) {
	emb := newStubEmbedder(4)
	emb.vectors["live"] = []float32{1, 0, 0, 0}
	s, sessID := newSessionTestServer(t, emb, 0)

	ctx := context.Background()
	_, liveOut, err := s.handleWrite(ctx, nil, WriteInput{Content: "live", Type: "user"})
	if err != nil {
		t.Fatalf("write live: %v", err)
	}

	// Inject a buffer referencing a non-existent ID alongside the live one.
	if err := s.store.UpdateSessionBuffer(ctx, sessID, memory.WorkingMemory{
		Buffer: []memory.MemoryRef{
			{ID: liveOut.ID, Layer: memory.TypeL2Semantic, Score: 0.9, Content: "live"},
			{ID: "deleted-fact-id", Layer: memory.TypeL2Semantic, Score: 0.5, Content: "ghost"},
			{ID: "deleted-episode-id", Layer: memory.TypeL3Episodic, Score: 0.5, Content: "ghost ep"},
		},
		BudgetUsed: 3,
		BudgetMax:  50,
	}); err != nil {
		t.Fatalf("inject buffer: %v", err)
	}

	_, endOut, err := s.handleSessionEnd(ctx, nil, SessionEndInput{SessionID: sessID})
	if err != nil {
		t.Fatalf("session_end: %v", err)
	}
	// Only the live fact's cluster should be counted.
	if endOut.ClustersTicked != 1 {
		t.Errorf("ClustersTicked = %d, want 1 (dead refs skipped)", endOut.ClustersTicked)
	}
}

// TestHandleSessionEnd_TicksEntitiesFromBuffer is the Phase-7 plumbing
// integration test: a buffered memory mentions an entity, the entity has
// non-zero turns_since pre-session_end, and session_end's scoped tick
// resets that entity's turns_since to 0 (and bumps an unrelated entity
// to demonstrate that the access set really is scoped).
func TestHandleSessionEnd_TicksEntitiesFromBuffer(t *testing.T) {
	emb := newStubEmbedder(4)
	emb.vectors["buffered fact"] = []float32{1, 0, 0, 0}
	emb.vectors["mentioned (file)"] = []float32{0, 1, 0, 0}
	emb.vectors["other (file)"] = []float32{0, 0, 1, 0}
	s, sessID := newSessionTestServer(t, emb, 0)

	ctx := context.Background()

	// Seed: one fact in the session buffer (via the write+session path),
	// one mentioned entity, one unrelated entity.
	_, writeOut, err := s.handleWrite(ctx, nil, WriteInput{
		Content:   "buffered fact",
		Type:      "project",
		SessionID: sessID,
	})
	if err != nil {
		t.Fatalf("write: %v", err)
	}
	_, entMentioned, err := s.handleEntityUpsert(ctx, nil, EntityUpsertInput{Name: "mentioned", EntityType: "file"})
	if err != nil {
		t.Fatalf("upsert mentioned: %v", err)
	}
	_, entOther, err := s.handleEntityUpsert(ctx, nil, EntityUpsertInput{Name: "other", EntityType: "file"})
	if err != nil {
		t.Fatalf("upsert other: %v", err)
	}
	if _, _, err := s.handleEntityMention(ctx, nil, EntityMentionInput{
		MemoryID:  writeOut.ID,
		EntityIDs: []string{entMentioned.EntityID},
	}); err != nil {
		t.Fatalf("entity_mention: %v", err)
	}

	// Drive both entities forward by ticking decay a few times — the
	// session-buffered memory's mentioned entity should reset on
	// session_end while the unrelated one keeps aging.
	for i := 0; i < 3; i++ {
		if _, _, err := s.handleDecayTick(ctx, nil, DecayTickInput{}); err != nil {
			t.Fatalf("decay_tick %d: %v", i, err)
		}
	}

	preMentioned, err := s.store.GetEntity(ctx, entMentioned.EntityID)
	if err != nil {
		t.Fatalf("pre GetEntity mentioned: %v", err)
	}
	if preMentioned.TurnsSince == 0 {
		t.Fatal("pre-condition: mentioned entity should have turns_since > 0 before session_end")
	}

	// End session — must reset the mentioned entity via the buffer →
	// entity_mentions lookup.
	_, _, err = s.handleSessionEnd(ctx, nil, SessionEndInput{SessionID: sessID})
	if err != nil {
		t.Fatalf("session_end: %v", err)
	}

	postMentioned, err := s.store.GetEntity(ctx, entMentioned.EntityID)
	if err != nil {
		t.Fatalf("post GetEntity mentioned: %v", err)
	}
	if postMentioned.TurnsSince != 0 {
		t.Errorf("mentioned entity turns_since = %d, want 0 (should reset via buffer mention)", postMentioned.TurnsSince)
	}

	postOther, err := s.store.GetEntity(ctx, entOther.EntityID)
	if err != nil {
		t.Fatalf("post GetEntity other: %v", err)
	}
	// The session-end tick bumps the unrelated entity by 1.
	if postOther.TurnsSince <= preMentioned.TurnsSince {
		t.Errorf("unrelated entity turns_since = %d, want > %d (should keep aging)", postOther.TurnsSince, preMentioned.TurnsSince)
	}
}
