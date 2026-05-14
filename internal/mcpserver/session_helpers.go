package mcpserver

import (
	"context"
	"fmt"
	"math"

	"github.com/diffsec/reverie/internal/memory"
)

// bufferMutation is a closure that modifies a WorkingMemory in-place during
// the load→mutate→snapshot cycle shared by recall/apply_judgment/reinforce/
// write handlers. Returning nil skips the snapshot write (used when the
// mutation has nothing to persist).
type bufferMutation func(wm *memory.WorkingMemory)

// applySessionMutation packages the "load session → mutate buffer → snapshot"
// path that every session-aware handler shares. Per the Phase 6 spec:
//   - SessionID == "" is a no-op (sessions are opt-in).
//   - Unknown session_id is an error, never silent-create.
//   - Closed session_id is an error (closed sessions are read-only).
//   - Snapshot failures don't fail the parent call: we log a warning and
//     return nil so the handler returns the primary op's result as-is.
func (s *Server) applySessionMutation(ctx context.Context, sessionID string, mutate bufferMutation) error {
	if sessionID == "" {
		return nil
	}
	sess, err := s.store.GetSession(ctx, sessionID)
	if err != nil {
		return fmt.Errorf("get session: %w", err)
	}
	if sess == nil {
		return fmt.Errorf("session not found: %s; call memory_session_init first", sessionID)
	}
	if sess.ClosedAt != nil {
		return fmt.Errorf("session closed: %s", sessionID)
	}
	wm := sess.WorkingMem
	if wm.Buffer == nil {
		wm.Buffer = []memory.MemoryRef{}
	}
	// Ensure BudgetMax reflects the current config ceiling. If a session was
	// written with an older smaller budget, the live config wins — handlers
	// treat BufferBudgetMax as the authoritative cap.
	wm.BudgetMax = s.bufferBudgetMax()

	mutate(&wm)

	if err := s.store.UpdateSessionBuffer(ctx, sessionID, wm); err != nil {
		// Snapshot failures don't fail the parent op (the recall/write/
		// reinforce/apply_judgment has already succeeded). Log and return
		// nil — the next successful snapshot will re-persist.
		s.logger.Warn("session snapshot failed", "session_id", sessionID, "err", err)
		return nil
	}
	return nil
}

// bufferBudgetMax returns the effective buffer budget for session snapshots.
// Falls back to 50 (matching the design-doc default) when the config field
// is zero, which keeps tests that build Config{}-style structs working.
func (s *Server) bufferBudgetMax() int {
	n := s.cfg.Session.BufferBudgetMax
	if n <= 0 {
		return 50
	}
	return n
}

// recallBufferEntry builds the MemoryRef appended to a session buffer for a
// recall hit. Score = similarity * max(retention, 0.01), mirroring the
// composite formula used by apply_judgment so the buffer is ordered
// consistently across the recall→judge→reinforce loop.
func recallBufferEntry(c memory.Candidate, retention float64) memory.MemoryRef {
	score := float64(c.Similarity) * math.Max(retention, 0.01)
	return memory.MemoryRef{
		ID:      c.ID(),
		Layer:   c.Layer(),
		Score:   score,
		Content: c.Content(),
	}
}

// appendWriteToSessionBuffer is the write-path shortcut over
// applySessionMutation. A freshly-written memory lands in the buffer with
// score 1.0 (maximum confidence: the agent just created it) per the Phase 6
// spec.
func (s *Server) appendWriteToSessionBuffer(ctx context.Context, sessionID, memoryID string, layer memory.MemoryType, content string) error {
	if sessionID == "" {
		return nil
	}
	ref := memory.MemoryRef{
		ID:      memoryID,
		Layer:   layer,
		Score:   1.0,
		Content: content,
	}
	budgetMax := s.bufferBudgetMax()
	return s.applySessionMutation(ctx, sessionID, func(wm *memory.WorkingMemory) {
		memory.AppendToBuffer(wm, ref, budgetMax)
	})
}

// reinforceNewScore bumps a buffer entry's score. The spec allows either
// "recompute composite" or "+0.1 capped at 1.0"; we take the simple capped
// bump because we don't have the candidate's current retention on hand
// inside the reinforce handler (the fact/episode isn't re-recalled). The
// bump is idempotent within the cap so repeated reinforce calls converge
// on 1.0.
func reinforceNewScore(current float64) float64 {
	next := current + 0.1
	if next > 1.0 {
		next = 1.0
	}
	return next
}
