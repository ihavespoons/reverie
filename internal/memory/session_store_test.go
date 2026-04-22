package memory

import (
	"context"
	"strings"
	"testing"
	"time"
)

// sessionStoreTable runs the shared assertions in one place against any
// Store implementation. Both the sqlite and in-memory stores must satisfy
// the same contract; parameterizing the tests catches drift between them.
func sessionStoreTable(t *testing.T, newStore func(t *testing.T) Store) {
	t.Helper()

	t.Run("CreateAndGetRoundTrip", func(t *testing.T) {
		s := newStore(t)
		ctx := context.Background()

		sess := Session{
			ID:          "s-1",
			ProjectHint: "reverie",
			Tags:        []string{"go", "mcp"},
			WorkingMem: WorkingMemory{
				Buffer: []MemoryRef{
					{ID: "m1", Layer: TypeL2Semantic, Score: 0.8, Content: "hello"},
				},
				BudgetMax: 50,
			},
		}
		if err := s.CreateSession(ctx, sess); err != nil {
			t.Fatalf("CreateSession: %v", err)
		}

		got, err := s.GetSession(ctx, "s-1")
		if err != nil {
			t.Fatalf("GetSession: %v", err)
		}
		if got == nil {
			t.Fatal("GetSession returned nil for existing session")
		}
		if got.ID != "s-1" {
			t.Errorf("ID = %q, want s-1", got.ID)
		}
		if got.ProjectHint != "reverie" {
			t.Errorf("ProjectHint = %q, want reverie", got.ProjectHint)
		}
		// Tags are normalized (sorted).
		if len(got.Tags) != 2 || got.Tags[0] != "go" || got.Tags[1] != "mcp" {
			t.Errorf("Tags = %v, want [go mcp]", got.Tags)
		}
		if len(got.WorkingMem.Buffer) != 1 {
			t.Fatalf("buffer len = %d, want 1", len(got.WorkingMem.Buffer))
		}
		if got.WorkingMem.Buffer[0].ID != "m1" {
			t.Errorf("buffer[0].ID = %q, want m1", got.WorkingMem.Buffer[0].ID)
		}
		if got.ClosedAt != nil {
			t.Errorf("ClosedAt should be nil on new session, got %v", *got.ClosedAt)
		}
	})

	t.Run("GetUnknownReturnsNilNilError", func(t *testing.T) {
		s := newStore(t)
		got, err := s.GetSession(context.Background(), "does-not-exist")
		if err != nil {
			t.Errorf("GetSession unknown should not error, got: %v", err)
		}
		if got != nil {
			t.Errorf("GetSession unknown should return nil, got %+v", got)
		}
	})

	t.Run("CreateDuplicateErrors", func(t *testing.T) {
		s := newStore(t)
		ctx := context.Background()
		sess := Session{ID: "dup"}
		if err := s.CreateSession(ctx, sess); err != nil {
			t.Fatalf("first CreateSession: %v", err)
		}
		if err := s.CreateSession(ctx, sess); err == nil {
			t.Errorf("second CreateSession should have errored on duplicate id")
		}
	})

	t.Run("UpdateBufferRoundTrip", func(t *testing.T) {
		s := newStore(t)
		ctx := context.Background()
		if err := s.CreateSession(ctx, Session{ID: "s-buf"}); err != nil {
			t.Fatalf("CreateSession: %v", err)
		}
		newWM := WorkingMemory{
			Buffer: []MemoryRef{
				{ID: "m1", Layer: TypeL2Semantic, Score: 0.9, Content: "first"},
				{ID: "m2", Layer: TypeL3Episodic, Score: 0.4, Content: "second"},
			},
			BudgetUsed: 2,
			BudgetMax:  50,
		}
		if err := s.UpdateSessionBuffer(ctx, "s-buf", newWM); err != nil {
			t.Fatalf("UpdateSessionBuffer: %v", err)
		}
		got, err := s.GetSession(ctx, "s-buf")
		if err != nil {
			t.Fatalf("GetSession: %v", err)
		}
		if len(got.WorkingMem.Buffer) != 2 {
			t.Fatalf("buffer len = %d, want 2", len(got.WorkingMem.Buffer))
		}
		if got.WorkingMem.Buffer[0].ID != "m1" || got.WorkingMem.Buffer[1].ID != "m2" {
			t.Errorf("unexpected buffer IDs: %v, %v", got.WorkingMem.Buffer[0].ID, got.WorkingMem.Buffer[1].ID)
		}
		if got.WorkingMem.BudgetMax != 50 {
			t.Errorf("BudgetMax = %d, want 50", got.WorkingMem.BudgetMax)
		}
	})

	t.Run("UpdateMetaReplaces", func(t *testing.T) {
		s := newStore(t)
		ctx := context.Background()
		if err := s.CreateSession(ctx, Session{ID: "s-meta", ProjectHint: "old", Tags: []string{"old"}}); err != nil {
			t.Fatalf("CreateSession: %v", err)
		}
		if err := s.UpdateSessionMeta(ctx, "s-meta", "new", []string{"New", "alpha", "New"}); err != nil {
			t.Fatalf("UpdateSessionMeta: %v", err)
		}
		got, err := s.GetSession(ctx, "s-meta")
		if err != nil {
			t.Fatalf("GetSession: %v", err)
		}
		if got.ProjectHint != "new" {
			t.Errorf("ProjectHint = %q, want new", got.ProjectHint)
		}
		// Tags normalized: lowercased, deduped, sorted.
		if len(got.Tags) != 2 || got.Tags[0] != "alpha" || got.Tags[1] != "new" {
			t.Errorf("Tags = %v, want [alpha new]", got.Tags)
		}
	})

	t.Run("CloseSessionSetsClosedAt", func(t *testing.T) {
		s := newStore(t)
		ctx := context.Background()
		if err := s.CreateSession(ctx, Session{ID: "s-close"}); err != nil {
			t.Fatalf("CreateSession: %v", err)
		}
		before := time.Now().UTC()
		if err := s.CloseSession(ctx, "s-close"); err != nil {
			t.Fatalf("CloseSession: %v", err)
		}
		got, err := s.GetSession(ctx, "s-close")
		if err != nil {
			t.Fatalf("GetSession: %v", err)
		}
		if got.ClosedAt == nil {
			t.Fatal("ClosedAt should be set after CloseSession")
		}
		// The timestamp should be at or after 'before' (allow 1s slop for
		// SQLite's second-precision datetime('now')).
		if got.ClosedAt.Before(before.Add(-2 * time.Second)) {
			t.Errorf("ClosedAt = %v, should be >= %v", got.ClosedAt, before)
		}
	})

	t.Run("CloseSessionIdempotent", func(t *testing.T) {
		s := newStore(t)
		ctx := context.Background()
		if err := s.CreateSession(ctx, Session{ID: "s-close2"}); err != nil {
			t.Fatalf("CreateSession: %v", err)
		}
		if err := s.CloseSession(ctx, "s-close2"); err != nil {
			t.Fatalf("first CloseSession: %v", err)
		}
		// Second close returns nil.
		if err := s.CloseSession(ctx, "s-close2"); err != nil {
			t.Errorf("second CloseSession (idempotent) errored: %v", err)
		}
	})

	t.Run("CloseUnknownErrors", func(t *testing.T) {
		s := newStore(t)
		err := s.CloseSession(context.Background(), "missing")
		if err == nil {
			t.Fatal("CloseSession on unknown id should error")
		}
		if !strings.Contains(err.Error(), "not found") {
			t.Errorf("error should mention not found: %v", err)
		}
	})

	t.Run("UpdateUnknownErrors", func(t *testing.T) {
		s := newStore(t)
		err := s.UpdateSessionBuffer(context.Background(), "missing", WorkingMemory{})
		if err == nil {
			t.Fatal("UpdateSessionBuffer on unknown id should error")
		}
		err = s.UpdateSessionMeta(context.Background(), "missing", "x", nil)
		if err == nil {
			t.Fatal("UpdateSessionMeta on unknown id should error")
		}
	})
}

func TestMemStore_Sessions(t *testing.T) {
	sessionStoreTable(t, func(t *testing.T) Store {
		return NewMemStore()
	})
}

func TestSQLiteStore_Sessions(t *testing.T) {
	sessionStoreTable(t, func(t *testing.T) Store {
		return openTestDB(t)
	})
}
