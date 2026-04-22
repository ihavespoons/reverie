package memory

import "testing"

func mkRef(id string, score float64) MemoryRef {
	return MemoryRef{
		ID:      id,
		Layer:   TypeL2Semantic,
		Score:   score,
		Content: "c-" + id,
	}
}

func TestAppendToBuffer_HappyPath(t *testing.T) {
	wm := &WorkingMemory{}
	AppendToBuffer(wm, mkRef("a", 0.5), 10)

	if len(wm.Buffer) != 1 {
		t.Fatalf("buffer len = %d, want 1", len(wm.Buffer))
	}
	if wm.Buffer[0].ID != "a" {
		t.Errorf("buffer[0].ID = %q, want a", wm.Buffer[0].ID)
	}
	if wm.BudgetUsed != 1 {
		t.Errorf("BudgetUsed = %d, want 1", wm.BudgetUsed)
	}
	if wm.BudgetMax != 10 {
		t.Errorf("BudgetMax = %d, want 10", wm.BudgetMax)
	}
}

func TestAppendToBuffer_Dedup(t *testing.T) {
	wm := &WorkingMemory{}
	AppendToBuffer(wm, mkRef("a", 0.3), 10)
	AppendToBuffer(wm, mkRef("b", 0.7), 10)

	// Re-append "a" with a new score + content; length should NOT grow.
	updated := MemoryRef{ID: "a", Layer: TypeL2Semantic, Score: 0.9, Content: "new-a"}
	AppendToBuffer(wm, updated, 10)

	if len(wm.Buffer) != 2 {
		t.Fatalf("buffer len = %d, want 2 (dedup failed)", len(wm.Buffer))
	}
	// After resort, "a" is highest (0.9) and first.
	if wm.Buffer[0].ID != "a" {
		t.Errorf("after resort, buffer[0].ID = %q, want a", wm.Buffer[0].ID)
	}
	if wm.Buffer[0].Score != 0.9 {
		t.Errorf("a.Score = %v, want 0.9", wm.Buffer[0].Score)
	}
	if wm.Buffer[0].Content != "new-a" {
		t.Errorf("a.Content = %q, want new-a", wm.Buffer[0].Content)
	}
}

func TestAppendToBuffer_Eviction(t *testing.T) {
	wm := &WorkingMemory{}
	AppendToBuffer(wm, mkRef("a", 0.9), 3)
	AppendToBuffer(wm, mkRef("b", 0.5), 3)
	AppendToBuffer(wm, mkRef("c", 0.3), 3)
	// Adding a fourth should drop the lowest-scored ("c").
	AppendToBuffer(wm, mkRef("d", 0.7), 3)

	if len(wm.Buffer) != 3 {
		t.Fatalf("buffer len = %d, want 3", len(wm.Buffer))
	}
	ids := []string{wm.Buffer[0].ID, wm.Buffer[1].ID, wm.Buffer[2].ID}
	// Expected descending: a (0.9), d (0.7), b (0.5). c dropped.
	if ids[0] != "a" || ids[1] != "d" || ids[2] != "b" {
		t.Errorf("post-eviction order = %v, want [a d b]", ids)
	}
	for _, r := range wm.Buffer {
		if r.ID == "c" {
			t.Errorf("c should have been evicted")
		}
	}
}

// TestAppendToBuffer_TieBreak locks in the rule: when two entries have the
// same lowest score, the earlier-inserted one is kept (the later one is
// evicted). This derives from the sortBufferByScoreDesc's stable sort +
// evict-from-tail policy.
func TestAppendToBuffer_TieBreak(t *testing.T) {
	wm := &WorkingMemory{}
	AppendToBuffer(wm, mkRef("first", 0.2), 2)
	AppendToBuffer(wm, mkRef("top", 0.9), 2)
	// Third append at the same low score as "first" — budget=2 so one of
	// {first, third} must be evicted. Earlier wins: "third" is dropped.
	AppendToBuffer(wm, mkRef("third", 0.2), 2)

	if len(wm.Buffer) != 2 {
		t.Fatalf("buffer len = %d, want 2", len(wm.Buffer))
	}
	kept := map[string]bool{}
	for _, r := range wm.Buffer {
		kept[r.ID] = true
	}
	if !kept["top"] {
		t.Errorf("top should remain (highest score)")
	}
	if !kept["first"] {
		t.Errorf("first should remain (tie broken by insertion order)")
	}
	if kept["third"] {
		t.Errorf("third should have been evicted on tie")
	}
}

func TestAppendToBuffer_UnboundedBudget(t *testing.T) {
	wm := &WorkingMemory{}
	for i := 0; i < 100; i++ {
		AppendToBuffer(wm, mkRef(string(rune('a'+i%26)), float64(i)), 0)
	}
	// With budgetMax=0 (unbounded) we dedup on key — 26 unique ids.
	if len(wm.Buffer) != 26 {
		t.Errorf("buffer len = %d, want 26", len(wm.Buffer))
	}
}

func TestReplaceBufferFiltered(t *testing.T) {
	wm := &WorkingMemory{
		Buffer: []MemoryRef{
			mkRef("a", 0.9),
			mkRef("b", 0.7),
			mkRef("c", 0.5),
		},
	}
	ReplaceBufferFiltered(wm, []string{"a", "c"})

	if len(wm.Buffer) != 2 {
		t.Fatalf("buffer len = %d, want 2", len(wm.Buffer))
	}
	if wm.Buffer[0].ID != "a" || wm.Buffer[1].ID != "c" {
		t.Errorf("got IDs %v, want [a c]", []string{wm.Buffer[0].ID, wm.Buffer[1].ID})
	}
	if wm.BudgetUsed != 2 {
		t.Errorf("BudgetUsed = %d, want 2", wm.BudgetUsed)
	}
}

func TestReplaceBufferFiltered_Empty(t *testing.T) {
	wm := &WorkingMemory{
		Buffer: []MemoryRef{mkRef("a", 0.9)},
	}
	ReplaceBufferFiltered(wm, nil)
	if len(wm.Buffer) != 0 {
		t.Errorf("buffer len = %d, want 0", len(wm.Buffer))
	}
	if wm.BudgetUsed != 0 {
		t.Errorf("BudgetUsed = %d, want 0", wm.BudgetUsed)
	}
}

func TestRescoreBuffer(t *testing.T) {
	wm := &WorkingMemory{
		Buffer: []MemoryRef{
			mkRef("a", 0.3),
			mkRef("b", 0.5),
			mkRef("c", 0.7),
		},
	}
	RescoreBuffer(wm, map[string]float64{
		"a": 0.95, // a jumps to top
		"c": 0.1,  // c drops to bottom
		"z": 0.99, // z is not in buffer — no-op
	})

	if wm.Buffer[0].ID != "a" {
		t.Errorf("expected a at top, got %s", wm.Buffer[0].ID)
	}
	if wm.Buffer[0].Score != 0.95 {
		t.Errorf("a.Score = %v, want 0.95", wm.Buffer[0].Score)
	}
	if wm.Buffer[len(wm.Buffer)-1].ID != "c" {
		t.Errorf("expected c at bottom, got %s", wm.Buffer[len(wm.Buffer)-1].ID)
	}
	// Unmentioned 'b' score unchanged.
	for _, r := range wm.Buffer {
		if r.ID == "b" && r.Score != 0.5 {
			t.Errorf("b.Score changed to %v, want 0.5", r.Score)
		}
	}
	// Verify 'z' wasn't injected.
	if len(wm.Buffer) != 3 {
		t.Errorf("buffer len = %d, want 3 (z should not be inserted)", len(wm.Buffer))
	}
}

func TestRescoreBuffer_NilInputs(t *testing.T) {
	// Guard against nil map / empty buffer degenerate cases.
	var wm *WorkingMemory
	RescoreBuffer(wm, map[string]float64{"a": 0.5}) // must not panic

	wm = &WorkingMemory{Buffer: []MemoryRef{mkRef("a", 0.5)}}
	RescoreBuffer(wm, nil)
	if wm.Buffer[0].Score != 0.5 {
		t.Errorf("nil update map should leave scores unchanged")
	}
}

func TestAppendToBuffer_NilReceiver(t *testing.T) {
	// Must not panic on nil *WorkingMemory.
	AppendToBuffer(nil, mkRef("a", 0.5), 10)
}
