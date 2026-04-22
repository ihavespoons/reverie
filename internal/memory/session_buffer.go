package memory

import "sort"

// This file holds pure buffer mutation helpers for session working memory
// (Phase 6b). They operate only on a *WorkingMemory and have no DB or MCP
// dependencies so they can be exercised in isolation. Three ops are needed:
//
//   AppendToBuffer        — recall/write append with dedup + eviction.
//   ReplaceBufferFiltered — apply_judgment drops rejected candidates.
//   RescoreBuffer         — reinforce bumps scores for in-buffer IDs.
//
// Ordering invariant maintained by every helper: Buffer is sorted by Score
// descending. Ties are broken by insertion order — when two entries have the
// same score, the one that was added earlier sorts first. This keeps
// eviction deterministic: the last entry in the slice is always the
// lowest-scored-and-newest, so it is evicted first when over-budget.

// AppendToBuffer adds ref to wm.Buffer, deduplicating by ID. If an entry with
// the same ID is already present, its Score and Content are replaced with
// the incoming values (newest wins, consistent with the spec's "dedup-update
// keeps tie-freshness naturally"). After the append, the buffer is re-sorted
// by Score descending with insertion order as the tiebreak, and evicted to
// respect budgetMax.
//
// budgetMax <= 0 means unbounded; the eviction step is skipped.
func AppendToBuffer(wm *WorkingMemory, ref MemoryRef, budgetMax int) {
	if wm == nil {
		return
	}

	// Dedup: look for an existing entry with the same ID and update it in
	// place. This preserves the original position until the sort re-orders.
	replaced := false
	for i := range wm.Buffer {
		if wm.Buffer[i].ID == ref.ID {
			wm.Buffer[i].Score = ref.Score
			wm.Buffer[i].Content = ref.Content
			wm.Buffer[i].Layer = ref.Layer
			replaced = true
			break
		}
	}
	if !replaced {
		wm.Buffer = append(wm.Buffer, ref)
	}

	sortBufferByScoreDesc(wm.Buffer)
	evictIfOverBudget(wm, budgetMax)

	wm.BudgetUsed = len(wm.Buffer)
	if budgetMax > 0 {
		wm.BudgetMax = budgetMax
	}
}

// ReplaceBufferFiltered keeps only entries whose ID appears in keepIDs. The
// relative order of surviving entries is preserved (so a subsequent resort
// remains stable against the score order established by the previous
// append). BudgetUsed is refreshed.
func ReplaceBufferFiltered(wm *WorkingMemory, keepIDs []string) {
	if wm == nil {
		return
	}
	if len(wm.Buffer) == 0 {
		wm.BudgetUsed = 0
		return
	}

	keep := make(map[string]struct{}, len(keepIDs))
	for _, id := range keepIDs {
		keep[id] = struct{}{}
	}

	filtered := wm.Buffer[:0]
	for _, r := range wm.Buffer {
		if _, ok := keep[r.ID]; ok {
			filtered = append(filtered, r)
		}
	}
	wm.Buffer = filtered
	wm.BudgetUsed = len(wm.Buffer)
}

// RescoreBuffer updates the Score of every buffer entry whose ID is present
// in idToScore (no-op for IDs not currently in the buffer). The buffer is
// re-sorted by score descending afterwards.
func RescoreBuffer(wm *WorkingMemory, idToScore map[string]float64) {
	if wm == nil || len(idToScore) == 0 || len(wm.Buffer) == 0 {
		return
	}
	for i := range wm.Buffer {
		if newScore, ok := idToScore[wm.Buffer[i].ID]; ok {
			wm.Buffer[i].Score = newScore
		}
	}
	sortBufferByScoreDesc(wm.Buffer)
}

// sortBufferByScoreDesc sorts buf by Score descending, stable so that ties
// fall back to insertion order (the slice's existing relative ordering).
func sortBufferByScoreDesc(buf []MemoryRef) {
	sort.SliceStable(buf, func(i, j int) bool {
		return buf[i].Score > buf[j].Score
	})
}

// evictIfOverBudget trims wm.Buffer down to budgetMax entries by repeatedly
// dropping the last (lowest-scored, tied-newest) entry. Callers must ensure
// the buffer was sorted descending by score first.
func evictIfOverBudget(wm *WorkingMemory, budgetMax int) {
	if budgetMax <= 0 {
		return
	}
	if len(wm.Buffer) <= budgetMax {
		return
	}
	wm.Buffer = wm.Buffer[:budgetMax]
}
