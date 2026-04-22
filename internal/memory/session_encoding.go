package memory

import (
	"encoding/json"
	"fmt"
	"strings"
)

// sessionWorkingMemoryJSON is the wire shape stored in sessions.working_memory.
// Phase 6a pinned this to buffer + budget only — Clusters, InteractionCtx,
// and TaskMeta are owned elsewhere (clusters by reverie://l1/index, raw
// turns by the harness, TaskMeta by the sessions row's project_hint/tags
// columns). Keeping the stored blob narrow means future schema tweaks don't
// rot old session snapshots.
type sessionWorkingMemoryJSON struct {
	Buffer     []MemoryRef `json:"buffer"`
	BudgetUsed int         `json:"budget_used"`
	BudgetMax  int         `json:"budget_max"`
}

// encodeWorkingMemory renders the buffer-only slice of a WorkingMemory as
// the JSON TEXT stored in sessions.working_memory. A nil buffer encodes as
// an empty array, not null, so round-trips through SQLite TEXT DEFAULT '{}'
// never produce a Go-nil slice on read.
func encodeWorkingMemory(wm WorkingMemory) (string, error) {
	buf := wm.Buffer
	if buf == nil {
		buf = []MemoryRef{}
	}
	payload := sessionWorkingMemoryJSON{
		Buffer:     buf,
		BudgetUsed: wm.BudgetUsed,
		BudgetMax:  wm.BudgetMax,
	}
	b, err := json.Marshal(payload)
	if err != nil {
		return "", fmt.Errorf("encode working memory: %w", err)
	}
	return string(b), nil
}

// decodeWorkingMemory parses the JSON blob back into a WorkingMemory. Empty
// strings and the "{}" default value yield a zero WorkingMemory with a
// non-nil (empty) buffer slice. Malformed input returns an error.
func decodeWorkingMemory(raw string) (WorkingMemory, error) {
	s := strings.TrimSpace(raw)
	if s == "" || s == "{}" {
		return WorkingMemory{Buffer: []MemoryRef{}}, nil
	}
	var payload sessionWorkingMemoryJSON
	if err := json.Unmarshal([]byte(s), &payload); err != nil {
		return WorkingMemory{}, fmt.Errorf("decode working memory: %w", err)
	}
	if payload.Buffer == nil {
		payload.Buffer = []MemoryRef{}
	}
	return WorkingMemory{
		Buffer:     payload.Buffer,
		BudgetUsed: payload.BudgetUsed,
		BudgetMax:  payload.BudgetMax,
	}, nil
}
