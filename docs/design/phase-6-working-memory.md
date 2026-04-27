# Phase 6 — Working memory & sessions

Status: **DESIGN** (6a is the design pass; 6b–6d implement)
Prereqs: Phase 1 migration framework.
Unblocks: nothing — Phase 6 completes the paper-faithful architecture.

## Goal

Make reverie's notion of a "session" real. Today `WorkingMemory` is a Go struct that lives only in RAM, `TaskMeta` has no MCP surface, and the `sessions` table is dead code. The paper's architecture calls for a bounded, persisted working memory per session so that agent conversations can resume with prior context intact. Phase 6 delivers exactly that — no more, no less.

The hard question up front: **who owns what.** The paper describes `WorkingMemory` as holding three things: interaction context (raw turns), always-resident clusters, and a dynamic buffer of L2/L3 refs. An MCP server cannot practically own the full interaction context without the agent streaming every turn, and doing so duplicates the harness's conversation log for no added capability. So we draw a line.

## Ownership decision (Phase 6a output)

| Concern | Owner | Why |
|---|---|---|
| Raw interaction context (turns) | **Harness** | The conversation window already lives in the agent's prompt context. Duplicating it in reverie adds latency and storage for no recall advantage. |
| Always-resident L1 clusters | **Reverie (already)** | `reverie://l1/index` + Phase 1's `reverie://l1/cluster/{id}` covers this. No change. |
| Dynamic L2/L3 buffer (recall results, reinforced memories) | **Reverie (new)** | This IS the valuable session state — what memories are in play this session. Persisting it enables resume. |
| `TaskMeta` (project hint, session_id, tags) | **Reverie (new)** | One-shot setup per session; persists with the buffer. |

So Phase 6 persists only the **buffer + TaskMeta** under a session_id. Raw turns are not our problem. This is a deliberate narrowing of the paper's `WorkingMemory` to what MCP can model well.

## Session lifecycle

1. **Init.** Agent calls `memory_session_init(session_id, project_hint, tags)` at the start of a session. If the session_id is new, a fresh buffer is created. If it exists, the stored buffer is returned for resume.
2. **Use.** Every subsequent `memory_recall` / `memory_apply_judgment` / `memory_reinforce` / `memory_write` call that includes `session_id` updates the buffer: recall appends matched IDs; reinforce bumps their scores; write adds the new ID.
3. **Snapshot.** Buffer is serialized to `sessions.working_memory` after each mutation (best-effort, non-blocking).
4. **Restore.** `memory_session_restore(session_id)` returns the buffer for agents that want an explicit re-hydrate outside of `init`.
5. **End.** `memory_session_end(session_id)` triggers a decay tick scoped to clusters that were accessed this session (not all clusters — targeted), optionally writes an L3 episode, and marks the session closed. Closed sessions are read-only.

Sessions are **optional**. All existing tools keep working without `session_id`. This keeps backward compat and lets harnesses adopt sessions incrementally.

## Buffer semantics

Buffer is a bounded list of `MemoryRef` ordered by composite score descending, with a budget cap. Reuses the existing `WorkingMemory.Buffer` type (`internal/memory/types.go:86`).

Eviction: when the buffer exceeds `BudgetMax` (from config), evict the lowest-scored entry. Score = `similarity * max(retention, 0.01)` (same formula as `apply_judgment`). Ties broken by `accessed_at` descending (newer wins).

Budget default: 50 items. Config field: `[session] buffer_budget_max = 50`.

## Migration 4 (introduced by 6b)

```sql
-- Extend sessions table.
ALTER TABLE sessions ADD COLUMN project_hint TEXT DEFAULT '';
ALTER TABLE sessions ADD COLUMN tags TEXT DEFAULT '[]'; -- JSON array, same shape as facts.tags
ALTER TABLE sessions ADD COLUMN created_at TEXT DEFAULT (datetime('now'));
ALTER TABLE sessions ADD COLUMN closed_at TEXT; -- nullable; set by session_end

CREATE INDEX IF NOT EXISTS idx_sessions_updated ON sessions(updated_at DESC);
```

`working_memory` column (already in schema) is reused: stores JSON-serialized buffer + budget metadata. Format:

```json
{
  "buffer": [{"id": "...", "layer": "l2_semantic", "score": 0.82, "content": "..."}],
  "budget_used": 12,
  "budget_max": 50
}
```

## Tool / resource API surfaces

### 6b — session plumbing

Add optional `session_id` to existing tools:

```go
// Added to: RecallInput, ApplyJudgmentInput, ReinforceInput, WriteInput
SessionID string `json:"session_id,omitempty" jsonschema:"attach this call to a session; buffer updates on recall/write/reinforce"`
```

Handler side-effect when `session_id != ""`:

- `recall`: append each returned candidate's ID to the session buffer (with computed score). Respect budget: evict if needed.
- `apply_judgment`: replace the buffer entries for this recall with the filtered set (drops rejected candidates).
- `reinforce`: re-score each ID in the buffer.
- `write`: append the new memory ID to the buffer with score 1.0.

If `session_id` is provided but unknown, the call **errors**: `fmt.Errorf("session not found: %s; call memory_session_init first", id)`. Silent session creation would hide typos.

### 6c — session tools

```go
// memory_session_init
type SessionInitInput struct {
    SessionID   string   `json:"session_id" jsonschema:"stable session identifier (client-generated)"`
    ProjectHint string   `json:"project_hint,omitempty"`
    Tags        []string `json:"tags,omitempty"`
}

type SessionInitOutput struct {
    SessionID   string       `json:"session_id"`
    Created     bool         `json:"created"`   // true if new; false if resuming
    Buffer      []MemoryRef  `json:"buffer"`    // empty for new sessions
    ProjectHint string       `json:"project_hint"`
    Tags        []string     `json:"tags"`
    CreatedAt   string       `json:"created_at"`
    ClosedAt    *string      `json:"closed_at,omitempty"` // set if previously closed
}
```

Behavior:

- New session: insert row, empty buffer, `Created=true`.
- Resume: load buffer + meta, `Created=false`. If `ClosedAt` is set → error: `session closed, cannot resume`.
- Tags / project_hint on resume: if provided, MERGE (tags add, project_hint replaces if non-empty). Allows incremental enrichment.

```go
// memory_session_snapshot
type SessionSnapshotInput struct { SessionID string `json:"session_id"` }
type SessionSnapshotOutput struct {
    Persisted bool   `json:"persisted"`
    UpdatedAt string `json:"updated_at"`
}
```

Forces a write of the current buffer. Normally implicit after each mutation, but this lets harnesses request an explicit checkpoint.

```go
// memory_session_restore
type SessionRestoreInput struct { SessionID string `json:"session_id"` }
type SessionRestoreOutput struct {
    Buffer      []MemoryRef `json:"buffer"`
    ProjectHint string      `json:"project_hint"`
    Tags        []string    `json:"tags"`
    UpdatedAt   string      `json:"updated_at"`
    ClosedAt    *string     `json:"closed_at,omitempty"`
}
```

Pure read. Useful when a harness wants to inspect a session without `init` semantics (e.g., audit tools).

```go
// memory_session_end
type SessionEndInput struct {
    SessionID string          `json:"session_id"`
    Episode   *EpisodePayload `json:"episode,omitempty" jsonschema:"if set, write an L3 episode summarizing the session"`
}

type SessionEndOutput struct {
    SessionID          string `json:"session_id"`
    EpisodeID          string `json:"episode_id,omitempty"`
    ClustersTicked     int    `json:"clusters_ticked"`
}
```

Behavior:

1. Load session. If already closed, error.
2. Extract cluster IDs touched by the buffer.
3. Call `MemoryManager.TickDecay(ctx, clusterIDs)` — scoped tick (reset `turns_since=0` for accessed clusters, bump all others). This is what the defunct Phase 3 `session_end=true` flag was trying to do — here it has a real home.
4. If `Episode` payload provided, write an L3 episode with `tags` including a `"session:<id>"` tag for traceability, linked to any fact IDs currently in the buffer.
5. Mark session `closed_at = now()`.

### 6d — resource + prompt

```go
// reverie://sessions/{id}
// Same shape as SessionRestoreOutput with added:
type sessionResourceResponse struct {
    SessionID   string      `json:"session_id"`
    Buffer      []MemoryRef `json:"buffer"`
    ProjectHint string      `json:"project_hint"`
    Tags        []string    `json:"tags"`
    CreatedAt   string      `json:"created_at"`
    UpdatedAt   string      `json:"updated_at"`
    ClosedAt    *string     `json:"closed_at,omitempty"`
    BufferBudget struct {
        Used int `json:"used"`
        Max  int `json:"max"`
    } `json:"buffer_budget"`
}
```

Update prompts (`internal/mcpserver/prompts.go`):

- `session_start` prompt accepts `session_id` argument. Text changes to:
  1. Call `memory_session_init` with the session_id.
  2. If `Created=false`, inspect the returned buffer — prior session context.
  3. Read `reverie://l1/index` as today.
  4. Call `memory_recall` with `session_id` so buffer auto-updates.
- `session_end` prompt: "call `memory_session_end` with the session_id, optionally with an episode payload summarizing the session."

## Behavior spec — shared rules

- **Sessions are opt-in.** Every tool still works without a session_id. Don't break anyone who hasn't adopted sessions.
- **Unknown session_id is an error, never silent-create.** Only `session_init` creates.
- **Buffer eviction is deterministic.** Lowest score first, ties by oldest `accessed_at`. Makes tests straightforward.
- **Snapshot failures don't fail the parent call.** If the session snapshot write errors (disk full, DB locked), log a warning and continue — the recall/write/reinforce succeeded. Next snapshot will persist.
- **Closed sessions are read-only.** Resource reads work; tool calls with a closed session_id error.

## Task breakdown (sequential, one subagent per sub-phase)

| # | Owner | Files touched | Depends on |
|---|---|---|---|
| 6a | primary + Plan agent (no impl) | `docs/design/phase-6-working-memory.md` finalization (this doc) | — |
| 6b | subagent | `internal/db/migrations.go` (migration 4), `internal/memory/store.go` + impls (session CRUD), `internal/memory/session_buffer.go` (new, buffer mutation helpers with eviction), `internal/mcpserver/tools.go` (add SessionID to RecallInput/ApplyJudgmentInput/ReinforceInput/WriteInput; buffer-update side effects), `internal/config/config.go` (BufferBudgetMax) | 6a |
| 6c | subagent | `internal/mcpserver/tools.go` (4 new handlers), `internal/mcpserver/server.go` (register), store method for ClusterIDsInBuffer helper | 6b |
| 6d | subagent | `internal/mcpserver/resources.go` (sessions resource), `internal/mcpserver/prompts.go` (updated prompts), `docs/*.md` (replacing-auto-memory.md and CLAUDE.md preamble updates) | 6c |

No parallelism within Phase 6 — each sub-phase hard-depends on the previous.

## Test matrix

### 6b
- Migration 4: fresh DB → extended `sessions` columns exist.
- Migration 4 on existing DB: new columns added with defaults; existing `sessions` rows (should be none) untouched.
- Store session CRUD: insert, get, list, update buffer, mark closed.
- Buffer eviction at budget limit: lowest-scored item drops.
- Buffer eviction tie-break: oldest `accessed_at` drops first.
- `memory_recall` with `session_id`: returned candidates appear in buffer; budget respected.
- `memory_recall` without `session_id`: no buffer interaction (regression guard).
- `memory_recall` with unknown `session_id`: error.
- `apply_judgment` with `session_id`: buffer entries filtered to judged set.
- `reinforce` with `session_id`: buffer entries re-scored.
- `write` with `session_id`: new ID lands in buffer.
- Snapshot failure mid-recall: recall succeeds, warning logged. (Use test hook to inject DB error.)

### 6c
- `session_init` new: empty buffer, `Created=true`.
- `session_init` resume: buffer populated from last snapshot.
- `session_init` resume of closed session: error.
- `session_init` resume with new project_hint/tags: merge semantics.
- `session_snapshot` persists current state.
- `session_restore` returns expected shape.
- `session_end`: scoped tick hits only accessed clusters; others bump.
- `session_end` with episode payload: episode written, linked to buffer's fact IDs.
- `session_end` on already-closed session: error.

### 6d
- `reverie://sessions/{id}` returns correct shape.
- `reverie://sessions/{unknown}` errors cleanly.
- Updated `session_start` prompt text references `memory_session_init` and `session_id` argument.
- Updated `session_end` prompt text references `memory_session_end`.

## Rollout

1. `go test ./...` green.
2. Manual smoke:
   - Init session "test-1", recall + write + reinforce with session_id, confirm buffer accumulates.
   - Kill server, restart, `memory_session_restore("test-1")` → same buffer returns.
   - `session_end` with an episode → episode written, session closed.
   - Attempt to use session "test-1" again → error.
3. Update `~/.claude/CLAUDE.md` preamble:
   - Recommend initializing a session per conversation using a stable id (e.g., project name + date).
   - Point to the `session_start` / `session_end` prompts.
4. Update `docs/replacing-auto-memory.md` — describes how sessions replace the auto-memory "conversation file" concept.

## Forward references

- A future phase could add cross-session search ("what did we decide in session X?") — buffer IDs per session are queryable by design.
- Phase 5's `daily_stats` schema could be extended with `sessions_started` / `sessions_ended` counters via triggers on the sessions table. Out of scope here but the shape permits it.

## Open questions retired in 6a

- ~~Does reverie own raw turns?~~ No. Harness owns. Decided above.
- ~~Is session_id server-generated or client-generated?~~ Client-generated. Server validates uniqueness but does not mint IDs. Rationale: harnesses already have conversation IDs they can reuse.
- ~~How does session_end relate to decay_tick?~~ `session_end` is the scoped version of `decay_tick`. Phase 3 removed the dead `session_end=true` flag from `decay_tick` precisely because this tool is its real home.
