# Phase 3 — Write fidelity

Status: **IMPLEMENTED** (`c9f57f6` dry_run, `6ea69f8` writable confidence, `fbfad14` decay_tick cleanup, `787838c` batched forget)
Prereqs: Phase 1 merged (tags wiring, `memory_get` shape)
Unblocks: nothing downstream depends on Phase 3 hard; Phase 4 is independent.

## Goal

Make the write path honest. Today `memory_write` hardcodes confidence, silently overwrites near-duplicates without preview, and exposes half-implemented knobs on `memory_decay_tick` that do nothing. `memory_forget` takes one ID at a time even though batch deletion is a common curation pattern. Phase 3 fixes all four: dry-run preview of supersede, writable confidence, dead-flag cleanup, and batch forget.

## Non-goals

- No change to cluster assignment behavior (that's Phase 2's territory).
- No change to the conflict-detection threshold (`ConflictThreshold` stays at 0.92 default).
- No change to the existing auto-supersede behavior when `dry_run=false` — Phase 3 only adds the preview path; the default remains "write and supersede automatically" to avoid breaking existing callers.

## Tool / resource API surfaces

### 3A — `dry_run` on `memory_write`

Add a flag that runs the full write pipeline except the INSERT/supersede, and returns what *would* happen.

```go
type WriteInput struct {
    // ... existing: Content, Type, Tags, Source, Episode ...
    DryRun bool `json:"dry_run,omitempty" jsonschema:"if true, preview cluster assignment and supersede candidate without writing"`
}

type WriteOutput struct {
    ID       string          `json:"id,omitempty"` // empty for dry-run
    Layer    string          `json:"layer"`
    DryRun   bool            `json:"dry_run"`      // echoes input for clarity
    Preview  *WritePreview   `json:"preview,omitempty"` // non-nil when DryRun=true
}

type WritePreview struct {
    ProposedClusterID   string            `json:"proposed_cluster_id"`
    ProposedClusterIsNew bool             `json:"proposed_cluster_is_new"`
    ProposedSupersedes  *SupersedeCandidate `json:"proposed_supersedes,omitempty"` // nil if no conflict
    ContentHash         string            `json:"content_hash"`
}

type SupersedeCandidate struct {
    ID         string  `json:"id"`
    Content    string  `json:"content"`
    Similarity float32 `json:"similarity"`
    Subtype    string  `json:"subtype"`
    CreatedAt  string  `json:"created_at"`
}
```

Behavior for `DryRun=true`:

1. Validate input (same validation as committed write).
2. Embed content / episode fields.
3. Run cluster assignment logic (`Assign`), but **do not call** `AfterInsert` (no centroid update, no tick).
4. Run conflict detection (`FindSimilarFacts` for facts; episodes have no conflict detection — preview omits supersede candidate).
5. Return `Preview` populated; no row inserted; no embedding cache write-through (see below).

Embedding cache policy during dry-run: the embedding is computed via the same `embedder.Embed` call that production writes use, so if the embedder uses a write-through cache (`CachedProvider`), the vector for this content hash is cached. That's fine — it's just a cache, same behavior as recall. No side effect on the memory store.

Error cases match committed write (invalid subtype, missing content/episode, etc.).

Non-episode dry-run for episodes: supported — only skips supersede (episodes don't supersede). Returns `ProposedSupersedes=nil`.

### 3B — writable `confidence` on `memory_write`

```go
type WriteInput struct {
    // ... existing ...
    Confidence float64 `json:"confidence,omitempty" jsonschema:"confidence in [0,1]; default 1.0"`
}
```

Behavior:

- Default: 1.0 (matches current hardcoded value — backwards compatible).
- Validation: `0.0 <= confidence <= 1.0`. Values outside range: error.
- Stored on the `facts.confidence` column (already exists — just stop hardcoding).
- Episodes: field ignored — episodes don't carry confidence. If `confidence` is set on an episode write, return an error to prevent silent loss. (Same pattern as tags silently dropping, which we're fixing elsewhere — don't replicate the bug.)

Recall / list already plumb `confidence` through. No changes needed there.

### 3C — clean half-implemented flags on `memory_decay_tick`

Current `DecayTickInput`:

```go
type DecayTickInput struct {
    TurnsElapsed int  `json:"turns_elapsed,omitempty"` // "ignored in Phase 2" per schema
    SessionEnd   bool `json:"session_end,omitempty"`   // handler is no-op
}
```

`SessionEnd=true` and `=false` produce identical behavior in `handleDecayTick` today. `TurnsElapsed` is documented-as-ignored.

Decision: **remove both**. Replace with a minimal, honest input:

```go
type DecayTickInput struct {
    // Reserved. No fields — tick is parameterless today.
}
```

(Or, if the SDK requires at least one field for schema generation, keep `Note string `json:"note,omitempty"` jsonschema:"optional log annotation"`` purely for audit.)

The CLAUDE.md global preamble and any docs referencing `session_end=true` must be updated to drop the flag. Grep hits:

- `docs/replacing-auto-memory.md` — check + update if mentioned.
- `internal/mcpserver/prompts.go` — `session_end` prompt currently says "call memory_decay_tick with session_end=true". Update to "call memory_decay_tick".
- `README.md` — scan.

If we want to preserve a session-end hook for future use (Phase 6), add a separate tool `memory_session_end` in Phase 6 instead of overloading this one. Don't keep dead flags as placeholders.

### 3D — batch `memory_forget`

Extend `ForgetInput` to accept an array. Keep `id` and `query` for backwards compat.

```go
type ForgetInput struct {
    ID    string   `json:"id,omitempty" jsonschema:"ID of a specific memory to delete"`
    IDs   []string `json:"ids,omitempty" jsonschema:"batch delete multiple memories by ID"`
    Query string   `json:"query,omitempty" jsonschema:"query to find candidates (returns without deleting)"`
}

type ForgetOutput struct {
    Deleted     int               `json:"deleted"`
    Failed      []ForgetFailure   `json:"failed,omitempty"`
    Candidates  []ForgetCandidate `json:"candidates,omitempty"` // query mode
}

type ForgetFailure struct {
    ID     string `json:"id"`
    Reason string `json:"reason"`
}
```

Validation: exactly one of `id`, `ids`, `query`. Multiple set → error.

Batch behavior:

1. Iterate IDs. For each, attempt fact-then-episode delete (same as single-ID path).
2. Collect failures (not-found, store error) into `Failed`. Continue on error — do not abort the batch. Caller decides what to do with partial success.
3. Return `Deleted` = successful count, `Failed` = per-ID reasons.
4. Single-transaction is NOT required — partial success is explicitly allowed and useful.

## Behavior spec — shared rules

- **Dry-run is stateless except for embedding cache.** No `turns_since` bump, no `accessed_at` update, no centroid drift, no superseded_by flip. Calling dry-run in a loop costs only embedding compute (cached after the first call for identical content).
- **Confidence is write-once per fact.** Phase 3 adds write-time setting; Phase 2's `memory_update_content` does NOT accept a new confidence. If the operator wants to change confidence on an existing fact, they forget + rewrite (cost of that decision is explicit). We can add `memory_update_confidence` later if needed.
- **Batch forget is best-effort.** The caller sees exactly which IDs succeeded / failed. This matches how a curator thinks: "delete these 5 typos — tell me which didn't exist."

## Task breakdown

| # | Owner | Files touched | Depends on | Blocks |
|---|---|---|---|---|
| 3A | subagent | `internal/mcpserver/tools.go` (WriteInput + WriteOutput + writeFact/writeEpisode branches) | Phase 1 | — |
| 3B | subagent | `internal/mcpserver/tools.go` (WriteInput + writeFact) | Phase 1 | — |
| 3C | subagent | `internal/mcpserver/tools.go` (DecayTickInput), `internal/mcpserver/prompts.go` (prompt text), `docs/*.md` grep+fix, `README.md` | — | — |
| 3D | subagent | `internal/mcpserver/tools.go` (ForgetInput + handleForget) | — | — |

All four parallelizable — they touch distinct regions of `tools.go` and can merge in any order with rebase. Schedule 3C last since it crosses docs and is the widest-grep.

## Test matrix

### 3A
- `dry_run=true` on fresh content → `ID=""`, `Preview.ProposedClusterID` set, `ProposedClusterIsNew=true` when applicable, `ProposedSupersedes=nil`.
- `dry_run=true` on near-duplicate content → `ProposedSupersedes` populated with expected fact's metadata.
- `dry_run=true` → no new rows in `facts` table (verify via direct store read).
- `dry_run=true` → no cluster `turns_since` change.
- `dry_run=true` on episode → no supersede candidate; cluster assignment populated.
- `dry_run=false` (default) → existing behavior unchanged (regression guard).
- Dry-run + invalid input (bad subtype, empty content) → same validation errors as non-dry-run.

### 3B
- `confidence=0.7` → fact stored with 0.7, readable via `memory_list` / `memory_get`.
- `confidence` omitted → stored as 1.0.
- `confidence=1.5` → error.
- `confidence=-0.1` → error.
- `confidence` on episode write → error.

### 3C
- `memory_decay_tick` with no input still advances decay.
- `memory_decay_tick` with old-shape input containing `session_end` or `turns_elapsed` → error that explicitly names the removed flags (helps callers discover the breaking change).
- `session_end` prompt text no longer mentions the flag.
- Grep of `docs/` and `README.md` has no remaining `session_end=true` references.

### 3D
- Batch of 3 valid IDs → `Deleted=3`, `Failed=nil`.
- Batch mixing valid + nonexistent → `Deleted=valid count`, `Failed` entries for missing IDs.
- Batch with store error mid-loop → continues processing remaining IDs.
- Both `id` and `ids` set → error.
- `ids=[]` → error (ambiguous with other modes).
- Single-ID path (`id="..."`) unchanged — regression guard.

## Rollout

1. `go test ./...` green.
2. Manual smoke:
   - Write with `dry_run=true` and confirm no row appears.
   - Write with `confidence=0.5` → `memory_get` returns 0.5.
   - Call `memory_decay_tick` with old shape — expect error. Update any calling scripts.
   - Batch-forget 3 IDs including one bogus — confirm partial success.
3. Update CLAUDE.md global preamble:
   - Mention `dry_run=true` as the recommended first call when uncertain about supersede.
   - Remove `session_end=true` guidance.
   - Note confidence is writable.

## Forward references

- Phase 4's `memory_unsupersede` complements 3A: if dry-run predicted a supersede and the operator committed it but then regretted it, `memory_unsupersede` is the undo.
- Phase 6 may add `memory_session_end` as a first-class tool; this Phase deliberately does NOT reintroduce the flag on tick.
