# Phase 4 — Links & supersede

Status: **IMPLEMENTED** (`44274c8` link/unlink/list_links, `34866d1` unsupersede). Note: Phase 7 (`3cd9c79`) retired the `memory_link*` tools and migrated `fact_episode_links` rows into `memory_edges`; this doc is preserved as historical context for the original cross-type link design.
Prereqs: Phase 1 merged (`memory_get` already exposes supersede chain and link IDs)
Unblocks: nothing downstream hard-depends on Phase 4.

## Goal

Make the cross-type link graph and temporal supersede chain mutable from the MCP. Today fact↔episode links can only be established at episode-write time (via `EpisodePayload.LinkedFactIDs`); there's no way to link two existing memories, remove a link, or see both directions. And `superseded_by` is set automatically but can never be reversed, which means every accidental near-duplicate is a permanent rewrite.

## Non-goals

- No change to auto-supersede trigger in `memory_write` (lives in Phase 3's dry-run).
- No change to link semantics at episode-write time (still accepts `LinkedFactIDs`).
- No new link types beyond `"evidence"` (only existing). Free-form `link_type` string is accepted, but we document `"evidence"` as canonical.

## Tool / resource API surfaces

### 4A — link operations

Three tools: create, remove, list.

```go
// memory_link
type LinkInput struct {
    FactID    string `json:"fact_id" jsonschema:"L2 fact ID"`
    EpisodeID string `json:"episode_id" jsonschema:"L3 episode ID"`
    LinkType  string `json:"link_type,omitempty" jsonschema:"default \"evidence\""`
}

type LinkOutput struct {
    FactID    string `json:"fact_id"`
    EpisodeID string `json:"episode_id"`
    LinkType  string `json:"link_type"`
    Created   bool   `json:"created"` // false if link already existed (idempotent)
}
```

Behavior:

- Validate both IDs exist (fact is a fact, episode is an episode — layer check).
- `link_type` defaults to `"evidence"`. Any non-empty string is accepted.
- Uses existing `Store.LinkFactEpisode` (already idempotent via `INSERT OR IGNORE`).
- Report `Created=false` when the row already existed. Requires a small store change: `LinkFactEpisode` currently returns only `error`; extend to return `(bool, error)` where bool = "row actually inserted." SQLite driver exposes `RowsAffected()` from `Exec` — use it.

```go
// memory_unlink
type UnlinkInput struct {
    FactID    string `json:"fact_id"`
    EpisodeID string `json:"episode_id"`
}

type UnlinkOutput struct {
    Deleted bool `json:"deleted"` // false if link didn't exist
}
```

Store addition:

```go
UnlinkFactEpisode(ctx context.Context, factID, episodeID string) (deleted bool, err error)
```

Implementation: `DELETE FROM fact_episode_links WHERE fact_id=? AND episode_id=?`, return `RowsAffected() > 0`.

```go
// memory_list_links
type ListLinksInput struct {
    MemoryID string `json:"memory_id" jsonschema:"fact or episode ID"`
}

type ListLinksOutput struct {
    MemoryID string        `json:"memory_id"`
    Layer    string        `json:"layer"`  // layer of the input memory
    Links    []LinkDetail  `json:"links"`
}

type LinkDetail struct {
    ID          string `json:"id"`          // the OTHER side of the link
    Layer       string `json:"layer"`       // "l2_semantic" or "l3_episodic"
    LinkType    string `json:"link_type"`
    ContentPreview string `json:"content_preview"` // truncated to 120 chars
}
```

Behavior:

- Resolve `memory_id` to fact or episode.
- If fact: call `GetFactLinks`, return episode-side details.
- If episode: call `GetEpisodeLinks`, return fact-side details.
- Error on unknown ID.

### 4B — `memory_unsupersede`

Reverse a supersede relationship: clear the superseded-by pointer on a fact so it becomes "active" again.

```go
type UnsupersedeInput struct {
    FactID string `json:"fact_id" jsonschema:"ID of the superseded fact to revive"`
}

type UnsupersedeOutput struct {
    FactID               string   `json:"fact_id"`
    PreviouslySupersededBy string `json:"previously_superseded_by"`
    Warning              string   `json:"warning,omitempty"` // set if the superseder is still active
}
```

Behavior:

1. Fetch fact by ID; error if not found or not a fact.
2. If `superseded_by` is nil: return error "fact is not superseded".
3. Record the current `superseded_by` value for the response.
4. Clear `superseded_by`. Update `accessed_at = now()`.
5. Check whether the formerly-superseding fact is still active (itself not superseded). If yes, populate `Warning`:
   `"both fact %s and fact %s are now active and may be treated as duplicates; consider memory_forget or memory_update_content on one of them"`
6. No cluster/centroid recompute (membership unchanged; only a flag flipped).

Store addition:

```go
ClearFactSuperseded(ctx context.Context, id string) (previouslySupersededBy string, err error)
```

## Behavior spec — shared rules

- **Idempotency.** `memory_link` on an existing link succeeds with `Created=false`. `memory_unlink` on a missing link succeeds with `Deleted=false`. `memory_unsupersede` on a non-superseded fact is an error (the operator is confused about state).
- **Link direction is symmetric at query time, asymmetric at creation.** The schema stores (fact_id, episode_id, link_type) one row per relationship. `memory_list_links` hides this by flipping based on which side you query from.
- **`memory_unsupersede` warns rather than refuses.** The operator might genuinely want two coexisting facts (e.g., "this was true in v1, this is true in v2"). That's their call. We warn but commit.

## Task breakdown

| # | Owner | Files touched | Depends on | Blocks |
|---|---|---|---|---|
| 4A | subagent | `internal/mcpserver/tools.go` (3 new handlers), `internal/mcpserver/server.go` (register), `internal/memory/store.go` (UnlinkFactEpisode; modify LinkFactEpisode signature), store impls | Phase 1 | — |
| 4B | subagent | `internal/mcpserver/tools.go` (new handler), `internal/mcpserver/server.go` (register), `internal/memory/store.go` (ClearFactSuperseded), store impls | Phase 1 | — |

4A and 4B are fully parallel — no shared files beyond tool registration.

Landing order: either first; rebase merge conflict in `server.go` (both add a line to register tools) is trivial.

## Test matrix

### 4A
- `memory_link` with valid fact + episode → success, `Created=true`.
- Repeat same link → success, `Created=false`.
- `memory_link` with fact ID that's actually an episode → error.
- `memory_link` with nonexistent IDs → error.
- Custom `link_type="cause"` → stored, readable via `memory_list_links`.
- `memory_unlink` on existing link → `Deleted=true`.
- `memory_unlink` on missing link → `Deleted=false`.
- `memory_list_links` from fact side: returns linked episodes.
- `memory_list_links` from episode side: returns linked facts.
- `memory_list_links` on memory with no links → empty array.
- Deletion cascade still works: delete fact → links gone (existing `ON DELETE CASCADE` regression guard).
- `LinkFactEpisode` store signature change: callers in existing code paths (`writeEpisode`) updated to ignore the new bool return or use it — verify the writeEpisode path still works after the signature change (regression guard).

### 4B
- `memory_unsupersede` on superseded fact: clears flag, returns previous superseder ID.
- `memory_unsupersede` on active fact: error.
- `memory_unsupersede` on nonexistent ID: error.
- `memory_unsupersede` on fact whose superseder is still active: warning populated.
- `memory_unsupersede` on fact whose superseder was itself superseded: no warning (the chain moved on).
- After unsupersede, `memory_recall` finds the formerly-hidden fact again (regression: verify that `ListFacts` / `GlobalSearch` no longer filter it out).

## Rollout

1. `go test ./...` green.
2. Manual smoke:
   - Write a fact, write a similar one → auto-supersede triggered.
   - `memory_get` on old fact → shows `superseded_by`.
   - `memory_unsupersede` on old fact → warning populated, both now active.
   - `memory_recall` returns both.
   - `memory_link` old fact to an arbitrary episode. `memory_list_links` on the episode shows the fact.
   - `memory_unlink` → gone.
3. Update CLAUDE.md:
   - Under "Reinforce & forget", add a note: "If you accidentally supersede a good fact (e.g., near-duplicate that was actually different), use `memory_unsupersede` to revive it; you may need to `memory_forget` the newer one."
   - Add link tools to the recall/introspection section.

## Forward references

- Phase 5 (5A) enriched status should include `superseded_count` — use `COUNT(*) FROM facts WHERE superseded_by IS NOT NULL` which gives operators visibility into how much history they're carrying.
- Phase 6 doesn't interact with links directly, but working-memory snapshots should NOT include link tables (they're queryable; buffer holds IDs only).
