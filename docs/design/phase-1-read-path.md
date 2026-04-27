# Phase 1 — Read-path truth

Status: **DESIGN** (not yet implemented)
Prereqs: none
Unblocks: Phase 2 (curation), Phase 4 (supersede chain exposure uses `memory_get`)

## Goal

Make the MCP read surface reflect what the store already knows. Today an agent can see that clusters exist, but can't see which memory is in which cluster, can't inspect a single memory's full record, and can't categorize anything beyond the 4 subtypes. Phase 1 fixes all three without changing any behavior — every change is additive.

## Non-goals

- No new write semantics (Phase 3).
- No cluster mutation (Phase 2).
- No supersede mutation (Phase 4); we only *expose* the chain here.
- No link mutation (Phase 4); we only *expose* existing links.
- No session/working memory (Phase 6).

## Migration framework (prerequisite, implemented by 1D)

Reverie currently re-runs `schema.sql` on every startup and relies on `IF NOT EXISTS`. That model doesn't support column additions. Phase 1 introduces a minimal versioned-migration system because Phase 1 (tags) and later phases (5, 6) need it.

Design:

```go
// internal/db/migrations.go
type migration struct {
    Version int
    Name    string
    SQL     string
}

var migrations = []migration{
    {1, "initial_schema", initialSchemaSQL}, // current schema.sql, verbatim
    {2, "add_tags_columns", addTagsSQL},     // Phase 1
    // future phases append here
}
```

On `db.Open`:

1. `CREATE TABLE IF NOT EXISTS schema_migrations (version INTEGER PRIMARY KEY, name TEXT, applied_at TEXT)`.
2. Read max applied version.
3. Apply each migration with version > current in a single transaction per migration, recording the row.

Backward compat: existing databases have all the "initial_schema" tables. When migration 1 runs against them, every statement is `IF NOT EXISTS` and no-ops. Migration 2 then applies cleanly.

`schema.sql` stays as the canonical initial snapshot (migration 1's SQL). Future migrations are pure `ALTER`/`CREATE` SQL strings embedded as Go constants or `//go:embed` files — 1D's choice, whichever is tidier.

## Schema changes

Migration 2 (`add_tags_columns`):

```sql
ALTER TABLE facts    ADD COLUMN tags TEXT DEFAULT '[]';
ALTER TABLE episodes ADD COLUMN tags TEXT DEFAULT '[]';
```

Tags are stored as a JSON array of strings (SQLite TEXT, parsed with `encoding/json` in Go). Rationale: SQLite's JSON1 extension supports indexing and queries, but for the v1 use case (substring match + equality) plain TEXT with Go-side parsing is enough and avoids JSON1-dependency. If tag queries become hot, we can add a `tags_facts` join table later without breaking the API.

No index on tags in Phase 1. Add one if recall-by-tag becomes a measured bottleneck.

## Type changes

`internal/memory/types.go`:

```go
type Fact struct {
    // ... existing fields ...
    Tags []string `json:"tags"` // JSON array, empty by default
}

type Episode struct {
    // ... existing fields ...
    Tags []string `json:"tags"`
}
```

`internal/memory/store.go` (`ListFilter`):

```go
type ListFilter struct {
    // ... existing fields ...
    // TagsAny filters to memories containing at least one of these tags.
    // Nil/empty means no tag filter.
    TagsAny []string `json:"tags_any"`
}
```

Persistence in `sqlite_store.go`: on insert/scan, marshal/unmarshal `tags` to/from the TEXT column. An empty or NULL column reads as `[]string{}`.

## Tool / resource API surfaces

### 1A — `memory_list` output additions

Append fields to `ListMemory` in `internal/mcpserver/tools.go`:

```go
type ListMemory struct {
    ID         string   `json:"id"`
    Content    string   `json:"content"`
    Layer      string   `json:"layer"`
    Subtype    string   `json:"subtype,omitempty"`
    Source     string   `json:"source,omitempty"`
    Confidence float64  `json:"confidence,omitempty"`
    CreatedAt  string   `json:"created_at"`
    AccessedAt string   `json:"accessed_at"`
    ClusterID  string   `json:"cluster_id"`          // NEW
    Tags       []string `json:"tags,omitempty"`      // NEW
}
```

`ListInput` gains a filter:

```go
type ListInput struct {
    // ... existing fields ...
    TagsAny []string `json:"tags_any,omitempty" jsonschema:"Filter to memories with at least one of these tags"`
}
```

`handleList` plumbs `in.TagsAny` into `filter.TagsAny`.

### 1B — `memory_recall` candidate additions

Append fields to `RecallCandidate`:

```go
type RecallCandidate struct {
    // ... existing fields ...
    ClusterID string   `json:"cluster_id"` // NEW
    Tags      []string `json:"tags,omitempty"` // NEW
}
```

The handler already has each candidate's underlying `Fact` or `Episode`; populate from there. `LinkedIDs` is already populated by existing code (tools.go:128–145).

No input change to `memory_recall` in this subagent — scoped filters come in Phase 2 (2E).

### 1C — `memory_get` tool

New tool. Takes a memory ID (fact or episode), returns the full record including supersede chain and links.

```go
type GetInput struct {
    ID string `json:"id" jsonschema:"memory ID (fact or episode)"`
}

type GetOutput struct {
    ID           string    `json:"id"`
    Layer        string    `json:"layer"` // "l2_semantic" or "l3_episodic"
    Content      string    `json:"content"` // rendered content (joined fields for episode)
    Subtype      string    `json:"subtype,omitempty"`
    Source       string    `json:"source,omitempty"`
    Confidence   float64   `json:"confidence,omitempty"`
    Tags         []string  `json:"tags,omitempty"`
    ClusterID    string    `json:"cluster_id"`
    ClusterSummary string  `json:"cluster_summary,omitempty"`
    CreatedAt    string    `json:"created_at"`
    AccessedAt   string    `json:"accessed_at"`
    ValidFrom    string    `json:"valid_from,omitempty"`    // facts only
    SupersededBy *string   `json:"superseded_by,omitempty"` // facts only
    Supersedes   []string  `json:"supersedes,omitempty"`    // IDs of facts this one replaced
    Links        []LinkRef `json:"links,omitempty"`         // cross-type links
    // Episode-only fields below are set when Layer=l3_episodic.
    Situation  string `json:"situation,omitempty"`
    Action     string `json:"action,omitempty"`
    Outcome    string `json:"outcome,omitempty"`
    Preemptive string `json:"preemptive,omitempty"`
}

type LinkRef struct {
    ID       string `json:"id"`
    Layer    string `json:"layer"`
    LinkType string `json:"link_type"`
}
```

Error cases:

- ID not found → `fmt.Errorf("memory not found: %s", id)`.
- Empty ID → `fmt.Errorf("id is required")`.

`Supersedes` is computed by a store query: `SELECT id FROM facts WHERE superseded_by = ?`. Implementation lives in the store.

Store addition:

```go
// in Store interface
GetFactSupersedes(ctx context.Context, id string) ([]string, error)
```

(Episodes do not supersede — field omitted for layer=l3.)

### 1D — Tags end-to-end

Owns:

- `internal/db/migrations.go` (new, migration framework).
- `internal/db/schema.sql` split into migration 1 if needed; keep the file as canonical initial snapshot.
- `internal/memory/types.go` `Tags` fields on `Fact` and `Episode`.
- `internal/memory/store.go` `ListFilter.TagsAny`.
- `internal/memory/sqlite_store.go` and `mem_store.go` — tag encode/decode, filter application.
- `internal/mcpserver/tools.go` `WriteInput.Tags` is ALREADY in the schema but silently dropped — wire it through `writeFact` and `writeEpisode` to the persisted `Fact.Tags` / `Episode.Tags`.

Validation:

- Tags are normalized: lowercased, trimmed, deduplicated, sorted at write time.
- Max 16 tags per memory, max 32 chars each. Reject with error on violation.
- Empty tags are stripped.

Subagents 1A, 1B, 1C do NOT touch schema or types — they read `Tags` from the struct once 1D has wired it. This is why 1D must land first.

### 1E — `reverie://l1/cluster/{id}` resource

Dedicated resource per the earlier decision. Returns the cluster's metadata plus paginated members.

URI template: `reverie://l1/cluster/{id}?limit=N&offset=N`

Response:

```go
type l1ClusterDetailResponse struct {
    Cluster l1ClusterEntry      `json:"cluster"` // same shape as entries in reverie://l1/index
    Members []l1ClusterMember   `json:"members"`
    Total   int                 `json:"total"`   // total members in cluster (for pagination)
    Limit   int                 `json:"limit"`
    Offset  int                 `json:"offset"`
}

type l1ClusterMember struct {
    ID         string   `json:"id"`
    Layer      string   `json:"layer"` // l2_semantic or l3_episodic
    Subtype    string   `json:"subtype,omitempty"`
    Content    string   `json:"content"` // truncated to 200 chars
    Tags       []string `json:"tags,omitempty"`
    CreatedAt  string   `json:"created_at"`
    AccessedAt string   `json:"accessed_at"`
}
```

Defaults: `limit=50`, `offset=0`. `limit` capped at 200.

Error cases:

- Cluster not found → return HTTP-ish 404 via MCP error: `fmt.Errorf("cluster not found: %s", id)`.
- Invalid `limit`/`offset` → return error.

Store additions for paginated membership:

```go
// in Store interface
ListFactsByCluster(ctx context.Context, clusterID string, limit, offset int) ([]Fact, error)
ListEpisodesByCluster(ctx context.Context, clusterID string, limit, offset int) ([]Episode, error)
CountFactsByCluster(ctx context.Context, clusterID string) (int, error)
CountEpisodesByCluster(ctx context.Context, clusterID string) (int, error)
```

Member ordering: by `created_at` ascending (oldest first — stable for pagination).

## Behavior spec — shared rules

- **Superseded facts are invisible to reads.** `memory_list`, `memory_recall`, and `l1/cluster/{id}` exclude them, matching existing `ListFacts` behavior. `memory_get` can return a superseded fact if asked by ID — that's the whole point of the history view.
- **Empty tags always serialize as `[]`, never `null`.** The `omitempty` on `Tags` in tool outputs is for compactness; the Go zero value `[]string(nil)` renders `null` via `encoding/json`, so normalize to `[]string{}` before returning if we want `[]`. Decision: emit `[]` (explicit) in tool results; omit only from DB-inert cases. 1D specifies this in the handler helpers.
- **`cluster_id` is always populated** on outputs that have a backing memory. No memory lives outside a cluster (store's default-cluster invariant).

## Task breakdown

| # | Owner | Files touched | Depends on | Blocks |
|---|---|---|---|---|
| 1D | subagent | `internal/db/migrations.go` (new), `internal/db/db.go`, `internal/db/schema.sql`, `internal/memory/types.go`, `internal/memory/store.go`, `internal/memory/sqlite_store.go`, `internal/memory/mem_store.go`, `internal/mcpserver/tools.go` (WriteInput.Tags wiring only) | — | 1A, 1B, 1C, 1E |
| 1A | subagent | `internal/mcpserver/tools.go` (ListMemory + handleList) | 1D | — |
| 1B | subagent | `internal/mcpserver/tools.go` (RecallCandidate + handleRecall) | 1D | — |
| 1C | subagent | `internal/mcpserver/tools.go` (new GetInput/GetOutput + handleGet), `internal/mcpserver/server.go` (register tool), `internal/memory/store.go` (GetFactSupersedes), `internal/memory/sqlite_store.go` + `mem_store.go` (implement GetFactSupersedes) | 1D | — |
| 1E | subagent | `internal/mcpserver/resources.go` (new handler + register), `internal/memory/store.go` (4 new methods), `internal/memory/sqlite_store.go` + `mem_store.go` (implement them) | 1D | — |

Landing order:

1. 1D merges first. Migration framework + tags end-to-end + `WriteInput.Tags` fix.
2. 1A, 1B, 1C, 1E branch from the post-1D main, work in parallel worktrees, merge independently.

## Test matrix

Each subagent must leave `go test ./...` green in its worktree. Specific tests required:

### 1D
- `internal/db/migrations_test.go`:
  - Fresh DB: all migrations apply, `schema_migrations` has expected rows.
  - Partially migrated DB: only pending migrations run.
  - Fully migrated DB: second `Open` call is a no-op.
  - Migration failure rolls back (no partial apply).
- `internal/memory/sqlite_store_test.go`:
  - Insert fact with tags → read back, tags round-trip (normalized).
  - Insert fact with no tags → reads as `[]string{}`.
  - Insert fact with 17 tags → rejected.
  - Insert fact with 33-char tag → rejected.
  - Duplicate/empty/mixed-case tags are normalized.
  - `ListFacts` with `TagsAny` filter returns only matches.
  - Same suite mirrored for `mem_store_test.go` and episodes.
- `internal/mcpserver/tools_test.go`:
  - `memory_write` with `tags` input persists and is readable via `memory_list`.

### 1A
- `memory_list` result includes `cluster_id` for every entry.
- `memory_list` result includes `tags` (empty list for untagged).
- `tags_any=["foo"]` filter returns only facts with `foo` tag.
- Existing `memory_list` tests remain green.

### 1B
- `memory_recall` candidates include `cluster_id` and `tags`.
- Existing recall tests remain green.
- `linked_ids` continues to work (regression guard).

### 1C
- `memory_get` by fact ID returns full record.
- `memory_get` by episode ID returns episode fields populated, supersede fields omitted.
- `memory_get` on superseded fact still returns it with `superseded_by` populated.
- `memory_get` on fact that supersedes others returns `supersedes` IDs.
- `memory_get` on fact with linked episodes returns links (bidirectional coverage).
- `memory_get` on nonexistent ID returns "not found" error.
- Empty ID returns "id is required" error.

### 1E
- `reverie://l1/cluster/{id}` returns cluster meta + first page of members.
- `limit` / `offset` pagination works across cluster boundaries.
- `limit` capped at 200.
- Unknown cluster returns error.
- Members order is stable (created_at asc).
- Superseded facts excluded from members.

## Rollout

1. Run `go test ./...` — must be green.
2. Manual smoke test: start `reverie serve`, use `memory_write` with a `tags: ["smoke"]` payload, `memory_list tags_any=["smoke"]` returns it, `memory_get` on its ID returns the full shape, `reverie://l1/cluster/{its_cluster_id}` shows it in members.
3. No change needed to `~/.claude/CLAUDE.md`; Phase 2 may update agent-facing guidance.

## Forward references

- Phase 2 (2E) adds `cluster_id`/`subtype`/`layer`/`tags` filters to `memory_recall` input — design of that filter uses the same `TagsAny` shape landed here.
- Phase 4 uses `memory_get`'s `supersedes` / `superseded_by` to drive `memory_unsupersede`.
- Phase 5 (5A) reuses the migration framework for new status counters.
