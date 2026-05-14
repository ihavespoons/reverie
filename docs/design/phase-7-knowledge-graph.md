# Phase 7 -- Knowledge Graph

Status: **IMPLEMENTED** (`3cd9c79` migration 5 + store + MCP surface + decay + status, `7f47df1` README + .gitignore)
Prereqs: Phase 1 merged (migration framework); Phase 4 merged (`memory_link`/`memory_unlink`/`memory_list_links` -- retired by this phase).
Unblocks: Phase 7C (graph-aware recall -- deferred from this phase).

## Goal

The proto-graph (`fact_episode_links` L2<->L3 evidence links with one free-form `link_type`, plus implicit `cluster_id` grouping and `superseded_by` chains) is too narrow to support entity/relationship reasoning over codebases. Phase 7 promotes Reverie's edge story to a real knowledge graph: (a) generalize `memory_edges` to any memory->memory (L2/L3/entity), typed and weighted; (b) introduce a first-class `entities` layer for files/repos/libraries/concepts with full Ebbinghaus decay; (c) link memories to entities via `entity_mentions`. Graph-aware recall (`expand_via_graph`) is explicitly out of scope and deferred to Phase 7C.

## Non-goals

- No entity resolution beyond exact `(name, entity_type)` match + cosine-similarity fallback. No NER, fuzzy person-name matching, or co-reference.
- No automatic edge inference from co-occurrence -- edges are explicit caller actions (Gate-A pattern preserved; no internal LLM calls).
- No graph-aware recall in this phase -- `memory_recall` is untouched. Reserved for 7C.
- No remote graph DB option -- SQLite stays.
- No deprecation shims for `memory_link`/`memory_unlink`/`memory_list_links` -- they're removed outright (sole user; data migrated forward in migration #5).

## Locked decisions

| Decision | Choice | Why |
|---|---|---|
| Edge retention model | Compute on read as `min(src_retention, dst_retention)` | Avoids a third decay tick path; matches "edges as ephemeral relationships between durable nodes" semantics. No `retention` column on `memory_edges`. |
| Entity-embedding text format | `name + " (" + entity_type + ")"` | Deterministic; cache-friendly; encodes both fields so `("foo", "file")` and `("foo", "concept")` have different embeddings. |
| Entity similarity-dedup threshold | Reuse `cfg.Memory.SimilarityThreshold` | No new config field; same threshold the recall path uses. |
| Session-end entity reset | At `memory_session_end`, query `entity_mentions` for memories in `WorkingMemory.Buffer`; reset those entities' `turns_since=0` | No buffer schema change; uses existing buffered memory IDs as the access set. |
| `ReplaceEpisodeLinks` fate | Keep signature; retarget implementation at `memory_edges` with `edge_type='evidence'` | Minimizes Phase-1 surface change; preserves the `EpisodePayload.LinkedFactIDs` write path. |
| Edge `hops` cap | `memory_edge_list` accepts `hops` 1..3, default 1; Go-side BFS | Bounded traversal; 3 hops is sufficient for evidence chains without runaway expansion. |
| Free-form `edge_type` strings | Caller-supplied; canonical types documented in README (`evidence`, `causes`, `contradicts`, `supports`, `refines`, `depends_on`, `references`) | Same precedent as Phase 4's `link_type`; no enforcement at the schema layer. |

## Migration 5

Single transaction. Three CREATE TABLEs + two indexes + data move + DROP. The data move uses `COALESCE(link_type, 'evidence')` to defend against any NULLs that crept in before Phase 4's default was enforced.

```sql
CREATE TABLE memory_edges (
  src_id      TEXT NOT NULL,
  dst_id      TEXT NOT NULL,
  edge_type   TEXT NOT NULL,
  weight      REAL DEFAULT 1.0,
  created_at  TEXT,
  PRIMARY KEY (src_id, dst_id, edge_type)
);
CREATE INDEX idx_edges_src ON memory_edges(src_id);
CREATE INDEX idx_edges_dst ON memory_edges(dst_id);

CREATE TABLE entities (
  id           TEXT PRIMARY KEY,
  name         TEXT NOT NULL,
  entity_type  TEXT NOT NULL,
  embedding    BLOB,
  utility      REAL DEFAULT 0.5,
  frequency    REAL DEFAULT 0.5,
  turns_since  INTEGER DEFAULT 0,
  retention    REAL DEFAULT 1.0,
  last_access  TEXT,
  created_at   TEXT,
  UNIQUE(name, entity_type)
);

CREATE TABLE entity_mentions (
  memory_id  TEXT NOT NULL,
  entity_id  TEXT NOT NULL,
  role       TEXT,
  PRIMARY KEY (memory_id, entity_id)
);

INSERT INTO memory_edges (src_id, dst_id, edge_type, created_at)
SELECT fact_id, episode_id, COALESCE(link_type, 'evidence'), datetime('now')
FROM fact_episode_links;

DROP TABLE fact_episode_links;
```

`entity_mentions` and `memory_edges` deliberately have no FKs -- `src_id`/`dst_id`/`memory_id` are polymorphic over facts, episodes, and entities. Application-level integrity is sufficient (small blast radius for orphans; cascade handler in the Store layer keeps it tight). Initial entity `utility` / `frequency` / `retention` defaults mirror new clusters so the first decay tick can't immediately drop a fresh entity below threshold.

## Tool / resource API surfaces

Six new tools. Names line up with the existing taxonomy (`memory_edge_*` for the typed graph, `memory_entity_*` for the new entity layer).

### `memory_edge_add` -- replaces `memory_link`

```go
type EdgeAddInput struct {
    SrcID    string  `json:"src_id" jsonschema:"source memory ID (fact, episode, or entity)"`
    DstID    string  `json:"dst_id" jsonschema:"destination memory ID"`
    EdgeType string  `json:"edge_type" jsonschema:"free-form; canonical types in README"`
    Weight   float64 `json:"weight,omitempty" jsonschema:"default 1.0"`
}

type EdgeAddOutput struct {
    SrcID    string  `json:"src_id"`
    DstID    string  `json:"dst_id"`
    EdgeType string  `json:"edge_type"`
    Weight   float64 `json:"weight"`
    Created  bool    `json:"created"` // false if edge already existed (idempotent)
}
```

Validates both IDs resolve (fact, episode, or entity). `Weight` clamps to `(0, 100]` and defaults to 1.0. Idempotency keyed by `(src_id, dst_id, edge_type)`; a repeat call with a different `Weight` does not overwrite -- `Created=false` and the stored row's weight is returned.

### `memory_edge_remove` -- replaces `memory_unlink`

```go
type EdgeRemoveInput struct {
    SrcID    string `json:"src_id"`
    DstID    string `json:"dst_id"`
    EdgeType string `json:"edge_type"`
}

type EdgeRemoveOutput struct {
    Deleted bool `json:"deleted"`
}
```

All three fields required. Missing edges return `Deleted=false` (no error).

### `memory_edge_list` -- replaces `memory_list_links`

```go
type EdgeListInput struct {
    MemoryID string `json:"memory_id"`
    Hops     int    `json:"hops,omitempty" jsonschema:"1..3, default 1"`
}

type EdgeListOutput struct {
    MemoryID string       `json:"memory_id"`
    Layer    string       `json:"layer"` // l2_semantic / l3_episodic / entity
    Edges    []EdgeDetail `json:"edges"`
}

type EdgeDetail struct {
    OtherID        string  `json:"other_id"`        // ID at the OTHER end of this edge from the BFS frontier
    OtherLayer     string  `json:"other_layer"`     // l2_semantic / l3_episodic / entity
    EdgeType       string  `json:"edge_type"`
    Weight         float64 `json:"weight"`
    Distance       int     `json:"distance"`        // hops from MemoryID (1..Hops)
    ContentPreview string  `json:"content_preview"` // truncated to 120 chars
}
```

`Hops` outside `[1, 3]` is an error. Default 1 -- callers who want the old `memory_list_links` shape get it unchanged.

### `memory_entity_upsert`

```go
type EntityUpsertInput struct {
    Name       string `json:"name"`
    EntityType string `json:"entity_type" jsonschema:"file/repo/library/concept/person/command/..."`
}

type EntityUpsertOutput struct {
    EntityID            string `json:"entity_id"`
    Created             bool   `json:"created"`
    MatchedBySimilarity bool   `json:"matched_by_similarity"` // true if matched via cosine, not exact
}
```

Embedding is computed on insert from `name + " (" + entity_type + ")"`. Dedup order: exact `(name, entity_type)` -> cosine within same `entity_type` >= `SimilarityThreshold` -> insert. `MatchedBySimilarity=true` lets the caller log/inspect fuzzy hits.

### `memory_entity_mention`

```go
type EntityMentionInput struct {
    MemoryID  string   `json:"memory_id"`
    EntityIDs []string `json:"entity_ids"`
    Role      string   `json:"role,omitempty"` // advisory: subject/object/mention
}

type EntityMentionOutput struct {
    Inserted int `json:"inserted"` // count of new rows (idempotent existing rows excluded)
}
```

`MemoryID` must resolve to a fact or episode (not another entity -- mentions go memory->entity). `EntityIDs` empty -> error.

### `memory_entity_neighbors`

```go
type EntityNeighborsInput struct {
    EntityID string `json:"entity_id"`
    Hops     int    `json:"hops,omitempty" jsonschema:"1..3, default 1"`
}

type EntityNeighborsOutput struct {
    EntityID string             `json:"entity_id"`
    Memories []NeighborMemory   `json:"memories"`
    Entities []NeighborEntity   `json:"entities"`
}

type NeighborMemory struct {
    ID             string `json:"id"`
    Layer          string `json:"layer"` // l2_semantic / l3_episodic
    ContentPreview string `json:"content_preview"`
    Distance       int    `json:"distance"`
}

type NeighborEntity struct {
    ID         string `json:"id"`
    Name       string `json:"name"`
    EntityType string `json:"entity_type"`
    Distance   int    `json:"distance"`
}
```

`Memories` is populated from `entity_mentions` (entity -> memory edges are implicit). `Entities` is populated by walking `memory_edges` rows where both endpoints are entities.

### Resource additions

`reverie://status` -> `countStatus` gains two int fields: `Entities` and `Edges`. Both are cheap `COUNT(*)` queries run under the existing `statusSubtaskBudget`. An entity-retention histogram (mirror of the cluster one) is **not** in this phase -- see Forward references.

## Behavior spec

### Edge retention computation

`memory_edges` rows store no retention. When the system needs an edge's effective retention (e.g., future graph-aware recall scoring, debug surfaces) it reads it:

```
read_retention(edge_row):
    src_r = retention_for(edge_row.src_id)
    dst_r = retention_for(edge_row.dst_id)
    return min(src_r, dst_r)

retention_for(memory_id):
    if memory_id resolves to an entity:
        return entities.retention for that row
    elif memory_id resolves to a fact or episode:
        return clusters.retention for that memory's cluster_id
    else:
        return 0.0
```

Facts and episodes have no per-row `retention` column; they inherit their cluster's retention. An edge between a fact and an entity therefore uses `min(fact_cluster.retention, entity.retention)`. Edges are not surfaced as durable nodes in the decay loop -- they fade when either endpoint fades. Phase 7C will consume this helper for `expand_via_graph` scoring; it's not used in any user-facing tool surface in 7A/7B.

### Hop traversal (Go-side BFS)

`ListEdges(memoryID, hops)` walks edges from `memoryID` up to `hops` levels deep:

```
visited = {memoryID}
frontier = [memoryID]
results = []
for depth in 1..hops:
    next_frontier = []
    for node in frontier:
        rows = SELECT * FROM memory_edges WHERE src_id=node OR dst_id=node
        for row in rows:
            other = row.dst_id if row.src_id == node else row.src_id
            if other in visited: continue
            visited.add(other)
            results.append(EdgeDetail{OtherID: other, EdgeType: row.edge_type,
                                      Weight: row.weight, Distance: depth, ...})
            next_frontier.append(other)
    frontier = next_frontier
return results
```

Each edge contributes exactly one `EdgeDetail` (the depth at which BFS first reached the other endpoint). Cycles cannot duplicate rows because the visited set is keyed by node ID. `OtherLayer` and `ContentPreview` are filled in by a layered lookup (`GetFact` -> `GetEpisode` -> `GetEntity`) per emitted result.

### Entity dedup

`UpsertEntity(name, entityType)`:

```
1. row = SELECT * FROM entities WHERE name=? AND entity_type=?
   if row: return (row.id, created=false, matchedBySimilarity=false)
2. text = name + " (" + entity_type + ")"
   vec  = embedder.Embed(text)
3. candidates = SELECT id, embedding FROM entities WHERE entity_type=?
   for each c in candidates: score = cosine(vec, c.embedding)
4. best = argmax(score)
   if best.score >= cfg.Memory.SimilarityThreshold:
       return (best.id, created=false, matchedBySimilarity=true)
5. id = new uuid; INSERT entities(id, name, entity_type, embedding, ...)
   return (id, created=true, matchedBySimilarity=false)
```

Similarity is intentionally scoped to the same `entity_type` -- `("foo", "file")` should never collapse onto `("foo", "concept")` even at high cosine.

### Entity decay

`Decayer.RetentionFromState(utility, frequency, turnsSince) float64` is factored out of `decay/decayer.go` (Phase 7B / plan-P3). `Retention(ClusterNode)` becomes a thin wrapper that pulls the three fields off the cluster and delegates. Entities call `RetentionFromState` directly -- they do **not** route through `Retention(ClusterNode)`. Same formula, two callers.

`TickAllEntities(accessedIDs []string)` mirrors `TickAllClusters`:

```
in a single transaction:
    UPDATE entities SET turns_since = turns_since + 1
    UPDATE entities SET turns_since = 0, last_access = now() WHERE id IN (accessedIDs)
    for each entity: retention = RetentionFromState(utility, frequency, turns_since)
                     UPDATE entities SET retention = ? WHERE id = ?
```

The per-row retention rewrite happens after the bulk increment/reset so it sees final `turns_since` values.

### Session-end entity reset

At `memory_session_end`, after the existing scoped cluster tick:

```
memoryIDs = [m.ID for m in session.WorkingMemory.Buffer]
entityIDs = SELECT DISTINCT entity_id FROM entity_mentions WHERE memory_id IN (memoryIDs)
TickAllEntities(ctx, entityIDs)
log.Info("memory_session_end", "clusters_ticked", n_clusters, "entities_ticked", len(entityIDs))
```

Buffer carries only `MemoryRef`; the mentions table is the canonical source of buffered-memory -> entity translation. No buffer schema change.

### Cascade on memory delete

When a fact or episode is deleted (via `memory_forget` or the existing cascade paths):

- Delete all `entity_mentions` rows with `memory_id = <deleted>`.
- Delete all `memory_edges` rows with `src_id = <deleted>` OR `dst_id = <deleted>` (both directions).

Entities themselves are **not** cascade-deleted. An entity may be referenced by many memories; surviving references keep it alive, and an orphaned entity simply fades through `TickAllEntities` until decay drops it below threshold (operator can run `memory_forget` on entities later -- not in this phase). Cascade runs inside the same transaction as the fact/episode delete so a crashed delete cannot leave dangling edges.

## Task breakdown

| # | Owner | Files touched | Depends on | Blocks |
|---|---|---|---|---|
| 7A | subagent (plan-P1) | `internal/db/migrations.go` (+migration 5), `internal/db/migrations_test.go`, `internal/memory/types.go` (+Entity/Edge/Mention; -EpisodeLink/FactLink), `internal/memory/store.go` (interface diff), `internal/memory/sqlite_store.go` + `mem_store.go` (impls + rewire `writeEpisode`/`getEpisode`/`ReplaceEpisodeLinks`), store tests | Phase 1, Phase 4 | 7B |
| 7B | subagent (plan-P2 + P3) | `internal/mcpserver/server.go` (remove 3 / register 6 tools), `internal/mcpserver/tools.go` (-LinkInput etc., +6 handlers), `internal/mcpserver/tools_test.go`, `internal/decay/decayer.go` (+RetentionFromState), `internal/manager/manager.go` (TickDecay extension), `internal/mcpserver/resources.go` (countStatus + handler), `internal/mcpserver/resources_test.go` | 7A | -- |

7A is one large task -- the schema, types, store interface, store implementations, and non-tool rewires all move together. 7B splits cleanly into "MCP surface" (P2) and "decay + status" (P3) but lands in one PR; the second slice depends on the first only at file-touch level (no semantic dependency on tool handlers from the decay path). No parallelism within 7A or between 7A and 7B.

## Test matrix

| **Test name** | **Layer** | **What it asserts** |
|---|---|---|
| `TestMigration5_ForwardOnly` | db | Seed v4 DB with sample `fact_episode_links` rows; run migration; every row appears in `memory_edges` with `edge_type='evidence'` and `created_at` populated; `fact_episode_links` is gone. |
| `TestMigration5_Idempotent` | db | Second run after migration is recorded is a no-op; schema unchanged; row count unchanged. |
| `TestStore_AddEdge_HappyPath` | memory | `AddEdge(src, dst, "refines", 1.0)` returns `created=true`; row visible in `memory_edges`. |
| `TestStore_AddEdge_Idempotent` | memory | Repeat call -> `created=false`; row count = 1; weight unchanged. |
| `TestStore_RemoveEdge_Missing` | memory | Remove of nonexistent edge -> `deleted=false`, no error. |
| `TestStore_ListEdges_EmptyResult` | memory | Memory ID with no edges returns empty slice. |
| `TestStore_ListEdges_TwoHops` | memory | A->B, B->C edges; `ListEdges(A, 2)` returns both with `distance=1` and `distance=2` respectively. |
| `TestStore_UpsertEntity_Exact` | memory | Two upserts with same `(name, entity_type)` -> same ID, second `created=false`. |
| `TestStore_UpsertEntity_SimilarityMatch` | memory | Two entities with high-cosine names (same `entity_type`) -> second returns first's ID with `MatchedBySimilarity=true`. |
| `TestStore_UpsertEntity_CrossTypeNoMatch` | memory | Same name, different `entity_type` -> two distinct IDs. |
| `TestStore_AddEntityMentions_Idempotent` | memory | Same `(memory_id, entity_id)` twice -> `Inserted=1` then `Inserted=0`. |
| `TestStore_TickAllEntities_NoAccess` | memory | Insert entity; tick twice with empty access set; `turns_since=2`, `retention` strictly decreased. |
| `TestStore_TickAllEntities_Access` | memory | Tick with entity in access set; `turns_since=0`; `last_access` bumped. |
| `TestStore_CascadeOnFactDelete` | memory | Delete a fact that has both edges and mentions; matching `memory_edges` and `entity_mentions` rows gone; the linked entities remain. |
| `TestStore_EpisodeLinks_RoundTrip` | memory | Write episode with `EpisodePayload.LinkedFactIDs=[F1,F2]`; `getEpisode` returns the same two IDs -- guards the rewire of `writeEpisode`/`getEpisode` from `fact_episode_links` onto `memory_edges`. |
| `TestStore_ReplaceEpisodeLinks` | memory | `ReplaceEpisodeLinks(ep, [F2,F3])` after `[F1,F2]` -> only F2 and F3 remain as `evidence` edges. |
| `TestTool_EdgeAdd_LayerCheck` | mcpserver | Edge with nonexistent IDs -> error. |
| `TestTool_EdgeAdd_Idempotent` | mcpserver | Two `memory_edge_add` calls -> second returns `Created=false` with original `Weight`. |
| `TestTool_EdgeRemove_Missing` | mcpserver | Returns `Deleted=false` cleanly. |
| `TestTool_EdgeList_HopsCap` | mcpserver | `Hops=4` -> error; `Hops=0` -> defaults to 1. |
| `TestTool_EdgeList_TwoHops_BFS` | mcpserver | Built graph A-B-C; `EdgeList(A, hops=2)` returns two `EdgeDetail` entries; distances and layers correct. |
| `TestTool_EntityUpsert_DedupExact` | mcpserver | Second call with same args -> `Created=false`, `MatchedBySimilarity=false`. |
| `TestTool_EntityUpsert_DedupSimilarity` | mcpserver | Returns the existing ID with `MatchedBySimilarity=true`. |
| `TestTool_EntityMention_BadMemoryID` | mcpserver | Memory ID resolves to entity -> error (mentions are memory->entity only). |
| `TestTool_EntityNeighbors_MixedShape` | mcpserver | Entity with both memory mentions and entity-to-entity edges -> both arms of the output populated. |
| `TestManager_TickDecay_EntitiesIncluded` | manager | After `TickDecay`, accessed entities have `turns_since=0`, others incremented; `SetLastTick` called. |
| `TestSessionEnd_EntitiesReset` | mcpserver | Session buffer contains memory M with mention of entity E; `memory_session_end` resets E's `turns_since=0`; log line includes entity count. |
| `TestStatusResource_GraphCounts` | mcpserver | `reverie://status` includes `counts.entities` and `counts.edges`; values match DB `COUNT(*)`. |

## Rollout

1. 7A and 7B ship together in one PR (plan phases P1+P2+P3). No feature flag; straight replacement of the link tools.
2. Plan phases P4 (docs) and P5 (verification) run concurrently after the implementation PR is up: P4 updates `README.md`, `CLAUDE.md`, `AGENTS.md`, the Phase-2 design doc's forward reference, and the harness setup docs; P5 runs `gofmt -l ./...`, `go vet ./...`, `go test ./...`, `go build ./...`, the `fact_episode_links` grep audit, and the MCP JSON-RPC smoke.
3. Manual smoke after merge: `memory_write` two facts, `memory_entity_upsert` a file entity, `memory_entity_mention` the first fact, `memory_entity_neighbors` returns it; `memory_edge_add` between the two facts with `edge_type="refines"`, `memory_edge_list hops=2` returns both edges; `memory_decay_tick` and confirm entity `turns_since` advanced.

## Forward references

- **Phase 7C -- Graph-aware recall.** Adds `expand_via_graph: true` flag on `memory_recall`. Walks `memory_edges` and `entity_mentions` from seed memories returned by the cosine pass and scores neighbors by `seed_similarity * neighbor_retention * decay_per_hop`. Deferred from this phase pending real usage data to tune the scoring constants and the per-hop decay coefficient.
- **Entity retention histogram in `reverie://status`.** Mirror of the existing cluster histogram (same `RetentionBucket` shape from Phase 5). Easy follow-up if entity decay turns out to need operator visibility.
- **Edge typing analytics.** Track which `edge_type` strings the caller is actually using; helps decide whether to formalize the canonical set into a code-level constant or keep it free-form.
- **Entity forget.** A `memory_entity_forget` tool to actively prune orphaned entities; currently they fade only through decay. Punted until decay alone proves insufficient.
