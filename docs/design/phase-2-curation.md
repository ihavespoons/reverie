# Phase 2 — Curation tools

Status: **IMPLEMENTED** (`4879609` reassign_cluster, `aac9e0f` merge_clusters, `4712446` split_cluster, `424e507` update_content, `b4a6374` recall filters)
Prereqs: Phase 1 merged (`cluster_id` on results, `memory_get`, `reverie://l1/cluster/{id}`, tags)
Unblocks: Phase 3 (`dry_run` reuses supersede preview logic), Phase 4 (link ops don't depend on this but land after)

## Goal

Give the operator first-class control over cluster shape and fact content. Today the auto-clusterer's mistakes are permanent — a misclustered fact can only be forgotten and rewritten, which changes IDs and breaks episode links. Phase 2 adds reassign / merge / split / update-content + scoped recall so curation is a first-class workflow.

## Non-goals

- No auto-clustering tuning (threshold changes, algorithm swap). Out of scope — Phase 2 is about operator tools, not algorithm quality.
- No write-path changes (dry-run lives in Phase 3).
- No link mutation (Phase 4).
- `memory_update_content` does NOT auto-reassign cluster based on new embedding. Operator does that explicitly via `memory_reassign_cluster` if they want it. Keeps one-tool-one-job.

## Centroid invariants (shared rule)

Every curation op that moves memories between clusters must leave centroids consistent with membership. Centralize the logic in `internal/memory/cluster_recompute.go`:

```go
// RecomputeCentroid reads all (non-superseded) members of cluster c, averages
// their embeddings, and writes the result via UpdateClusterCentroid.
// Returns ErrEmptyCluster if the cluster has zero members after the operation
// (caller decides whether to delete the cluster or leave it empty).
func RecomputeCentroid(ctx context.Context, store Store, clusterID string) error
```

Called from 2A (after reassign), 2B (after merge into target), 2C (for each new cluster), 2D (NOT called — update content preserves cluster; its embedding changes but cluster stays put and centroid drifts within tolerance).

Empty clusters: operations that would leave a cluster with zero members must delete the cluster row to avoid ghost entries in `reverie://l1/index`. (The existing `l1/index` handler already filters `item_count==0`, but deleting is tidier and prevents orphans.)

## Tool / resource API surfaces

### 2A — `memory_reassign_cluster`

Move a single fact or episode to a different cluster.

```go
type ReassignClusterInput struct {
    MemoryID        string `json:"memory_id" jsonschema:"fact or episode ID"`
    TargetClusterID string `json:"target_cluster_id" jsonschema:"destination cluster ID (must exist)"`
}

type ReassignClusterOutput struct {
    MemoryID         string `json:"memory_id"`
    OldClusterID     string `json:"old_cluster_id"`
    NewClusterID     string `json:"new_cluster_id"`
    OldClusterDeleted bool  `json:"old_cluster_deleted"` // true if old cluster is now empty
}
```

Behavior:

1. Fetch memory by ID (fact first, then episode).
2. Fetch target cluster; error if not found.
3. Record `OldClusterID`.
4. Update `cluster_id` on the fact/episode row.
5. Recompute centroid for old and new clusters.
6. If old cluster is now empty, delete the cluster row; set `OldClusterDeleted=true`.
7. Reset `turns_since=0` on new cluster (treat as access).

Errors: memory not found; target cluster not found; memory already in target cluster (`fmt.Errorf("memory %s already in cluster %s", memoryID, targetClusterID)`).

Store additions:

```go
SetMemoryCluster(ctx context.Context, memoryID, clusterID string) error
DeleteCluster(ctx context.Context, id string) error
```

`SetMemoryCluster` handles both facts and episodes — internally tries fact first, then episode, errors if neither exists. `DeleteCluster` refuses to delete a cluster with members (safety guard).

### 2B — `memory_merge_clusters`

Merge N source clusters into one target. Members of sources get reparented to target; sources are deleted.

```go
type MergeClustersInput struct {
    SourceClusterIDs []string `json:"source_cluster_ids" jsonschema:"clusters to merge and delete (must be non-empty)"`
    TargetClusterID  string   `json:"target_cluster_id" jsonschema:"destination cluster (must exist, cannot be in source list)"`
}

type MergeClustersOutput struct {
    TargetClusterID string   `json:"target_cluster_id"`
    MergedCount     int      `json:"merged_count"`     // total memories moved
    DeletedClusters []string `json:"deleted_clusters"` // source IDs that were deleted
    NewItemCount    int      `json:"new_item_count"`   // target cluster's item_count after merge
}
```

Behavior:

1. Validate: `SourceClusterIDs` non-empty, target not in sources, all IDs exist.
2. For each source: move all members to target (single `UPDATE` per source table).
3. Delete each source cluster row.
4. Recompute target centroid.

Errors: target in source list; any ID not found; source list empty.

Atomicity: all-or-nothing — wrap in a transaction. Store additions:

```go
MoveAllClusterMembers(ctx context.Context, sourceClusterID, targetClusterID string) (moved int, err error)
```

Called once per source inside a `WithTx` helper (add one if not present).

### 2C — `memory_split_cluster`

Partition a cluster's members into new clusters by explicit ID groups.

```go
type SplitClusterInput struct {
    ClusterID string     `json:"cluster_id" jsonschema:"cluster to split (must exist)"`
    Groups    [][]string `json:"groups" jsonschema:"non-overlapping partitions; each group becomes a new cluster"`
    Metas     []ClusterMeta `json:"metas,omitempty" jsonschema:"optional metadata for each new cluster (summary, domain, meta_instr). Must be same length as groups if provided."`
}

type ClusterMeta struct {
    Summary   string `json:"summary,omitempty"`
    Domain    string `json:"domain,omitempty"`
    MetaInstr string `json:"meta_instr,omitempty"`
}

type SplitClusterOutput struct {
    SourceClusterID   string   `json:"source_cluster_id"`
    NewClusterIDs     []string `json:"new_cluster_ids"` // same length as groups
    SourceDeleted     bool     `json:"source_deleted"`  // true if all members were partitioned
    RemainingInSource int      `json:"remaining_in_source"` // members not in any group
}
```

Behavior:

1. Validate: cluster exists; all member IDs in groups belong to this cluster; no ID appears in two groups; groups non-empty.
2. For each group: create a new cluster (UUID). Apply optional metadata from `Metas[i]`. Move each group's members to it. Recompute new cluster centroid.
3. Recompute source centroid with remaining members. If source is empty, delete it and set `SourceDeleted=true`.
4. Partial splits allowed: IDs not in any group stay in source. That is what `RemainingInSource` reports.

Errors: cluster not found; any ID not a member of the cluster; overlap between groups; empty group.

### 2D — `memory_update_content`

Amend a fact or episode's content in place. Preserves ID, cluster, links, timestamps. Re-embeds.

```go
type UpdateContentInput struct {
    ID      string          `json:"id" jsonschema:"fact or episode ID"`
    Content string          `json:"content,omitempty" jsonschema:"new content (for facts)"`
    Episode *EpisodePayload `json:"episode,omitempty" jsonschema:"new episode fields (for episodes); omit linked_fact_ids to preserve existing"`
    Tags    *[]string       `json:"tags,omitempty" jsonschema:"replace tags; omit to preserve"`
}

type UpdateContentOutput struct {
    ID        string `json:"id"`
    Layer     string `json:"layer"`
    Reembedded bool  `json:"reembedded"` // always true for successful updates
}
```

Behavior:

1. Resolve ID to fact or episode.
2. Validate: layer-matching payload (content for facts, episode for episodes).
3. Compute new embedding + content hash.
4. Update row: content/fields, embedding, content_hash, `accessed_at = now()`. `created_at` unchanged.
5. If the payload supplies `tags` (pointer non-nil), replace; otherwise preserve.
6. For episodes: if `EpisodePayload.LinkedFactIDs` is non-nil and non-empty, REPLACE the current link set with it. If nil, preserve existing. If explicit empty slice, clear links.
7. Do NOT trigger conflict detection / supersede. Update is not a write.
8. Do NOT reassign cluster. Centroid drifts; recompute is not triggered here.

Errors: ID not found; layer mismatch (episode payload to a fact ID); both `content` and `episode` set; neither set.

Store additions:

```go
UpdateFactContent(ctx context.Context, id, content, contentHash string, embedding []float32, tags []string) error
UpdateEpisodeContent(ctx context.Context, id string, e Episode) error // e has updated fields + embedding
ReplaceEpisodeLinks(ctx context.Context, episodeID string, factIDs []string) error
```

### 2E — Recall filters

Add optional filters to `memory_recall`. Applied post-cosine, pre-ranking. Matches Phase 1's `TagsAny` semantics for consistency.

```go
type RecallInput struct {
    // ... existing: Query, Limit, Hints, Round ...
    ClusterID string   `json:"cluster_id,omitempty" jsonschema:"restrict to members of this cluster"`
    Subtype   string   `json:"subtype,omitempty" jsonschema:"restrict to this fact subtype"`
    Layer     string   `json:"layer,omitempty" jsonschema:"l2, l3, or both (default)"`
    TagsAny   []string `json:"tags_any,omitempty" jsonschema:"restrict to memories with any of these tags"`
}
```

Behavior:

1. Embed query, run `GlobalSearch` as today.
2. Filter result set: drop candidates whose cluster_id, subtype, layer, or tags don't match (when respective filter is set). Empty filter = pass-through.
3. Continue with existing ranking (similarity, gate annotations, cache put).

Rationale for post-filter (not SQL-side): keeps `GlobalSearch` signature stable and filters are cheap on the candidate set. If recall volume grows, push to SQL later.

No store changes for 2E; all logic in `handleRecall`.

## Behavior spec — shared rules

- **All curation ops update `accessed_at = now()`** on moved memories — they're being touched. 
- **Centroid recompute uses only non-superseded members.** Superseded facts still have embeddings in the DB but are excluded from centroid math and member counts.
- **Reassigning a fact that's the last member of its old cluster deletes the old cluster.** Same for merge target exhaustion. The `l1/index` already filters empty clusters, but we clean up the row too.
- **Split + merge are transactional.** Either the whole operation lands or nothing does. Reassign is also transactional (single row update + two centroid updates in one tx).
- **Update-content preserves cluster, links, timestamps** (except `accessed_at`). ID stable — any external reference (reinforcement logs, episode links) keeps working.

## Task breakdown

| # | Owner | Files touched | Depends on | Blocks |
|---|---|---|---|---|
| 2A | subagent | `internal/mcpserver/tools.go` (new handler + register), `internal/memory/store.go` (SetMemoryCluster, DeleteCluster), `sqlite_store.go` + `mem_store.go`, `internal/memory/cluster_recompute.go` (new shared helper) | Phase 1 | 2B, 2C use `cluster_recompute.go` |
| 2B | subagent | `internal/mcpserver/tools.go` (new handler), `internal/memory/store.go` (MoveAllClusterMembers, transaction helper), store impls | 2A (for shared helper) | — |
| 2C | subagent | `internal/mcpserver/tools.go` (new handler), reuses helpers from 2A/2B | 2A (for shared helper) | — |
| 2D | subagent | `internal/mcpserver/tools.go` (new handler), `internal/memory/store.go` (UpdateFactContent, UpdateEpisodeContent, ReplaceEpisodeLinks), store impls | Phase 1 | — |
| 2E | subagent | `internal/mcpserver/tools.go` (RecallInput + handleRecall filter logic) | Phase 1 (cluster_id on candidates) | — |

Landing order:

1. **2A first** — it introduces `cluster_recompute.go` which 2B/2C depend on. Must merge before 2B/2C open worktrees.
2. **2B, 2C, 2D, 2E in parallel** after 2A lands.

## Test matrix

### 2A
- Reassign fact: `cluster_id` updates, both centroids recomputed, `accessed_at` bumped.
- Reassign to same cluster: error.
- Reassign to nonexistent cluster: error.
- Reassign nonexistent memory: error.
- Reassign last member: old cluster deleted; `OldClusterDeleted=true`.
- Reassign episode: symmetric behavior to fact.
- `reverie://l1/index` no longer shows the deleted cluster.

### 2B
- Merge 2 sources into target: all members moved, sources deleted, target centroid recomputed.
- Target in source list: error.
- Any source nonexistent: error (no partial merge).
- Empty source list: error.
- Atomicity: inject a store error mid-merge → all-or-nothing verified.

### 2C
- Split into 2 groups: 2 new clusters with correct members, source either deleted (full partition) or shrunk (partial).
- Overlapping groups: error.
- Empty group: error.
- Group containing ID not in source cluster: error.
- `Metas` applied to new clusters when provided.
- Partial split: `RemainingInSource` correct.

### 2D
- Update fact content: content + embedding + hash change; ID, cluster, timestamps (except accessed_at) stable.
- Update tags-only (nil content, tags pointer set): content unchanged.
- Tags nil pointer: tags preserved.
- Tags empty slice: tags cleared.
- Episode: `LinkedFactIDs` preservation / replacement / clearing semantics.
- Layer mismatch (episode payload on fact ID): error.
- Both content and episode set: error.
- Superseded fact update: allowed (operator decision) but behavior explicit — still updates and preserves superseded_by.
- Conflict-threshold-similar content does NOT trigger auto-supersede.

### 2E
- Filter by cluster_id: only members of that cluster returned.
- Filter by subtype: only matching fact subtypes; episodes excluded from filtered result.
- `layer=l2`: episodes excluded even if similar.
- `layer=l3`: facts excluded.
- `tags_any`: union match semantics (at least one tag overlap).
- Combined filters: intersection semantics (all filters must pass).
- No filters: behavior identical to pre-2E recall.

## Rollout

1. `go test ./...` green.
2. Manual smoke:
   - Write 3 unrelated facts. Confirm they land in 1-3 clusters depending on similarity.
   - If they ended up in one cluster, split with explicit groups. Confirm new clusters.
   - Merge two of them back. Confirm single cluster.
   - Reassign one member out. Confirm membership + centroids.
   - Update a fact's content. Confirm ID stable, recall finds it under new content.
   - Recall with cluster_id filter. Confirm scope.
3. Update `~/.claude/CLAUDE.md` curation section with "you now have split/merge/reassign — use them when clusters are mislabeled instead of forgetting".

## Forward references

- Phase 3 (3A) `dry_run` on `memory_write` reuses the conflict-detection path; if we expose supersede candidates in dry-run, that format mirrors `memory_get`'s `supersedes` field from Phase 1.
- Phase 4 `memory_edge_remove` does NOT replace 2D's link-clearing behavior; they're different affordances (2D for full replacement, Phase 4 for surgical edits).
