# Phase 7C -- Graph-aware Recall

Status: **DESIGN** (not yet implemented)
Prereqs: Phase 7 merged (`memory_edges`, `entities`, `entity_mentions` tables; `ListEdges`, `ListMemoriesByEntity`, `GetEntity`, `ListEntitiesByMemoryIDs` store methods).
Unblocks: nothing -- Phase 7C completes the knowledge-graph payoff story.

## Goal

Graph plumbing without graph-aware recall delivers zero user-facing value. Phase 7C adds opt-in `expand_via_graph` to `memory_recall`: after vector search returns seeds, walk `memory_edges` and `entity_mentions` from those seeds, score reachable neighbors with a per-hop decay formula, merge them into the candidate set, and rank by composite score. Hub entities (entities mentioned by many memories -- the canonical "popular file" case) expand without per-seed truncation; pathological blowup is bounded by a global visited-set cap and an in-BFS retention pre-filter.

## Non-goals

- Replacing vector search -- graph expansion is strictly additive.
- Per-edge-type scoring weights (`causes` / `contradicts` / `refines` treated equally for the first cut).
- Entities as first-class `RecallCandidate`s -- they remain pure expansion mediators.
- Round-1+ support -- `expand_via_graph` is silently ignored on `round >= 1`.
- Streaming or cursored neighborhoods -- the hop cap (3) plus global cap (2000) bound the working set; no pagination.
- Score-constant auto-tuning from usage telemetry -- a future operational improvement.
- New tools -- this is a `memory_recall` extension only.

## Locked decisions

| Decision | Choice | Why |
|---|---|---|
| Activation | Opt-in via `expand_via_graph bool` on `RecallInput`. Default false. | Pure vector recall stays the default; callers ask for the extra cost when they want it. |
| Round behavior | Honored only when `round == 0`. Silently ignored on `round >= 1`. | AND-logic narrowing is incompatible with graph broadening. |
| Default hop count | 2 (configurable via `graph_hops` on input, clamped 1..3). | 1-hop catches direct edges only; 2-hop catches "memories sharing an entity" -- the typical "what do I know about X" query shape. |
| Memory->entity->memory cost | Counts as 2 hops. | An entity is a real graph node, not a free pass. |
| Scoring formula | `composite = seed_similarity * neighbor_retention * (decay_per_hop ^ hop_count)` | Diminishes by hop, weighted by neighbor's standing retention, anchored to the seed that found it. |
| `decay_per_hop` default | 0.5, configurable via `[memory] graph_decay_per_hop` in `reverie.toml`. | Halving per hop is an opinionated starting point; tunable as usage data accrues. |
| Multi-seed neighbor scoring | If a neighbor is reachable from multiple seeds, take MAX composite score. | Most-favorable seed wins. |
| Vector + graph overlap | Same memory in both: keep the higher-scored entry, dedup by ID. | Composite score wins. |
| Per-seed cap | None -- hub entities must expand without truncation. | "Memories about file X" must return ALL memories about X, not an arbitrary 50. |
| Global blowup safety | Cap on total visited memory IDs during BFS. Default 2000, configurable via `[memory] graph_max_visited`. | Prevents 200x200=40k explosions at hop 2 on dense graphs. When cap is reached, BFS stops; remaining seeds' lower-priority neighbors are dropped. |
| Retention pre-filter | Skip neighbors during BFS if their cluster's (or entity's) retention < 0.05. Configurable via `[memory] graph_min_retention_for_expansion`. | Decayed memories don't pollute the candidate set; reduces global-cap pressure. |
| Gate B for graph-only neighbors | `gate_b_pass = false` deterministically. | They weren't found by similarity -- preserves the existing gate's meaning. |
| Gate C | Applied uniformly across vector + graph candidates. | Existing semantics unchanged. |
| Output shape | Add `Distance int` (0 for vector hit, >=1 for graph) and `CompositeScore float64` to `RecallCandidate`. Existing `Similarity` field unchanged (0 for graph-only neighbors that lack a cosine-to-query). | Provenance + ranking explicit; backward-compatible (existing fields preserved). |
| Entities in output | Entities do NOT appear as `RecallCandidate`s -- pure expansion mediators only. | Recall returns memories; entities help find them but aren't surfaced. Forward reference for a future phase. |

## Tool API surface diff

`RecallInput` adds two fields (existing fields -- Query, Limit, Hints, Round, ClusterID, Subtype, Layer, TagsAny, SessionID -- unchanged):

```go
// ExpandViaGraph: when true and round==0, recall additionally walks
// memory_edges and entity_mentions from the vector seeds to surface
// graph neighbors. Default false. Ignored on round>=1.
ExpandViaGraph bool `json:"expand_via_graph,omitempty" jsonschema:"opt-in graph expansion on top of vector recall"`

// GraphHops: BFS depth budget for graph expansion. Clamped to [1,3].
// Defaults to 2 when ExpandViaGraph is true. Memory->entity->memory
// counts as 2 hops.
GraphHops int `json:"graph_hops,omitempty" jsonschema:"BFS depth for graph expansion (1..3, default 2)"`
```

`RecallCandidate` adds two fields (existing fields -- ID, Content, Layer, Similarity, Retention, GateBPass, GateCPass, ClusterID, Subtype, Tags, LinkedIDs -- unchanged):

```go
// Distance: 0 for vector hits, >=1 for graph-expanded neighbors.
Distance int `json:"distance"`

// CompositeScore: the score used for ranking. For vector hits this
// equals Similarity; for graph hits it is
// seed_similarity * neighbor_retention * (decay_per_hop ^ Distance).
CompositeScore float64 `json:"composite_score"`
```

`RecallOutput` is unchanged in shape; its `Candidates` slice is now ordered by `CompositeScore` descending.

## Store API surface

One new method on the `Store` interface (both `sqlite_store.go` and `mem_store.go` implement):

```go
// ExpandViaGraph walks memory_edges and entity_mentions outward from the
// given seed memory IDs up to hops levels deep (clamped 1..3). Each hop
// follows: (a) memory_edges in either direction, and (b) memory->entity
// via entity_mentions then entity->memory via entity_mentions (2 hops).
// Decayed nodes (retention < minRetention) are pruned during BFS.
// Returns at most maxVisited distinct neighbor IDs; further frontier
// expansion is dropped when the cap is reached.
ExpandViaGraph(
    ctx context.Context,
    seedIDs []string,
    hops int,
    minRetention float64,
    maxVisited int,
) ([]GraphHit, error)
```

New type in `internal/memory/types.go`:

```go
type GraphHit struct {
    NeighborID    string  // memory ID (fact or episode)
    NeighborLayer string  // "l2_semantic" or "l3_episodic"
    SeedID        string  // the seed this neighbor was reached from (one row per (neighbor, seed) pair)
    Distance      int     // hop count from seed (1..hops)
}
```

The store returns one `GraphHit` per `(neighbor, seed)` pair reached during BFS — it does NOT collapse to one row per neighbor. Rationale: with `decay_per_hop = 0.5`, shortest distance does not imply best composite score. Example: seed A reaches M at distance 1 with `sim_A = 0.3` (composite = 0.15 × M.retention); seed B reaches M at distance 2 with `sim_B = 0.9` (composite = 0.225 × M.retention). Seed B wins despite being farther. The store cannot make this judgment without knowing per-seed similarities, so it surfaces every pair and lets the handler compute the MAX. Within a single BFS run, if the same `(neighbor, seed)` pair is reachable via multiple paths of different distances, the store keeps the shortest (per-pair-shortest is unambiguously best within a single seed because similarity is fixed per seed).

## Behavior spec

### BFS algorithm

```
# Per-pair shortest-distance tracking: each (neighbor, seed) gets at most
# one row, at its shortest distance from that seed. Different seeds
# reaching the same neighbor each produce their own row (handler computes
# MAX composite across them).
bestPair       = {(s, s) -> 0 for s in seedIDs}    // (memID, seedID) -> shortest dist
globalVisited  = set(seedIDs)                       // distinct memory IDs for the cap
frontier       = [(s, s) for s in seedIDs]         // list of (memID, seedID) tuples
results        = []

for depth in 1..hops:
    if len(globalVisited) >= maxVisited: break
    next_frontier = []

    # (a) memory_edges -- one-hop edge neighbors
    frontierIDs = {memID for (memID, _) in frontier}
    rows = SELECT src_id, dst_id FROM memory_edges
           WHERE src_id IN frontierIDs OR dst_id IN frontierIDs
    for row in rows:
        for n in (row.src_id, row.dst_id):
            if resolves_to_entity(n): continue
            if retention_for(n) < minRetention: continue
            # For every parent (memID, seedID) in frontier where memID is
            # an endpoint of this edge, attempt to record (n, seedID).
            for (parent, seedID) in frontier_endpoints_of(row, frontier):
                if n == parent: continue              // self-loop guard
                if (n, seedID) in bestPair: continue  // already reached from this seed
                bestPair[(n, seedID)] = depth
                results.append(GraphHit{n, layer_of(n), seedID, depth})
                next_frontier.append((n, seedID))
                if n not in globalVisited:
                    globalVisited.add(n)
                    if len(globalVisited) >= maxVisited: break-all

    # (b) memory->entity->memory -- 2-hop entity mediator
    if depth+1 <= hops:
        for (m, seedID) in frontier:
            entIDs = SELECT entity_id FROM entity_mentions WHERE memory_id=m
            for e in entIDs:
                if retention_for_entity(e) < minRetention: continue
                mIDs = SELECT memory_id FROM entity_mentions WHERE entity_id=e
                for n in mIDs:
                    if n == m: continue
                    if retention_for(n) < minRetention: continue
                    if (n, seedID) in bestPair: continue
                    bestPair[(n, seedID)] = depth+1
                    results.append(GraphHit{n, layer_of(n), seedID, depth+1})
                    if n not in globalVisited:
                        globalVisited.add(n)
                        if len(globalVisited) >= maxVisited: break-all

    frontier = next_frontier

return results
```

Two invariants the BFS preserves:

1. **Per-pair shortest distance.** For a given `(neighbor, seed)`, only the first (smallest-depth) reach is recorded. Within a single seed, similarity is fixed; the shortest distance is unambiguously the highest composite score from that seed.
2. **Multi-seed surfacing.** The same neighbor reached from different seeds produces multiple `GraphHit` rows -- one per seed. The handler computes composite for each and keeps the MAX. The store does not collapse across seeds because shortest distance is NOT a reliable proxy for best composite once per-seed similarities differ.

The entity mediator advances depth by 2 in one iteration -- the loop guard `depth+1 <= hops` keeps an in-budget hop-2 expansion legal at the outer `depth==1` iteration but illegal at `depth==2` when `hops==2`. The global visited cap counts distinct memory IDs, not `(memID, seedID)` pairs -- the cap protects against memory blowup, not per-seed work duplication. Entity IDs themselves never enter `globalVisited` or `bestPair` -- they're traversal mediators, not output rows. `retention_for(memory_id)` reuses the Phase 7 helper from `phase-7-knowledge-graph.md#edge-retention-computation`; `retention_for_entity(entity_id)` reads the entity's own retention column directly.

### Scoring

For each `GraphHit{neighborID, seedID, distance}`:

```
composite = seedSimilarity[seedID]
          * retention_for(neighborID)
          * (decay_per_hop ^ distance)
```

Worked example. Query embedding cosine-matches two seeds:

- Seed A at `sim = 0.8`
- Seed B at `sim = 0.6`

Both reach memory M:

- A links to M via one `memory_edges` row at depth 1.
- B mentions entity E; E is mentioned by M at depth 2.

Let `M.retention = 0.9` and `decay_per_hop = 0.5`. The two candidate composites:

- Via A: `0.8 * 0.9 * 0.5^1 = 0.360`
- Via B: `0.6 * 0.9 * 0.5^2 = 0.135`

`max(0.360, 0.135) = 0.360`. M is emitted once with `Distance=1`, `SeedID=A`, `CompositeScore=0.360`.

### Vector + graph merge

```
byID = {}
for v in vectorCandidates:
    byID[v.ID] = RecallCandidate{
        ...v...,
        Similarity:     v.similarity,
        Distance:       0,
        CompositeScore: float64(v.similarity),
        GateBPass:      v.similarity > threshold,
    }

for h in graphHits:
    rc := RecallCandidate{
        ID:             h.NeighborID,
        Layer:          h.NeighborLayer,
        Similarity:     0,
        Distance:       h.Distance,
        CompositeScore: seedSim[h.SeedID] * retention_for(h) * pow(decay, h.Distance),
        GateBPass:      false,
        // GateCPass + Retention populated from cluster lookup below
    }
    if existing, ok := byID[h.NeighborID]; ok {
        if rc.CompositeScore > existing.CompositeScore:
            byID[h.NeighborID] = rc            // graph beats vector
        else:
            continue                            // vector beats graph; keep
    } else {
        byID[h.NeighborID] = rc
    }

candidates = sortByCompositeDesc(values(byID))
return candidates[:Limit]
```

Content, ClusterID, Subtype, Tags, LinkedIDs, GateCPass, and Retention are populated on graph-only entries via the same per-candidate lookups the vector path already performs (`GetCluster`, `GetFact` / `GetEpisode`, `ListEdges` for `LinkedIDs`). For collisions where the vector entry wins, no re-lookup is needed -- it's already populated. `Limit` is applied **after** merge, so a 10-result request may return 5 vector hits and 5 graph hits, or 10-and-0, depending on composite scores.

### Round-aware behavior

```
if in.Round >= 1 && in.ExpandViaGraph {
    s.logger.Info("memory_recall: expand_via_graph ignored on round>=1",
                  "round", in.Round)
    // proceed with existing AND-logic refinement path; do NOT expand
}
```

No error. The flag is silently honored as false on round-1+. AND-logic narrowing operates over the prior round's cached candidate set, which already includes any graph hits from round 0 if the caller asked for them then.

### Config plumbing

Three new fields under the existing `[memory]` section in `internal/config/config.go`:

| Field | Default | Source |
|---|---|---|
| `GraphDecayPerHop` | 0.5 | `[memory] graph_decay_per_hop` |
| `GraphMaxVisited` | 2000 | `[memory] graph_max_visited` |
| `GraphMinRetentionForExpansion` | 0.05 | `[memory] graph_min_retention_for_expansion` |

The handler reads all three at call time (no hot-reload required; values bind once per process). Out-of-range values (`<=0` decay, `<=0` cap, `<0` or `>=1` retention) fall back to the default and emit a single warning at startup via the config validator.

### Gate semantics

- **Vector hits.** `GateBPass = Similarity > cfg.Memory.SimilarityThreshold` (today's behavior, preserved).
- **Graph-only hits.** `GateBPass = false` deterministically -- no exception even if the composite score is high.
- **Gate C.** `GateCPass = decayer.GateC(cluster)` for both vector and graph hits, looked up uniformly via `GetCluster` on the candidate's cluster ID.

## Task breakdown

Two execution phases (C1 + C2). Single PR.

| # | Owner | Files touched | Depends on | Blocks |
|---|---|---|---|---|
| C1 | subagent | `internal/memory/store.go` (interface +`ExpandViaGraph`), `internal/memory/types.go` (+`GraphHit`), `internal/memory/sqlite_store.go` + `mem_store.go` (BFS impl w/ retention pre-filter + visited cap), `internal/mcpserver/tools.go` (RecallInput/RecallCandidate diff + `handleRecall` rewrite), `internal/memory/store_test.go`, `internal/mcpserver/tools_test.go` | Phase 7 | C2 |
| C2 | subagent | `README.md` (memory_recall row + Knowledge-graph "Graph-aware recall" subsection), `reverie.toml.example` (three new tunables under `[memory]`), `internal/config/config.go` (+ three fields + defaults), full `gofmt`/`vet`/`test`/`build`/`grep` verification pass | C1 | -- |

C1 is the foundation -- store BFS, handler scoring, and merge. C2 is docs, config, and verification, parallelizable with C1's review cycle but landed in the same PR.

## Test matrix

| Test name | Layer | What it asserts |
|---|---|---|
| `TestExpandViaGraph_DirectEdge` | store | M1 edge->M2; expand from M1 hops=1 returns M2 distance=1. |
| `TestExpandViaGraph_EntityMention` | store | M1 mentions E, M2 mentions E; expand from M1 hops=2 returns M2 distance=2 via E. |
| `TestExpandViaGraph_TwoHopEdgeChain` | store | M1->M2->M3 edges; expand from M1 hops=2 returns M2 dist=1 and M3 dist=2. |
| `TestExpandViaGraph_HubEntityNoCap` | store | E mentioned by 100 memories; expand from one mention with hops=2 returns all 99 others (no per-seed cap). |
| `TestExpandViaGraph_GlobalCapHonored` | store | Synthesize 3000 neighbors at hops=2; assert exactly 2000 returned (cap), no panic. |
| `TestExpandViaGraph_RetentionPrefilter` | store | M1 -> M2 (retention 0.01) -> M3 (retention 0.9); expand from M1 hops=2 returns nothing past M2 (pre-filter prunes M2 from frontier). |
| `TestExpandViaGraph_MultiSeedReturnsAllPairs` | store | M1 and M2 both reach Mx; M1 at dist=1, M2 at dist=2; result contains TWO `GraphHit`s for Mx (one per seed); handler computes MAX composite across them. |
| `TestExpandViaGraph_SameSeedShortestPath` | store | Within a single seed S, Mx is reachable at dist=1 (via edge) and dist=2 (via entity hop); only the dist=1 row is returned for `(Mx, S)`. |
| `TestHandleRecall_ExpandFlagOff_Unchanged` | handler | `expand_via_graph: false` yields identical output (modulo new zero-valued `Distance` and `CompositeScore = Similarity`) to the pre-7C path. |
| `TestHandleRecall_ExpandViaGraph_AddsNeighbors` | handler | Seed via vector, neighbor reachable only via edge; appears in output with `Distance=1`, `CompositeScore` correctly computed. |
| `TestHandleRecall_VectorAndGraphMaxDedupe` | handler | Memory M appears as both vector hit (sim=0.6) and graph hit (composite=0.4); single entry with `CompositeScore=0.6`, `Distance=0`. |
| `TestHandleRecall_Round1_IgnoresExpand` | handler | Round 1 + `expand_via_graph=true`: output identical to round 1 without the flag; an info log line is emitted. |
| `TestHandleRecall_GateB_GraphHitsFalse` | handler | Graph-only neighbor has `gate_b_pass=false` regardless of any computed similarity. |
| `TestHandleRecall_HopsClamping` | handler | `graph_hops=5` clamps to 3; `graph_hops=0` defaults to 2 when `expand_via_graph=true`. |
| `TestHandleRecall_LimitAppliedPostMerge` | handler | `Limit=5` with 10 combined vector+graph candidates yields the top 5 by composite. |

## Rollout

Single PR bundles C1 + C2. No feature flag -- the `expand_via_graph` input field itself is the opt-in gate. Backward compatible: callers ignoring the new fields see identical behavior (new output fields default to zero and existing fields are untouched). Post-merge smoke: `memory_recall` with `expand_via_graph: true` against a session that has at least one edge and one entity mention; verify `distance>0` entries appear with non-zero `composite_score` and `gate_b_pass=false`.

## Forward references

- **Per-edge-type weighting.** Let `edge_type` (`causes`, `contradicts`, `refines`, ...) modulate the score. Needs usage telemetry to motivate sensible weights.
- **Entities as candidates.** Let `RecallCandidate` carry `Layer: "entity"` so callers can recall entities themselves. Requires deciding on entity content representation in the response and may affect Gate B semantics.
- **Score-constant tuning from telemetry.** `graph_decay_per_hop` and `graph_min_retention_for_expansion` are guesses. Once we have real usage data, derive sensible defaults from telemetry.
- **Streamed expansion.** For very large graphs, BFS could yield results incrementally rather than batching to 2000. Deferred until the cap actually bites in practice.
