# Phase 5 — Observability

Status: **DESIGN** (not yet implemented)
Prereqs: Phase 1 merged (migration framework; 5C adds migration 3)
Unblocks: nothing — Phase 5 is self-contained observability.

## Goal

Give the operator a truthful picture of the memory system's health over time. Today `reverie://status` shows counts and config — nothing about retention distribution, cache performance, supersede chain depth, or growth over time. An agent can't answer "is this memory at risk of being forgotten?" or "how many stale duplicates are we carrying?" without direct DB access.

## Non-goals

- No automated alerting / threshold crossing. Phase 5 is read-only telemetry.
- No retention / decay tuning UI — still config-file driven.
- No per-memory access counters beyond the existing `accessed_at`. A full hit-rate histogram per memory is out of scope.

## Migration 3 (introduced by 5C)

```sql
CREATE TABLE IF NOT EXISTS daily_stats (
  date        TEXT PRIMARY KEY,       -- "YYYY-MM-DD"
  facts_in    INTEGER DEFAULT 0,
  facts_out   INTEGER DEFAULT 0,      -- deletions
  episodes_in INTEGER DEFAULT 0,
  episodes_out INTEGER DEFAULT 0,
  supersedes  INTEGER DEFAULT 0       -- count of facts superseded that day
);
```

No index needed — PRIMARY KEY is enough, table stays small (one row per day of server activity).

Counters are maintained by SQL triggers rather than Go-side to avoid drift if the DB is touched outside the server (e.g., via `reverie view` / `reverie forget` CLIs which go through the Store but not the handlers). Triggers:

```sql
CREATE TRIGGER IF NOT EXISTS trg_facts_insert
AFTER INSERT ON facts BEGIN
  INSERT INTO daily_stats(date, facts_in) VALUES (date('now'), 1)
  ON CONFLICT(date) DO UPDATE SET facts_in = facts_in + 1;
END;

CREATE TRIGGER IF NOT EXISTS trg_facts_delete
AFTER DELETE ON facts BEGIN
  INSERT INTO daily_stats(date, facts_out) VALUES (date('now'), 1)
  ON CONFLICT(date) DO UPDATE SET facts_out = facts_out + 1;
END;

CREATE TRIGGER IF NOT EXISTS trg_facts_supersede
AFTER UPDATE OF superseded_by ON facts
WHEN NEW.superseded_by IS NOT NULL AND OLD.superseded_by IS NULL BEGIN
  INSERT INTO daily_stats(date, supersedes) VALUES (date('now'), 1)
  ON CONFLICT(date) DO UPDATE SET supersedes = supersedes + 1;
END;

-- mirror for episodes_in / episodes_out
```

Back-filling existing data: migration 3 runs one-shot backfill from `facts.created_at` and `episodes.created_at`:

```sql
INSERT OR IGNORE INTO daily_stats(date, facts_in)
SELECT date(created_at), COUNT(*) FROM facts GROUP BY date(created_at);
-- ON CONFLICT keys mean second run produces 0 additions
```

(Deletions and supersedes pre-migration are lost — we don't have the history. Accept that.)

## Embedding cache metrics

The existing `CachedProvider` (`internal/embed/`) needs to expose hit/miss counters. Add atomic counters to the struct and a `Stats()` method returning a snapshot:

```go
type ProviderStats struct {
    Hits   int64 `json:"hits"`
    Misses int64 `json:"misses"`
}

func (p *CachedProvider) Stats() ProviderStats
```

Counters are in-memory only — reset on server restart. That's fine for "how's the cache doing in this session" but not historical. Persistent cache metrics are out of scope.

## Tool / resource API surfaces

### 5A — enrich `reverie://status`

Extend `statusResponse`:

```go
type statusResponse struct {
    DBPath    string          `json:"db_path"`
    Embedding embeddingStatus `json:"embedding"`
    Decay     decayStatus     `json:"decay"`
    Counts    countStatus     `json:"counts"`
    Disabled  bool            `json:"disabled"`
    // NEW in 5A:
    Retention     retentionStatus     `json:"retention"`
    Supersede     supersedeStatus     `json:"supersede"`
    EmbeddingCache cacheStatus         `json:"embedding_cache"`
    LastTick      string              `json:"last_tick,omitempty"` // ISO timestamp
}

type retentionStatus struct {
    BelowThreshold  int `json:"below_threshold"`  // clusters with R < config.retention_threshold
    Buckets         []RetentionBucket `json:"buckets"` // histogram
}

type RetentionBucket struct {
    Min   float64 `json:"min"`   // inclusive
    Max   float64 `json:"max"`   // exclusive (except last bucket)
    Count int     `json:"count"`
}

type supersedeStatus struct {
    TotalSuperseded int `json:"total_superseded"` // facts with superseded_by != NULL
    LongestChain    int `json:"longest_chain"`    // max depth across all chains
}

type cacheStatus struct {
    Hits    int64   `json:"hits"`
    Misses  int64   `json:"misses"`
    HitRate float64 `json:"hit_rate"` // 0.0 if hits+misses == 0
}
```

Buckets: fixed at `[0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.001]` → 6 buckets (last is inclusive of 1.0).

`LastTick` is read from a new singleton row in a new small table `decay_state` introduced by migration 3 alongside `daily_stats`:

```sql
CREATE TABLE IF NOT EXISTS decay_state (
  id INTEGER PRIMARY KEY CHECK(id = 1),
  last_tick TEXT
);
INSERT OR IGNORE INTO decay_state (id, last_tick) VALUES (1, NULL);
```

`MemoryManager.TickDecay` updates this row after advancing. `handleStatusResource` reads it.

Longest chain computation: do it on the fly with recursive CTE:

```sql
WITH RECURSIVE chain(id, depth) AS (
  SELECT id, 1 FROM facts WHERE superseded_by IS NOT NULL AND id NOT IN (SELECT superseded_by FROM facts WHERE superseded_by IS NOT NULL)
  UNION ALL
  SELECT f.id, chain.depth + 1 FROM facts f JOIN chain ON f.superseded_by = chain.id
)
SELECT COALESCE(MAX(depth), 0) FROM chain;
```

With small memory counts this is instant; add a query timeout (100ms) as a guardrail.

### 5B — `reverie://l1/at_risk` resource

Lists clusters below the retention threshold, sorted by retention ascending (most-at-risk first).

URI: `reverie://l1/at_risk?threshold=0.3&limit=50`

Defaults: `threshold` = config `decay.retention_threshold`; `limit` = 50 (max 500).

Response (shape reuses `l1ClusterEntry` from Phase 1):

```go
type l1AtRiskResponse struct {
    Threshold float64          `json:"threshold"`
    Clusters  []l1ClusterEntry `json:"clusters"` // retention ascending
    Total     int              `json:"total"`    // total clusters below threshold
}
```

### 5C — `reverie://stats/daily` resource

URI: `reverie://stats/daily?from=2026-01-01&to=2026-04-18`

Defaults: `from` = 30 days ago; `to` = today. Max span 365 days.

```go
type dailyStatsResponse struct {
    From string            `json:"from"`
    To   string            `json:"to"`
    Days []dailyStatsEntry `json:"days"` // sorted oldest first
}

type dailyStatsEntry struct {
    Date        string `json:"date"`
    FactsIn     int    `json:"facts_in"`
    FactsOut    int    `json:"facts_out"`
    EpisodesIn  int    `json:"episodes_in"`
    EpisodesOut int    `json:"episodes_out"`
    Supersedes  int    `json:"supersedes"`
}
```

Gaps (days with no activity) are returned as zero-value rows — denser output, cleaner for graphing.

## Behavior spec — shared rules

- **Status endpoint stays cheap.** Retention histogram iterates all clusters (small N — O(hundreds) expected). Cache stats are atomic reads. Supersede longest-chain has a 100ms timeout. If any sub-computation exceeds 250ms, skip it with a `"degraded": true` flag on that subsection rather than blocking the whole response.
- **Triggers are trusted.** Once migration 3 lands, all writes to facts/episodes flow through them. Don't also increment in Go — pick one source of truth.
- **Date boundary: UTC.** `date('now')` in SQLite returns UTC. Aligns with `created_at` stored as UTC.

## Task breakdown

| # | Owner | Files touched | Depends on | Blocks |
|---|---|---|---|---|
| 5A | subagent | `internal/mcpserver/resources.go` (status handler), `internal/embed/cached.go` (Stats method), `internal/memory/manager.go` (LastTick persistence), `internal/memory/store.go` + impls (GetLastTick, SetLastTick, SupersedeLongestChain, RetentionBuckets) | migration 3 landed by 5C, Phase 1 migration framework | — |
| 5B | subagent | `internal/mcpserver/resources.go` (new handler + register) | Phase 1 | — |
| 5C | subagent | `internal/db/migrations.go` (migration 3), `internal/memory/sqlite_store.go` (trigger-aware queries or just rely on triggers), new resource handler, store method for `ListDailyStats(from, to)` | Phase 1 migration framework | 5A (provides `decay_state` table) |

Landing order:

1. **5C merges first** — migration 3 introduces `daily_stats` AND `decay_state`. Both 5A and 5B benefit.
2. **5A and 5B in parallel** after 5C.

## Test matrix

### 5A
- `status` response has all new fields populated on a DB with at least 1 memory.
- Retention buckets sum equals total cluster count.
- Supersede longest chain: write fact A, write A' (supersedes A), write A'' (supersedes A') → chain = 3.
- Cache hit/miss: call `memory_recall` twice with same query → second call has higher hits.
- LastTick updates after `memory_decay_tick`.
- Empty DB: status handler doesn't crash; numeric fields are 0, rate is 0.0.
- Longest-chain timeout: inject a fake slow query path and confirm `degraded=true`. (Use a test-only hook.)

### 5B
- Write clusters with varied retention (manipulate `turns_since` directly in a test). Query returns clusters below threshold, sorted ascending.
- `threshold=0.5` override works.
- `limit` cap works.
- `threshold=1.0` returns all non-empty clusters.

### 5C
- Migration 3: fresh DB → tables + triggers created, `decay_state` seeded, backfill populates `daily_stats` from existing facts/episodes.
- Migration 3 on existing DB with data: backfill is correct; second run is idempotent.
- Insert a fact → `daily_stats[today].facts_in` increments.
- Delete a fact → `facts_out` increments.
- Supersede: insert near-duplicate → `supersedes` increments by 1.
- `reverie://stats/daily` with default range returns last 30 days.
- Explicit range works; gaps are zero-filled.
- `from > to` → error.
- Span > 365 days → error.

## Rollout

1. `go test ./...` green.
2. Manual smoke:
   - `reverie://status` shows realistic buckets.
   - Write enough memories to see retention drop after ticks. Confirm `l1/at_risk` populates.
   - Write, delete, write — check `stats/daily` for today.
3. Update `docs/` — add a new `docs/observability.md` describing the resources and what an operator or dashboard should care about.

## Forward references

- Phase 6's session persistence may add a `session_count` to status. Leave space in `statusResponse` for future fields — don't lock the shape.
- A future phase could add histograms per subtype or per cluster; keep `RetentionBucket` reusable.
