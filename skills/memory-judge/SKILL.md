---
name: memory-judge
description: Use after calling memory_recall when the candidate list is large (>5) or the query is sensitive to staleness. Spawns a Task subagent with fresh context to judge each candidate, then calls memory_apply_judgment with the verdicts.
---

# Memory Judge

After `memory_recall` returns candidates, use this skill to apply Gate A
(uncertainty/staleness judgment) before the final filter.

## When to use

- A recall returned more than 5 candidates
- The query is about "current state" or otherwise staleness-sensitive
- Multiple candidates have similar similarity scores and you need to disambiguate

## Process

1. Call `memory_recall` with your query -- note the `recall_id` in the response.
2. Spawn a Task subagent using the Task tool. Use this prompt:

   ```
   You are a memory relevance judge. You see ONLY this query and this list of
   candidate memories -- nothing from the surrounding conversation. For each candidate,
   decide whether it should be kept as context for answering the query.

   Drop a memory if any of:
   - It is stale (a newer memory contradicts or supersedes it).
   - It is off-topic despite surface similarity.
   - Its content is too generic to be useful for this query.

   Query: {{query}}

   Candidates:
   {{for each: id, content, layer, subtype, created_at, similarity, retention}}

   Respond with a JSON array: [{"memory_id": "...", "keep": true|false, "reason": "..."}]
   ```

   Replace `{{query}}` with the actual query string, and replace the `{{for each ...}}`
   block with a formatted list of candidates from the `memory_recall` response. Include
   these fields for each candidate: `id`, `content`, `layer`, `subtype`, `created_at`,
   `similarity`, `retention`.

3. Feed the subagent: the `query`, and the list of candidates (id, content, layer, subtype, created_at, similarity, retention).
4. The subagent returns a JSON array: `[{"memory_id": "...", "keep": true|false, "reason": "..."}]`.
5. Call `memory_apply_judgment` with the `recall_id` and the `verdicts` array.

## Why a subagent

- **Fresh context**: the subagent sees only the query and candidates. Your main
  conversation's noise does not bias the judgment.
- **Your auth, your model**: no separate API key needed. If you are Opus, the
  judge is Opus; if Sonnet, the judge is Sonnet.
- **One spawn, all candidates**: a single Task call handles the full candidate
  set in one round-trip.
- **Transparent fallback**: callers without subagent capability (Claude Desktop,
  bare harnesses) skip `memory_apply_judgment` and use Phase-1 candidates under
  OR logic. Gates B and C still filter; only Gate A is lost.

## Example

Suppose you need to recall what database the project uses, and `memory_recall`
returns 7 candidates:

```
recall_id: "rec_abc123"
candidates:
  - id: "m-001", content: "Project uses PostgreSQL 15 for the main datastore", layer: "l2_semantic", subtype: "project", created_at: "2026-03-10", similarity: 0.88, retention: 0.92
  - id: "m-002", content: "Migrated from PostgreSQL to SQLite for local dev", layer: "l2_semantic", subtype: "project", created_at: "2026-04-01", similarity: 0.85, retention: 0.95
  - id: "m-003", content: "Database backups run nightly at 02:00 UTC", layer: "l2_semantic", subtype: "project", created_at: "2026-02-15", similarity: 0.72, retention: 0.60
  - id: "m-004", content: "The user prefers SQLite for single-binary deployments", layer: "l2_semantic", subtype: "user", created_at: "2026-01-20", similarity: 0.70, retention: 0.88
  - id: "m-005", content: "Evaluated CockroachDB but rejected for complexity", layer: "l3_episodic", subtype: "", created_at: "2026-02-28", similarity: 0.68, retention: 0.55
  - id: "m-006", content: "SQL is a declarative query language", layer: "l2_semantic", subtype: "reference", created_at: "2025-12-01", similarity: 0.65, retention: 0.30
  - id: "m-007", content: "Project uses WAL mode for SQLite", layer: "l2_semantic", subtype: "project", created_at: "2026-04-05", similarity: 0.82, retention: 0.97
```

Spawn a Task subagent with the prompt above, substituting the query and candidates:

```
You are a memory relevance judge. You see ONLY this query and this list of
candidate memories -- nothing from the surrounding conversation. For each candidate,
decide whether it should be kept as context for answering the query.

Drop a memory if any of:
- It is stale (a newer memory contradicts or supersedes it).
- It is off-topic despite surface similarity.
- Its content is too generic to be useful for this query.

Query: What database does this project use?

Candidates:
- id: "m-001", content: "Project uses PostgreSQL 15 for the main datastore", layer: "l2_semantic", subtype: "project", created_at: "2026-03-10", similarity: 0.88, retention: 0.92
- id: "m-002", content: "Migrated from PostgreSQL to SQLite for local dev", layer: "l2_semantic", subtype: "project", created_at: "2026-04-01", similarity: 0.85, retention: 0.95
- id: "m-003", content: "Database backups run nightly at 02:00 UTC", layer: "l2_semantic", subtype: "project", created_at: "2026-02-15", similarity: 0.72, retention: 0.60
- id: "m-004", content: "The user prefers SQLite for single-binary deployments", layer: "l2_semantic", subtype: "user", created_at: "2026-01-20", similarity: 0.70, retention: 0.88
- id: "m-005", content: "Evaluated CockroachDB but rejected for complexity", layer: "l3_episodic", subtype: "", created_at: "2026-02-28", similarity: 0.68, retention: 0.55
- id: "m-006", content: "SQL is a declarative query language", layer: "l2_semantic", subtype: "reference", created_at: "2025-12-01", similarity: 0.65, retention: 0.30
- id: "m-007", content: "Project uses WAL mode for SQLite", layer: "l2_semantic", subtype: "project", created_at: "2026-04-05", similarity: 0.82, retention: 0.97

Respond with a JSON array: [{"memory_id": "...", "keep": true|false, "reason": "..."}]
```

The subagent returns:

```json
[
  {"memory_id": "m-001", "keep": false, "reason": "Superseded by m-002 which shows migration to SQLite"},
  {"memory_id": "m-002", "keep": true, "reason": "Most recent database decision -- migrated to SQLite"},
  {"memory_id": "m-003", "keep": false, "reason": "About backup scheduling, not which database is used"},
  {"memory_id": "m-004", "keep": true, "reason": "Explains the motivation behind the SQLite choice"},
  {"memory_id": "m-005", "keep": true, "reason": "Useful context -- a rejected alternative"},
  {"memory_id": "m-006", "keep": false, "reason": "Too generic -- defines SQL itself, not project-specific"},
  {"memory_id": "m-007", "keep": true, "reason": "Directly relevant -- WAL mode config for the current DB"}
]
```

Then call `memory_apply_judgment`:

```json
{
  "recall_id": "rec_abc123",
  "verdicts": [
    {"memory_id": "m-001", "keep": false, "reason": "Superseded by m-002 which shows migration to SQLite"},
    {"memory_id": "m-002", "keep": true, "reason": "Most recent database decision -- migrated to SQLite"},
    {"memory_id": "m-003", "keep": false, "reason": "About backup scheduling, not which database is used"},
    {"memory_id": "m-004", "keep": true, "reason": "Explains the motivation behind the SQLite choice"},
    {"memory_id": "m-005", "keep": true, "reason": "Useful context -- a rejected alternative"},
    {"memory_id": "m-006", "keep": false, "reason": "Too generic -- defines SQL itself, not project-specific"},
    {"memory_id": "m-007", "keep": true, "reason": "Directly relevant -- WAL mode config for the current DB"}
  ]
}
```

Reverie combines these Gate A verdicts with the Gate B (similarity) and Gate C
(retention) results, applies the round's logic (OR for round 0, AND for round 1+),
curates under the working memory budget, and returns the final memory set.
