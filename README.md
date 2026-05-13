# Reverie

Persistent memory system for LLM coding agents.

Reverie is an MCP server that implements the Oblivion memory architecture (arXiv 2604.00131). It provides three memory layers -- L1 clusters (procedural), L2 facts (semantic), and L3 episodes (episodic) -- with Ebbinghaus-curve decay, two-gate retention filtering, and local embeddings via Ollama. Single Go binary, no CGO, no internal LLM calls.

## Quick start

The fastest path: clone and run the installer. It builds the binary, pulls the Ollama embedding model if missing, and wires reverie into Claude Code, Claude Desktop, and/or OpenCode (whichever it finds), preserving any existing MCP server entries.

```bash
git clone https://github.com/ihavespoons/reverie.git
cd reverie
./scripts/install.sh
```

The installer is re-run safe (existing config is backed up before merge) and supports `--code-only`, `--desktop-only`, `--opencode-only`, `--skip-ollama`, and `--uninstall` flags. See `./scripts/install.sh --help`.

### Manual install

```bash
# Prerequisites: Go 1.22+, Ollama with nomic-embed-text
ollama pull nomic-embed-text

# Build
go install ./cmd/reverie

# Run as an MCP server (Claude Code will invoke this)
reverie serve

# Or check status
reverie status
```

## Claude Code setup

Register the server with the `claude mcp` CLI -- it writes to `~/.claude.json` (the file Claude Code reads MCP entries from; `~/.claude/settings.json` is the wrong file and is ignored):

```bash
claude mcp add --scope user reverie /path/to/reverie serve
```

Replace `/path/to/reverie` with the actual binary path (e.g., the output of `go env GOPATH`/bin/reverie if installed via `go install`).

No API keys needed with Ollama. For Voyage, pass the key with `-e VOYAGE_API_KEY="$VOYAGE_API_KEY"`.

Verify with `claude mcp list`. Restart Claude Code after adding the entry.

See [docs/claude-code-setup.md](docs/claude-code-setup.md) for the full setup guide including the CLAUDE.md preamble.

## Claude Desktop setup

Add to `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "reverie": {
      "type": "stdio",
      "command": "/path/to/reverie",
      "args": ["serve"]
    }
  }
}
```

Desktop does NOT support `${ENV_VAR}` interpolation -- use a wrapper script if env vars are needed. With Ollama (default), no wrapper is necessary.

Desktop does NOT have Task/subagent support -- Gate A (`memory_apply_judgment`) is unavailable. Recall works fine with Gates B+C only.

See [docs/claude-desktop-setup.md](docs/claude-desktop-setup.md) for details.

## OpenCode setup

Add to `~/.config/opencode/opencode.json`:

```json
{
  "mcp": {
    "reverie": {
      "type": "local",
      "command": ["/path/to/reverie", "serve"],
      "enabled": true
    }
  }
}
```

Note: the field is `mcp` (not `mcpServers`), and `command` is a single array containing the executable plus its args -- copying the Claude Code shape verbatim will not work.

OpenCode uses `{env:VAR}` for env-var interpolation, not `${VAR}`.

Gate A (`memory_apply_judgment`) IS available in OpenCode -- unlike Desktop -- provided you copy `opencode/agents/memory-judge.md` into `~/.config/opencode/agents/`. See [opencode/README.md](opencode/README.md) for the copy step.

Restart OpenCode after adding the entry.

See [docs/opencode-setup.md](docs/opencode-setup.md) for the full setup guide.

## Custom harness setup

Subprocess the binary and speak MCP over stdio. Go: `exec.Command("reverie", "serve")` with piped stdin/stdout. Python: `subprocess.Popen(["reverie", "serve"])` with the `mcp` package.

See [docs/custom-harness.md](docs/custom-harness.md) for examples.

## Architecture

```
+----------------------+      stdio MCP      +-----------------------------+
| Claude Code, Desktop,| <-----------------> | reverie serve (Go binary)   |
| OpenCode, or a       |                     |   NO internal LLM calls     |
| custom Go/Py harness |                     |                             |
|  +-- spawns Task     |                     |  Executor                   |
|  |   subagent to     |                     |   +-- Decayer (gates B+C)   |
|  |   judge candidates|                     |   +-- MemoryManager         |
|  |   (Gate A)        |                     |   +-- WorkingMemory (RAM)   |
|  +-- calls write/    |                     |                             |
|      reinforce w/ own|                     |  Embed: OpenAI-compat HTTP  |
|      classification  |                     |  (Ollama by default)        |
+----------------------+                     |                             |
                                             |  Store: SQLite (WAL)        |
                                             |   +-- clusters (L1)         |
                                             |   +-- facts (L2)            |
                                             |   +-- episodes (L3)         |
                                             |   +-- entities (L-graph)    |
                                             |   +-- edges (L-graph)       |
                                             |   +-- embedding_cache       |
                                             +-----------------------------+
```

- **Executor**: orchestrates read/write paths, owns working memory lifecycle.
- **Decayer**: Gates B (similarity) + C (Ebbinghaus retention). Gate A is the caller's responsibility via subagent.
- **MemoryManager**: utility/frequency reinforcement, tick decay, budget curation.
- **Embed**: OpenAI-compatible HTTP client. Ollama on localhost by default; any `/v1/embeddings` endpoint works.
- **Store**: SQLite with WAL mode, pure Go driver (`modernc.org/sqlite`). Vectors stored as BLOBs, cosine computed in Go.

## Memory types

| Type | Layer | Description |
|---|---|---|
| user | L2 | Personal facts (role, preferences, skills) |
| feedback | L2 | Rules for agent behavior |
| project | L2 | Codebase facts, conventions, architecture |
| reference | L2 | Pointers to URLs, repos, external docs |
| episode | L3 | Situation, action, outcome, lesson |
| entity | L-graph | First-class nodes (files, repos, libraries, concepts) referenced by memory mentions; decay like clusters. |

## Tools

| Tool | Purpose | When to call |
|---|---|---|
| `memory_recall` | Search memory by query. Returns ranked candidates with gate pass flags. Accepts optional `session_id` to auto-update the session buffer; set `expand_via_graph: true` (with optional `graph_hops`) to walk the knowledge graph from vector seeds and surface reachable neighbors. | Session start; before architectural decisions; when referencing prior context; "what do I know about X" queries (with `expand_via_graph: true`). |
| `memory_write` | Store a new fact (L2) or episode (L3). Accepts optional `session_id`. | When the caller decides something is durable knowledge. |
| `memory_apply_judgment` | Apply Gate A verdicts from a subagent judge to a recall result. Accepts optional `session_id`. | After `memory_recall` when >5 candidates or staleness matters. |
| `memory_reinforce` | Boost utility of memories actually used in a response. Accepts optional `session_id`. | After using recalled memories. |
| `memory_forget` | Delete by ID, or search for deletion candidates by query. | On correction; on explicit "forget X". |
| `memory_list` | Browse/audit memories with filtering and pagination. | Inspection. |
| `memory_decay_tick` | Advance the decay clock (internal). | Scheduled jobs; not called by agents directly. Use `memory_session_end` for session-scoped ticks. |
| `memory_session_init` | Create or resume a named session; returns the persisted working-memory buffer. | At the start of every conversation that wants resumable memory. |
| `memory_session_snapshot` | Force-flush the current buffer to the session store. | Explicit checkpoint; normally implicit after each mutation. |
| `memory_session_restore` | Read the buffer and metadata for a session without `init` semantics. | Inspection / audit. |
| `memory_session_end` | Close a session, run a scoped decay tick, optionally write an L3 episode. | End of conversation. |
| `memory_edge_add` | Add a typed directed edge between two memories or entities. | When the host classifies a relation (causes/refines/contradicts/...) between two known nodes. |
| `memory_edge_remove` | Remove a specific edge; idempotent on missing. | On correction or stale-link cleanup. |
| `memory_edge_list` | List edges incident to a memory or entity, up to N hops (1-3). | Walk the graph to find related context. |
| `memory_entity_upsert` | Create or dedupe an entity by (name, entity_type); exact match then similarity fallback. | After noticing a recurring named thing (file, library, concept). |
| `memory_entity_mention` | Attach a memory to one or more entities; idempotent. | Right after `memory_write` when the host has extracted entities from the new memory. |
| `memory_entity_neighbors` | Walk the graph from an entity to nearby memories and entities. | To answer "what do I know about X" queries. |

## Resources

| URI | Content |
|---|---|
| `reverie://status` | Counts per layer, last decay, DB size, cache hit rate. |
| `reverie://l1/index` | L1 cluster meta-index -- always-resident procedural memory. |
| `reverie://l1/cluster/{id}` | Per-cluster metadata + paginated members (facts + episodes). |
| `reverie://l1/at_risk` | Clusters with retention below threshold, most-at-risk first. |
| `reverie://l3/recent` | Recent episodic traces. |
| `reverie://stats/daily` | Per-day facts/episodes in/out + supersedes. |
| `reverie://sessions/{id}` | Per-session working-memory buffer, metadata, and budget. |

## Prompts

| Name | Purpose |
|---|---|
| `session_start` | Walk through `memory_session_init` + `reverie://l1/index` + session-scoped `memory_recall`. Takes `session_id` (required) and `project_hint` (optional). |
| `session_end` | Walk through `memory_session_end` (with optional episode payload). Takes `session_id` (required). |

## Knowledge graph

Reverie's graph layer connects memories (L2 facts, L3 episodes) and entities through typed directed edges. Edge types and entity types are caller-supplied strings -- nothing is enforced at the schema layer -- but the lists below are the canonical taxonomy the system understands.

### Edge types

- `evidence` -- supporting reference (episodes evidencing facts, citations).
- `causes` -- cause-to-effect relationship.
- `contradicts` -- known conflict between two memories.
- `supports` -- soft endorsement (weaker than `evidence`).
- `refines` -- successor that clarifies or extends without superseding.
- `depends_on` -- prerequisite relationship.
- `references` -- non-directional pointer; default for generic links.

### Entity types

- `file`, `repo`, `library`, `concept`, `person`, `command` -- the hosts' typical extractions.
- Free-form: callers can store any string; reverie does not enforce a closed set.
- Dedup: two entities with the same `(name, entity_type)` always merge; entities differing only in `entity_type` are distinct, so `"foo" (file)` and `"foo" (concept)` are two different entities.

### Graph-aware recall

Vector recall finds memories whose text is similar to the query; graph expansion finds memories related to those seeds *through structure* -- direct edges (`causes` / `refines` / `contradicts` / ...) and shared entity mentions. Set `expand_via_graph: true` on `memory_recall` to walk the graph from each vector seed and merge reachable neighbors into the candidate set. This is the recommended mode for "what do I know about file X" questions, where the answer memories often don't share keywords with the query.

Neighbors are scored by `composite = seed_similarity * neighbor_retention * (graph_decay_per_hop ^ distance)`. With the default `graph_decay_per_hop = 0.5` and the default hop budget of 2, a memory that only shares an entity with a vector seed (memory -> entity -> memory, distance 2) still reaches the candidate set, scored at a quarter of the seed's contribution. `graph_hops` (1-3) overrides the budget per call.

Each `RecallCandidate` carries a `distance` field: `0` for vector hits, `>= 1` for graph neighbors at that BFS depth. Graph-only neighbors have `similarity = 0`, `gate_b_pass = false` deterministically (they were not found by cosine similarity, so the similarity gate does not apply). The `limit` is applied after merge -- top-N by `composite_score` survives, so a request for 10 results may return any mix of vector and graph hits.

Hub entities (entities mentioned by many memories) expand without per-seed truncation -- "memories about popular-file.go" should return all of them, not an arbitrary subset. A global `graph_max_visited` cap (default 2000) bounds pathological blowup on dense graphs, and a `graph_min_retention_for_expansion` pre-filter (default 0.05) skips heavily-decayed neighbors during BFS so they don't pollute the candidate set or waste the visited budget. Both knobs live under `[memory]` in `reverie.toml`.

Recall filters (`cluster_id`, `subtype`, `layer`, `tags_any`) apply uniformly to vector and graph hits. `expand_via_graph` is honored only on `round == 0`; on round 1+ it is silently ignored (an info log line is emitted) and recall falls back to pre-7C behavior.

## Configuration

Copy `reverie.toml.example` to `~/.config/reverie/reverie.toml` and adjust as needed.

Key settings:

- **`[embedding] provider`**: `"openai_compat"` (default, for Ollama/LM Studio/OpenAI) or `"voyage"` (hosted, requires `VOYAGE_API_KEY`).
- **`[embedding] base_url`**: Ollama default is `http://localhost:11434/v1`.
- **`[memory] similarity_threshold`**: `0.70` for Voyage/mxbai-large; drop to `0.55-0.60` for nomic-embed-text.
- **`[decay] temperature`**: controls how slowly memories fade. Higher = more gradual.

See [reverie.toml.example](reverie.toml.example) for the full annotated config.

Environment variable overrides: `REVERIE_DB_PATH`, `REVERIE_EMBED_URL`, `REVERIE_EMBED_MODEL`, `REVERIE_CONFIG`, `REVERIE_LOG_LEVEL`, `REVERIE_DISABLED=1`.

## Replacing auto-memory

Reverie is designed to replace Claude Code's built-in auto-memory system entirely. See [docs/replacing-auto-memory.md](docs/replacing-auto-memory.md) for the migration guide.

## Development

```bash
go test ./...
```

Smoke tests (require Ollama running):

```bash
REVERIE_SMOKE_TEST=1 go test ./internal/embed/ -run Smoke -v
```

## License

TODO: pick a license.
