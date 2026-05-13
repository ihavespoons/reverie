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

Add to `~/.claude/settings.json`:

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

Replace `/path/to/reverie` with the actual binary path (e.g., the output of `go env GOPATH`/bin/reverie if installed via `go install`).

No API keys needed with Ollama. For Voyage, add `"env": {"VOYAGE_API_KEY": "${VOYAGE_API_KEY}"}`.

Restart Claude Code after adding the entry.

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

## Tools

| Tool | Purpose | When to call |
|---|---|---|
| `memory_recall` | Search memory by query. Returns ranked candidates with gate pass flags. Accepts optional `session_id` to auto-update the session buffer. | Session start; before architectural decisions; when referencing prior context. |
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
