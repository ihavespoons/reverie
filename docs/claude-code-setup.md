# Claude Code Setup

Step-by-step guide for wiring reverie into Claude Code.

## 1. Build reverie

Option A -- install to `$GOPATH/bin`:

```bash
cd /path/to/reverie
go install ./cmd/reverie
```

The binary lands at `$(go env GOPATH)/bin/reverie`. Make sure that directory is on your `$PATH`.

Option B -- build in place:

```bash
cd /path/to/reverie
go build -o reverie ./cmd/reverie
```

Use the absolute path to the binary in the config below.

## 2. Ensure Ollama is running

Reverie uses Ollama for local embeddings by default. Pull the model if you haven't already:

```bash
ollama pull nomic-embed-text
```

Verify Ollama is running:

```bash
curl -s http://localhost:11434/v1/models | head -5
```

If Ollama isn't running when reverie starts, `memory_write` and `memory_recall` return clean errors -- no data is corrupted.

## 3. Add to settings.json

Edit `~/.claude/settings.json` and add the `reverie` entry under `mcpServers`:

```json
{
  "mcpServers": {
    "reverie": {
      "type": "stdio",
      "command": "reverie",
      "args": ["serve"]
    }
  }
}
```

If `reverie` isn't on your `$PATH`, use the full path:

```json
{
  "mcpServers": {
    "reverie": {
      "type": "stdio",
      "command": "/Users/you/Code/personal/reverie/reverie",
      "args": ["serve"]
    }
  }
}
```

No API keys or `env` block needed with Ollama. If using Voyage instead:

```json
{
  "mcpServers": {
    "reverie": {
      "type": "stdio",
      "command": "reverie",
      "args": ["serve"],
      "env": {
        "VOYAGE_API_KEY": "${VOYAGE_API_KEY}"
      }
    }
  }
}
```

## 4. Add the reverie preamble to CLAUDE.md

Add the following to your `~/.claude/CLAUDE.md` (append it after any existing content):

```markdown
## Memory — Reverie

All persistent memory goes through the `reverie` MCP server. Do not write to ~/.claude/projects/*/memory/ files — that system is disabled.

### Recall
- At session start, call `memory_recall` with the project/task context. Prefer reading `reverie://l1/index` before your first recall.
- Before architectural decisions, recall relevant project/reference memories.
- If a recall returns more than ~5 candidates OR the query is sensitive to staleness (user asking about "current" state, deciding between competing facts), follow up with the `memory-judge` skill: spawn a Task subagent with the candidates, collect keep/drop verdicts, then call `memory_apply_judgment` with the results. For quick lookups, use the candidates as-is.

### Write (type must be one of user | feedback | project | reference)
- user — stable personal facts (role, preferences, skills)
- feedback — how to behave (corrections you want preserved)
- project — architecture, conventions, decisions for a codebase
- reference — pointers to docs/repos/URLs
- If the content is retrospective (situation → action → outcome → lesson), pass an `episode` payload to promote to L3.
Do NOT write transient task state.

### Reinforce & forget
- After using recalled memories in a response, call `memory_reinforce` with their IDs.
- On user correction, `memory_forget` the stale memory and write the correction.
```

This teaches Claude Code how and when to use reverie's tools in every session.

## 5. Restart Claude Code

Quit and relaunch the `claude` CLI. MCP servers are discovered at startup.

## 6. Verify

Start a Claude Code session and run `/mcp`. You should see `reverie` listed as a connected server with 7+ tools:

- `memory_recall`
- `memory_write`
- `memory_apply_judgment`
- `memory_reinforce`
- `memory_forget`
- `memory_list`
- `memory_decay_tick`

If reverie is not listed, check:
- Is Ollama running? (`ollama list` should show `nomic-embed-text`)
- Is the binary path correct in settings.json?
- Check stderr output: `reverie serve 2>reverie.log` and inspect `reverie.log`.

## 7. Test

Ask Claude to write a test memory:

> "Write a memory that my preferred language is Go."

Then recall it:

> "Recall memories about my language preferences."

You should see the fact returned with a similarity score and gate pass flags. If you see results, reverie is working end-to-end.
