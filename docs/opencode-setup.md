# OpenCode Setup

Step-by-step guide for wiring reverie into OpenCode (sst/opencode).

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

## 3. Add to opencode.json

OpenCode reads config from `~/.config/opencode/opencode.json` (global) or `opencode.json` / `opencode.jsonc` in your project root (project-local, highest precedence). The field is `mcp` -- **not** `mcpServers` -- and the `command` is a **single array** that includes the executable plus its args.

Edit your `opencode.json` and add the `reverie` entry under `mcp`:

```json
{
  "$schema": "https://opencode.ai/config.json",
  "mcp": {
    "reverie": {
      "type": "local",
      "command": ["reverie", "serve"],
      "enabled": true
    }
  }
}
```

If `reverie` isn't on your `$PATH`, use the full path:

```json
{
  "$schema": "https://opencode.ai/config.json",
  "mcp": {
    "reverie": {
      "type": "local",
      "command": ["/Users/you/Code/personal/reverie/reverie", "serve"],
      "enabled": true
    }
  }
}
```

No API keys or `environment` block needed with Ollama. If using Voyage instead, add `environment` with `{env:VAR}` interpolation (note: this is OpenCode's syntax, not `${VAR}`):

```json
{
  "$schema": "https://opencode.ai/config.json",
  "mcp": {
    "reverie": {
      "type": "local",
      "command": ["reverie", "serve"],
      "enabled": true,
      "environment": {
        "VOYAGE_API_KEY": "{env:VOYAGE_API_KEY}"
      }
    }
  }
}
```

## 4. Add the reverie preamble to AGENTS.md

OpenCode uses `AGENTS.md` as the equivalent of Claude Code's `CLAUDE.md` -- it's loaded into every session as project/global guidance. Place it at `~/.config/opencode/AGENTS.md` (global) or `AGENTS.md` in your project root (project-local).

Append the following to your `AGENTS.md`:

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

This teaches OpenCode how and when to use reverie's tools in every session.

## 5. Install the memory-judge subagent (for Gate A)

To use `memory_apply_judgment` (Gate A -- the uncertainty/staleness filter), OpenCode needs a subagent that can judge recall candidates. Copy the judge definition from this repo:

```bash
# Global (available in every project)
mkdir -p ~/.config/opencode/agents
cp opencode/agents/memory-judge.md ~/.config/opencode/agents/memory-judge.md

# Or project-local
mkdir -p .opencode/agents
cp opencode/agents/memory-judge.md .opencode/agents/memory-judge.md
```

OpenCode invokes subagents either automatically via its Task tool (when the primary agent matches the subagent's description) or manually via `@memory-judge` in a message.

Without this file, recall still works -- candidates are returned under OR logic with Gates B (similarity) + C (Ebbinghaus retention) only. The `superseded_by` chain on L2 facts catches the most common staleness case, so recall remains usable -- just less discriminating on ambiguous candidates than with full Gate A.

## 6. Restart OpenCode

Exit and relaunch OpenCode. MCP servers are discovered at startup.

## 7. Verify

OpenCode ships a `opencode mcp list` CLI subcommand that prints configured servers and their status:

```bash
opencode mcp list
```

You should see `reverie` listed. Inside an OpenCode session, ask "What MCP tools do you have access to?" -- consult OpenCode's docs for any in-TUI equivalent slash command. The expected tool list is the same 7+ tools as with Claude Code:

- `memory_recall`
- `memory_write`
- `memory_apply_judgment`
- `memory_reinforce`
- `memory_forget`
- `memory_list`
- `memory_decay_tick`

If reverie is not listed, check:
- Is Ollama running? (`ollama list` should show `nomic-embed-text`)
- Is the binary path correct in `opencode.json`?
- Did you use `mcp` (not `mcpServers`) and put the executable inside the `command` array?
- Check stderr output: `reverie serve 2>reverie.log` and inspect `reverie.log`. OpenCode also exposes `opencode mcp debug` for protocol-level inspection.

## 8. Test

Ask OpenCode to write a test memory:

> "Write a memory that my preferred language is Go."

Then recall it:

> "Recall memories about my language preferences."

You should see the fact returned with a similarity score and gate pass flags. If you see results, reverie is working end-to-end.
