# Replacing Claude Code's Auto-Memory with Reverie

## Background

Claude Code has a built-in auto-memory system that stores memories as markdown files in `~/.claude/projects/<project>/memory/`. It works but has limitations:

- **Code-only**: memories are not accessible from Claude Desktop or custom API harnesses.
- **No recall logic**: all memories are injected into context every session (no search, no ranking).
- **No decay**: memories accumulate forever with no staleness filtering.
- **No structured types**: no distinction between facts and episodes, no cross-references.

Reverie replaces this with a proper memory system: vector search, Ebbinghaus decay, three memory layers, and MCP-based access from any client.

## Step 1: Add the CLAUDE.md preamble

Append the following to `~/.claude/CLAUDE.md` (after any existing content):

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

This preamble teaches Claude Code to use reverie's tools instead of the built-in memory system.

## Step 2: Disable auto-memory

This is an open question. The auto-memory preamble is injected by Claude Code's built-in subsystem at the harness level -- it is not controlled by CLAUDE.md. Until a proper off-switch is available, **both** instruction blocks will appear in the system prompt and may compete.

Known options being investigated:
- A `settings.json` flag to disable auto-memory (not confirmed to exist yet).
- Removing or disabling the auto-memory skill/subsystem.
- Filing a feature request with Anthropic for an explicit opt-out.

In practice, the CLAUDE.md preamble's explicit "Do not write to ~/.claude/projects/*/memory/ files" instruction is usually sufficient to redirect the agent. The auto-memory preamble will still appear but the agent should follow the more specific reverie instructions.

## Step 3: Migrate existing memories (Phase 5)

Phase 5 will add `reverie import` to migrate existing auto-memory files into reverie:

```bash
# Import all projects
reverie import --all-projects

# Import a specific project
reverie import --project-dir ~/.claude/projects/-Users-you-Code-project/memory
```

The importer will:
- Walk `~/.claude/projects/*/memory/*.md` files.
- Parse YAML frontmatter (`name`, `description`, `type`).
- Map to reverie subtypes (user, feedback, project, reference).
- Embed via the configured provider.
- Write as L2 facts (or L3 episodes if the body has situation/action/outcome structure).
- Deduplicate via content hash.

**This is not yet implemented.** Until it ships:

- Existing memories in `~/.claude/projects/*/memory/` remain accessible to Claude Code through the built-in system.
- They will not be indexed by reverie.
- You can manually re-create important memories via `memory_write` in a Claude Code session.
- No data is lost -- the old files stay on disk untouched.

## Step 4: Verify

After setup, start a new Claude Code session and test:

1. Ask: "Write a memory that I prefer local-first tools." -- Should call `memory_write`.
2. Ask: "What are my preferences?" -- Should call `memory_recall`, not read from disk.
3. Check: `reverie status` -- Should show the new fact.

If Claude Code is still writing to `~/.claude/projects/*/memory/`, the auto-memory subsystem is overriding the preamble. Escalate to the disable-auto-memory investigation.
