# OpenCode templates for Reverie

This directory holds OpenCode subagent definitions that Reverie callers can drop
into their OpenCode config.

## Contents

- `agents/memory-judge.md` -- subagent that applies Gate A (relevance/staleness
  judgment) to candidates returned by `memory_recall`. Spawned by a primary
  OpenCode agent after recall, then its JSON verdicts are passed to
  `memory_apply_judgment`.

## Install

Copy `agents/memory-judge.md` to one of:

- `~/.config/opencode/agents/memory-judge.md` -- global, available in every
  OpenCode session.
- `<project>/.opencode/agents/memory-judge.md` -- project-local, scoped to a
  single repo.

Without this agent, Reverie's `memory_apply_judgment` (Gate A) is unavailable,
but recall still works -- Gates B (similarity) and C (retention) continue to
filter under OR logic.

See `../docs/opencode-setup.md` for the full setup flow.
