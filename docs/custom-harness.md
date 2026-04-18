# Custom Harness Setup

Guide for integrating reverie into custom API harnesses (Go or Python programs that call the Anthropic API directly).

## Overview

Reverie is an MCP server that communicates over stdio. Your harness spawns it as a subprocess, pipes JSON-RPC messages over stdin/stdout, and uses the MCP client SDK to call tools.

## Go

```go
import (
    "os/exec"
    "github.com/modelcontextprotocol/go-sdk/mcp"
)

cmd := exec.Command("reverie", "serve")
// The MCP SDK handles stdin/stdout piping.
client, err := mcp.NewStdioClient(cmd)
if err != nil {
    log.Fatal(err)
}
defer client.Close()

// Call memory_recall
result, err := client.CallTool(ctx, "memory_recall", map[string]any{
    "query": "what database does this project use",
    "limit": 5,
})
```

See the [MCP Go SDK](https://github.com/modelcontextprotocol/go-sdk) for full client examples.

## Python

```python
import subprocess
from mcp import StdioServerParameters, ClientSession

server = StdioServerParameters(command="reverie", args=["serve"])
async with ClientSession(server) as session:
    result = await session.call_tool("memory_recall", {
        "query": "what database does this project use",
        "limit": 5,
    })
```

See the [MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk) for full client examples.

## Gate A: the judgment step

The harness author is responsible for implementing Gate A (the uncertainty/staleness judge) if they want it. The flow:

1. Call `memory_recall` -- get back a `recall_id` and a list of candidates.
2. Make a separate Claude API call with a fresh system prompt (the judge prompt) and the candidate list. The judge prompt is documented in [`skills/memory-judge/SKILL.md`](../skills/memory-judge/SKILL.md).
3. Parse the judge's response -- a JSON array of `{"memory_id": "...", "keep": true|false, "reason": "..."}`.
4. Call `memory_apply_judgment` with the `recall_id` and the verdicts.

Example judge call (Go, using the Anthropic SDK):

```go
resp, err := anthropic.Messages.Create(ctx, anthropic.MessageCreateParams{
    Model: "claude-sonnet-4-20250514",
    System: judgeSystemPrompt, // from SKILL.md
    Messages: []anthropic.Message{
        {Role: "user", Content: formatCandidatesForJudge(query, candidates)},
    },
})
// Parse resp into []Verdict, then:
client.CallTool(ctx, "memory_apply_judgment", map[string]any{
    "recall_id": recallID,
    "verdicts":  verdicts,
})
```

If you don't need Gate A, skip step 2-4 and use the candidates from `memory_recall` directly. Gates B (similarity) and C (Ebbinghaus retention) still apply.

## Notes

- Reverie makes no Anthropic API calls itself. The only outbound call is to the embedding endpoint (Ollama by default).
- Each harness instance spawns its own reverie process over stdio. This is standard MCP -- one server per client.
- If you need multiple harnesses to share the same memory store, they can -- SQLite WAL supports concurrent readers. But only one writer at a time, so avoid concurrent writes from multiple processes.
