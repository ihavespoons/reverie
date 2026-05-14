# Claude Desktop Setup

Guide for wiring reverie into Claude Desktop on macOS.

## 1. Build reverie

```bash
cd /path/to/reverie
go build -o reverie ./cmd/reverie
```

Note the absolute path to the binary -- Desktop requires a full path.

## 2. Add to claude_desktop_config.json

Edit `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "reverie": {
      "type": "stdio",
      "command": "/Users/you/Code/github.com/diffsec/reverie/reverie",
      "args": ["serve"]
    }
  }
}
```

Replace the `command` path with your actual binary location.

## 3. Environment variables and wrapper scripts

Desktop does **NOT** support `${ENV_VAR}` interpolation in config values. With Ollama (default), no env vars are needed -- the config above works as-is.

If you need environment variables (e.g., for Voyage), create a wrapper script:

```bash
#!/bin/bash
export VOYAGE_API_KEY="your-key-here"
exec /path/to/reverie serve
```

Save it as e.g. `~/bin/reverie-wrapper.sh`, make it executable (`chmod +x`), and point the config at it:

```json
{
  "mcpServers": {
    "reverie": {
      "type": "stdio",
      "command": "/Users/you/bin/reverie-wrapper.sh"
    }
  }
}
```

## 4. Gate A limitations

Desktop does **NOT** have Task/subagent support. This means:

- `memory_apply_judgment` (Gate A -- the uncertainty/staleness judge) is unavailable.
- `memory_recall` still works -- candidates are returned under OR logic with Gates B (similarity) + C (Ebbinghaus retention) only.
- The `superseded_by` chain on L2 facts catches the most common staleness case (explicit corrections), so Desktop recall is still usable -- just less discriminating on ambiguous candidates than Claude Code with full Gate A.

This is a known trade-off, not a bug. The tool descriptions make this explicit so the agent won't attempt to spawn a subagent.

## 5. Restart Desktop

Quit Claude Desktop completely (Cmd+Q, not just close the window) and relaunch. MCP servers are discovered at startup.

## 6. Verify

Start a new conversation and ask: "What MCP tools do you have access to?" You should see the reverie tools listed. Test with a write and recall:

> "Write a memory that my preferred language is Go."
> "Recall memories about my language preferences."
