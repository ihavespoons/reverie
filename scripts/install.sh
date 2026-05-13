#!/usr/bin/env bash
# Reverie installer: builds the binary and wires it into Claude Code,
# Claude Desktop, and/or OpenCode on macOS / Linux.
#
# Defaults: Ollama provider with nomic-embed-text (no API keys required).
# Re-run safe -- config merges are idempotent; existing MCP servers are
# preserved. Existing config files are backed up before rewrite.
#
# Usage:
#   ./install.sh                  # build + install + configure detected clients
#   ./install.sh --skip-build     # skip go install (use the binary already on PATH)
#   ./install.sh --skip-ollama    # don't touch Ollama (assume already configured)
#   ./install.sh --code-only      # configure Claude Code only
#   ./install.sh --desktop-only   # configure Claude Desktop only
#   ./install.sh --opencode-only  # configure OpenCode only
#   ./install.sh --uninstall      # remove the reverie entry from configured clients

set -uo pipefail

# --- styling ---
if [ -t 1 ]; then
    BOLD=$(tput bold 2>/dev/null || true)
    DIM=$(tput dim 2>/dev/null || true)
    GREEN=$(tput setaf 2 2>/dev/null || true)
    YELLOW=$(tput setaf 3 2>/dev/null || true)
    RED=$(tput setaf 1 2>/dev/null || true)
    RESET=$(tput sgr0 2>/dev/null || true)
else
    BOLD=""; DIM=""; GREEN=""; YELLOW=""; RED=""; RESET=""
fi

ok()    { printf "%s✓%s %s\n" "$GREEN" "$RESET" "$1"; }
info()  { printf "%s•%s %s\n" "$BOLD" "$RESET" "$1"; }
warn()  { printf "%s!%s %s\n" "$YELLOW" "$RESET" "$1" >&2; }
fail()  { printf "%s✗%s %s\n" "$RED"   "$RESET" "$1" >&2; exit 1; }

# --- args ---
DO_BUILD=1
DO_OLLAMA=1
DO_CODE=1
DO_DESKTOP=1
DO_OPENCODE=1
UNINSTALL=0

for arg in "$@"; do
    case "$arg" in
        --skip-build)    DO_BUILD=0 ;;
        --skip-ollama)   DO_OLLAMA=0 ;;
        --code-only)     DO_DESKTOP=0; DO_OPENCODE=0 ;;
        --desktop-only)  DO_CODE=0; DO_OPENCODE=0 ;;
        --opencode-only) DO_CODE=0; DO_DESKTOP=0 ;;
        --uninstall)     UNINSTALL=1; DO_BUILD=0; DO_OLLAMA=0 ;;
        -h|--help)
            sed -n '2,16p' "$0" | sed 's/^# \{0,1\}//'
            exit 0 ;;
        *)
            fail "unknown argument: $arg (see --help)" ;;
    esac
done

# --- paths ---
OS="$(uname -s)"
case "$OS" in
    Darwin)
        DESKTOP_CONFIG="$HOME/Library/Application Support/Claude/claude_desktop_config.json"
        ;;
    Linux)
        DESKTOP_CONFIG="$HOME/.config/Claude/claude_desktop_config.json"
        ;;
    *)
        warn "unsupported OS: $OS — Desktop config path may be wrong"
        DESKTOP_CONFIG="$HOME/.config/Claude/claude_desktop_config.json"
        ;;
esac
CODE_CONFIG="$HOME/.claude/settings.json"
OPENCODE_CONFIG="$HOME/.config/opencode/opencode.json"

# --- utilities ---
require_cmd() {
    command -v "$1" >/dev/null 2>&1 || fail "$1 not found in PATH${2:+ — $2}"
}

merge_config() {
    # Args: $1 = config path, $2 = binary path, $3 = client label (for messages)
    local cfg="$1"
    local bin="$2"
    local label="$3"

    if [ ! -e "$cfg" ]; then
        info "$label: creating $cfg"
        mkdir -p "$(dirname "$cfg")"
        printf '{\n  "mcpServers": {}\n}\n' > "$cfg"
    else
        local backup="$cfg.bak.$(date +%s)"
        cp "$cfg" "$backup"
        ok "$label: backed up existing config to $backup"
    fi

    local entry
    entry=$(jq -n --arg cmd "$bin" '{type:"stdio", command:$cmd, args:["serve"]}')

    local merged
    merged=$(jq --argjson entry "$entry" '
        .mcpServers = (.mcpServers // {}) |
        .mcpServers.reverie = $entry
    ' "$cfg") || fail "$label: jq merge failed"

    printf '%s\n' "$merged" > "$cfg"
    ok "$label: wired reverie into $cfg"
}

remove_config() {
    local cfg="$1"
    local label="$2"
    if [ ! -e "$cfg" ]; then
        info "$label: no config file at $cfg, nothing to remove"
        return
    fi
    if ! jq -e '.mcpServers.reverie' "$cfg" >/dev/null 2>&1; then
        info "$label: no reverie entry in $cfg"
        return
    fi
    local backup="$cfg.bak.$(date +%s)"
    cp "$cfg" "$backup"
    local stripped
    stripped=$(jq 'del(.mcpServers.reverie)' "$cfg") \
        || fail "$label: jq strip failed"
    printf '%s\n' "$stripped" > "$cfg"
    ok "$label: removed reverie entry (backup at $backup)"
}

merge_opencode_config() {
    # Args: $1 = config path, $2 = binary path, $3 = client label (for messages)
    # OpenCode uses a different schema -- `.mcp` (not `.mcpServers`) with
    # `command` as an array [executable, ...args] and a `type: "local"` tag.
    local cfg="$1"
    local bin="$2"
    local label="$3"

    if [ ! -e "$cfg" ]; then
        info "$label: creating $cfg"
        mkdir -p "$(dirname "$cfg")"
        printf '{\n  "mcp": {}\n}\n' > "$cfg"
    else
        local backup="$cfg.bak.$(date +%s)"
        cp "$cfg" "$backup"
        ok "$label: backed up existing config to $backup"
    fi

    local entry
    entry=$(jq -n --arg cmd "$bin" '{type:"local", command:[$cmd, "serve"], enabled:true}')

    local merged
    merged=$(jq --argjson entry "$entry" '
        .mcp = (.mcp // {}) |
        .mcp.reverie = $entry
    ' "$cfg") || fail "$label: jq merge failed"

    printf '%s\n' "$merged" > "$cfg"
    ok "$label: wired reverie into $cfg"
}

remove_opencode_config() {
    local cfg="$1"
    local label="$2"
    if [ ! -e "$cfg" ]; then
        info "$label: no config file at $cfg, nothing to remove"
        return
    fi
    if ! jq -e '.mcp.reverie' "$cfg" >/dev/null 2>&1; then
        info "$label: no reverie entry in $cfg"
        return
    fi
    local backup="$cfg.bak.$(date +%s)"
    cp "$cfg" "$backup"
    local stripped
    stripped=$(jq 'del(.mcp.reverie)' "$cfg") \
        || fail "$label: jq strip failed"
    printf '%s\n' "$stripped" > "$cfg"
    ok "$label: removed reverie entry (backup at $backup)"
}

# --- preflight ---
require_cmd jq "install via brew install jq, apt install jq, or equivalent"

# --- uninstall path ---
if [ "$UNINSTALL" -eq 1 ]; then
    info "uninstall mode"
    [ "$DO_CODE"     -eq 1 ] && remove_config "$CODE_CONFIG"             "Claude Code"
    [ "$DO_DESKTOP"  -eq 1 ] && remove_config "$DESKTOP_CONFIG"          "Claude Desktop"
    [ "$DO_OPENCODE" -eq 1 ] && remove_opencode_config "$OPENCODE_CONFIG" "OpenCode"
    info "binary at $(command -v reverie 2>/dev/null || echo "<not on PATH>") left in place -- remove manually if desired"
    exit 0
fi

# --- preflight (install path) ---
if [ "$DO_BUILD" -eq 1 ]; then
    require_cmd go "install Go 1.22+ from https://go.dev/dl/"
    GO_VERSION=$(go version | awk '{print $3}' | sed 's/^go//')
    GO_MAJOR=${GO_VERSION%%.*}
    GO_MINOR=$(printf '%s' "$GO_VERSION" | awk -F. '{print $2}')
    if [ "$GO_MAJOR" -lt 1 ] || { [ "$GO_MAJOR" -eq 1 ] && [ "$GO_MINOR" -lt 22 ]; }; then
        fail "Go 1.22+ required, found $GO_VERSION"
    fi
    ok "Go $GO_VERSION"
fi

# --- Ollama check + model pull ---
if [ "$DO_OLLAMA" -eq 1 ]; then
    if curl -fsS --max-time 2 http://localhost:11434/api/tags >/dev/null 2>&1; then
        ok "Ollama running on :11434"
    else
        warn "Ollama not reachable on :11434 — start it (brew services start ollama, or ollama serve in another terminal)"
        warn "continuing — pull will be skipped"
        DO_OLLAMA=0
    fi
fi

if [ "$DO_OLLAMA" -eq 1 ]; then
    if curl -fsS --max-time 2 http://localhost:11434/api/tags 2>/dev/null \
        | jq -e '.models[]?.name | select(startswith("nomic-embed-text"))' >/dev/null 2>&1; then
        ok "nomic-embed-text already pulled"
    else
        info "pulling nomic-embed-text (one-time, ~270MB)"
        if ! ollama pull nomic-embed-text; then
            warn "ollama pull failed — install will continue but recall won't work until you pull manually"
        fi
    fi
fi

# --- build ---
if [ "$DO_BUILD" -eq 1 ]; then
    SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
    REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
    if [ -f "$REPO_DIR/go.mod" ] && [ -d "$REPO_DIR/cmd/reverie" ]; then
        info "go install from $REPO_DIR"
        ( cd "$REPO_DIR" && go install ./cmd/reverie ) || fail "go install failed"
    else
        info "go install github.com/ihavespoons/reverie/cmd/reverie@latest"
        go install github.com/ihavespoons/reverie/cmd/reverie@latest \
            || fail "go install from module path failed — clone the repo and re-run from there, or check network"
    fi
fi

# --- locate binary ---
BIN="$(command -v reverie || true)"
if [ -z "$BIN" ]; then
    GOPATH_BIN="$(go env GOPATH 2>/dev/null)/bin/reverie"
    [ -x "$GOPATH_BIN" ] && BIN="$GOPATH_BIN"
fi
[ -n "$BIN" ] || fail "reverie binary not found after install — check that \$(go env GOPATH)/bin is on PATH"
ok "binary: $BIN"

# --- configure clients ---
if [ "$DO_CODE" -eq 1 ]; then
    if [ -e "$CODE_CONFIG" ] || [ -d "$HOME/.claude" ]; then
        merge_config "$CODE_CONFIG" "$BIN" "Claude Code"
    else
        info "Claude Code config dir not detected at ~/.claude — skipping (use --code-only to force)"
    fi
fi

if [ "$DO_DESKTOP" -eq 1 ]; then
    if [ -e "$DESKTOP_CONFIG" ] || [ -d "$(dirname "$DESKTOP_CONFIG")" ]; then
        merge_config "$DESKTOP_CONFIG" "$BIN" "Claude Desktop"
    else
        info "Claude Desktop config dir not detected -- skipping (use --desktop-only to force)"
    fi
fi

if [ "$DO_OPENCODE" -eq 1 ]; then
    if [ -e "$OPENCODE_CONFIG" ] || [ -d "$(dirname "$OPENCODE_CONFIG")" ]; then
        merge_opencode_config "$OPENCODE_CONFIG" "$BIN" "OpenCode"
    else
        info "OpenCode config dir not detected at ~/.config/opencode -- skipping (use --opencode-only to force)"
    fi
fi

# --- restart hints ---
echo
ok "install complete"
echo
echo "${BOLD}Next steps:${RESET}"
echo "  • Claude Code: type /exit and reopen, or restart the IDE."
echo "  • Claude Desktop: ${BOLD}fully quit${RESET} (Cmd-Q on macOS) and reopen -- closing the window is not enough."
echo "  • OpenCode: exit (Ctrl-C or :quit) and relaunch."
echo
echo "${DIM}Test: reverie status${RESET}"
