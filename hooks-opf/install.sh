#!/usr/bin/env bash
# Installs hooks-opf and wires UserPromptSubmit + PostToolUse on whichever agents are present.
#
# Detects Claude Code (~/.claude/) and Codex (~/.codex/). Auto-wires each that exists.
#
# Usage:
#   curl -fsSL https://raw.githubusercontent.com/CJHwong/agent-seatbelt/main/hooks-opf/install.sh | bash
#   curl -fsSL .../install.sh | bash -s -- --prompt-only   # skip PostToolUse
#   curl -fsSL .../install.sh | bash -s -- --no-codex      # skip Codex even if present
#
# Scripts are installed to ~/.claude/hooks/ regardless of agent. Both agents reference that path.
# Idempotent. Re-running won't duplicate hook entries.

set -euo pipefail

REPO_BASE="${HOOKS_OPF_BASE_URL:-https://raw.githubusercontent.com/CJHwong/agent-seatbelt/main/hooks-opf}"
HOOKS_DIR="$HOME/.claude/hooks"
SERVER_DEST="$HOOKS_DIR/pii-server.py"
CHECK_DEST="$HOOKS_DIR/pii-check.sh"
CLAUDE_SETTINGS="$HOME/.claude/settings.json"
CODEX_HOOKS="$HOME/.codex/hooks.json"

PROMPT_ONLY=0
SKIP_CODEX=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --prompt-only) PROMPT_ONLY=1; shift ;;
        --no-codex)    SKIP_CODEX=1; shift ;;
        -h|--help)
            sed -n '2,13p' "$0" | sed 's/^# \{0,1\}//'
            exit 0
            ;;
        *) echo "unknown arg: $1" >&2; exit 1 ;;
    esac
done

need_cmd() {
    command -v "$1" >/dev/null 2>&1 || {
        echo "Error: $1 is required but not on PATH." >&2
        exit 1
    }
}

need_cmd curl
need_cmd jq
need_cmd uv

mkdir -p "$HOOKS_DIR"

echo "Downloading hook files..."
curl -fsSL "$REPO_BASE/pii-server.py" -o "$SERVER_DEST"
curl -fsSL "$REPO_BASE/pii-check.sh"  -o "$CHECK_DEST"
chmod +x "$CHECK_DEST"
echo "Installed: $SERVER_DEST"
echo "Installed: $CHECK_DEST"

# Add an entry to a hooks-shaped JSON file, idempotent on the command string.
# Args: $1 = target file, $2 = event key, $3 = entry JSON, $4 = command string to match.
add_entry() {
    local target="$1" event="$2" entry="$3" cmd="$4"
    if [ ! -f "$target" ]; then
        mkdir -p "$(dirname "$target")"
        echo '{}' > "$target"
    fi
    local tmp
    tmp=$(mktemp)
    jq \
        --arg event "$event" \
        --arg cmd "$cmd" \
        --argjson entry "$entry" \
        '
        .hooks = (.hooks // {}) |
        .hooks[$event] = (
          (.hooks[$event] // []) as $entries |
          if any($entries[]?; any(.hooks[]?; .command == $cmd)) then $entries
          else $entries + [$entry] end
        )
        ' "$target" > "$tmp"
    mv "$tmp" "$target"
}

wire_agent() {
    local label="$1" target="$2" posttool_mode="$3"

    local prompt_cmd="$CHECK_DEST --mode prompt"
    local posttool_cmd="$CHECK_DEST --mode $posttool_mode"

    local prompt_entry
    prompt_entry=$(jq -cn --arg cmd "$prompt_cmd" \
        '{hooks:[{type:"command",command:$cmd,timeout:20}]}')
    add_entry "$target" "UserPromptSubmit" "$prompt_entry" "$prompt_cmd"
    echo "  [$label] UserPromptSubmit -> $prompt_cmd"

    if [ "$PROMPT_ONLY" -eq 0 ]; then
        local posttool_entry
        posttool_entry=$(jq -cn --arg cmd "$posttool_cmd" \
            '{matcher:"*",hooks:[{type:"command",command:$cmd,timeout:20}]}')
        add_entry "$target" "PostToolUse" "$posttool_entry" "$posttool_cmd"
        echo "  [$label] PostToolUse (*) -> $posttool_cmd"
    else
        echo "  [$label] PostToolUse skipped (--prompt-only)"
    fi
}

CLAUDE_PRESENT=0
CODEX_PRESENT=0
[ -d "$HOME/.claude" ] && CLAUDE_PRESENT=1
[ -d "$HOME/.codex" ] && CODEX_PRESENT=1
[ "$SKIP_CODEX" -eq 1 ] && CODEX_PRESENT=0

if [ "$CLAUDE_PRESENT" -eq 0 ] && [ "$CODEX_PRESENT" -eq 0 ]; then
    echo "Neither ~/.claude/ nor ~/.codex/ found. Install at least one agent first." >&2
    exit 1
fi

echo
echo "Wiring hooks..."
if [ "$CLAUDE_PRESENT" -eq 1 ]; then
    wire_agent "claude" "$CLAUDE_SETTINGS" "claude-posttool"
fi
if [ "$CODEX_PRESENT" -eq 1 ]; then
    wire_agent "codex"  "$CODEX_HOOKS"     "codex-posttool"
fi

echo
echo "Done. Restart any running agent for the changes to take effect."
echo "First matching prompt may be slow — uv resolves deps and ~30MB model assets download to ~/.cache/opf/."
echo
echo "Tuning:"
echo "  PII_BLOCK_LEVEL=off       disable all checks"
echo "  PII_BLOCK_LEVEL=relaxed   block only secrets + account numbers"
echo "  PII_BLOCK_LEVEL=standard  + emails, phones, addresses (default)"
echo "  PII_BLOCK_LEVEL=strict    + names, urls, dates"
echo "  Prefix a prompt with 'pii:off ' to bypass a single submission."
