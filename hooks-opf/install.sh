#!/usr/bin/env bash
# Installs hooks-opf and wires UserPromptSubmit + PostToolUse on whichever agents are present.
#
# Detects Claude Code (~/.claude/) and Codex (~/.codex/). Auto-wires each that exists.
#
# Usage:
#   curl -fsSL https://raw.githubusercontent.com/CJHwong/agent-seatbelt/main/hooks-opf/install.sh | bash
#   curl -fsSL .../install.sh | bash -s -- --prompt-only   # skip PostToolUse
#   curl -fsSL .../install.sh | bash -s -- --no-codex      # skip Codex even if present
#   curl -fsSL .../install.sh | bash -s -- --no-pilot      # skip the model warm-up run
#
# Before wiring, a pilot run resolves uv deps and downloads the ~30MB model, then
# leaves the server warm — so the first agent session skips the cold start. The
# server is a shared singleton on 127.0.0.1:9123 that both agents reuse.
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
PORT="${PII_PORT:-9123}"
SERVER_LOG="${PII_SERVER_LOG:-$HOME/.cache/opf/server.log}"

# Tools whose output can carry external PII. Edit/Write/Glob/LS/Todo etc. only
# emit structural metadata, so scanning them is wasted work. Codex aliases file
# edits to apply_patch; both Claude's Edit/Write and Codex's apply_patch fall
# outside this pattern and are skipped.
POSTTOOL_MATCHER='^(Bash|Read|NotebookRead|WebFetch|WebSearch|Agent|Task|mcp__.*)$'

PROMPT_ONLY=0
SKIP_CODEX=0
RUN_PILOT=1
while [[ $# -gt 0 ]]; do
    case "$1" in
        --prompt-only) PROMPT_ONLY=1; shift ;;
        --no-codex)    SKIP_CODEX=1; shift ;;
        --no-pilot)    RUN_PILOT=0; shift ;;
        -h|--help)
            sed -n '2,18p' "$0" | sed 's/^# \{0,1\}//'
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

# Add or update an entry in a hooks-shaped JSON file.
# Idempotent on command string: if an entry already references $cmd, replace it
# with $entry (this is how matcher changes propagate to existing installs);
# otherwise append.
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
          ($entries | map(if any(.hooks[]?; .command == $cmd) then $entry else . end)) as $mapped |
          if any($mapped[]?; any(.hooks[]?; .command == $cmd)) then $mapped
          else $mapped + [$entry] end
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
        posttool_entry=$(jq -cn --arg cmd "$posttool_cmd" --arg matcher "$POSTTOOL_MATCHER" \
            '{matcher:$matcher,hooks:[{type:"command",command:$cmd,timeout:20}]}')
        add_entry "$target" "PostToolUse" "$posttool_entry" "$posttool_cmd"
        echo "  [$label] PostToolUse ($POSTTOOL_MATCHER) -> $posttool_cmd"
    else
        echo "  [$label] PostToolUse skipped (--prompt-only)"
    fi
}

# Start the server once so uv deps + the ~30MB model download happen now, not in
# the user's first agent turn. Waits far longer than the hook's 10s (a cold
# download can take ~30s), then smoke-tests one known-PII string. Leaves the
# server running — it's the same 127.0.0.1:$PORT singleton the hooks reuse.
# Sets PILOT_OK on success. Fail-soft: a miss here just means the first real
# prompt pays the cold start, same as before this step existed.
pilot_run() {
    local health="http://127.0.0.1:$PORT/health"
    if curl -sSf --max-time 1 "$health" >/dev/null 2>&1; then
        echo "  server already warm on 127.0.0.1:$PORT — nothing to do"
        PILOT_OK=1
        return
    fi
    echo "  resolving deps + downloading the ~30MB model (one-time)..."
    mkdir -p "$(dirname "$SERVER_LOG")"
    nohup uv run "$SERVER_DEST" --port "$PORT" >"$SERVER_LOG" 2>&1 </dev/null &
    disown
    local i
    for i in $(seq 1 120); do   # up to ~60s for a cold download
        curl -sSf --max-time 1 "$health" >/dev/null 2>&1 && break
        sleep 0.5
    done
    if ! curl -sSf --max-time 1 "$health" >/dev/null 2>&1; then
        echo "  server not up after ~60s; model may still be downloading in the" >&2
        echo "  background. It will finish on first agent use. Log: $SERVER_LOG" >&2
        return
    fi
    local resp spans
    resp=$(curl -sS --max-time 10 -X POST "http://127.0.0.1:$PORT/" \
        -H 'Content-Type: application/json' \
        -d '{"text":"reach me at pilot@example.com"}' 2>/dev/null) || true
    spans=$(printf '%s' "$resp" | jq -r '.spans // [] | length' 2>/dev/null)
    if [ "${spans:-0}" -ge 1 ]; then
        echo "  ok — server warm, smoke test flagged ${spans} span(s), cached at ~/.cache/opf/"
        PILOT_OK=1
    else
        echo "  server up but smoke test flagged nothing; check $SERVER_LOG" >&2
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

PILOT_OK=0
if [ "$RUN_PILOT" -eq 1 ]; then
    echo
    echo "Pilot run (before wiring)..."
    pilot_run
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
if [ "$PILOT_OK" -eq 1 ]; then
    echo "Model is warm (pilot run) — the first agent prompt won't wait on the download."
else
    echo "First matching prompt may be slow — uv resolves deps and ~30MB model assets download to ~/.cache/opf/."
fi
if [ "$CODEX_PRESENT" -eq 1 ]; then
    echo
    echo "Codex only: hooks require trust before they run. Launch codex, run /hooks,"
    echo "and trust the pii-check entries (trust is remembered in ~/.codex/config.toml"
    echo "under [hooks.state]). Re-running this installer changes the command string,"
    echo "so Codex will ask you to re-trust. Claude Code needs no trust step."
fi
echo
echo "Tuning:"
echo "  PII_BLOCK_LEVEL=off       disable all checks"
echo "  PII_BLOCK_LEVEL=relaxed   block only secrets + account numbers"
echo "  PII_BLOCK_LEVEL=standard  + emails, phones, addresses (default)"
echo "  PII_BLOCK_LEVEL=strict    + names, urls, dates"
echo "  PII_ALLOW_LABELS=private_url,private_date  allow specific labels within the selected tier"
echo "  Prefix a prompt with 'pii:off ' to bypass a single submission."
