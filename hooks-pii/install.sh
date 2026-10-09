#!/usr/bin/env bash
# Installs hooks-pii and wires UserPromptSubmit + PostToolUse on whichever agents are present.
#
# Detects Claude Code (~/.claude/) and Codex (~/.codex/). Auto-wires each that exists.
#
# Usage:
#   curl -fsSL https://raw.githubusercontent.com/CJHwong/agent-seatbelt/main/hooks-pii/install.sh | bash
#   curl -fsSL .../install.sh | bash -s -- --prompt-only   # skip the tool hooks
#   curl -fsSL .../install.sh | bash -s -- --no-codex      # skip Codex even if present
#   curl -fsSL .../install.sh | bash -s -- --no-pilot      # skip the model warm-up run
#   curl -fsSL .../install.sh | bash -s -- -y              # skip the risk prompt (agents, CI)
#
# Every run first warns that it runs downloaded code as you and asks [Y/n] on the
# terminal. With no terminal and no -y, it refuses before it writes anything.
#
# A server already running on the port is stopped, so the new files take effect.
# Before wiring, a pilot run resolves uv deps and starts the selected model, then
# leaves the server warm so the first agent session skips the cold start. The
# server is a shared singleton on 127.0.0.1:9123 that both agents reuse.
#
# Everything is installed to ~/.claude/hooks/pii/ regardless of agent. Both agents reference that path.
# An install from before that folder is moved into it, and its old files are removed.
# Idempotent. Re-running won't duplicate hook entries.

set -euo pipefail

REPO_BASE="${HOOKS_PII_BASE_URL:-https://raw.githubusercontent.com/CJHwong/agent-seatbelt/main/hooks-pii}"
# The native rules engine is a compiled file, so it ships as a release asset and
# not as a file in the tree.
NATIVE_BASE="${HOOKS_PII_NATIVE_BASE_URL:-https://github.com/CJHwong/agent-seatbelt/releases/latest/download}"
HOOKS_DIR="$HOME/.claude/hooks"
PII_DIR="$HOOKS_DIR/pii"
SERVER_DEST="$PII_DIR/server.py"
NATIVE_DEST="$PII_DIR/rules/pii_rules_native.abi3.so"
CHECK_DEST="$PII_DIR/check.sh"
CLIENT_DEST="$PII_DIR/hook"
# The server's files, at the same path under server/ in the repo and under $PII_DIR.
SERVER_FILES=(server.py answer.py)
SERVER_FILES+=(detectors/__init__.py detectors/redact.py detectors/redact_torch.py)
SERVER_FILES+=(detectors/privacy_filter.py detectors/tagger.py)
SERVER_FILES+=(rules/__init__.py rules/engine.py rules/tagger_rules.py rules/secrets.py)
SERVER_FILES+=(rules/cl100k_base.tokens.gz)
# What an install before $PII_DIR put in $HOOKS_DIR. Removed once the hooks point at
# $PII_DIR, and nothing else in $HOOKS_DIR is touched.
LEGACY_FILES=(pii-server.py pii_hook.py pii_redact_lite.py pii_redact_torch.py pii_opf.py)
LEGACY_FILES+=(pii_tagger.py pii_rules.py pii_tagger_rules.py pii_secret_patterns.py)
LEGACY_FILES+=(cl100k_base.tokens.gz pii_rules_native.abi3.so pii-check.sh pii-hook)
LEGACY_CHECK="$HOOKS_DIR/pii-check.sh"
LEGACY_CLIENT="$HOOKS_DIR/pii-hook"
CLAUDE_SETTINGS="$HOME/.claude/settings.json"
CODEX_HOOKS="$HOME/.codex/hooks.json"
PORT="${PII_PORT:-9123}"
SERVER_LOG="${PII_SERVER_LOG:-$HOME/.cache/pii/server.log}"
SERVER_MODE="${PII_SERVER_MODE:-redact}"
ACTION_MODE="${PII_ACTION_MODE:-warn}"

# Tools whose output can carry external PII. Edit/Write/Glob/LS/Todo etc. only
# emit structural metadata, so scanning them is wasted work. Codex aliases file
# edits to apply_patch; both Claude's Edit/Write and Codex's apply_patch fall
# outside this pattern and are skipped. Codex uses a wildcard because its tool
# identifiers vary by runtime, and the hook filters the returned text itself.
POSTTOOL_MATCHER_CLAUDE='^(Bash|Read|NotebookRead|WebFetch|WebSearch|Agent|Task|exec_command|mcp__.*)$'
POSTTOOL_MATCHER_CODEX='*'
# Tools whose INPUT can carry a value that must not leave. The matcher names tools,
# not binaries, so wget, curl, scp and every other command run inside Bash are covered
# by that one entry, and a tool nobody has written yet needs one more. WebSearch is
# here because a query string is URL-shaped, the same channel as WebFetch. Read and
# NotebookRead are absent because their input is a path, and Edit and Write because
# they carry the agent's own content, which is the same reasoning the post-tool
# matcher uses.
PRETOOL_MATCHER_CLAUDE='^(Bash|exec_command|WebFetch|WebSearch|Agent|Task|mcp__.*)$'
# Codex resolves Edit, Write and apply_patch to one tool name, and its shell tool is
# named differently across runtime versions, so the wildcard is the stable choice
# here for the same reason the post-tool matcher uses it. The hook filters the
# returned text itself, and a tool whose input is a path costs one short request.
PRETOOL_MATCHER_CODEX='*'

PROMPT_ONLY=0
SKIP_CODEX=0
RUN_PILOT=1
ASSUME_YES=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        -y|--yes)      ASSUME_YES=1; shift ;;
        --prompt-only) PROMPT_ONLY=1; shift ;;
        --no-codex)    SKIP_CODEX=1; shift ;;
        --no-pilot)    RUN_PILOT=0; shift ;;
        -h|--help)
            # Print the text below, never this file's own header. Piped from curl,
            # $0 is "bash" and there is no file to dump, so reading the header here
            # printed a sed error instead of usage.
            cat <<'USAGE'
Installs hooks-pii and wires the PII hooks on whichever agents are present.

Usage: install.sh [-y] [--prompt-only] [--no-codex] [--no-pilot]
  -y, --yes       skip the risk prompt (for agents and CI)
  --prompt-only   wire the prompt hook only, skipping both tool hooks
  --no-codex      ignore Codex even if ~/.codex exists
  --no-pilot      skip the model warm-up run

Environment: PII_SERVER_MODE, PII_ACTION_MODE, PII_LEVEL, PII_PORT,
PII_SERVER_LOG, PII_SKIP_EVENT_PATH, HOOKS_PII_BASE_URL, HOOKS_PII_NATIVE_BASE_URL.

Full notes: https://github.com/CJHwong/agent-seatbelt/blob/main/hooks-pii/README.md
USAGE
            exit 0
            ;;
        *) echo "unknown arg: $1" >&2; exit 1 ;;
    esac
done

# A piped install runs whatever the server sends, with your permissions, so ask
# first. The answer comes from /dev/tty because stdin is the script itself.
confirm_install() {
    cat >&2 <<'EOF'
WARNING: this installer downloads code from the internet and runs it as you.
It can read, change, or delete anything your user account can.
The server can send different code each time, so read the script first:
  https://github.com/CJHwong/agent-seatbelt/blob/main/hooks-pii/install.sh
Pass -y to skip this question (for agents and CI).
EOF
    if ! (: </dev/tty) 2>/dev/null; then
        echo "install.sh: no terminal to ask on; re-run with -y to accept the risk" >&2
        exit 1
    fi
    printf 'Proceed? [Y/n] ' >&2
    local answer=""
    read -r answer </dev/tty || answer=""
    case "$answer" in
        n|N|no|No|NO) echo "install.sh: aborted" >&2; exit 1 ;;
    esac
}
[ "$ASSUME_YES" -eq 1 ] || confirm_install

need_cmd() {
    command -v "$1" >/dev/null 2>&1 || {
        echo "Error: $1 is required but not on PATH." >&2
        exit 1
    }
}

need_cmd curl
need_cmd jq
need_cmd pgrep
if [ "$SERVER_MODE" = "rules" ]; then
    need_cmd python3
else
    need_cmd uv
fi

case "$SERVER_MODE" in
    redact|redact-torch|openai|rules|tagger) ;;
    *) echo "Error: PII_SERVER_MODE must be redact, redact-torch, openai, rules, or tagger." >&2; exit 1 ;;
esac

case "$ACTION_MODE" in
    block|warn) ;;
    *) echo "Error: PII_ACTION_MODE must be block or warn." >&2; exit 1 ;;
esac

# Decide which agents are present BEFORE anything is created. mkdir -p "$HOOKS_DIR"
# below would make $HOME/.claude exist, which would then satisfy the check that
# $HOME/.claude exists, so testing it afterwards installed into a machine with no
# agent and reported success. Checking first also means a refusal writes nothing.
CLAUDE_PRESENT=0
CODEX_PRESENT=0
[ -d "$HOME/.claude" ] && CLAUDE_PRESENT=1
[ -d "$HOME/.codex" ] && CODEX_PRESENT=1
[ "$SKIP_CODEX" -eq 1 ] && CODEX_PRESENT=0

if [ "$CLAUDE_PRESENT" -eq 0 ] && [ "$CODEX_PRESENT" -eq 0 ]; then
    echo "Neither ~/.claude/ nor ~/.codex/ found. Install at least one agent first." >&2
    exit 1
fi

mkdir -p "$PII_DIR/detectors" "$PII_DIR/rules"

echo "Downloading hook files..."
for file in "${SERVER_FILES[@]}"; do
    curl -fsSL "$REPO_BASE/server/$file" -o "$PII_DIR/$file"
    echo "Installed: $PII_DIR/$file"
done
curl -fsSL "$REPO_BASE/hook/check.sh" -o "$CHECK_DEST"
chmod +x "$CHECK_DEST"
echo "Installed: $CHECK_DEST"

# The release carries one native build per OS and CPU. Any other platform keeps
# the Python engine, which returns the same spans, only slower.
native_platform() {
    case "$(uname -s)-$(uname -m)" in
        Darwin-arm64) echo "darwin-arm64" ;;
        Linux-x86_64) echo "linux-x86_64" ;;
        *) return 1 ;;
    esac
}

# A failure here leaves no native file behind, so a stale build from an earlier
# install cannot stay in use. The download goes to a side file and then moves
# into place: a running server has the old file mapped, and writing over it in
# place would change the code under that server.
install_native() {
    local platform
    if ! platform=$(native_platform); then
        rm -f "$NATIVE_DEST"
        echo "No native rules engine is built for $(uname -s) $(uname -m). The rules run on the Python engine."
        return
    fi
    if curl -fsSL "$NATIVE_BASE/pii_rules_native-$platform.abi3.so" -o "$NATIVE_DEST.part"; then
        mv "$NATIVE_DEST.part" "$NATIVE_DEST"
        echo "Installed: $NATIVE_DEST"
    else
        rm -f "$NATIVE_DEST.part" "$NATIVE_DEST"
        echo "Could not download the native rules engine for $platform. The rules run on the Python engine." >&2
    fi
}
install_native

# The native hook command sends the request without starting a shell, and hands every
# other case to check.sh. Both agents run it. Codex binds hook trust to the command
# string, so a move between the script and the build stops its scan until someone
# trusts the new command; the closing notes say so.
#
# The file runs once before it is used: a hook command that cannot start fails open
# on every call, so a build for the wrong CPU must leave the script in place.
HOOK_COMMAND="$CHECK_DEST"
install_client() {
    local platform
    if ! platform=$(native_platform); then
        rm -f "$CLIENT_DEST"
        echo "No native hook command is built for $(uname -s) $(uname -m). The agents run check.sh."
        return
    fi
    if curl -fsSL "$NATIVE_BASE/pii-hook-$platform" -o "$CLIENT_DEST.part" &&
        chmod +x "$CLIENT_DEST.part" &&
        "$CLIENT_DEST.part" --mode prompt </dev/null >/dev/null 2>&1; then
        mv "$CLIENT_DEST.part" "$CLIENT_DEST"
        HOOK_COMMAND="$CLIENT_DEST"
        echo "Installed: $CLIENT_DEST"
    else
        rm -f "$CLIENT_DEST.part" "$CLIENT_DEST"
        echo "Could not install the native hook command for $platform. The agents run check.sh." >&2
    fi
}
install_client

# Add or update an entry in a hooks-shaped JSON file.
# Idempotent on command string: if an entry already references $cmd or one of $others,
# replace it with $entry (this is how matcher changes propagate to existing installs,
# and how an install moves between the script, the native command and an older
# install's paths); otherwise append.
# Args: $1 = target file, $2 = event key, $3 = entry JSON, $4 = command string to match,
# $5 = a JSON array of the same command through every other hook file.
add_entry() {
    local target="$1" event="$2" entry="$3" cmd="$4" others="$5"
    if [ ! -f "$target" ]; then
        mkdir -p "$(dirname "$target")"
        echo '{}' > "$target"
    fi
    local tmp
    tmp=$(mktemp)
    # The program is built in a plain assignment, and the call below is one line.
    # A command continued with backslashes AND carrying a multi-line quoted program
    # is attributed to different lines by different bash versions: 5.3 reports the
    # command's first line here and 3.2 reports the first argument line. No single
    # prediction can match both, so the construct is avoided rather than guessed at.
    local program
    # shellcheck disable=SC2016  # the single quotes are deliberate: this is jq source
    program='
        .hooks = (.hooks // {}) |
        .hooks[$event] = (
          (.hooks[$event] // []) as $entries |
          ($entries | map(if any(.hooks[]?; .command as $found | $found == $cmd or any($others[]; . == $found)) then $entry else . end)) as $mapped |
          if any($mapped[]?; any(.hooks[]?; .command == $cmd)) then $mapped
          else $mapped + [$entry] end
        )
        '
    jq --arg event "$event" --arg cmd "$cmd" --argjson others "$others" --argjson entry "$entry" "$program" "$target" > "$tmp"
    mv "$tmp" "$target"
}

# The command for one mode through every hook file but $1, as a JSON array.
other_commands() {
    local hook="$1" mode="$2" candidate
    local candidates=()
    for candidate in "$CLIENT_DEST" "$CHECK_DEST" "$LEGACY_CLIENT" "$LEGACY_CHECK"; do
        [ "$candidate" = "$hook" ] || candidates+=("$candidate --mode $mode")
    done
    jq -cn '$ARGS.positional' --args "${candidates[@]}"
}

wire_agent() {
    local label="$1" target="$2" posttool_mode="$3" pretool_mode="$4" hook="$5"
    local posttool_matcher="$POSTTOOL_MATCHER_CLAUDE"
    if [ "$label" = "codex" ]; then
        posttool_matcher="$POSTTOOL_MATCHER_CODEX"
    fi

    local prompt_cmd="$hook --mode prompt"
    local posttool_cmd="$hook --mode $posttool_mode"

    local prompt_entry
    prompt_entry=$(jq -cn --arg cmd "$prompt_cmd" \
        '{hooks:[{type:"command",command:$cmd,timeout:20}]}')
    add_entry "$target" "UserPromptSubmit" "$prompt_entry" "$prompt_cmd" "$(other_commands "$hook" prompt)"
    echo "  [$label] UserPromptSubmit -> $prompt_cmd"

    if [ "$PROMPT_ONLY" -eq 0 ]; then
        local posttool_entry
        posttool_entry=$(jq -cn --arg cmd "$posttool_cmd" --arg matcher "$posttool_matcher" \
            '{matcher:$matcher,hooks:[{type:"command",command:$cmd,timeout:20}]}')
        add_entry "$target" "PostToolUse" "$posttool_entry" "$posttool_cmd" "$(other_commands "$hook" "$posttool_mode")"
        echo "  [$label] PostToolUse ($posttool_matcher) -> $posttool_cmd"

        # PreToolUse is wired for both agents with the same flag as PostToolUse:
        # --prompt-only means the prompt hook alone.
        #
        # The timeout has to exceed the hook's own detector budget: a timed-out hook
        # fails open on PreToolUse, so the call would proceed unscanned and look checked.
        # 20 seconds against a 5 second POST budget.
        local pretool_cmd="$hook --mode $pretool_mode"
        local pretool_matcher="$PRETOOL_MATCHER_CLAUDE"
        if [ "$label" = "codex" ]; then
            pretool_matcher="$PRETOOL_MATCHER_CODEX"
        fi
        local pretool_entry
        pretool_entry=$(jq -cn --arg cmd "$pretool_cmd" --arg matcher "$pretool_matcher" \
            '{matcher:$matcher,hooks:[{type:"command",command:$cmd,timeout:20}]}')
        add_entry "$target" "PreToolUse" "$pretool_entry" "$pretool_cmd" "$(other_commands "$hook" "$pretool_mode")"
        echo "  [$label] PreToolUse ($pretool_matcher) -> $pretool_cmd"
    else
        echo "  [$label] PreToolUse and PostToolUse skipped (--prompt-only)"
    fi
}

# Start the server once so uv deps and model loading happen now, not in the
# user's first agent turn. Waits far longer than the hook's 10s, then smoke-tests
# one known-PII string. Leaves the
# server running — it's the same 127.0.0.1:$PORT singleton the hooks reuse.
# Sets PILOT_OK on success. Fail-soft: a miss here just means the first real
# prompt pays the cold start, same as before this step existed.
pilot_run() {
    local health="http://127.0.0.1:$PORT/health"
    if curl -sSf --max-time 1 "$health" >/dev/null 2>&1; then
        echo "  a server the installer cannot find still answers on 127.0.0.1:$PORT;" >&2
        echo "  stop it and rerun the installer" >&2
        return
    fi
    mkdir -p "$(dirname "$SERVER_LOG")"
    if [ "$SERVER_MODE" = "rules" ]; then
        # Rules mode needs no dependencies, so it runs on the system python3.
        echo "  starting the rules server on python3 (no dependencies to resolve)..."
        nohup python3 "$SERVER_DEST" --port "$PORT" --mode "$SERVER_MODE" >"$SERVER_LOG" 2>&1 </dev/null &
    else
        echo "  resolving deps + loading the $SERVER_MODE model (one-time)..."
        nohup uv run "$SERVER_DEST" --port "$PORT" --mode "$SERVER_MODE" >"$SERVER_LOG" 2>&1 </dev/null &
    fi
    disown
    for _ in $(seq 1 120); do   # up to ~60s for a cold download
        curl -sSf --max-time 1 "$health" | jq -e --arg mode "$SERVER_MODE" \
            '.status == "ok" and .mode == $mode' >/dev/null 2>&1 && break
        sleep 0.5
    done
    if ! curl -sSf --max-time 1 "$health" | jq -e --arg mode "$SERVER_MODE" \
        '.status == "ok" and .mode == $mode' >/dev/null 2>&1; then
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
        echo "  ok — $SERVER_MODE server warm, smoke test flagged ${spans} span(s)"
        PILOT_OK=1
    else
        echo "  server up but smoke test flagged nothing; check $SERVER_LOG" >&2
    fi
}

# A running server keeps the code it started with, so the new files take effect
# only once it stops. Match the script name and the port, not the full path: a server
# started by hand from the hooks directory shows only "pii/server.py", and one from an
# install before that folder shows "pii-server.py".
SERVER_PATTERN="pii([-_]|/)server\.py .*--port $PORT( |\$)"
stop_running_server() {
    local pids
    pids=$(pgrep -f "$SERVER_PATTERN" | tr '\n' ' ' || true)
    [ -n "$pids" ] || return 0
    echo "  stopping the running server on 127.0.0.1:$PORT (pid ${pids% })"
    # shellcheck disable=SC2086 # one pid per word
    kill $pids 2>/dev/null || true
    for _ in $(seq 1 20); do
        pgrep -f "$SERVER_PATTERN" >/dev/null || return 0
        sleep 0.5
    done
    pids=$(pgrep -f "$SERVER_PATTERN" | tr '\n' ' ' || true)
    echo "  server pid ${pids% } did not stop; stop it and rerun the installer" >&2
    return 1
}

echo
echo "Restarting the server..."
SERVER_STOPPED=1
stop_running_server || SERVER_STOPPED=0

PILOT_OK=0
if [ "$RUN_PILOT" -eq 1 ] && [ "$SERVER_STOPPED" -eq 1 ]; then
    echo
    echo "Pilot run (before wiring)..."
    pilot_run
fi

echo
echo "Wiring hooks..."
if [ "$CLAUDE_PRESENT" -eq 1 ]; then
    wire_agent "claude" "$CLAUDE_SETTINGS" "claude-posttool" "claude-pretool" "$HOOK_COMMAND"
fi
# Read before the wiring rewrites the file: Codex runs a command only once trusted.
CODEX_NEEDS_TRUST=0
if [ "$CODEX_PRESENT" -eq 1 ]; then
    grep -qF "\"$HOOK_COMMAND --mode prompt\"" "$CODEX_HOOKS" 2>/dev/null || CODEX_NEEDS_TRUST=1
    wire_agent "codex"  "$CODEX_HOOKS"     "codex-posttool" "codex-pretool" "$HOOK_COMMAND"
fi

# Only now: until the wiring above, a hook could still run an old file.
for file in "${LEGACY_FILES[@]}"; do
    [ -e "$HOOKS_DIR/$file" ] || continue
    rm -f "$HOOKS_DIR/$file"
    echo "Removed the old $HOOKS_DIR/$file"
done
# Python cached the old modules beside them. Other hooks may share the folder.
rm -f "$HOOKS_DIR"/__pycache__/pii_*.pyc
rmdir "$HOOKS_DIR/__pycache__" 2>/dev/null || true

echo
echo "Done. Restart any running agent for the changes to take effect."
if [ "$PILOT_OK" -eq 1 ]; then
    echo "Model is warm (pilot run) — the first agent prompt won't wait on the download."
else
    echo "First matching prompt may be slow. The server will resolve deps and load the selected model."
fi
if [ "$CODEX_NEEDS_TRUST" -eq 1 ]; then
    echo
    echo "WARNING: Codex has a new hook command, $HOOK_COMMAND."
    echo "Codex scans nothing until you trust it. Do the step below now."
fi
if [ "$CODEX_PRESENT" -eq 1 ]; then
    echo
    echo "Codex only: hooks require trust before they run. Launch codex, run /hooks,"
    echo "and trust the pii entries (trust is remembered in ~/.codex/config.toml"
    echo "under [hooks.state]). Trust binds to the command string, so a new entry needs"
    echo "trusting once. A content update does not, because the hook's path does not"
    echo "change. Claude Code needs no trust step."
fi
echo
echo "Tuning:"
echo "  PII_SERVER_MODE=redact        use the published Redact graph on CPU (default; downloads)"
echo "  PII_SERVER_MODE=redact-torch  use the Redact checkpoint instead; needs a compatible local cache"
echo "  PII_SERVER_MODE=openai        use the OpenAI Privacy Filter on CPU"
echo "  PII_SERVER_MODE=rules         use the deterministic rules only; no model, no accelerator"
echo "  PII_ACTION_MODE=warn    allow input and warn the agent about detected PII (default)"
echo "  PII_ACTION_MODE=block   reject detected PII"
echo "  PII_LEVEL=off       disable all checks"
echo "  PII_LEVEL=relaxed   act on secrets + account numbers only"
echo "  PII_LEVEL=standard  + emails, phones, addresses (default)"
echo "  PII_LEVEL=strict    + names, urls, dates"
echo "  (legacy PII_BLOCK_LEVEL is still read when PII_LEVEL is unset)"
echo "  PII_ALLOW_LABELS=private_url,private_date  allow specific labels within the selected tier"
echo "  REDACT_CHUNK_OVERLAP_TOKENS=128  overlap long Redact chunks by this many tokens"
echo "  REDACT_MAX_INPUT_TOKENS=32768    reject a Redact request above this token count"
echo "  Prefix a prompt with 'pii:off ' to bypass a single submission."
