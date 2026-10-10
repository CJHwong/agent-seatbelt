#!/usr/bin/env bash
# PII scanner hook — works for Claude Code, Codex, and UserPromptSubmit.
# Auto-starts the local ONNX int8 server on first call, fail-open on any error.
#
# Usage: check.sh --mode <mode>
#   prompt           UserPromptSubmit (Claude Code + Codex)
#   claude-posttool  Claude Code PostToolUse
#   codex-posttool   Codex PostToolUse
#   (default)        Auto-detect from stdin (prompt vs tool_output)
#
# Claude Code and Codex share the prompt contract: input carries .prompt, and
# {decision:"block",reason} blocks. The post-tool modes share one detector path.
# Verified with codex-cli 0.154.0 in a fresh TUI session after hook trust.
#
# PII_ACTION_MODE (default: warn):
#   warn     allow input and add a masked warning to agent context
#   block    reject input when a span matches PII_LEVEL
# PII_LEVEL (default: standard, legacy name PII_BLOCK_LEVEL still accepted):
#   off      — disable all PII checks
#   relaxed  — select only critical (secrets, account numbers)
#   standard — select critical + moderate (emails, phones, addresses)
#   strict   — select all categories including low (names, URLs, dates)
# The level selects which labels the hook acts on. Block mode rejects a selected
# span; warn mode reports one. A span below the level goes to stderr only.
# PII_ALLOW_LABELS (default: empty):
#   comma-separated labels to allow even when included by PII_LEVEL

set -euo pipefail
export PATH="$HOME/.local/bin:/opt/homebrew/bin:/usr/local/bin:$PATH"

MODE="auto"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --mode) MODE="$2"; shift 2 ;;
        --mode=*) MODE="${1#*=}"; shift ;;
        *) shift ;;
    esac
done

PORT="${PII_PORT:-9123}"
HOST="127.0.0.1"
SERVER_SCRIPT="${PII_SERVER_SCRIPT:-$HOME/.claude/hooks/pii/server.py}"
SERVER_MODE="${PII_SERVER_MODE:-tagger}"
HEALTH="http://$HOST:$PORT/health"
HOOK="http://$HOST:$PORT/hook"
# Per user and per port. The old fixed /tmp/pii-server.starting was shared by every
# session and every user on the host, so a second one skipped its own start, waited
# out the full health poll, and then failed closed.
LOCK="${TMPDIR:-/tmp}/pii-server.${EUID}.${PORT}.starting"
SERVER_LOG="${PII_SERVER_LOG:-$HOME/.cache/pii/server.log}"
ACTION_MODE="${PII_ACTION_MODE:-warn}"
# Default 1 keeps the documented one-shot bypass working for existing users. A
# multi-user deployment sets 0, because anyone who can reach the agent can forge the
# prefix, and a prefix carried in-band cannot be authenticated.
ALLOW_BYPASS="${PII_ALLOW_BYPASS:-1}"

# Record a skipped scan where a daemon can see it, on the host, without the agent.
#
# The in-band warning reaches the agent, and the agent is the one party this hook
# cannot vouch for, so that warning is not a signal a deployment can rest on by
# itself. It also cannot be: the scanner is killable, so a skip is not always an
# operational accident. A host-side line needs no cooperation from anyone, and the
# daemon is the component that sees enough state to tell the two apart.
#
# One line per skipped request. Rotation belongs to whatever reads it.
SKIP_EVENT_PATH="${PII_SKIP_EVENT_PATH:-$HOME/.cache/pii/pii-skips.log}"
signal_skip() {
    mkdir -p "$(dirname "$SKIP_EVENT_PATH")" 2>/dev/null || true
    printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$1" >>"$SKIP_EVENT_PATH" 2>/dev/null || true
}

# Say that scanning was skipped, why, and how to restore it, instead of exiting
# silently. A silent exit downgrades the deployment to no scanning at all, and the
# transcript shows nothing.
#
# Every trigger here is a standing condition rather than a one-off: a missing tool or
# a bad variable stays that way, so the next request is unscanned too. The message
# says so, because "nothing in this request was checked" reads as a single miss. The
# agent gets the fact that whoever relies on this check does not know it is off, and
# decides for itself whether and how to pass that on.
#
# jq may itself be the thing that is missing, so that case writes the JSON literally.
scanner_skipped() {
    local detail="$1"
    local remedy="$2"
    signal_skip "$detail"
    local summary="PII scanner skipped: ${detail}. ${remedy}. Nothing was checked, and the scanner stays off until this is fixed."
    if command -v jq >/dev/null 2>&1; then
        jq -cn --arg message "$summary" \
            --arg context "${summary} Whatever this request carried reached the agent unscanned, so treat it as sensitive. Whoever relies on this check does not know it is off." \
            '{continue: true, systemMessage: $message, hookSpecificOutput: {hookEventName: "UserPromptSubmit", additionalContext: $context}}'
    else
        printf '{"continue":true,"systemMessage":"%s","hookSpecificOutput":{"hookEventName":"UserPromptSubmit","additionalContext":"%s Whatever this request carried reached the agent unscanned, so treat it as sensitive. Whoever relies on this check does not know it is off."}}\n' \
            "$summary" "$summary"
    fi
    exit 0
}

command -v jq >/dev/null 2>&1 || scanner_skipped "jq is not installed, so the scanner cannot run" "Install jq"
command -v curl >/dev/null 2>&1 || scanner_skipped "curl is not installed, so the scanner cannot reach the detector" "Install curl"

case "$SERVER_MODE" in
    redact|redact-torch|openai|rules|tagger) ;;
    *) scanner_skipped "PII_SERVER_MODE is '${SERVER_MODE}', which is not redact, redact-torch, openai, rules, or tagger" "Set it to one of those five" ;;
esac

case "$ACTION_MODE" in
    block|warn) ;;
    *) scanner_skipped "PII_ACTION_MODE is '${ACTION_MODE}', which is neither block nor warn" "Set it to block or warn, spelled exactly" ;;
esac

case "$ALLOW_BYPASS" in
    0|1) ;;
    *) scanner_skipped "PII_ALLOW_BYPASS is '${ALLOW_BYPASS}', which is neither 0 nor 1" "Set it to 0 or 1" ;;
esac

LEVEL="${PII_LEVEL:-${PII_BLOCK_LEVEL:-standard}}"
ALLOW_LABELS="${PII_ALLOW_LABELS:-}"

# Read stdin with a builtin, without starting cat, which costs about 2 ms on every
# tool call. Not $(</dev/stdin): Claude Code passes stdin as a socket, and on Linux
# /dev/stdin is a path that a socket cannot be opened through. read takes one byte
# per system call, so a 64 KB tool output costs about 12 ms more than cat; half the
# payloads are under 500 bytes, where read is the faster. A JSON payload holds no NUL.
IFS= read -r -d '' payload || true

# --- Extract text based on mode ---
# Tool inputs and tool responses vary in shape per tool: Bash uses .command, Read
# uses .file_path, WebFetch uses .url, and each tool reports its result differently.
# Rather than chase each shape, one recursion collects every leaf string under the
# field that matters — the NER labels patterns, so incidental strings (paths, type
# markers) are inert. The same recursion serves an input and a response.
leaf_strings() {
    printf '%s' "$payload" | jq -r --arg field "$1" '
        (.[$field] // empty) |
        if type == "string" then .
        elif type == "object" then [.. | strings] | join("\n")
        else empty end
    '
}

# The order in the auto branch matters. A PostToolUse payload carries BOTH
# .tool_input and .tool_response, so the response is tested first: reading the input
# of a call that already ran would scan the request and miss the result.
extract_text() {
    case "$MODE" in
        prompt)
            printf '%s' "$payload" | jq -r '.prompt // empty'
            ;;
        claude-pretool|codex-pretool)
            leaf_strings tool_input
            ;;
        claude-posttool|codex-posttool)
            leaf_strings tool_response
            ;;
        *)
            if printf '%s' "$payload" | jq -e 'has("prompt")' >/dev/null 2>&1; then
                printf '%s' "$payload" | jq -r '.prompt // empty'
            elif printf '%s' "$payload" | jq -e 'has("tool_response")' >/dev/null 2>&1; then
                leaf_strings tool_response
            else
                leaf_strings tool_input
            fi
            ;;
    esac
}

# The server settles the text, the level and the bypass on every request. These
# checks run here only when no server answers, so that an input with nothing to scan
# never starts one, exactly as before the server took them over.
local_checks() {
    if ! text=$(extract_text 2>/dev/null); then
        # The payload could not be parsed at all. Under set -e an unguarded substitution
        # here aborted the whole script with jq's exit status and no output, which a
        # runtime reads as "no decision" and therefore as a pass.
        scanner_skipped "the hook could not parse its own input, so it could not extract any text to scan" "Check that the runtime sends a JSON payload"
    fi
    [ -z "$text" ] && exit 0

    # --- Bypass ---
    [ "$LEVEL" = "off" ] && exit 0

    resolve_emit_mode

    # The pii:off prefix applies to a user prompt and to nothing else. Routing it on the
    # resolved event contract, rather than on a second guess about the payload shape, is
    # what keeps it off tool output. Tool output is content an attacker controls, so a
    # prefix there would be a bypass anyone could plant in a web page, a file, or an MCP
    # result, and it would switch off the scan for that entire response.
    if [ "$ALLOW_BYPASS" = "1" ] && [ "$emit_mode" = "prompt" ] && [[ "$text" == "pii:off"* ]]; then
        exit 0
    fi
}

# The event contract that words a failure. Only a failure needs it here, because the
# server words every answer it gives.
emit_mode=""
resolve_emit_mode() {
    emit_mode="$MODE"
    if [ "$emit_mode" = "auto" ]; then
        if printf '%s' "$payload" | jq -e 'has("prompt")' >/dev/null 2>&1; then
            emit_mode="prompt"
        elif printf '%s' "$payload" | jq -e 'has("tool_response")' >/dev/null 2>&1; then
            emit_mode="claude-posttool"
        else
            emit_mode="claude-pretool"
        fi
    fi
}

# The resolved event contract is the only source for the wording of a response.
event_subject() {
    [ -n "$emit_mode" ] || resolve_emit_mode
    case "$emit_mode" in
        prompt)
            event_name="UserPromptSubmit"
            detected_location="in the user prompt"
            allowed_subject="The user prompt"
            ;;
        claude-pretool|codex-pretool)
            event_name="PreToolUse"
            detected_location="in the tool input"
            allowed_subject="The tool input"
            ;;
        claude-posttool|codex-posttool)
            event_name="PostToolUse"
            detected_location="in tool output"
            allowed_subject="The tool output"
            ;;
        *)
            event_name="UserPromptSubmit"
            detected_location="in the input"
            allowed_subject="The input"
            ;;
    esac
}

# Emit a block in the shape the event expects. Claude Code documents
# hookSpecificOutput.permissionDecision for PreToolUse, and the top-level decision for
# the other events. On 2.1.276 all three deny forms block: both shapes and exit code 2.
# So this follows the documented contract rather than the one form that works today.
# Codex takes the deny shape too, so both pre-tool modes use it and the two runtimes
# cannot drift apart.
block_response() {
    local reason="$1"
    case "$emit_mode" in
        claude-pretool|codex-pretool)
        jq -cn --arg reason "$reason" \
            '{hookSpecificOutput: {hookEventName: "PreToolUse", permissionDecision: "deny", permissionDecisionReason: $reason}}'
            ;;
        *)
            jq -cn --arg reason "$reason" '{decision: "block", reason: $reason}'
            ;;
    esac
}

detector_failure() {
    local detail="$1"
    event_subject
    # A dead detector leaves the request unscanned exactly as a missing tool does, so
    # it earns the same host-side line. Without it this branch is invisible: the agent
    # holds the only copy of the cause, and nothing on the host can tell a recurring
    # failure from a one-off.
    signal_skip "$detail"

    if [ "$ACTION_MODE" = "warn" ]; then
        # The cause belongs in the message the human reads. All six failure branches
        # share this sentence, so without the detail a restart-the-server fault and a
        # malformed-response fault look identical.
        local warning_message="PII detector unavailable while checking ${detected_location}. ${allowed_subject} was allowed because PII_ACTION_MODE=warn, but the detector did not complete. Treat the content as sensitive. ${detail}."
        local warning_context="PII detector unavailable while checking ${detected_location}. ${allowed_subject} was allowed because PII_ACTION_MODE=warn, but the detector did not complete. Do not repeat or expose unscanned values. ${detail}."
        jq -cn \
            --arg message "$warning_message" \
            --arg context "$warning_context" \
            --arg event "$event_name" \
            '{continue: true, systemMessage: $message, hookSpecificOutput: {hookEventName: $event, additionalContext: $context}}'
    else
        local reason="PII detector unavailable while checking ${detected_location}. Blocked because PII_ACTION_MODE=block. ${detail}."
        block_response "$reason"
    fi
    exit 0
}

# A rejected input is not a broken detector. Both make curl exit non-zero, but the
# agent's next move differs: retrying is useless here, and the fix is on the server's
# limit rather than on the input.
oversize_failure() {
    local detail="$1"
    event_subject
    local advice="Raise the detector's input limit, or lower PII_LEVEL, if this content has to be checked."
    signal_skip "$detail"

    if [ "$ACTION_MODE" = "warn" ]; then
        local warning_message="The input was too large for the detector to scan while checking ${detected_location}, so ${allowed_subject} went unscanned. ${advice} ${detail}."
        local warning_context="The input was too large for the detector to scan while checking ${detected_location}, so ${allowed_subject} went unscanned. Do not treat it as checked. ${detail}."
        jq -cn \
            --arg message "$warning_message" \
            --arg context "$warning_context" \
            --arg event "$event_name" \
            '{continue: true, systemMessage: $message, hookSpecificOutput: {hookEventName: $event, additionalContext: $context}}'
    else
        local reason="The input is too large for the detector to scan, so ${detected_location} can never be checked. This is not a retryable failure and the data is not a detector fault. ${advice} ${detail}."
        block_response "$reason"
    fi
    exit 0
}

health_json() { curl -sSf --max-time 0.5 "$HEALTH" 2>/dev/null; }
health_ok() {
    health_json | jq -e --arg mode "$SERVER_MODE" \
        '.status == "ok" and .mode == $mode' >/dev/null 2>&1
}

# Start the server and wait for it. Runs only when nothing answered on the port.
start_server() {
    current_health=$(health_json || true)
    current_status=$(printf '%s' "$current_health" | jq -r '.status // empty' 2>/dev/null || true)
    current_mode=$(printf '%s' "$current_health" | jq -r '.mode // "unknown"' 2>/dev/null || true)
    if [ "$current_status" = "ok" ]; then
        detector_failure "server mode is $current_mode, requested $SERVER_MODE; restart the server"
    fi
    if [ -d "$LOCK" ] && [ -n "$(find "$LOCK" -maxdepth 0 -mmin +1 2>/dev/null)" ]; then
        rmdir "$LOCK" 2>/dev/null
    fi

    if mkdir "$LOCK" 2>/dev/null; then
        mkdir -p "$(dirname "$SERVER_LOG")"
        # Rules mode imports the standard library only, so it runs on the system
        # python3. uv would resolve the script's declared model dependencies and
        # install torch for a mode that never loads it.
        if [ "$SERVER_MODE" = "rules" ]; then
            command -v python3 >/dev/null 2>&1 || {
                rmdir "$LOCK" 2>/dev/null || true
                detector_failure "python3 is not available"
            }
            launcher=(python3)
        else
            command -v uv >/dev/null 2>&1 || {
                rmdir "$LOCK" 2>/dev/null || true
                detector_failure "uv is not available"
            }
            launcher=(uv run)
        fi
        [ -f "$SERVER_SCRIPT" ] || {
            rmdir "$LOCK" 2>/dev/null || true
            detector_failure "$SERVER_SCRIPT was not found"
        }
        nohup "${launcher[@]}" "$SERVER_SCRIPT" --port "$PORT" --mode "$SERVER_MODE" >"$SERVER_LOG" 2>&1 </dev/null &
        disown
    fi

    for _ in $(seq 1 40); do
        health_ok && break
        sleep 0.25
    done
    rmdir "$LOCK" 2>/dev/null

    health_ok || detector_failure "server did not become healthy"
}

# The hash code_version in answer.py takes: every .py file in the server's folder and
# below, in the byte order of its relative path. LC_ALL=C gives sort that order; a
# UTF-8 locale can put redact_torch.py before redact.py.
installed_version() {
    [ -f "$SERVER_SCRIPT" ] || return 0
    (cd "${SERVER_SCRIPT%/*}" && find . -name '*.py' | LC_ALL=C sort | tr '\n' '\0' | xargs -0 cat) 2>/dev/null |
        shasum -a 256 2>/dev/null | cut -d' ' -f1 || true
}

# A server keeps the code it started with. When the hook files change under it, as a
# copy synced from another machine does, /health still answers ok while the old code
# fails each request. Stop such a server so start_server brings up the installed code.
# Returns 1 when the server is current, when no version can be compared, or when the
# process is not a pii server on this port, the installer's pattern: the new
# pii/server.py or the pii-server.py of an install before it. The status the caller
# holds then stands.
SERVER_PATTERN="pii([-_]|/)server\.py .*--port $PORT( |\$)"
stop_stale_server() {
    local current_health running expected pids
    current_health=$(health_json) || return 1
    running=$(jq -er 'select(.status == "ok") | .version // ""' <<<"$current_health" 2>/dev/null) || return 1
    expected=$(installed_version)
    [ -n "$expected" ] && [ "$running" != "$expected" ] || return 1
    pids=$(pgrep -f "$SERVER_PATTERN" | tr '\n' ' ' || true)
    [ -n "$pids" ] || return 1
    # shellcheck disable=SC2086 # one pid per word
    kill $pids 2>/dev/null || true
    for _ in $(seq 1 20); do
        health_json >/dev/null || return 0
        sleep 0.1
    done
    detector_failure "the server on $HOST:$PORT runs older code than $SERVER_SCRIPT and did not stop; restart the server"
}

# One request does the whole check. The server extracts the text, runs the detector,
# and returns the exact stdout and stderr this script used to build with jq, about
# fifteen processes per call. The policy goes in headers, because each agent's
# environment can differ while every agent shares one server. The payload goes through
# stdin, never argv: the kernel caps argv, at 1 MB on macOS.
request_timeout_seconds=5

# One request over bash's own socket. It skips the curl process, about 5 ms of a hook
# that runs on every tool call. It fails when nothing listens on the port, when this
# bash was built without /dev/tcp, or when no whole answer comes back in time to tell
# a timeout from a closed connection. curl then makes the request, and reports what it
# gets as it always did.
post_hook_tcp() {
    # ${#payload} must count bytes, because that is what Content-Length counts.
    local LC_ALL=C read_exit=0 written=0 started=$SECONDS
    exec 3<>"/dev/tcp/$HOST/$PORT" || return 1
    # The server answers 413 before it reads a body far above its cap, and closes.
    # The rest of the write then raises SIGPIPE, which would end the hook with no
    # output, so it is ignored for the write and the failure comes back as a status.
    trap '' PIPE
    printf 'POST /hook HTTP/1.0\r\nContent-Type: application/json\r\nX-Pii-Mode: %s\r\nX-Pii-Level: %s\r\nX-Pii-Allow-Labels: %s\r\nX-Pii-Action-Mode: %s\r\nX-Pii-Allow-Bypass: %s\r\nX-Pii-Server-Mode: %s\r\nContent-Length: %d\r\n\r\n%s' \
        "$MODE" "$LEVEL" "$ALLOW_LABELS" "$ACTION_MODE" "$ALLOW_BYPASS" "$SERVER_MODE" "${#payload}" "$payload" >&3 || written=$?
    trap - PIPE
    if [ "$written" -ne 0 ]; then
        exec 3<&-
        return 1
    fi
    # The server closes an HTTP/1.0 connection after its answer, so the read ends at
    # end of file. bash 3.2 leaves the variable unset when the read times out.
    response=""
    IFS= read -r -d '' -t "$request_timeout_seconds" response <&3 || read_exit=$?
    exec 3<&-
    curl_exit=0
    case "$response" in
        HTTP/*$'\r\n\r\n'*) ;;
        *)
            # No whole answer arrived. bash 4 and later end a timed-out read with a code
            # above 128. bash 3.2, the macOS /bin/bash, ends it with 1, as at end of
            # file, so the time spent waiting tells a timeout from a closed connection.
            # A timeout is reported here, as curl's 28: asking again would double the wait.
            if [ "$read_exit" -gt 128 ] || [ $((SECONDS - started)) -ge $((request_timeout_seconds - 1)) ]; then
                curl_exit=28
                return 0
            fi
            return 1
            ;;
    esac
    http_status="${response#* }"
    http_status="${http_status%% *}"
    response="${response#*$'\r\n\r\n'}"
}

post_hook_curl() {
    curl_exit=0
    response=$(curl -sS --max-time "$request_timeout_seconds" -w $'\n%{http_code}' -X POST "$HOOK" \
        -H 'Content-Type: application/json' \
        -H "X-Pii-Mode: $MODE" \
        -H "X-Pii-Level: $LEVEL" \
        -H "X-Pii-Allow-Labels: $ALLOW_LABELS" \
        -H "X-Pii-Action-Mode: $ACTION_MODE" \
        -H "X-Pii-Allow-Bypass: $ALLOW_BYPASS" \
        -H "X-Pii-Server-Mode: $SERVER_MODE" \
        --data-binary @- <<<"$payload" 2>/dev/null) || curl_exit=$?
    http_status="${response##*$'\n'}"
    response="${response%$'\n'*}"
}

post_hook() {
    post_hook_tcp 2>/dev/null || post_hook_curl
}

post_hook
# curl exit 7: nothing listens on the port. Settle what needs no server, start one,
# and ask again.
if [ "$curl_exit" -eq 7 ]; then
    local_checks
    start_server
    post_hook
fi

# curl exit 28 is its own timeout. No response arrived, so http_code is 000, and that
# read as a dead server. The server is up but slow: a large input, or other hooks
# queued on its single inference lock.
if [ "$curl_exit" -eq 28 ]; then
    detector_failure "the detector request timed out after ${request_timeout_seconds}s; the server is up but did not answer in time, usually because the input is large or other checks are queued ahead of it"
fi

# A status the server has no reason to send can come from a server that runs older
# code. Replace that server once and ask again.
if ! [[ "$http_status" =~ ^(200|409|413|422|502)$ ]] && stop_stale_server; then
    start_server
    post_hook
fi

# The answer is stdout, a record separator, stderr, and a closing separator. The
# closing one keeps the command substitution from trimming stderr's last newline.
case "$http_status" in
    200)
        answer="${response%$'\x1e'}"
        printf '%s' "${answer%%$'\x1e'*}"
        printf '%s' "${answer#*$'\x1e'}" >&2
        exit 0
        ;;
    409) detector_failure "server mode is $response, requested $SERVER_MODE; restart the server" ;;
    413) oversize_failure "The detector answered HTTP 413 for this input" ;;
    422) scanner_skipped "the hook could not parse its own input, so it could not extract any text to scan" "Check that the runtime sends a JSON payload" ;;
    502) detector_failure "the detector response did not have the expected shape" ;;
    *)   detector_failure "the detector request failed with HTTP status '${http_status:-none}'" ;;
esac
