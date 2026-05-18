#!/usr/bin/env bash
# PII scanner hook — works for Claude Code, Codex, and UserPromptSubmit.
# Auto-starts the local ONNX int8 server on first call, fail-open on any error.
#
# Usage: pii-check.sh --mode <mode>
#   prompt           UserPromptSubmit — blocks with decision:block
#   claude-posttool  Claude Code PostToolUse — blocks with decision:block
#   codex-posttool   Codex PostToolUse — blocks with continue:false
#   (default)        Auto-detect from stdin (prompt vs tool_output)
#
# PII_BLOCK_LEVEL (default: standard):
#   off      — disable all PII checks
#   relaxed  — block only critical (secrets, account numbers)
#   standard — block critical + moderate (emails, phones, addresses)
#   strict   — block all categories including low (names, URLs, dates)

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
SERVER_SCRIPT="${PII_SERVER_SCRIPT:-$HOME/.claude/hooks/pii-server.py}"
HEALTH="http://$HOST:$PORT/health"
PREDICT="http://$HOST:$PORT/"
LOCK="/tmp/pii-server.starting"
SERVER_LOG="${PII_SERVER_LOG:-$HOME/.cache/opf/server.log}"

# --- Category tiers ---
CRITICAL=('secret' 'account_number')
MODERATE=('private_email' 'private_phone' 'private_address')
LOW=('private_person' 'private_url' 'private_date')

BLOCK_LEVEL="${PII_BLOCK_LEVEL:-standard}"

mkdir -p "$(dirname "$SERVER_LOG")"

command -v jq >/dev/null 2>&1 || { echo "pii-check: jq not found, skipping" >&2; exit 0; }
command -v curl >/dev/null 2>&1 || { echo "pii-check: curl not found, skipping" >&2; exit 0; }

payload=$(cat)

# --- Extract text based on mode ---
extract_text() {
    case "$MODE" in
        prompt)
            printf '%s' "$payload" | jq -r '.prompt // empty'
            ;;
        claude-posttool)
            printf '%s' "$payload" | jq -r '.tool_response.stdout // empty'
            ;;
        codex-posttool)
            # Codex sends tool_response as a bare string for shell tools,
            # and as an object (e.g. with .stdout) for MCP tools. Handle both.
            printf '%s' "$payload" | jq -r '
                if (.tool_response | type) == "string" then .tool_response
                elif (.tool_response | type) == "object" then (.tool_response.stdout // .tool_response.text // empty)
                else empty end
            '
            ;;
        *)
            # Auto-detect: try fields in order
            printf '%s' "$payload" | jq -r '
                .prompt // .tool_response.stdout // .aggregated_output // .tool_output // empty
            '
            ;;
    esac
}

text=$(extract_text 2>/dev/null)
[ -z "$text" ] && exit 0

# --- Bypass ---
[ "$BLOCK_LEVEL" = "off" ] && exit 0

# pii:off prefix only for user prompts
if [[ "$MODE" == "prompt" || "$MODE" == "auto" ]] && [[ "$text" == "pii:off"* ]]; then
    exit 0
fi

health_ok() { curl -sSf --max-time 0.5 "$HEALTH" >/dev/null 2>&1; }

if ! health_ok; then
    if [ -d "$LOCK" ] && [ -n "$(find "$LOCK" -maxdepth 0 -mmin +1 2>/dev/null)" ]; then
        rmdir "$LOCK" 2>/dev/null
    fi

    if mkdir "$LOCK" 2>/dev/null; then
        command -v uv >/dev/null 2>&1 || { rmdir "$LOCK"; exit 0; }
        [ -f "$SERVER_SCRIPT" ] || { rmdir "$LOCK"; echo "pii-check: $SERVER_SCRIPT not found" >&2; exit 0; }
        nohup uv run "$SERVER_SCRIPT" --port "$PORT" >"$SERVER_LOG" 2>&1 </dev/null &
        disown
    fi

    for _ in $(seq 1 40); do
        health_ok && break
        sleep 0.25
    done
    rmdir "$LOCK" 2>/dev/null

    health_ok || exit 0
fi

response=$(curl -sS --max-time 5 -X POST "$PREDICT" \
    -H 'Content-Type: application/json' \
    -d "$(jq -cn --arg t "$text" '{text:$t}')" 2>/dev/null) || exit 0

count=$(printf '%s' "$response" | jq -r '.spans // [] | length' 2>/dev/null)
[ "${count:-0}" -eq 0 ] && exit 0

# --- Build blocked-labels array based on BLOCK_LEVEL ---
case "$BLOCK_LEVEL" in
    strict)   blocked_labels=("${CRITICAL[@]}" "${MODERATE[@]}" "${LOW[@]}") ;;
    standard) blocked_labels=("${CRITICAL[@]}" "${MODERATE[@]}") ;;
    relaxed)  blocked_labels=("${CRITICAL[@]}") ;;
    *)        blocked_labels=("${CRITICAL[@]}" "${MODERATE[@]}") ;;
esac

blocked_json=$(printf '%s\n' "${blocked_labels[@]}" | jq -R . | jq -s .)
processing_ms=$(printf '%s' "$response" | jq -r '.processing_ms // "?"')

# Split spans
blocked_spans=$(printf '%s' "$response" | jq -c --argjson labels "$blocked_json" \
    '[.spans[] | select(.label as $l | $labels | index($l))]')
warned_spans=$(printf '%s' "$response" | jq -c --argjson labels "$blocked_json" \
    '[.spans[] | select(.label as $l | $labels | index($l) | not)]')

# Print warnings for non-blocked spans
printf '%s' "$warned_spans" | jq -r '.[] | "PII warn: [\(.label)] \(.text)"' >&2 || true

blocked_count=$(printf '%s' "$blocked_spans" | jq -r 'length' 2>/dev/null)
[ "${blocked_count:-0}" -eq 0 ] && exit 0

# Print blocked spans
printf '%s' "$blocked_spans" | jq -r '.[] | "PII block: [\(.label)] \(.text)"' >&2

blocked_labels_str=$(printf '%s' "$blocked_spans" | jq -r '[.[].label] | unique | join(", ")')

# --- Output decision based on mode ---
case "$MODE" in
    prompt|claude-posttool)
        printf '%s' "$blocked_spans" | jq -c --arg ms "$processing_ms" --arg level "$BLOCK_LEVEL" --arg labels "$blocked_labels_str" '{
            decision: "block",
            reason: ("PII [level=\($level)]: " + ($ms + "ms") + " — " + $labels + ". Prefix with pii:off to bypass, or set PII_BLOCK_LEVEL=off.")
        }'
        ;;
    codex-posttool)
        printf '%s' "$blocked_spans" | jq -c --arg ms "$processing_ms" --arg level "$BLOCK_LEVEL" --arg labels "$blocked_labels_str" '{
            decision: "block",
            reason: ("PII in tool output [level=\($level)]: " + ($ms + "ms") + " — " + $labels + ". Set PII_BLOCK_LEVEL=off to disable.")
        }'
        ;;
    *)
        # Auto: check if prompt-like or tool-output-like
        if printf '%s' "$payload" | jq -e '.prompt' >/dev/null 2>&1; then
            printf '%s' "$blocked_spans" | jq -c --arg ms "$processing_ms" --arg level "$BLOCK_LEVEL" --arg labels "$blocked_labels_str" '{
                decision: "block",
                reason: ("PII [level=\($level)]: " + ($ms + "ms") + " — " + $labels + ". Prefix with pii:off to bypass, or set PII_BLOCK_LEVEL=off.")
            }'
        else
            # Tool output: block so model doesn't see content
            printf '%s' "$blocked_spans" | jq -c --arg ms "$processing_ms" --arg level "$BLOCK_LEVEL" --arg labels "$blocked_labels_str" '{
                decision: "block",
                reason: ("PII in tool output [level=\($level)]: " + ($ms + "ms") + " — " + $labels + ". Set PII_BLOCK_LEVEL=off.")
            }'
        fi
        ;;
esac
