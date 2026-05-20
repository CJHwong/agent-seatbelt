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
# Tool responses vary in shape per tool: Bash uses .stdout, Read uses .file.content,
# WebFetch/MCP tools use .text or nested fields. Rather than chase each shape, we
# recursively collect every leaf string under .tool_response — the NER labels
# patterns, so incidental strings (paths, type markers) are inert.
extract_text() {
    case "$MODE" in
        prompt)
            printf '%s' "$payload" | jq -r '.prompt // empty'
            ;;
        claude-posttool|codex-posttool)
            printf '%s' "$payload" | jq -r '
                (.tool_response // empty) |
                if type == "string" then .
                elif type == "object" then [.. | strings] | join("\n")
                else empty end
            '
            ;;
        *)
            # Auto-detect: prompt field first, then tool_response in either shape.
            printf '%s' "$payload" | jq -r '
                if has("prompt") then (.prompt // empty)
                else
                    (.tool_response // empty) |
                    if type == "string" then .
                    elif type == "object" then [.. | strings] | join("\n")
                    else empty end
                end
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

# Tier map derived from the CRITICAL/MODERATE/LOW arrays — single source of truth.
tier_map=$(jq -cn \
    --argjson crit "$(printf '%s\n' "${CRITICAL[@]}" | jq -R . | jq -s .)" \
    --argjson mod  "$(printf '%s\n' "${MODERATE[@]}" | jq -R . | jq -s .)" \
    --argjson low_ "$(printf '%s\n' "${LOW[@]}"      | jq -R . | jq -s .)" \
    '($crit | map({(.):"critical"}) | add) +
     ($mod  | map({(.):"moderate"}) | add) +
     ($low_ | map({(.):"low"})      | add)')

# Split spans into blocked vs warned, then tier-annotate both.
blocked_spans=$(printf '%s' "$response" | jq -c --argjson labels "$blocked_json" --argjson tm "$tier_map" \
    '[.spans[] | select(.label as $l | $labels | index($l)) | . + {tier: ($tm[.label] // "unknown")}]')
warned_spans=$(printf '%s' "$response" | jq -c --argjson labels "$blocked_json" --argjson tm "$tier_map" \
    '[.spans[] | select(.label as $l | $labels | index($l) | not) | . + {tier: ($tm[.label] // "unknown")}]')

# Stderr: tier-annotated warnings for non-blocked spans.
printf '%s' "$warned_spans" | jq -r '.[] | "PII warn: [\(.label)(\(.tier))] \(.text)"' >&2 || true

blocked_count=$(printf '%s' "$blocked_spans" | jq -r 'length' 2>/dev/null)
[ "${blocked_count:-0}" -eq 0 ] && exit 0

# Stderr: tier-annotated blocked spans + one-line summary with processing_ms.
printf '%s' "$blocked_spans" | jq -r '.[] | "PII block: [\(.label)(\(.tier))] \(.text)"' >&2
echo "pii-check: blocked $blocked_count span(s) in ${processing_ms}ms at level=$BLOCK_LEVEL" >&2

# Tier-annotated label list for the model-facing reason, e.g. "secret(critical), private_email(moderate)".
blocked_labels_annotated=$(printf '%s' "$blocked_spans" | jq -r \
    '[.[] | "\(.label)(\(.tier))"] | unique | join(", ")')

# Highest tier that fired determines the remediation hint.
highest_tier=$(printf '%s' "$blocked_spans" | jq -r '
    [.[].tier] |
    if any(. == "critical") then "critical"
    elif any(. == "moderate") then "moderate"
    elif any(. == "low") then "low"
    else "unknown" end')

case "$highest_tier" in
    critical) hint="Only PII_BLOCK_LEVEL=off would allow this." ;;
    moderate) hint="Drop to PII_BLOCK_LEVEL=relaxed to allow moderate categories (emails/phones/addresses)." ;;
    low)      hint="Drop to PII_BLOCK_LEVEL=standard to allow low categories (names/urls/dates)." ;;
    *)        hint="" ;;
esac

# Detect prompt-vs-tool-output for auto mode.
emit_mode="$MODE"
if [ "$emit_mode" = "auto" ]; then
    if printf '%s' "$payload" | jq -e '.prompt' >/dev/null 2>&1; then
        emit_mode="prompt"
    else
        emit_mode="claude-posttool"
    fi
fi

case "$emit_mode" in
    prompt)
        reason="PII in prompt: ${blocked_labels_annotated}. Blocked at PII_BLOCK_LEVEL=${BLOCK_LEVEL}. ${hint} One-shot bypass: prefix prompt with 'pii:off '."
        ;;
    claude-posttool|codex-posttool)
        reason="PII in tool output: ${blocked_labels_annotated}. Blocked at PII_BLOCK_LEVEL=${BLOCK_LEVEL}. ${hint} Do not retry the same command. The model has not seen the output."
        ;;
    *)
        reason="PII detected: ${blocked_labels_annotated}. Blocked at PII_BLOCK_LEVEL=${BLOCK_LEVEL}. ${hint}"
        ;;
esac

jq -cn --arg reason "$reason" '{decision: "block", reason: $reason}'
