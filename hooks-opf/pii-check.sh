#!/usr/bin/env bash
# PII scanner hook — works for Claude Code, Codex, and UserPromptSubmit.
# Auto-starts the local ONNX int8 server on first call, fail-open on any error.
#
# Usage: pii-check.sh --mode <mode>
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
SERVER_SCRIPT="${PII_SERVER_SCRIPT:-$HOME/.claude/hooks/pii-server.py}"
SERVER_MODE="${PII_SERVER_MODE:-redact}"
HEALTH="http://$HOST:$PORT/health"
PREDICT="http://$HOST:$PORT/"
LOCK="/tmp/pii-server.starting"
SERVER_LOG="${PII_SERVER_LOG:-$HOME/.cache/opf/server.log}"
ACTION_MODE="${PII_ACTION_MODE:-warn}"

case "$SERVER_MODE" in
    redact|openai|rules) ;;
    *) echo "pii-check: PII_SERVER_MODE must be redact, openai, or rules" >&2; exit 0 ;;
esac

case "$ACTION_MODE" in
    block|warn) ;;
    *) echo "pii-check: PII_ACTION_MODE must be block or warn" >&2; exit 0 ;;
esac

# --- Category tiers ---
CRITICAL=('secret' 'account_number')
MODERATE=('private_email' 'private_phone' 'private_address')
LOW=('private_person' 'private_url' 'private_date')

LEVEL="${PII_LEVEL:-${PII_BLOCK_LEVEL:-standard}}"
ALLOW_LABELS="${PII_ALLOW_LABELS:-}"

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
[ "$LEVEL" = "off" ] && exit 0

# pii:off prefix only for user prompts
if [[ "$MODE" == "prompt" || "$MODE" == "auto" ]] && [[ "$text" == "pii:off"* ]]; then
    exit 0
fi

# Detect the event contract before the server call so failures use the same output shape.
emit_mode="$MODE"
if [ "$emit_mode" = "auto" ]; then
    if printf '%s' "$payload" | jq -e '.prompt' >/dev/null 2>&1; then
        emit_mode="prompt"
    else
        emit_mode="claude-posttool"
    fi
fi

detector_failure() {
    local detail="$1"
    local detected_location="in the input"
    local allowed_subject="The input"
    local event_name="PostToolUse"
    case "$emit_mode" in
        prompt)
            detected_location="in the user prompt"
            allowed_subject="The user prompt"
            event_name="UserPromptSubmit"
            ;;
        claude-posttool|codex-posttool)
            detected_location="in tool output"
            allowed_subject="The tool output"
            ;;
    esac

    if [ "$ACTION_MODE" = "warn" ]; then
        local warning_message="PII detector unavailable while checking ${detected_location}. ${allowed_subject} was allowed because PII_ACTION_MODE=warn, but the detector did not complete. Treat the content as sensitive."
        local warning_context="PII detector unavailable while checking ${detected_location}. ${allowed_subject} was allowed because PII_ACTION_MODE=warn, but the detector did not complete. Do not repeat or expose unscanned values. ${detail}."
        jq -cn \
            --arg message "$warning_message" \
            --arg context "$warning_context" \
            --arg event "$event_name" \
            '{continue: true, systemMessage: $message, hookSpecificOutput: {hookEventName: $event, additionalContext: $context}}'
    else
        local reason="PII detector unavailable while checking ${detected_location}. Blocked because PII_ACTION_MODE=block. ${detail}."
        jq -cn --arg reason "$reason" '{decision: "block", reason: $reason}'
    fi
    exit 0
}

health_json() { curl -sSf --max-time 0.5 "$HEALTH" 2>/dev/null; }
health_ok() {
    health_json | jq -e --arg mode "$SERVER_MODE" \
        '.status == "ok" and .mode == $mode' >/dev/null 2>&1
}

if ! health_ok; then
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
fi

if ! response=$(curl -fsS --max-time 5 -X POST "$PREDICT" \
    -H 'Content-Type: application/json' \
    -d "$(jq -cn --arg t "$text" '{text:$t}')" 2>/dev/null); then
    detector_failure "detector request failed"
fi

if ! count=$(printf '%s' "$response" | jq -er '.spans | if type == "array" then length else error("spans is not an array") end' 2>/dev/null); then
    detector_failure "detector returned invalid JSON"
fi
[ "${count:-0}" -eq 0 ] && exit 0

# --- Build the selected-labels array from the level ---
case "$LEVEL" in
    strict)   selected_labels=("${CRITICAL[@]}" "${MODERATE[@]}" "${LOW[@]}") ;;
    standard) selected_labels=("${CRITICAL[@]}" "${MODERATE[@]}") ;;
    relaxed)  selected_labels=("${CRITICAL[@]}") ;;
    *)        selected_labels=("${CRITICAL[@]}" "${MODERATE[@]}") ;;
esac

selected_json=$(printf '%s\n' "${selected_labels[@]}" | jq -R . | jq -s .)
if [ -n "$ALLOW_LABELS" ]; then
    allow_json=$(printf '%s' "$ALLOW_LABELS" | jq -R 'split(",") | map(gsub("^\\s+|\\s+$"; "")) | map(select(length > 0))')
    selected_json=$(printf '%s' "$selected_json" | jq -c --argjson allow "$allow_json" 'map(select(. as $l | $allow | index($l) | not))')
fi
processing_ms=$(printf '%s' "$response" | jq -r '.processing_ms // "?"')

# Tier map derived from the CRITICAL/MODERATE/LOW arrays — single source of truth.
tier_map=$(jq -cn \
    --argjson crit "$(printf '%s\n' "${CRITICAL[@]}" | jq -R . | jq -s .)" \
    --argjson mod  "$(printf '%s\n' "${MODERATE[@]}" | jq -R . | jq -s .)" \
    --argjson low_ "$(printf '%s\n' "${LOW[@]}"      | jq -R . | jq -s .)" \
    '($crit | map({(.):"critical"}) | add) +
     ($mod  | map({(.):"moderate"}) | add) +
     ($low_ | map({(.):"low"})      | add)')

mask_jq='
    def mask_value:
        (.text // "" | tostring | gsub("[\r\n\t]+"; " ") | gsub(" +"; " ")) as $s |
        ($s | length) as $n |
        if $n == 0 then "[empty]"
        elif $n <= 6 then "[redacted]"
        elif $n <= 14 then ($s[0:2] + "..." + $s[-2:])
        else ($s[0:4] + "..." + $s[-4:])
        end;
'

# Split spans at the level: selected spans drive the response, the rest go to stderr.
selected_spans=$(printf '%s' "$response" | jq -c --argjson labels "$selected_json" --argjson tm "$tier_map" \
    "$mask_jq [.spans[] | select(.label as \$l | \$labels | index(\$l)) | . + {tier: (\$tm[.label] // \"unknown\"), masked: mask_value}]")
ignored_spans=$(printf '%s' "$response" | jq -c --argjson labels "$selected_json" --argjson tm "$tier_map" \
    "$mask_jq [.spans[] | select(.label as \$l | \$labels | index(\$l) | not) | . + {tier: (\$tm[.label] // \"unknown\"), masked: mask_value}]")

# Stderr only: spans the level does not select. The agent never sees these.
printf '%s' "$ignored_spans" | jq -r '.[] | "PII below level: [\(.label)(\(.tier))] \(.masked)"' >&2 || true

selected_count=$(printf '%s' "$selected_spans" | jq -r 'length' 2>/dev/null)
[ "${selected_count:-0}" -eq 0 ] && exit 0

# Tier-annotated, masked span list for the model-facing text, e.g. "secret(critical): sk_t...p7dc".
selected_spans_masked=$(printf '%s' "$selected_spans" | jq -r \
    '[.[] | "\(.label)(\(.tier)): \(.masked)"] | unique | join(", ")')

if [ "$ACTION_MODE" = "warn" ]; then
    case "$emit_mode" in
        prompt)
            event_name="UserPromptSubmit"
            detected_location="in the user prompt"
            allowed_subject="The user prompt"
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
    warning_context="PII detector warning: possible sensitive data was identified ${detected_location}: ${selected_spans_masked}. ${allowed_subject} was allowed because PII_ACTION_MODE=warn. Check whether each detection is valid. If the detection is valid, do not repeat or expose the value. Use a redacted form. Rotate or revoke a valid secret."
    warning_message="PII detector warning: possible sensitive data was identified ${detected_location}: ${selected_spans_masked}. ${allowed_subject} was allowed because PII_ACTION_MODE=warn."
    jq -cn \
        --arg message "$warning_message" \
        --arg context "$warning_context" \
        --arg event "$event_name" \
        '{continue: true, systemMessage: $message, hookSpecificOutput: {hookEventName: $event, additionalContext: $context}}'
    exit 0
fi

# Stderr: tier-annotated masked blocked spans + one-line summary with processing_ms.
printf '%s' "$selected_spans" | jq -r '.[] | "PII block: [\(.label)(\(.tier))] \(.masked)"' >&2
echo "pii-check: blocked $selected_count span(s) in ${processing_ms}ms at level=$LEVEL" >&2

# Highest tier that fired determines the remediation hint.
highest_tier=$(printf '%s' "$selected_spans" | jq -r '
    [.[].tier] |
    if any(. == "critical") then "critical"
    elif any(. == "moderate") then "moderate"
    elif any(. == "low") then "low"
    else "unknown" end')

case "$highest_tier" in
    critical) hint="Only PII_LEVEL=off would allow this." ;;
    moderate) hint="Drop to PII_LEVEL=relaxed to allow moderate categories (emails/phones/addresses)." ;;
    low)      hint="Drop to PII_LEVEL=standard to allow low categories (names/urls/dates)." ;;
    *)        hint="" ;;
esac

case "$emit_mode" in
    prompt)
        reason="PII in prompt: ${selected_spans_masked}. Blocked at PII_LEVEL=${LEVEL}. ${hint} One-shot bypass: prefix prompt with 'pii:off '."
        ;;
    claude-posttool|codex-posttool)
        reason="PII in tool output: ${selected_spans_masked}. Blocked at PII_LEVEL=${LEVEL}. ${hint} Do not retry the same command. Treat every value in that output as already exposed: do not repeat it, and do not write it to a file or a message."
        ;;
    *)
        reason="PII detected: ${selected_spans_masked}. Blocked at PII_LEVEL=${LEVEL}. ${hint}"
        ;;
esac

jq -cn --arg reason "$reason" '{decision: "block", reason: $reason}'
