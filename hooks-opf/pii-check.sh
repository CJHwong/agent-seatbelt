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
# Per user and per port. The old fixed /tmp/pii-server.starting was shared by every
# session and every user on the host, so a second one skipped its own start, waited
# out the full health poll, and then failed closed.
LOCK="${TMPDIR:-/tmp}/pii-server.$(id -u).${PORT}.starting"
SERVER_LOG="${PII_SERVER_LOG:-$HOME/.cache/opf/server.log}"
ACTION_MODE="${PII_ACTION_MODE:-warn}"
# Default 1 keeps the documented one-shot bypass working for existing users. A
# multi-user deployment sets 0, because anyone who can reach the agent can forge the
# prefix, and a prefix carried in-band cannot be authenticated.
ALLOW_BYPASS="${PII_ALLOW_BYPASS:-1}"

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
    redact|openai|rules) ;;
    *) scanner_skipped "PII_SERVER_MODE is '${SERVER_MODE}', which is not redact, openai, or rules" "Set it to one of those three" ;;
esac

case "$ACTION_MODE" in
    block|warn) ;;
    *) scanner_skipped "PII_ACTION_MODE is '${ACTION_MODE}', which is neither block nor warn" "Set it to block or warn, spelled exactly" ;;
esac

case "$ALLOW_BYPASS" in
    0|1) ;;
    *) scanner_skipped "PII_ALLOW_BYPASS is '${ALLOW_BYPASS}', which is neither 0 nor 1" "Set it to 0 or 1" ;;
esac

# --- Category tiers ---
CRITICAL=('secret' 'account_number')
MODERATE=('private_email' 'private_phone' 'private_address')
LOW=('private_person' 'private_url' 'private_date')

LEVEL="${PII_LEVEL:-${PII_BLOCK_LEVEL:-standard}}"
ALLOW_LABELS="${PII_ALLOW_LABELS:-}"

mkdir -p "$(dirname "$SERVER_LOG")"

payload=$(cat)

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
        claude-pretool)
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

if ! text=$(extract_text 2>/dev/null); then
    # The payload could not be parsed at all. Under set -e an unguarded substitution
    # here aborted the whole script with jq's exit status and no output, which a
    # runtime reads as "no decision" and therefore as a pass.
    scanner_skipped "the hook could not parse its own input, so it could not extract any text to scan" "Check that the runtime sends a JSON payload"
fi
[ -z "$text" ] && exit 0

# --- Bypass ---
[ "$LEVEL" = "off" ] && exit 0

# Detect the event contract before the server call so failures use the same output shape.
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

# The pii:off prefix applies to a user prompt and to nothing else. Routing it on the
# resolved event contract, rather than on a second guess about the payload shape, is
# what keeps it off tool output. Tool output is content an attacker controls, so a
# prefix there would be a bypass anyone could plant in a web page, a file, or an MCP
# result, and it would switch off the scan for that entire response.
if [ "$ALLOW_BYPASS" = "1" ] && [ "$emit_mode" = "prompt" ] && [[ "$text" == "pii:off"* ]]; then
    exit 0
fi

# The resolved event contract is the only source for the wording of a response.
event_subject() {
    case "$emit_mode" in
        prompt)
            event_name="UserPromptSubmit"
            detected_location="in the user prompt"
            allowed_subject="The user prompt"
            ;;
        claude-pretool)
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

detector_failure() {
    local detail="$1"
    event_subject

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

# A rejected input is not a broken detector. Both make curl exit non-zero, but the
# agent's next move differs: retrying is useless here, and the fix is on the server's
# limit rather than on the input.
oversize_failure() {
    local detail="$1"
    event_subject
    local advice="Raise the detector's input limit, or lower PII_LEVEL, if this content has to be checked."

    if [ "$ACTION_MODE" = "warn" ]; then
        local warning_message="The input was too large for the detector to scan while checking ${detected_location}, so ${allowed_subject} went unscanned. ${advice}"
        local warning_context="The input was too large for the detector to scan while checking ${detected_location}, so ${allowed_subject} went unscanned. Do not treat it as checked. ${detail}."
        jq -cn \
            --arg message "$warning_message" \
            --arg context "$warning_context" \
            --arg event "$event_name" \
            '{continue: true, systemMessage: $message, hookSpecificOutput: {hookEventName: $event, additionalContext: $context}}'
    else
        local reason="The input is too large for the detector to scan, so ${detected_location} can never be checked. This is not a retryable failure and the data is not a detector fault. ${advice} ${detail}."
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

# Build and send the request through pipes, never through argv. `jq --arg t "$text"`
# puts the whole payload into the argument list, and the kernel caps that, at 1 MB on
# macOS: a larger input died with "Argument list too long" before the detector saw it,
# and the hook reported that as a failed request rather than as a size problem.
# An empty body here would surface as HTTP 400 from the detector, which is honest: the
# request really was malformed. The `|| true` is what keeps set -e from aborting the
# script with no output at all.
request_body=$(printf '%s' "$text" | jq -Rs '{text:.}' 2>/dev/null) || true

# Keep the status code. curl -f collapses every non-2xx into one failure, which is why
# an input the detector rejected as oversized used to be reported as a dead detector.
response=$(printf '%s' "$request_body" | \
    curl -sS --max-time 5 -w $'\n%{http_code}' -X POST "$PREDICT" \
    -H 'Content-Type: application/json' --data-binary @- 2>/dev/null) || true
http_status="${response##*$'\n'}"
response="${response%$'\n'*}"

case "$http_status" in
    200) ;;
    413) oversize_failure "The detector answered HTTP 413 for this input" ;;
    *)   detector_failure "the detector request failed with HTTP status '${http_status:-none}'" ;;
esac

# Validate the shape, not only the container. A spans array of numbers passes a length
# check and then kills the script later, when the label is read off a number.
if ! count=$(printf '%s' "$response" | jq -er '
        .spans |
        if type != "array" then error("spans is not an array")
        elif any(.[]; type != "object") then error("a span is not an object")
        elif any(.[]; (.label | type) != "string") then error("a span has no string label")
        else length end' 2>/dev/null); then
    detector_failure "the detector response did not have the expected shape"
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

# shellcheck disable=SC2016  # the single quotes are deliberate: this is jq source
mask_jq='
    def mask_value:
        (.text // "" | tostring | gsub("[\r\n\t]+"; " ") | gsub(" +"; " ")) as $s |
        ($s | length) as $n |
        if $n < 12 then "[redacted]"
        else ($s[0:2] + "..." + $s[-2:])
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

# Tier-annotated, masked span list for the model-facing text, e.g. "secret(critical): sk...dc".
selected_spans_masked=$(printf '%s' "$selected_spans" | jq -r \
    '[.[] | "\(.label)(\(.tier)): \(.masked)"] | unique | join(", ")')

if [ "$ACTION_MODE" = "warn" ]; then
    # event_subject, not a second copy of it. The wording of an event lives in one
    # place, so a new contract cannot be added to one copy and missed in the other.
    event_subject
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

# Highest tier that fired determines the remediation hint. Every label the
# detector can emit is in the tier map, so the case covers every reachable value;
# hint is initialized rather than defaulted in a case arm, because set -u is on
# and every reason string below interpolates it.
highest_tier=$(printf '%s' "$selected_spans" | jq -r '
    [.[].tier] |
    if any(. == "critical") then "critical"
    elif any(. == "moderate") then "moderate"
    elif any(. == "low") then "low"
    else "unknown" end')

hint=""
case "$highest_tier" in
    critical) hint="Only PII_LEVEL=off would allow this." ;;
    moderate) hint="Drop to PII_LEVEL=relaxed to allow moderate categories (emails/phones/addresses)." ;;
    low)      hint="Drop to PII_LEVEL=standard to allow low categories (names/urls/dates)." ;;
esac

case "$emit_mode" in
    prompt)
        reason="PII in prompt: ${selected_spans_masked}. Blocked at PII_LEVEL=${LEVEL}. ${hint}"
        ;;
    claude-pretool)
        reason="PII in tool input: ${selected_spans_masked}. Blocked at PII_LEVEL=${LEVEL}. ${hint} The tool did not run, so the value has not left this machine. Do not send it another way."
        ;;
    claude-posttool|codex-posttool)
        reason="PII in tool output: ${selected_spans_masked}. Blocked at PII_LEVEL=${LEVEL}. ${hint} Do not retry the same command. Treat every value in that output as already exposed: do not repeat it, and do not write it to a file or a message."
        ;;
    *)
        reason="PII detected: ${selected_spans_masked}. Blocked at PII_LEVEL=${LEVEL}. ${hint}"
        ;;
esac

jq -cn --arg reason "$reason" '{decision: "block", reason: $reason}'
