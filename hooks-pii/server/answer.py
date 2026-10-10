"""The answer check.sh prints, computed in the server instead of in jq.

The hook script used to extract the text, call the detector, and build its output
with about fifteen jq processes per call. Each process costs about 3 ms, which made
the shell the slowest part of a hook. POST /hook gets the raw hook payload and the
caller's policy, and returns the exact stdout and stderr the script used to print.

Everything here copies the script's behaviour, including its order: an unparsable
payload, an empty text, PII_LEVEL=off and the pii:off prefix are all settled before
the detector runs. Failures come back as a status for the script to report, so the
failure messages and the skip log stay in one place, the script.

Standard library only: rules mode runs the server on the system python3.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

CRITICAL = ("secret", "account_number")
MODERATE = ("private_email", "private_phone", "private_address", "private_username")
LOW = ("private_person", "private_url", "private_date")
TIERS = {
    **{label: "critical" for label in CRITICAL},
    **{label: "moderate" for label in MODERATE},
    **{label: "low" for label in LOW},
}
LEVELS = {
    "strict": CRITICAL + MODERATE + LOW,
    "standard": CRITICAL + MODERATE,
    "relaxed": CRITICAL,
}
HINTS = {
    "critical": "Only PII_LEVEL=off would allow this.",
    "moderate": "Drop to PII_LEVEL=relaxed to allow moderate categories (emails/phones/addresses).",
    "low": "Drop to PII_LEVEL=standard to allow low categories (names/urls/dates).",
}
# Separates stdout from stderr in a /hook answer. Neither ever holds it: the script
# reads both from jq, which escapes every control character in JSON, and a masked
# value keeps at most four characters of a span.
SEPARATOR = "\x1e"


def code_version(script: Path) -> str:
    """A hash of the server code: every .py file in the script's folder and below.

    The files go in the byte order of their path relative to that folder. A running
    server keeps the code it started with. check.sh hashes the installed files the
    same way, and a different hash means the server started before those files
    changed.
    """
    folder = script.parent
    digest = hashlib.sha256()
    for module in sorted(
        folder.rglob("*.py"), key=lambda path: path.relative_to(folder).as_posix()
    ):
        digest.update(module.read_bytes())
    return digest.hexdigest()


class HookError(Exception):
    """A request the script must report itself, as an HTTP status."""

    def __init__(self, status: int, detail: str = "") -> None:
        super().__init__(detail)
        self.status, self.detail = status, detail


@dataclass(frozen=True)
class Policy:
    """The caller's settings, which can differ per agent while the server is shared."""

    mode: str
    level: str
    allow_labels: str
    action_mode: str
    allow_bypass: str
    server_mode: str

    @classmethod
    def from_headers(cls, headers: Any, serving_mode: str) -> Policy:
        """The policy check.sh sends in X-Pii-* headers. A missing one takes the default."""
        return cls(
            mode=headers.get("X-Pii-Mode", "auto"),
            level=headers.get("X-Pii-Level", "standard"),
            allow_labels=headers.get("X-Pii-Allow-Labels", ""),
            action_mode=headers.get("X-Pii-Action-Mode", "warn"),
            allow_bypass=headers.get("X-Pii-Allow-Bypass", "1"),
            server_mode=headers.get("X-Pii-Server-Mode", serving_mode),
        )


def answer(
    payload: bytes,
    policy: Policy,
    serving_mode: str,
    max_body: int,
    detect: Callable[[str], tuple[list[dict], int]],
) -> str:
    """The script's stdout and stderr for one payload, each closed by SEPARATOR."""
    stdout, stderr = outputs(payload, policy, serving_mode, max_body, detect)
    return stdout + SEPARATOR + stderr + SEPARATOR


def outputs(
    payload: bytes,
    policy: Policy,
    serving_mode: str,
    max_body: int,
    detect: Callable[[str], tuple[list[dict], int]],
) -> tuple[str, str]:
    document = parse(payload)
    text = hook_text(document, policy.mode)
    if not text or policy.level == "off":
        return "", ""
    emit_mode = event_mode(document, policy.mode)
    if (
        policy.allow_bypass == "1"
        and emit_mode == "prompt"
        and text.startswith("pii:off")
    ):
        return "", ""
    if serving_mode != policy.server_mode:
        raise HookError(409, serving_mode)
    if len(request_body(text)) > max_body:
        raise HookError(413)
    spans, processing_ms = detect(text)
    if not well_formed(spans):
        raise HookError(502)
    return render(spans, processing_ms, policy, emit_mode)


def well_formed(spans: Any) -> bool:
    """A list of objects with a string label, the shape the script used to insist on."""
    return isinstance(spans, list) and all(
        isinstance(span, dict) and isinstance(span.get("label"), str) for span in spans
    )


def parse(payload: bytes) -> Any:
    """The payload as jq reads it. Blank input is null, as jq then runs no filter."""
    decoded = payload.decode("utf-8", "replace")
    if not decoded.strip():
        return None
    try:
        return json.loads(decoded)
    except ValueError as error:
        raise HookError(422) from error


def hook_text(document: Any, mode: str) -> str:
    """The text the script extracted, after bash dropped NULs and trailing newlines."""
    if mode == "prompt":
        text = prompt_text(document)
    elif mode in ("claude-pretool", "codex-pretool"):
        text = leaf_strings(document, "tool_input")
    elif mode in ("claude-posttool", "codex-posttool"):
        text = leaf_strings(document, "tool_response")
    elif isinstance(document, dict) and "prompt" in document:
        text = prompt_text(document)
    elif isinstance(document, dict) and "tool_response" in document:
        text = leaf_strings(document, "tool_response")
    else:
        text = leaf_strings(document, "tool_input")
    return text.replace("\0", "").rstrip("\n")


def field(document: Any, name: str) -> Any:
    """jq's .[name]: null on null, an error on anything but an object."""
    if document is None:
        return None
    if not isinstance(document, dict):
        raise HookError(422)
    return document.get(name)


def prompt_text(document: Any) -> str:
    """jq -r '.prompt // empty'."""
    value = field(document, "prompt")
    if value is None or value is False:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, indent=2)
    return json.dumps(value)


def leaf_strings(document: Any, name: str) -> str:
    """Every string under one field, in jq's `..` order, one per line."""
    value = field(document, name)
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return "\n".join(strings_under(value))
    return ""


def strings_under(value: Any) -> list[str]:
    found: list[str] = []
    pending = [value]
    while pending:
        current = pending.pop()
        if isinstance(current, str):
            found.append(current)
        elif isinstance(current, dict):
            pending.extend(reversed(list(current.values())))
        elif isinstance(current, list):
            pending.extend(reversed(current))
    return found


def event_mode(document: Any, mode: str) -> str:
    """The event contract that words the answer. Auto mode reads it off the payload."""
    if mode != "auto":
        return mode
    if isinstance(document, dict) and "prompt" in document:
        return "prompt"
    if isinstance(document, dict) and "tool_response" in document:
        return "claude-posttool"
    return "claude-pretool"


def request_body(text: str) -> bytes:
    """The body the script sent to POST /, which the size limit was measured on."""
    return ('{\n  "text": ' + jq_json(text) + "\n}").encode("utf-8")


def jq_json(value: Any) -> str:
    """Compact JSON as jq writes it: UTF-8 as is, and DEL escaped like a control."""
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).replace(
        "\x7f", "\\u007f"
    )


def mask(span: dict) -> str:
    """The span as the script showed it: two characters each end, or [redacted]."""
    raw = span.get("text")
    shown = (
        ""
        if raw is None or raw is False
        else raw
        if isinstance(raw, str)
        else jq_json(raw)
    )
    shown = re.sub(" +", " ", re.sub("[\r\n\t]+", " ", shown))
    return "[redacted]" if len(shown) < 12 else shown[:2] + "..." + shown[-2:]


def selected_labels(level: str, allow_labels: str) -> tuple[str, ...]:
    labels = LEVELS.get(level, LEVELS["standard"])
    allowed = (
        {label.strip() for label in allow_labels.split(",")} if allow_labels else set()
    )
    return tuple(label for label in labels if label not in allowed)


def render(
    spans: list[dict], processing_ms: int, policy: Policy, emit_mode: str
) -> tuple[str, str]:
    """stdout and stderr for the detector's spans."""
    if not spans:
        return "", ""
    labels = selected_labels(policy.level, policy.allow_labels)
    marked = [
        (span["label"], TIERS.get(span["label"], "unknown"), mask(span))
        for span in spans
    ]
    selected = [item for item in marked if item[0] in labels]
    stderr = "".join(
        f"PII below level: [{label}({tier})] {masked}\n"
        for label, tier, masked in marked
        if label not in labels
    )
    if not selected:
        return "", stderr
    listed = ", ".join(
        sorted({f"{label}({tier}): {masked}" for label, tier, masked in selected})
    )
    event, location, subject = WORDING.get(emit_mode, WORDING[""])
    if policy.action_mode == "warn":
        found = f"PII detector warning: possible sensitive data was identified {location}: {listed}. {subject} was allowed because PII_ACTION_MODE=warn."
        context = f"{found} Check whether each detection is valid. If the detection is valid, do not repeat or expose the value. Use a redacted form. Rotate or revoke a valid secret."
        stdout = jq_json(
            {
                "continue": True,
                "systemMessage": found,
                "hookSpecificOutput": {
                    "hookEventName": event,
                    "additionalContext": context,
                },
            }
        )
        return stdout + "\n", stderr
    stderr += "".join(
        f"PII block: [{label}({tier})] {masked}\n" for label, tier, masked in selected
    )
    stderr += f"pii-check: blocked {len(selected)} span(s) in {processing_ms}ms at level={policy.level}\n"
    return block_response(
        block_reason(listed, selected, policy.level, emit_mode), emit_mode
    ), stderr


# event name, where the data was found, and what was let through, per event contract
WORDING = {
    "prompt": ("UserPromptSubmit", "in the user prompt", "The user prompt"),
    "claude-pretool": ("PreToolUse", "in the tool input", "The tool input"),
    "codex-pretool": ("PreToolUse", "in the tool input", "The tool input"),
    "claude-posttool": ("PostToolUse", "in tool output", "The tool output"),
    "codex-posttool": ("PostToolUse", "in tool output", "The tool output"),
    "": ("UserPromptSubmit", "in the input", "The input"),
}


def block_reason(listed: str, selected: list, level: str, emit_mode: str) -> str:
    tiers = {tier for _, tier, _ in selected}
    highest = next(
        (tier for tier in ("critical", "moderate", "low") if tier in tiers), ""
    )
    hint = HINTS.get(highest, "")
    if emit_mode == "prompt":
        return f"PII in prompt: {listed}. Blocked at PII_LEVEL={level}. {hint}"
    if emit_mode in ("claude-pretool", "codex-pretool"):
        return f"PII in tool input: {listed}. Blocked at PII_LEVEL={level}. {hint} The tool did not run, so the value has not left this machine. Do not send it another way."
    if emit_mode in ("claude-posttool", "codex-posttool"):
        return f"PII in tool output: {listed}. Blocked at PII_LEVEL={level}. {hint} Do not retry the same command. Treat every value in that output as already exposed: do not repeat it, and do not write it to a file or a message."
    return f"PII detected: {listed}. Blocked at PII_LEVEL={level}. {hint}"


def block_response(reason: str, emit_mode: str) -> str:
    if emit_mode in ("claude-pretool", "codex-pretool"):
        decision = {
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "deny",
                "permissionDecisionReason": reason,
            }
        }
    else:
        decision = {"decision": "block", "reason": reason}
    return jq_json(decision) + "\n"
