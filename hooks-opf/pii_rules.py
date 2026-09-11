"""Deterministic PII rules.

Standard library only. The Redact model, the OpenAI Privacy Filter, and the
hook all consume these spans. Nothing here loads a checkpoint.
"""

from __future__ import annotations

import re
from typing import cast


SPAN_PRIORITY = {
    "secret": 6,
    "account_number": 5,
    "private_email": 4,
    "private_phone": 4,
    "private_address": 3,
    "private_person": 2,
    "private_url": 2,
    "private_date": 1,
}

EMAIL_PATTERN = re.compile(
    r"(?<![\w.+-])[A-Za-z0-9][A-Za-z0-9._%+-]*"
    r"@[A-Za-z0-9](?:[A-Za-z0-9.-]*[A-Za-z0-9])?\.[A-Za-z]{2,}(?![\w.-])"
)
URL_PATTERN = re.compile(r"https?://[^\s<>\"']+", re.IGNORECASE)
PHONE_PATTERN = re.compile(
    r"(?<!\w)(?:\+\d{1,3}[\s.-]?)?"
    r"(?:\(\d{2,4}\)|\d{2,4})[\s.-]\d{3}[\s.-]\d{3,4}(?!\w)"
)
CARD_PATTERN = re.compile(r"(?<!\d)(?:\d[ -]?){13,19}(?!\d)")
BANK_ACCOUNT_PATTERN = re.compile(
    r"(?i)\b(?:account|acct)(?:\s+(?:number|no\.?))?"
    r"\s*(?:is|=|:)?\s*(?P<value>\d{8,})\b"
)
IBAN_PATTERN = re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b")
IP_PATTERN = re.compile(r"(?<![\w.])(?:\d{1,3}\.){3}\d{1,3}(?![\w.])")
AWS_ACCESS_KEY_PATTERN = re.compile(r"\bAKIA[0-9A-Z]{16}\b")
SECRET_PREFIX_PATTERN = re.compile(
    r"\b(?:sk_(?:live|test)_[A-Za-z0-9]+|gh[pousr]_[A-Za-z0-9]+|"
    r"xox[baprs]-[A-Za-z0-9-]+|AIza[0-9A-Za-z_-]{20,})\b"
)
PROVIDER_SECRET_PATTERN = re.compile(
    r"(?<![A-Za-z0-9])(?:"
    r"ocid1\.securitytoken\.oc1\.\.[A-Za-z0-9._-]{20,}|"
    r"ASIA[0-9A-Z]{16}|GOCSPX-[A-Za-z0-9_-]{20,}|"
    r"(?:cfp|dop_v1|hrk|vercel|nfp|sbp)_[A-Za-z0-9]{20,}|"
    r"sk-(?:proj-|ant-(?:api\d+-)?|)[A-Za-z0-9_-]{20,}|"
    r"rk_(?:live|test)_[A-Za-z0-9]{20,}|"
    r"SG\.[A-Za-z0-9_-]{20,}\.[A-Za-z0-9_-]{8,}|"
    r"whsec_[A-Za-z0-9_-]{20,}|glc_[A-Za-z0-9_-]{20,}|"
    r"snyk_[A-Za-z0-9_-]{20,}|pul-[A-Za-z0-9_-]{20,}|"
    r"sk\.[A-Za-z0-9]{20,}|"
    r"dd_api_[A-Za-z0-9]{20,}|NRAK-[A-Za-z0-9]{20,}|"
    r"github_pat_[A-Za-z0-9_]{20,}|"
    r"glpat-[A-Za-z0-9_-]{20,}|bb_app_[A-Za-z0-9_-]{20,}|"
    r"npm_[A-Za-z0-9_-]{20,}|pypi-[A-Za-z0-9_-]{20,}|"
    r"dckr_pat_[A-Za-z0-9_-]{20,}|cci_[A-Za-z0-9_-]{20,}|"
    r"bk_[A-Za-z0-9_-]{20,}|travis_[A-Za-z0-9_-]{20,}|"
    r"codecov_[A-Za-z0-9_-]{20,}|dp\.st\.[A-Za-z0-9_-]{20,}|"
    r"hvs\.[A-Za-z0-9_-]{20,}|"
    r"lin_api_[A-Za-z0-9_-]{20,}|sntrys_[A-Za-z0-9_-]{20,}|"
    r"webex_[A-Za-z0-9_-]{20,}|intercom_[A-Za-z0-9_-]{20,}|"
    r"\d{8,12}:AA[A-Za-z0-9_-]{20,}"
    r")(?![A-Za-z0-9])"
)
JWT_PATTERN = re.compile(r"\beyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\b")
PRIVATE_KEY_PATTERN = re.compile(r"-----BEGIN (?:[A-Z0-9]+ )*PRIVATE KEY-----")
SECRET_CONTEXT_PATTERN = re.compile(
    r"(?i)\b(?:aws_secret_access_key|secret(?:\s+key|_access_key)?)"
    r"(?![a-z0-9_])"
    r"\s*(?:(?:is|=|:)\s*)?(?P<value>[A-Za-z0-9/+=!@#$%^&*_-]{16,})"
)
PASSWORD_CONTEXT_PATTERN = re.compile(
    r"(?i)\b(?:password|passphrase)\s*(?:is|=|:)\s*"
    r"(?P<value>[^\s,.;]{4,})"
)
SECRET_CONTEXT_KEYS = (
    r"(?:api(?:[_-]?key|\s+key)|api[_-]?token|client[_-]?secret|"
    r"access[_-]?token|refresh[_-]?token|session[_-]?token|"
    r"private[_-]?(?:key|token)|auth[_-]?token|"
    r"database[_-]?password|db[_-]?password|docker[_-]?password|"
    r"aws[_-]?session[_-]?token|credential[_-]?value|password|passphrase|"
    r"secret(?:[_-]?(?:key|token|value|access[_-]?key))?|token|"
    r"x-api-key|_authToken|deploy[_-]?(?:secret|token)|"
    r"service[_-]?(?:secret|token))"
)
GENERIC_SECRET_CONTEXT_PATTERN = re.compile(
    r"(?ix)(?<![a-z0-9])"
    + SECRET_CONTEXT_KEYS
    + r"(?![a-z0-9_])\s*[\"']?\s*(?:is|=|:)\s*"
    r"(?P<quote>[\"'`]?)(?P<value>"
    r"[a-z0-9][a-z0-9._~+/=:@$!%*&?{}-]{7,}"
    r")(?(quote)(?P=quote))"
)
BEARER_SECRET_PATTERN = re.compile(r"(?i)\bBearer\s+(?P<value>[A-Za-z0-9._~+/=-]{20,})")
BASIC_SECRET_PATTERN = re.compile(r"(?i)\bBasic\s+(?P<value>[A-Za-z0-9+/]{16,}={0,2})")
DATABASE_URL_SECRET_PATTERN = re.compile(
    r"(?i)\b(?:postgres(?:ql)?|mysql|mongodb(?:\+srv)?|redis)s?://"
    r"(?:[^/\s:@]+:|:)(?P<value>[^@\s/?#]+)@"
)
MAILGUN_SECRET_PATTERN = re.compile(
    r"(?i)\bmailgun(?:\s+api)?\s+(?:key|token)\s+"
    r"(?P<value>key-[A-Za-z0-9]{20,})"
)
NOTION_SECRET_PATTERN = re.compile(
    r"(?i)\bnotion(?:\s+integration)?\s+"
    r"(?P<value>secret_[A-Za-z0-9_-]{20,})"
)
WEBHOOK_SECRET_PATTERN = re.compile(
    r"https://(?:hooks\.slack\.com/services/"
    r"[A-Za-z0-9]+/[A-Za-z0-9]+/[A-Za-z0-9]+|"
    r"discord(?:app)?\.com/api/webhooks/[A-Za-z0-9]+/[A-Za-z0-9_-]+)",
    re.IGNORECASE,
)
XML_SECRET_CONTEXT_PATTERN = re.compile(
    r"(?is)<(?:password|passphrase|secret|token|api[_-]?key)\b[^>]*>\s*"
    r"(?P<value>[A-Za-z0-9][A-Za-z0-9._~+/=:@$!%*&?{}-]{7,})\s*</"
)
JWK_PRIVATE_VALUE_PATTERN = re.compile(
    r"(?is)(?=[^{}\n]{0,200}[\"']kty[\"']\s*:)"
    r"[^{}\n]{0,200}[\"']d[\"']\s*:\s*[\"']"
    r"(?P<value>[A-Za-z0-9_-]{16,})[\"']"
)
TERRAFORM_SECRET_VALUE_PATTERN = re.compile(
    r"(?is)\bresource\s+[\"']secret[\"']\s+[\"'][^\"']+[\"']\s*"
    r"\{[^{}]{0,1000}?\bvalue\s*=\s*[\"']"
    r"(?P<value>[A-Za-z0-9][A-Za-z0-9._~+/=:@$!%*&?{}-]{7,})[\"']"
)
COOKIE_SECRET_PATTERN = re.compile(
    r"(?i)\bSet-Cookie:\s*(?:session|sid|auth|access|refresh)[^=;\s]*="
    r"(?P<value>[A-Za-z0-9._~+/=-]{16,})"
)
SIGNED_URL_SECRET_PATTERN = re.compile(
    r"(?i)\bhttps?://[^\s?]+(?:\?[^\s#]*)?[?&]sig="
    r"(?P<value>[A-Za-z0-9%._~-]{16,})"
)
STRUCTURED_SECRET_VALUE_PATTERN = re.compile(
    r"(?is)\bsecret\s*=\s*\{[^{}]{0,200}?\bvalue\s*=\s*[\"']"
    r"(?P<value>[A-Za-z0-9][A-Za-z0-9._~+/=:@$!%*&?{}-]{7,})[\"']"
)
CJK_SECRET_CONTEXT_PATTERN = re.compile(
    r"(?:密碼|密鑰|金鑰|令牌)\s*(?:是|為|：|:|=)\s*"
    r"(?P<value>[^\s,，。；;\"'<>]{8,})"
)
ISO_DATE_PATTERN = re.compile(r"\b(?:19|20)\d{2}[-/]\d{1,2}[-/]\d{1,2}\b")
MONTH_DATE_PATTERN = re.compile(
    r"(?i)\b(?:january|february|march|april|may|june|july|august|"
    r"september|october|november|december)\s+\d{1,2}(?:st|and|rd|th)?"
    r"(?:,\s*|\s+)(?:19|20)\d{2}\b"
)


def deterministic_spans(text: str) -> list[dict[str, object]]:
    spans: list[dict[str, object]] = []
    _append_matches(text, EMAIL_PATTERN, "private_email", spans)
    _append_matches(text, URL_PATTERN, "private_url", spans, trim_url=True)
    _append_matches(text, PHONE_PATTERN, "private_phone", spans)
    _append_card_matches(text, spans)
    _append_matches(
        text, BANK_ACCOUNT_PATTERN, "account_number", spans, group_name="value"
    )
    _append_matches(text, IBAN_PATTERN, "account_number", spans)
    _append_ip_matches(text, spans)

    _append_secret_matches(text, AWS_ACCESS_KEY_PATTERN, spans)
    _append_secret_matches(text, SECRET_PREFIX_PATTERN, spans)
    _append_secret_matches(text, PROVIDER_SECRET_PATTERN, spans)
    _append_secret_matches(text, JWT_PATTERN, spans)
    _append_secret_matches(text, PRIVATE_KEY_PATTERN, spans)
    _append_secret_matches(
        text,
        PASSWORD_CONTEXT_PATTERN,
        spans,
        group_name="value",
    )
    _append_secret_matches(
        text,
        SECRET_CONTEXT_PATTERN,
        spans,
        group_name="value",
    )
    _append_secret_matches(
        text,
        GENERIC_SECRET_CONTEXT_PATTERN,
        spans,
        group_name="value",
    )
    _append_secret_matches(
        text,
        BEARER_SECRET_PATTERN,
        spans,
        group_name="value",
    )
    _append_secret_matches(
        text,
        BASIC_SECRET_PATTERN,
        spans,
        group_name="value",
    )
    _append_secret_matches(
        text,
        DATABASE_URL_SECRET_PATTERN,
        spans,
        group_name="value",
    )
    _append_secret_matches(
        text,
        MAILGUN_SECRET_PATTERN,
        spans,
        group_name="value",
    )
    _append_secret_matches(
        text,
        NOTION_SECRET_PATTERN,
        spans,
        group_name="value",
    )
    _append_secret_matches(text, WEBHOOK_SECRET_PATTERN, spans)
    _append_secret_matches(
        text,
        SIGNED_URL_SECRET_PATTERN,
        spans,
        group_name="value",
    )
    _append_secret_matches(
        text,
        XML_SECRET_CONTEXT_PATTERN,
        spans,
        group_name="value",
    )
    _append_secret_matches(
        text,
        JWK_PRIVATE_VALUE_PATTERN,
        spans,
        group_name="value",
    )
    _append_secret_matches(
        text,
        TERRAFORM_SECRET_VALUE_PATTERN,
        spans,
        group_name="value",
    )
    _append_secret_matches(
        text,
        COOKIE_SECRET_PATTERN,
        spans,
        group_name="value",
    )
    _append_secret_matches(
        text,
        STRUCTURED_SECRET_VALUE_PATTERN,
        spans,
        group_name="value",
    )
    _append_secret_matches(
        text,
        CJK_SECRET_CONTEXT_PATTERN,
        spans,
        group_name="value",
    )
    _append_csv_secret_matches(text, spans)

    _append_matches(text, ISO_DATE_PATTERN, "private_date", spans)
    _append_matches(text, MONTH_DATE_PATTERN, "private_date", spans)
    return merge_spans(text, spans)


def _append_matches(
    text: str,
    pattern: re.Pattern[str],
    label: str,
    spans: list[dict[str, object]],
    *,
    group_name: str | None = None,
    trim_url: bool = False,
) -> None:
    for match in pattern.finditer(text):
        start, end = match.span(group_name) if group_name else match.span()
        if trim_url:
            end = _trim_url_end(text, start, end)
        if start < end:
            spans.append({"start": start, "end": end, "label": label})


def _append_secret_matches(
    text: str,
    pattern: re.Pattern[str],
    spans: list[dict[str, object]],
    *,
    group_name: str | None = None,
) -> None:
    for match in pattern.finditer(text):
        start, end = match.span(group_name) if group_name else match.span()
        end = _trim_secret_end(text, start, end)
        if start >= end or _is_secret_placeholder(text[start:end]):
            continue
        spans.append({"start": start, "end": end, "label": "secret"})


def _trim_secret_end(text: str, start: int, end: int) -> int:
    while end > start and text[end - 1] in ".,;:":
        end -= 1
    return end


def _is_secret_placeholder(value: str) -> bool:
    normalized = value.strip().casefold()
    if not normalized:
        return True
    if normalized[0] in "<$[{":
        return True
    if normalized in {
        "changeme",
        "demo",
        "null",
        "none",
        "redacted",
        "removed",
        "test",
    }:
        return True
    return normalized.startswith(
        (
            "your_",
            "your-",
            "example_",
            "example-",
            "sample_",
            "sample-",
            "test_",
            "test-",
            "demo_",
            "demo-",
            "placeholder",
            "removed ",
        )
    )


def _append_csv_secret_matches(
    text: str,
    spans: list[dict[str, object]],
) -> None:
    lines = text.splitlines(keepends=True)
    line_offset = 0
    for line_index, header_line in enumerate(lines[:-1]):
        delimiter = "\t" if "\t" in header_line else ","
        header_values = header_line.rstrip("\r\n").split(delimiter)
        sensitive_columns = {
            column_index
            for column_index, header_value in enumerate(header_values)
            if _is_secret_column(header_value)
        }
        if not sensitive_columns:
            line_offset += len(header_line)
            continue

        row_line = lines[line_index + 1]
        _append_csv_row_matches(
            row_line,
            line_offset + len(header_line),
            delimiter,
            sensitive_columns,
            spans,
        )
        line_offset += len(header_line)


def _append_csv_row_matches(
    row_line: str,
    row_offset: int,
    delimiter: str,
    sensitive_columns: set[int],
    spans: list[dict[str, object]],
) -> None:
    field_offset = row_offset
    row_text = row_line.rstrip("\r\n")
    for column_index, field_value in enumerate(row_text.split(delimiter)):
        field_start = field_offset
        field_offset += len(field_value) + len(delimiter)
        if column_index in sensitive_columns:
            _append_csv_field_match(field_value, field_start, spans)


def _append_csv_field_match(
    field_value: str,
    field_start: int,
    spans: list[dict[str, object]],
) -> None:
    value = field_value.strip()
    if not value or _is_secret_placeholder(value):
        return
    value_start = field_start + len(field_value) - len(field_value.lstrip())
    spans.append(
        {
            "start": value_start,
            "end": value_start + len(value),
            "label": "secret",
        }
    )


def _is_secret_column(header_value: str) -> bool:
    normalized = header_value.strip().strip("\"'").casefold()
    return normalized in {
        "api_key",
        "apikey",
        "auth_token",
        "credential",
        "credential_value",
        "password",
        "passphrase",
        "secret",
        "secret_key",
        "token",
    }


def _append_card_matches(text: str, spans: list[dict[str, object]]) -> None:
    for match in CARD_PATTERN.finditer(text):
        digits = re.sub(r"[ -]", "", match.group())
        if 13 <= len(digits) <= 19 and luhn_valid(digits):
            spans.append(
                {"start": match.start(), "end": match.end(), "label": "account_number"}
            )


def _append_ip_matches(text: str, spans: list[dict[str, object]]) -> None:
    for match in IP_PATTERN.finditer(text):
        octets = match.group().split(".")
        if all(0 <= int(octet) <= 255 for octet in octets):
            spans.append(
                {"start": match.start(), "end": match.end(), "label": "private_url"}
            )


def luhn_valid(digits: str) -> bool:
    checksum = 0
    for index, digit in enumerate(reversed(digits)):
        value = int(digit)
        if index % 2 == 1:
            value *= 2
            if value > 9:
                value -= 9
        checksum += value
    return checksum % 10 == 0


def _trim_url_end(text: str, start: int, end: int) -> int:
    while end > start and text[end - 1] in ".,;:!?)]}":
        end -= 1
    return end


def merge_spans(text: str, spans: list[dict[str, object]]) -> list[dict[str, object]]:
    normalized = []
    for span in spans:
        start = cast(int, span["start"])
        end = cast(int, span["end"])
        label = str(span["label"])
        if start >= end or start < 0 or end > len(text):
            continue
        normalized.append(
            {
                "start": start,
                "end": end,
                "label": label,
                "text": text[start:end],
            }
        )

    prioritized = sorted(
        normalized,
        key=lambda span: (
            -SPAN_PRIORITY.get(str(span["label"]), 0),
            -(int(span["end"]) - int(span["start"])),
            int(span["start"]),
        ),
    )
    selected: list[dict[str, object]] = []
    for candidate in prioritized:
        if any(_spans_overlap(candidate, selected_span) for selected_span in selected):
            continue
        selected.append(candidate)
    return sorted(selected, key=lambda span: (int(span["start"]), int(span["end"])))


def _spans_overlap(
    first_span: dict[str, object],
    second_span: dict[str, object],
) -> bool:
    first_start = cast(int, first_span["start"])
    first_end = cast(int, first_span["end"])
    second_start = cast(int, second_span["start"])
    second_end = cast(int, second_span["end"])
    return first_start < second_end and second_start < first_end
