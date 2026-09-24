"""Deterministic PII rules.

Standard library only. The Redact model, the OpenAI Privacy Filter, and the
hook all consume these spans. Nothing here loads a checkpoint.
"""

from __future__ import annotations

import itertools
import math
import re
import unicodedata
from bisect import bisect_left
from collections import Counter
from typing import Protocol, cast

from pii_secret_patterns import GITLEAKS_RULES


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

# Characters that carry no width: a value split by one of them must still
# match, and a value that ends on one must still be reported whole. The set
# excludes tab, LF and CR, which carry the line and column structure that the
# CSV rule reads.
INVISIBLE_CODES = (
    *range(0x00, 0x09),
    0x0B,
    0x0C,
    *range(0x0E, 0x20),
    *range(0x7F, 0xA0),
    0xAD,
    0x180E,
    *range(0x200B, 0x2010),
    *range(0x202A, 0x202F),
    *range(0x2060, 0x2070),
    0xFEFF,
)
INVISIBLE_CHARACTERS = frozenset(map(chr, INVISIBLE_CODES))
INVISIBLE_PATTERN = re.compile(
    "[" + re.escape("".join(sorted(INVISIBLE_CHARACTERS))) + "]"
)

# A card is written either as one run of digits or as digit groups held by a
# space or a hyphen. Both forms must start and end on a digit: a run that ends
# on a separator swallows the character after it, and a run that continues past
# a separator merges the card with the next digit it happens to precede.
CARD_PATTERN = re.compile(r"(?<!\d)(?:\d{13,19}|\d{1,6}(?:[ -]\d{2,6}){1,5})(?!\d)")

# A phone is one of: a country code marked by + or 00, a parenthesised area
# code, a dotted NANP number, a bare run of ten digits, or digit groups held by
# a space or a hyphen. The dotted form is pinned to three-three-four digits, so
# a dotted quad never reads as a phone. A dotted run of any other length is a
# version number, and neither a quad nor a version is a phone. The grouped form
# carries no length in its shape, so _append_phone_matches counts its digits.
#
# The guard on the right differs per shape. A sentence may end on the number,
# so a trailing full stop must not reject it. The dotted shape instead refuses
# a fourth group, which is what makes it leave a dotted quad alone. The bare
# ten-digit run refuses a letter after it: that is the start of a hex id, like
# a docker layer. The guard on the left refuses a word character and a longer
# dotted run, nothing else.
#
# Two shapes are deliberately narrow. The 00 prefix needs a separator after the
# country code, because a bare run behind it is a reference number and not a
# dialled one. The grouped form needs its middle groups to carry three or four
# digits, because a two-digit middle group is the shape of a tax identifier.
PHONE_PATTERN = re.compile(
    r"(?<!\w)(?<!\d\.)"
    r"(?:"
    r"\+\d{1,3}[ .-]?(?:\(\d{1,4}\)[ .-]?)?\d(?:[ .-]?\d){5,12}(?!\w)"
    r"|00\d{1,3}[ .-](?:\(\d{1,4}\)[ .-]?)?\d(?:[ .-]?\d){5,12}(?!\w)"
    r"|\(\d{1,4}\)[ .-]?\d(?:[ .-]?\d){6,12}(?!\w)"
    r"|\d{3}\.\d{3}\.\d{4}(?!\.?\d)"
    r"|(?<!\d)\d{10}(?!\w)"
    r"|(?<!\d[ -])\d{2,4}(?:[ -]\d{3,4}){0,2}[ -]\d{3,4}(?!\d)"
    r")"
)
PHONE_DIGIT_FLOOR = 9
PHONE_DIGIT_CEILING = 15

# The guards on both sides reject a character that continues the token, and
# they must not reject a full stop that ends a sentence. A dot is only a
# continuation when a word character sits beside it: in front, the dot of a
# longer dotted run; behind, the dot of a longer domain.
EMAIL_PATTERN = re.compile(
    r"(?<![\w+-])(?<!\w\.)[A-Za-z0-9][A-Za-z0-9._%+-]*"
    r"@[A-Za-z0-9](?:[A-Za-z0-9.-]*[A-Za-z0-9])?\.[A-Za-z]{2,}(?![\w-])"
)
URL_PATTERN = re.compile(r"https?://[^\s<>\"']+", re.IGNORECASE)
BANK_ACCOUNT_PATTERN = re.compile(
    r"(?i)\b(?:account|acct)(?:\s+(?:number|no\.?))?"
    r"\s*(?:is|=|:)?\s*(?P<value>\d{8,})\b"
)
IBAN_PATTERN = re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b")
# The same guard as the address rule: refuse a word character on either side,
# refuse to sit inside a longer dotted run, and let a full stop end a sentence.
IP_PATTERN = re.compile(r"(?<!\w)(?<!\d\.)(?:\d{1,3}\.){3}\d{1,3}(?!\w)(?!\.\d)")
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
# A value never starts with a hyphen. Joined to the keyword by one, it is the
# rest of a name, like the secret-scanning path. A slash still joins them: a
# one-time secret link carries its key right after /secret/.
SECRET_CONTEXT_PATTERN = re.compile(
    r"(?i)\b(?:aws_secret_access_key|secret(?:\s+key|_access_key)?)"
    r"(?![a-z0-9_])"
    r"\s*(?:(?:is|=|:)\s*)?(?P<value>(?!-)[A-Za-z0-9/+=!@#$%^&*_-]{16,})"
)
PASSWORD_CONTEXT_PATTERN = re.compile(
    r"(?i)\b(?:password|passphrase)\s*(?:is|=|:)\s*"
    r"(?P<value>[^\s,.;]{4,})"
)
SECRET_CONTEXT_KEYS = (
    r"(?:api(?:[_-]?key|\s+key)|api[_-]?token|client[_-]?secret|"
    r"access[_-]?token|refresh[_-]?token|session[_-]?token|"
    r"private[_-]?(?:key|token)|auth[_-]?token|"
    r"aws[_-]?session[_-]?token|credential[_-]?value|"
    r"secret(?:[_-]?(?:key|token|value|access[_-]?key))?|token|"
    r"x-api-key|_authToken|deploy[_-]?(?:secret|token)|"
    r"service[_-]?(?:secret|token))"
)
# The password keys run as their own pattern, because only the other keys
# skip a value of plain words. See SECRET_ALLOWLISTS.
PASSWORD_CONTEXT_KEYS = (
    r"(?:database[_-]?password|db[_-]?password|docker[_-]?password|"
    r"password|passphrase)"
)
CONTEXT_VALUE = (
    r"(?![a-z0-9_])\s*[\"']?\s*(?:is|=|:)\s*"
    r"(?P<quote>[\"'`]?)(?P<value>"
    r"[a-z0-9][a-z0-9._~+/=:@$!%*&?{}-]{7,}"
    r")(?(quote)(?P=quote))"
)
GENERIC_SECRET_CONTEXT_PATTERN = re.compile(
    r"(?ix)(?<![a-z0-9])" + SECRET_CONTEXT_KEYS + CONTEXT_VALUE
)
GENERIC_PASSWORD_CONTEXT_PATTERN = re.compile(
    r"(?ix)(?<![a-z0-9])" + PASSWORD_CONTEXT_KEYS + CONTEXT_VALUE
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
GITLEAKS_PATTERNS = {
    rule_id: re.compile(source) for rule_id, source, _, _, _ in GITLEAKS_RULES
}
# Each gitleaks pattern with the checks its secret must pass: the entropy it
# must exceed, and the regexes that mark it as a known false positive.
GITLEAKS_CHECKS = tuple(
    (
        GITLEAKS_PATTERNS[rule_id],
        entropy_floor,
        tuple(re.compile(allowed) for allowed in allowlist),
    )
    for rule_id, _, _, entropy_floor, allowlist in GITLEAKS_RULES
)

# Every rule pattern, with the literals it cannot match without. A pattern
# whose keywords are all absent from the text is skipped, matched ignoring ASCII
# case. That skip is where the scan time went: the JWK pattern alone tried a
# lookahead of 200 characters at every position of every text, and almost no
# text holds "kty". A keyword must be required by the pattern's own shape. One
# that is only common would skip a real match. An empty tuple means the pattern
# always runs. The order is the order the native engine reports matches in.
RULE_KEYWORDS: dict[re.Pattern[str], tuple[str, ...]] = {
    EMAIL_PATTERN: ("@",),
    URL_PATTERN: ("://",),
    PHONE_PATTERN: (),
    CARD_PATTERN: (),
    BANK_ACCOUNT_PATTERN: ("acc",),
    IBAN_PATTERN: (),
    IP_PATTERN: (),
    AWS_ACCESS_KEY_PATTERN: ("akia",),
    SECRET_PREFIX_PATTERN: (
        "sk_",
        "ghp_",
        "gho_",
        "ghu_",
        "ghs_",
        "ghr_",
        "xox",
        "aiza",
    ),
    PROVIDER_SECRET_PATTERN: (
        "ocid1.securitytoken",
        "asia",
        "gocspx-",
        "cfp_",
        "dop_v1_",
        "hrk_",
        "vercel_",
        "nfp_",
        "sbp_",
        "sk-",
        "rk_",
        "sg.",
        "whsec_",
        "glc_",
        "snyk_",
        "pul-",
        "sk.",
        "dd_api_",
        "nrak-",
        "github_pat_",
        "glpat-",
        "bb_app_",
        "npm_",
        "pypi-",
        "dckr_pat_",
        "cci_",
        "bk_",
        "travis_",
        "codecov_",
        "dp.st.",
        "hvs.",
        "lin_api_",
        "sntrys_",
        "webex_",
        "intercom_",
        ":aa",
    ),
    JWT_PATTERN: ("eyj",),
    PRIVATE_KEY_PATTERN: ("-----begin ",),
    PASSWORD_CONTEXT_PATTERN: ("pass",),
    SECRET_CONTEXT_PATTERN: ("secret",),
    GENERIC_SECRET_CONTEXT_PATTERN: (
        "api",
        "secret",
        "token",
        "private",
        "credential",
    ),
    GENERIC_PASSWORD_CONTEXT_PATTERN: ("pass",),
    BEARER_SECRET_PATTERN: ("bearer",),
    BASIC_SECRET_PATTERN: ("basic",),
    DATABASE_URL_SECRET_PATTERN: ("://",),
    MAILGUN_SECRET_PATTERN: ("mailgun",),
    NOTION_SECRET_PATTERN: ("notion",),
    WEBHOOK_SECRET_PATTERN: ("hooks.slack.com/services/", "discord"),
    SIGNED_URL_SECRET_PATTERN: ("sig=",),
    XML_SECRET_CONTEXT_PATTERN: (
        "<password",
        "<passphrase",
        "<secret",
        "<token",
        "<api",
    ),
    JWK_PRIVATE_VALUE_PATTERN: ("kty",),
    TERRAFORM_SECRET_VALUE_PATTERN: ("resource",),
    COOKIE_SECRET_PATTERN: ("set-cookie:",),
    STRUCTURED_SECRET_VALUE_PATTERN: ("secret",),
    CJK_SECRET_CONTEXT_PATTERN: ("密碼", "密鑰", "金鑰", "令牌"),
    ISO_DATE_PATTERN: (),
    MONTH_DATE_PATTERN: (
        "jan",
        "feb",
        "mar",
        "apr",
        "may",
        "jun",
        "jul",
        "aug",
        "sep",
        "oct",
        "nov",
        "dec",
    ),
    # gitleaks keywords are not all required by their pattern. They are the
    # gate gitleaks itself applies, so gating on them keeps gitleaks' results.
    **{
        GITLEAKS_PATTERNS[rule_id]: keywords
        for rule_id, _, keywords, _, _ in GITLEAKS_RULES
    },
}

# The non-ASCII characters that IGNORECASE matches to an ASCII letter: the
# dotted and the dotless i, the long s, and the Kelvin sign. An ASCII keyword
# search cannot see them, so a text holding one runs every pattern.
FOLDING_CHARACTERS = frozenset("\u0130\u0131\u017f\u212a")

# How re reads \w in a str pattern: a letter, a number, or the underscore.
# PCRE2 also counts combining marks and connector punctuation, so the native
# engine is handed this class in place of \w, and a word boundary built from it
# in place of \b.
PYTHON_WORD_CHARACTERS = r"\p{L}\p{N}_"
PYTHON_WORD_BOUNDARY = (
    rf"(?:(?<=[{PYTHON_WORD_CHARACTERS}])(?![{PYTHON_WORD_CHARACTERS}])"
    rf"|(?<![{PYTHON_WORD_CHARACTERS}])(?=[{PYTHON_WORD_CHARACTERS}]))"
)

# Where PCRE2's own \w differs from re's when both know the same Unicode
# version: a combining mark, or connector punctuation other than "_". A text
# without one gets the patterns as written, which PCRE2 matches twice as fast
# as the spelled-out \b.
PCRE2_WORD_DIFFERENCE = r"(?!_)[\p{Mn}\p{Pc}]"

# The classes the rules are built from, as re spells them and as the native
# engine is handed them, and whether the pattern ignores case. Every other
# construct in the rules names characters one by one. Where the two answers for
# a class differ on a character, the Unicode tables of Python and of PCRE2
# disagree about it: Python 3.11 knows Unicode 14 and PCRE2 knows a later one.
UNICODE_CLASS_CHECKS = (
    (r"\w", f"[{PYTHON_WORD_CHARACTERS}]", False),
    (r"\d", r"\d", False),
    (r"\s", r"\s", False),
    (r"[a-z]", r"[a-z]", True),
)

# The version of pii_rules_native this file was written against. A module of
# any other version is ignored, so a stale build cannot answer for these rules.
NATIVE_ENGINE_VERSION = 2


class NativeEngine(Protocol):
    """What pii_rules calls on pii_rules_native.Engine."""

    def scan(
        self, text: str
    ) -> list[list[tuple[int, int, list[tuple[int, int] | None]]]] | None: ...

    def strip_invisibles(self, text: str) -> tuple[str, list[int]] | None: ...


class _NativeMatch:
    """The part of re.Match that the rule helpers read."""

    __slots__ = ("_groups", "_span", "_text")

    def __init__(
        self,
        text: str,
        span: tuple[int, int],
        groups: dict[str, tuple[int, int] | None],
    ) -> None:
        self._text = text
        self._span = span
        self._groups = groups

    def span(self, group: str | None = None) -> tuple[int, int]:
        if group is None:
            return self._span
        # re.Match answers (-1, -1) for a group that took no part in the match.
        return self._groups[group] or (-1, -1)

    def start(self) -> int:
        return self._span[0]

    def end(self) -> int:
        return self._span[1]

    def group(self) -> str:
        return self._text[self._span[0] : self._span[1]]


def _load_native_engine() -> tuple[NativeEngine | None, str]:
    """Return the native engine, or None and the reason it is not in use."""
    try:
        import pii_rules_native  # ty: ignore[unresolved-import]
    except ImportError as error:
        return None, f"python ({error})"
    version = getattr(pii_rules_native, "ENGINE_VERSION", None)
    if version != NATIVE_ENGINE_VERSION:
        return None, (
            f"python (pii_rules_native is version {version}, "
            f"these rules need version {NATIVE_ENGINE_VERSION})"
        )
    invisible = "".join(sorted(INVISIBLE_CHARACTERS))
    try:
        sources = [
            (
                pattern.pattern,
                _pcre2_source(pattern.pattern),
                bool(pattern.flags & re.IGNORECASE),
                list(pattern.groupindex),
                list(keywords),
            )
            for pattern, keywords in RULE_KEYWORDS.items()
        ]
        decline = _native_decline_characters(pii_rules_native)
        engine = pii_rules_native.Engine(
            sources, decline, PCRE2_WORD_DIFFERENCE, invisible
        )
    except ValueError as error:
        return None, f"python (pii_rules_native refused a pattern: {error})"
    return engine, "native"


def _pcre2_source(source: str) -> str:
    """Spell \\w and \\b in a pattern the way re reads them."""
    parts: list[str] = []
    index = 0
    in_class = False
    while index < len(source):
        character = source[index]
        if character == "\\":
            escape = source[index : index + 2]
            if escape in (r"\W", r"\B"):
                raise ValueError(f"{escape} has no translation for PCRE2")
            if escape == r"\w":
                word = PYTHON_WORD_CHARACTERS
                parts.append(word if in_class else f"[{word}]")
            elif escape == r"\b" and not in_class:
                parts.append(PYTHON_WORD_BOUNDARY)
            else:
                parts.append(escape)
            index += 2
            continue
        if character == "[" and not in_class:
            in_class = True
        elif character == "]" and in_class:
            in_class = False
        parts.append(character)
        index += 1
    return "".join(parts)


def _native_decline_characters(native_module) -> str:
    """Return the characters on which the native engine could differ from re.

    The folding characters are always declined, because an ASCII keyword
    search cannot see them. The classes can also differ where the two Unicode
    tables disagree, which only happens when their versions differ. Deriving
    those characters costs 1.3 s on a slow CPU, so it runs only then.
    """
    if native_module.UNICODE_VERSION == unicodedata.unidata_version:
        return "".join(sorted(FOLDING_CHARACTERS))
    return _derive_decline_characters(native_module.class_members)


def _derive_decline_characters(class_members) -> str:
    """Run each class over every code point in both engines, and return the
    characters the two place differently, plus the folding characters."""
    every_character = "".join(
        map(chr, itertools.chain(range(0xD800), range(0xE000, 0x110000)))
    )
    differing = set(FOLDING_CHARACTERS)
    for python_class, native_class, ignore_case in UNICODE_CLASS_CHECKS:
        flags = re.IGNORECASE if ignore_case else 0
        python_members = set(re.findall(python_class, every_character, flags))
        native_members = set(class_members(native_class, ignore_case, every_character))
        differing |= python_members ^ native_members
    # The strip removes the invisible characters before any scan, so none can
    # reach the engine. Leaving them out keeps every declined character outside
    # ASCII, which is what lets the engine skip the check on an ASCII text.
    return "".join(sorted(differing - INVISIBLE_CHARACTERS))


# RULES_ENGINE names the engine in use, and why when it is not the native one.
# The server logs it at start.
_NATIVE_ENGINE, RULES_ENGINE = _load_native_engine()


def deterministic_spans(text: str) -> list[dict[str, object]]:
    """Return the merged spans of text, with offsets into text itself.

    Every rule matches against the text with the invisible characters removed,
    so a zero-width character cannot hide a value. The offsets are then put
    back onto the original text, because the hook and the redactor both address
    the original string by offset.
    """
    clean_text, index_map = _strip_invisibles(text)
    spans = _scan_spans(clean_text)
    if index_map is not None:
        spans = [_restore_offsets(span, index_map) for span in spans]
    return merge_spans(text, spans)


def _strip_invisibles(text: str) -> tuple[str, list[int] | None]:
    """Remove the invisible characters and return the clean text plus a map.

    The map holds the original index of each clean character, plus one final
    entry holding len(text). None means the text was already clean, which is
    the common case and costs one scan.
    """
    if _NATIVE_ENGINE is not None:
        try:
            stripped = _NATIVE_ENGINE.strip_invisibles(text)
        except UnicodeEncodeError:
            # A lone surrogate cannot cross into Rust. The loop below handles it.
            pass
        else:
            return (text, None) if stripped is None else stripped
    if not INVISIBLE_PATTERN.search(text):
        return text, None
    characters: list[str] = []
    index_map: list[int] = []
    for index, character in enumerate(text):
        if character in INVISIBLE_CHARACTERS:
            continue
        characters.append(character)
        index_map.append(index)
    index_map.append(len(text))
    return "".join(characters), index_map


def _restore_offsets(
    span: dict[str, object], index_map: list[int]
) -> dict[str, object]:
    """Move a clean-text span onto the original text.

    The end maps through the entry after the last matched character, so a
    match that ends on an invisible character keeps that character inside the
    span instead of leaving it behind.
    """
    return {
        "start": index_map[cast(int, span["start"])],
        "end": index_map[cast(int, span["end"])],
        "label": span["label"],
    }


# The secret patterns, and the group that holds the value when the match
# carries a label or a key in front of it.
SECRET_RULES: tuple[tuple[re.Pattern[str], str | None], ...] = (
    (AWS_ACCESS_KEY_PATTERN, None),
    (SECRET_PREFIX_PATTERN, None),
    (PROVIDER_SECRET_PATTERN, None),
    (JWT_PATTERN, None),
    (PRIVATE_KEY_PATTERN, None),
    (PASSWORD_CONTEXT_PATTERN, "value"),
    (SECRET_CONTEXT_PATTERN, "value"),
    (GENERIC_SECRET_CONTEXT_PATTERN, "value"),
    (GENERIC_PASSWORD_CONTEXT_PATTERN, "value"),
    (BEARER_SECRET_PATTERN, "value"),
    (BASIC_SECRET_PATTERN, "value"),
    (DATABASE_URL_SECRET_PATTERN, "value"),
    (MAILGUN_SECRET_PATTERN, "value"),
    (NOTION_SECRET_PATTERN, "value"),
    (WEBHOOK_SECRET_PATTERN, None),
    (SIGNED_URL_SECRET_PATTERN, "value"),
    (XML_SECRET_CONTEXT_PATTERN, "value"),
    (JWK_PRIVATE_VALUE_PATTERN, "value"),
    (TERRAFORM_SECRET_VALUE_PATTERN, "value"),
    (COOKIE_SECRET_PATTERN, "value"),
    (STRUCTURED_SECRET_VALUE_PATTERN, "value"),
    (CJK_SECRET_CONTEXT_PATTERN, "value"),
)

# Values a rule matches that are not secrets. After an api-key or a token
# keyword, lowercase words joined by hyphens are prose or a name, as in
# "airtable-api-key: keyword-context rule". A token carries a digit or a capital.
# The password rules keep no such list, because a passphrase is words.
SECRET_ALLOWLISTS = {
    GENERIC_SECRET_CONTEXT_PATTERN: (re.compile(r"\A[a-z]+(?:[-_][a-z]+)*\Z"),),
}

# What each engine answers: the matches of every pattern that matched.
Matches = dict[re.Pattern[str], list]


def _scan_spans(text: str) -> list[dict[str, object]]:
    found = _find_matches(text)
    spans: list[dict[str, object]] = []
    _append_matches(text, found, EMAIL_PATTERN, "private_email", spans)
    _append_matches(text, found, URL_PATTERN, "private_url", spans, trim_url=True)
    _append_phone_matches(found, spans)
    _append_card_matches(found, spans)
    _append_matches(
        text, found, BANK_ACCOUNT_PATTERN, "account_number", spans, group_name="value"
    )
    _append_matches(text, found, IBAN_PATTERN, "account_number", spans)
    _append_ip_matches(found, spans)

    for pattern, group_name in SECRET_RULES:
        _append_secret_matches(
            text,
            found,
            pattern,
            spans,
            group_name=group_name,
            allowlist=SECRET_ALLOWLISTS.get(pattern, ()),
        )
    for pattern, entropy_floor, allowlist in GITLEAKS_CHECKS:
        _append_secret_matches(
            text,
            found,
            pattern,
            spans,
            group_name="value" if "value" in pattern.groupindex else None,
            entropy_floor=entropy_floor,
            allowlist=allowlist,
        )
    _append_csv_secret_matches(text, spans)

    _append_matches(text, found, ISO_DATE_PATTERN, "private_date", spans)
    _append_matches(text, found, MONTH_DATE_PATTERN, "private_date", spans)
    return spans


def _find_matches(text: str) -> Matches:
    if _NATIVE_ENGINE is not None:
        found = _native_matches(text)
        if found is not None:
            return found
    return _python_matches(text)


def _native_matches(text: str) -> Matches | None:
    """Return the native engine's matches, or None when re must scan the text."""
    try:
        per_pattern = _NATIVE_ENGINE.scan(text)
    except (UnicodeEncodeError, ValueError):
        # A lone surrogate cannot cross into Rust, and PCRE2 reports a pattern
        # that hit its resource limit as an error. re handles both.
        return None
    if per_pattern is None:
        return None
    found: Matches = {}
    for pattern, matches in zip(RULE_KEYWORDS, per_pattern):
        if not matches:
            continue
        names = tuple(pattern.groupindex)
        found[pattern] = [
            _NativeMatch(text, (start, end), dict(zip(names, groups)))
            for start, end, groups in matches
        ]
    return found


def _python_matches(text: str) -> Matches:
    every_gate_open = not FOLDING_CHARACTERS.isdisjoint(text)
    lowered = text.lower()
    found: Matches = {}
    for pattern, keywords in RULE_KEYWORDS.items():
        if keywords and not every_gate_open:
            if not any(keyword in lowered for keyword in keywords):
                continue
        matches = list(pattern.finditer(text))
        if matches:
            found[pattern] = matches
    return found


def _append_phone_matches(found: Matches, spans: list[dict[str, object]]) -> None:
    """Keep the phone-shaped runs that carry a plausible digit count.

    The shape of a grouped number fixes no length, so the count decides: below
    the floor a grouped run is a date or a version, and above the ceiling it is
    not a dialable number. A miss is cheaper than a false positive here, which
    blocks ordinary work at the default level.
    """
    for match in found.get(PHONE_PATTERN, ()):
        digit_count = sum(character.isdigit() for character in match.group())
        if not PHONE_DIGIT_FLOOR <= digit_count <= PHONE_DIGIT_CEILING:
            continue
        spans.append(
            {"start": match.start(), "end": match.end(), "label": "private_phone"}
        )


def _append_matches(
    text: str,
    found: Matches,
    pattern: re.Pattern[str],
    label: str,
    spans: list[dict[str, object]],
    *,
    group_name: str | None = None,
    trim_url: bool = False,
) -> None:
    for match in found.get(pattern, ()):
        start, end = match.span(group_name) if group_name else match.span()
        if trim_url:
            # The trim cannot empty a match: every URL keeps "https://", whose
            # last character is not in the trim set. A pattern that did return
            # an empty span would be dropped by merge_spans, which normalizes
            # start >= end away.
            end = _trim_url_end(text, start, end)
        spans.append({"start": start, "end": end, "label": label})


def _append_secret_matches(
    text: str,
    found: Matches,
    pattern: re.Pattern[str],
    spans: list[dict[str, object]],
    *,
    group_name: str | None = None,
    entropy_floor: float | None = None,
    allowlist: tuple[re.Pattern[str], ...] = (),
) -> None:
    for match in found.get(pattern, ()):
        start, end = match.span(group_name) if group_name else match.span()
        # gitleaks judges the secret as matched, before any trim.
        secret = text[start:end]
        if entropy_floor and _shannon_entropy(secret) <= entropy_floor:
            continue
        if any(allowed.search(secret) for allowed in allowlist):
            continue
        end = _trim_secret_end(text, start, end)
        if start >= end or _is_secret_placeholder(text[start:end]):
            continue
        spans.append({"start": start, "end": end, "label": "secret"})


def _shannon_entropy(value: str) -> float:
    """Bits per character, as gitleaks computes it for its entropy floor."""
    if not value:
        return 0.0
    length = len(value)
    return -sum(
        count / length * math.log2(count / length) for count in Counter(value).values()
    )


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
        # Every secret column name holds one of these keywords, so a line
        # holding none cannot be the header. casefold, as the column compare
        # uses: it folds "ß" to "ss", which lower() leaves alone.
        folded_line = header_line.casefold()
        if not any(keyword in folded_line for keyword in SECRET_COLUMN_KEYWORDS):
            line_offset += len(header_line)
            continue
        delimiter = "\t" if "\t" in header_line else ","
        header_values = header_line.rstrip("\r\n").split(delimiter)
        # A header names two columns or more. One name and a trailing comma is
        # a line of code that lists quoted names, one per line.
        if sum(bool(value.strip()) for value in header_values) < 2:
            line_offset += len(header_line)
            continue
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


SECRET_COLUMN_NAMES = frozenset(
    {
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
)
SECRET_COLUMN_KEYWORDS = ("api", "credential", "pass", "secret", "token")


def _is_secret_column(header_value: str) -> bool:
    normalized = header_value.strip().strip("\"'").casefold()
    return normalized in SECRET_COLUMN_NAMES


def _append_card_matches(found: Matches, spans: list[dict[str, object]]) -> None:
    for match in found.get(CARD_PATTERN, ()):
        digits = re.sub(r"[ -]", "", match.group())
        if 13 <= len(digits) <= 19 and luhn_valid(digits):
            spans.append(
                {"start": match.start(), "end": match.end(), "label": "account_number"}
            )


def _append_ip_matches(found: Matches, spans: list[dict[str, object]]) -> None:
    for match in found.get(IP_PATTERN, ()):
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


# A trimmed fragment shorter than this carries nothing worth reporting. The
# split leaves a blank run or a single character between two claimed ranges,
# and a labelled blank reaches the hook as a span it cannot act on.
MIN_FRAGMENT_LENGTH = 2


def merge_spans(text: str, spans: list[dict[str, object]]) -> list[dict[str, object]]:
    """Return the disjoint spans of text, highest priority first.

    A span that overlaps a selected one keeps the parts the selected span does
    not cover, and it is dropped only when nothing is left. So a signature
    value inside a URL leaves the rest of the URL reported, instead of erasing
    the whole URL from the answer. A trimmed part that carries no value is
    dropped as well: a fragment is only reported when it says something.

    The selected ranges are held sorted, and each candidate is placed into them
    with a bisect. Scanning the whole selected list per candidate made this
    quadratic, and the merge is the dominant cost on large input.
    """
    selected: list[dict[str, object]] = []
    claimed_starts: list[int] = []
    claimed_ranges: list[tuple[int, int]] = []
    for candidate in sorted(
        _normalized_spans(text, spans),
        key=lambda span: (
            -SPAN_PRIORITY.get(str(span["label"]), 0),
            -(int(span["end"]) - int(span["start"])),
            int(span["start"]),
        ),
    ):
        candidate_start = cast(int, candidate["start"])
        candidate_end = cast(int, candidate["end"])
        for start, end in _uncovered_parts(
            candidate_start,
            candidate_end,
            claimed_starts,
            claimed_ranges,
        ):
            trimmed = start != candidate_start or end != candidate_end
            if trimmed and _is_degenerate_fragment(text, start, end):
                continue
            selected.append(
                {
                    "start": start,
                    "end": end,
                    "label": candidate["label"],
                    "text": text[start:end],
                }
            )
            _claim_range(claimed_starts, claimed_ranges, start, end)
    return sorted(selected, key=lambda span: (int(span["start"]), int(span["end"])))


def _normalized_spans(
    text: str, spans: list[dict[str, object]]
) -> list[dict[str, object]]:
    normalized: list[dict[str, object]] = []
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
    return normalized


def _is_degenerate_fragment(text: str, start: int, end: int) -> bool:
    """True when a trimmed fragment is blank or a single character."""
    return len(text[start:end].strip()) < MIN_FRAGMENT_LENGTH


def _uncovered_parts(
    start: int,
    end: int,
    claimed_starts: list[int],
    claimed_ranges: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    """Return the parts of [start, end) that no claimed range covers."""
    parts: list[tuple[int, int]] = []
    index = bisect_left(claimed_starts, start)
    if index > 0 and claimed_ranges[index - 1][1] > start:
        index -= 1
    cursor = start
    while index < len(claimed_ranges) and claimed_ranges[index][0] < end:
        claimed_start, claimed_end = claimed_ranges[index]
        if claimed_start > cursor:
            parts.append((cursor, claimed_start))
        cursor = max(cursor, claimed_end)
        if cursor >= end:
            # A claim reaches the end of the range, so nothing is left open.
            return parts
        index += 1
    # Every claim that reached the end returned above, so the tail is open and
    # cursor is still inside the range.
    parts.append((cursor, end))
    return parts


def _claim_range(
    claimed_starts: list[int],
    claimed_ranges: list[tuple[int, int]],
    start: int,
    end: int,
) -> None:
    """Add [start, end) to the sorted claimed ranges, joining its neighbours."""
    index = bisect_left(claimed_starts, start)
    if index > 0 and claimed_ranges[index - 1][1] >= start:
        index -= 1
        start = claimed_ranges[index][0]
    last = index
    while last < len(claimed_ranges) and claimed_ranges[last][0] <= end:
        end = max(end, claimed_ranges[last][1])
        last += 1
    claimed_ranges[index:last] = [(start, end)]
    claimed_starts[index:last] = [start]
