"""The rules that run after the bilingual tagger, in the order the tagger was scored with.

The tagger reads form. It cannot know that a URL is a documentation page, that a name
belongs to a public figure, or that a handle sits in an author field. These rules can:

1. post: drop a Chinese place with no street part, a public-host or repository URL, a
   shared mailbox, a default account, a code identifier read as a username, and a
   username inside a dropped URL.
2. persons: drop a full name only one listed public figure holds, with no other private
   value within 200 characters, and later partial mentions of that name.
3. places: drop a Han person span that names a listed place.
4. offices: drop a name next to a public-office title, and every other mention of it.
5. thresholds: drop url and username spans the tagger is unsure of.
6. keys: add a username for a handle in an author, reviewer or chat field.

The code is copied from the training repository (experiments/vocab_prune/prune_lib.py in
pii-model), so the hook reproduces the scored results. Standard library only.
"""

from __future__ import annotations

import re
from urllib.parse import urlsplit


# The guideline marks a street-level address, never a city or district alone. Only Chinese
# has a closed set of street-level characters; street names in other languages carry no
# common marker (Jalan Muhibbah, Intrarea Tabacu), so the check covers Han spans only.
HAN = re.compile(r"[\u4e00-\u9fff]")


STREET_PART = re.compile(r"[0-9０-９路街道巷弄段號号樓楼室幢栋棟座]|單元|单元")


# Documentation, package, standards and reserved hosts point at no private person.
PUBLIC_HOST_SUFFIXES = (
    "example.com",
    "example.org",
    "example.net",
    ".example",
    ".test",
    ".invalid",
    "localhost",
    "readthedocs.io",
    "pypi.org",
    "npmjs.com",
    "w3.org",
    "schema.org",
    # Licence, standards, language and vendor documentation hosts: library code links them.
    "apache.org",
    "opensource.org",
    "gnu.org",
    "ietf.org",
    "openid.net",
    "wikipedia.org",
    "wikimedia.org",
    "python.org",
    "golang.org",
    "rust-lang.org",
    "nodejs.org",
    "kubernetes.io",
    "k8s.io",
    "docker.com",
    "learn.microsoft.com",
    "developers.google.com",
    "cloud.google.com",
    "googleapis.com",
    "amazonaws.com",
    "developer.apple.com",
    "developer.mozilla.org",
    "stackoverflow.com",
    "stackexchange.com",
    "cdnjs.cloudflare.com",
    "jsdelivr.net",
    "unpkg.com",
    "platform.openai.com",
)


PUBLIC_HOST_PREFIXES = ("docs.", "schemas.")


# A user page on a code host is one path segment (github.com/name); a repository has two.
REPO_HOSTS = ("github.com", "gitlab.com", "bitbucket.org", "huggingface.co")


# Code reads: a call, an attribute, an assignment target or a definition. Prose words such
# as "from" are not cues: "a message from jake_99" names a user. A call has no space
# before its parenthesis; "user cpohl (email ...)" does.
IDENTIFIER_AFTER = re.compile(r"^(?:\(|\s*=(?!=))")


IDENTIFIER_BEFORE = re.compile(r"(?:\.|\b(?:def|class)\s+)$")


PERSONAL_PATH = re.compile(r"/(?:users?|u)/[^/]+|/~[^/]+|/@[^/]+", re.IGNORECASE)


ROLE_MAILBOXES = {
    "noreply",
    "no-reply",
    "support",
    "info",
    "admin",
    "press",
    "service",
    "security",
    "abuse",
    "postmaster",
}


DEFAULT_ACCOUNTS = {
    "root",
    "admin",
    "postgres",
    "ubuntu",
    "ec2-user",
    "guest",
    "nobody",
    "www-data",
    "sa",
}


def public_url(value: str) -> bool:
    """A URL on a public host, with no personal path and no user:password part. A span that
    does not parse (an unclosed [ reads as a broken IPv6 host) is not shown public, so it stays."""
    try:
        parts = urlsplit(value if "://" in value else f"http://{value}")
    except ValueError:
        return False
    host = (parts.hostname or "").lower()
    if "://" not in value and "." not in host:
        return True  # docs/source/page.md is a path, not a URL
    segments = [segment for segment in parts.path.split("/") if segment]
    repository = host.endswith(REPO_HOSTS) and len(segments) >= 2
    public = (
        host.endswith(PUBLIC_HOST_SUFFIXES)
        or host.startswith(PUBLIC_HOST_PREFIXES)
        or repository
    )
    return public and parts.username is None and not PERSONAL_PATH.search(parts.path)


def shared_email(value: str) -> bool:
    """An organisation's shared mailbox. Reserved domains stay: synthetic sources give
    private people example.com addresses, and 42% of dev-split gold emails use one."""
    return value.lower().partition("@")[0] in ROLE_MAILBOXES


def _policy_drops(span: dict, value: str) -> bool:
    if span["label"] == "address":
        return bool(HAN.search(value)) and not STREET_PART.search(value)
    if span["label"] == "url":
        return public_url(value)
    if span["label"] == "email":
        return shared_email(value)
    return span["label"] == "username" and value.lower() in DEFAULT_ACCOUNTS


def _code_identifier(span: dict, text: str) -> bool:
    """A username span that code reads as an identifier: called, an attribute, assigned or
    defined. The value's own shape is no cue: jeanette.blair and nx.Graph look alike."""
    if span["label"] != "username":
        return False
    before, after = text[: span["start"]], text[span["end"] :]
    return bool(IDENTIFIER_AFTER.match(after)) or bool(IDENTIFIER_BEFORE.search(before))


# Developer text names people by handle in fields the model reads as code: a roster's
# slack or github key, author= and reviewer= in a log, and an @mention in a table cell.
# login, username and handle keys stay out: in a config file they name service accounts.
# The key stands alone (not gf-github:pr-create), a colon or a bare = follows it
# (author = x is code), and the value is no call (currentUser()).
HANDLE = r"([A-Za-z0-9][A-Za-z0-9._-]{1,38})"


HANDLE_KEY = re.compile(
    r"(?i)(?<![\w./-])(?:slack|github|gitlab|author|reviewer|assignee|telegram|twitter)"
    r"[\"']?(?:\s*:|=)\s*[\"']?@?" + HANDLE + r"(?![A-Za-z0-9._\-\[<{(])"
)


HANDLE_CELL = re.compile(r"\|\s*@" + HANDLE + r"\s*(?=\|)")


NOT_A_HANDLE = {
    "null", "none", "nil", "true", "false", "tbd", "todo", "unknown", "me", "self", "user", "username", "login",
    "example", "test", "admin", "root", "anonymous", "ghost", "human", "unassigned", "nobody", "reviewer",
    "assignee", "author", "user_name", "handle", "dependabot", "renovate", "github-actions", "coderabbit",
    "coderabbitai", "codecov", "sonarcloud", "deepsource", "sourcery-ai",
}  # fmt: skip


BOT_HANDLE = re.compile(r"(?i)(?:^|[-_.])bot$|^copilot\b")


def _handle(value: str) -> bool:
    return (
        value.lower() not in NOT_A_HANDLE
        and not value.isdigit()
        and not BOT_HANDLE.search(value)
    )


def username_keys(spans: list[dict], text: str) -> list[dict]:
    """Add a username span for a handle in a username field the spans do not cover."""
    added = []
    for pattern in (HANDLE_KEY, HANDLE_CELL):
        for match in pattern.finditer(text):
            span = {
                "start": match.start(1),
                "end": match.end(1),
                "label": "username",
                "score": 1.0,
            }
            taken = [*spans, *added]
            if _handle(match.group(1)) and not any(
                s["start"] < span["end"] and span["start"] < s["end"] for s in taken
            ):
                added.append(span)
    return sorted([*spans, *added], key=lambda s: (s["start"], s["end"]))


def post_rules(spans: list[dict], text: str) -> list[dict]:
    """Drop the spans the guideline excludes by form: a Chinese place name with no street
    part, a public-host, repository or documentation URL, a path that is not a URL, a shared
    mailbox, a default account, a code identifier read as a username, and a username inside
    a dropped URL. The model flags these by shape; training them away cost recall."""
    dropped = [
        span
        for span in spans
        if _policy_drops(span, text[span["start"] : span["end"]])
        or _code_identifier(span, text)
    ]
    dropped_urls = [span for span in dropped if span["label"] == "url"]
    inside_url = [
        span
        for span in spans
        if span["label"] == "username"
        and any(
            u["start"] <= span["start"] and span["end"] <= u["end"]
            for u in dropped_urls
        )
    ]
    return [span for span in spans if span not in dropped and span not in inside_url]


NAME_DOTS = re.compile(r"[·・‧•]")


# A public figure is rarely written next to a private phone or ID; a private namesake is.
PRIVATE_CONTEXT_CHARS = 200


NOT_CONTEXT = {"person", "date"}


def name_key(value: str) -> str:
    """One spelling per name: one kind of middle dot, single spaces, no case."""
    return " ".join(NAME_DOTS.sub("·", value).split()).casefold()


def full_name(value: str) -> bool:
    """Two Han characters or two words. A lone given name or surname never matches."""
    han = len(HAN.findall(value))
    return han >= 2 if han else len(value.split()) >= 2


def _near_private_pii(span: dict, others: list[dict]) -> bool:
    return any(
        other["start"] < span["end"] + PRIVATE_CONTEXT_CHARS
        and span["start"] - PRIVATE_CONTEXT_CHARS < other["end"]
        for other in others
    )


def _part_of(key: str, public_keys: list[str]) -> bool:
    """A surname or given name of a public name met earlier in the text."""
    if HAN.search(key):
        return any(key in public for public in public_keys)
    return any(set(key.split()) <= set(public.split()) for public in public_keys)


def public_filter(spans: list[dict], text: str, public_names: set[str]) -> list[dict]:
    """Drop person spans that name a listed public figure: a full-name match with no other
    PII nearby, and later partial mentions of the same person. The model reads form and
    cannot know who is famous; the list can."""
    others = [span for span in spans if span["label"] not in NOT_CONTEXT]
    persons = sorted(
        (span for span in spans if span["label"] == "person"),
        key=lambda span: span["start"],
    )
    public, public_keys = [], []
    for span in persons:
        key = name_key(text[span["start"] : span["end"]])
        listed = (
            full_name(key)
            and key in public_names
            and not _near_private_pii(span, others)
        )
        if listed or (public_keys and _part_of(key, public_keys)):
            public.append(span)
        if listed:
            public_keys.append(key)
    return [span for span in spans if span not in public]


# Public offices, in Traditional and Simplified. The guideline excludes politicians and
# officials; a name next to one of these titles is one. Private titles (經理, 老師) are out.
OFFICES_HAN = (
    "總統 总统 總理 总理 首相 主席 議員 议员 立委 部長 部长 外長 外长 國務卿 国务卿 發言人 发言人 "
    "市長 市长 縣長 县长 州長 州长 省長 省长 議長 议长 大使 特首 局長 局长 署長 署长 司長 司长 "
    "總書記 总书记 委員長 委员长 國王 国王 女王 王儲 王储 總督 总督 教宗 將軍 将军 上將 上将 司令"
).split()


OFFICES_LATIN = (
    "President|Vice President|Senator|Sen\\.|Rep\\.|Representative|Congressman|Congresswoman|Governor|"
    "Mayor|Prime Minister|Minister|Secretary of State|Chancellor|King|Queen|Prince|Princess|Pope|Ambassador"
)


OFFICE_BEFORE = re.compile(
    f"(?:{'|'.join(OFFICES_HAN)})\\s*$|\\b(?:{OFFICES_LATIN})\\s+$"
)


OFFICE_AFTER = re.compile(f"^\\s*[副前]?(?:{'|'.join(OFFICES_HAN)})")


OFFICE_WINDOW = 12


def _titled(span: dict, text: str) -> bool:
    before = text[max(0, span["start"] - OFFICE_WINDOW) : span["start"]]
    after = text[span["end"] : span["end"] + OFFICE_WINDOW]
    return bool(
        OFFICE_BEFORE.search(before)
        or (
            HAN.search(text[span["start"] : span["end"]]) and OFFICE_AFTER.search(after)
        )
    )


def _same_person(key: str, public_keys: list[str]) -> bool:
    """The same key, or a part of a public full name (a later surname alone)."""
    return key in public_keys or _part_of(
        key, [public for public in public_keys if full_name(public)]
    )


def office_filter(spans: list[dict], text: str) -> list[dict]:
    """Drop person spans next to a public-office title (議員郭榮鏗, Senator John Smith), unless
    private PII sits nearby, and every other mention of the same name in the text."""
    others = [span for span in spans if span["label"] not in NOT_CONTEXT]
    persons = [span for span in spans if span["label"] == "person"]
    public_keys = [
        name_key(text[span["start"] : span["end"]])
        for span in persons
        if _titled(span, text) and not _near_private_pii(span, others)
    ]
    if not public_keys:
        return spans
    public = [
        span
        for span in persons
        if _same_person(name_key(text[span["start"] : span["end"]]), public_keys)
    ]
    return [span for span in spans if span not in public]


def place_filter(spans: list[dict], text: str, place_names: set[str]) -> list[dict]:
    """Drop Han person spans that name a listed place (德國, 伯恩州). The list leaves out
    every string that is also a person's name. Latin text is left alone: full names such
    as Anna Maria and Maria Helena are also towns, and the dev split holds both."""
    return [
        span
        for span in spans
        if not (
            span["label"] == "person"
            and HAN.search(key := name_key(text[span["start"] : span["end"]]))
            and full_name(key)
            and key in place_names
        )
    ]


def apply_chain(
    spans: list[dict],
    text: str,
    persons: set[str],
    places: set[str],
    thresholds: dict[str, float],
) -> list[dict]:
    """Every rule, in the scored order. Spans carry the tagger's labels and scores."""
    spans = post_rules(spans, text)
    spans = public_filter(spans, text, persons)
    spans = place_filter(spans, text, places)
    spans = office_filter(spans, text)
    spans = [
        span for span in spans if span["score"] >= thresholds.get(span["label"], 0.0)
    ]
    return username_keys(spans, text)
