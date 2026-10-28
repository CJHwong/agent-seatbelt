#!/usr/bin/env python3
"""Plant random fake secrets and account numbers in the shapes the tagger's noise filter
must keep, and report how many each server still flags.

Each case is a template with one {v} slot. The runner fills it with a random value per
trial, from a fixed seed, so every server sees the same texts. A plant is found when a
span with the expected label overlaps it; for a control (no expected label) any
overlapping span counts. Only gated cases decide the exit code: every trial must be found.
The rest are reported, such as the shapes the filter drops by design.

Usage:
    ./run-planted.py --server NAME=URL [--server NAME=URL ...] [--trials N]
"""

from __future__ import annotations

import argparse
import json
import random
import string
import sys
import urllib.error
import urllib.request
from pathlib import Path

DEFAULT_FIXTURE = Path(__file__).resolve().parents[1] / "tests" / "planted-cases.jsonl"
SEED = 20261028
LETTERS_AND_DIGITS = string.ascii_letters + string.digits


def fake_secret(rng: random.Random) -> str:
    """Mixed-case letters and digits, the shape the filter judges by its key word."""
    while True:
        value = "".join(
            rng.choice(LETTERS_AND_DIGITS) for _ in range(rng.choice((24, 32, 40)))
        )
        if (
            any(c.isdigit() for c in value)
            and any(c.isupper() for c in value)
            and any(c.islower() for c in value)
        ):
            return value


def fake_digits(rng: random.Random) -> str:
    """A 9 to 12 digit run with no separators, so it is never phone-shaped."""
    return str(rng.randint(1, 9)) + "".join(
        rng.choice(string.digits) for _ in range(rng.randint(8, 11))
    )


FAKES = {"secret": fake_secret, "digits": fake_digits}


def load_cases(fixture: Path) -> list[dict]:
    cases = [
        json.loads(line)
        for line in fixture.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    for case in cases:
        if case["value"] not in FAKES or case["template"].count("{v}") != 1:
            raise ValueError(
                f"case {case['id']}: needs one {{v}} slot and a value of {sorted(FAKES)}"
            )
    return cases


def plants(cases: list[dict], trials: int) -> list[tuple[dict, str, int, int]]:
    rng = random.Random(SEED)
    out = []
    for case in cases:
        for _ in range(trials):
            value = FAKES[case["value"]](rng)
            text = case["template"].format(v=value)
            start = text.index(value)
            out.append((case, text, start, start + len(value)))
    return out


def spans(server: str, text: str) -> list[dict]:
    request = urllib.request.Request(
        server.rstrip("/") + "/",
        data=json.dumps({"text": text}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            payload = json.loads(response.read())
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"request to {server} failed: {exc}") from exc
    if not isinstance(payload.get("spans"), list):
        raise TypeError(f"{server} returned no span list")
    return payload["spans"]


def found(case: dict, returned: list[dict], start: int, end: int) -> bool:
    overlapping = [
        s for s in returned if s.get("start", 0) < end and start < s.get("end", 0)
    ]
    if case["expected"] is None:
        return bool(overlapping)
    return any(s.get("label") == case["expected"] for s in overlapping)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--server", action="append", required=True, metavar="NAME=URL")
    parser.add_argument(
        "--trials", type=int, default=20, help="random values per case (default 20)"
    )
    args = parser.parse_args()

    servers = {}
    for spec in args.server:
        name, separator, url = spec.partition("=")
        if not separator or not name or not url or name in servers:
            print(
                f"planted setup error: server must be a unique NAME=URL: {spec}",
                file=sys.stderr,
            )
            return 2
        servers[name] = url
    try:
        cases = load_cases(args.fixture)
    except (OSError, KeyError, ValueError) as exc:
        print(f"planted setup error: {exc}", file=sys.stderr)
        return 2

    counts = {(name, case["id"]): 0 for name in servers for case in cases}
    try:
        for case, text, start, end in plants(cases, args.trials):
            for name, url in servers.items():
                counts[(name, case["id"])] += found(case, spans(url, text), start, end)
    except (RuntimeError, TypeError) as exc:
        print(f"planted run error: {exc}", file=sys.stderr)
        return 2

    print(
        f"trials per case: {args.trials}. Positives: higher is better. Controls: lower is better."
    )
    print("case | gated | " + " | ".join(servers) + " | note")
    failed = []
    for case in cases:
        row = [counts[(name, case["id"])] for name in servers]
        print(
            f"{case['id']} | {'yes' if case['gated'] else 'no'} | "
            + " | ".join(map(str, row))
            + f" | {case['note']}"
        )
        failed += [
            f"{name}:{case['id']}"
            for name, n in zip(servers, row)
            if case["gated"] and n < args.trials
        ]
    if failed:
        print(f"gated cases with a missed plant: {', '.join(failed)}")
        return 1
    print("every gated plant was found")
    return 0


if __name__ == "__main__":
    sys.exit(main())
