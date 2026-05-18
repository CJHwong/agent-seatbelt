#!/usr/bin/env python3
"""Run hooks-opf test cases against a live pii-server.

Usage:
    ./run-tests.py [--fixture PATH] [--server URL]

Defaults:
    fixture: tests/test-cases.jsonl alongside this script
    server:  http://127.0.0.1:9123

Exit code is 0 if every case's expected label appears in the returned spans, 1 otherwise.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path


def run(fixture: Path, server: str) -> int:
    cases = [json.loads(line) for line in fixture.read_text().splitlines() if line.strip()]

    results: dict[str, dict[str, int]] = {}
    failures: list[tuple[int, str, list[str]]] = []
    passed = 0

    for i, case in enumerate(cases, 1):
        expected = case["category"]
        req = urllib.request.Request(
            server.rstrip("/") + "/",
            data=json.dumps({"text": case["prompt"]}).encode(),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=10) as r:
                resp = json.loads(r.read())
        except (urllib.error.URLError, TimeoutError) as exc:
            print(f"server error on case {i}: {exc}", file=sys.stderr)
            return 2

        labels = sorted({s["label"] for s in resp.get("spans", [])})
        bucket = results.setdefault(expected, {"pass": 0, "fail": 0})
        if expected in labels:
            passed += 1
            bucket["pass"] += 1
        else:
            bucket["fail"] += 1
            failures.append((i, expected, labels))

    total = len(cases)
    print(f"total: {passed}/{total} passed\n")
    print("by category:")
    for cat in sorted(results):
        s = results[cat]
        print(f"  {cat:20s} {s['pass']}/{s['pass'] + s['fail']}")

    if failures:
        print("\nmisses (expected label not in returned spans):")
        for i, exp, got in failures:
            print(f"  case {i:2d}  expected={exp:20s}  got={got}")

    return 0 if not failures else 1


def main() -> int:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, default=here / "test-cases.jsonl")
    parser.add_argument("--server", default="http://127.0.0.1:9123")
    args = parser.parse_args()
    return run(args.fixture, args.server)


if __name__ == "__main__":
    sys.exit(main())
