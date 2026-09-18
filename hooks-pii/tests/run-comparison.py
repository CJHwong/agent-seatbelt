#!/usr/bin/env python3
"""Compare privacy-filter servers against the expanded false-positive corpus."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import cast

DEFAULT_FIXTURE = Path(__file__).with_name("false-positive-cases.jsonl")


def load_cases(fixture_path: Path) -> list[dict[str, object]]:
    cases: list[dict[str, object]] = []
    for line_number, line in enumerate(fixture_path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            case_record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON on line {line_number}: {exc}") from exc
        if not isinstance(case_record, dict):
            raise TypeError(f"case on line {line_number} is not an object")
        _validate_case(case_record, line_number)
        cases.append(case_record)
    if not cases:
        raise ValueError(f"fixture is empty: {fixture_path}")
    return cases


def _validate_case(case_record: dict[str, object], line_number: int) -> None:
    required_fields = {"id", "class", "group", "expected", "text"}
    missing_fields = required_fields.difference(case_record)
    if missing_fields:
        missing_text = ", ".join(sorted(missing_fields))
        raise ValueError(f"case on line {line_number} misses: {missing_text}")
    if case_record["class"] not in {"clean", "positive", "ambiguous"}:
        raise ValueError(f"case on line {line_number} has an invalid class")
    if not isinstance(case_record["expected"], list):
        raise TypeError(f"case on line {line_number} has a non-list expected value")
    if not isinstance(case_record["text"], str):
        raise TypeError(f"case on line {line_number} has non-string text")


def parse_server_specs(server_specs: list[str]) -> dict[str, str]:
    servers: dict[str, str] = {}
    for specification in server_specs:
        name, separator, url = specification.partition("=")
        if not separator or not name or not url:
            raise ValueError(f"server must use NAME=URL: {specification}")
        if name in servers:
            raise ValueError(f"duplicate server name: {name}")
        servers[name] = url
    if not servers:
        raise ValueError("at least one --server NAME=URL is required")
    return servers


def parse_unsupported_specs(unsupported_specs: list[str]) -> dict[str, set[str]]:
    unsupported: dict[str, set[str]] = {}
    for specification in unsupported_specs:
        name, separator, labels_text = specification.partition("=")
        labels = {label.strip() for label in labels_text.split(",") if label.strip()}
        if not separator or not name or not labels:
            raise ValueError(
                f"unsupported labels must use NAME=LABEL,...: {specification}"
            )
        if name in unsupported:
            raise ValueError(f"duplicate unsupported-label name: {name}")
        unsupported[name] = labels
    return unsupported


def query_server(
    server_url: str, text: str
) -> tuple[set[str], list[dict[str, object]], float]:
    request = urllib.request.Request(
        server_url.rstrip("/") + "/",
        data=json.dumps({"text": text}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    started_at = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            payload = json.loads(response.read())
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"request failed: {exc}") from exc
    elapsed_ms = (time.perf_counter() - started_at) * 1000
    returned_spans = payload.get("spans")
    if not isinstance(returned_spans, list):
        raise TypeError("server response has no span list")
    spans = [span for span in returned_spans if isinstance(span, dict)]
    labels = {str(span.get("label")) for span in spans if span.get("label")}
    return labels, spans, elapsed_ms


def _expected_labels(case_record: dict[str, object]) -> list[str]:
    """The labels a case expects, narrowed out of the record's object values.

    Every case carries a list here, which `load_cases` checks before returning,
    but a record's values are typed as object, so the narrowing happens once
    here rather than at each use.
    """
    expected = case_record["expected"]
    if not isinstance(expected, list):
        return []
    return [str(label) for label in expected]


def compare_server(
    server_name: str,
    server_url: str,
    cases: list[dict[str, object]],
    unsupported_labels: set[str],
) -> dict[str, object]:
    clean_cases = [case for case in cases if case["class"] == "clean"]
    positive_cases = [
        case
        for case in cases
        if case["class"] == "positive"
        and (set(_expected_labels(case)) - unsupported_labels)
    ]
    ambiguous_cases = [case for case in cases if case["class"] == "ambiguous"]
    false_positives: list[tuple[str, str, list[str], list[str]]] = []
    missing_detections: list[tuple[str, str, list[str], list[str]]] = []
    ambiguous_flags: list[tuple[str, str, list[str]]] = []
    latencies: list[float] = []

    for case_record in cases:
        labels, spans, elapsed_ms = query_server(server_url, str(case_record["text"]))
        latencies.append(elapsed_ms)
        case_type = str(case_record["class"])
        case_id = str(case_record["id"])
        case_group = str(case_record["group"])
        span_descriptions = _span_descriptions(spans)
        if case_type == "clean" and labels:
            false_positives.append(
                (case_id, case_group, sorted(labels), span_descriptions)
            )
        if case_type == "positive":
            expected_labels = set(_expected_labels(case_record)) - unsupported_labels
            if not expected_labels:
                continue
            missing_labels = sorted(expected_labels.difference(labels))
            if missing_labels:
                missing_detections.append(
                    (case_id, case_group, missing_labels, sorted(labels))
                )
        if case_type == "ambiguous" and labels:
            ambiguous_flags.append((case_id, case_group, sorted(labels)))

    return {
        "name": server_name,
        "clean_total": len(clean_cases),
        "positive_total": len(positive_cases),
        "ambiguous_total": len(ambiguous_cases),
        "false_positives": false_positives,
        "missing_detections": missing_detections,
        "ambiguous_flags": ambiguous_flags,
        "latencies": latencies,
    }


def _span_descriptions(spans: list[dict[str, object]]) -> list[str]:
    descriptions = []
    for span in spans:
        label = str(span.get("label", "unknown"))
        span_text = str(span.get("text", ""))
        descriptions.append(f"{label}:{span_text}")
    return descriptions


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, math.ceil(fraction * len(ordered)) - 1)
    return ordered[index]


def rate_text(passed: int, total: int) -> str:
    if total == 0:
        return "0/0 (n/a)"
    return f"{passed}/{total} ({passed / total:.1%})"


def print_summary(result: dict[str, object]) -> None:
    # compare_server fixes these shapes, but a record's values are typed as
    # object, so each is narrowed once here rather than at each use.
    false_positives = cast(
        "list[tuple[str, str, list[str], list[str]]]", result["false_positives"]
    )
    missing_detections = cast(
        "list[tuple[str, str, list[str], list[str]]]", result["missing_detections"]
    )
    ambiguous_flags = cast(
        "list[tuple[str, str, list[str]]]", result["ambiguous_flags"]
    )
    latencies = cast("list[float]", result["latencies"])
    clean_total = cast(int, result["clean_total"])
    positive_total = cast(int, result["positive_total"])
    print(
        f"{result['name']}: "
        f"clean={rate_text(clean_total - len(false_positives), clean_total)} "
        f"positive={rate_text(positive_total - len(missing_detections), positive_total)} "
        f"ambiguous_flagged={len(ambiguous_flags)}/{result['ambiguous_total']} "
        f"latency_ms(p50/p95)={percentile(latencies, 0.50):.1f}/"
        f"{percentile(latencies, 0.95):.1f}"
    )
    for case_id, case_group, labels, spans in false_positives:
        print(
            f"  false-positive {case_id} [{case_group}]: labels={labels} spans={spans}"
        )
    for case_id, case_group, missing, returned in missing_detections:
        print(
            f"  miss {case_id} [{case_group}]: expected={missing} returned={returned}"
        )
    if ambiguous_flags:
        flagged_ids = ", ".join(
            f"{case_id}[{case_group}]" for case_id, case_group, _ in ambiguous_flags
        )
        print(f"  ambiguous flags: {flagged_ids}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument(
        "--server",
        action="append",
        required=True,
        metavar="NAME=URL",
        help="repeat for each server, such as redact=http://127.0.0.1:9124",
    )
    parser.add_argument(
        "--unsupported",
        action="append",
        default=[],
        metavar="NAME=LABEL,...",
        help="exclude documented unsupported positive labels for a server",
    )
    args = parser.parse_args()

    try:
        cases = load_cases(args.fixture)
        servers = parse_server_specs(args.server)
        unsupported = parse_unsupported_specs(args.unsupported)
    except (OSError, TypeError, ValueError) as exc:
        print(f"comparison setup error: {exc}", file=sys.stderr)
        return 2

    print(f"cases: {len(cases)}")
    print(
        "classes: "
        f"clean={sum(case['class'] == 'clean' for case in cases)} "
        f"positive={sum(case['class'] == 'positive' for case in cases)} "
        f"ambiguous={sum(case['class'] == 'ambiguous' for case in cases)}"
    )
    comparison_failed = False
    for server_name, server_url in servers.items():
        try:
            result = compare_server(
                server_name,
                server_url,
                cases,
                unsupported.get(server_name, set()),
            )
        except (RuntimeError, TypeError) as exc:
            print(f"{server_name}: comparison error: {exc}", file=sys.stderr)
            comparison_failed = True
            continue
        print_summary(result)
        comparison_failed |= bool(
            result["false_positives"] or result["missing_detections"]
        )
    return 1 if comparison_failed else 0


if __name__ == "__main__":
    sys.exit(main())
