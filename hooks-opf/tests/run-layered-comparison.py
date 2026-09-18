#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "classifinder-engine>=0.1,<1",
#     "detect-secrets>=1.5,<2",
#     "huggingface_hub>=0.23,<2",
#     "numpy>=1.24,<3",
#     "onnxruntime>=1.17,<2",
#     "tokenizers>=0.15,<1",
#     "torch>=2.2,<3",
#     "transformers>=4.40,<6",
# ]
# ///
"""Compare raw and layered local privacy detectors on the challenge corpus."""

from __future__ import annotations

import argparse
import importlib.util
import math
import os
import resource
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, cast

from classifinder_engine import (  # ty: ignore[unresolved-import]
    scan as classifinder_scan,
)
from detect_secrets.core.scan import scan_line  # ty: ignore[unresolved-import]
from detect_secrets.settings import default_settings  # ty: ignore[unresolved-import]
import numpy as np  # ty: ignore[unresolved-import]


TEST_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TEST_DIR))
sys.path.insert(0, str(TEST_DIR.parent))

from challenge_cases import CASES, validate_cases  # noqa: E402  # ty: ignore[unresolved-import]
from final_holdout_cases import (  # noqa: E402  # ty: ignore[unresolved-import]
    FINAL_HOLDOUT_CASES,
    validate_final_holdout_cases,
)
from holdout_cases import HOLDOUT_CASES, validate_holdout_cases  # noqa: E402  # ty: ignore[unresolved-import]
from redact_server import (  # noqa: E402  # ty: ignore[unresolved-import]
    RedactModel,
    deterministic_spans,
    merge_spans,
    viterbi_decode,
)


CaseRecord = dict[str, object]
Span = dict[str, object]
Labels = set[str]


def load_openai_model():
    module_path = TEST_DIR.parent / "pii-server.py"
    module_spec = importlib.util.spec_from_file_location(
        "openai_filter_server", module_path
    )
    if module_spec is None or module_spec.loader is None:
        raise RuntimeError(f"cannot load OpenAI filter module: {module_path}")
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module.Model(Path.home() / ".cache" / "of")


def raw_redact_spans(model: RedactModel, text: str) -> list[Span]:
    if not text:
        return []
    encoding = model.tokenizer.encode(text, add_special_tokens=False)
    if not encoding.ids:
        return []
    token_probabilities = model._predict_token_probabilities(encoding.ids)
    log_probabilities = np.log(np.clip(token_probabilities, 1e-7, 1.0))
    path = viterbi_decode(
        log_probabilities,
        model.start,
        model.end,
        model.transition,
    )
    return model._path_to_spans(
        path,
        token_probabilities,
        encoding.offsets,
        text,
    )


def span_labels(spans: list[Span]) -> Labels:
    return {str(span["label"]) for span in spans}


def scan_detect_secrets(text: str) -> list[Any]:
    lines = text.splitlines() or [text]
    findings: list[Any] = []
    for line in lines:
        findings.extend(scan_line(line))
    return findings


def classifinder_labels(findings: list[Any]) -> Labels:
    labels: Labels = set()
    for finding in findings:
        if str(getattr(finding, "type", "")).lower() == "credit_card_number":
            labels.add("account_number")
            continue
        labels.add("secret")
    return labels


def secret_detector_labels(findings: list[Any]) -> Labels:
    return {"secret"} if findings else set()


def percentile(values: list[float], fraction: float) -> float:
    ordered_values = sorted(values)
    if not ordered_values:
        return 0.0
    index = max(0, math.ceil(fraction * len(ordered_values)) - 1)
    return ordered_values[index]


def format_rate(passed: int, total: int) -> str:
    if total == 0:
        return "0/0"
    return f"{passed}/{total} ({passed / total:.1%})"


def _expected_labels(case_record: CaseRecord) -> list[str]:
    """The labels a case expects, narrowed out of the record's object values.

    A case always carries a list here, but a record's values are typed as
    object, so the narrowing happens once here rather than at each use.
    """
    expected = case_record["expected"]
    if not isinstance(expected, list):
        return []
    return [str(label) for label in expected]


def measure_layer(
    layer_name: str,
    layer_labels: dict[str, Labels],
    cases: list[CaseRecord],
    timings: list[float],
) -> dict[str, object]:
    positive_cases = [case for case in cases if case["class"] == "positive"]
    secret_cases = [
        case for case in positive_cases if "secret" in set(_expected_labels(case))
    ]
    robustness_cases = [case for case in cases if case["class"] == "robustness"]
    clean_cases = [case for case in cases if case["class"] == "clean"]
    ambiguous_cases = [case for case in cases if case["class"] == "ambiguous"]

    def matches(case: CaseRecord) -> bool:
        expected_labels = set(_expected_labels(case))
        return expected_labels.issubset(layer_labels[str(case["id"])])

    positive_misses = [case for case in positive_cases if not matches(case)]
    secret_misses = [case for case in secret_cases if not matches(case)]
    robustness_misses = [case for case in robustness_cases if not matches(case)]
    clean_false_positives = [
        case for case in clean_cases if layer_labels[str(case["id"])]
    ]
    secret_clean_false_positives = [
        case for case in clean_cases if "secret" in layer_labels[str(case["id"])]
    ]
    ambiguous_flags = [
        case for case in ambiguous_cases if layer_labels[str(case["id"])]
    ]
    secret_ambiguous_flags = [
        case for case in ambiguous_cases if "secret" in layer_labels[str(case["id"])]
    ]
    return {
        "name": layer_name,
        "positive_passed": len(positive_cases) - len(positive_misses),
        "positive_total": len(positive_cases),
        "secret_passed": len(secret_cases) - len(secret_misses),
        "secret_total": len(secret_cases),
        "robustness_passed": len(robustness_cases) - len(robustness_misses),
        "robustness_total": len(robustness_cases),
        "clean_false_positives": clean_false_positives,
        "secret_clean_false_positives": secret_clean_false_positives,
        "ambiguous_flags": ambiguous_flags,
        "secret_ambiguous_flags": secret_ambiguous_flags,
        "positive_misses": positive_misses,
        "secret_misses": secret_misses,
        "robustness_misses": robustness_misses,
        "clean_total": len(clean_cases),
        "ambiguous_total": len(ambiguous_cases),
        "p50_ms": percentile(timings, 0.50),
        "p95_ms": percentile(timings, 0.95),
        "p99_ms": percentile(timings, 0.99),
        "max_ms": max(timings) if timings else 0.0,
    }


def print_layer_summary(summary: dict[str, object]) -> None:
    # measure_layer fixes these shapes, but a record's values are typed as
    # object, so each is narrowed once here rather than at each use.
    false_positives = cast("list[CaseRecord]", summary["clean_false_positives"])
    ambiguous_flags = cast("list[CaseRecord]", summary["ambiguous_flags"])
    secret_clean_false_positives = cast(
        "list[CaseRecord]", summary["secret_clean_false_positives"]
    )
    secret_ambiguous_flags = cast("list[CaseRecord]", summary["secret_ambiguous_flags"])
    positive_passed = cast(int, summary["positive_passed"])
    positive_total = cast(int, summary["positive_total"])
    secret_passed = cast(int, summary["secret_passed"])
    secret_total = cast(int, summary["secret_total"])
    robustness_passed = cast(int, summary["robustness_passed"])
    robustness_total = cast(int, summary["robustness_total"])
    print(
        f"{summary['name']}: "
        f"positive={format_rate(positive_passed, positive_total)} "
        f"secret={format_rate(secret_passed, secret_total)} "
        f"robustness={format_rate(robustness_passed, robustness_total)} "
        f"clean_fp={len(false_positives)}/{summary['clean_total']} "
        f"ambiguous={len(ambiguous_flags)}/{summary['ambiguous_total']} "
        f"secret_clean_fp={len(secret_clean_false_positives)} "
        f"secret_ambiguous={len(secret_ambiguous_flags)} "
        f"latency_ms(p50/p95/p99)={summary['p50_ms']:.3f}/"
        f"{summary['p95_ms']:.3f}/{summary['p99_ms']:.3f}"
    )


def print_case_ids(prefix: str, cases: list[CaseRecord]) -> None:
    if cases:
        print(f"  {prefix}: {', '.join(str(case['id']) for case in cases)}")


def print_detailed_summary(summary: dict[str, object]) -> None:
    # measure_layer fixes these shapes, but a record's values are typed as
    # object, so each list is narrowed to the type print_case_ids takes.
    print_case_ids("secret misses", cast("list[CaseRecord]", summary["secret_misses"]))
    print_case_ids(
        "clean false positives",
        cast("list[CaseRecord]", summary["clean_false_positives"]),
    )
    print_case_ids(
        "secret clean false positives",
        cast("list[CaseRecord]", summary["secret_clean_false_positives"]),
    )
    print_case_ids(
        "ambiguous flags", cast("list[CaseRecord]", summary["ambiguous_flags"])
    )
    print_case_ids(
        "secret ambiguous flags",
        cast("list[CaseRecord]", summary["secret_ambiguous_flags"]),
    )
    print_case_ids(
        "robustness misses", cast("list[CaseRecord]", summary["robustness_misses"])
    )


def local_layers(
    redact_model: RedactModel,
    cases: list[CaseRecord],
) -> tuple[dict[str, dict[str, Labels]], dict[str, list[float]], dict[str, int]]:
    layer_labels: dict[str, dict[str, Labels]] = defaultdict(dict)
    layer_timings: dict[str, list[float]] = defaultdict(list)
    target_hits: dict[str, int] = defaultdict(int)

    with default_settings() as settings:
        settings.disable_plugins("Base64HighEntropyString", "HexHighEntropyString")
        redact_model.predict("warmup")
        scan_detect_secrets("warmup")
        classifinder_scan("warmup")

        for case in cases:
            case_id = str(case["id"])
            text = str(case["text"])

            raw_started = time.perf_counter()
            model_spans = raw_redact_spans(redact_model, text)
            raw_elapsed = (time.perf_counter() - raw_started) * 1000

            rules_started = time.perf_counter()
            rule_spans = deterministic_spans(text)
            rules_elapsed = (time.perf_counter() - rules_started) * 1000

            wrapper_spans = merge_spans(text, rule_spans + model_spans)

            secrets_started = time.perf_counter()
            detect_findings = scan_detect_secrets(text)
            secrets_elapsed = (time.perf_counter() - secrets_started) * 1000
            detect_labels = secret_detector_labels(detect_findings)

            classifinder_started = time.perf_counter()
            classifinder_findings = classifinder_scan(text)
            classifinder_elapsed = (time.perf_counter() - classifinder_started) * 1000
            classifinder_result = classifinder_labels(classifinder_findings)

            model_result = span_labels(model_spans)
            rules_result = span_labels(rule_spans)
            wrapper_result = span_labels(wrapper_spans)
            layer_results = {
                "raw_redact": model_result,
                "redact_rules": rules_result,
                "redact_wrapper": wrapper_result,
                "classifinder": classifinder_result,
                "detect_secrets": detect_labels,
                "redact_plus_classifinder": wrapper_result | classifinder_result,
                "redact_plus_detect_secrets": wrapper_result | detect_labels,
                "redact_plus_both": wrapper_result
                | detect_labels
                | classifinder_result,
            }
            for layer_name, labels in layer_results.items():
                layer_labels[layer_name][case_id] = labels

            layer_timings["raw_redact"].append(raw_elapsed)
            layer_timings["redact_rules"].append(rules_elapsed)
            layer_timings["redact_wrapper"].append(raw_elapsed + rules_elapsed)
            layer_timings["classifinder"].append(classifinder_elapsed)
            layer_timings["detect_secrets"].append(secrets_elapsed)
            layer_timings["redact_plus_classifinder"].append(
                raw_elapsed + rules_elapsed + classifinder_elapsed
            )
            layer_timings["redact_plus_detect_secrets"].append(
                raw_elapsed + rules_elapsed + secrets_elapsed
            )
            layer_timings["redact_plus_both"].append(
                raw_elapsed + rules_elapsed + secrets_elapsed + classifinder_elapsed
            )

            target_text = case.get("target")
            if isinstance(target_text, str) and target_text:
                for layer_name, spans in (
                    ("redact_rules", rule_spans),
                    ("redact_wrapper", wrapper_spans),
                ):
                    if any(target_text in str(span.get("text", "")) for span in spans):
                        target_hits[layer_name] += 1

    return layer_labels, layer_timings, target_hits


OPENAI_LAYER_NAMES = (
    "openai_privacy_filter",
    "openai_plus_redact_rules",
    "openai_plus_classifinder",
    "openai_plus_detect_secrets",
    "openai_plus_all",
)


def openai_layers(
    openai_model: Any,
    cases: list[CaseRecord],
) -> tuple[dict[str, dict[str, Labels]], dict[str, list[float]]]:
    layer_labels: dict[str, dict[str, Labels]] = defaultdict(dict)
    layer_timings: dict[str, list[float]] = defaultdict(list)
    with default_settings() as settings:
        settings.disable_plugins("Base64HighEntropyString", "HexHighEntropyString")
        openai_model.predict("warmup")
        scan_detect_secrets("warmup")
        classifinder_scan("warmup")
        for case in cases:
            case_id = str(case["id"])
            text = str(case["text"])

            openai_started = time.perf_counter()
            openai_result = span_labels(openai_model.predict(text))
            openai_elapsed = (time.perf_counter() - openai_started) * 1000

            rules_started = time.perf_counter()
            rule_result = span_labels(deterministic_spans(text))
            rules_elapsed = (time.perf_counter() - rules_started) * 1000

            secrets_started = time.perf_counter()
            detect_result = secret_detector_labels(scan_detect_secrets(text))
            secrets_elapsed = (time.perf_counter() - secrets_started) * 1000

            classifinder_started = time.perf_counter()
            classifinder_result = classifinder_labels(classifinder_scan(text))
            classifinder_elapsed = (time.perf_counter() - classifinder_started) * 1000

            layer_results = {
                "openai_privacy_filter": openai_result,
                "openai_plus_redact_rules": openai_result | rule_result,
                "openai_plus_classifinder": openai_result | classifinder_result,
                "openai_plus_detect_secrets": openai_result | detect_result,
                "openai_plus_all": openai_result
                | rule_result
                | classifinder_result
                | detect_result,
            }
            for layer_name, labels in layer_results.items():
                layer_labels[layer_name][case_id] = labels

            layer_timings["openai_privacy_filter"].append(openai_elapsed)
            layer_timings["openai_plus_redact_rules"].append(
                openai_elapsed + rules_elapsed
            )
            layer_timings["openai_plus_classifinder"].append(
                openai_elapsed + classifinder_elapsed
            )
            layer_timings["openai_plus_detect_secrets"].append(
                openai_elapsed + secrets_elapsed
            )
            layer_timings["openai_plus_all"].append(
                openai_elapsed + rules_elapsed + classifinder_elapsed + secrets_elapsed
            )
    return layer_labels, layer_timings


LAYER_NAMES = (
    "raw_redact",
    "redact_rules",
    "redact_wrapper",
    "classifinder",
    "detect_secrets",
    "redact_plus_classifinder",
    "redact_plus_detect_secrets",
    "redact_plus_both",
)


def print_layer_summaries(
    layer_labels: dict[str, dict[str, Labels]],
    layer_timings: dict[str, list[float]],
    cases: list[CaseRecord],
) -> None:
    for layer_name in LAYER_NAMES:
        summary = measure_layer(
            layer_name,
            layer_labels[layer_name],
            cases,
            layer_timings[layer_name],
        )
        print_layer_summary(summary)
        if layer_name in {
            "redact_wrapper",
            "redact_plus_detect_secrets",
            "redact_plus_both",
        }:
            print_detailed_summary(summary)


def print_openai_layer_summaries(
    layer_labels: dict[str, dict[str, Labels]],
    layer_timings: dict[str, list[float]],
    cases: list[CaseRecord],
) -> None:
    for layer_name in OPENAI_LAYER_NAMES:
        summary = measure_layer(
            layer_name,
            layer_labels[layer_name],
            cases,
            layer_timings[layer_name],
        )
        print_layer_summary(summary)
        if layer_name in {"openai_privacy_filter", "openai_plus_all"}:
            print_detailed_summary(summary)


def tail_timings(
    layer_timings: dict[str, list[float]],
    count: int,
) -> dict[str, list[float]]:
    return {
        layer_name: timings[-count:] for layer_name, timings in layer_timings.items()
    }


def run_local(
    min_score: float,
    cases: list[CaseRecord],
    extra_cases: list[CaseRecord],
    extra_name: str | None,
) -> None:
    os.environ["REDACT_MIN_SCORE"] = str(min_score)
    load_started = time.perf_counter()
    redact_model = RedactModel(Path.home() / ".cache" / "redact", "mps")
    load_elapsed = (time.perf_counter() - load_started) * 1000
    layer_labels, layer_timings, target_hits = local_layers(redact_model, cases)
    print(f"mode=local device={redact_model.device} model_load_ms={load_elapsed:.1f}")
    print(
        f"peak_process_memory_mb={resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024):.1f}"
    )
    print("corpus=all")
    print_layer_summaries(layer_labels, layer_timings, cases)
    if extra_cases and extra_name:
        print(f"corpus={extra_name}")
        print_layer_summaries(
            layer_labels,
            tail_timings(layer_timings, len(extra_cases)),
            extra_cases,
        )
    print(f"secret_target_containment={dict(target_hits)}")


def run_openai(
    cases: list[CaseRecord],
    extra_cases: list[CaseRecord],
    extra_name: str | None,
) -> None:
    load_started = time.perf_counter()
    openai_model = load_openai_model()
    load_elapsed = (time.perf_counter() - load_started) * 1000
    layer_labels, layer_timings = openai_layers(openai_model, cases)
    print(f"mode=openai model_load_ms={load_elapsed:.1f}")
    print(
        f"peak_process_memory_mb={resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024):.1f}"
    )
    print("corpus=all")
    print_openai_layer_summaries(layer_labels, layer_timings, cases)
    if extra_cases and extra_name:
        print(f"corpus={extra_name}")
        print_openai_layer_summaries(
            layer_labels,
            tail_timings(layer_timings, len(extra_cases)),
            extra_cases,
        )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("local", "openai"),
        required=True,
        help="run local layers or the OpenAI Privacy Filter",
    )
    parser.add_argument(
        "--redact-min-score",
        type=float,
        default=0.90,
        help="Redact model threshold for local mode",
    )
    corpus_group = parser.add_mutually_exclusive_group()
    corpus_group.add_argument(
        "--include-holdout",
        action="store_true",
        help="include the fresh holdout corpus in the run",
    )
    corpus_group.add_argument(
        "--include-final-holdout",
        action="store_true",
        help="include the untouched final holdout corpus in the run",
    )
    return parser.parse_args()


def main() -> int:
    validate_cases()
    validate_holdout_cases()
    validate_final_holdout_cases()
    arguments = parse_arguments()
    if arguments.include_final_holdout:
        extra_cases = FINAL_HOLDOUT_CASES
        extra_name = "final_holdout"
    elif arguments.include_holdout:
        extra_cases = HOLDOUT_CASES
        extra_name = "holdout"
    else:
        extra_cases = []
        extra_name = None
    cases = CASES + extra_cases
    if arguments.mode == "local":
        run_local(arguments.redact_min_score, cases, extra_cases, extra_name)
    else:
        run_openai(cases, extra_cases, extra_name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
