#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "huggingface_hub>=0.23,<2",
#     "numpy>=1.24,<3",
#     "onnxruntime>=1.17,<2",
#     "tokenizers>=0.15,<1",
#     "torch>=2.2,<3",
#     "transformers>=4.40,<6",
# ]
# ///
"""Measure resource use for the two local privacy filter modes."""

from __future__ import annotations

import argparse
import importlib.util
import math
import os
import resource
import sys
import time
from pathlib import Path
from typing import Callable


TEST_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TEST_DIR))
sys.path.insert(0, str(TEST_DIR.parent))

from final_holdout_cases import (  # noqa: E402  # ty: ignore[unresolved-import]
    FINAL_HOLDOUT_CASES,
    validate_final_holdout_cases,
)


Predict = Callable[[str], object]


def load_server_module():
    module_path = TEST_DIR.parent / "pii-server.py"
    module_spec = importlib.util.spec_from_file_location(
        "privacy_filter_server",
        module_path,
    )
    if module_spec is None or module_spec.loader is None:
        raise RuntimeError(f"cannot load privacy filter module: {module_path}")
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module


def percentile(values: list[float], fraction: float) -> float:
    ordered_values = sorted(values)
    if not ordered_values:
        return 0.0
    index = max(0, math.ceil(fraction * len(ordered_values)) - 1)
    return ordered_values[index]


def memory_peak_mb() -> float:
    peak_bytes = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return peak_bytes / (1024 * 1024)
    return peak_bytes / 1024


def accelerator_memory_mb(device: object) -> tuple[float, float]:
    import torch  # ty: ignore[unresolved-import]

    device_type = getattr(device, "type", str(device))
    if device_type == "mps":
        torch.mps.synchronize()
        current_bytes = torch.mps.current_allocated_memory()
        driver_bytes = torch.mps.driver_allocated_memory()
        return current_bytes / (1024 * 1024), driver_bytes / (1024 * 1024)
    if device_type == "cuda":
        torch.cuda.synchronize(device)
        current_bytes = torch.cuda.memory_allocated(device)
        reserved_bytes = torch.cuda.memory_reserved(device)
        return current_bytes / (1024 * 1024), reserved_bytes / (1024 * 1024)
    return 0.0, 0.0


def measure_predictor(
    predictor: Predict,
    cases: list[dict[str, object]],
    repetitions: int,
    device: object,
) -> dict[str, float]:
    texts = [str(case["text"]) for case in cases]
    for text in texts[:2]:
        predictor(text)

    wall_ms: list[float] = []
    cpu_ms: list[float] = []
    gpu_current_mb = 0.0
    gpu_driver_mb = 0.0
    run_started = time.perf_counter()
    cpu_started = time.process_time()
    for _ in range(repetitions):
        for text in texts:
            wall_started = time.perf_counter()
            cpu_started_case = time.process_time()
            predictor(text)
            wall_ms.append((time.perf_counter() - wall_started) * 1000)
            cpu_ms.append((time.process_time() - cpu_started_case) * 1000)
            current_mb, driver_mb = accelerator_memory_mb(device)
            gpu_current_mb = max(gpu_current_mb, current_mb)
            gpu_driver_mb = max(gpu_driver_mb, driver_mb)

    total_wall_seconds = time.perf_counter() - run_started
    total_cpu_seconds = time.process_time() - cpu_started
    return {
        "request_count": float(len(wall_ms)),
        "wall_p50_ms": percentile(wall_ms, 0.50),
        "wall_p95_ms": percentile(wall_ms, 0.95),
        "wall_p99_ms": percentile(wall_ms, 0.99),
        "cpu_p50_ms": percentile(cpu_ms, 0.50),
        "cpu_p95_ms": percentile(cpu_ms, 0.95),
        "cpu_p99_ms": percentile(cpu_ms, 0.99),
        "cpu_util_pct": total_cpu_seconds / total_wall_seconds * 100,
        "gpu_current_peak_mb": gpu_current_mb,
        "gpu_driver_peak_mb": gpu_driver_mb,
    }


def print_measurement(
    mode: str,
    device: object,
    load_wall_ms: float,
    load_cpu_ms: float,
    metrics: dict[str, float],
    usage_before: resource.struct_rusage,
) -> None:
    usage_after = resource.getrusage(resource.RUSAGE_SELF)
    user_ms = (usage_after.ru_utime - usage_before.ru_utime) * 1000
    system_ms = (usage_after.ru_stime - usage_before.ru_stime) * 1000
    print(f"mode={mode} device={device}")
    print(f"logical_cpu_count={os.cpu_count() or 0}")
    print(f"model_load_wall_ms={load_wall_ms:.1f}")
    print(f"model_load_cpu_ms={load_cpu_ms:.1f}")
    print(f"requests={int(metrics['request_count'])}")
    print(
        "wall_ms(p50/p95/p99)="
        f"{metrics['wall_p50_ms']:.3f}/"
        f"{metrics['wall_p95_ms']:.3f}/"
        f"{metrics['wall_p99_ms']:.3f}"
    )
    print(
        "process_cpu_ms(p50/p95/p99)="
        f"{metrics['cpu_p50_ms']:.3f}/"
        f"{metrics['cpu_p95_ms']:.3f}/"
        f"{metrics['cpu_p99_ms']:.3f}"
    )
    print(f"process_cpu_utilization_pct={metrics['cpu_util_pct']:.1f}")
    print(f"process_user_cpu_ms={user_ms:.1f}")
    print(f"process_system_cpu_ms={system_ms:.1f}")
    print(f"peak_process_memory_mb={memory_peak_mb():.1f}")
    print(f"peak_gpu_current_allocated_mb={metrics['gpu_current_peak_mb']:.1f}")
    print(f"peak_gpu_driver_allocated_mb={metrics['gpu_driver_peak_mb']:.1f}")
    print("gpu_utilization_pct=unavailable_from_pytorch")


def run_local(repetitions: int) -> None:
    usage_before = resource.getrusage(resource.RUSAGE_SELF)
    load_wall_started = time.perf_counter()
    load_cpu_started = time.process_time()
    server_module = load_server_module()
    model = server_module.load_selected_model("redact")
    load_wall_ms = (time.perf_counter() - load_wall_started) * 1000
    load_cpu_ms = (time.process_time() - load_cpu_started) * 1000
    metrics = measure_predictor(
        model.predict,
        FINAL_HOLDOUT_CASES,
        repetitions,
        model.device,
    )
    print_measurement(
        "redact_plus_rules",
        model.device,
        load_wall_ms,
        load_cpu_ms,
        metrics,
        usage_before,
    )


def run_openai(repetitions: int) -> None:
    usage_before = resource.getrusage(resource.RUSAGE_SELF)
    load_wall_started = time.perf_counter()
    load_cpu_started = time.process_time()
    server_module = load_server_module()
    model = server_module.load_selected_model("openai")
    load_wall_ms = (time.perf_counter() - load_wall_started) * 1000
    load_cpu_ms = (time.process_time() - load_cpu_started) * 1000
    metrics = measure_predictor(
        model.predict,
        FINAL_HOLDOUT_CASES,
        repetitions,
        object(),
    )
    print_measurement(
        "openai_privacy_filter",
        "cpu",
        load_wall_ms,
        load_cpu_ms,
        metrics,
        usage_before,
    )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("local", "openai"), required=True)
    parser.add_argument("--repetitions", type=int, default=3)
    return parser.parse_args()


def main() -> int:
    validate_final_holdout_cases()
    arguments = parse_arguments()
    if arguments.repetitions <= 0:
        raise ValueError("--repetitions must be positive")
    if arguments.mode == "local":
        run_local(arguments.repetitions)
    else:
        run_openai(arguments.repetitions)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
