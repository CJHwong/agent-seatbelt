#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "huggingface_hub>=0.23,<2",
#     "numpy>=1.24,<3",
#     "tokenizers>=0.15,<1",
#     "torch>=2.2,<3",
#     "transformers>=4.40,<6",
# ]
# ///
"""GPU-backed local PII server for Desert Ant Redact."""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, cast

import numpy as np  # ty: ignore[unresolved-import]
import torch  # ty: ignore[unresolved-import]
from tokenizers import Tokenizer  # ty: ignore[unresolved-import]
from transformers import (  # ty: ignore[unresolved-import]
    BertConfig,
    BertForTokenClassification,
)

from pii_rules import deterministic_spans, merge_spans

REDACT_REPO = os.environ.get("REDACT_REPO", "desert-ant-labs/redact")
REDACT_REVISION = os.environ.get("REDACT_REVISION", "v0.4.0")
CACHE_DIR = Path(os.environ.get("REDACT_CACHE_DIR", Path.home() / ".cache" / "redact"))
MAX_SEQUENCE_LENGTH = 256
CONTENT_WINDOW_LENGTH = MAX_SEQUENCE_LENGTH - 2
WINDOW_STRIDE = 64
NEG_INF = -1e9
DEFAULT_MAX_TOKENS = 4096
DEFAULT_CHUNK_OVERLAP_TOKENS = 128
DEFAULT_MAX_INPUT_TOKENS = 32768


class InputTooLargeError(ValueError):
    """Raised when a request exceeds REDACT_MAX_INPUT_TOKENS.

    Subclasses ValueError so a caller that already maps ValueError to 413 keeps
    its behavior.
    """


REQUIRED_FILES = [
    "config.json",
    "labels.json",
    "tokenizer.json",
    "redact.pt",
]

MODEL_LABEL_MAP = {
    "GIVEN_NAME": "private_person",
    "SURNAME": "private_person",
    "STREET_NAME": "private_address",
    "BUILDING_NUMBER": "private_address",
    "SECONDARY_ADDRESS": "private_address",
    "CITY": "private_address",
    "STATE": "private_address",
    "ZIP_CODE": "private_address",
    "EMAIL": "private_email",
    "PHONE": "private_phone",
    "URL": "private_url",
    "IP_ADDRESS": "private_url",
    "CREDIT_CARD": "account_number",
    "BANK_ACCOUNT": "account_number",
    "ROUTING_NUMBER": "account_number",
    "GOVERNMENT_ID": "account_number",
    "PASSPORT": "account_number",
    "DRIVERS_LICENSE": "account_number",
    "TAX_ID": "account_number",
    "SSN": "account_number",
}


def choose_device(requested: str | None = None) -> torch.device:
    """Select an accelerator and reject implicit CPU fallback."""
    choice = (requested or os.environ.get("REDACT_DEVICE", "auto")).strip().lower()
    if choice == "cpu":
        return torch.device("cpu")
    if choice == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("requested accelerator cuda is not available")
        return torch.device("cuda")
    if choice == "mps":
        if not _mps_is_available():
            raise RuntimeError("requested accelerator mps is not available")
        _reject_mps_cpu_fallback()
        return torch.device("mps")
    if choice == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if _mps_is_available():
            _reject_mps_cpu_fallback()
            return torch.device("mps")
        raise RuntimeError(
            "no GPU accelerator is available; set REDACT_DEVICE=cpu to use CPU explicitly"
        )
    raise ValueError("REDACT_DEVICE must be auto, cuda, mps, or cpu")


def _mps_is_available() -> bool:
    return hasattr(torch.backends, "mps") and torch.backends.mps.is_available()


def _reject_mps_cpu_fallback() -> None:
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1":
        raise RuntimeError(
            "PYTORCH_ENABLE_MPS_FALLBACK=1 would hide CPU execution; unset it"
        )


def ensure_assets() -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    missing_files = [
        filename for filename in REQUIRED_FILES if not (CACHE_DIR / filename).is_file()
    ]
    if missing_files:
        missing = ", ".join(missing_files)
        raise FileNotFoundError(
            "Redact requires a compatible PyTorch cache. "
            f"Missing: {missing}. "
            "The public Redact release does not publish redact.pt. "
            f"Populate {CACHE_DIR} or set REDACT_CACHE_DIR to a converted cache. "
            f"Configured repository: {REDACT_REPO}@{REDACT_REVISION}."
        )
    return CACHE_DIR


class LabelSpace:
    """BIOES labels from the Redact configuration."""

    def __init__(self, id2label: dict[int, str]):
        self.id2label = id2label
        self.num_classes = len(id2label)
        self.bg = next(label_id for label_id, name in id2label.items() if name == "O")
        self.tag: dict[int, str | None] = {}
        self.span_name: dict[int, str | None] = {}
        for label_id, name in id2label.items():
            if name == "O":
                self.tag[label_id] = None
                self.span_name[label_id] = None
                continue
            tag, _, span_name = name.partition("-")
            self.tag[label_id] = tag
            self.span_name[label_id] = span_name


def build_transition_tables(
    label_space: LabelSpace,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    class_count = label_space.num_classes
    start = np.full(class_count, NEG_INF, dtype=np.float32)
    end = np.full(class_count, NEG_INF, dtype=np.float32)
    transition = np.full((class_count, class_count), NEG_INF, dtype=np.float32)

    for label_id in range(class_count):
        tag = label_space.tag[label_id]
        if tag in {"B", "S"} or label_id == label_space.bg:
            start[label_id] = 0.0
        if tag in {"E", "S"} or label_id == label_space.bg:
            end[label_id] = 0.0

    for previous_id in range(class_count):
        for next_id in range(class_count):
            if valid_transition(label_space, previous_id, next_id):
                transition[previous_id, next_id] = 0.0
    return start, end, transition


def valid_transition(label_space: LabelSpace, previous_id: int, next_id: int) -> bool:
    previous_is_background = previous_id == label_space.bg
    next_is_background = next_id == label_space.bg
    previous_tag = label_space.tag[previous_id]
    next_tag = label_space.tag[next_id]

    if previous_is_background:
        return next_is_background or next_tag in {"B", "S"}
    if previous_tag in {"E", "S"}:
        return next_is_background or next_tag in {"B", "S"}
    if previous_tag in {"B", "I"}:
        return (
            not next_is_background
            and next_tag in {"I", "E"}
            and label_space.span_name[previous_id] == label_space.span_name[next_id]
        )
    return False


def viterbi_decode(
    log_probs: np.ndarray,
    start: np.ndarray,
    end: np.ndarray,
    transition: np.ndarray,
) -> list[int]:
    token_count, class_count = log_probs.shape
    if token_count == 0:
        return []

    scores = log_probs[0] + start
    backpointers = np.empty((token_count - 1, class_count), dtype=np.int64)
    for token_index in range(1, token_count):
        candidates = scores[:, None] + transition
        best_previous = np.argmax(candidates, axis=0)
        scores = (
            candidates[best_previous, np.arange(class_count)] + log_probs[token_index]
        )
        backpointers[token_index - 1] = best_previous

    path = np.empty(token_count, dtype=np.int64)
    path[-1] = int(np.argmax(scores + end))
    for token_index in range(token_count - 2, -1, -1):
        path[token_index] = backpointers[token_index, path[token_index + 1]]
    return path.tolist()


def labels_to_spans(
    path: list[int],
    label_space: LabelSpace,
) -> list[tuple[int, int, str]]:
    spans: list[tuple[int, int, str]] = []
    active_start: int | None = None
    active_name: str | None = None

    for token_index, label_id in enumerate(path):
        tag = label_space.tag[label_id]
        name = label_space.span_name[label_id]
        if tag == "S":
            if active_start is not None and active_name is not None:
                spans.append((active_start, token_index - 1, active_name))
            active_start = None
            active_name = None
            spans.append((token_index, token_index, name or ""))
            continue
        if tag == "B":
            if active_start is not None and active_name is not None:
                spans.append((active_start, token_index - 1, active_name))
            active_start = token_index
            active_name = name
            continue
        if tag == "I":
            if active_name != name:
                active_start = token_index
                active_name = name
            continue
        if tag == "E":
            if active_start is None:
                active_start = token_index
                active_name = name
            spans.append((active_start, token_index, active_name or name or ""))
            active_start = None
            active_name = None
            continue
        if active_start is not None and active_name is not None:
            spans.append((active_start, token_index - 1, active_name))
        active_start = None
        active_name = None

    if active_start is not None and active_name is not None:
        spans.append((active_start, len(path) - 1, active_name))
    return spans


def map_model_label(source_label: str) -> str | None:
    return MODEL_LABEL_MAP.get(source_label)


def chunk_token_ranges(
    token_count: int,
    chunk_size: int,
    overlap: int,
) -> list[tuple[int, int]]:
    if token_count < 0:
        raise ValueError("token_count must not be negative")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be non-negative and smaller than chunk_size")
    if token_count == 0:
        return []

    step = chunk_size - overlap
    ranges: list[tuple[int, int]] = []
    start = 0
    while start < token_count:
        end = min(start + chunk_size, token_count)
        ranges.append((start, end))
        if end == token_count:
            break
        start += step
    return ranges


def _window_ranges(token_count: int) -> list[tuple[int, int]]:
    if token_count <= 0:
        return []
    if WINDOW_STRIDE >= CONTENT_WINDOW_LENGTH:
        raise RuntimeError("WINDOW_STRIDE must be less than the content window length")

    ranges: list[tuple[int, int]] = []
    start = 0
    while start < token_count:
        end = min(start + CONTENT_WINDOW_LENGTH, token_count)
        ranges.append((start, end))
        if end == token_count:
            break
        start = end - WINDOW_STRIDE
    return ranges


class RedactModel:
    def __init__(self, cache_dir: Path, requested_device: str | None = None):
        config_path = cache_dir / "config.json"
        config_data = json.loads(config_path.read_text())
        id2label = {
            int(label_id): name for label_id, name in config_data["id2label"].items()
        }
        self.label_space = LabelSpace(id2label)
        self.start, self.end, self.transition = build_transition_tables(
            self.label_space
        )
        self.device = choose_device(requested_device)
        self.min_score = float(os.environ.get("REDACT_MIN_SCORE", "0.6"))
        self.batch_size = int(os.environ.get("REDACT_BATCH_SIZE", "8"))
        self.max_tokens = int(
            os.environ.get("REDACT_MAX_TOKENS", str(DEFAULT_MAX_TOKENS))
        )
        self.chunk_overlap = int(
            os.environ.get(
                "REDACT_CHUNK_OVERLAP_TOKENS",
                str(DEFAULT_CHUNK_OVERLAP_TOKENS),
            )
        )
        self.max_input_tokens = int(
            os.environ.get("REDACT_MAX_INPUT_TOKENS", str(DEFAULT_MAX_INPUT_TOKENS))
        )
        if not 0.0 <= self.min_score <= 1.0:
            raise ValueError("REDACT_MIN_SCORE must be between 0 and 1")
        if self.batch_size <= 0 or self.max_tokens <= 0:
            raise ValueError("REDACT_BATCH_SIZE and REDACT_MAX_TOKENS must be positive")
        if self.max_input_tokens < self.max_tokens:
            raise ValueError(
                "REDACT_MAX_INPUT_TOKENS must be at least REDACT_MAX_TOKENS"
            )
        if self.chunk_overlap < 0 or self.chunk_overlap >= self.max_tokens:
            raise ValueError(
                "REDACT_CHUNK_OVERLAP_TOKENS must be non-negative and smaller "
                "than REDACT_MAX_TOKENS"
            )

        self.tokenizer = Tokenizer.from_file(str(cache_dir / "tokenizer.json"))
        self.bos_id = self._token_id("<s>", 0)
        self.eos_id = self._token_id("</s>", 2)
        self.pad_id = self._token_id("<pad>", 1)
        self.model = self._load_model(cache_dir, config_path)
        self._synchronize()

    def _token_id(self, token: str, fallback: int) -> int:
        token_id = self.tokenizer.token_to_id(token)
        return fallback if token_id is None else int(token_id)

    def _load_model(
        self, cache_dir: Path, config_path: Path
    ) -> BertForTokenClassification:
        model_config = BertConfig.from_json_file(str(config_path))
        model = BertForTokenClassification(model_config)
        checkpoint = torch.load(
            cache_dir / "redact.pt",
            map_location="cpu",
            weights_only=True,
        )
        state_dict = (
            checkpoint.get("state_dict", checkpoint)
            if isinstance(checkpoint, dict)
            else None
        )
        if not isinstance(state_dict, dict):
            raise TypeError("redact.pt does not contain a PyTorch state dictionary")
        model.load_state_dict(state_dict, strict=True)
        model.eval()
        model.to(self.device)
        parameter_devices = {parameter.device.type for parameter in model.parameters()}
        if parameter_devices != {self.device.type}:
            raise RuntimeError(
                f"model device mismatch: expected {self.device.type}, found {sorted(parameter_devices)}"
            )
        return model

    def predict(self, text: str) -> list[dict[str, object]]:
        if not text:
            return []
        encoding = self.tokenizer.encode(text, add_special_tokens=False)
        token_count = len(encoding.ids)
        if token_count > self.max_input_tokens:
            raise InputTooLargeError(
                f"input has {token_count} tokens, exceeds max {self.max_input_tokens}"
            )
        rule_spans = deterministic_spans(text)
        if token_count == 0:
            return rule_spans

        model_spans: list[dict[str, object]] = []
        for token_start, token_end in chunk_token_ranges(
            token_count, self.max_tokens, self.chunk_overlap
        ):
            chunk_char_start = encoding.offsets[token_start][0]
            chunk_char_end = encoding.offsets[token_end - 1][1]
            if chunk_char_end <= chunk_char_start:
                continue

            chunk_text = text[chunk_char_start:chunk_char_end]
            chunk_offsets = [
                (start - chunk_char_start, end - chunk_char_start)
                for start, end in encoding.offsets[token_start:token_end]
            ]
            chunk_spans = self._predict_model_spans(
                encoding.ids[token_start:token_end],
                chunk_offsets,
                chunk_text,
            )
            model_spans.extend(
                {
                    "start": cast(int, span["start"]) + chunk_char_start,
                    "end": cast(int, span["end"]) + chunk_char_start,
                    "label": str(span["label"]),
                }
                for span in chunk_spans
            )

        return merge_spans(text, rule_spans + model_spans)

    def _predict_model_spans(
        self,
        token_ids: list[int],
        offsets: list[tuple[int, int]],
        text: str,
    ) -> list[dict[str, object]]:
        token_probabilities = self._predict_token_probabilities(token_ids)
        log_probs = np.log(np.clip(token_probabilities, 1e-7, 1.0))
        path = viterbi_decode(log_probs, self.start, self.end, self.transition)
        return self._path_to_spans(path, token_probabilities, offsets, text)

    def _predict_token_probabilities(self, token_ids: list[int]) -> np.ndarray:
        token_count = len(token_ids)
        aggregate = np.zeros(
            (token_count, self.label_space.num_classes),
            dtype=np.float32,
        )
        windows = _window_ranges(token_count)
        for batch_start in range(0, len(windows), self.batch_size):
            batch_windows = windows[batch_start : batch_start + self.batch_size]
            input_ids, attention_mask = self._build_batch(token_ids, batch_windows)
            with torch.inference_mode():
                output = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
                probabilities = torch.softmax(output.logits, dim=-1)
            self._synchronize()
            probability_array = probabilities.detach().float().cpu().numpy()
            for row_index, (start, end) in enumerate(batch_windows):
                content_length = end - start
                content_probabilities = probability_array[
                    row_index, 1 : content_length + 1
                ]
                aggregate[start:end] = np.maximum(
                    aggregate[start:end], content_probabilities
                )

        row_sums = aggregate.sum(axis=1, keepdims=True)
        np.divide(aggregate, row_sums, out=aggregate, where=row_sums > 0)
        return aggregate

    def _build_batch(
        self,
        token_ids: list[int],
        windows: list[tuple[int, int]],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_count = len(windows)
        input_id_array = np.full(
            (batch_count, MAX_SEQUENCE_LENGTH),
            self.pad_id,
            dtype=np.int64,
        )
        attention_array = np.zeros(
            (batch_count, MAX_SEQUENCE_LENGTH),
            dtype=np.int64,
        )
        for row_index, (start, end) in enumerate(windows):
            sequence = [self.bos_id, *token_ids[start:end], self.eos_id]
            sequence_length = len(sequence)
            input_id_array[row_index, :sequence_length] = sequence
            attention_array[row_index, :sequence_length] = 1
        return (
            torch.from_numpy(input_id_array).to(self.device),
            torch.from_numpy(attention_array).to(self.device),
        )

    def _path_to_spans(
        self,
        path: list[int],
        token_probabilities: np.ndarray,
        offsets: list[tuple[int, int]],
        text: str,
    ) -> list[dict[str, object]]:
        spans: list[dict[str, object]] = []
        for token_start, token_end, source_label in labels_to_spans(
            path, self.label_space
        ):
            target_label = map_model_label(source_label)
            if target_label is None:
                continue
            selected_probabilities = token_probabilities[
                token_start : token_end + 1,
                [
                    path[token_index]
                    for token_index in range(token_start, token_end + 1)
                ],
            ]
            score = float(np.min(np.diag(selected_probabilities)))
            if score < self.min_score:
                continue
            char_start = offsets[token_start][0]
            char_end = offsets[token_end][1]
            char_start, char_end = _trim_span_whitespace(text, char_start, char_end)
            if char_start < char_end:
                spans.append(
                    {
                        "start": char_start,
                        "end": char_end,
                        "label": target_label,
                        "text": text[char_start:char_end],
                    }
                )
        return spans

    def _synchronize(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        elif self.device.type == "mps":
            torch.mps.synchronize()


def _trim_span_whitespace(text: str, start: int, end: int) -> tuple[int, int]:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


class Handler(BaseHTTPRequestHandler):
    model: RedactModel
    inference_lock = threading.Lock()

    def log_message(self, format: str, *args: Any) -> None:
        sys.stderr.write(f"[redact] {self.address_string()} {format % args}\n")

    def _send_json(self, status_code: int, payload: dict[str, object]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self._send_json(400, {"error": "invalid Content-Length"})
            return
        if content_length <= 0:
            self._send_json(400, {"error": "empty body"})
            return

        try:
            request_body = json.loads(self.rfile.read(content_length).decode("utf-8"))
            request_text = request_body["text"]
            if not isinstance(request_text, str):
                raise TypeError("text must be a string")
        except (
            UnicodeDecodeError,
            json.JSONDecodeError,
            KeyError,
            TypeError,
            ValueError,
        ) as exc:
            self._send_json(400, {"error": f"bad request: {exc}"})
            return

        started_at = time.perf_counter()
        try:
            with self.inference_lock:
                spans = self.model.predict(request_text)
        except InputTooLargeError as exc:
            self._send_json(413, {"error": str(exc)})
            return
        except RuntimeError as exc:
            self._send_json(500, {"error": str(exc)})
            return

        processing_ms = round((time.perf_counter() - started_at) * 1000)
        self._send_json(
            200,
            {"processing_ms": processing_ms, "spans": spans},
        )

    def do_GET(self) -> None:
        if self.path == "/health":
            self._send_json(
                200,
                {"status": "ok", "device": str(self.model.device)},
            )
            return
        self._send_json(404, {"error": "not found"})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9124)
    parser.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"))
    args = parser.parse_args()

    print(f"[redact] cache: {CACHE_DIR}", file=sys.stderr, flush=True)
    cache_dir = ensure_assets()
    print("[redact] loading model...", file=sys.stderr, flush=True)
    Handler.model = RedactModel(cache_dir, args.device)
    print(
        f"[redact] device: {Handler.model.device}",
        file=sys.stderr,
        flush=True,
    )
    print(
        f"[redact] ready on http://{args.host}:{args.port}",
        file=sys.stderr,
        flush=True,
    )

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: server.shutdown())
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
