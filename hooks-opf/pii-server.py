# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "onnxruntime>=1.17",
#     "tokenizers>=0.15",
#     "huggingface_hub>=0.23",
#     "numpy>=1.24",
# ]
# ///
"""Local int8 PII server for openai/privacy-filter.

POST / {"text": "..."} -> {"spans": [{"start": int, "end": int, "label": str, "text": str}, ...]}
"""

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

import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

HF_REPO = "openai/privacy-filter"
CACHE_DIR = Path(os.environ.get("OPF_CACHE_DIR", Path.home() / ".cache" / "opf"))
NEG_INF = -1e9

REQUIRED_FILES = [
    "config.json",
    "tokenizer.json",
    "viterbi_calibration.json",
    "onnx/model_quantized.onnx",
    "onnx/model_quantized.onnx_data",
]


def ensure_assets() -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    for rel in REQUIRED_FILES:
        hf_hub_download(
            repo_id=HF_REPO,
            filename=rel,
            local_dir=str(CACHE_DIR),
        )
    return CACHE_DIR


class LabelSpace:
    """BIOES label space derived from config.json's id2label."""

    def __init__(self, id2label: dict[int, str]):
        self.id2label = id2label
        self.num_classes = len(id2label)
        self.bg = next(i for i, n in id2label.items() if n == "O")

        self.tag: dict[int, str | None] = {}
        self.span_name: dict[int, str | None] = {}
        for idx, name in id2label.items():
            if name == "O":
                self.tag[idx] = None
                self.span_name[idx] = None
            else:
                head, _, rest = name.partition("-")
                self.tag[idx] = head
                self.span_name[idx] = rest


def build_transition_tables(ls: LabelSpace) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (start_scores, end_scores, transition_scores) with invalid edges = NEG_INF.

    All default biases are 0, so valid edges carry 0 and invalid edges are masked out.
    """
    n = ls.num_classes
    start = np.full(n, NEG_INF, dtype=np.float32)
    end = np.full(n, NEG_INF, dtype=np.float32)
    trans = np.full((n, n), NEG_INF, dtype=np.float32)

    for i in range(n):
        if ls.tag[i] in {"B", "S"} or i == ls.bg:
            start[i] = 0.0
        if ls.tag[i] in {"E", "S"} or i == ls.bg:
            end[i] = 0.0

    for i in range(n):
        for j in range(n):
            if _valid(ls, i, j):
                trans[i, j] = 0.0
    return start, end, trans


def _valid(ls: LabelSpace, prev: int, nxt: int) -> bool:
    prev_bg = prev == ls.bg
    nxt_bg = nxt == ls.bg
    prev_tag = ls.tag[prev]
    nxt_tag = ls.tag[nxt]

    if prev_bg:
        return nxt_bg or nxt_tag in {"B", "S"}
    if prev_tag in {"E", "S"}:
        return nxt_bg or nxt_tag in {"B", "S"}
    if prev_tag in {"B", "I"}:
        return (not nxt_bg) and nxt_tag in {"I", "E"} and ls.span_name[prev] == ls.span_name[nxt]
    return False


def viterbi_decode(
    logits: np.ndarray,
    start: np.ndarray,
    end: np.ndarray,
    trans: np.ndarray,
) -> list[int]:
    """Decode [T, num_classes] log-softmax scores into a label path."""
    t, n = logits.shape
    if t == 0:
        return []

    scores = logits[0] + start
    backpointers = np.empty((t - 1, n), dtype=np.int64)

    for step in range(1, t):
        candidates = scores[:, None] + trans
        best_prev = np.argmax(candidates, axis=0)
        best = candidates[best_prev, np.arange(n)]
        scores = best + logits[step]
        backpointers[step - 1] = best_prev

    final = scores + end
    if not np.isfinite(final).any():
        return logits.argmax(axis=1).tolist()

    path = np.empty(t, dtype=np.int64)
    path[-1] = int(np.argmax(final))
    for step in range(t - 2, -1, -1):
        path[step] = backpointers[step, path[step + 1]]
    return path.tolist()


def labels_to_spans(
    path: list[int],
    ls: LabelSpace,
) -> list[tuple[int, int, str]]:
    """Convert a BIOES label path into [token_start, token_end_inclusive, span_name] tuples."""
    spans: list[tuple[int, int, str]] = []
    active_start: int | None = None
    active_name: str | None = None

    for i, label_id in enumerate(path):
        tag = ls.tag[label_id]
        name = ls.span_name[label_id]

        if tag == "S":
            if active_start is not None and active_name is not None:
                spans.append((active_start, i - 1, active_name))
                active_start = None
                active_name = None
            spans.append((i, i, name))
        elif tag == "B":
            if active_start is not None and active_name is not None:
                spans.append((active_start, i - 1, active_name))
            active_start = i
            active_name = name
        elif tag == "I":
            if active_name != name:
                active_start = i
                active_name = name
        elif tag == "E":
            if active_start is None:
                active_start = i
                active_name = name
            spans.append((active_start, i, active_name or name))
            active_start = None
            active_name = None
        else:
            if active_start is not None and active_name is not None:
                spans.append((active_start, i - 1, active_name))
                active_start = None
                active_name = None

    if active_start is not None and active_name is not None:
        spans.append((active_start, len(path) - 1, active_name))

    return spans


class Model:
    def __init__(self, cache_dir: Path):
        config = json.loads((cache_dir / "config.json").read_text())
        id2label = {int(k): v for k, v in config["id2label"].items()}
        self.ls = LabelSpace(id2label)
        self.start, self.end, self.trans = build_transition_tables(self.ls)

        self.tokenizer = Tokenizer.from_file(str(cache_dir / "tokenizer.json"))
        so = ort.SessionOptions()
        so.intra_op_num_threads = max(1, (os.cpu_count() or 4) // 2)
        self.session = ort.InferenceSession(
            str(cache_dir / "onnx" / "model_quantized.onnx"),
            sess_options=so,
            providers=["CPUExecutionProvider"],
        )
        self.max_len = config.get("max_position_embeddings", 131072)

    def predict(self, text: str) -> list[dict]:
        if not text:
            return []

        enc = self.tokenizer.encode(text, add_special_tokens=False)
        if len(enc.ids) == 0:
            return []
        if len(enc.ids) > self.max_len:
            raise ValueError(
                f"input has {len(enc.ids)} tokens, exceeds max {self.max_len}"
            )

        input_ids = np.array([enc.ids], dtype=np.int64)
        attention_mask = np.ones_like(input_ids)

        logits = self.session.run(
            ["logits"],
            {"input_ids": input_ids, "attention_mask": attention_mask},
        )[0][0]

        log_probs = logits - _logsumexp(logits, axis=-1, keepdims=True)
        path = viterbi_decode(log_probs, self.start, self.end, self.trans)

        spans = []
        for tok_start, tok_end, name in labels_to_spans(path, self.ls):
            char_start = enc.offsets[tok_start][0]
            char_end = enc.offsets[tok_end][1]
            while char_start < char_end and text[char_start].isspace():
                char_start += 1
            while char_end > char_start and text[char_end - 1].isspace():
                char_end -= 1
            if char_start >= char_end:
                continue
            spans.append({
                "start": char_start,
                "end": char_end,
                "label": name,
                "text": text[char_start:char_end],
            })
        return spans


def _logsumexp(x: np.ndarray, axis: int, keepdims: bool) -> np.ndarray:
    m = np.max(x, axis=axis, keepdims=True)
    out = m + np.log(np.sum(np.exp(x - m), axis=axis, keepdims=True))
    return out if keepdims else np.squeeze(out, axis=axis)


class Handler(BaseHTTPRequestHandler):
    model: Model  # set on the class by main()
    inference_lock = threading.Lock()  # ORT run() is thread-safe, but tokenizer may not be

    def log_message(self, fmt, *args):
        sys.stderr.write(f"[opf] {self.address_string()} {fmt % args}\n")

    def _send_json(self, code: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0:
            self._send_json(400, {"error": "empty body"})
            return
        try:
            body = json.loads(self.rfile.read(length).decode("utf-8"))
            text = body["text"]
            if not isinstance(text, str):
                raise ValueError("text must be a string")
        except (json.JSONDecodeError, KeyError, ValueError) as exc:
            self._send_json(400, {"error": f"bad request: {exc}"})
            return

        started_at = time.perf_counter()
        try:
            with self.inference_lock:
                spans = self.model.predict(text)
        except ValueError as exc:
            self._send_json(413, {"error": str(exc)})
            return

        processing_ms = round((time.perf_counter() - started_at) * 1000)
        self._send_json(200, {"processing_ms": processing_ms, "spans": spans})

    def do_GET(self):
        if self.path == "/health":
            self._send_json(200, {"status": "ok"})
            return
        self._send_json(404, {"error": "not found"})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9123)
    args = parser.parse_args()

    print(f"[opf] cache: {CACHE_DIR}", file=sys.stderr)
    cache_dir = ensure_assets()
    print("[opf] loading model...", file=sys.stderr)
    Handler.model = Model(cache_dir)
    print(f"[opf] ready on http://{args.host}:{args.port}", file=sys.stderr)

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
