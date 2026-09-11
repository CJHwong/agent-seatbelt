# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "huggingface_hub>=0.23,<2",
#     "onnxruntime>=1.17",
#     "tokenizers>=0.15",
#     "numpy>=1.24",
#     "torch>=2.2,<3",
#     "transformers>=4.40,<6",
# ]
# ///
"""Local PII server with Redact as the default mode.

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
from typing import Any

SUPPORTED_MODES = ("redact", "openai", "rules")
DEFAULT_MODE = "redact"


def resolve_mode(requested: str | None = None) -> str:
    mode = (
        (requested or os.environ.get("PII_SERVER_MODE", DEFAULT_MODE)).strip().lower()
    )
    if mode not in SUPPORTED_MODES:
        supported = ", ".join(SUPPORTED_MODES)
        raise ValueError(f"PII_SERVER_MODE must be one of: {supported}")
    return mode


def health_payload(mode: str, model: object) -> dict[str, str]:
    payload = {"status": "ok", "mode": mode}
    if mode == "redact":
        payload["device"] = str(getattr(model, "device", "unknown"))
    else:
        payload["device"] = "cpu"
    return payload


class RulesModel:
    """Deterministic rules only. No checkpoint, no accelerator."""

    device = "cpu"

    def predict(self, text: str) -> list[dict]:
        if not text:
            return []
        # pii_rules is standard library only, so this import costs no model load.
        from pii_rules import deterministic_spans

        return deterministic_spans(text)


def load_selected_model(mode: str) -> object:
    server_dir = Path(__file__).resolve().parent
    if str(server_dir) not in sys.path:
        sys.path.insert(0, str(server_dir))

    if mode == "rules":
        return RulesModel()
    if mode == "openai":
        from pii_opf import Model, ensure_assets

        return Model(ensure_assets())

    from redact_server import RedactModel, ensure_assets as ensure_redact_assets

    requested_device = os.environ.get("REDACT_DEVICE", "auto")
    return RedactModel(ensure_redact_assets(), requested_device)


class Handler(BaseHTTPRequestHandler):
    model: Any
    mode: str
    inference_lock = threading.Lock()

    def log_message(self, format: str, *args: Any) -> None:
        sys.stderr.write(f"[{self.mode}] {self.address_string()} {format % args}\n")

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
            self._send_json(200, health_payload(self.mode, self.model))
            return
        self._send_json(404, {"error": "not found"})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9123)
    parser.add_argument("--mode", choices=SUPPORTED_MODES)
    args = parser.parse_args()

    mode = resolve_mode(args.mode)
    print(f"[{mode}] loading model...", file=sys.stderr, flush=True)
    Handler.model = load_selected_model(mode)
    Handler.mode = mode
    print(
        f"[{mode}] ready on http://{args.host}:{args.port}",
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
