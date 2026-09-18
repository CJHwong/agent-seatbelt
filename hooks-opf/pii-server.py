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
DEFAULT_MAX_BODY_BYTES = 2 * 1024 * 1024
HANDLER_TIMEOUT_SECONDS = 30


def resolve_mode(requested: str | None = None) -> str:
    mode = (
        (requested or os.environ.get("PII_SERVER_MODE", DEFAULT_MODE)).strip().lower()
    )
    if mode not in SUPPORTED_MODES:
        supported = ", ".join(SUPPORTED_MODES)
        raise ValueError(f"PII_SERVER_MODE must be one of: {supported}")
    return mode


def max_body_bytes() -> int:
    """The largest request body the server reads. A bad setting uses the default."""
    try:
        configured = int(os.environ.get("PII_MAX_BODY_BYTES", DEFAULT_MAX_BODY_BYTES))
    except ValueError:
        return DEFAULT_MAX_BODY_BYTES
    return configured if configured > 0 else DEFAULT_MAX_BODY_BYTES


def health_payload(mode: str, model: object, busy: bool = False) -> dict[str, object]:
    payload: dict[str, object] = {"status": "ok", "mode": mode}
    if mode == "redact":
        payload["device"] = str(getattr(model, "device", "unknown"))
    else:
        payload["device"] = "cpu"
    payload["busy"] = busy
    return payload


def request_shutdown(server: ThreadingHTTPServer) -> None:
    """Stop serve_forever() from a thread that the serving loop does not own.

    shutdown() blocks until serve_forever() returns. A signal handler runs on
    the main thread, which is inside serve_forever(), so calling shutdown()
    there waits on itself. socketserver expects a separate thread.
    """
    threading.Thread(target=server.shutdown, daemon=True).start()


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
    timeout = HANDLER_TIMEOUT_SECONDS

    def log_message(self, format: str, *args: Any) -> None:
        sys.stderr.write(f"[{self.mode}] {self.address_string()} {format % args}\n")

    def _send_json(self, code: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        try:
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            # The client left before the answer. One line says so, and a
            # routine disconnect stops burying a real failure in a traceback.
            self.log_message("client disconnected before the response")

    def _content_length(self) -> int | None:
        declared = self.headers.get("Content-Length", "0")
        try:
            return int(declared)
        except ValueError:
            self._send_json(400, {"error": "invalid Content-Length"})
            return None

    def _read_text(self, length: int) -> str | None:
        try:
            body = json.loads(self.rfile.read(length).decode("utf-8"))
            if not isinstance(body, dict):
                raise ValueError("body must be a JSON object")
            text = body["text"]
            if not isinstance(text, str):
                raise ValueError("text must be a string")
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            self._send_json(400, {"error": f"bad request: {exc}"})
            return None
        return text

    def do_POST(self):
        length = self._content_length()
        if length is None:
            return
        if length <= 0:
            self._send_json(400, {"error": "empty body"})
            return
        cap = max_body_bytes()
        if length > cap:
            # Answer before the read. Reading a body this large costs memory in
            # proportion to the body, not to the cap.
            error = f"request body of {length} bytes exceeds the {cap} byte limit"
            self._send_json(413, {"error": error})
            return

        text = self._read_text(length)
        if text is None:
            return

        started_at = time.perf_counter()
        try:
            with self.inference_lock:
                spans = self.model.predict(text)
        except ValueError as exc:
            self._send_json(413, {"error": str(exc)})
            return
        except RuntimeError as exc:
            # A model that cannot produce the expected output shape is a broken
            # detector, not an oversized input. It must not read as 413.
            self.log_message("model failure: %s", exc)
            self._send_json(500, {"error": str(exc)})
            return

        processing_ms = round((time.perf_counter() - started_at) * 1000)
        self._send_json(200, {"processing_ms": processing_ms, "spans": spans})

    def do_GET(self):
        if self.path == "/health":
            self._send_json(
                200,
                health_payload(self.mode, self.model, self.inference_lock.locked()),
            )
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
        signal.signal(sig, lambda *_: request_shutdown(server))
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
