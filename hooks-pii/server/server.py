# /// script
# # coremltools ships its Core ML bindings for Python 3.13 at most; on 3.14 tagger
# # mode loses the Neural Engine and falls back to the CPU.
# requires-python = ">=3.11,<3.14"
# dependencies = [
#     "ai-edge-litert>=2.0",
#     "coremltools>=8; sys_platform == 'darwin' and platform_machine == 'arm64'",
#     "huggingface_hub>=0.23,<2",
#     "onnxruntime>=1.17",
#     "tokenizers>=0.15",
#     "numpy>=1.24",
#     "torch>=2.2,<3",
#     "transformers>=4.40,<6",
# ]
# ///
"""Local PII server with Redact as the default mode.

`redact` runs the published LiteRT graph, which a fresh install can download.
`redact-torch` runs the checkpoint that release no longer publishes, for a host
that already has one.

POST / {"text": "..."} -> {"spans": [{"start": int, "end": int, "label": str, "text": str}, ...]}
POST /hook <raw hook payload>, policy in X-Pii-* headers -> what check.sh prints (answer.py)
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import socketserver
import sys
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from answer import HookError, Policy, answer, code_version

SUPPORTED_MODES = ("redact", "redact-torch", "openai", "rules", "tagger")
DEFAULT_MODE = "tagger"
DEFAULT_MAX_BODY_BYTES = 2 * 1024 * 1024
# The largest text one scan covers. The model runs at about 9 ms per KB on the Neural
# Engine and the hook waits 5 s for the answer, so 256 KB is the largest text that
# finishes inside the caller's budget with one scan queued behind it.
DEFAULT_SCAN_MAX_CHARS = 256 * 1024
HANDLER_TIMEOUT_SECONDS = 30
# A hook payload wraps the text in JSON with the tool's other fields, so the body may
# run larger than the text. This guards memory in the handler, before the read. What
# the scan costs is a separate number, DEFAULT_SCAN_MAX_CHARS, checked on the text.
HOOK_PAYLOAD_FACTOR = 4


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


def scan_max_chars() -> int:
    """The largest text one scan covers. A bad setting uses the default."""
    try:
        configured = int(os.environ.get("PII_SCAN_MAX_CHARS", DEFAULT_SCAN_MAX_CHARS))
    except ValueError:
        return DEFAULT_SCAN_MAX_CHARS
    return configured if configured > 0 else DEFAULT_SCAN_MAX_CHARS


class ScanQueue:
    """One scan runs, one waits, and anyone past that is refused instead of queued.

    The model saturates the machine on its own: three concurrent 1 MB scans measured
    slower in total than one, so a scan makes no progress beside another. A queue
    would only move that wait into the caller, and the hook waits 5 s for its answer.
    A caller queued behind one full scan already risks reporting a live server as
    unavailable. Refusing at once keeps every failure of this server honest.
    """

    def __init__(self, depth: int = 1) -> None:
        self.free = threading.Condition()
        self.depth = depth
        self.active = 0
        self.waiting = 0

    @property
    def busy(self) -> bool:
        """True while a scan runs or a caller waits for one."""
        with self.free:
            return self.active > 0 or self.waiting > 0

    @contextmanager
    def hold(self) -> Iterator[None]:
        """Hold the scan slot, or raise HookError(503) when the queue is full."""
        with self.free:
            if self.active > 0:
                if self.waiting >= self.depth:
                    raise HookError(
                        503,
                        f"a scan is running and {self.waiting} caller already waits for it",
                    )
                self.waiting += 1
                # No timeout: the caller's own budget ends a request it has given up
                # on, and the queue cannot grow past depth, so this waits for one scan.
                self.free.wait_for(lambda: self.active == 0)
                self.waiting -= 1
            self.active += 1
        try:
            yield
        finally:
            with self.free:
                self.active -= 1
                self.free.notify()


def health_payload(
    mode: str, model: object, busy: bool = False, version: str = ""
) -> dict[str, object]:
    payload: dict[str, object] = {"status": "ok", "mode": mode}
    if mode in ("redact", "redact-torch", "tagger"):
        payload["device"] = str(getattr(model, "device", "unknown"))
    else:
        payload["device"] = "cpu"
    payload["busy"] = busy
    payload["version"] = version
    return payload


class NoLookupHTTPServer(ThreadingHTTPServer):
    """ThreadingHTTPServer without the reverse lookup of its own address.

    HTTPServer.server_bind names the server with socket.getfqdn(host). That lookup
    took 35 s on a macos-15 runner, and the hook waits about 10 s for a cold start.
    Nothing reads server_name, so the host serves as the name.
    """

    def server_bind(self) -> None:
        socketserver.TCPServer.server_bind(self)
        host, port = self.server_address[:2]
        self.server_name = str(host)
        self.server_port = int(port)


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

    def __init__(self) -> None:
        # Imported here, not on the first request: the native engine compiles
        # every pattern on import, and that cost belongs to start-up.
        from rules.engine import deterministic_spans

        self.deterministic_spans = deterministic_spans

    def predict(self, text: str) -> list[dict]:
        if not text:
            return []
        return self.deterministic_spans(text)

    def rules_only(self, text: str) -> list[dict]:
        """This mode is the rules, so the whole model is the cheap layer."""
        return self.predict(text)


def load_selected_model(mode: str) -> object:
    server_dir = Path(__file__).resolve().parent
    if str(server_dir) not in sys.path:
        sys.path.insert(0, str(server_dir))

    if mode == "rules":
        return RulesModel()
    if mode == "openai":
        from detectors.privacy_filter import Model, ensure_assets

        return Model(ensure_assets())
    if mode == "tagger":
        # The bilingual tagger and its rules, from the release or PII_TAGGER_DIR.
        from detectors.tagger import TaggerModel, asset_dir

        return TaggerModel(asset_dir())
    if mode == "redact-torch":
        # The checkpoint backend, kept for a host that already has redact.pt.
        from detectors.redact_torch import (
            RedactModel,
            ensure_assets as ensure_torch_assets,
        )

        requested_device = os.environ.get("REDACT_DEVICE", "auto")
        return RedactModel(ensure_torch_assets(), requested_device)

    # The default. Downloads its own assets, so a fresh install can provision it.
    from detectors.redact import LitertRedactModel, ensure_assets as ensure_graph_assets

    return LitertRedactModel(ensure_graph_assets())


class Handler(BaseHTTPRequestHandler):
    model: Any
    mode: str
    version = ""
    scans = ScanQueue()
    timeout = HANDLER_TIMEOUT_SECONDS

    def log_message(self, format: str, *args: Any) -> None:
        sys.stderr.write(f"[{self.mode}] {self.address_string()} {format % args}\n")

    def _send_json(self, code: int, payload: dict) -> None:
        self._send(code, json.dumps(payload).encode("utf-8"), "application/json")

    def _send_text(self, code: int, text: str) -> None:
        self._send(code, text.encode("utf-8"), "text/plain; charset=utf-8")

    def _send(self, code: int, body: bytes, content_type: str) -> None:
        try:
            self.send_response(code)
            self.send_header("Content-Type", content_type)
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

    def _detect(self, text: str) -> tuple[list[dict], int]:
        started_at = time.perf_counter()
        try:
            with self.scans.hold():
                spans = self.model.predict(text)
        except ValueError as exc:
            raise HookError(413) from exc
        except RuntimeError as exc:
            self.log_message("model failure: %s", exc)
            raise HookError(500) from exc
        return spans, round((time.perf_counter() - started_at) * 1000)

    def _hook(self) -> None:
        length = self._content_length()
        if length is None:
            return
        cap = max_body_bytes()
        if length > HOOK_PAYLOAD_FACTOR * cap:
            self._send_text(413, "")
            return
        payload = self.rfile.read(length) if length > 0 else b""
        policy = Policy.from_headers(self.headers, self.mode)
        # The mode may not have a cheap layer at all: the redact and openai backends
        # carry no rules, and a text past the cap is refused for them.
        rules_only = getattr(self.model, "rules_only", None)
        try:
            body = answer(
                payload, policy, self.mode, scan_max_chars(), self._detect, rules_only
            )
        except HookError as error:
            self._send_text(error.status, error.detail)
            return
        self._send_text(200, body)

    def do_POST(self):
        if self.path == "/hook":
            self._hook()
            return
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

        # The same cap as a hook request: this endpoint drives the same model, so an
        # uncapped text here would hold the scan slot for as long as it takes and
        # starve every hook waiting behind it.
        scan_max = scan_max_chars()
        if len(text) > scan_max:
            self._send_json(
                413,
                {
                    "error": f"text of {len(text)} characters exceeds the {scan_max} character scan limit"
                },
            )
            return
        started_at = time.perf_counter()
        try:
            with self.scans.hold():
                spans = self.model.predict(text)
        except ValueError as exc:
            self._send_json(413, {"error": str(exc)})
            return
        except HookError as exc:
            # A full queue is this request's problem, not the caller's text.
            self._send_json(exc.status, {"error": exc.detail})
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
                health_payload(
                    self.mode,
                    self.model,
                    self.scans.busy,
                    self.version,
                ),
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
    # Hashed before the slow model load, so the version is the code this process
    # imported, not a later copy of the files.
    Handler.version = code_version(Path(__file__))
    print(f"[{mode}] loading model...", file=sys.stderr, flush=True)
    Handler.model = load_selected_model(mode)
    Handler.mode = mode
    # Only the modes that run the rules have loaded them. The line says which
    # engine matches the rules, so a missing native module shows in the log.
    engine = sys.modules.get("rules.engine")
    if engine is not None:
        print(
            f"[{mode}] rules engine: {engine.RULES_ENGINE}", file=sys.stderr, flush=True
        )
    print(
        f"[{mode}] ready on http://{args.host}:{args.port}",
        file=sys.stderr,
        flush=True,
    )

    server = NoLookupHTTPServer((args.host, args.port), Handler)
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
