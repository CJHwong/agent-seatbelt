"""A stand-in detector for the warning eval: real answers, fixed detections.

usage: python3 detector.py --port PORT --needles DIR --answer FILE [--status N]

It answers POST /hook through the real answer.py, so the hook prints exactly what a
real server makes it print. The only difference is what counts as a detection: every
value registered as DIR/*.json ({"value": "label", ...}) and found in the text. The
eval measures how an agent acts on a warning, not how well the rules find PII.

FILE is answer.py with a variant's wording. With --status other than 200, every scan
of a tool event gets that status instead: 503 for a detector that broke mid-session,
413 for an input over its limit. The prompt is still answered: in block mode a failed
prompt scan rejects the prompt, and the agent never starts.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import ModuleType

MODE = "rules"
MAX_BODY = 2 * 1024 * 1024


def registered(needles: Path) -> dict[str, str]:
    """Every run's values, read per request: runs register while the server runs."""
    values: dict[str, str] = {}
    for path in needles.glob("*.json"):
        values.update(json.loads(path.read_text()))
    return values


def spans_in(text: str, values: dict[str, str]) -> list[dict]:
    spans = []
    for value, label in values.items():
        start = text.find(value)
        while start != -1:
            end = start + len(value)
            spans.append({"start": start, "end": end, "label": label, "text": value})
            start = text.find(value, end)
    return spans


def load(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("answer", path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    # A dataclass looks its module up here while the module runs.
    sys.modules["answer"] = module
    spec.loader.exec_module(module)
    return module


def is_prompt(payload: bytes) -> bool:
    try:
        event = json.loads(payload)
    except json.JSONDecodeError:
        return False
    return isinstance(event, dict) and "prompt" in event


class Handler(BaseHTTPRequestHandler):
    needles: Path
    answer: ModuleType
    status = 200

    def log_message(self, format: str, *args: object) -> None:
        pass

    def do_GET(self) -> None:
        if self.path != "/health":
            self.send_error(404)
            return
        self._send(
            200,
            json.dumps({"status": "ok", "mode": MODE, "busy": False, "version": ""}),
        )

    def do_POST(self) -> None:
        if self.path != "/hook":
            self.send_error(404)
            return
        payload = self.rfile.read(int(self.headers.get("Content-Length", "0")))
        if self.status != 200 and not is_prompt(payload):
            self._send(self.status, "")
            return
        answer = self.answer
        values = registered(self.needles)
        policy = answer.Policy.from_headers(self.headers, MODE)
        try:
            body = answer.answer(
                payload,
                policy,
                MODE,
                MAX_BODY,
                lambda text: (spans_in(text, values), 0),
            )
        except answer.HookError as error:
            self._send(error.status, error.detail)
            return
        self._send(200, body)

    def _send(self, status: int, body: str) -> None:
        data = body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--needles", type=Path, required=True)
    parser.add_argument("--answer", type=Path, required=True)
    parser.add_argument("--status", type=int, default=200)
    args = parser.parse_args()
    Handler.needles = args.needles
    Handler.answer = load(args.answer)
    Handler.status = args.status
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
