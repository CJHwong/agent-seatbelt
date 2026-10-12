"""A stand-in detector for the warning eval: real answers, fixed detections.

usage: python3 detector.py --port PORT --needles DIR [--guidance FILE] [--down]

It answers POST /hook through the real answer.py, so the hook prints exactly what a
real server makes it print. The only difference is what counts as a detection: every
value registered as DIR/*.json ({"value": "label", ...}) and found in the text. The
eval measures how an agent acts on a warning, not how well the rules find PII.

The text of FILE replaces answer.WARN_GUIDANCE, which is the wording under test.
With --down it answers every scan with HTTP 503, a detector that is up but broken.
"""

from __future__ import annotations

import argparse
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "server"))

import answer  # noqa: E402

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


class Handler(BaseHTTPRequestHandler):
    needles: Path
    down = False

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
        if self.down:
            self._send(503, "")
            return
        payload = self.rfile.read(int(self.headers.get("Content-Length", "0")))
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
    parser.add_argument("--guidance", type=Path)
    parser.add_argument("--down", action="store_true")
    args = parser.parse_args()
    if args.guidance:
        answer.WARN_GUIDANCE = args.guidance.read_text().strip()
    Handler.needles = args.needles
    Handler.down = args.down
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
