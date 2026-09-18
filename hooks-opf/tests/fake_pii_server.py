#!/usr/bin/env python3
"""Stand-in for pii-server.py, launched by the hook during the autostart tests.

pii-check.sh starts the detector itself when the port is cold, so the test cannot
use the in-process fake. This script takes the same `--port` and `--mode` flags
and answers the same two endpoints, with the response controlled by environment
variables the test sets before running the hook.

    FAKE_PII_HEALTH_STATUS   200 (default). Anything else fails /health.
    FAKE_PII_HEALTH_MODE     mode reported by /health. Defaults to --mode.
    FAKE_PII_RESPONSE        JSON body for POST /. Defaults to no spans.
    FAKE_PII_RESPONSE_STATUS HTTP status for POST /. Defaults to 200.
    FAKE_PII_RESPONSE_BODY   Raw body for POST /, for the invalid-JSON test.
    FAKE_PII_START_DELAY     Seconds to wait before binding. Defaults to 0.
"""

from __future__ import annotations

import json
import os
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class FakeDetectorHandler(BaseHTTPRequestHandler):
    health_status = 200
    health_mode = "redact"
    response_status = 200
    response_body = '{"spans": [], "processing_ms": 1.0}'

    def do_GET(self) -> None:
        if self.path != "/health":
            self.send_error(404)
            return
        if self.health_status != 200:
            self.send_error(self.health_status)
            return
        self._send_json(
            json.dumps({"status": "ok", "mode": self.health_mode, "device": "cpu"})
        )

    def do_POST(self) -> None:
        request_length = int(self.headers.get("Content-Length", "0"))
        self.rfile.read(request_length)
        self._send_raw(self.response_body, self.response_status)

    def _send_json(self, body: str, status_code: int = 200) -> None:
        self._send_raw(body, status_code)

    def _send_raw(self, body: str, status_code: int) -> None:
        encoded_body = body.encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded_body)))
        self.end_headers()
        self.wfile.write(encoded_body)

    def log_message(self, format: str, *args: object) -> None:
        return


def parse_arguments(argv: list[str]) -> tuple[str | None, int]:
    server_mode: str | None = None
    port = 0
    index = 0
    while index < len(argv):
        if argv[index] == "--mode":
            server_mode = argv[index + 1]
            index += 2
        elif argv[index] == "--port":
            port = int(argv[index + 1])
            index += 2
        else:
            index += 1
    return server_mode, port


def main() -> int:
    server_mode, port = parse_arguments(sys.argv[1:])
    FakeDetectorHandler.health_status = int(
        os.environ.get("FAKE_PII_HEALTH_STATUS", "200")
    )
    FakeDetectorHandler.health_mode = os.environ.get(
        "FAKE_PII_HEALTH_MODE", server_mode or "redact"
    )
    FakeDetectorHandler.response_status = int(
        os.environ.get("FAKE_PII_RESPONSE_STATUS", "200")
    )
    response_body = os.environ.get("FAKE_PII_RESPONSE_BODY")
    if response_body is None:
        response_body = os.environ.get(
            "FAKE_PII_RESPONSE", '{"spans": [], "processing_ms": 1.0}'
        )
    FakeDetectorHandler.response_body = response_body

    start_delay = float(os.environ.get("FAKE_PII_START_DELAY", "0"))
    if start_delay:
        time.sleep(start_delay)

    server = ThreadingHTTPServer(("127.0.0.1", port), FakeDetectorHandler)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
