#!/usr/bin/env python3
"""Stand-in for server.py, launched by the hook during the autostart tests.

check.sh starts the detector itself when the port is cold, so the test cannot
use the in-process fake. This script takes the same `--port` and `--mode` flags
and answers the same two endpoints, with the response controlled by environment
variables the test sets before running the hook.

    FAKE_PII_HEALTH_STATUS   200 (default). Anything else fails /health.
    FAKE_PII_HEALTH_MODE     mode reported by /health. Defaults to --mode.
    FAKE_PII_HEALTH_VERSION  version reported by /health. Defaults to the hash of
                             this script, the version check.sh expects of it.
    FAKE_PII_RESPONSE        JSON body for POST /, and the spans POST /hook answers on.
                             Defaults to no spans.
    FAKE_PII_RESPONSE_STATUS HTTP status for POST / and POST /hook. Defaults to 200.
    FAKE_PII_RESPONSE_BODY   Raw body for POST /, for the invalid-JSON test.
    FAKE_PII_START_DELAY     Seconds to wait before binding. Defaults to 0.
"""

from __future__ import annotations

import json
import os
import socketserver
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

# answer.py sits one level up in the checkout, and next to this file once the
# install tests copy it in as pii-server.py.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "server"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from answer import HookError, Policy, answer, code_version  # noqa: E402


class NoLookupHTTPServer(ThreadingHTTPServer):
    """The real server's bind, without socket.getfqdn: 35 s on a macos-15 runner."""

    def server_bind(self) -> None:
        socketserver.TCPServer.server_bind(self)
        host, port = self.server_address[:2]
        self.server_name = str(host)
        self.server_port = int(port)


class FakeDetectorHandler(BaseHTTPRequestHandler):
    health_status = 200
    health_mode = "redact"
    health_version = ""
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
            json.dumps(
                {
                    "status": "ok",
                    "mode": self.health_mode,
                    "device": "cpu",
                    "version": self.health_version,
                }
            )
        )

    def do_POST(self) -> None:
        request_length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(request_length)
        if self.path != "/hook":
            self._send_raw(self.response_body, self.response_status)
            return
        status = (
            self.health_status if self.health_status != 200 else self.response_status
        )
        if status != 200:
            self._send_raw("", status)
            return
        try:
            spans = json.loads(self.response_body)["spans"]
            text = answer(
                body,
                Policy.from_headers(self.headers, self.health_mode),
                self.health_mode,
                2 * 1024 * 1024,
                lambda _: (spans, 1),
            )
        except HookError as error:
            self._send_raw(error.detail, error.status)
            return
        except (ValueError, KeyError, TypeError):
            self._send_raw("", 500)
            return
        self._send_raw(text, 200)

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
    FakeDetectorHandler.health_version = os.environ.get(
        "FAKE_PII_HEALTH_VERSION", code_version(Path(__file__))
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

    server = NoLookupHTTPServer(("127.0.0.1", port), FakeDetectorHandler)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
