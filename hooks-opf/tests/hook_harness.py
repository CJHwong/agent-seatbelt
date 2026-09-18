"""Shared harness for the pii-check.sh tests.

Holds no tests. Both test_hook_modes.py and test_hook_paths.py build on it, so
the fake detector, the environment scrubbing, and the subprocess call live in
one place.
"""

from __future__ import annotations

import contextlib
import json
import os
import shutil
import socket
import subprocess
import tempfile
import threading
import unittest
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, ClassVar

TESTS_DIR = Path(__file__).resolve().parent
HOOK_PATH = TESTS_DIR.parent / "pii-check.sh"
FAKE_SERVER_PATH = TESTS_DIR / "fake_pii_server.py"
COV_ENV_PATH = TESTS_DIR / "cov_env.sh"

# Every key the hook reads. Popped before each run so a stray value in the
# developer's own shell cannot change a test's result.
HOOK_ENV_KEYS = (
    "PII_PORT",
    "PII_SERVER_MODE",
    "PII_SERVER_SCRIPT",
    "PII_SERVER_LOG",
    "PII_ACTION_MODE",
    "PII_LEVEL",
    "PII_BLOCK_LEVEL",
    "PII_ALLOW_LABELS",
)

REQUIRED_TOOLS = ("curl", "jq")


def tools_available() -> bool:
    return all(shutil.which(tool) for tool in REQUIRED_TOOLS)


def free_port() -> int:
    """A port nothing is listening on. Racy by nature; adequate for a test."""
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


class FakePiiHandler(BaseHTTPRequestHandler):
    """Return one fixed detector span without loading a model."""

    response_status = 200
    health_status = 200
    health_mode = "redact"
    received_body_bytes = 0

    detector_response: ClassVar[dict[str, object]] = {
        "spans": [
            {
                "start": 8,
                "end": 25,
                "label": "secret",
                "text": "sk_test_1234567890",
            }
        ],
        "processing_ms": 1.2,
    }

    @classmethod
    @contextlib.contextmanager
    def responding_with(cls, spans: list[dict[str, object]]) -> Iterator[None]:
        """Swap the fixed span set for one test."""
        original_response = cls.detector_response
        cls.detector_response = {"spans": spans, "processing_ms": 1.2}
        try:
            yield
        finally:
            cls.detector_response = original_response

    @classmethod
    @contextlib.contextmanager
    def unhealthy(cls, status: int = 503) -> Iterator[None]:
        """Report a health failure, which sends the hook down its failure path."""
        original_status = cls.health_status
        cls.health_status = status
        try:
            yield
        finally:
            cls.health_status = original_status

    def do_GET(self) -> None:
        if self.path != "/health":
            self.send_error(404)
            return
        if self.health_status != 200:
            self.send_error(self.health_status)
            return
        self._send_json(
            {"status": "ok", "mode": self.health_mode, "device": "cpu"},
            self.health_status,
        )

    def do_POST(self) -> None:
        request_length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(request_length)
        FakePiiHandler.received_body_bytes = len(body)
        self._send_json(self.detector_response, self.response_status)

    def _send_json(
        self, response_body: dict[str, object], status_code: int = 200
    ) -> None:
        encoded_body = json.dumps(response_body).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded_body)))
        self.end_headers()
        self.wfile.write(encoded_body)

    def log_message(self, format: str, *args: Any) -> None:
        return


class HookRunner(unittest.TestCase):
    """Runs pii-check.sh with a controlled environment. Holds no tests."""

    detector_port: int = 0
    server_mode = "redact"

    def run_hook_raw(
        self,
        mode: str,
        payload: dict[str, object] | str,
        action_mode: str | None = None,
        level: str | None = "standard",
        legacy_level: str | None = None,
        allow_labels: str | None = None,
        extra_env: dict[str, str] | None = None,
        extra_args: list[str] | None = None,
        server_script: str | None = None,
        server_mode: str | None = None,
        hook_path: Path | None = None,
        timeout: int = 90,
    ) -> subprocess.CompletedProcess[str]:
        """Run the hook once. `payload` is JSON-encoded unless it is already str."""
        environment = os.environ.copy()
        for key in HOOK_ENV_KEYS:
            environment.pop(key, None)
        environment["PII_PORT"] = str(self.detector_port)
        environment["PII_SERVER_MODE"] = server_mode or self.server_mode
        if action_mode is not None:
            environment["PII_ACTION_MODE"] = action_mode
        if level is not None:
            environment["PII_LEVEL"] = level
        if legacy_level is not None:
            environment["PII_BLOCK_LEVEL"] = legacy_level
        if allow_labels is not None:
            environment["PII_ALLOW_LABELS"] = allow_labels
        if server_script is not None:
            environment["PII_SERVER_SCRIPT"] = server_script
        if extra_env:
            environment.update(extra_env)

        with tempfile.TemporaryDirectory() as temporary_directory:
            environment.setdefault(
                "PII_SERVER_LOG", str(Path(temporary_directory) / "server.log")
            )
            hook_input = payload if isinstance(payload, str) else json.dumps(payload)
            command = ["bash", str(hook_path or HOOK_PATH), "--mode", mode]
            if extra_args:
                command.extend(extra_args)
            result = subprocess.run(
                command,
                cwd=TESTS_DIR.parent,
                env=environment,
                input=hook_input,
                text=True,
                capture_output=True,
                check=False,
                timeout=timeout,
            )

        if result.returncode != 0:
            raise AssertionError(
                f"hook exited {result.returncode}\nstderr: {result.stderr}"
            )
        return result

    def run_hook(
        self,
        mode: str,
        payload: dict[str, object] | str,
        action_mode: str | None = None,
        level: str | None = "standard",
        legacy_level: str | None = None,
        allow_labels: str | None = None,
        extra_env: dict[str, str] | None = None,
        extra_args: list[str] | None = None,
        server_script: str | None = None,
        server_mode: str | None = None,
    ) -> dict[str, Any]:
        """Run the hook and parse its JSON response."""
        result = self.run_hook_raw(
            mode,
            payload,
            action_mode,
            level,
            legacy_level,
            allow_labels,
            extra_env=extra_env,
            extra_args=extra_args,
            server_script=server_script,
            server_mode=server_mode,
        )
        self.assertTrue(result.stdout.strip(), result.stderr)
        return json.loads(result.stdout)


class HookHarness(HookRunner):
    """A hook runner wired to the in-process fake detector."""

    @classmethod
    def setUpClass(cls) -> None:
        if not tools_available():
            raise unittest.SkipTest("the shell hook requires curl and jq")
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), FakePiiHandler)
        cls.detector_port = cls.server.server_address[1]
        cls.server_thread = threading.Thread(
            target=cls.server.serve_forever, daemon=True
        )
        cls.server_thread.start()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.shutdown()
        cls.server.server_close()
        cls.server_thread.join(timeout=2)
