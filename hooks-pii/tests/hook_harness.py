"""Shared harness for the check.sh tests.

Holds no tests. Both test_hook_modes.py and test_hook_paths.py build on it, so
the fake detector, the environment scrubbing, and the subprocess call live in
one place.
"""

from __future__ import annotations

import atexit
import contextlib
import functools
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from typing import Any, ClassVar

TESTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TESTS_DIR.parent / "server"))

from fake_pii_server import NoLookupHTTPServer  # noqa: E402
from answer import HookError, Policy, answer  # noqa: E402

HOOK_PATH = TESTS_DIR.parent / "hook" / "check.sh"
FAKE_SERVER_PATH = TESTS_DIR / "fake_pii_server.py"
COV_ENV_PATH = TESTS_DIR / "cov_env.sh"
# The native hook command, when set. The suites then run it in place of the script,
# and every assertion holds for both. It hands failures to the check.sh beside it.
HOOK_CLIENT = os.environ.get("PII_TEST_HOOK_CLIENT", "")


@functools.cache
def client_beside_script() -> Path:
    """A copy of the client in a folder where check.sh is the script under test."""
    folder = Path(tempfile.mkdtemp(prefix="pii-hook-client."))
    atexit.register(shutil.rmtree, folder, ignore_errors=True)
    client = folder / "hook"
    shutil.copy2(HOOK_CLIENT, client)
    (folder / "check.sh").symlink_to(HOOK_PATH)
    return client


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
    "PII_SKIP_EVENT_PATH",
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
    response_delay_seconds = 0.0
    health_status = 200
    health_mode = "redact"
    received_body_bytes = 0
    # Above this many bytes, answer 413 before reading the body, as server.py does
    # for a payload far above its cap.
    refuse_body_above: ClassVar[int | None] = None

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
        if (
            self.refuse_body_above is not None
            and request_length > self.refuse_body_above
        ):
            self._send_text("", 413)
            return
        body = self.rfile.read(request_length)
        FakePiiHandler.received_body_bytes = len(body)
        if self.response_delay_seconds:
            time.sleep(self.response_delay_seconds)
        if self.path != "/hook":
            self._send_json(self.detector_response, self.response_status)
            return
        # A server that fails its health check fails every request as well.
        status = (
            self.health_status if self.health_status != 200 else self.response_status
        )
        if status != 200:
            self._send_text("", status)
            return
        try:
            text = answer(
                body,
                Policy.from_headers(self.headers, self.health_mode),
                self.health_mode,
                2 * 1024 * 1024,
                self._detect,
            )
        except HookError as error:
            self._send_text(error.detail, error.status)
            return
        self._send_text(text, 200)

    def _detect(self, text: str) -> tuple[list[dict], Any]:
        response = self.detector_response
        return response["spans"], response.get("processing_ms", "?")  # type: ignore[return-value]

    def _send_text(self, text: str, status_code: int) -> None:
        encoded_body = text.encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded_body)))
        self.end_headers()
        self.wfile.write(encoded_body)

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
    """Runs check.sh with a controlled environment. Holds no tests."""

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
            if HOOK_CLIENT and hook_path is None:
                command = [str(client_beside_script()), "--mode", mode]
            else:
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
        cls.server = NoLookupHTTPServer(("127.0.0.1", 0), FakePiiHandler)
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
