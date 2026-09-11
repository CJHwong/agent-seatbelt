#!/usr/bin/env python3
"""Check block and warning output from the shell hook."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
HOOK_PATH = ROOT / "hooks-opf" / "pii-check.sh"


class FakePiiHandler(BaseHTTPRequestHandler):
    """Return one fixed detector span without loading a model."""

    detector_response = {
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

    def do_GET(self) -> None:  # noqa: N802
        if self.path != "/health":
            self.send_error(404)
            return
        self._send_json({"status": "ok", "mode": "redact", "device": "cpu"})

    def do_POST(self) -> None:  # noqa: N802
        request_length = int(self.headers.get("Content-Length", "0"))
        self.rfile.read(request_length)
        self._send_json(self.detector_response)

    def _send_json(self, response_body: dict[str, object]) -> None:
        encoded_body = json.dumps(response_body).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded_body)))
        self.end_headers()
        self.wfile.write(encoded_body)

    def log_message(self, format: str, *args: Any) -> None:
        return


class HookModeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if shutil.which("curl") is None or shutil.which("jq") is None:
            raise unittest.SkipTest("the shell hook requires curl and jq")
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), FakePiiHandler)
        cls.server_thread = threading.Thread(
            target=cls.server.serve_forever, daemon=True
        )
        cls.server_thread.start()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.shutdown()
        cls.server.server_close()
        cls.server_thread.join(timeout=2)

    def run_hook(
        self, mode: str, payload: dict[str, object], action_mode: str | None = None
    ) -> dict[str, Any]:
        environment = os.environ.copy()
        environment.update(
            {
                "PII_PORT": str(self.server.server_address[1]),
                "PII_SERVER_MODE": "redact",
                "PII_BLOCK_LEVEL": "standard",
            }
        )
        if action_mode is not None:
            environment["PII_ACTION_MODE"] = action_mode
        else:
            environment.pop("PII_ACTION_MODE", None)

        with tempfile.TemporaryDirectory() as temporary_directory:
            environment["PII_SERVER_LOG"] = str(
                Path(temporary_directory) / "server.log"
            )
            result = subprocess.run(
                ["bash", str(HOOK_PATH), "--mode", mode],
                cwd=ROOT,
                env=environment,
                input=json.dumps(payload),
                text=True,
                capture_output=True,
                check=False,
            )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(result.stdout.strip(), result.stderr)
        return json.loads(result.stdout)

    def test_explicit_block_action_blocks_prompt(self) -> None:
        hook_output = self.run_hook(
            "prompt",
            {"prompt": "send this secret"},
            action_mode="block",
        )

        self.assertEqual(hook_output["decision"], "block")
        self.assertIn("secret(critical)", hook_output["reason"])
        self.assertIn("sk_t...7890", hook_output["reason"])

    def test_default_action_warns_on_prompt(self) -> None:
        hook_output = self.run_hook("prompt", {"prompt": "send this secret"})

        self.assertTrue(hook_output["continue"])
        self.assertNotIn("decision", hook_output)
        self.assertNotIn("reason", hook_output)
        self.assertIn("PII detector warning", hook_output["systemMessage"])

    def test_warn_action_passes_prompt_with_agent_context(self) -> None:
        hook_output = self.run_hook(
            "prompt",
            {"prompt": "send this secret"},
            action_mode="warn",
        )

        self.assertTrue(hook_output["continue"])
        self.assertNotIn("decision", hook_output)
        self.assertNotIn("reason", hook_output)
        hook_specific_output = hook_output["hookSpecificOutput"]
        self.assertEqual(hook_specific_output["hookEventName"], "UserPromptSubmit")
        self.assertIn("PII detector warning", hook_specific_output["additionalContext"])
        self.assertIn(
            "If the detection is valid", hook_specific_output["additionalContext"]
        )
        self.assertIn("sk_t...7890", hook_specific_output["additionalContext"])

    def test_warn_action_uses_posttool_event(self) -> None:
        for posttool_mode in ("claude-posttool", "codex-posttool"):
            with self.subTest(posttool_mode=posttool_mode):
                hook_output = self.run_hook(
                    posttool_mode,
                    {"tool_response": {"stdout": "send this secret"}},
                    action_mode="warn",
                )

                self.assertTrue(hook_output["continue"])
                self.assertNotIn("reason", hook_output)
                hook_specific_output = hook_output["hookSpecificOutput"]
                self.assertEqual(hook_specific_output["hookEventName"], "PostToolUse")
                self.assertIn("tool output", hook_specific_output["additionalContext"])


if __name__ == "__main__":
    unittest.main()
