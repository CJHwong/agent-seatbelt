#!/usr/bin/env python3
"""Check level selection, block output, and warning output from the shell hook."""

from __future__ import annotations

import contextlib
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

    response_status = 200

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

    @classmethod
    @contextlib.contextmanager
    def responding_with(cls, spans: list[dict[str, object]]):
        """Swap the fixed span set for one test."""
        original_response = cls.detector_response
        cls.detector_response = {"spans": spans, "processing_ms": 1.2}
        try:
            yield
        finally:
            cls.detector_response = original_response

    def do_GET(self) -> None:  # noqa: N802
        if self.path != "/health":
            self.send_error(404)
            return
        self._send_json({"status": "ok", "mode": "redact", "device": "cpu"})

    def do_POST(self) -> None:  # noqa: N802
        request_length = int(self.headers.get("Content-Length", "0"))
        self.rfile.read(request_length)
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


class HookHarness(unittest.TestCase):
    """Shared stub server and hook runner. Holds no tests."""

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

    def run_hook_raw(
        self,
        mode: str,
        payload: dict[str, object],
        action_mode: str | None = None,
        level: str | None = "standard",
        legacy_level: str | None = None,
        allow_labels: str | None = None,
    ) -> subprocess.CompletedProcess[str]:
        environment = os.environ.copy()
        environment.update(
            {
                "PII_PORT": str(self.server.server_address[1]),
                "PII_SERVER_MODE": "redact",
            }
        )
        environment.pop("PII_LEVEL", None)
        environment.pop("PII_BLOCK_LEVEL", None)
        environment.pop("PII_ALLOW_LABELS", None)
        if allow_labels is not None:
            environment["PII_ALLOW_LABELS"] = allow_labels
        if level is not None:
            environment["PII_LEVEL"] = level
        if legacy_level is not None:
            environment["PII_BLOCK_LEVEL"] = legacy_level
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
        return result

    def run_hook(
        self,
        mode: str,
        payload: dict[str, object],
        action_mode: str | None = None,
        level: str | None = "standard",
        legacy_level: str | None = None,
        allow_labels: str | None = None,
    ) -> dict[str, Any]:
        result = self.run_hook_raw(
            mode, payload, action_mode, level, legacy_level, allow_labels
        )
        self.assertTrue(result.stdout.strip(), result.stderr)
        return json.loads(result.stdout)


class HookModeTests(HookHarness):
    """Block and warn output for the default span set."""

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
        self.assertIn("secret(critical)", hook_output["systemMessage"])
        self.assertIn("sk_t...7890", hook_output["systemMessage"])
        self.assertNotIn("sk_test_1234567890", json.dumps(hook_output))

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
                self.assertIn("secret(critical)", hook_output["systemMessage"])
                self.assertIn("sk_t...7890", hook_output["systemMessage"])
                self.assertIn("tool output", hook_specific_output["additionalContext"])
                self.assertNotIn("sk_test_1234567890", json.dumps(hook_output))

    def test_detector_error_warns_instead_of_silent_pass(self) -> None:
        FakePiiHandler.response_status = 413
        try:
            hook_output = self.run_hook(
                "codex-posttool",
                {"tool_response": {"stdout": "send this secret"}},
                action_mode="warn",
            )
        finally:
            FakePiiHandler.response_status = 200

        self.assertTrue(hook_output["continue"])
        self.assertIn("PII detector unavailable", hook_output["systemMessage"])
        self.assertIn(
            "PII detector unavailable",
            hook_output["hookSpecificOutput"]["additionalContext"],
        )

    def test_detector_error_blocks_in_block_mode(self) -> None:
        FakePiiHandler.response_status = 413
        try:
            hook_output = self.run_hook(
                "codex-posttool",
                {"tool_response": {"stdout": "send this secret"}},
                action_mode="block",
            )
        finally:
            FakePiiHandler.response_status = 200

        self.assertEqual(hook_output["decision"], "block")
        self.assertIn("PII detector unavailable", hook_output["reason"])


class LevelSelectionTests(HookHarness):
    """The level decides what the agent sees, in both action modes."""

    LOW_SPAN = [
        {"start": 0, "end": 13, "label": "private_person", "text": "Kevin Nakamura"}
    ]

    def test_warn_action_stays_silent_below_the_level(self) -> None:
        with FakePiiHandler.responding_with(self.LOW_SPAN):
            result = self.run_hook_raw(
                "prompt",
                {"prompt": "Kevin Nakamura called"},
                action_mode="warn",
                level="relaxed",
            )

        self.assertEqual(result.stdout.strip(), "")
        self.assertIn("PII below level: [private_person(low)]", result.stderr)

    def test_warn_action_reports_the_span_at_its_level(self) -> None:
        with FakePiiHandler.responding_with(self.LOW_SPAN):
            hook_output = self.run_hook(
                "prompt",
                {"prompt": "Kevin Nakamura called"},
                action_mode="warn",
                level="strict",
            )

        self.assertTrue(hook_output["continue"])
        self.assertIn("private_person(low)", hook_output["systemMessage"])
        self.assertNotIn("Kevin Nakamura", json.dumps(hook_output))

    def test_allow_labels_silences_the_agent_warning(self) -> None:
        with FakePiiHandler.responding_with(self.LOW_SPAN):
            result = self.run_hook_raw(
                "prompt",
                {"prompt": "Kevin Nakamura called"},
                action_mode="warn",
                level="strict",
                allow_labels="private_person",
            )

        self.assertEqual(result.stdout.strip(), "")
        self.assertIn("PII below level: [private_person(low)]", result.stderr)

    def test_level_off_skips_the_detector_entirely(self) -> None:
        with FakePiiHandler.responding_with(self.LOW_SPAN):
            result = self.run_hook_raw(
                "prompt",
                {"prompt": "Kevin Nakamura called"},
                action_mode="block",
                level="off",
            )

        self.assertEqual(result.stdout.strip(), "")
        self.assertEqual(result.stderr.strip(), "")

    def test_legacy_block_level_still_applies(self) -> None:
        with FakePiiHandler.responding_with(self.LOW_SPAN):
            hook_output = self.run_hook(
                "prompt",
                {"prompt": "Kevin Nakamura called"},
                action_mode="block",
                level=None,
                legacy_level="strict",
            )

        self.assertEqual(hook_output["decision"], "block")
        self.assertIn("private_person(low)", hook_output["reason"])
        self.assertIn("Blocked at PII_LEVEL=strict", hook_output["reason"])

    def test_pii_level_wins_over_the_legacy_name(self) -> None:
        with FakePiiHandler.responding_with(self.LOW_SPAN):
            result = self.run_hook_raw(
                "prompt",
                {"prompt": "Kevin Nakamura called"},
                action_mode="block",
                level="relaxed",
                legacy_level="strict",
            )

        self.assertEqual(result.stdout.strip(), "")


if __name__ == "__main__":
    unittest.main()
