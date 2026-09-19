#!/usr/bin/env python3
"""Check level selection, block output, and warning output from the shell hook."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from hook_harness import FakePiiHandler, HookHarness


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
        self.assertIn("sk...90", hook_output["reason"])

    def test_default_action_warns_on_prompt(self) -> None:
        hook_output = self.run_hook("prompt", {"prompt": "send this secret"})

        self.assertTrue(hook_output["continue"])
        self.assertNotIn("decision", hook_output)
        self.assertNotIn("reason", hook_output)
        self.assertIn("PII detector warning", hook_output["systemMessage"])
        self.assertIn("secret(critical)", hook_output["systemMessage"])
        self.assertIn("sk...90", hook_output["systemMessage"])
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
        self.assertIn("sk...90", hook_specific_output["additionalContext"])

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
                self.assertIn("sk...90", hook_output["systemMessage"])
                self.assertIn("tool output", hook_specific_output["additionalContext"])
                self.assertNotIn("sk_test_1234567890", json.dumps(hook_output))

    def test_detector_error_warns_instead_of_silent_pass(self) -> None:
        FakePiiHandler.response_status = 500
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
        FakePiiHandler.response_status = 500
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

    def test_the_user_message_names_the_cause_of_a_detector_failure(self) -> None:
        """The person who can restart the server reads systemMessage, not the context.

        Without the cause, every one of the failure branches produces the same
        sentence, so a recurring failure cannot be told apart from a one-off.
        """
        FakePiiHandler.response_status = 500
        try:
            hook_output = self.run_hook(
                "prompt",
                {"prompt": "send this secret"},
                action_mode="warn",
            )
        finally:
            FakePiiHandler.response_status = 200

        self.assertIn("HTTP status '500'", hook_output["systemMessage"])

    def test_a_detector_failure_is_recorded_on_the_host(self) -> None:
        """A skipped scan reaches the host log whether a tool or the detector caused it.

        The in-band warning goes to the agent, and the agent is the one party the hook
        cannot vouch for. A detector failure leaves the request unscanned exactly as a
        missing tool does, so it needs the same host-side line.
        """
        with tempfile.TemporaryDirectory() as temporary_directory:
            skip_log = Path(temporary_directory) / "pii-skips.log"
            FakePiiHandler.response_status = 500
            try:
                self.run_hook(
                    "prompt",
                    {"prompt": "send this secret"},
                    action_mode="warn",
                    extra_env={"PII_SKIP_EVENT_PATH": str(skip_log)},
                )
            finally:
                FakePiiHandler.response_status = 200

            self.assertTrue(skip_log.exists(), "no host-side line was written")
            self.assertIn("HTTP status '500'", skip_log.read_text())

    def test_the_user_message_names_the_cause_of_an_oversize_failure(self) -> None:
        FakePiiHandler.response_status = 413
        try:
            hook_output = self.run_hook(
                "prompt",
                {"prompt": "send this secret"},
                action_mode="warn",
            )
        finally:
            FakePiiHandler.response_status = 200

        self.assertIn("HTTP 413", hook_output["systemMessage"])

    def test_an_oversize_failure_is_recorded_on_the_host(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            skip_log = Path(temporary_directory) / "pii-skips.log"
            FakePiiHandler.response_status = 413
            try:
                self.run_hook(
                    "prompt",
                    {"prompt": "send this secret"},
                    action_mode="warn",
                    extra_env={"PII_SKIP_EVENT_PATH": str(skip_log)},
                )
            finally:
                FakePiiHandler.response_status = 200

            self.assertTrue(skip_log.exists(), "no host-side line was written")
            self.assertIn("HTTP 413", skip_log.read_text())

    def test_oversized_input_is_not_reported_as_a_dead_detector(self) -> None:
        """A 413 is a rejected input, not a broken detector, and the agent can act on it."""
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
        self.assertIn(
            "too large for the detector to scan", hook_output["systemMessage"]
        )
        self.assertNotIn("PII detector unavailable", hook_output["systemMessage"])

    def test_oversized_input_blocks_with_an_accurate_reason(self) -> None:
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
        self.assertIn("too large for the detector to scan", hook_output["reason"])
        self.assertIn("not a retryable failure", hook_output["reason"])
        self.assertNotIn("PII detector unavailable", hook_output["reason"])


class LevelSelectionTests(HookHarness):
    """The level decides what the agent sees, in both action modes."""

    LOW_SPAN: list[dict[str, object]] = [
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
