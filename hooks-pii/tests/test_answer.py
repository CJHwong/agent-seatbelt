"""Unit tests for answer.py, the answer POST /hook returns.

The hook suites run check.sh against a fake detector that calls this module, so
they cover the whole path. These tests pin the edge cases the script inherited from
jq, which the hook suites do not reach.
"""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "server"))

from answer import (
    SEPARATOR,
    HookError,
    Policy,
    answer,
    code_version,
    hook_text,
    mask,
    parse,
)

KEY_SPAN = {
    "start": 0,
    "end": 20,
    "label": "secret",
    "text": "AKIA" + "Q3EGRZ7MXN2PLW4T",
}
EMAIL_SPAN = {
    "start": 0,
    "end": 20,
    "label": "private_email",
    "text": "jane.doe@example.com",
}
PERSON_SPAN = {"start": 0, "end": 8, "label": "private_person", "text": "Jane Doe"}


def policy(**overrides: str) -> Policy:
    values = {
        "mode": "prompt",
        "level": "standard",
        "allow_labels": "",
        "action_mode": "warn",
        "allow_bypass": "1",
        "server_mode": "tagger",
    }
    values.update(overrides)
    return Policy(**values)


def run(payload: object, spans: list, **overrides: str) -> tuple[str, str]:
    """stdout and stderr of one answer, split on the separator."""
    body = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
    framed = answer(
        body, policy(**overrides), "tagger", 2 * 1024 * 1024, lambda text: (spans, 3)
    )
    stdout, stderr, closing = framed.split(SEPARATOR)
    assert closing == ""
    return stdout, stderr


class CodeVersionTests(unittest.TestCase):
    """check.sh hashes the installed files the same way, so the order is a contract."""

    def test_every_module_below_the_folder_in_relative_path_order(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            (folder / "rules").mkdir()
            (folder / "server.py").write_bytes(b"server")
            (folder / "rules" / "engine.py").write_bytes(b"engine")
            (folder / "answer.py").write_bytes(b"answer")
            (folder / "notes.txt").write_bytes(b"ignored")

            version = code_version(folder / "server.py")

        self.assertEqual(
            version, hashlib.sha256(b"answer" + b"engine" + b"server").hexdigest()
        )

    def test_a_dot_sorts_before_an_underscore(self) -> None:
        """Byte order, as LC_ALL=C sort gives. A UTF-8 locale can sort them the other way."""
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            (folder / "server.py").write_bytes(b"server")
            (folder / "redact_torch.py").write_bytes(b"torch")
            (folder / "redact.py").write_bytes(b"redact")

            version = code_version(folder / "server.py")

        self.assertEqual(
            version, hashlib.sha256(b"redact" + b"torch" + b"server").hexdigest()
        )

    def test_a_changed_module_changes_the_version(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            (folder / "detectors").mkdir()
            (folder / "server.py").write_bytes(b"server")
            (folder / "detectors" / "tagger.py").write_bytes(b"old")
            before = code_version(folder / "server.py")
            (folder / "detectors" / "tagger.py").write_bytes(b"new")

            self.assertNotEqual(code_version(folder / "server.py"), before)


class TextTests(unittest.TestCase):
    def test_leaf_strings_join_in_document_order(self) -> None:
        document = {"tool_response": {"a": "one", "b": ["two", {"c": "three"}], "d": 4}}
        self.assertEqual(hook_text(document, "claude-posttool"), "one\ntwo\nthree")

    def test_a_string_field_is_taken_whole_and_an_array_field_is_empty(self) -> None:
        self.assertEqual(
            hook_text({"tool_input": "ls -la"}, "claude-pretool"), "ls -la"
        )
        self.assertEqual(hook_text({"tool_input": ["ls"]}, "claude-pretool"), "")

    def test_trailing_newlines_and_nuls_drop_as_bash_dropped_them(self) -> None:
        self.assertEqual(hook_text({"prompt": "a\0b\n\n"}, "prompt"), "ab")

    def test_a_falsy_prompt_is_empty_and_a_number_is_its_text(self) -> None:
        self.assertEqual(hook_text({"prompt": None}, "prompt"), "")
        self.assertEqual(hook_text({"prompt": False}, "prompt"), "")
        self.assertEqual(hook_text({"prompt": 42}, "prompt"), "42")

    def test_auto_mode_reads_the_response_before_the_input(self) -> None:
        document = {"tool_input": {"command": "in"}, "tool_response": {"stdout": "out"}}
        self.assertEqual(hook_text(document, "auto"), "out")

    def test_a_document_that_is_not_an_object_is_unparsable(self) -> None:
        with self.assertRaises(HookError) as raised:
            hook_text(["a"], "prompt")
        self.assertEqual(raised.exception.status, 422)

    def test_blank_input_is_null_and_bad_json_is_unparsable(self) -> None:
        self.assertIsNone(parse(b"  \n"))
        with self.assertRaises(HookError) as raised:
            parse(b"not json")
        self.assertEqual(raised.exception.status, 422)


class MaskTests(unittest.TestCase):
    def test_short_values_are_redacted_and_long_ones_keep_two_each_end(self) -> None:
        self.assertEqual(mask({"text": "short"}), "[redacted]")
        self.assertEqual(mask({"text": "jane.doe@example.com"}), "ja...om")

    def test_whitespace_runs_collapse_before_the_length_check(self) -> None:
        self.assertEqual(mask({"text": "ab\r\n\t   cd    ef"}), "[redacted]")
        self.assertEqual(mask({"text": "\nabcdefghijkl"}), " a...kl")


class AnswerTests(unittest.TestCase):
    def test_nothing_found_prints_nothing(self) -> None:
        self.assertEqual(run({"prompt": "hello"}, []), ("", ""))

    def test_a_secret_in_warn_mode_is_one_json_line(self) -> None:
        stdout, stderr = run({"prompt": "key"}, [KEY_SPAN])
        output = json.loads(stdout)
        self.assertTrue(stdout.endswith("}\n"))
        self.assertEqual(
            output["hookSpecificOutput"]["hookEventName"], "UserPromptSubmit"
        )
        self.assertIn("secret(critical): AK...4T", output["systemMessage"])
        self.assertEqual(stderr, "")

    def test_a_span_below_the_level_goes_to_stderr_only(self) -> None:
        self.assertEqual(
            run({"prompt": "Jane"}, [PERSON_SPAN]),
            ("", "PII below level: [private_person(low)] [redacted]\n"),
        )

    def test_block_mode_denies_a_pretool_call_and_logs_each_span(self) -> None:
        stdout, stderr = run(
            {"tool_input": {"command": "x"}},
            [KEY_SPAN, EMAIL_SPAN],
            mode="claude-pretool",
            action_mode="block",
        )
        decision = json.loads(stdout)["hookSpecificOutput"]
        self.assertEqual(decision["permissionDecision"], "deny")
        self.assertIn(
            "Only PII_LEVEL=off would allow this.", decision["permissionDecisionReason"]
        )
        self.assertTrue(
            stderr.endswith("pii-check: blocked 2 span(s) in 3ms at level=standard\n")
        )

    def test_an_allowed_label_leaves_the_level(self) -> None:
        stdout, stderr = run(
            {"prompt": "x"}, [EMAIL_SPAN], allow_labels=" private_email , secret"
        )
        self.assertEqual(stdout, "")
        self.assertIn("PII below level: [private_email(moderate)]", stderr)

    def test_off_and_the_bypass_prefix_skip_the_detector(self) -> None:
        self.assertEqual(run({"prompt": "x"}, [KEY_SPAN], level="off"), ("", ""))
        self.assertEqual(run({"prompt": "pii:off x"}, [KEY_SPAN]), ("", ""))
        self.assertNotEqual(
            run({"prompt": "pii:off x"}, [KEY_SPAN], allow_bypass="0"), ("", "")
        )

    def test_the_bypass_prefix_does_not_apply_to_tool_output(self) -> None:
        stdout, _ = run(
            {"tool_response": {"stdout": "pii:off x"}},
            [KEY_SPAN],
            mode="claude-posttool",
        )
        self.assertIn("in tool output", stdout)

    def test_failures_come_back_as_statuses(self) -> None:
        cases = [
            (dict(server_mode="redact"), b'{"prompt": "x"}', [KEY_SPAN], 409),
            ({}, b"not json", [KEY_SPAN], 422),
            ({}, b'{"prompt": "x"}', [1, 2], 502),
        ]
        for overrides, payload, spans, status in cases:
            with self.subTest(status=status), self.assertRaises(HookError) as raised:
                answer(
                    payload,
                    policy(**overrides),
                    "tagger",
                    1000,
                    lambda text: (spans, 1),
                )
            self.assertEqual(raised.exception.status, status)

    def test_a_text_over_the_body_limit_is_refused(self) -> None:
        with self.assertRaises(HookError) as raised:
            answer(
                b'{"prompt": "' + b"x" * 100 + b'"}',
                policy(),
                "tagger",
                50,
                lambda text: ([], 1),
            )
        self.assertEqual(raised.exception.status, 413)

    def test_jq_escapes_delete_as_a_control_character(self) -> None:
        span = {**KEY_SPAN, "text": "AKIA\x7f" + "Q3EGRZ7MXN2PLW4T"}
        stdout, _ = run({"prompt": "x"}, [span])
        self.assertNotIn("\x7f", stdout)


if __name__ == "__main__":
    unittest.main()
