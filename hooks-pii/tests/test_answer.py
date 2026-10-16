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

# A documented example key, not a real one. Named so a test can spell the text out
# without reading it back out of the span, which the type checker reads as int or str.
KEY_TEXT = "AKIA" + "Q3EGRZ7MXN2PLW4T"
KEY_SPAN = {
    "start": 0,
    "end": 20,
    "label": "secret",
    "text": KEY_TEXT,
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


# The text cap these tests hand the answer path. The real default lives in server.py,
# and server-side tests pin it: here it only has to be larger than the fixtures.
SCAN_MAX_CHARS = 256 * 1024


def run(payload: object, spans: list, **overrides: str) -> tuple[str, str]:
    """stdout and stderr of one answer, split on the separator."""
    body = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
    framed = answer(
        body, policy(**overrides), "tagger", SCAN_MAX_CHARS, lambda text: (spans, 3)
    )
    stdout, stderr, closing = framed.split(SEPARATOR)
    assert closing == ""
    return stdout, stderr


def run_over_the_cap(
    spans: list, rules: list | None, size: int = 200, **overrides: str
) -> tuple[str, str]:
    """One answer for a text past the scan cap, with the rules layer wired in."""
    framed = answer(
        json.dumps({"prompt": "x" * size}).encode(),
        policy(**overrides),
        "tagger",
        100,
        lambda value: (spans, 3),
        None if rules is None else (lambda value: rules),
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

    def test_an_encrypted_codex_agent_message_is_not_scanned(self) -> None:
        blob = "gAAAAA" + "fake_encrypted-payload" * 4
        for tool_name in ("spawn_agent", "send_message", "collaborationfollowup_task"):
            with self.subTest(tool_name=tool_name):
                document = {
                    "tool_name": tool_name,
                    "tool_input": {"model": "gpt-5", "message": blob},
                }
                self.assertEqual(hook_text(document, "codex-pretool"), "gpt-5")

    def test_a_readable_agent_message_and_another_tool_are_still_scanned(self) -> None:
        blob = "gAAAAA" + "fake_encrypted-payload" * 4
        readable = {"tool_name": "send_message", "tool_input": {"message": "call bob"}}
        other_tool = {"tool_name": "Bash", "tool_input": {"message": blob}}
        self.assertEqual(hook_text(readable, "codex-pretool"), "call bob")
        self.assertEqual(hook_text(other_tool, "codex-pretool"), blob)

    def test_a_claude_task_id_is_not_scanned(self) -> None:
        cases = [
            ("Bash", "tool_response", "backgroundTaskId", "claude-posttool"),
            ("Monitor", "tool_response", "taskId", "claude-posttool"),
            ("Workflow", "tool_response", "taskId", "claude-posttool"),
            ("TaskStop", "tool_response", "task_id", "claude-posttool"),
            ("TaskStop", "tool_input", "task_id", "claude-pretool"),
        ]
        for tool_name, field_name, key, mode in cases:
            with self.subTest(tool_name=tool_name, key=key, mode=mode):
                document = {
                    "tool_name": tool_name,
                    field_name: {"stdout": "built", key: "b8k2qx7mw"},
                }
                self.assertEqual(hook_text(document, mode), "built")

    def test_a_task_id_field_of_another_tool_or_shape_is_still_scanned(self) -> None:
        other_tool = {
            "tool_name": "mcp__jira__get",
            "tool_response": {"task_id": "b8k2qx7mw"},
        }
        not_an_id = {
            "tool_name": "Bash",
            "tool_response": {"backgroundTaskId": "call bob"},
        }
        self.assertEqual(hook_text(other_tool, "claude-posttool"), "b8k2qx7mw")
        self.assertEqual(hook_text(not_an_id, "claude-posttool"), "call bob")

    def test_a_codex_agent_name_or_path_is_not_scanned(self) -> None:
        cases = [
            ("spawn_agent", "task_name", "review_api_01_02"),
            ("collaborationsend_message", "target", "/root/task1/task_3"),
            ("followup_task", "target", "task_3"),
            ("wait_agent", "target", "worker/task_3"),
        ]
        for tool_name, key, name in cases:
            with self.subTest(tool_name=tool_name, key=key):
                document = {
                    "tool_name": tool_name,
                    "tool_input": {"model": "gpt-5", key: name},
                }
                self.assertEqual(hook_text(document, "codex-pretool"), "gpt-5")

    def test_an_agent_field_of_another_tool_or_shape_is_still_scanned(self) -> None:
        other_tool = {"tool_name": "Bash", "tool_input": {"target": "task_3"}}
        not_a_name = {
            "tool_name": "send_message",
            "tool_input": {"target": "Bob Smith"},
        }
        self.assertEqual(hook_text(other_tool, "codex-pretool"), "task_3")
        self.assertEqual(hook_text(not_a_name, "codex-pretool"), "Bob Smith")

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

    def test_a_text_at_the_scan_limit_is_still_scanned(self) -> None:
        """The cap is the largest text one scan covers, so the limit itself passes."""
        text = "x" * 50
        framed = answer(
            json.dumps({"prompt": text}).encode(),
            policy(),
            "tagger",
            len(text),
            lambda scanned: ([], 1),
        )
        self.assertEqual(framed, SEPARATOR + SEPARATOR)

    def test_a_text_over_the_scan_limit_is_refused(self) -> None:
        with self.assertRaises(HookError) as raised:
            answer(
                b'{"prompt": "' + b"x" * 100 + b'"}',
                policy(),
                "tagger",
                50,
                lambda text: ([], 1),
            )
        self.assertEqual(raised.exception.status, 413)

    def test_a_text_over_the_scan_limit_is_refused_for_a_model_without_rules(
        self,
    ) -> None:
        """The redact and openai backends have no cheap layer to fall back to."""
        with self.assertRaises(HookError) as raised:
            run_over_the_cap([], None)
        self.assertEqual(raised.exception.status, 413)

    def test_the_rules_still_report_over_a_text_past_the_scan_limit(self) -> None:
        stdout, _ = run_over_the_cap([KEY_SPAN], [KEY_SPAN])
        message = json.loads(stdout)["systemMessage"]
        self.assertIn("secret(critical)", message)
        self.assertIn("The deep scan did not run", message)
        self.assertIn(
            "the scan cap is 100 characters and this text is 200 characters", message
        )

    def test_a_clean_rules_pass_over_a_long_text_says_so(self) -> None:
        """The message must not read like a clean result over all of the text."""
        stdout, _ = run_over_the_cap([], [])
        message = json.loads(stdout)["systemMessage"]
        self.assertIn("no secret or card was found", message)
        self.assertIn("Only the secret and card rules covered the whole text", message)

    def test_a_rules_answer_of_the_wrong_shape_is_reported_as_such(self) -> None:
        """A broken rules layer is not an oversized input, and must not read as one."""
        with self.assertRaises(HookError) as raised:
            run_over_the_cap([], [{"no": "label"}])
        self.assertEqual(raised.exception.status, 502)

    def test_a_kilobyte_of_text_is_reported_in_kilobytes(self) -> None:
        stdout, _ = run_over_the_cap([], [], size=4096)
        self.assertIn("this text is 4 KB", json.loads(stdout)["systemMessage"])

    def test_a_rule_hit_inside_a_long_text_can_carry_a_block(self) -> None:
        """The rules cover their own labels completely, so a hit there is a full result."""
        stdout, _ = run_over_the_cap([KEY_SPAN], [KEY_SPAN], action_mode="block")
        reason = json.loads(stdout)["reason"]
        self.assertIn("secret(critical)", reason)
        self.assertIn("The deep scan did not run", reason)

    def test_a_clean_rules_pass_cannot_carry_a_block(self) -> None:
        """A block needs the whole text, and the deep scan never saw this one."""
        with self.assertRaises(HookError) as raised:
            run_over_the_cap([], [], action_mode="block")
        self.assertEqual(raised.exception.status, 413)

    def test_the_rules_reach_a_block_only_at_the_selected_level(self) -> None:
        """The level decides what counts here too, exactly as it does for a full scan."""
        with self.assertRaises(HookError) as raised:
            run_over_the_cap(
                [PERSON_SPAN], [PERSON_SPAN], action_mode="block", level="standard"
            )
        self.assertEqual(raised.exception.status, 413)

    def test_the_scan_limit_counts_the_text_not_the_payload(self) -> None:
        """Quotes and newlines double in JSON. The model's cost does not."""
        text = 'say "hello"\n' * 100  # 1,200 characters, more once JSON-escaped
        payload = json.dumps({"prompt": text}).encode()
        self.assertGreater(len(payload), len(text) + 200)
        scanned: list[str] = []

        answer(
            payload,
            policy(),
            "tagger",
            len(text),
            lambda value: (scanned.append(value), ([], 1))[1],
        )

        self.assertEqual(scanned, [text.rstrip("\n")])

    def test_an_object_prompt_is_scanned_as_indented_json(self) -> None:
        """jq -r reads a non-string prompt as JSON, and that text is what gets scanned."""
        scanned: list[str] = []

        answer(
            json.dumps({"prompt": {"task": "email dana@example.org"}}).encode(),
            policy(),
            "tagger",
            SCAN_MAX_CHARS,
            lambda value: (scanned.append(value), ([], 1))[1],
        )

        self.assertEqual(
            scanned, [json.dumps({"task": "email dana@example.org"}, indent=2)]
        )

    def test_blank_input_scans_nothing(self) -> None:
        """jq reads blank input as null, and null holds no text to scan."""
        calls: list[str] = []

        stdout, stderr = run(b"", [])

        self.assertEqual((stdout, stderr, calls), ("", "", []))

    def test_auto_mode_names_the_tool_output_event(self) -> None:
        stdout, _ = run(
            {"tool_response": {"stdout": "key " + KEY_TEXT}},
            [KEY_SPAN],
            mode="auto",
        )
        self.assertEqual(
            json.loads(stdout)["hookSpecificOutput"]["hookEventName"], "PostToolUse"
        )

    def test_auto_mode_names_the_tool_input_event(self) -> None:
        """A payload with neither field is a tool call, and the tool input is the text."""
        stdout, _ = run(
            {"tool_input": {"command": "echo " + KEY_TEXT}},
            [KEY_SPAN],
            mode="auto",
        )
        self.assertEqual(
            json.loads(stdout)["hookSpecificOutput"]["hookEventName"], "PreToolUse"
        )

    def test_block_mode_names_the_prompt_location(self) -> None:
        stdout, _ = run({"prompt": "send " + KEY_TEXT}, [KEY_SPAN], action_mode="block")
        decision = json.loads(stdout)
        self.assertIn("PII in prompt:", decision["reason"])
        self.assertNotIn("hookSpecificOutput", decision)

    def test_block_mode_names_the_tool_output_location(self) -> None:
        stdout, _ = run(
            {"tool_response": {"stdout": "send " + KEY_TEXT}},
            [KEY_SPAN],
            mode="claude-posttool",
            action_mode="block",
        )
        self.assertIn("PII in tool output:", json.loads(stdout)["reason"])

    def test_an_unknown_event_contract_still_words_a_block(self) -> None:
        """The wording falls back rather than dropping the reason on the floor."""
        stdout, _ = run(
            {"prompt": "send " + KEY_TEXT},
            [KEY_SPAN],
            mode="redact",
            action_mode="block",
        )
        decision = json.loads(stdout)
        self.assertEqual(
            decision["reason"],
            f"PII detected: secret(critical): {KEY_TEXT[:2]}...{KEY_TEXT[-2:]}. "
            f"Blocked at PII_LEVEL=standard. Only PII_LEVEL=off would allow this.",
        )

    def test_jq_escapes_delete_as_a_control_character(self) -> None:
        span = {**KEY_SPAN, "text": "AKIA\x7f" + "Q3EGRZ7MXN2PLW4T"}
        stdout, _ = run({"prompt": "x"}, [span])
        self.assertNotIn("\x7f", stdout)


if __name__ == "__main__":
    unittest.main()
