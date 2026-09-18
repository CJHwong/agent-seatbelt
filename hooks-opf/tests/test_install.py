#!/usr/bin/env python3
"""Tests for install.sh, run against a throwaway HOME.

install.sh is the only thing that writes the agent configuration, and it is the one
place where a mistake is silent: a hook entry with the wrong command or matcher
never runs, and nothing tells anyone. So these tests assert on the configuration
that comes out, not on the fact that the script exited zero.

Every test runs the real script with three things pinned, which is all the
isolation it needs:

  HOME                 a temporary directory, so every write lands there and the
                       real ~/.claude and ~/.codex are never referenced
  HOOKS_OPF_BASE_URL   file:// pointing at this checkout, so nothing reaches the
                       network and the installed files are this repo's
  --no-pilot           so no model is resolved, downloaded, or left running

No test starts a server, downloads anything, or needs a model dependency.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time
import unittest

from hook_harness import free_port


TESTS_DIR = Path(__file__).resolve().parent
HOOKS_DIR = TESTS_DIR.parent
INSTALL = HOOKS_DIR / "install.sh"
SOURCE_URL = f"file://{HOOKS_DIR}"
FAKE_SERVER = TESTS_DIR / "fake_pii_server.py"

# Pin the interpreter by absolute path. subprocess resolves a bare name against the
# CHILD's PATH (os.get_exec_path(env)), so the test that strips PATH to hide jq
# would otherwise swap in the system bash 3.2: that shell has no BASH_XTRACEFD, so
# cov_env.sh points stderr at the trace file and the refusal message vanishes from
# the stderr the test is asserting on.
BASH = shutil.which("bash") or "/bin/bash"

# The scripts the installer copies, and the only files it should ever create under
# a HOME apart from the two agent configuration files.
HOOK_FILES = (
    "pii-server.py",
    "redact_server.py",
    "pii_rules.py",
    "pii_opf.py",
    "pii-check.sh",
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class InstallerHarness(unittest.TestCase):
    """Runs install.sh in an isolated HOME. Holds no tests."""

    def setUp(self) -> None:
        self._temporary = tempfile.TemporaryDirectory(prefix="opf-install-test.")
        self.addCleanup(self._temporary.cleanup)
        self.home = Path(self._temporary.name)

    def add_agent(self, name: str) -> None:
        (self.home / f".{name}").mkdir(parents=True, exist_ok=True)

    def run_installer(
        self,
        *args: str,
        extra_env: dict[str, str] | None = None,
        source: Path | None = None,
        pilot: bool = False,
    ) -> subprocess.CompletedProcess[str]:
        """Run the real installer against the temporary HOME.

        `source` overrides where the scripts are fetched from, which is how the
        cold-start pilot test substitutes a fake server. `pilot` enables the pilot
        run, which is off by default because it resolves model dependencies and
        leaves a listening process behind.
        """
        environment = os.environ.copy()
        environment.update(
            {
                "HOME": str(self.home),
                "HOOKS_OPF_BASE_URL": f"file://{source or HOOKS_DIR}",
                "PII_PORT": str(free_port()),
                "PII_SERVER_LOG": str(self.home / "server.log"),
            }
        )
        for key in (
            "PII_SERVER_MODE",
            "PII_ACTION_MODE",
            "PII_LEVEL",
            "PII_SERVER_LOG",
        ):
            if extra_env is None or key not in extra_env:
                environment.pop(key, None)
        if extra_env:
            environment.update(extra_env)
        command = [BASH, str(INSTALL)]
        if not pilot:
            command.append("--no-pilot")
        command.extend(args)
        return subprocess.run(
            command,
            env=environment,
            text=True,
            capture_output=True,
            check=False,
        )

    def fake_source(self, server_script: Path | None = None) -> Path:
        """A copy of this checkout whose pii-server.py is a stand-in.

        The installer downloads whatever the base URL serves and then runs it as
        the server, so pointing the base URL here lets the pilot start something
        that answers /health in milliseconds instead of loading a model.
        """
        source = self.home / "source"
        source.mkdir()
        for name in HOOK_FILES:
            if name != "pii-server.py":
                shutil.copy2(HOOKS_DIR / name, source / name)
        shutil.copy2(server_script or FAKE_SERVER, source / "pii-server.py")
        return source

    def reap(self, port: int) -> None:
        """Kill whatever is listening. The pilot leaves its server running."""
        listing = subprocess.run(
            ["lsof", "-ti", f"tcp:{port}"], text=True, capture_output=True, check=False
        )
        for pid in listing.stdout.split():
            if pid.isdigit():
                with contextlib.suppress(OSError):
                    os.kill(int(pid), 15)

    def installed_dir(self) -> Path:
        return self.home / ".claude" / "hooks"

    def settings(self) -> dict:
        return json.loads((self.home / ".claude" / "settings.json").read_text())

    def entries(self, document: dict, event: str) -> list[dict]:
        return document["hooks"][event]

    def commands(self, document: dict, event: str) -> list[str]:
        return [
            hook["command"]
            for group in self.entries(document, event)
            for hook in group["hooks"]
        ]

    def files_under_home(self) -> set[str]:
        return {
            str(path.relative_to(self.home))
            for path in self.home.rglob("*")
            if path.is_file()
        }


class InstalledFileTests(InstallerHarness):
    """The scripts land where both agents look for them."""

    def test_all_five_scripts_are_installed(self) -> None:
        self.add_agent("claude")
        result = self.run_installer("--no-codex")

        self.assertEqual(result.returncode, 0, result.stderr)
        for name in HOOK_FILES:
            self.assertTrue((self.installed_dir() / name).is_file(), f"{name} missing")

    def test_the_hook_is_executable(self) -> None:
        self.add_agent("claude")
        self.run_installer("--no-codex")

        self.assertTrue(os.access(self.installed_dir() / "pii-check.sh", os.X_OK))

    def test_the_installed_files_are_this_checkout(self) -> None:
        """A copy that silently differs from the source is the failure to catch."""
        self.add_agent("claude")
        self.run_installer("--no-codex")

        for name in HOOK_FILES:
            self.assertEqual(
                digest(self.installed_dir() / name),
                digest(HOOKS_DIR / name),
                f"{name} was installed with different content",
            )


class ClaudeWiringTests(InstallerHarness):
    """The entries Claude Code reads."""

    def test_prompt_and_tool_use_are_both_wired(self) -> None:
        self.add_agent("claude")
        self.run_installer("--no-codex")

        document = self.settings()
        hook = str(self.installed_dir() / "pii-check.sh")
        self.assertEqual(
            self.commands(document, "UserPromptSubmit"), [f"{hook} --mode prompt"]
        )
        self.assertEqual(
            self.commands(document, "PostToolUse"), [f"{hook} --mode claude-posttool"]
        )

    def test_each_entry_carries_a_timeout(self) -> None:
        self.add_agent("claude")
        self.run_installer("--no-codex")

        for event in ("UserPromptSubmit", "PostToolUse"):
            for group in self.entries(self.settings(), event):
                for hook in group["hooks"]:
                    self.assertEqual(hook["type"], "command")
                    self.assertIsInstance(hook["timeout"], int)

    def test_the_matcher_scans_reads_and_leaves_writes_alone(self) -> None:
        """The matcher is the whole decision about which tool output gets scanned.

        Read-capable tools carry external content, so a leaked secret can arrive
        through them. Edit and Write only carry what the agent already wrote, so
        scanning them is wasted work.
        """
        self.add_agent("claude")
        self.run_installer("--no-codex")

        pattern = self.entries(self.settings(), "PostToolUse")[0]["matcher"]
        for scanned in (
            "Bash",
            "Read",
            "NotebookRead",
            "WebFetch",
            "WebSearch",
            "mcp__jira__get",
        ):
            self.assertRegex(scanned, pattern, f"{scanned} should be scanned")
        for skipped in ("Edit", "Write", "Glob", "LS", "apply_patch"):
            self.assertIsNone(
                re.fullmatch(pattern, skipped), f"{skipped} should be skipped"
            )

    def test_prompt_only_skips_the_tool_use_entry(self) -> None:
        self.add_agent("claude")
        self.run_installer("--no-codex", "--prompt-only")

        document = self.settings()
        self.assertEqual(len(self.commands(document, "UserPromptSubmit")), 1)
        self.assertNotIn("PostToolUse", document["hooks"])

    def test_the_tool_input_entry_is_wired(self) -> None:
        """The one intercept point where the value has not left yet."""
        self.add_agent("claude")
        self.run_installer("--no-codex")

        hook = str(self.installed_dir() / "pii-check.sh")
        self.assertEqual(
            self.commands(self.settings(), "PreToolUse"),
            [f"{hook} --mode claude-pretool"],
        )

    def test_the_tool_input_matcher_covers_commands_not_paths(self) -> None:
        """The matcher names tools, so one Bash entry covers wget, curl and scp.

        Read's input is a path and Write's is the agent's own content, so neither is
        worth scanning before it runs. This is the same reasoning the post-tool
        matcher uses to leave Edit and Write out.
        """
        self.add_agent("claude")
        self.run_installer("--no-codex")

        pattern = self.entries(self.settings(), "PreToolUse")[0]["matcher"]
        for scanned in (
            "Bash",
            "exec_command",
            "WebFetch",
            "WebSearch",
            "Agent",
            "Task",
            "mcp__github__push",
        ):
            self.assertRegex(scanned, pattern, f"{scanned} should be scanned")
        for skipped in ("Read", "NotebookRead", "Edit", "Write", "Glob", "LS"):
            self.assertIsNone(
                re.fullmatch(pattern, skipped), f"{skipped} should not be scanned"
            )

    def test_prompt_only_skips_the_tool_input_entry_too(self) -> None:
        self.add_agent("claude")
        self.run_installer("--no-codex", "--prompt-only")

        document = self.settings()
        self.assertNotIn("PreToolUse", document["hooks"])
        self.assertNotIn("PostToolUse", document["hooks"])


class CodexWiringTests(InstallerHarness):
    """Codex reads a different file and needs a different matcher."""

    def codex_settings(self) -> dict:
        return json.loads((self.home / ".codex" / "hooks.json").read_text())

    def test_codex_is_wired_when_present(self) -> None:
        self.add_agent("claude")
        self.add_agent("codex")
        self.run_installer()

        document = self.codex_settings()
        hook = str(self.installed_dir() / "pii-check.sh")
        self.assertEqual(
            self.commands(document, "UserPromptSubmit"), [f"{hook} --mode prompt"]
        )
        self.assertEqual(
            self.commands(document, "PostToolUse"), [f"{hook} --mode codex-posttool"]
        )

    def test_codex_uses_a_wildcard_matcher(self) -> None:
        """Codex tool identifiers vary by runtime, so the hook filters the text itself."""
        self.add_agent("codex")
        self.run_installer()

        self.assertEqual(
            self.entries(self.codex_settings(), "PostToolUse")[0]["matcher"], "*"
        )

    def test_no_codex_skips_a_present_codex(self) -> None:
        self.add_agent("claude")
        self.add_agent("codex")
        self.run_installer("--no-codex")

        self.assertFalse((self.home / ".codex" / "hooks.json").exists())

    def test_codex_gets_a_tool_input_entry_with_a_wildcard_matcher(self) -> None:
        """Codex resolves Edit, Write and apply_patch to one name and renames its shell
        tool across versions, so the wildcard is the stable choice, as for its post-tool
        entry. The hook filters the text itself."""
        self.add_agent("codex")
        self.run_installer()

        hook = str(self.installed_dir() / "pii-check.sh")
        self.assertEqual(
            self.commands(self.codex_settings(), "PreToolUse"),
            [f"{hook} --mode codex-pretool"],
        )
        self.assertEqual(
            self.entries(self.codex_settings(), "PreToolUse")[0]["matcher"], "*"
        )

    def test_both_agents_share_one_installed_script(self) -> None:
        self.add_agent("claude")
        self.add_agent("codex")
        self.run_installer()

        codex_hook = str(self.installed_dir() / "pii-check.sh")
        self.assertIn(
            f"{codex_hook} --mode prompt",
            self.commands(self.codex_settings(), "UserPromptSubmit"),
        )
        self.assertIn(
            f"{codex_hook} --mode prompt",
            self.commands(self.settings(), "UserPromptSubmit"),
        )


class IdempotencyTests(InstallerHarness):
    """Re-running is documented as safe, and it is how a matcher change propagates."""

    def test_running_twice_does_not_duplicate_an_entry(self) -> None:
        self.add_agent("claude")
        self.run_installer("--no-codex")
        self.run_installer("--no-codex")

        document = self.settings()
        self.assertEqual(len(self.commands(document, "UserPromptSubmit")), 1)
        self.assertEqual(len(self.entries(document, "PostToolUse")), 1)

    def test_a_new_matcher_replaces_the_existing_entry(self) -> None:
        """The documented upgrade path: an existing install picks up a matcher change."""
        self.add_agent("claude")
        self.run_installer("--no-codex")

        stale = self.home / ".claude" / "settings.json"
        document = json.loads(stale.read_text())
        document["hooks"]["PostToolUse"][0]["matcher"] = "old-matcher"
        stale.write_text(json.dumps(document))

        self.run_installer("--no-codex")

        self.assertEqual(len(self.entries(self.settings(), "PostToolUse")), 1)
        self.assertNotEqual(
            self.entries(self.settings(), "PostToolUse")[0]["matcher"], "old-matcher"
        )


class RefusalTests(InstallerHarness):
    """Everything the installer refuses, and what it leaves behind when it does."""

    def test_no_agent_is_refused(self) -> None:
        result = self.run_installer("--no-codex")

        self.assertEqual(result.returncode, 1)
        self.assertIn("Neither ~/.claude/ nor ~/.codex/ found", result.stderr)

    def test_a_refusal_writes_nothing(self) -> None:
        """A refusal that leaves half an install behind is worse than no installer.

        The agent check used to run after mkdir of ~/.claude/hooks, which created
        the very directory the check looked for, so on a machine with no agent the
        installer wired hooks anyway and reported success.
        """
        result = self.run_installer("--no-codex")

        self.assertEqual(result.returncode, 1)
        self.assertEqual(
            self.files_under_home(), set(), "a refused install wrote files"
        )

    def test_an_invalid_server_mode_is_refused_before_writing(self) -> None:
        self.add_agent("claude")
        result = self.run_installer(
            "--no-codex", extra_env={"PII_SERVER_MODE": "bogus"}
        )

        self.assertEqual(result.returncode, 1)
        self.assertIn("PII_SERVER_MODE must be", result.stderr)
        self.assertEqual(self.files_under_home(), set())

    def test_an_invalid_action_mode_is_refused_before_writing(self) -> None:
        self.add_agent("claude")
        result = self.run_installer(
            "--no-codex", extra_env={"PII_ACTION_MODE": "Block"}
        )

        self.assertEqual(result.returncode, 1)
        self.assertIn("PII_ACTION_MODE must be", result.stderr)
        self.assertEqual(self.files_under_home(), set())

    def test_an_unknown_argument_is_refused(self) -> None:
        self.add_agent("claude")
        result = self.run_installer("--no-codex", "--not-a-flag")

        self.assertEqual(result.returncode, 1)
        self.assertIn("unknown arg", result.stderr)
        self.assertEqual(self.files_under_home(), set())

    def test_a_missing_tool_is_refused(self) -> None:
        """need_cmd runs before anything is written, so this refusal is also clean."""
        self.add_agent("claude")
        # Drop every PATH entry that holds jq, whatever it is on this machine.
        stripped = os.pathsep.join(
            part
            for part in os.environ.get("PATH", "").split(os.pathsep)
            if part and not (Path(part) / "jq").exists()
        )
        result = self.run_installer("--no-codex", extra_env={"PATH": stripped})

        self.assertEqual(result.returncode, 1)
        self.assertIn("is required but not on PATH", result.stderr)
        self.assertEqual(self.files_under_home(), set())


class PilotTests(InstallerHarness):
    """The pilot run, which is the part that starts a process.

    The pilot exists so the first agent turn does not pay a cold model load. Only
    /health and one POST matter to it, so a stand-in server answers both in
    milliseconds: no model is resolved, downloaded, or loaded, and every server
    this class starts is reaped.
    """

    SPAN_RESPONSE = json.dumps(
        {
            "spans": [
                {
                    "start": 0,
                    "end": 16,
                    "label": "private_email",
                    "text": "pilot@example.com",
                }
            ],
            "processing_ms": 1.0,
        }
    )

    def start_fake(
        self, port: int, mode: str = "rules", response: str | None = None
    ) -> None:
        environment = os.environ.copy()
        if response is not None:
            environment["FAKE_PII_RESPONSE"] = response
        server = subprocess.Popen(
            [sys.executable, str(FAKE_SERVER), "--port", str(port), "--mode", mode],
            env=environment,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        # Terminate AND wait: terminating alone leaves the process alive long enough
        # to be garbage collected still running, which surfaces as a ResourceWarning
        # and can outlive the test process.
        self.addCleanup(self.stop, server)
        self.addCleanup(self.reap, port)
        for _ in range(200):
            probe = subprocess.run(
                ["curl", "-sSf", "--max-time", "1", f"http://127.0.0.1:{port}/health"],
                capture_output=True,
                check=False,
            )
            if probe.returncode == 0:
                return
            time.sleep(0.05)
        self.fail("the stand-in server never became healthy")

    @staticmethod
    def stop(server: subprocess.Popen) -> None:
        if server.poll() is not None:
            return
        server.terminate()
        try:
            server.wait(timeout=5)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait(timeout=5)

    def test_pilot_reports_an_already_warm_server(self) -> None:
        self.add_agent("claude")
        port = free_port()
        self.start_fake(port, mode="rules")

        result = self.run_installer(
            "--no-codex",
            pilot=True,
            extra_env={"PII_PORT": str(port), "PII_SERVER_MODE": "rules"},
        )

        self.assertIn("server already warm", result.stdout)
        self.assertIn("Model is warm", result.stdout)

    def test_pilot_reports_a_mode_mismatch(self) -> None:
        """A server on the port running the wrong mode is worth saying out loud."""
        self.add_agent("claude")
        port = free_port()
        self.start_fake(port, mode="redact")

        result = self.run_installer(
            "--no-codex",
            pilot=True,
            extra_env={"PII_PORT": str(port), "PII_SERVER_MODE": "rules"},
        )

        self.assertIn("server mode is redact, requested rules", result.stderr)
        self.assertIn("First matching prompt may be slow", result.stdout)

    def test_pilot_starts_a_cold_server_and_smoke_tests_it(self) -> None:
        self.add_agent("claude")
        port = free_port()
        self.addCleanup(self.reap, port)

        result = self.run_installer(
            "--no-codex",
            pilot=True,
            source=self.fake_source(),
            extra_env={
                "PII_PORT": str(port),
                "PII_SERVER_MODE": "rules",
                "FAKE_PII_RESPONSE": self.SPAN_RESPONSE,
            },
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("server warm, smoke test flagged 1 span(s)", result.stdout)
        self.assertIn("Model is warm", result.stdout)

    def test_pilot_starts_a_cold_server_with_uv_for_a_model_mode(self) -> None:
        """The non-rules branch resolves dependencies through uv before starting."""
        self.add_agent("claude")
        port = free_port()
        self.addCleanup(self.reap, port)

        result = self.run_installer(
            "--no-codex",
            pilot=True,
            source=self.fake_source(),
            extra_env={
                "PII_PORT": str(port),
                "PII_SERVER_MODE": "redact",
                "FAKE_PII_RESPONSE": self.SPAN_RESPONSE,
            },
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("resolving deps + loading the redact model", result.stdout)
        self.assertIn("server warm, smoke test flagged 1 span(s)", result.stdout)

    def test_pilot_reports_a_server_that_flagged_nothing(self) -> None:
        """A server that answers but finds nothing is a misconfiguration, not a pass."""
        self.add_agent("claude")
        port = free_port()
        self.addCleanup(self.reap, port)

        result = self.run_installer(
            "--no-codex",
            pilot=True,
            source=self.fake_source(),
            extra_env={
                "PII_PORT": str(port),
                "PII_SERVER_MODE": "rules",
                "FAKE_PII_RESPONSE": '{"spans": [], "processing_ms": 1}',
            },
        )

        self.assertIn("smoke test flagged nothing", result.stderr)
        self.assertIn("First matching prompt may be slow", result.stdout)

    def test_pilot_gives_up_when_the_server_never_answers(self) -> None:
        """The poll is 120 tries at half a second, so this test stands in for sleep.

        What is under test is the loop's logic, which is that a server that never
        answers is reported and the install continues rather than failing.
        """
        self.add_agent("claude")
        dead = self.home / "dead_server.py"
        dead.write_text("import sys\nsys.exit(0)\n")

        stub_dir = self.home / "stubs"
        stub_dir.mkdir()
        stub = stub_dir / "sleep"
        stub.write_text("#!/bin/sh\nexit 0\n")
        stub.chmod(0o755)

        result = self.run_installer(
            "--no-codex",
            pilot=True,
            source=self.fake_source(server_script=dead),
            extra_env={
                "PII_SERVER_MODE": "rules",
                "PATH": f"{stub_dir}{os.pathsep}{os.environ['PATH']}",
            },
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("server not up after ~60s", result.stderr)
        self.assertIn("First matching prompt may be slow", result.stdout)


class SideEffectTests(InstallerHarness):
    """What the installer writes, and what it must not."""

    def test_it_writes_only_inside_the_home_it_was_given(self) -> None:
        self.add_agent("claude")
        self.run_installer("--no-codex")

        self.assertEqual(
            self.files_under_home(),
            {
                ".claude/settings.json",
                *{f".claude/hooks/{name}" for name in HOOK_FILES},
            },
        )

    def test_the_real_agent_configuration_is_untouched(self) -> None:
        """A tripwire for a future hardcoded path.

        The installer must reach the real ~/.claude only when HOME points at it.
        This runs it with a temporary HOME and checks the real file's bytes.
        """
        real_settings = Path(os.environ["HOME"]) / ".claude" / "settings.json"
        if not real_settings.is_file():
            self.skipTest("this machine has no real ~/.claude/settings.json to protect")
        before = digest(real_settings)

        self.add_agent("claude")
        self.run_installer("--no-codex")

        self.assertEqual(
            digest(real_settings), before, "the real settings.json changed"
        )

    def test_no_server_is_started(self) -> None:
        self.add_agent("claude")
        port = free_port()
        self.run_installer("--no-codex", extra_env={"PII_PORT": str(port)})

        listing = subprocess.run(
            ["lsof", "-ti", f"tcp:{port}"], text=True, capture_output=True, check=False
        )
        self.assertEqual(
            listing.stdout.strip(), "", "the installer left a server running"
        )

    def test_rules_mode_installs_without_a_model_toolchain(self) -> None:
        """Rules mode is the one path that must work with no uv and no accelerator."""
        self.add_agent("claude")
        result = self.run_installer(
            "--no-codex", extra_env={"PII_SERVER_MODE": "rules"}
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(self.commands(self.settings(), "UserPromptSubmit")), 1)


class HelpTests(InstallerHarness):
    """--help is the one path that must work before anything is written."""

    def test_help_prints_the_header_and_writes_nothing(self) -> None:
        self.add_agent("claude")
        result = self.run_installer("--help")

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Installs hooks-opf", result.stdout)
        self.assertEqual(self.files_under_home(), set())


if __name__ == "__main__":
    unittest.main()
