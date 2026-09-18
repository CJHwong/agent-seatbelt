#!/usr/bin/env python3
"""Canaries: does each control actually refuse?

Every other test in this directory asserts a *response*: the fields, the shape, the
wording. None asserts the *effect*, and a hook has two ways to look right while
stopping nothing. The runtime can drop its decision, and the wrapper that invokes it
can swallow the decision before the runtime ever sees it. The second kind is easy to
write by accident and it fails silently, so a shape test cannot catch it.

So these run the real CLI against a scripted API and assert that the effect did not
happen. The scripted API is what makes it a test rather than a demonstration: with a
real model in the loop, the model decides whether to call the tool at all and may
refuse a prompt that reads like a probe, so an absent side effect proves nothing.

Each control is measured twice, once off and once on, for that same reason. "The
sentinel is absent" alone cannot tell a refused call from a call that never happened.

Measured on Claude Code 2.1.276: PreToolUse refuses on a nested permissionDecision,
on a top-level decision, and on exit code 2. The hook emits the nested form because
Claude Code documents it, not because the others fail today.

These need the `claude` CLI and skip without it. They use no model and no API quota:
the stub answers.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from http.server import ThreadingHTTPServer
from urllib.error import URLError
from urllib.request import urlopen

from hook_harness import FakePiiHandler, free_port


TESTS_DIR = Path(__file__).resolve().parent
HOOKS_DIR = TESTS_DIR.parent
STUB = TESTS_DIR / "stub_anthropic_api.py"

# Not under /tmp. Claude Code resolves a project settings path under /tmp to
# /private/tmp, reports it as a broken symlink, and loads no hooks from it, so a
# canary placed there measures nothing at all.
PROJECT_ROOT = Path.home() / ".cache"


class CanaryHarness(unittest.TestCase):
    """Runs one control against the real CLI. Holds no tests."""

    @classmethod
    def setUpClass(cls) -> None:
        if shutil.which("claude") is None:
            raise unittest.SkipTest("the claude CLI is not on PATH; a canary needs it")
        cls.detector = ThreadingHTTPServer(("127.0.0.1", 0), FakePiiHandler)
        cls.detector_port = cls.detector.server_address[1]
        cls.detector_thread = threading.Thread(
            target=cls.detector.serve_forever, daemon=True
        )
        cls.detector_thread.start()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.detector.shutdown()
        cls.detector.server_close()
        cls.detector_thread.join(timeout=2)

    def setUp(self) -> None:
        PROJECT_ROOT.mkdir(parents=True, exist_ok=True)
        self._project_dir = tempfile.TemporaryDirectory(
            prefix="opf-canary.", dir=PROJECT_ROOT
        )
        self.addCleanup(self._project_dir.cleanup)
        self.project = Path(self._project_dir.name)
        self.sentinel = self.project / "sentinel"
        self.stub_port = free_port()
        # A marker in the prompt, so the stub can say whether the prompt reached the
        # API. Counting requests cannot: the CLI may ask for a session title even when
        # the prompt hook blocked.
        self.marker = "canary-marker-9f3a1c"

    def wire(self, event: str, hook_mode: str, action_mode: str) -> None:
        """Point one event at the repo's hook, with the mode inside the command.

        The mode goes in the command rather than the launching environment because an
        inherited variable did not reach the hook in an earlier attempt, and the run
        then measured the default mode while appearing to measure the chosen one.

        `event` is the settings key the CLI dispatches on. `hook_mode` is the value
        install.sh passes to `--mode`, which is a separate name.
        """
        settings = self.project / ".claude" / "settings.json"
        settings.parent.mkdir(parents=True, exist_ok=True)
        command = (
            f"PII_PORT={self.detector_port} PII_SERVER_MODE=redact "
            f"PII_ACTION_MODE={action_mode} PII_LEVEL=strict "
            f"PII_SERVER_LOG={self.project}/server.log "
            f"bash {HOOKS_DIR}/pii-check.sh --mode {hook_mode}"
        )
        settings.write_text(
            json.dumps(
                {
                    "hooks": {
                        event: [
                            {
                                "matcher": "*",
                                "hooks": [
                                    {
                                        "type": "command",
                                        "command": command,
                                        "timeout": 20,
                                    }
                                ],
                            }
                        ]
                    },
                    "permissions": {"allow": ["Bash"]},
                }
            )
        )

    def start_stub(self) -> None:
        self.stub = subprocess.Popen(
            [
                sys.executable,
                str(STUB),
                "--port",
                str(self.stub_port),
                "--command",
                f"touch {self.sentinel}",
                "--marker",
                self.marker,
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        self.addCleanup(self.stop_stub, self.stub)

    @staticmethod
    def stop_stub(stub: subprocess.Popen) -> None:
        if stub.poll() is None:
            stub.terminate()
            try:
                stub.wait(timeout=5)
            except subprocess.TimeoutExpired:
                stub.kill()
                stub.wait(timeout=5)

    def stub_state(self) -> dict:
        for _ in range(40):
            try:
                with urlopen(f"http://127.0.0.1:{self.stub_port}/", timeout=1) as r:
                    return json.loads(r.read())
            except (URLError, OSError):
                time.sleep(0.05)
        self.fail("the stub never answered")

    def run_cli(self, prompt: str) -> subprocess.CompletedProcess[str]:
        environment = os.environ.copy()
        environment.update(
            {
                "ANTHROPIC_BASE_URL": f"http://127.0.0.1:{self.stub_port}",
                "ANTHROPIC_API_KEY": "stub",
                "ANTHROPIC_MODEL": "claude-sonnet-5",
            }
        )
        return subprocess.run(
            ["claude", "-p", prompt],
            cwd=self.project,
            env=environment,
            stdin=subprocess.DEVNULL,
            text=True,
            capture_output=True,
            check=False,
            timeout=180,
        )


class ControlRefusalTests(CanaryHarness):
    """A control exists only when its refusal is observable."""

    def test_the_control_runs_when_it_is_off(self) -> None:
        """The baseline. Without it, a missing sentinel proves nothing.

        If the harness cannot make the sentinel appear with the control disabled, then
        every other assertion here is measuring the harness, not the hook.
        """
        self.wire("PreToolUse", "claude-pretool", "warn")
        self.start_stub()

        self.run_cli("anything")

        self.assertTrue(
            self.sentinel.exists(),
            "the sentinel never appeared even with the control in warn mode, so this "
            "harness cannot tell a block from a run that never happened",
        )

    def test_pretooluse_deny_stops_the_command(self) -> None:
        """The canary for the defect this suite exists for.

        A shape assertion passed while the command ran: the hook returned a perfectly
        formed block that the runtime dropped silently.
        """
        self.wire("PreToolUse", "claude-pretool", "block")
        self.start_stub()

        self.run_cli("anything")

        self.assertFalse(self.sentinel.exists(), "the control did not stop the command")
        self.assertGreater(
            self.stub_state()["requests"],
            0,
            "the stub was never asked for anything, so no tool call was attempted and "
            "the missing sentinel says nothing",
        )

    def test_the_prompt_control_runs_when_it_is_off(self) -> None:
        """The baseline for the prompt path."""
        self.wire("UserPromptSubmit", "prompt", "warn")
        self.start_stub()

        self.run_cli(f"please remember {self.marker}")

        self.assertTrue(
            self.sentinel.exists(),
            "the prompt did not reach the tool even with the control in warn mode, so "
            "a missing sentinel proves nothing about the block",
        )

    def test_prompt_block_stops_the_turn(self) -> None:
        """A blocked prompt is never processed.

        The assertion is the sentinel, not the marker. Claude Code asks the provider
        for a session title before any hook runs, and that call carries the prompt
        text, so the marker reaches the API even when the block works. Asserting on it
        would assert a guarantee this runtime does not offer. The turn never starting
        is the guarantee the hook does offer, and a tool call is what proves it ran.
        """
        self.wire("UserPromptSubmit", "prompt", "block")
        self.start_stub()

        self.run_cli(f"please remember {self.marker}")

        self.assertFalse(
            self.sentinel.exists(),
            f"the turn ran although the prompt hook blocked it; the stub saw "
            f"{self.stub_state()['detail']}",
        )


if __name__ == "__main__":
    unittest.main()
