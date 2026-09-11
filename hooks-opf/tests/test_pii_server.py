from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path
from unittest import TestCase
from unittest.mock import patch


HOOKS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HOOKS_DIR))

MODULE_PATH = HOOKS_DIR / "pii-server.py"
MODULE_SPEC = importlib.util.spec_from_file_location("pii_server", MODULE_PATH)
if MODULE_SPEC is None or MODULE_SPEC.loader is None:
    raise RuntimeError(f"cannot load server module: {MODULE_PATH}")
PII_SERVER = importlib.util.module_from_spec(MODULE_SPEC)
MODULE_SPEC.loader.exec_module(PII_SERVER)


class PiiServerModeTests(TestCase):
    def test_default_mode_is_redact(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(PII_SERVER.resolve_mode(), "redact")

    def test_environment_selects_openai_mode(self) -> None:
        with patch.dict(os.environ, {"PII_SERVER_MODE": "openai"}, clear=True):
            self.assertEqual(PII_SERVER.resolve_mode(), "openai")

    def test_argument_overrides_environment(self) -> None:
        with patch.dict(os.environ, {"PII_SERVER_MODE": "openai"}, clear=True):
            self.assertEqual(PII_SERVER.resolve_mode("redact"), "redact")

    def test_invalid_mode_fails_explicitly(self) -> None:
        with self.assertRaisesRegex(ValueError, "PII_SERVER_MODE"):
            PII_SERVER.resolve_mode("unknown")

    def test_health_payload_identifies_mode_and_device(self) -> None:
        model = type("Model", (), {"device": "mps"})()
        self.assertEqual(
            PII_SERVER.health_payload("redact", model),
            {"status": "ok", "mode": "redact", "device": "mps"},
        )
        self.assertEqual(
            PII_SERVER.health_payload("openai", model),
            {"status": "ok", "mode": "openai", "device": "cpu"},
        )

    def test_environment_selects_rules_mode(self) -> None:
        with patch.dict(os.environ, {"PII_SERVER_MODE": "rules"}, clear=True):
            self.assertEqual(PII_SERVER.resolve_mode(), "rules")

    def test_rules_mode_reports_no_accelerator(self) -> None:
        self.assertEqual(
            PII_SERVER.health_payload("rules", PII_SERVER.RulesModel()),
            {"status": "ok", "mode": "rules", "device": "cpu"},
        )


class RulesModelTests(TestCase):
    def test_empty_input_returns_no_spans(self) -> None:
        self.assertEqual(PII_SERVER.RulesModel().predict(""), [])

    def test_secret_is_found_without_a_checkpoint(self) -> None:
        spans = PII_SERVER.RulesModel().predict(
            "Staging key sk_live_9Kx3Lm2Qp7Yt4Nf8Rw1Zc6Vb sits in .env."
        )
        self.assertEqual([span["label"] for span in spans], ["secret"])

    def test_email_and_phone_are_found(self) -> None:
        spans = PII_SERVER.RulesModel().predict(
            "Ping dana.reyes@example.org or +1 (415) 555-0134."
        )
        self.assertEqual(
            sorted(span["label"] for span in spans),
            ["private_email", "private_phone"],
        )

    def test_person_name_is_not_found(self) -> None:
        self.assertEqual(
            PII_SERVER.RulesModel().predict("Add Avery Coleman to the invite."),
            [],
        )

    def test_rules_module_pulls_no_model_framework(self) -> None:
        probe = (
            "import sys; sys.path.insert(0, %r); import pii_rules; "
            "print(sorted(m for m in ('torch', 'numpy', 'onnxruntime', 'tokenizers') "
            "if m in sys.modules))" % str(HOOKS_DIR)
        )
        result = subprocess.run(
            [sys.executable, "-c", probe], capture_output=True, text=True, check=True
        )
        self.assertEqual(result.stdout.strip(), "[]")

    def test_server_module_pulls_no_model_framework(self) -> None:
        probe = (
            "import importlib.util, sys; sys.path.insert(0, %r); "
            "spec = importlib.util.spec_from_file_location('pii_server', %r); "
            "module = importlib.util.module_from_spec(spec); "
            "spec.loader.exec_module(module); "
            "print(sorted(m for m in ('torch', 'numpy', 'onnxruntime', 'tokenizers') "
            "if m in sys.modules))" % (str(HOOKS_DIR), str(MODULE_PATH))
        )
        result = subprocess.run(
            [sys.executable, "-c", probe], capture_output=True, text=True, check=True
        )
        self.assertEqual(result.stdout.strip(), "[]")
