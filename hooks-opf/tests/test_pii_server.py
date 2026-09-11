from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from unittest import TestCase
from unittest.mock import patch


MODULE_PATH = Path(__file__).resolve().parents[1] / "pii-server.py"
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
