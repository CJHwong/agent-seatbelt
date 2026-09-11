#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "numpy>=1.24,<3",
#     "tokenizers>=0.15,<1",
#     "torch>=2.2,<3",
#     "transformers>=4.40,<6",
# ]
# ///
"""Unit tests for the local Redact candidate."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from redact_server import (
    choose_device,
    chunk_token_ranges,
    deterministic_spans,
    ensure_assets,
    map_model_label,
    merge_spans,
    RedactModel,
)


class DeviceSelectionTests(unittest.TestCase):
    @patch("redact_server.torch.backends.mps.is_available", return_value=True)
    @patch("redact_server.torch.cuda.is_available", return_value=False)
    def test_auto_selects_mps_on_apple_gpu(self, cuda_available, mps_available):
        self.assertEqual(choose_device("auto").type, "mps")

    @patch("redact_server.torch.backends.mps.is_available", return_value=False)
    @patch("redact_server.torch.cuda.is_available", return_value=True)
    def test_auto_selects_cuda_before_mps(self, cuda_available, mps_available):
        self.assertEqual(choose_device("auto").type, "cuda")

    def test_cpu_requires_explicit_selection(self):
        self.assertEqual(choose_device("cpu").type, "cpu")

    @patch("redact_server.torch.backends.mps.is_available", return_value=False)
    @patch("redact_server.torch.cuda.is_available", return_value=False)
    def test_auto_rejects_missing_accelerator(self, cuda_available, mps_available):
        with self.assertRaisesRegex(RuntimeError, "accelerator"):
            choose_device("auto")


class AssetValidationTests(unittest.TestCase):
    def test_missing_pytorch_cache_fails_with_setup_message(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            with patch("redact_server.CACHE_DIR", Path(temporary_directory)):
                with self.assertRaisesRegex(FileNotFoundError, "redact.pt"):
                    ensure_assets()


class LabelMappingTests(unittest.TestCase):
    def test_redact_labels_map_to_hook_policy(self):
        expected = {
            "GIVEN_NAME": "private_person",
            "SURNAME": "private_person",
            "STREET_NAME": "private_address",
            "EMAIL": "private_email",
            "PHONE": "private_phone",
            "URL": "private_url",
            "CREDIT_CARD": "account_number",
            "BANK_ACCOUNT": "account_number",
            "SSN": "account_number",
        }
        for source_label, target_label in expected.items():
            with self.subTest(source_label=source_label):
                self.assertEqual(map_model_label(source_label), target_label)

    def test_unneeded_label_is_ignored(self):
        self.assertIsNone(map_model_label("ORG"))
        self.assertIsNone(map_model_label("unknown"))


class DeterministicDetectionTests(unittest.TestCase):
    def test_detects_fixture_sensitive_values(self):
        text = (
            "Email avery@example.com, use https://internal.example.test/x, "
            "call +1 (415) 555-0186, charge 4242 4242 4242 4242, "
            "and keep June 3, 2026 private. Key AKIAIOSFODNN7EXAMPLE."
        )
        spans = deterministic_spans(text)
        labels = {span["label"] for span in spans}
        self.assertEqual(
            labels,
            {
                "private_email",
                "private_url",
                "private_phone",
                "account_number",
                "private_date",
                "secret",
            },
        )

    def test_secret_context_captures_password_value(self):
        spans = deterministic_spans("Temporary admin password is StormGlass!492.")
        self.assertIn(
            {"label": "secret", "text": "StormGlass!492"},
            [{"label": span["label"], "text": span["text"]} for span in spans],
        )

    def test_detects_provider_and_transport_secrets(self):
        text = (
            "OpenAI sk-proj-5aB7cD9eF1gH3jK5mN7pQ9rS, "
            "GitHub github_pat_11AAbbCCDDeeFF00112233445566778899, "
            "X-Api-Key: CurlSecret7vN4xP8mL2, "
            "postgresql://worker:DbSecret7vN4xP8mL2@db.example.test/app"
        )
        secret_values = {
            span["text"]
            for span in deterministic_spans(text)
            if span["label"] == "secret"
        }
        self.assertTrue(
            {
                "sk-proj-5aB7cD9eF1gH3jK5mN7pQ9rS",
                "github_pat_11AAbbCCDDeeFF00112233445566778899",
                "CurlSecret7vN4xP8mL2",
                "DbSecret7vN4xP8mL2",
            }.issubset(secret_values)
        )

    def test_detects_structured_and_localized_secret_context(self):
        text = (
            '{"client_secret":"ClientValue7vN4xP8mL2",'
            '"refresh_token":"RefreshValue7vN4xP8mL2"} '
            "<password>XmlValue7vN4xP8mL2</password> "
            "密碼：本地測試密碼7vN4xP8mL2"
        )
        secret_values = {
            span["text"]
            for span in deterministic_spans(text)
            if span["label"] == "secret"
        }
        self.assertTrue(
            {
                "ClientValue7vN4xP8mL2",
                "RefreshValue7vN4xP8mL2",
                "XmlValue7vN4xP8mL2",
                "本地測試密碼7vN4xP8mL2",
            }.issubset(secret_values)
        )

    def test_detects_private_key_webhook_and_empty_user_dsn(self):
        text = (
            "-----BEGIN PRIVATE KEY----- "
            "https://discord.com/api/webhooks/117QvN4/7mL2kD6sF9hJ3wC5bT1yU0 "
            "redis://:redisP7vN4xL2kD6@cache.example/0"
        )
        secret_values = {
            span["text"]
            for span in deterministic_spans(text)
            if span["label"] == "secret"
        }
        self.assertTrue(
            {
                "-----BEGIN PRIVATE KEY-----",
                "https://discord.com/api/webhooks/117QvN4/7mL2kD6sF9hJ3wC5bT1yU0",
                "redisP7vN4xL2kD6",
            }.issubset(secret_values)
        )

    def test_detects_secret_value_in_csv_column(self):
        text = "service,environment,password\nworker,prod,CSVPass7vN4xP8mL2kD6"
        self.assertIn(
            "CSVPass7vN4xP8mL2kD6",
            {
                span["text"]
                for span in deterministic_spans(text)
                if span["label"] == "secret"
            },
        )

    def test_detects_new_provider_signatures_and_transport_fields(self):
        text = (
            "ASIA8Q2M4N6P8R0T2V4X6 GOCSPX-7vN4xP8mL2kD6sF9hJ3wC5bT1 "
            "sk-7vN4xP8mL2kD6sF9hJ3wC5bT1 "
            "sk-ant-7vN4xP8mL2kD6sF9hJ3wC5bT1 "
            "whsec_7vN4xP8mL2kD6sF9hJ3wC5bT1 "
            "glc_7vN4xP8mL2kD6sF9hJ3wC5bT1 "
            "snyk_7vN4xP8mL2kD6sF9hJ3wC5bT1 "
            "pul-7vN4xP8mL2kD6sF9hJ3wC5bT1 "
            "https://blob.example.test/file?sv=1&sig=SignedValue7vN4xP8mL2kD6 "
            "Set-Cookie: sid=SidCookie7vN4xP8mL2kD6sF9 "
            'secret = { value = "HclSecret7vN4xP8mL2kD6" }'
        )
        secret_values = {
            span["text"]
            for span in deterministic_spans(text)
            if span["label"] == "secret"
        }
        self.assertGreaterEqual(len(secret_values), 10)

    def test_does_not_treat_provider_like_identifiers_as_secrets(self):
        text = (
            "request_id=secret_7vN4xP8mL2kD6sF9hJ3wC5bT1 "
            "cache_key=key-7vN4xP8mL2kD6sF9hJ3wC5bT1 "
            "build_id=SK7vN4xP8mL2kD6sF9hJ3wC5bT1yU0 "
            "project_id=EAAA7vN4xP8mL2kD6sF9hJ3wC5bT1 "
            "model_id=hf_7vN4xP8mL2kD6sF9hJ3wC5bT1"
        )
        self.assertEqual(
            [],
            [span for span in deterministic_spans(text) if span["label"] == "secret"],
        )

    def test_ignores_secret_placeholders(self):
        text = (
            "password=${PASSWORD}; api_key=YOUR_API_KEY; "
            "Authorization: Bearer <token>; secret=REDACTED"
        )
        self.assertEqual(
            [],
            [span for span in deterministic_spans(text) if span["label"] == "secret"],
        )


class SpanMergeTests(unittest.TestCase):
    def test_overlap_keeps_the_higher_priority_span(self):
        text = "secret@example.com"
        spans = merge_spans(
            text,
            [
                {"start": 0, "end": len(text), "label": "private_email"},
                {"start": 0, "end": 6, "label": "secret"},
            ],
        )
        self.assertEqual([span["label"] for span in spans], ["secret"])


class ChunkingTests(unittest.TestCase):
    def test_chunk_ranges_keep_overlap_and_cover_all_tokens(self):
        self.assertEqual(
            chunk_token_ranges(8, chunk_size=4, overlap=1),
            [(0, 4), (3, 7), (6, 8)],
        )

    def test_chunk_ranges_keep_short_input_in_one_chunk(self):
        self.assertEqual(
            chunk_token_ranges(3, chunk_size=4, overlap=1),
            [(0, 3)],
        )

    def test_chunk_ranges_reject_invalid_overlap(self):
        with self.assertRaisesRegex(ValueError, "overlap"):
            chunk_token_ranges(8, chunk_size=4, overlap=4)

    def test_long_input_scans_deterministic_rules_once_and_remaps_model_spans(self):
        class StubRedactModel(RedactModel):
            def __init__(self):
                self.tokenizer = SimpleNamespace(
                    encode=lambda text, add_special_tokens=False: SimpleNamespace(
                        ids=list(range(8)),
                        offsets=[(index * 2, index * 2 + 1) for index in range(8)],
                    )
                )
                self.max_tokens = 4
                self.chunk_overlap = 1
                self.chunk_sizes = []

            def _predict_model_spans(self, token_ids, offsets, text):
                self.chunk_sizes.append(len(token_ids))
                return [{"start": 0, "end": 1, "label": "private_person"}]

        text = "a " * 8
        deterministic = [{"start": 2, "end": 3, "label": "secret"}]
        model = StubRedactModel()
        with patch(
            "redact_server.deterministic_spans", return_value=deterministic
        ) as scan:
            spans = model.predict(text)

        scan.assert_called_once_with(text)
        self.assertEqual(model.chunk_sizes, [4, 4, 2])
        self.assertEqual(
            [(span["start"], span["label"]) for span in spans],
            [
                (0, "private_person"),
                (2, "secret"),
                (6, "private_person"),
                (12, "private_person"),
            ],
        )


if __name__ == "__main__":
    unittest.main()
