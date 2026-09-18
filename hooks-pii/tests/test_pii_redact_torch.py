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

import io
import json
import os
import shutil
import signal
import socket
import struct
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from contextlib import redirect_stderr, suppress
from http.server import ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import Mock, patch

import numpy as np  # ty: ignore[unresolved-import]
import torch  # ty: ignore[unresolved-import]
from tokenizers import (  # ty: ignore[unresolved-import]
    Tokenizer,
    models,
    pre_tokenizers,
)
from transformers import (  # ty: ignore[unresolved-import]
    BertConfig,
    BertForTokenClassification,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pii_redact_torch import (
    build_transition_tables,
    choose_device,
    chunk_token_ranges,
    CONTENT_WINDOW_LENGTH,
    deterministic_spans,
    ensure_assets,
    Handler,
    InputTooLargeError,
    LabelSpace,
    labels_to_spans,
    main,
    map_model_label,
    max_body_bytes,
    MAX_SEQUENCE_LENGTH,
    merge_spans,
    MODEL_LABEL_MAP,
    NEG_INF,
    RedactModel,
    valid_transition,
    viterbi_decode,
    WINDOW_OVERLAP,
    WINDOW_STEP,
)
from pii_redact_torch import _trim_span_whitespace, _window_ranges

# The fixture cache holds a real, locally built checkpoint: a two-layer-free BERT
# with a tiny hidden size and a WordLevel tokenizer. Nothing is downloaded. The
# shapes are small enough to build in milliseconds, so the module's inference path
# runs for real instead of standing behind a stub.

FIXTURE_CACHE = Path(tempfile.mkdtemp(prefix="redact-server-fixture-"))
CJK_FIXTURE_CACHE = Path(tempfile.mkdtemp(prefix="redact-server-cjk-"))

FIXTURE_ID2LABEL = {
    0: "O",
    1: "B-GIVEN_NAME",
    2: "E-GIVEN_NAME",
    3: "S-EMAIL",
    4: "B-ORG",
    5: "E-ORG",
    6: "S-ORG",
}

FIXTURE_VOCAB = {
    "[UNK]": 0,
    "<s>": 1,
    "</s>": 2,
    "<pad>": 3,
    "hello": 4,
    "world": 5,
    "acme": 6,
    "inc": 7,
}

# Multi-byte words. One CJK character is three bytes, so a byte offset used as a
# character index would slice mojibake out of the text.
CJK_FIXTURE_VOCAB = {
    "[UNK]": 0,
    "<s>": 1,
    "</s>": 2,
    "<pad>": 3,
    "hello": 4,
    "world": 5,
    "密碼": 6,
    "本地": 7,
    "測試": 8,
}

FIXTURE_ENVIRONMENT = {
    "REDACT_MIN_SCORE": "0.0",
    "REDACT_BATCH_SIZE": "2",
    "REDACT_MAX_TOKENS": "4",
    "REDACT_CHUNK_OVERLAP_TOKENS": "1",
    "REDACT_MAX_INPUT_TOKENS": "16",
}

# The labels a completed prediction may carry: the hook policy labels the model
# label map targets, plus the two the deterministic rules emit.
HOOK_LABELS = set(MODEL_LABEL_MAP.values()) | {"private_date", "secret"}

# Consecutive windows advance by the content window minus the overlap, because
# _window_ranges sets the next start to `end - WINDOW_OVERLAP`. The module
# derives WINDOW_STEP the same way; WindowRangeTests pins the two together.


def write_fixture_cache(
    cache_dir: Path,
    checkpoint: str = "wrapped",
    vocab: dict[str, int] | None = None,
) -> None:
    """Write a loadable Redact cache into cache_dir."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    vocabulary = FIXTURE_VOCAB if vocab is None else vocab
    tokenizer = Tokenizer(models.WordLevel(vocab=dict(vocabulary), unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.save(str(cache_dir / "tokenizer.json"))
    (cache_dir / "labels.json").write_text(json.dumps(FIXTURE_ID2LABEL))
    config = BertConfig(
        vocab_size=len(vocabulary),
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=16,
        max_position_embeddings=MAX_SEQUENCE_LENGTH,
        num_labels=len(FIXTURE_ID2LABEL),
        id2label=FIXTURE_ID2LABEL,
        label2id={name: key for key, name in FIXTURE_ID2LABEL.items()},
    )
    config.to_json_file(str(cache_dir / "config.json"))
    torch.manual_seed(0)
    state_dict = BertForTokenClassification(config).state_dict()
    if checkpoint == "wrapped":
        torch.save({"state_dict": state_dict}, cache_dir / "redact.pt")
    elif checkpoint == "bare":
        torch.save(state_dict, cache_dir / "redact.pt")
    else:
        torch.save(["not", "a", "state dict"], cache_dir / "redact.pt")


def write_fixture_config_only(cache_dir: Path) -> None:
    """Write only the config file, enough for the constructor's validation paths."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    config = BertConfig(
        vocab_size=len(FIXTURE_VOCAB),
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=16,
        max_position_embeddings=MAX_SEQUENCE_LENGTH,
        num_labels=len(FIXTURE_ID2LABEL),
        id2label=FIXTURE_ID2LABEL,
        label2id={name: key for key, name in FIXTURE_ID2LABEL.items()},
    )
    config.to_json_file(str(cache_dir / "config.json"))


def build_model(
    cache_dir: Path | None = None,
    device: str | None = "cpu",
    **overrides: str,
) -> RedactModel:
    """Build a RedactModel against the fixture cache under a controlled environment."""
    environment = dict(FIXTURE_ENVIRONMENT)
    environment.update(overrides)
    with patch.dict(os.environ, environment):
        return RedactModel(cache_dir or FIXTURE_CACHE, device)


def label_id(label_space: LabelSpace, name: str) -> int:
    for identifier, label_name in label_space.id2label.items():
        if label_name == name:
            return identifier
    raise AssertionError(f"fixture has no label named {name}")


def probability_table(path: list[int], score: float, class_count: int) -> np.ndarray:
    table = np.zeros((len(path), class_count), dtype=np.float32)
    for token_index, identifier in enumerate(path):
        table[token_index, 0] = 1.0 - score
        table[token_index, identifier] = score
    return table


def setUpModule() -> None:
    write_fixture_cache(FIXTURE_CACHE)
    write_fixture_cache(CJK_FIXTURE_CACHE, vocab=CJK_FIXTURE_VOCAB)


def tearDownModule() -> None:
    shutil.rmtree(FIXTURE_CACHE, ignore_errors=True)
    shutil.rmtree(CJK_FIXTURE_CACHE, ignore_errors=True)


class DeviceSelectionTests(unittest.TestCase):
    @patch("pii_redact_torch.torch.backends.mps.is_available", return_value=True)
    @patch("pii_redact_torch.torch.cuda.is_available", return_value=False)
    def test_auto_selects_mps_on_apple_gpu(self, cuda_available, mps_available):
        self.assertEqual(choose_device("auto").type, "mps")

    @patch("pii_redact_torch.torch.backends.mps.is_available", return_value=False)
    @patch("pii_redact_torch.torch.cuda.is_available", return_value=True)
    def test_auto_selects_cuda_before_mps(self, cuda_available, mps_available):
        self.assertEqual(choose_device("auto").type, "cuda")

    def test_cpu_requires_explicit_selection(self):
        self.assertEqual(choose_device("cpu").type, "cpu")

    @patch("pii_redact_torch.torch.backends.mps.is_available", return_value=False)
    @patch("pii_redact_torch.torch.cuda.is_available", return_value=False)
    def test_auto_rejects_missing_accelerator(self, cuda_available, mps_available):
        with self.assertRaisesRegex(RuntimeError, "accelerator"):
            choose_device("auto")

    @patch("pii_redact_torch.torch.cuda.is_available", return_value=False)
    def test_requested_cuda_that_is_missing_is_an_error(self, cuda_available):
        with self.assertRaisesRegex(RuntimeError, "cuda is not available"):
            choose_device("cuda")

    @patch("pii_redact_torch.torch.cuda.is_available", return_value=True)
    def test_requested_cuda_is_used_when_present(self, cuda_available):
        self.assertEqual(choose_device("cuda").type, "cuda")

    @patch("pii_redact_torch.torch.backends.mps.is_available", return_value=False)
    def test_requested_mps_that_is_missing_is_an_error(self, mps_available):
        with self.assertRaisesRegex(RuntimeError, "mps is not available"):
            choose_device("mps")

    @patch("pii_redact_torch.torch.backends.mps.is_available", return_value=True)
    def test_requested_mps_is_used_when_present(self, mps_available):
        self.assertEqual(choose_device("mps").type, "mps")

    @patch("pii_redact_torch.torch.backends.mps.is_available", return_value=True)
    def test_mps_fallback_environment_variable_is_rejected(self, mps_available):
        with patch.dict(os.environ, {"PYTORCH_ENABLE_MPS_FALLBACK": "1"}):
            with self.assertRaisesRegex(RuntimeError, "PYTORCH_ENABLE_MPS_FALLBACK=1"):
                choose_device("mps")

    @patch("pii_redact_torch.torch.backends.mps.is_available", return_value=True)
    @patch("pii_redact_torch.torch.cuda.is_available", return_value=False)
    def test_auto_also_rejects_the_mps_fallback_variable(
        self, cuda_available, mps_available
    ):
        with patch.dict(os.environ, {"PYTORCH_ENABLE_MPS_FALLBACK": "1"}):
            with self.assertRaisesRegex(RuntimeError, "hide CPU execution"):
                choose_device("auto")

    def test_cpu_ignores_the_mps_fallback_variable(self):
        with patch.dict(os.environ, {"PYTORCH_ENABLE_MPS_FALLBACK": "1"}):
            self.assertEqual(choose_device("cpu").type, "cpu")

    def test_unknown_device_name_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "auto, cuda, mps, or cpu"):
            choose_device("tpu")

    def test_environment_variable_supplies_the_default_device(self):
        with patch.dict(os.environ, {"REDACT_DEVICE": "  CPU  "}):
            self.assertEqual(choose_device().type, "cpu")

    def test_missing_mps_backend_reads_as_unavailable(self):
        with patch.object(cast(Any, torch), "backends", SimpleNamespace()):
            with self.assertRaisesRegex(RuntimeError, "mps is not available"):
                choose_device("mps")

    def test_auto_falls_through_to_the_plain_error_without_an_mps_backend(self):
        with patch.object(cast(Any, torch), "backends", SimpleNamespace()):
            with patch("pii_redact_torch.torch.cuda.is_available", return_value=False):
                with self.assertRaisesRegex(RuntimeError, "no GPU accelerator"):
                    choose_device("auto")


class AssetValidationTests(unittest.TestCase):
    def test_missing_pytorch_cache_fails_with_setup_message(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            with patch("pii_redact_torch.CACHE_DIR", Path(temporary_directory)):
                with self.assertRaisesRegex(FileNotFoundError, "redact.pt"):
                    ensure_assets()

    def test_error_lists_every_missing_file_in_declared_order(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            cache_dir = Path(temporary_directory)
            (cache_dir / "config.json").write_text("{}")
            with patch("pii_redact_torch.CACHE_DIR", cache_dir):
                with self.assertRaises(FileNotFoundError) as raised:
                    ensure_assets()

        self.assertIn("labels.json, tokenizer.json, redact.pt", str(raised.exception))

    def test_error_names_the_configured_repository_and_revision(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            with patch("pii_redact_torch.CACHE_DIR", Path(temporary_directory)):
                with patch("pii_redact_torch.REDACT_REPO", "example/redact"):
                    with patch("pii_redact_torch.REDACT_REVISION", "v9.9.9"):
                        with self.assertRaises(FileNotFoundError) as raised:
                            ensure_assets()

        self.assertIn("example/redact@v9.9.9", str(raised.exception))

    def test_complete_cache_creates_the_directory_and_returns_it(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            cache_dir = Path(temporary_directory) / "nested" / "redact"
            write_fixture_cache(cache_dir)
            with patch("pii_redact_torch.CACHE_DIR", cache_dir):
                self.assertEqual(ensure_assets(), cache_dir)

    def test_absent_cache_directory_is_created(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            cache_dir = Path(temporary_directory) / "nested" / "redact"
            with patch("pii_redact_torch.CACHE_DIR", cache_dir):
                with self.assertRaises(FileNotFoundError):
                    ensure_assets()
            self.assertTrue(cache_dir.is_dir())


class LabelSpaceTests(unittest.TestCase):
    def test_bioes_names_split_into_tag_and_span_name(self):
        space = LabelSpace({0: "O", 1: "B-GIVEN_NAME", 2: "I-GIVEN_NAME"})

        self.assertEqual(space.tag[1], "B")
        self.assertEqual(space.span_name[1], "GIVEN_NAME")
        self.assertEqual(space.tag[2], "I")
        self.assertEqual(space.span_name[2], "GIVEN_NAME")

    def test_background_label_is_found_and_has_no_tag(self):
        space = LabelSpace({7: "O", 1: "S-EMAIL"})

        self.assertEqual(space.bg, 7)
        self.assertIsNone(space.tag[7])
        self.assertIsNone(space.span_name[7])

    def test_class_count_is_the_label_table_size(self):
        self.assertEqual(LabelSpace({0: "O", 1: "S-EMAIL"}).num_classes, 2)

    def test_span_name_without_a_hyphen_becomes_empty(self):
        space = LabelSpace({0: "O", 1: "ORGANIZATION"})

        self.assertEqual(space.tag[1], "ORGANIZATION")
        self.assertEqual(space.span_name[1], "")


class TransitionTableTests(unittest.TestCase):
    def setUp(self):
        self.space = LabelSpace(
            {
                0: "O",
                1: "B-GIVEN_NAME",
                2: "I-GIVEN_NAME",
                3: "E-GIVEN_NAME",
                4: "S-EMAIL",
                5: "B-ORG",
                6: "X-WEIRD",
            }
        )
        self.start, self.end, self.transition = build_transition_tables(self.space)

    def test_start_allows_begin_span_and_background_only(self):
        self.assertEqual(self.start[0], 0.0)
        self.assertEqual(self.start[1], 0.0)
        self.assertEqual(self.start[4], 0.0)
        self.assertEqual(self.start[2], NEG_INF)
        self.assertEqual(self.start[3], NEG_INF)

    def test_end_allows_end_span_and_background_only(self):
        self.assertEqual(self.end[0], 0.0)
        self.assertEqual(self.end[3], 0.0)
        self.assertEqual(self.end[4], 0.0)
        self.assertEqual(self.end[1], NEG_INF)
        self.assertEqual(self.end[2], NEG_INF)

    def test_transition_matrix_marks_legal_and_illegal_pairs(self):
        self.assertEqual(self.transition[0, 0], 0.0)
        self.assertEqual(self.transition[0, 1], 0.0)
        self.assertEqual(self.transition[0, 2], NEG_INF)
        self.assertEqual(self.transition[1, 2], 0.0)
        self.assertEqual(self.transition[1, 5], NEG_INF)
        self.assertEqual(self.transition[3, 0], 0.0)
        self.assertEqual(self.transition[3, 5], 0.0)
        self.assertEqual(self.transition[3, 2], NEG_INF)

    def test_background_to_background_and_to_begin(self):
        self.assertTrue(valid_transition(self.space, 0, 0))
        self.assertTrue(valid_transition(self.space, 0, 1))
        self.assertFalse(valid_transition(self.space, 0, 2))

    def test_finished_span_restarts_or_closes(self):
        self.assertTrue(valid_transition(self.space, 3, 0))
        self.assertTrue(valid_transition(self.space, 3, 1))
        self.assertFalse(valid_transition(self.space, 3, 2))

    def test_inside_span_continues_only_the_same_span_name(self):
        self.assertTrue(valid_transition(self.space, 1, 2))
        self.assertTrue(valid_transition(self.space, 1, 3))
        self.assertFalse(valid_transition(self.space, 1, 5))
        self.assertFalse(valid_transition(self.space, 1, 0))

    def test_unknown_tag_never_transitions(self):
        self.assertFalse(valid_transition(self.space, 6, 0))
        self.assertFalse(valid_transition(self.space, 1, 6))


class ViterbiTests(unittest.TestCase):
    def setUp(self):
        self.space = LabelSpace(
            {
                0: "O",
                1: "B-GIVEN_NAME",
                2: "I-GIVEN_NAME",
                3: "E-GIVEN_NAME",
            }
        )
        self.start, self.end, self.transition = build_transition_tables(self.space)

    def test_empty_input_decodes_to_an_empty_path(self):
        empty = np.zeros((0, 4), dtype=np.float32)
        self.assertEqual(
            viterbi_decode(empty, self.start, self.end, self.transition), []
        )

    def test_path_length_matches_the_token_count(self):
        log_probs = np.log(
            np.array([[0.6, 0.2, 0.1, 0.1], [0.1, 0.1, 0.2, 0.6]], dtype=np.float32)
        )
        path = viterbi_decode(log_probs, self.start, self.end, self.transition)
        self.assertEqual(len(path), 2)

    def test_illegal_start_label_is_not_chosen(self):
        # Class 2 (I-GIVEN_NAME) carries the highest score but cannot start a
        # sequence, so the decoder must pick a label the start table allows.
        log_probs = np.log(
            np.array([[0.05, 0.1, 0.8, 0.05], [0.05, 0.1, 0.8, 0.05]], dtype=np.float32)
        )
        path = viterbi_decode(log_probs, self.start, self.end, self.transition)
        self.assertNotEqual(path[0], 2)

    def test_illegal_end_label_is_not_chosen(self):
        # Symmetrically, a path may not finish inside a span.
        log_probs = np.log(
            np.array([[0.05, 0.8, 0.1, 0.05], [0.05, 0.1, 0.8, 0.05]], dtype=np.float32)
        )
        path = viterbi_decode(log_probs, self.start, self.end, self.transition)
        self.assertNotEqual(path[-1], 2)

    def test_highest_scoring_legal_path_wins(self):
        # The tokens prefer B, I then E, and that run is a legal B-GIVEN_NAME span.
        log_probs = np.log(
            np.array(
                [
                    [0.05, 0.90, 0.03, 0.02],
                    [0.05, 0.03, 0.90, 0.02],
                    [0.05, 0.03, 0.02, 0.90],
                ],
                dtype=np.float32,
            )
        )
        path = viterbi_decode(log_probs, self.start, self.end, self.transition)
        self.assertEqual(path, [1, 2, 3])


class LabelToSpanTests(unittest.TestCase):
    def setUp(self):
        self.space = LabelSpace(
            {
                0: "O",
                1: "B-GIVEN_NAME",
                2: "I-GIVEN_NAME",
                3: "E-GIVEN_NAME",
                4: "S-EMAIL",
                5: "B-ORG",
            }
        )
        self.background = 0
        self.begin = 1
        self.inside = 2
        self.finish = 3
        self.single = 4
        self.other_begin = 5

    def test_single_token_label_becomes_a_one_token_span(self):
        self.assertEqual(labels_to_spans([self.single], self.space), [(0, 0, "EMAIL")])

    def test_begin_then_end_covers_the_whole_run(self):
        spans = labels_to_spans([self.begin, self.finish], self.space)
        self.assertEqual(spans, [(0, 1, "GIVEN_NAME")])

    def test_inside_tokens_join_the_open_span(self):
        spans = labels_to_spans([self.begin, self.inside, self.finish], self.space)
        self.assertEqual(spans, [(0, 2, "GIVEN_NAME")])

    def test_background_closes_an_open_span(self):
        spans = labels_to_spans(
            [self.begin, self.inside, self.background, self.single], self.space
        )
        self.assertEqual(spans, [(0, 1, "GIVEN_NAME"), (3, 3, "EMAIL")])

    def test_unclosed_span_at_the_end_is_reported(self):
        spans = labels_to_spans([self.begin, self.inside], self.space)
        self.assertEqual(spans, [(0, 1, "GIVEN_NAME")])

    def test_single_token_label_closes_an_open_span(self):
        spans = labels_to_spans([self.begin, self.single], self.space)
        self.assertEqual(spans, [(0, 0, "GIVEN_NAME"), (1, 1, "EMAIL")])

    def test_end_label_without_a_begin_starts_at_itself(self):
        spans = labels_to_spans([self.background, self.finish], self.space)
        self.assertEqual(spans, [(1, 1, "GIVEN_NAME")])

    def test_inside_label_of_another_name_restarts_the_span(self):
        spans = labels_to_spans(
            [self.begin, self.other_begin, self.inside, self.background], self.space
        )
        self.assertEqual(spans, [(0, 0, "GIVEN_NAME"), (2, 2, "GIVEN_NAME")])

    def test_background_only_produces_no_spans(self):
        self.assertEqual(
            labels_to_spans([self.background, self.background], self.space), []
        )

    def test_empty_path_produces_no_spans(self):
        self.assertEqual(labels_to_spans([], self.space), [])


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
        self.assertEqual(
            [(span["start"], span["end"], span["label"]) for span in spans],
            [(0, 6, "secret"), (6, 18, "private_email")],
        )


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

    def test_chunk_ranges_reject_negative_overlap(self):
        with self.assertRaisesRegex(ValueError, "overlap"):
            chunk_token_ranges(8, chunk_size=4, overlap=-1)

    def test_chunk_ranges_reject_a_negative_token_count(self):
        with self.assertRaisesRegex(ValueError, "must not be negative"):
            chunk_token_ranges(-1, chunk_size=4, overlap=1)

    def test_chunk_ranges_reject_a_non_positive_chunk_size(self):
        with self.assertRaisesRegex(ValueError, "must be positive"):
            chunk_token_ranges(8, chunk_size=0, overlap=0)

    def test_chunk_ranges_of_no_tokens_are_empty(self):
        self.assertEqual(chunk_token_ranges(0, chunk_size=4, overlap=1), [])

    def test_chunk_ranges_end_exactly_on_the_last_token(self):
        self.assertEqual(
            chunk_token_ranges(4, chunk_size=4, overlap=1),
            [(0, 4)],
        )

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
                self.max_input_tokens = 32
                self.chunk_sizes = []

            def _predict_model_spans(self, token_ids, offsets, text):
                self.chunk_sizes.append(len(token_ids))
                return [{"start": 0, "end": 1, "label": "private_person"}]

        text = "a " * 8
        deterministic = [{"start": 2, "end": 3, "label": "secret"}]
        model = StubRedactModel()
        with patch(
            "pii_redact_torch.deterministic_spans", return_value=deterministic
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

    def test_input_above_the_cap_is_rejected_before_inference(self):
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
                self.max_input_tokens = 4
                self.chunk_sizes = []

            def _predict_model_spans(self, token_ids, offsets, text):
                self.chunk_sizes.append(len(token_ids))
                return []

        model = StubRedactModel()
        with patch("pii_redact_torch.deterministic_spans") as scan:
            with self.assertRaisesRegex(InputTooLargeError, "exceeds max 4"):
                model.predict("a " * 8)

        scan.assert_not_called()
        self.assertEqual(model.chunk_sizes, [])

    def test_chunk_with_no_characters_is_skipped(self):
        class StubRedactModel(RedactModel):
            def __init__(self):
                self.tokenizer = SimpleNamespace(
                    encode=lambda text, add_special_tokens=False: SimpleNamespace(
                        ids=[0, 1],
                        offsets=[(0, 0), (0, 0)],
                    )
                )
                self.max_tokens = 4
                self.chunk_overlap = 1
                self.max_input_tokens = 32
                self.chunk_sizes = []

            def _predict_model_spans(self, token_ids, offsets, text):
                self.chunk_sizes.append(len(token_ids))
                return []

        model = StubRedactModel()
        with patch("pii_redact_torch.deterministic_spans", return_value=[]):
            spans = model.predict("zero width tokens")

        self.assertEqual(model.chunk_sizes, [])
        self.assertEqual(spans, [])

    def test_empty_text_skips_the_tokenizer(self):
        model = RedactModel.__new__(RedactModel)
        model.tokenizer = SimpleNamespace(encode=Mock(side_effect=AssertionError))
        self.assertEqual(model.predict(""), [])
        model.tokenizer.encode.assert_not_called()

    def test_text_without_tokens_returns_the_rule_spans_unchanged(self):
        model = RedactModel.__new__(RedactModel)
        model.tokenizer = SimpleNamespace(
            encode=lambda text, add_special_tokens=False: SimpleNamespace(
                ids=[], offsets=[]
            )
        )
        model.max_input_tokens = 16
        rule_spans = [{"start": 0, "end": 3, "label": "secret", "text": "abc"}]
        with patch("pii_redact_torch.deterministic_spans", return_value=rule_spans):
            self.assertEqual(model.predict("   "), rule_spans)


class WindowRangeTests(unittest.TestCase):
    def test_no_tokens_produce_no_windows(self):
        self.assertEqual(_window_ranges(0), [])

    def test_negative_token_count_produces_no_windows(self):
        self.assertEqual(_window_ranges(-5), [])

    def test_token_count_below_the_window_stays_in_one_range(self):
        self.assertEqual(_window_ranges(10), [(0, 10)])

    def test_token_count_equal_to_the_window_stays_in_one_range(self):
        self.assertEqual(
            _window_ranges(CONTENT_WINDOW_LENGTH), [(0, CONTENT_WINDOW_LENGTH)]
        )

    def test_long_input_slides_by_the_step_and_reaches_the_end(self):
        token_count = CONTENT_WINDOW_LENGTH + WINDOW_OVERLAP
        ranges = _window_ranges(token_count)

        self.assertEqual(len(ranges), 2)
        self.assertEqual(ranges[0], (0, CONTENT_WINDOW_LENGTH))
        self.assertEqual(ranges[1][1], token_count)
        self.assertEqual(ranges[1][0], CONTENT_WINDOW_LENGTH - WINDOW_OVERLAP)
        self.assertEqual(ranges[0][1] - ranges[1][0], WINDOW_OVERLAP)
        self.assertEqual(ranges[1][0] - ranges[0][0], WINDOW_STEP)

    def test_step_is_the_content_window_minus_the_overlap(self):
        self.assertEqual(WINDOW_STEP, CONTENT_WINDOW_LENGTH - WINDOW_OVERLAP)

    def test_overlap_at_the_window_length_is_rejected(self):
        with patch("pii_redact_torch.WINDOW_OVERLAP", CONTENT_WINDOW_LENGTH):
            with self.assertRaisesRegex(RuntimeError, "WINDOW_OVERLAP"):
                _window_ranges(10)


class TrimSpanWhitespaceTests(unittest.TestCase):
    def test_surrounding_whitespace_is_removed(self):
        self.assertEqual(_trim_span_whitespace("  value  ", 0, 9), (2, 7))

    def test_all_whitespace_collapses_to_an_empty_range(self):
        self.assertEqual(_trim_span_whitespace("   ", 0, 3), (3, 3))

    def test_span_without_padding_is_unchanged(self):
        self.assertEqual(_trim_span_whitespace("value", 0, 5), (0, 5))


class ChunkBoundaryTests(unittest.TestCase):
    """Properties of the chunk arithmetic, asserted against the input length."""

    def chunk_parameters(self):
        return [
            (1, 0),
            (2, 1),
            (4, 0),
            (4, 1),
            (4, 3),
            (256, 64),
            (4096, 128),
        ]

    def test_every_token_lands_in_at_least_one_chunk(self):
        for chunk_size, overlap in self.chunk_parameters():
            step = chunk_size - overlap
            for token_count in list(range(1, 14)) + [255, 256, 257, 4096, 32768]:
                with self.subTest(chunk_size=chunk_size, token_count=token_count):
                    ranges = chunk_token_ranges(token_count, chunk_size, overlap)
                    self.assertEqual(ranges[0][0], 0)
                    self.assertEqual(ranges[-1][1], token_count)
                    for previous, following in zip(ranges, ranges[1:]):
                        self.assertLessEqual(following[0], previous[1])
                        self.assertEqual(
                            following[0], previous[1] - overlap, "overlap drifted"
                        )
                        self.assertEqual(following[0] - previous[0], step)

    def test_no_chunk_is_longer_than_the_chunk_size(self):
        for chunk_size, overlap in self.chunk_parameters():
            for token_count in list(range(1, 14)) + [255, 256, 257, 4096]:
                with self.subTest(chunk_size=chunk_size, token_count=token_count):
                    for start, end in chunk_token_ranges(
                        token_count, chunk_size, overlap
                    ):
                        self.assertGreater(start, 0 - 1)
                        self.assertLessEqual(end - start, chunk_size)

    def test_chunk_count_matches_the_stride_formula(self):
        for chunk_size, overlap in self.chunk_parameters():
            step = chunk_size - overlap
            for token_count in list(range(1, 14)) + [255, 256, 257, 4096, 32768]:
                with self.subTest(chunk_size=chunk_size, token_count=token_count):
                    expected = 1 + max(0, -(-(token_count - chunk_size) // step))
                    self.assertEqual(
                        len(chunk_token_ranges(token_count, chunk_size, overlap)),
                        expected,
                    )

    def test_exactly_one_chunk_of_input_is_a_single_range(self):
        for chunk_size, overlap in self.chunk_parameters():
            with self.subTest(chunk_size=chunk_size):
                self.assertEqual(
                    chunk_token_ranges(chunk_size, chunk_size, overlap),
                    [(0, chunk_size)],
                )

    def test_one_token_short_of_a_chunk_stays_in_one_range(self):
        for chunk_size, overlap in self.chunk_parameters():
            if chunk_size < 2:
                continue
            with self.subTest(chunk_size=chunk_size):
                self.assertEqual(
                    chunk_token_ranges(chunk_size - 1, chunk_size, overlap),
                    [(0, chunk_size - 1)],
                )

    def test_a_single_token_chunk_size_leaves_no_partial_range(self):
        self.assertEqual(chunk_token_ranges(0, chunk_size=1, overlap=0), [])

    def test_one_token_over_a_chunk_starts_a_second_range(self):
        for chunk_size, overlap in self.chunk_parameters():
            step = chunk_size - overlap
            with self.subTest(chunk_size=chunk_size):
                ranges = chunk_token_ranges(chunk_size + 1, chunk_size, overlap)
                self.assertEqual(ranges[0], (0, chunk_size))
                self.assertEqual(ranges[-1][1], chunk_size + 1)
                self.assertGreaterEqual(len(ranges), 2)
                self.assertEqual(ranges[1][0], step)

    def test_final_chunk_is_partial_and_reaches_the_last_token(self):
        ranges = chunk_token_ranges(11, chunk_size=4, overlap=1)

        self.assertEqual(ranges, [(0, 4), (3, 7), (6, 10), (9, 11)])
        self.assertEqual(ranges[-1][1] - ranges[-1][0], 2)

    def test_final_chunk_can_end_exactly_on_the_last_token(self):
        ranges = chunk_token_ranges(10, chunk_size=4, overlap=1)

        self.assertEqual(ranges, [(0, 4), (3, 7), (6, 10)])
        self.assertEqual(ranges[-1][1] - ranges[-1][0], 4)

    def test_no_overlap_walks_chunk_by_chunk(self):
        self.assertEqual(
            chunk_token_ranges(10, chunk_size=4, overlap=0),
            [(0, 4), (4, 8), (8, 10)],
        )

    def test_maximum_overlap_walks_one_token_at_a_time(self):
        ranges = chunk_token_ranges(6, chunk_size=4, overlap=3)
        self.assertEqual(ranges, [(0, 4), (1, 5), (2, 6)])

    def test_single_token_chunks_are_allowed(self):
        self.assertEqual(
            chunk_token_ranges(3, chunk_size=1, overlap=0), [(0, 1), (1, 2), (2, 3)]
        )


class WindowBoundaryTests(unittest.TestCase):
    """Properties of the model window arithmetic, asserted against the input length."""

    def window_counts(self):
        return list(range(1, 270)) + [300, 511, 512, 4096]

    def test_every_token_lands_in_at_least_one_window(self):
        for token_count in self.window_counts():
            with self.subTest(token_count=token_count):
                ranges = _window_ranges(token_count)
                self.assertEqual(ranges[0][0], 0)
                self.assertEqual(ranges[-1][1], token_count)
                for previous, following in zip(ranges, ranges[1:]):
                    self.assertLessEqual(following[0], previous[1], "a token is lost")
                    self.assertGreater(following[0], previous[0], "windows do not move")

    def test_no_window_is_longer_than_the_content_window(self):
        for token_count in self.window_counts():
            with self.subTest(token_count=token_count):
                for start, end in _window_ranges(token_count):
                    self.assertLessEqual(end - start, CONTENT_WINDOW_LENGTH)

    def test_window_count_matches_the_step_formula(self):
        for token_count in self.window_counts():
            with self.subTest(token_count=token_count):
                expected = 1 + max(
                    0, -(-(token_count - CONTENT_WINDOW_LENGTH) // WINDOW_STEP)
                )
                self.assertEqual(len(_window_ranges(token_count)), expected)

    def test_input_shorter_than_a_window_is_one_window(self):
        for token_count in (1, 2, CONTENT_WINDOW_LENGTH - 1, CONTENT_WINDOW_LENGTH):
            with self.subTest(token_count=token_count):
                self.assertEqual(_window_ranges(token_count), [(0, token_count)])

    def test_one_token_past_a_window_starts_a_second_window(self):
        ranges = _window_ranges(CONTENT_WINDOW_LENGTH + 1)

        self.assertEqual(len(ranges), 2)
        self.assertEqual(ranges[0], (0, CONTENT_WINDOW_LENGTH))
        self.assertEqual(ranges[1], (WINDOW_STEP, CONTENT_WINDOW_LENGTH + 1))

    def test_final_window_is_partial(self):
        ranges = _window_ranges(CONTENT_WINDOW_LENGTH + WINDOW_OVERLAP)

        self.assertEqual(
            ranges[-1], (WINDOW_STEP, CONTENT_WINDOW_LENGTH + WINDOW_OVERLAP)
        )
        self.assertLess(ranges[-1][1] - ranges[-1][0], CONTENT_WINDOW_LENGTH)

    def test_adjacent_windows_overlap_by_the_overlap(self):
        ranges = _window_ranges(4096)
        for previous, following in zip(ranges, ranges[1:]):
            with self.subTest(previous=previous):
                self.assertEqual(previous[1] - following[0], WINDOW_OVERLAP)
                self.assertEqual(previous[1] - previous[0], CONTENT_WINDOW_LENGTH)
                self.assertEqual(following[0] - previous[0], WINDOW_STEP)

    def test_overlap_leaves_room_inside_the_window(self):
        self.assertLess(WINDOW_OVERLAP, CONTENT_WINDOW_LENGTH)
        self.assertGreater(WINDOW_OVERLAP, 0)


class MultiByteOffsetTests(unittest.TestCase):
    """Offsets must be character indices, and must be rebased per chunk."""

    TEXT = "hello 密碼 本地 測試 world"

    def test_chunk_offsets_are_character_indices(self):
        model = build_model(CJK_FIXTURE_CACHE, REDACT_MAX_TOKENS="8")
        encoding = model.tokenizer.encode(self.TEXT, add_special_tokens=False)

        self.assertEqual(
            [self.TEXT[start:end] for start, end in encoding.offsets],
            ["hello", "密碼", "本地", "測試", "world"],
        )
        self.assertEqual(encoding.offsets[-1][1], len(self.TEXT))

    def test_multibyte_span_survives_the_offset_round_trip(self):
        model = build_model(CJK_FIXTURE_CACHE, REDACT_MAX_TOKENS="8")
        path = [
            label_id(model.label_space, "B-GIVEN_NAME"),
            label_id(model.label_space, "E-GIVEN_NAME"),
        ]
        probabilities = probability_table(path, 0.9, model.label_space.num_classes)
        spans = model._path_to_spans(path, probabilities, [(0, 2), (3, 5)], "密碼 本地")

        self.assertEqual([span["text"] for span in spans], ["密碼 本地"])

    def test_every_chunk_reports_character_offsets_in_the_original_text(self):
        model = build_model(CJK_FIXTURE_CACHE, REDACT_MAX_TOKENS="2")
        seen = []

        def record(token_ids, offsets, text):
            seen.append((text, list(offsets)))
            return [{"start": 0, "end": len(text), "label": "private_person"}]

        with patch.object(model, "_predict_model_spans", side_effect=record):
            model.predict(self.TEXT)

        self.assertEqual(
            [text for text, _ in seen],
            ["hello 密碼", "密碼 本地", "本地 測試", "測試 world"],
        )
        for text, offsets in seen:
            with self.subTest(text=text):
                self.assertEqual(offsets[0][0], 0, "chunk origin is not rebased")
                self.assertEqual(
                    text[offsets[0][0] : offsets[0][1]], text.split(" ")[0]
                )
                for start, end in offsets:
                    self.assertGreaterEqual(start, 0)
                    self.assertLessEqual(end, len(text))

    def test_overlapping_chunk_spans_do_not_duplicate_a_multibyte_word(self):
        model = build_model(CJK_FIXTURE_CACHE, REDACT_MAX_TOKENS="2")
        with patch.object(
            model,
            "_predict_model_spans",
            side_effect=lambda token_ids, offsets, text: [
                {"start": 0, "end": len(text), "label": "private_person"}
            ],
        ):
            spans = model.predict(self.TEXT)

        words = set(CJK_FIXTURE_VOCAB) - {"[UNK]", "<s>", "</s>", "<pad>"}
        # The windows overlap, so the merge sees the same word more than once.
        # What matters is that the output stays ordered and disjoint, that no
        # span cuts a character, and that no fragment splits a word. The exact
        # fragments the trim leaves behind are not pinned, because the
        # degenerate ones are dropped; the coverage below is what must not
        # shrink, and it is asserted against the text rather than a copied list.
        previous_end = 0
        covered: set[int] = set()
        for span in spans:
            with self.subTest(span=span):
                start = cast(int, span["start"])
                end = cast(int, span["end"])
                self.assertEqual(span["text"], self.TEXT[start:end])
                self.assertGreaterEqual(start, previous_end, "spans overlap")
                previous_end = end
                covered.update(range(start, end))
                for word in str(span["text"]).split(" "):
                    if word:
                        self.assertIn(word, words)

        # Every character the detector reported is still covered, so the trim
        # never loses ground that the old drop-based merge would have lost.
        # Whitespace may go uncovered: a fragment that is only whitespace is
        # dropped on purpose.
        for index, character in enumerate(self.TEXT):
            if not character.isspace():
                with self.subTest(index=index):
                    self.assertIn(index, covered)

        self.assertEqual(
            [
                (spans[0]["start"], spans[0]["end"]),
                (spans[-1]["start"], spans[-1]["end"]),
            ],
            [(0, 8), (12, 20)],
        )

    def test_masked_text_never_splits_a_multibyte_character(self):
        model = build_model(CJK_FIXTURE_CACHE, REDACT_MAX_TOKENS="2")
        with patch.object(
            model,
            "_predict_model_spans",
            side_effect=lambda token_ids, offsets, text: [
                {"start": 0, "end": len(text), "label": "private_person"}
            ],
        ):
            spans = model.predict(self.TEXT)

        words = set(CJK_FIXTURE_VOCAB) - {"[UNK]", "<s>", "</s>", "<pad>"}
        for span in spans:
            with self.subTest(span=span):
                self.assertEqual(
                    span["text"],
                    self.TEXT[cast(int, span["start"]) : cast(int, span["end"])],
                )
                for word in str(span["text"]).split(" "):
                    if word:
                        self.assertIn(word, words)


class SpanSeamTests(unittest.TestCase):
    """Merging at and around the chunk overlap seam."""

    def test_adjacent_spans_are_both_kept(self):
        text = "aaaabbbb"
        spans = merge_spans(
            text,
            [
                {"start": 0, "end": 4, "label": "private_person"},
                {"start": 4, "end": 8, "label": "private_person"},
            ],
        )

        self.assertEqual(
            [(span["start"], span["end"]) for span in spans], [(0, 4), (4, 8)]
        )

    def test_identical_spans_collapse_to_one(self):
        text = "aaaa"
        spans = merge_spans(
            text,
            [
                {"start": 0, "end": 4, "label": "private_person"},
                {"start": 0, "end": 4, "label": "private_person"},
            ],
        )

        self.assertEqual(len(spans), 1)

    def test_higher_priority_span_wins_over_the_span_containing_it(self):
        text = "secret@example.com"
        spans = merge_spans(
            text,
            [
                {"start": 0, "end": 18, "label": "private_person"},
                {"start": 0, "end": 6, "label": "secret"},
            ],
        )

        # The container is trimmed around the winner, not dropped: the value
        # the secret sits in stays reported for the characters the secret does
        # not cover.
        self.assertEqual(
            [(span["start"], span["end"], span["label"]) for span in spans],
            [(0, 6, "secret"), (6, 18, "private_person")],
        )

    def test_container_wins_when_it_holds_the_higher_priority(self):
        text = "user@example.com"
        spans = merge_spans(
            text,
            [
                {"start": 0, "end": 16, "label": "private_email"},
                {"start": 0, "end": 4, "label": "private_person"},
            ],
        )

        self.assertEqual(
            [(span["start"], span["end"], span["label"]) for span in spans],
            [(0, 16, "private_email")],
        )

    def test_partially_overlapping_spans_are_trimmed_not_dropped(self):
        text = "aaaaaaaaaa"
        spans = merge_spans(
            text,
            [
                {"start": 0, "end": 6, "label": "private_person"},
                {"start": 4, "end": 10, "label": "private_person"},
            ],
        )

        # Equal priority: the earlier span wins the overlap and the later one
        # keeps the characters it alone covers. Nothing is dropped.
        self.assertEqual(
            [(span["start"], span["end"]) for span in spans], [(0, 6), (6, 10)]
        )

    def test_spans_that_touch_the_seam_are_ordered_and_disjoint(self):
        model = build_model(REDACT_MIN_SCORE="0.0")
        spans = model.predict("hello world avery@example.com acme inc hello world")
        previous_end = 0
        for span in spans:
            with self.subTest(span=span):
                self.assertGreaterEqual(cast(int, span["start"]), previous_end)
                previous_end = cast(int, span["end"])


class InputCapBoundaryTests(unittest.TestCase):
    """The request cap must accept the cap itself and reject only what is over."""

    def tokens(self, count: int) -> str:
        return " ".join(["hello"] * count)

    def test_a_request_exactly_at_the_cap_is_accepted(self):
        model = build_model(REDACT_MAX_INPUT_TOKENS="16")
        text = self.tokens(16)

        self.assertEqual(
            len(model.tokenizer.encode(text, add_special_tokens=False).ids), 16
        )
        self.assertIsInstance(model.predict(text), list)

    def test_a_request_one_token_under_the_cap_is_accepted(self):
        model = build_model(REDACT_MAX_INPUT_TOKENS="16")

        self.assertIsInstance(model.predict(self.tokens(15)), list)

    def test_a_request_one_token_over_the_cap_is_rejected(self):
        model = build_model(REDACT_MAX_INPUT_TOKENS="16")

        with self.assertRaisesRegex(InputTooLargeError, "exceeds max 16"):
            model.predict(self.tokens(17))

    def test_the_cap_counts_model_tokens_not_characters(self):
        model = build_model(REDACT_MAX_INPUT_TOKENS="16")
        text = self.tokens(17)

        self.assertEqual(len(text), 101)
        with self.assertRaises(InputTooLargeError):
            model.predict(text)

    def test_a_rejected_request_leaves_the_model_usable(self):
        model = build_model(REDACT_MAX_INPUT_TOKENS="16")
        with self.assertRaises(InputTooLargeError):
            model.predict(self.tokens(17))

        self.assertIsInstance(model.predict(self.tokens(3)), list)


class SerializingModel:
    """Records the highest number of concurrent predict calls."""

    device = "cpu"

    def __init__(self, delay: float = 0.2):
        self.delay = delay
        self.guard = threading.Lock()
        self.active = 0
        self.peak_active = 0
        self.calls = 0

    def predict(self, text):
        with self.guard:
            self.active += 1
            self.calls += 1
            self.peak_active = max(self.peak_active, self.active)
        time.sleep(self.delay)
        with self.guard:
            self.active -= 1
        return []


class FlakyModel:
    """Fails the first call, then behaves."""

    device = "cpu"

    def __init__(self, error):
        self.error = error
        self.calls = 0

    def predict(self, text):
        self.calls += 1
        if self.calls == 1:
            raise self.error
        return []


class ConcurrencyTests(unittest.TestCase):
    def setUp(self):
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.addCleanup(self.stop_server)
        self.log_patch = patch.object(Handler, "log_message", Mock())
        self.log_patch.start()
        self.addCleanup(self.log_patch.stop)

    def stop_server(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=10)

    def use_model(self, model):
        model_patch = patch.object(Handler, "model", model, create=True)
        model_patch.start()
        self.addCleanup(model_patch.stop)

    def post(self, text: str):
        body = json.dumps({"text": text}).encode()
        return raw_request(self.port, post_request(body))

    def test_the_inference_lock_is_shared_by_every_handler_instance(self):
        self.use_model(SerializingModel())
        first = Handler.__new__(Handler)
        second = Handler.__new__(Handler)

        self.assertIs(first.inference_lock, second.inference_lock)

    def test_overlapping_requests_never_enter_predict_together(self):
        model = SerializingModel(delay=0.25)
        self.use_model(model)
        results = []

        def send():
            results.append(self.post("hello")[0])

        threads = [threading.Thread(target=send) for _ in range(3)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=20)

        self.assertEqual(results, [200, 200, 200])
        self.assertEqual(model.calls, 3)
        self.assertEqual(model.peak_active, 1)

    def test_a_request_rejected_as_oversized_does_not_hold_the_lock(self):
        model = FlakyModel(InputTooLargeError("input has 17 tokens, exceeds max 16"))
        self.use_model(model)

        self.assertEqual(self.post("hello")[0], 413)
        self.assertEqual(self.post("hello")[0], 200)

    def test_a_failed_inference_does_not_hold_the_lock(self):
        self.use_model(FlakyModel(RuntimeError("model device mismatch")))

        self.assertEqual(self.post("hello")[0], 500)
        self.assertEqual(self.post("hello")[0], 200)

    def test_a_bad_request_does_not_hold_the_lock(self):
        self.use_model(SerializingModel(delay=0.0))

        status, _, _ = raw_request(self.port, post_request(b"{not json"))
        self.assertEqual(status, 400)
        self.assertEqual(self.post("hello")[0], 200)


class ModelConfigurationTests(unittest.TestCase):
    def test_score_above_one_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            cache_dir = Path(temporary_directory)
            write_fixture_config_only(cache_dir)
            with self.assertRaisesRegex(ValueError, "between 0 and 1"):
                build_model(cache_dir, REDACT_MIN_SCORE="1.5")

    def test_negative_score_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            cache_dir = Path(temporary_directory)
            write_fixture_config_only(cache_dir)
            with self.assertRaisesRegex(ValueError, "between 0 and 1"):
                build_model(cache_dir, REDACT_MIN_SCORE="-0.1")

    def test_zero_batch_size_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            cache_dir = Path(temporary_directory)
            write_fixture_config_only(cache_dir)
            with self.assertRaisesRegex(ValueError, "must be positive"):
                build_model(cache_dir, REDACT_BATCH_SIZE="0")

    def test_zero_max_tokens_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            cache_dir = Path(temporary_directory)
            write_fixture_config_only(cache_dir)
            with self.assertRaisesRegex(ValueError, "must be positive"):
                build_model(cache_dir, REDACT_MAX_TOKENS="0")

    def test_input_cap_below_the_chunk_size_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            cache_dir = Path(temporary_directory)
            write_fixture_config_only(cache_dir)
            with self.assertRaisesRegex(ValueError, "at least REDACT_MAX_TOKENS"):
                build_model(cache_dir, REDACT_MAX_INPUT_TOKENS="2")

    def test_overlap_at_the_chunk_size_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            cache_dir = Path(temporary_directory)
            write_fixture_config_only(cache_dir)
            with self.assertRaisesRegex(ValueError, "non-negative"):
                build_model(cache_dir, REDACT_CHUNK_OVERLAP_TOKENS="4")

    def test_negative_overlap_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            cache_dir = Path(temporary_directory)
            write_fixture_config_only(cache_dir)
            with self.assertRaisesRegex(ValueError, "non-negative"):
                build_model(cache_dir, REDACT_CHUNK_OVERLAP_TOKENS="-1")

    def test_defaults_are_used_when_the_environment_is_absent(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            cache_dir = Path(temporary_directory)
            write_fixture_cache(cache_dir)
            with patch.dict(os.environ, dict(FIXTURE_ENVIRONMENT)):
                for name in (
                    "REDACT_MIN_SCORE",
                    "REDACT_BATCH_SIZE",
                    "REDACT_MAX_TOKENS",
                    "REDACT_CHUNK_OVERLAP_TOKENS",
                    "REDACT_MAX_INPUT_TOKENS",
                ):
                    os.environ.pop(name, None)
                model = RedactModel(cache_dir, "cpu")

        self.assertEqual(model.min_score, 0.6)
        self.assertEqual(model.batch_size, 8)
        self.assertEqual(model.max_tokens, 4096)
        self.assertEqual(model.chunk_overlap, 128)
        self.assertEqual(model.max_input_tokens, 32768)

    def test_label_space_comes_from_the_config_file(self):
        model = build_model()

        self.assertEqual(model.label_space.num_classes, len(FIXTURE_ID2LABEL))
        self.assertEqual(
            model.label_space.id2label[label_id(model.label_space, "S-EMAIL")],
            "S-EMAIL",
        )
        self.assertEqual(model.label_space.bg, 0)

    def test_transition_tables_are_built_for_the_label_space(self):
        model = build_model()

        self.assertEqual(model.start.shape, (len(FIXTURE_ID2LABEL),))
        self.assertEqual(model.transition.shape, (len(FIXTURE_ID2LABEL),) * 2)


class CheckpointLoadingTests(unittest.TestCase):
    def test_public_special_token_ids_are_read_from_the_tokenizer(self):
        model = build_model()

        self.assertEqual(model.bos_id, FIXTURE_VOCAB["<s>"])
        self.assertEqual(model.eos_id, FIXTURE_VOCAB["</s>"])
        self.assertEqual(model.pad_id, FIXTURE_VOCAB["<pad>"])

    def test_unknown_special_token_falls_back_to_the_default_id(self):
        model = build_model()

        self.assertEqual(model._token_id("<not-in-vocab>", 99), 99)

    def test_known_token_reports_its_real_id(self):
        model = build_model()

        self.assertEqual(model._token_id("hello", 99), FIXTURE_VOCAB["hello"])

    def test_bare_state_dict_checkpoint_loads(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            cache_dir = Path(temporary_directory) / "redact"
            write_fixture_cache(cache_dir, checkpoint="bare")
            model = build_model(cache_dir)

        self.assertEqual(
            {parameter.device.type for parameter in model.model.parameters()}, {"cpu"}
        )

    def test_checkpoint_without_a_state_dict_raises_a_type_error(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            cache_dir = Path(temporary_directory) / "redact"
            write_fixture_cache(cache_dir, checkpoint="junk")
            with self.assertRaisesRegex(TypeError, "does not contain"):
                build_model(cache_dir)

    def test_a_parameter_that_did_not_move_is_refused_at_startup(self):
        """The one precondition that reaches the mismatch arm.

        Module.to() moves every parameter, so the constructor cannot reach this
        arm on its own: over 12 model shapes across hidden sizes, layer counts
        and devices it never fired. Send the move to a device the model did not
        ask for, which is the state the guard exists to catch, and it fires.
        """
        original_to = torch.nn.Module.to

        def move_elsewhere(module, *args, **kwargs):
            return original_to(module, "meta")

        with patch.object(torch.nn.Module, "to", move_elsewhere):
            with self.assertRaisesRegex(
                RuntimeError,
                r"model device mismatch: expected cpu, found \['meta'\]",
            ):
                build_model()

    def test_model_is_put_in_evaluation_mode(self):
        self.assertFalse(build_model().model.training)

    def test_device_synchronize_is_a_no_op_on_cpu(self):
        model = build_model()
        with patch("pii_redact_torch.torch.cuda.synchronize") as cuda_sync:
            with patch("pii_redact_torch.torch.mps.synchronize") as mps_sync:
                model._synchronize()

        cuda_sync.assert_not_called()
        mps_sync.assert_not_called()

    def test_device_synchronize_waits_on_cuda(self):
        model = build_model()
        cuda_device = SimpleNamespace(type="cuda")
        with patch.object(model, "device", cuda_device):
            with patch("pii_redact_torch.torch.cuda.synchronize") as cuda_sync:
                model._synchronize()

        cuda_sync.assert_called_once_with(cuda_device)

    def test_device_synchronize_waits_on_mps(self):
        model = build_model()
        with patch.object(model, "device", SimpleNamespace(type="mps")):
            with patch("pii_redact_torch.torch.mps.synchronize") as mps_sync:
                model._synchronize()

        mps_sync.assert_called_once_with()


class BatchBuildingTests(unittest.TestCase):
    def test_batch_is_padded_to_the_model_sequence_length(self):
        model = build_model()
        input_ids, attention_mask = model._build_batch([4, 5, 6], [(0, 2), (1, 3)])

        self.assertEqual(tuple(input_ids.shape), (2, MAX_SEQUENCE_LENGTH))
        self.assertEqual(tuple(attention_mask.shape), (2, MAX_SEQUENCE_LENGTH))

    def test_sequence_is_wrapped_in_special_tokens(self):
        model = build_model()
        input_ids, _ = model._build_batch([4, 5], [(0, 2)])
        row = input_ids[0].tolist()

        self.assertEqual(
            row[:4],
            [
                model.bos_id,
                FIXTURE_VOCAB["hello"],
                FIXTURE_VOCAB["world"],
                model.eos_id,
            ],
        )
        self.assertEqual(set(row[4:]), {model.pad_id})

    def test_attention_mask_is_one_only_over_the_real_sequence(self):
        model = build_model()
        _, attention_mask = model._build_batch([4, 5], [(0, 2)])
        row = attention_mask[0].tolist()

        self.assertEqual(row[:4], [1, 1, 1, 1])
        self.assertEqual(sum(row), 4)

    def test_window_slice_selects_the_requested_tokens(self):
        model = build_model()
        input_ids, _ = model._build_batch([4, 5, 6, 7], [(1, 3)])
        row = input_ids[0].tolist()

        self.assertEqual(
            row[:4],
            [model.bos_id, FIXTURE_VOCAB["world"], FIXTURE_VOCAB["acme"], model.eos_id],
        )

    def test_batch_tensors_live_on_the_selected_device(self):
        model = build_model()
        input_ids, attention_mask = model._build_batch([4], [(0, 1)])

        self.assertEqual(input_ids.device, model.device)
        self.assertEqual(attention_mask.device, model.device)


class InferenceTests(unittest.TestCase):
    def test_empty_text_produces_no_spans(self):
        self.assertEqual(build_model().predict(""), [])

    def test_token_probabilities_are_a_distribution_per_token(self):
        model = build_model()
        probabilities = model._predict_token_probabilities([4, 5, 6])

        self.assertEqual(probabilities.shape, (3, len(FIXTURE_ID2LABEL)))
        for row in probabilities:
            self.assertAlmostEqual(float(row.sum()), 1.0, places=4)

    def test_token_probabilities_are_all_non_negative(self):
        model = build_model()
        probabilities = model._predict_token_probabilities([4, 5])

        self.assertTrue(bool(np.all(probabilities >= 0.0)))

    def test_probabilities_are_renormalized_per_token(self):
        model = build_model(REDACT_MAX_TOKENS="4096", REDACT_MAX_INPUT_TOKENS="32768")
        probabilities = model._predict_token_probabilities([4] * 255)

        for row in probabilities:
            self.assertAlmostEqual(float(row.sum()), 1.0, places=4)

    def test_a_token_on_a_window_seam_does_not_score_higher(self):
        # Token 200 sits in the first window alone for a CONTENT_WINDOW_LENGTH
        # token input, and in both the first and second window for an input one
        # token longer. The two inputs give window 0 the identical tensor, so
        # only the aggregation differs.
        model = build_model(REDACT_MAX_TOKENS="4096", REDACT_MAX_INPUT_TOKENS="32768")
        one_window = model._predict_token_probabilities([4] * CONTENT_WINDOW_LENGTH)
        two_windows = model._predict_token_probabilities(
            [4] * (CONTENT_WINDOW_LENGTH + 1)
        )

        self.assertLessEqual(
            float(two_windows[200].max()), float(one_window[200].max())
        )

    def test_batching_covers_every_token(self):
        model = build_model(REDACT_BATCH_SIZE="1")
        probabilities = model._predict_token_probabilities([4, 5, 6, 7])

        self.assertTrue(bool(np.all(probabilities.sum(axis=1) > 0.0)))

    def test_spans_stay_inside_the_text_and_carry_hook_labels(self):
        model = build_model(REDACT_MIN_SCORE="0.0")
        text = "hello world acme inc"
        spans = model.predict(text)

        for span in spans:
            with self.subTest(span=span):
                start = cast(int, span["start"])
                end = cast(int, span["end"])
                self.assertGreaterEqual(start, 0)
                self.assertLessEqual(end, len(text))
                self.assertLess(start, end)
                self.assertEqual(span["text"], text[start:end])
                self.assertIn(str(span["label"]), HOOK_LABELS)

    def test_spans_never_overlap_each_other(self):
        model = build_model(REDACT_MIN_SCORE="0.0")
        text = "hello avery@example.com world"
        spans = model.predict(text)

        occupied: set[int] = set()
        for span in spans:
            for index in range(cast(int, span["start"]), cast(int, span["end"])):
                self.assertNotIn(index, occupied)
                occupied.add(index)

    def test_a_rule_span_inside_a_chunk_overlap_is_reported_once(self):
        # "avery@example.com" is the token where the first two chunks overlap, so
        # only the merge keeps it from being reported twice.
        model = build_model(REDACT_MIN_SCORE="0.0")
        text = "hello world avery@example.com acme inc hello world"
        spans = model.predict(text)
        start = text.index("avery@example.com")
        end = start + len("avery@example.com")
        matching = [
            span for span in spans if (span["start"], span["end"]) == (start, end)
        ]

        self.assertEqual(len(matching), 1)
        self.assertEqual(matching[0]["label"], "private_email")

    def test_long_input_is_split_into_overlapping_chunks(self):
        model = build_model(REDACT_MIN_SCORE="0.0")
        text = "hello world acme inc hello world acme inc"
        encoding = model.tokenizer.encode(text, add_special_tokens=False)

        self.assertEqual(
            chunk_token_ranges(
                len(encoding.ids), model.max_tokens, model.chunk_overlap
            ),
            [(0, 4), (3, 7), (6, 8)],
        )
        for span in model.predict(text):
            self.assertEqual(
                span["text"], text[cast(int, span["start"]) : cast(int, span["end"])]
            )

    def test_each_chunk_costs_one_forward_pass_at_batch_size_one(self):
        text = "hello world acme inc hello world acme inc hello world"
        model = build_model(REDACT_BATCH_SIZE="1")
        token_count = len(model.tokenizer.encode(text, add_special_tokens=False).ids)
        counter = CountingModule(model.model)
        cast(Any, model).model = counter
        model.predict(text)

        self.assertEqual(token_count, 10)
        self.assertEqual(
            chunk_token_ranges(token_count, model.max_tokens, model.chunk_overlap),
            [(0, 4), (3, 7), (6, 10)],
        )
        self.assertEqual(counter.calls, 3)

    def test_deterministic_rules_still_run_when_no_model_span_scores(self):
        model = build_model(REDACT_MIN_SCORE="1.0")
        text = "hello avery@example.com world"
        spans = model.predict(text)

        self.assertEqual([span["label"] for span in spans], ["private_email"])

    def test_input_over_the_cap_is_rejected(self):
        model = build_model(REDACT_MAX_INPUT_TOKENS="4")
        with self.assertRaisesRegex(InputTooLargeError, "exceeds max 4"):
            model.predict("hello world acme inc hello")

    def test_input_too_large_is_a_value_error(self):
        self.assertTrue(issubclass(InputTooLargeError, ValueError))

    def test_a_single_chunk_covers_a_short_input(self):
        model = build_model()
        spans = model._predict_model_spans([4, 5], [(0, 5), (6, 11)], "hello world")

        self.assertLessEqual(len(spans), 2)
        for span in spans:
            self.assertIn(str(span["label"]), HOOK_LABELS)
            self.assertGreaterEqual(cast(int, span["start"]), 0)
            self.assertLessEqual(cast(int, span["end"]), len("hello world"))


class PathToSpanTests(unittest.TestCase):
    def setUp(self):
        self.model = build_model()

    def test_mapped_single_token_label_becomes_a_span(self):
        path = [label_id(self.model.label_space, "S-EMAIL")]
        probabilities = probability_table(path, 0.9, self.model.label_space.num_classes)
        spans = self.model._path_to_spans(
            path, probabilities, [(0, 17)], "avery@example.com"
        )

        self.assertEqual(
            spans,
            [
                {
                    "start": 0,
                    "end": 17,
                    "label": "private_email",
                    "text": "avery@example.com",
                }
            ],
        )

    def test_whitespace_around_a_span_is_trimmed(self):
        path = [label_id(self.model.label_space, "S-EMAIL")]
        probabilities = probability_table(path, 0.9, self.model.label_space.num_classes)
        spans = self.model._path_to_spans(
            path, probabilities, [(0, 21)], "  avery@example.com  "
        )

        self.assertEqual(len(spans), 1)
        self.assertEqual((spans[0]["start"], spans[0]["end"]), (2, 19))

    def test_all_whitespace_span_is_dropped(self):
        path = [label_id(self.model.label_space, "S-EMAIL")]
        probabilities = probability_table(path, 0.9, self.model.label_space.num_classes)
        spans = self.model._path_to_spans(path, probabilities, [(0, 3)], "   ")

        self.assertEqual(spans, [])

    def test_unmapped_label_is_skipped_however_confident(self):
        path = [label_id(self.model.label_space, "S-ORG")]
        probabilities = probability_table(path, 1.0, self.model.label_space.num_classes)
        spans = self.model._path_to_spans(path, probabilities, [(0, 3)], "acme")

        self.assertEqual(spans, [])

    def test_span_below_the_score_floor_is_skipped(self):
        strict_model = build_model(REDACT_MIN_SCORE="0.6")
        path = [label_id(strict_model.label_space, "S-EMAIL")]
        probabilities = probability_table(
            path, 0.2, strict_model.label_space.num_classes
        )
        spans = strict_model._path_to_spans(
            path, probabilities, [(0, 17)], "avery@example.com"
        )

        self.assertEqual(spans, [])

    def test_span_at_the_score_floor_is_kept(self):
        strict_model = build_model(REDACT_MIN_SCORE="0.6")
        path = [label_id(strict_model.label_space, "S-EMAIL")]
        probabilities = probability_table(
            path, 0.6, strict_model.label_space.num_classes
        )
        spans = strict_model._path_to_spans(
            path, probabilities, [(0, 17)], "avery@example.com"
        )

        self.assertEqual([span["label"] for span in spans], ["private_email"])

    def test_multi_token_span_uses_the_lowest_token_score(self):
        # The run's lowest token score is below the floor, so the whole span goes,
        # even though its first token scored well.
        strict_model = build_model(REDACT_MIN_SCORE="0.6")
        path = [
            label_id(strict_model.label_space, "B-GIVEN_NAME"),
            label_id(strict_model.label_space, "E-GIVEN_NAME"),
        ]
        probabilities = probability_table(
            path, 0.9, strict_model.label_space.num_classes
        )
        probabilities[1, 0] = 0.8
        probabilities[1, path[1]] = 0.2
        spans = strict_model._path_to_spans(
            path, probabilities, [(0, 4), (5, 9)], "Jody Smith"
        )

        self.assertEqual(spans, [])

    def test_offsets_place_the_span_in_the_original_text(self):
        path = [
            label_id(self.model.label_space, "B-GIVEN_NAME"),
            label_id(self.model.label_space, "E-GIVEN_NAME"),
        ]
        probabilities = probability_table(path, 0.9, self.model.label_space.num_classes)
        spans = self.model._path_to_spans(
            path, probabilities, [(7, 11), (12, 17)], "prefix Jody Smith"
        )

        self.assertEqual(
            spans,
            [
                {
                    "start": 7,
                    "end": 17,
                    "label": "private_person",
                    "text": "Jody Smith",
                }
            ],
        )


class CountingModule:
    """Delegates to the real module and counts how often it was called."""

    def __init__(self, module):
        self.module = module
        self.calls = 0

    def __call__(self, **kwargs):
        self.calls += 1
        return self.module(**kwargs)


class RecordingModel:
    """Stands in for RedactModel so the HTTP layer can be driven directly."""

    device = "cpu"

    def __init__(self, spans=None, error=None):
        self.spans = [] if spans is None else spans
        self.error = error
        self.seen_texts = []

    def predict(self, text):
        self.seen_texts.append(text)
        if self.error is not None:
            raise self.error
        return self.spans


def raw_request(port: int, request: bytes) -> tuple[int, dict[str, str], bytes]:
    with socket.create_connection(("127.0.0.1", port), timeout=10) as connection:
        connection.sendall(request)
        response = bytearray()
        while True:
            block = connection.recv(65536)
            if not block:
                break
            response.extend(block)

    head, _, body = bytes(response).partition(b"\r\n\r\n")
    lines = head.decode("iso-8859-1").split("\r\n")
    status = int(lines[0].split(" ")[1])
    headers = {}
    for line in lines[1:]:
        name, _, value = line.partition(": ")
        headers[name] = value
    return status, headers, body


def post_request(body: bytes, content_length: str | None = None) -> bytes:
    length = str(len(body)) if content_length is None else content_length
    return (
        b"POST / HTTP/1.0\r\n"
        b"Host: 127.0.0.1\r\n"
        + f"Content-Length: {length}\r\n".encode()
        + b"\r\n"
        + body
    )


def free_port() -> int:
    """A port nothing is listening on. Racy by nature; adequate for a test."""
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def wait_until(predicate, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return bool(predicate())


class HandlerTests(unittest.TestCase):
    def setUp(self):
        self.model = RecordingModel()
        self.model_patch = patch.object(Handler, "model", self.model, create=True)
        self.model_patch.start()
        self.addCleanup(self.model_patch.stop)

        self.log_message = Mock()
        self.log_patch = patch.object(Handler, "log_message", self.log_message)
        self.log_patch.start()
        self.addCleanup(self.log_patch.stop)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.addCleanup(self.stop_server)

    def stop_server(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=10)

    def post(self, body: bytes, content_length: str | None = None):
        return raw_request(self.port, post_request(body, content_length))

    def health(self):
        return raw_request(
            self.port, b"GET /health HTTP/1.0\r\nHost: 127.0.0.1\r\n\r\n"
        )

    def post_declaring(self, content_length: str):
        """Declare a Content-Length, send no body, and read the answer.

        A handler that reads the declared body before it answers blocks here, so
        any reply at all proves the status came back ahead of the read.
        """
        request = (
            b"POST / HTTP/1.0\r\n"
            b"Host: 127.0.0.1\r\n"
            + f"Content-Length: {content_length}\r\n".encode()
            + b"\r\n"
        )
        return raw_request(self.port, request)

    def test_valid_request_returns_the_spans_and_a_timing(self):
        self.model.spans = [{"start": 0, "end": 3, "label": "secret", "text": "abc"}]
        status, _, body = self.post(json.dumps({"text": "abc"}).encode())

        payload = json.loads(body)
        self.assertEqual(status, 200)
        self.assertEqual(payload["spans"], self.model.spans)
        self.assertIsInstance(payload["processing_ms"], int)
        self.assertGreaterEqual(payload["processing_ms"], 0)

    def test_valid_request_reaches_the_model_with_the_request_text(self):
        self.post(json.dumps({"text": "hello world"}).encode())

        self.assertEqual(self.model.seen_texts, ["hello world"])

    def test_response_declares_json_and_its_own_length(self):
        status, headers, body = self.post(json.dumps({"text": "abc"}).encode())

        self.assertEqual(status, 200)
        self.assertEqual(headers["Content-Type"], "application/json")
        self.assertEqual(int(headers["Content-Length"]), len(body))

    def test_empty_span_list_is_returned_as_an_empty_list(self):
        status, _, body = self.post(json.dumps({"text": "abc"}).encode())

        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body)["spans"], [])

    def test_non_numeric_content_length_is_rejected(self):
        status, _, body = self.post(b"{}", content_length="not-a-number")

        self.assertEqual(status, 400)
        self.assertEqual(json.loads(body), {"error": "invalid Content-Length"})

    def test_missing_content_length_is_rejected(self):
        request = b"POST / HTTP/1.0\r\nHost: 127.0.0.1\r\n\r\n"
        status, _, body = raw_request(self.port, request)

        self.assertEqual(status, 400)
        self.assertEqual(json.loads(body), {"error": "empty body"})

    def test_zero_content_length_is_rejected(self):
        status, _, body = self.post(b"", content_length="0")

        self.assertEqual(status, 400)
        self.assertEqual(json.loads(body), {"error": "empty body"})

    def test_negative_content_length_is_rejected(self):
        status, _, body = self.post(b"", content_length="-5")

        self.assertEqual(status, 400)
        self.assertEqual(json.loads(body), {"error": "empty body"})

    def test_malformed_json_is_rejected(self):
        status, _, body = self.post(b"{not json")

        self.assertEqual(status, 400)
        self.assertTrue(json.loads(body)["error"].startswith("bad request:"))

    def test_missing_text_field_is_rejected(self):
        status, _, body = self.post(json.dumps({"prompt": "x"}).encode())

        self.assertEqual(status, 400)
        self.assertIn("text", json.loads(body)["error"])

    def test_non_string_text_is_rejected(self):
        status, _, body = self.post(json.dumps({"text": 17}).encode())

        self.assertEqual(status, 400)
        self.assertIn("text must be a string", json.loads(body)["error"])

    def test_non_utf8_body_is_rejected(self):
        status, _, body = self.post(b'{"text": "\xff\xfe"}')

        self.assertEqual(status, 400)
        self.assertTrue(json.loads(body)["error"].startswith("bad request:"))

    def test_non_object_json_body_is_rejected(self):
        for body in (b'"hello"', b"[1,2,3]", b"null", b"42", b"true"):
            with self.subTest(body=body):
                status, _, response = self.post(body)

                self.assertEqual(status, 400)
                self.assertEqual(
                    json.loads(response),
                    {"error": "bad request: body must be a JSON object"},
                )

    def test_declared_body_above_the_cap_is_answered_before_the_read(self):
        with patch.dict(os.environ, {"PII_MAX_BODY_BYTES": "64"}):
            status, _, body = self.post_declaring("65")

        self.assertEqual(status, 413)
        self.assertEqual(
            json.loads(body),
            {"error": "request body of 65 bytes exceeds the 64 byte limit"},
        )

    def test_body_at_the_cap_is_read_and_judged_on_its_content(self):
        with patch.dict(os.environ, {"PII_MAX_BODY_BYTES": "64"}):
            status, _, body = self.post(b"x" * 64)

        self.assertEqual(status, 400)
        self.assertTrue(json.loads(body)["error"].startswith("bad request:"))

    def test_default_body_cap_is_two_mebibytes(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(max_body_bytes(), 2 * 1024 * 1024)

    def test_body_cap_setting_comes_from_the_environment(self):
        with patch.dict(os.environ, {"PII_MAX_BODY_BYTES": "4096"}):
            self.assertEqual(max_body_bytes(), 4096)

    def test_malformed_body_cap_setting_falls_back_to_the_default(self):
        with patch.dict(os.environ, {"PII_MAX_BODY_BYTES": "not-a-number"}):
            self.assertEqual(max_body_bytes(), 2 * 1024 * 1024)

    def test_a_stalled_connection_is_dropped_by_the_handler_timeout(self):
        with patch.object(Handler, "timeout", 0.3):
            with socket.create_connection(("127.0.0.1", self.port), timeout=10) as conn:
                conn.sendall(
                    b"POST / HTTP/1.0\r\nHost: 127.0.0.1\r\nContent-Length: 50\r\n\r\n"
                )
                started = time.monotonic()
                self.assertEqual(conn.recv(1024), b"")
                elapsed = time.monotonic() - started

        self.assertLess(elapsed, 5)

    def test_oversized_input_returns_413(self):
        self.model.error = InputTooLargeError("input has 99 tokens, exceeds max 4")
        status, _, body = self.post(json.dumps({"text": "abc"}).encode())

        self.assertEqual(status, 413)
        self.assertEqual(
            json.loads(body), {"error": "input has 99 tokens, exceeds max 4"}
        )

    def test_model_failure_returns_500(self):
        self.model.error = RuntimeError("model device mismatch")
        status, _, body = self.post(json.dumps({"text": "abc"}).encode())

        self.assertEqual(status, 500)
        self.assertEqual(json.loads(body), {"error": "model device mismatch"})

    def test_a_client_that_leaves_before_the_answer_is_logged_not_traced(self):
        """A swallowed disconnect handler fails silently, so drive a real one.

        The client hangs up between the request and the answer, which is the
        ordinary case the handler exists for.
        """
        reached_model = threading.Event()
        release_model = threading.Event()

        class SignallingModel:
            device = "cpu"

            def predict(self, text):
                reached_model.set()
                release_model.wait(timeout=10)
                return []

        unexpected = io.StringIO()
        with patch.object(Handler, "model", SignallingModel()):
            connection = socket.create_connection(("127.0.0.1", self.port), timeout=10)
            with redirect_stderr(unexpected):
                try:
                    connection.sendall(
                        post_request(json.dumps({"text": "abc"}).encode())
                    )
                    self.assertTrue(
                        reached_model.wait(timeout=5),
                        "the request never reached the model",
                    )

                    # Leave without reading the answer. SO_LINGER 0 sends a
                    # reset, so the server's write fails rather than racing the
                    # close.
                    connection.setsockopt(
                        socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0)
                    )
                    connection.close()
                    release_model.set()

                    self.assertTrue(
                        wait_until(
                            lambda: any(
                                "client disconnected before the response" in str(call)
                                for call in self.log_message.call_args_list
                            )
                        ),
                        self.log_message.call_args_list,
                    )
                finally:
                    release_model.set()
                    with suppress(OSError):
                        connection.close()

        # An escaping exception would print here, because log_message is a stub
        # in this class and cannot carry a traceback.
        self.assertNotIn("Traceback", unexpected.getvalue())
        self.assertEqual(self.post(json.dumps({"text": "abc"}).encode())[0], 200)

    def test_health_reports_the_device(self):
        status, headers, body = self.health()

        self.assertEqual(status, 200)
        self.assertEqual(headers["Content-Type"], "application/json")
        self.assertEqual(
            json.loads(body), {"status": "ok", "device": "cpu", "busy": False}
        )

    def test_health_reports_busy_while_inference_holds_the_lock(self):
        class BlockingModel:
            device = "cpu"

            def __init__(self):
                self.holding = threading.Event()
                self.release = threading.Event()

            def predict(self, text):
                self.holding.set()
                self.release.wait(timeout=10)
                return []

        model = BlockingModel()
        with patch.object(Handler, "model", model):
            caller = threading.Thread(
                target=self.post,
                args=(json.dumps({"text": "abc"}).encode(),),
                daemon=True,
            )
            caller.start()
            try:
                self.assertTrue(model.holding.wait(timeout=5))
                status, _, body = self.health()

                self.assertEqual(status, 200)
                self.assertEqual(
                    json.loads(body), {"status": "ok", "device": "cpu", "busy": True}
                )
            finally:
                model.release.set()
                caller.join(timeout=10)

            self.assertFalse(json.loads(self.health()[2])["busy"])

    def test_unknown_path_returns_404(self):
        status, _, body = raw_request(
            self.port, b"GET /nope HTTP/1.0\r\nHost: 127.0.0.1\r\n\r\n"
        )

        self.assertEqual(status, 404)
        self.assertEqual(json.loads(body), {"error": "not found"})


class LogMessageTests(unittest.TestCase):
    def test_log_line_names_the_client_and_the_request(self):
        handler = Handler.__new__(Handler)
        handler.client_address = ("127.0.0.1", 41234)
        captured = io.StringIO()
        with redirect_stderr(captured):
            handler.log_message('"%s" %s', "GET /health", 200)

        self.assertEqual(captured.getvalue(), '[redact] 127.0.0.1 "GET /health" 200\n')


class MainWiringTests(unittest.TestCase):
    def run_main(self, argv: list[str], serve_forever=None):
        recorded: dict[str, Any] = {}
        signals: dict[int, Any] = {}

        class RecordingServer:
            def __init__(self, address, handler):
                recorded["address"] = address
                recorded["handler"] = handler

            def serve_forever(self):
                recorded["served"] = True
                if serve_forever is not None:
                    serve_forever()

            def shutdown(self):
                recorded["shutdown"] = True
                recorded["shutdown_thread"] = threading.current_thread()

            def server_close(self):
                recorded["closed"] = True

        class RecordingModel:
            def __init__(self, cache_dir, requested_device=None):
                self.device = torch.device("cpu")
                recorded["cache_dir"] = cache_dir
                recorded["requested_device"] = requested_device

        handler_had_model = hasattr(Handler, "model")
        # Cast: getattr widens to Any | None, and the attribute is typed RedactModel.
        # Putting the value back is exactly what was there, so the cast states the
        # intent rather than hiding a mismatch.
        handler_previous_model = cast(RedactModel, getattr(Handler, "model", None))

        def restore_handler():
            if handler_had_model:
                Handler.model = handler_previous_model
            elif hasattr(Handler, "model"):
                del Handler.model

        self.addCleanup(restore_handler)

        stderr = io.StringIO()
        with (
            patch(
                "pii_redact_torch.ensure_assets", return_value=Path("/fixture/cache")
            ),
            patch("pii_redact_torch.RedactModel", RecordingModel),
            patch("pii_redact_torch.ThreadingHTTPServer", RecordingServer),
            patch(
                "pii_redact_torch.signal.signal",
                side_effect=lambda number, function: signals.__setitem__(
                    number, function
                ),
            ),
            patch.object(sys, "argv", ["pii_redact_torch.py", *argv]),
            redirect_stderr(stderr),
        ):
            main()

        return recorded, signals, stderr.getvalue()

    def test_both_shutdown_signals_are_registered(self):
        _, signals, _ = self.run_main([])

        self.assertEqual(set(signals), {signal.SIGINT, signal.SIGTERM})

    def test_signal_handler_shuts_the_server_down(self):
        recorded, signals, _ = self.run_main([])

        signals[signal.SIGTERM]()
        self.assertTrue(wait_until(lambda: recorded.get("shutdown")))

    def test_interrupt_signal_shuts_the_server_down(self):
        recorded, signals, _ = self.run_main([])

        signals[signal.SIGINT]()
        self.assertTrue(wait_until(lambda: recorded.get("shutdown")))

    def test_shutdown_runs_off_the_thread_that_serves(self):
        recorded, signals, _ = self.run_main([])

        signals[signal.SIGTERM]()

        self.assertTrue(wait_until(lambda: recorded.get("shutdown")))
        self.assertIsNot(recorded["shutdown_thread"], threading.current_thread())
        self.assertTrue(recorded["shutdown_thread"].daemon)

    def test_server_serves_then_closes(self):
        recorded, _, _ = self.run_main([])

        self.assertTrue(recorded["served"])
        self.assertTrue(recorded["closed"])

    def test_keyboard_interrupt_still_closes_the_server(self):
        def interrupt():
            raise KeyboardInterrupt

        recorded, _, _ = self.run_main([], serve_forever=interrupt)

        self.assertTrue(recorded["closed"])

    def test_default_address_is_loopback_on_the_default_port(self):
        recorded, _, _ = self.run_main([])

        self.assertEqual(recorded["address"], ("127.0.0.1", 9124))

    def test_host_and_port_arguments_reach_the_server(self):
        recorded, _, _ = self.run_main(["--host", "0.0.0.0", "--port", "9999"])

        self.assertEqual(recorded["address"], ("0.0.0.0", 9999))

    def test_device_argument_reaches_the_model(self):
        recorded, _, _ = self.run_main(["--device", "cpu"])

        self.assertEqual(recorded["requested_device"], "cpu")

    def test_device_defaults_to_automatic_selection(self):
        recorded, _, _ = self.run_main([])

        self.assertIsNone(recorded["requested_device"])

    def test_unknown_device_argument_is_rejected(self):
        with self.assertRaises(SystemExit):
            self.run_main(["--device", "tpu"])

    def test_model_is_built_from_the_resolved_cache_directory(self):
        recorded, _, _ = self.run_main([])

        self.assertEqual(recorded["cache_dir"], Path("/fixture/cache"))

    def test_handler_carries_the_loaded_model(self):
        recorded, _, _ = self.run_main([])

        self.assertIs(recorded["handler"], Handler)
        self.assertEqual(str(Handler.model.device), "cpu")

    def test_startup_progress_is_written_to_stderr(self):
        _, _, stderr = self.run_main([])

        self.assertIn("[redact] cache: ", stderr)
        self.assertIn("[redact] loading model...\n", stderr)
        self.assertIn("[redact] device: cpu\n", stderr)
        self.assertIn("[redact] ready on http://127.0.0.1:9124\n", stderr)


SIGTERM_CHILD = """
import sys
from pathlib import Path

sys.path.insert(0, {hooks_dir!r})
import pii_redact_torch


class StubModel:
    device = "cpu"

    def __init__(self, cache_dir, requested_device=None):
        pass


pii_redact_torch.ensure_assets = lambda: Path("/fixture/cache")
pii_redact_torch.RedactModel = StubModel
pii_redact_torch.main()
"""


class ScriptEntryPointTests(unittest.TestCase):
    """Run main() in a real process, signals and all.

    Only the model load is stubbed, because the checkpoint is 88 MB and this
    test is about the signal wiring. The rest is the real path: signal.signal,
    request_shutdown, ThreadingHTTPServer, serve_forever.
    """

    def start_server(self, port: int, device: str = "cpu") -> subprocess.Popen[str]:
        code = SIGTERM_CHILD.format(hooks_dir=str(Path(__file__).resolve().parents[1]))
        return subprocess.Popen(
            [
                sys.executable,
                "-c",
                code,
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
                "--device",
                device,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

    def wait_for_listening(self, process: subprocess.Popen[str], port: int) -> None:
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            if process.poll() is not None:
                self.fail(f"server exited early: {process.communicate()[1]}")
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=1):
                    return
            except OSError:
                time.sleep(0.05)
        self.fail(f"the server on port {port} never accepted a connection")

    def stop_server(self, process: subprocess.Popen[str]) -> None:
        if process.poll() is None:
            process.kill()
        process.communicate(timeout=20)

    def test_sigterm_stops_the_server_process(self):
        """The regression test for the signal deadlock.

        The handler used to call shutdown() on the thread that was inside
        serve_forever(), so the process ignored SIGTERM, held the port, and
        answered nothing. Only a live process shows that.
        """
        port = free_port()
        process = self.start_server(port)
        try:
            self.wait_for_listening(process, port)
            process.send_signal(signal.SIGTERM)

            self.assertEqual(process.wait(timeout=10), 0)
            with self.assertRaises(OSError):
                socket.create_connection(("127.0.0.1", port), timeout=1).close()
        finally:
            self.stop_server(process)

    def test_interrupt_stops_the_server_process(self):
        port = free_port()
        process = self.start_server(port)
        try:
            self.wait_for_listening(process, port)
            process.send_signal(signal.SIGINT)

            self.assertEqual(process.wait(timeout=10), 0)
        finally:
            self.stop_server(process)


if __name__ == "__main__":
    unittest.main()
