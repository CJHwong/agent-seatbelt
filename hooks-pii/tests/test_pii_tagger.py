"""Unit tests for the bilingual tagger backend.

The module imports numpy, onnxruntime and tokenizers at module scope, so every test here
needs those three installed. The windowing, the decoder and the folder checks run for
real. The end-to-end test runs only when PII_TAGGER_DIR names an exported tagger.
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # ty: ignore[unresolved-import]

import pii_tagger
from pii_rules import SPAN_PRIORITY
from pii_tagger import (
    LABEL_MAP,
    REQUIRED_FILES,
    asset_dir,
    bioes_tables,
    chunk_text,
    decode_spans,
    join_touching,
    viterbi,
)

TYPES = ["person", "secret"]
LABELS = ["O"] + [
    f"{prefix}-{entity_type}" for entity_type in TYPES for prefix in "BIES"
]


def log_probs_for(path: list[str], sure: float = 0.9) -> np.ndarray:
    """Rows whose argmax is the given tag path."""
    rest = (1 - sure) / (len(LABELS) - 1)
    rows = np.full((len(path), len(LABELS)), rest)
    for row, tag in enumerate(path):
        rows[row, LABELS.index(tag)] = sure
    return np.log(rows)


class ChunkTextTests(unittest.TestCase):
    def test_whole_sentences_pack_up_to_the_limit(self) -> None:
        text = "第一句。第二句。第三句很長很長。"
        self.assertEqual(
            chunk_text(text, 8), [(0, "第一句。第二句。"), (8, "第三句很長很長。")]
        )

    def test_a_sentence_longer_than_the_limit_is_cut_hard(self) -> None:
        self.assertEqual(
            chunk_text("abcdefghij", 4), [(0, "abcd"), (4, "efgh"), (8, "ij")]
        )


class DecoderTests(unittest.TestCase):
    def test_a_legal_greedy_path_is_kept(self) -> None:
        path = ["B-person", "E-person", "O"]
        found = viterbi(log_probs_for(path), bioes_tables(LABELS))
        self.assertEqual([LABELS[index] for index in found], path)

    def test_an_illegal_greedy_path_becomes_a_legal_one(self) -> None:
        found = viterbi(log_probs_for(["O", "I-person", "O"]), bioes_tables(LABELS))
        tags = [LABELS[index] for index in found]
        self.assertNotIn("I-person", tags[:1])
        self.assertTrue(
            all(tag == "O" or tag.startswith(("B", "I", "E", "S")) for tag in tags)
        )
        self.assertNotEqual(tags, ["O", "I-person", "O"])

    def test_a_span_scores_its_least_sure_token(self) -> None:
        spans = decode_spans(
            ["B-person", "E-person", "S-secret"],
            [(0, 3), (4, 7), (8, 12)],
            [0.9, 0.6, 0.8],
        )
        self.assertEqual(
            spans,
            [
                {"start": 0, "end": 7, "label": "person", "score": 0.6},
                {"start": 8, "end": 12, "label": "secret", "score": 0.8},
            ],
        )

    def test_touching_secrets_join_but_touching_names_do_not(self) -> None:
        spans = [
            {"start": 0, "end": 4, "label": "secret", "score": 0.9},
            {"start": 4, "end": 9, "label": "secret", "score": 0.7},
            {"start": 9, "end": 11, "label": "person", "score": 0.9},
            {"start": 11, "end": 13, "label": "person", "score": 0.9},
        ]
        self.assertEqual(
            [
                (span["start"], span["end"], span["label"], span["score"])
                for span in join_touching(spans)
            ],
            [(0, 9, "secret", 0.7), (9, 11, "person", 0.9), (11, 13, "person", 0.9)],
        )


class LabelMapTests(unittest.TestCase):
    def test_every_tagger_type_maps_to_a_label_the_merge_ranks(self) -> None:
        self.assertEqual(len(LABEL_MAP), 18)
        self.assertTrue(set(LABEL_MAP.values()) <= set(SPAN_PRIORITY))


class AssetDirTests(unittest.TestCase):
    def test_an_unset_folder_is_an_error(self) -> None:
        with (
            patch.dict(os.environ, {"PII_TAGGER_DIR": ""}),
            self.assertRaisesRegex(RuntimeError, "not set"),
        ):
            asset_dir()

    def test_a_folder_missing_files_names_them(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            (Path(folder) / "tokenizer.json").write_text("{}")
            with patch.dict(
                os.environ, {"PII_TAGGER_DIR": folder, "PII_TAGGER_FP32": ""}
            ):
                with self.assertRaisesRegex(
                    RuntimeError, "model.int8.onnx, labels.json"
                ):
                    asset_dir()

    def test_fp32_asks_for_the_fp32_graph(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            for name in ("model.onnx", *REQUIRED_FILES):
                (Path(folder) / name).write_text("")
            with patch.dict(
                os.environ, {"PII_TAGGER_DIR": folder, "PII_TAGGER_FP32": "1"}
            ):
                self.assertEqual(asset_dir(), Path(folder))


@unittest.skipUnless(
    os.environ.get("PII_TAGGER_DIR"), "PII_TAGGER_DIR names no exported tagger"
)
class ExportedTaggerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.model = pii_tagger.TaggerModel(asset_dir())

    def test_a_handle_and_a_key_are_found_and_a_docs_url_is_not(self) -> None:
        key = "AKIA" + "Q3EGRZ7MXN2PLW4T"
        text = f"author=mchen pushed with {key}; see https://docs.python.org/3/library/re.html"
        labels = {
            (text[span["start"] : span["end"]], span["label"])
            for span in self.model.predict(text)
        }
        self.assertIn(("mchen", "private_username"), labels)
        self.assertIn((key, "secret"), labels)
        self.assertFalse(any(label == "private_url" for _, label in labels))

    def test_a_long_text_spans_several_windows(self) -> None:
        text = "今天天氣很好。" * 200 + "聯絡王小明 0912-345-678。"
        spans = self.model.predict(text)
        self.assertTrue(any(span["label"] == "private_phone" for span in spans))


if __name__ == "__main__":
    unittest.main()
