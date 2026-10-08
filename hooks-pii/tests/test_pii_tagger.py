"""Unit tests for the bilingual tagger backend.

The module imports numpy, onnxruntime and tokenizers at module scope, so every test here
needs those three installed. The windowing, the decoder and the folder checks run for
real. The end-to-end test runs only when PII_TAGGER_DIR names an exported tagger, so the suite
never downloads the release.
"""

from __future__ import annotations

import io
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
    COREML_FILES,
    LABEL_MAP,
    REQUIRED_FILES,
    NeuralEngineForward,
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
            ["B-person", "E-person", "O", "S-secret"],
            [(0, 3), (4, 7), (8, 9), (10, 14)],
            [0.9, 0.6, 0.99, 0.8],
        )
        self.assertEqual(
            spans,
            [
                {"start": 0, "end": 7, "label": "person", "score": 0.6},
                {"start": 10, "end": 14, "label": "secret", "score": 0.8},
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
    def test_an_unset_folder_downloads_the_pinned_release(self) -> None:
        calls = []

        def download(**kwargs: str) -> str:
            calls.append(kwargs)
            path = Path(kwargs["local_dir"]) / kwargs["filename"]
            path.write_text("")
            return str(path)

        with tempfile.TemporaryDirectory() as folder:
            with (
                patch.dict(os.environ, {"PII_TAGGER_DIR": "", "PII_TAGGER_FP32": ""}),
                patch.object(pii_tagger, "CACHE_DIR", Path(folder)),
                patch.object(pii_tagger, "neural_engine_available", return_value=False),
                patch("huggingface_hub.hf_hub_download", download),
            ):
                self.assertEqual(asset_dir(), Path(folder))
        self.assertEqual(
            [call["filename"] for call in calls], ["model.int8.onnx", *REQUIRED_FILES]
        )
        self.assertTrue(
            all(
                call["repo_id"] == pii_tagger.TAGGER_REPO
                and call["revision"] == pii_tagger.TAGGER_REVISION
                for call in calls
            )
        )

    def test_apple_silicon_also_downloads_the_core_ml_package(self) -> None:
        names = []

        def download(**kwargs: str) -> str:
            names.append(kwargs["filename"])
            path = Path(kwargs["local_dir"]) / kwargs["filename"]
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("")
            return str(path)

        with tempfile.TemporaryDirectory() as folder:
            with (
                patch.dict(os.environ, {"PII_TAGGER_DIR": "", "PII_TAGGER_FP32": ""}),
                patch.object(pii_tagger, "CACHE_DIR", Path(folder)),
                patch.object(pii_tagger, "neural_engine_available", return_value=True),
                patch("huggingface_hub.hf_hub_download", download),
            ):
                asset_dir()
        self.assertEqual(names, ["model.int8.onnx", *REQUIRED_FILES, *COREML_FILES])

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


class NeuralEngineAvailableTests(unittest.TestCase):
    def check(self, system: str, machine: str, fp32: str, installed: bool) -> bool:
        with (
            patch.object(pii_tagger.platform, "system", return_value=system),
            patch.object(pii_tagger.platform, "machine", return_value=machine),
            patch.object(
                pii_tagger.importlib.util,
                "find_spec",
                return_value=object() if installed else None,
            ),
            patch.dict(os.environ, {"PII_TAGGER_FP32": fp32}),
        ):
            return pii_tagger.neural_engine_available()

    def test_apple_silicon_with_coremltools_takes_it(self) -> None:
        self.assertTrue(self.check("Darwin", "arm64", "", True))

    def test_linux_intel_fp32_or_no_coremltools_do_not(self) -> None:
        self.assertFalse(self.check("Linux", "x86_64", "", True))
        self.assertFalse(self.check("Darwin", "x86_64", "", True))
        self.assertFalse(self.check("Darwin", "arm64", "1", True))
        self.assertFalse(self.check("Darwin", "arm64", "", False))


class FakeFunction:
    """A Core ML function of one length: records its input, returns ranked logits."""

    def __init__(self, length: int, seen: list) -> None:
        self.length, self.seen = length, seen

    def predict(self, feed: dict) -> dict:
        self.seen.append((self.length, feed["input_ids"], feed["attention_mask"]))
        rows = np.arange(self.length * 3, dtype=np.float16).reshape(1, self.length, 3)
        return {"logits": rows}


class NeuralEngineForwardTests(unittest.TestCase):
    def forward(self, seen: list) -> NeuralEngineForward:
        return NeuralEngineForward(
            {length: FakeFunction(length, seen) for length in (64, 128, 256, 512)}
        )

    def test_a_window_pads_to_the_shortest_length_that_holds_it(self) -> None:
        seen: list = []
        logits = self.forward(seen).logits(list(range(1, 71)))
        length, ids, mask = seen[0]
        self.assertEqual(length, 128)
        self.assertEqual(ids.dtype, np.int32)
        self.assertEqual(ids[0, :70].tolist(), list(range(1, 71)))
        self.assertEqual(mask[0].tolist(), [1] * 70 + [0] * 58)
        self.assertEqual(logits.shape, (70, 3))
        self.assertEqual(logits.dtype, np.float32)

    def test_a_thread_reuses_one_kept_buffer_per_length(self) -> None:
        seen: list = []
        forward = self.forward(seen)
        forward.logits([5] * 70)
        forward.logits([6] * 90)
        self.assertIs(seen[0][1], seen[1][1])
        self.assertIs(seen[0][2], seen[1][2])
        self.assertIn(seen[0][1], [feed["input_ids"] for feed in forward.kept])

    def test_a_shorter_window_clears_the_longer_one_before_it(self) -> None:
        seen: list = []
        forward = self.forward(seen)
        forward.logits([5] * 90)
        forward.logits([6] * 70)
        _, ids, mask = seen[1]
        self.assertEqual(ids[0].tolist(), [6] * 70 + [0] * 58)
        self.assertEqual(mask[0].tolist(), [1] * 70 + [0] * 58)

    def test_a_window_longer_than_every_length_is_refused(self) -> None:
        with self.assertRaisesRegex(ValueError, "513 tokens"):
            self.forward([]).logits([1] * 513)


class LoadForwardTests(unittest.TestCase):
    def test_a_core_ml_failure_falls_back_to_the_cpu_and_says_why(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            (Path(folder) / "model.mlpackage").mkdir()
            with (
                patch.object(pii_tagger, "neural_engine_available", return_value=True),
                patch.object(
                    pii_tagger.NeuralEngineForward,
                    "load",
                    side_effect=RuntimeError("no Neural Engine"),
                ),
                patch.object(pii_tagger, "OnnxForward", return_value="cpu forward"),
                patch("sys.stderr", new_callable=io.StringIO) as stderr,
            ):
                forward = pii_tagger.load_forward(Path(folder), "model.int8.onnx")
        self.assertEqual(forward, "cpu forward")
        self.assertIn("no Neural Engine", stderr.getvalue())

    def test_a_folder_without_the_package_uses_the_cpu(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            with (
                patch.object(pii_tagger, "neural_engine_available", return_value=True),
                patch.object(pii_tagger, "OnnxForward", return_value="cpu forward"),
            ):
                forward = pii_tagger.load_forward(Path(folder), "model.int8.onnx")
        self.assertEqual(forward, "cpu forward")


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

    def test_a_window_tags_the_same_beside_any_neighbour(self) -> None:
        window = (
            "author=mchen pushed with AKIA" + "Q3EGRZ7MXN2PLW4T, ask Wang Xiaoming.\n"
        )
        filler = "今天天氣很好。" * 57
        alone = self.model.tag(window)
        beside = [
            {
                **span,
                "start": span["start"] - len(filler),
                "end": span["end"] - len(filler),
            }
            for span in self.model.tag(filler + window)
            if span["start"] >= len(filler)
        ]
        self.assertTrue(alone)
        self.assertEqual(alone, beside)


if __name__ == "__main__":
    unittest.main()
