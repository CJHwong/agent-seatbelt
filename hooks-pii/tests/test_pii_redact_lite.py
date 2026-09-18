#!/usr/bin/env python3
"""Tests for pii_redact_lite.py.

This suite needs the model dependencies, so it is not in coverage.sh's offline
set and CI does not run it. Run it with:

  uv run --with ai-edge-litert --with tokenizers --with numpy \
      --with torch --with transformers python -m unittest discover -s . -p 'test_pii_redact_lite.py'

The interpreter is stubbed rather than loaded. A real 23 MB graph would test the
LiteRT runtime, and what these tests are for is the code around it: the graph
contract checked at load, the per-window loop, and the pooling that has to match
the PyTorch backend's.
"""

from __future__ import annotations

import json
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np  # ty: ignore[unresolved-import]
import numpy.typing as npt  # ty: ignore[unresolved-import]
from tokenizers import (  # ty: ignore[unresolved-import]
    Tokenizer,
    models,
    pre_tokenizers,
)

# The module under test lives one directory up, beside the server that imports it.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pii_redact_lite
from pii_redact_lite import LitertRedactModel
from pii_redact_torch import MAX_SEQUENCE_LENGTH, _window_ranges


LITE_ID2LABEL = {
    0: "O",
    1: "B-EMAIL",
    2: "I-EMAIL",
    3: "E-EMAIL",
    4: "S-EMAIL",
    5: "S-GIVEN_NAME",
}
CLASS_COUNT = len(LITE_ID2LABEL)
FIXTURE_VOCAB = {"[UNK]": 0, "<s>": 1, "</s>": 2, "<pad>": 3, "hello": 4, "world": 5}


def write_lite_cache(
    cache_dir: Path,
    id2label: dict[int, str] | None = None,
    window_width: int = MAX_SEQUENCE_LENGTH,
    class_count: int | None = None,
) -> None:
    """Write a cache the LiteRT backend can load, with a scripted graph contract."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = Tokenizer(
        models.WordLevel(vocab=dict(FIXTURE_VOCAB), unk_token="[UNK]")
    )
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.save(str(cache_dir / "tokenizer.json"))
    labels = LITE_ID2LABEL if id2label is None else id2label
    (cache_dir / "config.json").write_text(
        json.dumps({"id2label": {str(key): name for key, name in labels.items()}})
    )
    FakeInterpreter.window_width = window_width
    FakeInterpreter.class_count = CLASS_COUNT if class_count is None else class_count
    FakeInterpreter.logits = []


class FakeInterpreter:
    """The slice of the LiteRT interpreter the backend uses."""

    window_width = MAX_SEQUENCE_LENGTH
    class_count = CLASS_COUNT
    logits: list[npt.NDArray[np.float32]] = []

    def __init__(self, model_path: str, num_threads: int | None = None) -> None:
        self.model_path = model_path
        self.num_threads = num_threads
        self.allocated = False
        self.written: list[tuple[npt.NDArray[np.int32], npt.NDArray[np.int32]]] = []
        self.invocations = 0
        self._pending = np.full(
            (1, self.window_width, self.class_count), -10.0, np.float32
        )

    def allocate_tensors(self) -> None:
        self.allocated = True

    def get_input_details(self) -> list[dict[str, object]]:
        return [
            {
                "name": "serving_default_input_ids",
                "shape": np.array([1, self.window_width]),
                "index": 0,
            },
            {
                "name": "serving_default_attention_mask",
                "shape": np.array([1, self.window_width]),
                "index": 1,
            },
        ]

    def get_output_details(self) -> list[dict[str, object]]:
        return [
            {
                "name": "serving_default_logits_output",
                "shape": np.array([1, self.window_width, self.class_count]),
                "index": 2,
            }
        ]

    def set_tensor(self, index: int, value: npt.NDArray[np.int32]) -> None:
        if index == 0:
            self._input_ids = value
        else:
            self._attention = value
        if index == 1:
            self.written.append((self._input_ids.copy(), value.copy()))

    def invoke(self) -> None:
        scripted = FakeInterpreter.logits
        if self.invocations < len(scripted):
            self._pending = scripted[self.invocations]
        self.invocations += 1

    def get_tensor(self, index: int) -> npt.NDArray[np.float32]:
        return self._pending


def load(cache_dir: Path, num_threads: int | None = None) -> LitertRedactModel:
    """Build a model with the interpreter swapped for the fake."""
    with patch.object(pii_redact_lite, "Interpreter", FakeInterpreter):
        return LitertRedactModel(cache_dir, num_threads)


class EnsureAssetsTests(unittest.TestCase):
    def test_the_three_assets_are_downloaded_at_the_pinned_revision(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as temporary:
            cache_dir = Path(temporary) / "assets"
            calls: list[dict[str, str]] = []

            def record(**kwargs: str) -> str:
                calls.append(kwargs)
                return str(cache_dir / kwargs["filename"])

            with (
                patch.object(pii_redact_lite, "CACHE_DIR", cache_dir),
                patch("huggingface_hub.hf_hub_download", record),
            ):
                returned = pii_redact_lite.ensure_assets()

            self.assertEqual(returned, cache_dir)
            self.assertTrue(cache_dir.is_dir())
            self.assertEqual(
                [call["filename"] for call in calls],
                list(pii_redact_lite.REQUIRED_FILES),
            )
            for call in calls:
                self.assertEqual(call["repo_id"], pii_redact_lite.REDACT_REPO)
                self.assertEqual(call["revision"], pii_redact_lite.REDACT_REVISION)
                self.assertEqual(call["local_dir"], str(cache_dir))


class GraphContractTests(unittest.TestCase):
    """A graph that disagrees with the config beside it must not load.

    It would otherwise run, produce rows of the wrong width, and read as a
    detector that simply finds less.
    """

    def setUp(self) -> None:
        import tempfile

        self._temporary = tempfile.TemporaryDirectory(prefix="lite-test.")
        self.addCleanup(self._temporary.cleanup)
        self.cache_dir = Path(self._temporary.name) / "cache"

    def test_a_matching_graph_loads_and_records_its_tensors(self) -> None:
        write_lite_cache(self.cache_dir)

        model = load(self.cache_dir)

        self.assertTrue(model.model.allocated)
        self.assertEqual(model._num_threads, os.cpu_count() or 1)
        self.assertEqual(model.device.type, "cpu")

    def test_the_thread_count_can_be_given(self) -> None:
        write_lite_cache(self.cache_dir)

        self.assertEqual(load(self.cache_dir, num_threads=3).model.num_threads, 3)

    def test_a_graph_with_the_wrong_window_width_is_refused(self) -> None:
        write_lite_cache(self.cache_dir, window_width=128)

        with self.assertRaisesRegex(RuntimeError, "expected \\(1, 256\\)"):
            load(self.cache_dir)

    def test_a_graph_with_the_wrong_class_count_is_refused(self) -> None:
        """The graph cannot name its labels, so the count is the only check."""
        write_lite_cache(self.cache_dir, class_count=CLASS_COUNT + 3)

        with self.assertRaisesRegex(
            RuntimeError, f"declares {CLASS_COUNT + 3} classes"
        ):
            load(self.cache_dir)

    def test_a_graph_without_the_named_inputs_is_refused(self) -> None:
        write_lite_cache(self.cache_dir)

        def unnamed(self: FakeInterpreter) -> list[dict[str, object]]:
            return [{"name": "x", "shape": np.array([1, 256]), "index": 0}]

        with (
            patch.object(pii_redact_lite, "Interpreter", FakeInterpreter),
            patch.object(FakeInterpreter, "get_input_details", unnamed),
        ):
            with self.assertRaisesRegex(RuntimeError, "no input named"):
                LitertRedactModel(self.cache_dir)


class BatchRefusalTests(unittest.TestCase):
    def test_asking_for_a_batch_is_refused_rather_than_served_wrongly(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as temporary:
            cache_dir = Path(temporary) / "cache"
            write_lite_cache(cache_dir)
            model = load(cache_dir)

            with self.assertRaises(NotImplementedError):
                model._build_batch([1, 2, 3], [(0, 3)])


class WindowProbabilityTests(unittest.TestCase):
    """The per-window loop, and the pooling that has to match the PyTorch path."""

    def setUp(self) -> None:
        import tempfile

        self._temporary = tempfile.TemporaryDirectory(prefix="lite-window.")
        self.addCleanup(self._temporary.cleanup)
        self.cache_dir = Path(self._temporary.name) / "cache"
        write_lite_cache(self.cache_dir)
        self.model = load(self.cache_dir)

    def test_a_short_input_runs_one_window_and_its_rows_sum_to_one(self) -> None:
        probabilities = self.model._predict_token_probabilities([4, 5, 4])

        self.assertEqual(self.model.model.invocations, 1)
        for row in probabilities:
            self.assertAlmostEqual(float(row.sum()), 1.0, places=5)

    def test_one_window_is_run_per_window_of_the_input(self) -> None:
        token_ids = [4] * 255

        self.model._predict_token_probabilities(token_ids)

        self.assertEqual(self.model.model.invocations, len(_window_ranges(255)))

    def test_the_input_is_padded_to_the_window_width_with_bos_and_eos(self) -> None:
        self.model._predict_token_probabilities([4, 5, 4])

        input_ids, attention_mask = self.model.model.written[0]
        self.assertEqual(input_ids.shape, (1, MAX_SEQUENCE_LENGTH))
        self.assertEqual(input_ids.dtype, np.int32)
        self.assertEqual([int(value) for value in input_ids[0][:5]], [1, 4, 5, 4, 2])
        self.assertEqual([int(value) for value in input_ids[0][5:8]], [3, 3, 3])
        self.assertEqual(
            [int(value) for value in attention_mask[0][:5]], [1, 1, 1, 1, 1]
        )
        self.assertEqual([int(value) for value in attention_mask[0][5:8]], [0, 0, 0])

    def test_a_token_covered_twice_keeps_both_readings(self) -> None:
        """Pooling by maximum, not by last writer.

        The overlap region belongs to two windows. If the later window replaced
        the earlier one, a token the first window read confidently could lose
        that reading, so the two arms are given different high classes at the
        same token and both must survive.
        """
        from pii_redact_torch import CONTENT_WINDOW_LENGTH, WINDOW_OVERLAP

        # The last token the second window starts on, so both windows cover it:
        # content index `overlap_token` in the first, and index 0 in the second.
        overlap_token = CONTENT_WINDOW_LENGTH - WINDOW_OVERLAP
        first = np.full((1, MAX_SEQUENCE_LENGTH, CLASS_COUNT), -10.0, np.float32)
        second = np.full((1, MAX_SEQUENCE_LENGTH, CLASS_COUNT), -10.0, np.float32)
        # The graph adds a slot for BOS, so content index i is graph index i + 1.
        first[0, overlap_token + 1, 4] = 10.0
        second[0, 1, 5] = 10.0
        FakeInterpreter.logits = [first, second]
        self.addCleanup(setattr, FakeInterpreter, "logits", [])

        probabilities = self.model._predict_token_probabilities([4] * 255)

        row = probabilities[overlap_token]
        self.assertGreater(float(row[4]), 0.2)
        self.assertGreater(float(row[5]), 0.2)
        self.assertAlmostEqual(float(row.sum()), 1.0, places=5)


if __name__ == "__main__":
    unittest.main()
