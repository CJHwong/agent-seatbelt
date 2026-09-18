"""Redact on LiteRT, so the assets can be downloaded.

The PyTorch path needs a checkpoint the public Redact release no longer serves.
`redact.pt` is absent from every published revision, and the one commit that
still lists it answers 403 from the storage layer, so a machine without the file
already in its cache cannot start the detector at all.

The published LiteRT graph carries the same v0.4.0 weights, the same 89 labels
and the same 256-wide window, needs no accelerator, and downloads. That is what
makes it the backend a fresh install can provision, so `redact` selects it.
`redact-torch` keeps the checkpoint path for a host that already has one.

The cost is speed. The graph is fixed at one window per call, so it cannot batch
the way the PyTorch path does, and it has no usable accelerator: LiteRT's Metal
delegate fails to build this graph on a batch-axis bug of its own. Measured on an
M1 it is about 2.2x the PyTorch path for a whole request, and about 3x on an M5
Pro. A 100-token prompt pays 7 ms; a 16k-token tool output pays 0.9 s.

Only redact mode imports this module.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np  # ty: ignore[unresolved-import]
from ai_edge_litert.interpreter import Interpreter  # ty: ignore[unresolved-import]

from pii_redact_torch import (
    CACHE_DIR,
    MAX_SEQUENCE_LENGTH,
    REDACT_REPO,
    REDACT_REVISION,
    RedactModel,
    # The window geometry belongs to the graph, not to the PyTorch backend: both
    # backends run 254-token windows over a 256-wide input. pii_redact_torch names
    # it with a leading underscore, but its own tests import it by that name, so
    # it is already shared rather than private.
    _window_ranges,
)

GRAPH_FILE = "redact.tflite"
# The graph declares how many classes it has but not what they are called, so the
# label space still comes from config.json. tokenizer.json is the tokenizer the
# PyTorch path loads, unchanged.
REQUIRED_FILES = (GRAPH_FILE, "tokenizer.json", "config.json")


def ensure_assets() -> Path:
    """Download the assets into the cache and return the directory.

    Unlike the PyTorch path this takes no preparation by hand, so it downloads
    instead of reporting what is missing. The revision is pinned, where the
    OpenAI backend tracks its repository's default branch: the graph and the
    label space beside it have to come from the same release, and
    `_check_graph_contract` refuses the pair when they do not.
    """
    from huggingface_hub import hf_hub_download  # ty: ignore[unresolved-import]

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    for name in REQUIRED_FILES:
        hf_hub_download(
            repo_id=REDACT_REPO,
            filename=name,
            revision=REDACT_REVISION,
            local_dir=str(CACHE_DIR),
        )
    return CACHE_DIR


class LitertRedactModel(RedactModel):
    """The same pipeline over the LiteRT graph.

    Three things differ from the PyTorch backend: which object is loaded, how
    one window is assembled, and how the model is called. The tokenizer, the
    chunking, the window geometry, the Viterbi decode, the score threshold, the
    label mapping and the span merge are inherited, so the two backends cannot
    drift apart in anything but the forward pass.
    """

    def __init__(self, cache_dir: Path, num_threads: int | None = None):
        # LiteRT runs this graph on the CPU and has no accelerator here, so the
        # device is not a parameter. Asking for "cpu" skips choose_device's
        # refusal to run without one, which is what redact-torch still requires.
        self._graph_path = cache_dir / GRAPH_FILE
        self._num_threads = num_threads or (os.cpu_count() or 1)
        super().__init__(cache_dir, "cpu")

    def _load_model(self, cache_dir: Path, config_path: Path) -> Any:
        # The interpreter is single threaded unless told otherwise, and one
        # thread costs about 2.4x on this graph: 32.4 ms a window against 13.7
        # with ten, measured on an M1.
        interpreter = Interpreter(
            model_path=str(self._graph_path), num_threads=self._num_threads
        )
        interpreter.allocate_tensors()
        self._bind_graph(interpreter)
        return interpreter

    def _bind_graph(self, interpreter: Any) -> None:
        """Check the graph against the configuration beside it, and index it.

        A graph with a different window width or class count would otherwise
        run, produce probabilities of the wrong width, and read as a detector
        that simply finds less. That is the silent failure this suite keeps
        having to fix, so a mismatch is refused at load instead of at the first
        prompt.
        """
        named: dict[str, dict[str, Any]] = {}
        for detail in interpreter.get_input_details():
            for wanted in ("input_ids", "attention_mask"):
                if str(detail["name"]).endswith(wanted):
                    named[wanted] = detail
        missing = {"input_ids", "attention_mask"} - set(named)
        if missing:
            raise RuntimeError(
                f"{self._graph_path} has no input named {sorted(missing)}"
            )
        for wanted, detail in sorted(named.items()):
            shape = tuple(int(size) for size in detail["shape"])
            expected = (1, MAX_SEQUENCE_LENGTH)
            if shape != expected:
                raise RuntimeError(
                    f"{self._graph_path} input {wanted} is {shape}, expected {expected}"
                )

        output = interpreter.get_output_details()[0]
        classes = int(tuple(output["shape"])[-1])
        if classes != self.label_space.num_classes:
            raise RuntimeError(
                f"{self._graph_path} declares {classes} classes, "
                f"config.json declares {self.label_space.num_classes}"
            )

        self._input_indexes = {
            wanted: int(detail["index"]) for wanted, detail in named.items()
        }
        self._output_index = int(output["index"])

    def _build_batch(self, token_ids: list[int], windows: list[tuple[int, int]]) -> Any:
        """Refuse the batch path rather than fall back to it silently.

        The inherited `_predict_token_probabilities` is replaced below, so the
        batching entry point is never reached. A graph whose input is fixed at
        one row cannot serve a batch of eight, and a caller that found this out
        by getting wrong answers would be worse than one that finds out here.
        """
        raise NotImplementedError(f"{GRAPH_FILE} has a fixed input of one window")

    def _predict_token_probabilities(self, token_ids: list[int]) -> np.ndarray:
        token_count = len(token_ids)
        aggregate = np.zeros(
            (token_count, self.label_space.num_classes), dtype=np.float32
        )
        for start, end in _window_ranges(token_count):
            sequence = [self.bos_id, *token_ids[start:end], self.eos_id]
            input_ids = np.full((1, MAX_SEQUENCE_LENGTH), self.pad_id, dtype=np.int32)
            attention_mask = np.zeros((1, MAX_SEQUENCE_LENGTH), dtype=np.int32)
            input_ids[0, : len(sequence)] = sequence
            attention_mask[0, : len(sequence)] = 1

            probabilities = self._window_probabilities(input_ids, attention_mask)
            # Pool by maximum, exactly as the PyTorch path does, so a token on a
            # window seam keeps the best reading of it rather than the last.
            aggregate[start:end] = np.maximum(
                aggregate[start:end], probabilities[0, 1 : end - start + 1]
            )

        row_sums = aggregate.sum(axis=1, keepdims=True)
        np.divide(aggregate, row_sums, out=aggregate, where=row_sums > 0)
        return aggregate

    def _window_probabilities(
        self, input_ids: np.ndarray, attention_mask: np.ndarray
    ) -> np.ndarray:
        """Run one window and return its softmax probabilities."""
        self.model.set_tensor(self._input_indexes["input_ids"], input_ids)
        self.model.set_tensor(self._input_indexes["attention_mask"], attention_mask)
        self.model.invoke()
        logits = self.model.get_tensor(self._output_index)
        # Shift by the row maximum before exponentiating, which is what torch's
        # softmax does and what keeps a large logit from overflowing.
        shifted = logits - logits.max(axis=-1, keepdims=True)
        exponentials = np.exp(shifted)
        return exponentials / exponentials.sum(axis=-1, keepdims=True)
