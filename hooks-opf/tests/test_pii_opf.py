"""Unit tests for the OpenAI Privacy Filter wrapper.

The module imports numpy, onnxruntime, tokenizers and huggingface_hub at module
scope, so every test here needs those four installed. No test downloads a
checkpoint: the ONNX session and, where the offsets must be exact, the tokenizer
are injected at the module boundary. Everything else runs for real.
"""

from __future__ import annotations

import importlib.util
import json
import math
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType
from typing import Any, Protocol
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # ty: ignore[unresolved-import]
from tokenizers import (  # ty: ignore[unresolved-import]
    Tokenizer,
    models,
    pre_tokenizers,
)

import pii_opf
from pii_opf import (
    NEG_INF,
    LabelSpace,
    Model,
    ModelOutputShapeError,
    _logsumexp,
    _valid,
    build_transition_tables,
    ensure_assets,
    labels_to_spans,
    viterbi_decode,
)


MODULE_PATH = Path(__file__).resolve().parents[1] / "pii_opf.py"

# The real checkpoint ships 33 BioES labels with "O" at index 0. This keeps the
# layout and drops the entities the tests do not need.
ID2LABEL = {
    0: "O",
    1: "B-account_number",
    2: "I-account_number",
    3: "E-account_number",
    4: "S-account_number",
    5: "B-private_person",
    6: "I-private_person",
    7: "E-private_person",
    8: "S-private_person",
    9: "B-secret",
    10: "I-secret",
    11: "E-secret",
    12: "S-secret",
}
NUM_CLASSES = len(ID2LABEL)

VOCAB = {
    "[UNK]": 0,
    "my": 1,
    "card": 2,
    "is": 3,
    "4111": 4,
    "1111": 5,
    "2222": 6,
    "3333": 7,
}


def load_module_copy(environ: dict[str, str]) -> ModuleType:
    """Import a second copy of pii_opf so module-level env reads can be observed."""
    with patch.dict(os.environ, environ, clear=True):
        spec = importlib.util.spec_from_file_location("pii_opf_reloaded", MODULE_PATH)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"cannot load module: {MODULE_PATH}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    return module


class RecordingSession:
    """Stand-in for an ONNX Runtime session.

    It maps each input token id to a class, then puts a hard peak on that class
    at that position. The transition tables and the Viterbi decoder are the real
    ones, so an illegal peak still gets overruled.
    """

    def __init__(self, class_by_token_id: dict[int, int]):
        self.class_by_token_id = class_by_token_id
        self.feeds: Any = None

    def run(self, output_names: list[str], feeds: dict) -> list:
        self.feeds = feeds
        ids = feeds["input_ids"][0]
        logits = np.zeros((1, len(ids), NUM_CLASSES), dtype=np.float32)
        for position, token_id in enumerate(ids):
            logits[0, position, self.class_by_token_id[int(token_id)]] = 10.0
        return [logits]


def span_logits(
    token_count: int,
    positions: list[int],
    begin: int,
    inside: int,
    end: int,
    classes: int = NUM_CLASSES,
) -> np.ndarray:
    """Logits that label the given token positions as one span of one entity."""
    logits = np.zeros((1, token_count, classes), dtype=np.float32)
    for order, position in enumerate(positions):
        if len(positions) == 1:
            label = end + 1
        else:
            label = (
                begin
                if order == 0
                else (end if order == len(positions) - 1 else inside)
            )
        logits[0, position, label] = 10.0
    return logits


class FixedLogitsSession:
    """Returns whatever logits the test hands it, whatever the input is."""

    def __init__(self, logits: np.ndarray):
        self.logits = logits
        self.feeds: Any = None

    def run(self, output_names: list[str], feeds: dict) -> list:
        self.feeds = feeds
        return [self.logits]


class Session(Protocol):
    """The slice of an ONNX Runtime session that Model uses."""

    feeds: Any

    def run(self, output_names: list[str], feeds: dict) -> list: ...


class StubEncoding:
    def __init__(self, ids: list[int], offsets: list[tuple[int, int]]):
        self.ids = ids
        self.offsets = offsets


class StubTokenizer:
    """Returns fixed token ids and character offsets, whatever the text is."""

    def __init__(self, encoding: StubEncoding):
        self.encoding = encoding
        self.calls = 0

    def encode(self, text: str, add_special_tokens: bool = False) -> StubEncoding:
        self.calls += 1
        return self.encoding


class PiiOpfTestCase(unittest.TestCase):
    def temporary_directory(self) -> Path:
        holder = tempfile.TemporaryDirectory()
        self.addCleanup(holder.cleanup)
        return Path(holder.name)

    def write_tokenizer(self, directory: Path, vocab: dict[str, int]) -> None:
        tokenizer = Tokenizer(models.WordLevel(vocab=vocab, unk_token="[UNK]"))
        tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
        tokenizer.save(str(directory / "tokenizer.json"))

    def write_byte_level_tokenizer(self, directory: Path) -> Tokenizer:
        """A byte-level tokenizer, like the one the checkpoint ships.

        It is what makes the offset question real: a byte-level model can hand
        back offsets that disagree with Python string indices.
        """
        alphabet = pre_tokenizers.ByteLevel.alphabet()
        tokenizer = Tokenizer(
            models.BPE(
                vocab={char: index for index, char in enumerate(alphabet)}, merges=[]
            )
        )
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(
            add_prefix_space=True, use_regex=True
        )
        path = directory / "tokenizer.json"
        tokenizer.save(str(path))
        return Tokenizer.from_file(str(path))

    def build_model(
        self,
        directory: Path,
        session: Session,
        config: dict | None = None,
        vocab: dict[str, int] | None = None,
    ) -> tuple[Model, Any]:
        settings = {"id2label": ID2LABEL, "max_position_embeddings": 512}
        settings.update(config or {})
        (directory / "config.json").write_text(json.dumps(settings))
        self.write_tokenizer(directory, vocab or VOCAB)
        with patch.object(
            pii_opf.ort, "InferenceSession", return_value=session
        ) as constructor:
            model = Model(directory)
        return model, constructor


class LabelSpaceTests(PiiOpfTestCase):
    def test_background_index_points_at_the_outside_label(self) -> None:
        self.assertEqual(LabelSpace(ID2LABEL).bg, 0)

    def test_bioes_names_split_into_a_tag_and_a_span_name(self) -> None:
        ls = LabelSpace(ID2LABEL)
        self.assertEqual((ls.tag[1], ls.span_name[1]), ("B", "account_number"))
        self.assertEqual((ls.tag[2], ls.span_name[2]), ("I", "account_number"))
        self.assertEqual((ls.tag[3], ls.span_name[3]), ("E", "account_number"))
        self.assertEqual((ls.tag[4], ls.span_name[4]), ("S", "account_number"))

    def test_the_outside_label_carries_no_tag_and_no_name(self) -> None:
        ls = LabelSpace(ID2LABEL)
        self.assertIsNone(ls.tag[0])
        self.assertIsNone(ls.span_name[0])

    def test_num_classes_counts_every_label(self) -> None:
        self.assertEqual(LabelSpace(ID2LABEL).num_classes, NUM_CLASSES)


class TransitionTableTests(PiiOpfTestCase):
    def setUp(self) -> None:
        self.start, self.end, self.trans = build_transition_tables(LabelSpace(ID2LABEL))

    def test_a_span_may_start_on_begin_single_or_background(self) -> None:
        for index in (0, 1, 4, 5, 8, 9, 12):
            self.assertEqual(float(self.start[index]), 0.0, msg=str(index))

    def test_a_span_may_not_start_on_inside_or_end(self) -> None:
        for index in (2, 3, 6, 7, 10, 11):
            self.assertEqual(float(self.start[index]), NEG_INF, msg=str(index))

    def test_a_span_may_end_on_end_single_or_background(self) -> None:
        for index in (0, 3, 4, 7, 8, 11, 12):
            self.assertEqual(float(self.end[index]), 0.0, msg=str(index))

    def test_a_span_may_not_end_on_begin_or_inside(self) -> None:
        for index in (1, 2, 5, 6, 9, 10):
            self.assertEqual(float(self.end[index]), NEG_INF, msg=str(index))

    def test_open_and_close_transitions_are_free(self) -> None:
        pairs = [(0, 0), (0, 5), (4, 0), (4, 9), (3, 0), (3, 5), (7, 9)]
        for previous, following in pairs:
            self.assertEqual(float(self.trans[previous, following]), 0.0)

    def test_a_closed_span_may_be_followed_by_a_single_label(self) -> None:
        pairs = [(3, 4), (3, 12), (4, 12), (12, 9), (12, 12)]
        for previous, following in pairs:
            self.assertEqual(float(self.trans[previous, following]), 0.0)

    def test_inside_may_follow_begin_or_inside_of_the_same_entity(self) -> None:
        self.assertEqual(float(self.trans[5, 6]), 0.0)
        self.assertEqual(float(self.trans[6, 6]), 0.0)

    def test_end_may_follow_begin_or_inside_of_the_same_entity(self) -> None:
        self.assertEqual(float(self.trans[5, 7]), 0.0)
        self.assertEqual(float(self.trans[6, 7]), 0.0)

    def test_a_continuation_may_not_switch_entity(self) -> None:
        self.assertEqual(float(self.trans[5, 10]), NEG_INF)
        self.assertEqual(float(self.trans[6, 2]), NEG_INF)

    def test_background_may_not_be_absorbed_into_a_span(self) -> None:
        self.assertEqual(float(self.trans[0, 2]), NEG_INF)
        self.assertEqual(float(self.trans[6, 0]), NEG_INF)

    def test_a_span_may_not_restart_inside_itself(self) -> None:
        self.assertEqual(float(self.trans[5, 1]), NEG_INF)
        self.assertEqual(float(self.trans[6, 4]), NEG_INF)

    def test_an_unknown_tag_is_never_a_valid_predecessor(self) -> None:
        ls = LabelSpace({0: "O", 1: "X-thing"})
        self.assertFalse(_valid(ls, 1, 1))
        self.assertFalse(_valid(ls, 1, 0))


class ViterbiTests(PiiOpfTestCase):
    def setUp(self) -> None:
        self.start, self.end, self.trans = build_transition_tables(LabelSpace(ID2LABEL))

    def test_no_tokens_decodes_to_an_empty_path(self) -> None:
        self.assertEqual(
            viterbi_decode(
                np.zeros((0, NUM_CLASSES)), self.start, self.end, self.trans
            ),
            [],
        )

    def test_one_token_takes_the_best_start_and_end_pair(self) -> None:
        logits = np.zeros((1, NUM_CLASSES), dtype=np.float32)
        logits[0, 12] = 5.0
        path = viterbi_decode(logits, self.start, self.end, self.trans)
        self.assertEqual(path, [12])

    def test_a_token_that_cannot_start_or_end_is_not_chosen(self) -> None:
        logits = np.zeros((1, NUM_CLASSES), dtype=np.float32)
        logits[0, 2] = 5.0
        path = viterbi_decode(logits, self.start, self.end, self.trans)
        self.assertEqual(path, [0])

    def test_an_illegal_transition_loses_to_the_legal_path(self) -> None:
        logits = np.zeros((2, NUM_CLASSES), dtype=np.float32)
        logits[0, 5] = 10.0
        logits[1, 10] = 10.0
        logits[1, 7] = 1.0
        path = viterbi_decode(logits, self.start, self.end, self.trans)
        self.assertEqual(path, [5, 7])

    def test_every_infinite_path_falls_back_to_the_per_token_argmax(self) -> None:
        # NEG_INF is -1e9, which is finite, so the fallback guard needs a caller
        # that passes genuinely infinite tables.
        masked = np.full(NUM_CLASSES, -np.inf, dtype=np.float32)
        masked_trans = np.full((NUM_CLASSES, NUM_CLASSES), -np.inf, dtype=np.float32)
        logits = np.zeros((2, NUM_CLASSES), dtype=np.float32)
        logits[0, 1] = 3.0
        logits[1, 9] = 3.0
        path = viterbi_decode(logits, masked, masked, masked_trans)
        self.assertEqual(path, [1, 9])

    def test_the_masking_constant_is_finite(self) -> None:
        self.assertTrue(np.isfinite(NEG_INF))


class SpanConversionTests(PiiOpfTestCase):
    def setUp(self) -> None:
        self.ls = LabelSpace(ID2LABEL)

    def test_empty_path_produces_no_spans(self) -> None:
        self.assertEqual(labels_to_spans([], self.ls), [])

    def test_a_single_label_is_its_own_span(self) -> None:
        self.assertEqual(
            labels_to_spans([0, 4, 0], self.ls), [(1, 1, "account_number")]
        )

    def test_begin_inside_end_collapses_into_one_span(self) -> None:
        self.assertEqual(
            labels_to_spans([0, 5, 6, 6, 7, 0], self.ls), [(1, 4, "private_person")]
        )

    def test_begin_end_collapses_into_one_span(self) -> None:
        self.assertEqual(labels_to_spans([5, 7], self.ls), [(0, 1, "private_person")])

    def test_background_closes_an_open_span(self) -> None:
        self.assertEqual(
            labels_to_spans([5, 6, 6, 0, 5, 6], self.ls),
            [(0, 2, "private_person"), (4, 5, "private_person")],
        )

    def test_an_open_span_runs_to_the_last_token(self) -> None:
        self.assertEqual(labels_to_spans([0, 9, 10], self.ls), [(1, 2, "secret")])

    def test_a_second_begin_closes_the_first_span(self) -> None:
        self.assertEqual(
            labels_to_spans([5, 5], self.ls),
            [(0, 0, "private_person"), (1, 1, "private_person")],
        )

    def test_a_single_label_closes_an_open_span(self) -> None:
        self.assertEqual(
            labels_to_spans([5, 6, 12], self.ls),
            [(0, 1, "private_person"), (2, 2, "secret")],
        )

    def test_an_inside_label_of_another_entity_starts_a_new_span(self) -> None:
        self.assertEqual(labels_to_spans([6, 10], self.ls), [(1, 1, "secret")])

    def test_an_end_label_without_a_begin_still_forms_a_span(self) -> None:
        self.assertEqual(labels_to_spans([0, 7], self.ls), [(1, 1, "private_person")])

    def test_an_s_label_with_no_name_is_skipped(self) -> None:
        # labels_to_spans takes the label space as a parameter, so a caller can
        # hand it maps that disagree with each other. LabelSpace.__init__ cannot
        # build one: every name except "O" splits into a tag and a span name.
        # Without the guard this appends (0, 0, None), and a None label is a
        # malformed detector response that fails the whole response.
        space = LabelSpace({0: "O", 1: "S-private_email"})
        space.span_name[1] = None
        self.assertEqual(labels_to_spans([1], space), [])

    def test_an_e_label_with_no_name_is_skipped(self) -> None:
        # A lone E, so there is no active span name to report either. An E with
        # no name that closes an open span still reports the active name.
        space = LabelSpace({0: "O", 1: "E-private_email"})
        space.span_name[1] = None
        self.assertEqual(labels_to_spans([1], space), [])

    def test_a_single_label_closes_a_span_then_a_begin_reopens_it(self) -> None:
        self.assertEqual(
            labels_to_spans([9, 12, 9], self.ls),
            [(0, 0, "secret"), (1, 1, "secret"), (2, 2, "secret")],
        )


class LogSumExpTests(PiiOpfTestCase):
    def test_keepdims_returns_the_per_row_denominator(self) -> None:
        values = np.zeros((1, 2), dtype=np.float32)
        result = _logsumexp(values, axis=-1, keepdims=True)
        self.assertEqual(result.shape, (1, 1))
        self.assertAlmostEqual(float(result[0, 0]), math.log(2), places=5)

    def test_without_keepdims_the_axis_is_squeezed(self) -> None:
        values = np.zeros((2, 3), dtype=np.float32)
        result = _logsumexp(values, axis=-1, keepdims=False)
        self.assertEqual(result.shape, (2,))
        self.assertAlmostEqual(float(result[1]), math.log(3), places=5)

    def test_large_scores_do_not_overflow(self) -> None:
        values = np.full((1, 2), 1000.0, dtype=np.float32)
        result = _logsumexp(values, axis=-1, keepdims=True)
        self.assertAlmostEqual(float(result[0, 0]), 1000.0 + math.log(2), places=3)


class AssetResolutionTests(PiiOpfTestCase):
    def test_opf_cache_dir_overrides_the_default_cache(self) -> None:
        target = self.temporary_directory() / "elsewhere"
        module = load_module_copy({"OPF_CACHE_DIR": str(target)})
        self.assertEqual(module.CACHE_DIR, target)

    def test_default_cache_lives_under_the_user_home(self) -> None:
        module = load_module_copy({})
        self.assertEqual(module.CACHE_DIR, Path.home() / ".cache" / "opf")

    def test_ensure_assets_requests_every_required_file_into_the_cache(self) -> None:
        cache = self.temporary_directory() / "nested" / "opf"
        with patch.object(pii_opf, "CACHE_DIR", cache):
            with patch.object(pii_opf, "hf_hub_download") as download:
                result = ensure_assets()

        self.assertEqual(result, cache)
        self.assertTrue(cache.is_dir())
        self.assertEqual(
            [call.kwargs["filename"] for call in download.call_args_list],
            [
                "config.json",
                "tokenizer.json",
                "onnx/model_quantized.onnx",
                "onnx/model_quantized.onnx_data",
            ],
        )
        self.assertEqual(
            {call.kwargs["repo_id"] for call in download.call_args_list},
            {"openai/privacy-filter"},
        )
        self.assertEqual(
            {call.kwargs["local_dir"] for call in download.call_args_list},
            {str(cache)},
        )

    def test_a_failed_download_is_not_swallowed(self) -> None:
        cache = self.temporary_directory()
        with patch.object(pii_opf, "CACHE_DIR", cache):
            with patch.object(
                pii_opf, "hf_hub_download", side_effect=OSError("offline")
            ):
                with self.assertRaises(OSError):
                    ensure_assets()


class ModelConstructionTests(PiiOpfTestCase):
    def test_the_checkpoint_is_loaded_from_the_cache_and_pinned_to_cpu(self) -> None:
        directory = self.temporary_directory()
        session = RecordingSession({})
        _, constructor = self.build_model(directory, session)

        self.assertEqual(
            constructor.call_args.args[0],
            str(directory / "onnx" / "model_quantized.onnx"),
        )
        self.assertEqual(
            constructor.call_args.kwargs["providers"], ["CPUExecutionProvider"]
        )

    def test_the_session_options_trade_memory_for_latency(self) -> None:
        directory = self.temporary_directory()
        _, constructor = self.build_model(directory, RecordingSession({}))
        options = constructor.call_args.kwargs["sess_options"]

        self.assertFalse(options.enable_cpu_mem_arena)
        self.assertFalse(options.enable_mem_pattern)
        self.assertEqual(
            options.intra_op_num_threads, max(1, (os.cpu_count() or 4) // 2)
        )

    def test_the_window_is_capped_by_the_environment_limit(self) -> None:
        directory = self.temporary_directory()
        with patch.dict(os.environ, {"OPF_MAX_TOKENS": "64"}):
            model, _ = self.build_model(
                directory,
                RecordingSession({}),
                config={"max_position_embeddings": 131072},
            )
        self.assertEqual(model.max_len, 64)

    def test_the_window_never_exceeds_the_checkpoint_position_limit(self) -> None:
        directory = self.temporary_directory()
        with patch.dict(os.environ, {"OPF_MAX_TOKENS": "4096"}):
            model, _ = self.build_model(
                directory,
                RecordingSession({}),
                config={"max_position_embeddings": 512},
            )
        self.assertEqual(model.max_len, 512)

    def test_the_default_window_is_the_documented_request_limit(self) -> None:
        directory = self.temporary_directory()
        with patch.dict(os.environ, {}, clear=True):
            model, _ = self.build_model(
                directory,
                RecordingSession({}),
                config={"max_position_embeddings": 131072},
            )
        # 1024 tokens fits the hook's 5 s POST budget on a two-core box. A larger
        # default does not, so the value is pinned here and the README states it.
        self.assertEqual(model.max_len, 1024)

    def test_the_label_space_comes_from_config(self) -> None:
        directory = self.temporary_directory()
        model, _ = self.build_model(directory, RecordingSession({}))
        self.assertEqual(model.ls.num_classes, NUM_CLASSES)
        self.assertEqual(model.start.shape, (NUM_CLASSES,))
        self.assertEqual(model.trans.shape, (NUM_CLASSES, NUM_CLASSES))


class PredictTests(PiiOpfTestCase):
    def model_with(
        self,
        directory: Path,
        session: RecordingSession,
        config: dict | None = None,
    ) -> Model:
        model, _ = self.build_model(directory, session, config=config)
        return model

    def test_empty_text_produces_no_spans(self) -> None:
        model = self.model_with(self.temporary_directory(), RecordingSession({}))
        self.assertEqual(model.predict(""), [])

    def test_text_that_tokenizes_to_nothing_produces_no_spans(self) -> None:
        session = RecordingSession({})
        model = self.model_with(self.temporary_directory(), session)

        self.assertEqual(model.predict("   "), [])
        # The model must not be handed an empty tensor.
        self.assertIsNone(session.feeds)

    def test_a_labeled_token_maps_back_to_its_character_offsets(self) -> None:
        session = RecordingSession({1: 0, 2: 0, 3: 0, 4: 4})
        model = self.model_with(self.temporary_directory(), session)

        spans = model.predict("my card is 4111")

        self.assertEqual(
            spans,
            [{"start": 11, "end": 15, "label": "account_number", "text": "4111"}],
        )
        self.assertEqual(session.feeds["input_ids"].shape, (1, 4))
        self.assertEqual(session.feeds["attention_mask"].tolist(), [[1, 1, 1, 1]])

    def test_a_multi_token_span_covers_the_whole_value(self) -> None:
        session = RecordingSession({1: 0, 2: 0, 3: 0, 4: 0, 5: 5, 6: 6, 7: 7})
        model = self.model_with(self.temporary_directory(), session)

        spans = model.predict("my card is 4111 1111 2222 3333")

        self.assertEqual(
            spans,
            [
                {
                    "start": 16,
                    "end": 30,
                    "label": "private_person",
                    "text": "1111 2222 3333",
                }
            ],
        )

    def test_leading_whitespace_is_trimmed_from_a_span(self) -> None:
        session = RecordingSession({7: 12})
        model = self.model_with(self.temporary_directory(), session)
        model.tokenizer = StubTokenizer(StubEncoding([7], [(0, 8)]))

        self.assertEqual(
            model.predict("  secret  "),
            [{"start": 2, "end": 8, "label": "secret", "text": "secret"}],
        )

    def test_trailing_whitespace_is_trimmed_from_a_span(self) -> None:
        session = RecordingSession({7: 12})
        model = self.model_with(self.temporary_directory(), session)
        model.tokenizer = StubTokenizer(StubEncoding([7], [(2, 10)]))

        self.assertEqual(
            model.predict("  secret  "),
            [{"start": 2, "end": 8, "label": "secret", "text": "secret"}],
        )

    def test_a_span_that_is_only_whitespace_is_dropped(self) -> None:
        session = RecordingSession({7: 12})
        model = self.model_with(self.temporary_directory(), session)
        model.tokenizer = StubTokenizer(StubEncoding([7], [(0, 4)]))

        self.assertEqual(model.predict("    "), [])

    def test_input_above_the_request_limit_raises_instead_of_chunking(self) -> None:
        session = RecordingSession({})
        with patch.dict(os.environ, {"OPF_MAX_TOKENS": "512"}):
            model = self.model_with(
                self.temporary_directory(),
                session,
                config={"max_position_embeddings": 131072},
            )
        model.tokenizer = StubTokenizer(
            StubEncoding(list(range(5000)), [(0, 1)] * 5000)
        )

        with self.assertRaisesRegex(ValueError, "5000 tokens, exceeds max 512"):
            model.predict("x" * 5000)
        self.assertIsNone(session.feeds)

    def test_input_at_the_request_limit_is_accepted(self) -> None:
        session = RecordingSession({0: 0})
        with patch.dict(os.environ, {"OPF_MAX_TOKENS": "512"}):
            model = self.model_with(
                self.temporary_directory(),
                session,
                config={"max_position_embeddings": 131072},
            )
        model.tokenizer = StubTokenizer(StubEncoding([0] * 512, [(0, 1)] * 512))

        self.assertEqual(model.predict("x"), [])
        self.assertEqual(session.feeds["input_ids"].shape, (1, 512))

    def test_input_one_token_under_the_limit_is_accepted(self) -> None:
        session = RecordingSession({0: 0})
        with patch.dict(os.environ, {"OPF_MAX_TOKENS": "512"}):
            model = self.model_with(
                self.temporary_directory(),
                session,
                config={"max_position_embeddings": 131072},
            )
        model.tokenizer = StubTokenizer(StubEncoding([0] * 511, [(0, 1)] * 511))

        self.assertEqual(model.predict("x"), [])
        self.assertEqual(session.feeds["input_ids"].shape, (1, 511))

    def test_input_one_token_over_the_limit_is_rejected(self) -> None:
        session = RecordingSession({0: 0})
        with patch.dict(os.environ, {"OPF_MAX_TOKENS": "512"}):
            model = self.model_with(
                self.temporary_directory(),
                session,
                config={"max_position_embeddings": 131072},
            )
        model.tokenizer = StubTokenizer(StubEncoding([0] * 513, [(0, 1)] * 513))

        with self.assertRaisesRegex(ValueError, "513 tokens, exceeds max 512"):
            model.predict("x")
        self.assertIsNone(session.feeds)

    def test_a_single_character_is_accepted(self) -> None:
        model = self.model_with(self.temporary_directory(), RecordingSession({0: 0}))
        model.tokenizer = StubTokenizer(StubEncoding([0], [(0, 1)]))

        self.assertEqual(model.predict("a"), [])


class MultibyteOffsetTests(PiiOpfTestCase):
    """The reported span must slice the original string back to the match.

    A byte-based offset would shift every span by the extra UTF-8 bytes in the
    prefix, and the hook's masked preview would then print the wrong substring.
    These use a real byte-level tokenizer, which is what the checkpoint ships.
    """

    def byte_level_model(self, text: str, target: str) -> tuple[Model, int]:
        directory = self.temporary_directory()
        tokenizer = self.write_byte_level_tokenizer(directory)
        encoding = tokenizer.encode(text, add_special_tokens=False)
        start = text.index(target)
        labelled = [
            index
            for index, (token_start, token_end) in enumerate(encoding.offsets)
            if token_start < start + len(target) and token_end > start
        ]
        logits = span_logits(len(encoding.ids), labelled, 9, 10, 11)
        model, _ = self.build_model(directory, FixedLogitsSession(logits))
        model.tokenizer = tokenizer
        self.assertTrue(labelled, "the target must tokenize to at least one token")
        return model, start

    def assert_span_slices_the_text(self, text: str, target: str) -> None:
        model, start = self.byte_level_model(text, target)
        self.assertEqual(
            model.predict(text),
            [
                {
                    "start": start,
                    "end": start + len(target),
                    "label": "secret",
                    "text": target,
                }
            ],
            msg=f"offsets drifted for {text!r}",
        )

    def test_an_ascii_target_after_ascii_text(self) -> None:
        self.assert_span_slices_the_text("xx sk_live_ABC123", "sk_live_ABC123")

    def test_a_cjk_prefix_does_not_shift_the_span(self) -> None:
        self.assert_span_slices_the_text("密钥 sk_live_ABC123", "sk_live_ABC123")

    def test_an_emoji_prefix_does_not_shift_the_span(self) -> None:
        self.assert_span_slices_the_text("\U0001f511 sk_live_ABC123", "sk_live_ABC123")

    def test_a_combining_mark_prefix_does_not_shift_the_span(self) -> None:
        self.assert_span_slices_the_text("café sk_live_ABC123", "sk_live_ABC123")

    def test_a_mixed_multibyte_prefix_does_not_shift_the_span(self) -> None:
        self.assert_span_slices_the_text(
            "\U0001f511é密 é sk_live_ABC123", "sk_live_ABC123"
        )

    def test_a_crlf_prefix_does_not_shift_the_span(self) -> None:
        self.assert_span_slices_the_text("line1\r\nsk_live_ABC123", "sk_live_ABC123")

    def test_a_multibyte_target_is_sliced_correctly(self) -> None:
        self.assert_span_slices_the_text("密钥 café here", "café")

    def test_an_emoji_target_is_sliced_correctly(self) -> None:
        self.assert_span_slices_the_text(
            "pin \U0001f511\U0001f512 here", "\U0001f511\U0001f512"
        )


class ModelResponseShapeTests(PiiOpfTestCase):
    """What predict does when the session's output disagrees with the input.

    None of this is reachable with the shipped checkpoint. Its ONNX graph is
    dynamic in sequence_length and emits exactly the 33 classes that
    config.json declares, so a row count, class count or rank mismatch means the
    export, the config, or the download is broken.

    Every one of them raises ModelOutputShapeError, which is deliberately not a
    ValueError. The row count is the one that used to fail silently: a short
    response dropped its trailing detections and returned no spans at all.
    """

    def session_with(self, logits: np.ndarray) -> FixedLogitsSession:
        return FixedLogitsSession(logits)

    def test_a_short_response_is_rejected_instead_of_losing_trailing_tokens(
        self,
    ) -> None:
        text = "my card is 4111 1111"
        full = self.session_with(span_logits(5, [4], 9, 10, 11))
        short = self.session_with(span_logits(4, [], 9, 10, 11))

        with_full, _ = self.build_model(self.temporary_directory(), full)
        with_short, _ = self.build_model(self.temporary_directory(), short)

        self.assertEqual([span["text"] for span in with_full.predict(text)], ["1111"])
        # The model labelled the last token, but a four-row response cannot
        # carry that label. An empty result would lose it without a signal.
        with self.assertRaisesRegex(ModelOutputShapeError, r"shape \(4, 13\)"):
            with_short.predict(text)

    def test_a_long_response_is_rejected(self) -> None:
        session = self.session_with(span_logits(6, [5], 9, 10, 11))
        model, _ = self.build_model(self.temporary_directory(), session)

        with self.assertRaisesRegex(ModelOutputShapeError, r"shape \(6, 13\)"):
            model.predict("my card is 4111 1111")

    def test_a_class_count_mismatch_is_rejected(self) -> None:
        session = self.session_with(span_logits(5, [4], 9, 10, 11, classes=40))
        model, _ = self.build_model(self.temporary_directory(), session)

        with self.assertRaisesRegex(ModelOutputShapeError, r"shape \(5, 40\)"):
            model.predict("my card is 4111 1111")

    def test_a_flat_response_is_rejected(self) -> None:
        session = self.session_with(span_logits(5, [4], 9, 10, 11)[0])
        model, _ = self.build_model(self.temporary_directory(), session)

        with self.assertRaisesRegex(ModelOutputShapeError, r"shape \(13,\)"):
            model.predict("my card is 4111 1111")

    def test_the_shape_error_is_not_a_value_error(self) -> None:
        # The server maps ValueError to HTTP 413 "input too large". A shape
        # mismatch must not reach the operator as an oversized request.
        self.assertTrue(issubclass(ModelOutputShapeError, RuntimeError))
        self.assertFalse(issubclass(ModelOutputShapeError, ValueError))

    def test_the_request_limit_error_stays_a_value_error(self) -> None:
        session = RecordingSession({})
        with patch.dict(os.environ, {"OPF_MAX_TOKENS": "512"}):
            model, _ = self.build_model(
                self.temporary_directory(),
                session,
                config={"max_position_embeddings": 131072},
            )
        model.tokenizer = StubTokenizer(StubEncoding([0] * 513, [(0, 1)] * 513))

        with self.assertRaises(ValueError) as caught:
            model.predict("x")
        self.assertNotIsInstance(caught.exception, ModelOutputShapeError)
        self.assertIsNone(session.feeds)

    def test_all_zero_logits_produce_no_spans(self) -> None:
        session = self.session_with(np.zeros((1, 5, NUM_CLASSES), dtype=np.float32))
        model, _ = self.build_model(self.temporary_directory(), session)

        self.assertEqual(model.predict("my card is 4111 1111"), [])
