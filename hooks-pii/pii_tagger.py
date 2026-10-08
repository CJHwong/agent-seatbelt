"""The bilingual tagger: a 35M-parameter BIOES tagger for English, Simplified and
Traditional Chinese, then the rules it was scored with, then the deterministic secret
and card rules.

The files download from the Hugging Face release on first use, or come from the local
folder `PII_TAGGER_DIR` names: the int8 ONNX graph (fp32 with `PII_TAGGER_FP32=1`), the tokenizer, the label list, the
public persons and places lists, and `tagger.json` with the window sizes and the url and
username cuts. On Apple Silicon with coremltools, the release's Core ML package runs the
network on the Neural Engine instead: fp16, each window padded to a fixed length. Its
spans match the fp32 network's on all but 93 of 33,479 comparison rows, against 1,963
for int8, and a window takes 0.4 ms against 1.4 ms on an M5 Pro. When Core ML fails to
load, the int8 graph runs on the CPU and the server log says why.

Text is cut into windows of whole sentences, at most `chunk_chars` characters each, the
way the tagger was trained and scored. Each window runs alone, without padding, and
several windows run at once on a few threads each. Alone, a window's result does not
depend on its neighbours: dynamic int8 quantizes the activations of a whole call together.
On an 18-core M5 Pro, 6 calls of 3 threads tag 36k tokens a second, one batched call of
every thread 23k. The decoder is the training repository's: hard
BIOES transitions, and a span's score is the probability of its least sure token.

Only tagger mode imports this module.
"""

from __future__ import annotations

import gzip
import importlib.util
import json
import os
import platform
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np  # ty: ignore[unresolved-import]
import onnxruntime  # ty: ignore[unresolved-import]
from tokenizers import Tokenizer  # ty: ignore[unresolved-import]

from pii_rules import deterministic_spans, merge_spans
from pii_tagger_rules import apply_chain

# Release 2026.10 of the model repository.
TAGGER_REPO = "cjhwong/seatbelt-pii-tagger"
TAGGER_REVISION = "4e4c9d818bdcbcf4a4b10d8ec1144b35e02c997a"
CACHE_DIR = Path.home() / ".cache" / "pii-tagger"
GRAPH_INT8 = "model.int8.onnx"
GRAPH_FP32 = "model.onnx"
REQUIRED_FILES = (
    "tokenizer.json",
    "labels.json",
    "persons.txt.gz",
    "places.txt.gz",
    "tagger.json",
)
# One function per padded length over shared weights; the Neural Engine runs only fixed
# shapes. tagger.json caps a window at 512 tokens, so the longest length holds any window.
COREML_PACKAGE = "model.mlpackage"
COREML_LENGTHS = (64, 128, 256, 512)
COREML_FILES = tuple(
    f"{COREML_PACKAGE}/{name}"
    for name in (
        "Manifest.json",
        "Data/com.apple.CoreML/model.mlmodel",
        "Data/com.apple.CoreML/weights/weight.bin",
    )
)
SPACE_MARK = "▁"
SENTENCE_ENDS = "。！？\n"
THREADS_PER_CALL = 3
# The deterministic rules flag every URL, email, phone and date by shape, which would put
# back the documentation links and shared mailboxes the tagger's rules drop. Only their
# credential and card findings join the tagger's spans.
RULE_LABELS = {"secret", "account_number"}
JOIN_TOUCHING = {"secret"}
# Eighteen tagger types onto the hook's labels. Every identity number is an
# account_number, as the Redact map does with passports and tax IDs.
LABEL_MAP = {
    "person": "private_person",
    "username": "private_username",
    "email": "private_email",
    "phone": "private_phone",
    "address": "private_address",
    "url": "private_url",
    "date": "private_date",
    "secret": "secret",
    "account_number": "account_number",
    "national_id": "account_number",
    "passport": "account_number",
    "driver_license": "account_number",
    "nhi_card": "account_number",
    "household_no": "account_number",
    "military_id": "account_number",
    "medical_license": "account_number",
    "license_plate": "account_number",
    "company_id": "account_number",
}


def neural_engine_available() -> bool:
    """Apple Silicon with coremltools installed, and no request for the fp32 graph."""
    return (
        platform.system() == "Darwin"
        and platform.machine() == "arm64"
        and os.environ.get("PII_TAGGER_FP32") != "1"
        and importlib.util.find_spec("coremltools") is not None
    )


def download(graph: str) -> Path:
    """Download the release files into the cache and return the directory.

    The revision is pinned to a commit, not the tag, so a moved tag cannot swap the
    graph under a running hook. The CPU graph comes too on Apple Silicon, so a failed
    Core ML load still has something to fall back to.
    """
    from huggingface_hub import hf_hub_download  # ty: ignore[unresolved-import]

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    package = COREML_FILES if neural_engine_available() else ()
    for name in (graph, *REQUIRED_FILES, *package):
        hf_hub_download(
            repo_id=TAGGER_REPO,
            filename=name,
            revision=TAGGER_REVISION,
            local_dir=str(CACHE_DIR),
        )
    return CACHE_DIR


def asset_dir() -> Path:
    """The folder with every file the backend reads: PII_TAGGER_DIR, else the release."""
    graph = GRAPH_FP32 if os.environ.get("PII_TAGGER_FP32") == "1" else GRAPH_INT8
    configured = os.environ.get("PII_TAGGER_DIR", "")
    folder = Path(configured).expanduser() if configured else download(graph)
    missing = [
        name for name in (graph, *REQUIRED_FILES) if not (folder / name).is_file()
    ]
    if missing:
        raise RuntimeError(f"tagger folder {folder} lacks {', '.join(missing)}")
    return folder


def read_names(path: Path) -> set[str]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return {line.rstrip("\n") for line in handle}


def chunk_text(text: str, max_chars: int) -> list[tuple[int, str]]:
    """Pack whole sentences into chunks of at most max_chars, as (offset, chunk) pairs.

    A single sentence longer than max_chars is cut hard at max_chars.
    """
    sentences, start = [], 0
    for index, character in enumerate(text):
        if character in SENTENCE_ENDS:
            sentences.append((start, index + 1))
            start = index + 1
    if start < len(text):
        sentences.append((start, len(text)))
    pieces = [
        (s, min(s + max_chars, e)) for s, e in sentences for s in range(s, e, max_chars)
    ]
    chunks: list[tuple[int, int]] = []
    for piece_start, piece_end in pieces:
        if chunks and piece_end - chunks[-1][0] <= max_chars:
            chunks[-1] = (chunks[-1][0], piece_end)
        else:
            chunks.append((piece_start, piece_end))
    return [
        (chunk_start, text[chunk_start:chunk_end]) for chunk_start, chunk_end in chunks
    ]


def split_tag(tag: str) -> tuple[str, str | None]:
    if tag == "O":
        return "O", None
    prefix, entity_type = tag.split("-", 1)
    return prefix, entity_type


def allowed_transition(previous: str, current: str) -> bool:
    previous_prefix, previous_type = split_tag(previous)
    current_prefix, current_type = split_tag(current)
    if previous_prefix in ("B", "I"):
        return current_prefix in ("I", "E") and current_type == previous_type
    return current_prefix in ("O", "B", "S")


def bioes_tables(labels: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Transition scores (0 allowed, -inf forbidden) and the legal first and last tags."""
    allowed = np.array([[allowed_transition(a, b) for b in labels] for a in labels])
    can_start = np.array([split_tag(label)[0] in ("O", "B", "S") for label in labels])
    can_end = np.array([split_tag(label)[0] in ("O", "E", "S") for label in labels])
    return np.where(allowed, 0.0, -np.inf), can_start, can_end


def viterbi(
    log_probs: np.ndarray, tables: tuple[np.ndarray, np.ndarray, np.ndarray]
) -> list[int]:
    """The best BIOES path. Transitions are 0 or -inf, so a legal greedy path is the answer."""
    transitions, can_start, can_end = tables
    greedy = log_probs.argmax(axis=1)
    legal = (
        can_start[greedy[0]]
        and can_end[greedy[-1]]
        and not np.isinf(transitions[greedy[:-1], greedy[1:]]).any()
    )
    if legal:
        return greedy.tolist()
    scores = np.where(can_start, log_probs[0], -np.inf)
    back = []
    for row in log_probs[1:]:
        candidates = scores[:, None] + transitions
        best = candidates.argmax(axis=0)
        scores = candidates[best, np.arange(len(can_start))] + row
        back.append(best)
    path = [int(np.where(can_end, scores, -np.inf).argmax())]
    for best in reversed(back):
        path.append(int(best[path[-1]]))
    return path[::-1]


def decode_spans(
    tags: list[str], spans: list[tuple[int, int]], confidence: list[float]
) -> list[dict]:
    """Character spans from a legal tag path; a span's score is its least sure token."""
    found, start, least = [], None, 1.0
    for tag, (token_start, token_end), sure in zip(
        tags, spans, confidence, strict=True
    ):
        prefix, entity_type = split_tag(tag)
        if prefix == "S":
            found.append(
                {
                    "start": token_start,
                    "end": token_end,
                    "label": entity_type,
                    "score": sure,
                }
            )
        elif prefix == "B":
            start, least = token_start, sure
        elif prefix in ("I", "E"):
            least = min(least, sure)
            if prefix == "E" and start is not None:
                found.append(
                    {
                        "start": start,
                        "end": token_end,
                        "label": entity_type,
                        "score": least,
                    }
                )
                start = None
    return found


def join_touching(spans: list[dict]) -> list[dict]:
    """Two secret spans that touch are one secret: a long key can end at a chunk edge."""
    joined: list[dict] = []
    for span in sorted(spans, key=lambda item: item["start"]):
        previous = joined[-1] if joined else None
        if (
            previous is not None
            and previous["label"] == span["label"]
            and previous["end"] == span["start"]
            and span["label"] in JOIN_TOUCHING
        ):
            joined[-1] = {
                **previous,
                "end": span["end"],
                "score": min(previous["score"], span["score"]),
            }
        else:
            joined.append(span)
    return joined


class OnnxForward:
    """The ONNX graph on onnxruntime's CPU provider, one unpadded window per call."""

    device = "cpu"

    def __init__(self, graph: Path) -> None:
        options = onnxruntime.SessionOptions()
        options.intra_op_num_threads = THREADS_PER_CALL
        self.session = onnxruntime.InferenceSession(
            str(graph), options, providers=["CPUExecutionProvider"]
        )

    def logits(self, ids: list[int]) -> np.ndarray:
        batch = np.array([ids], dtype=np.int64)
        return self.session.run(
            ["logits"], {"input_ids": batch, "attention_mask": np.ones_like(batch)}
        )[0][0].astype(np.float32)


class NeuralEngineForward:
    """The Core ML package on the Neural Engine: a window padded to the shortest length.

    The padding is masked out, so a window's logits do not depend on its length bucket.

    The inputs are buffers that live as long as the process, one pair per thread and
    length, filled in place. coremltools wraps each numpy input in an object that holds a
    reference to it, and Core ML releases that object on its own thread when it resets an
    idle stream, without the GIL. If that release dropped the last reference, Python would
    free the array off its thread and the server would segfault after its first quiet
    spell. `kept` holds every buffer, so that release never frees one.
    """

    device = "neural_engine"

    def __init__(self, functions: dict) -> None:
        self.functions = functions
        self.local = threading.local()
        self.kept: list[dict[str, np.ndarray]] = []

    @classmethod
    def load(cls, package: Path) -> NeuralEngineForward:
        import coremltools  # ty: ignore[unresolved-import]

        return cls(
            {
                length: coremltools.models.MLModel(
                    str(package),
                    function_name=f"length_{length}",
                    compute_units=coremltools.ComputeUnit.CPU_AND_NE,
                )
                for length in COREML_LENGTHS
            }
        )

    def logits(self, ids: list[int]) -> np.ndarray:
        length = next(
            (size for size in sorted(self.functions) if size >= len(ids)), None
        )
        if length is None:
            raise ValueError(
                f"a window of {len(ids)} tokens exceeds the longest Core ML length"
            )
        feed = self.feed(length)
        feed["input_ids"][0, : len(ids)] = ids
        feed["input_ids"][0, len(ids) :] = 0
        feed["attention_mask"][0, : len(ids)] = 1
        feed["attention_mask"][0, len(ids) :] = 0
        out = self.functions[length].predict(feed)
        return np.asarray(out["logits"], dtype=np.float32)[0, : len(ids)]

    def feed(self, length: int) -> dict[str, np.ndarray]:
        """This thread's input buffers for one length, made once and never released."""
        feeds = self.local.__dict__.setdefault("feeds", {})
        if length not in feeds:
            feeds[length] = {
                "input_ids": np.zeros((1, length), dtype=np.int32),
                "attention_mask": np.zeros((1, length), dtype=np.int32),
            }
            self.kept.append(feeds[length])
        return feeds[length]


def load_forward(folder: Path, graph: str) -> OnnxForward | NeuralEngineForward:
    """The Neural Engine when this machine and the folder allow it, else the CPU graph."""
    package = folder / COREML_PACKAGE
    if neural_engine_available() and package.is_dir():
        try:
            return NeuralEngineForward.load(package)
        except Exception as error:  # Core ML reports a bad host through many types
            print(
                f"[tagger] Core ML failed to load ({type(error).__name__}: {error}); "
                f"using {graph} on the CPU",
                file=sys.stderr,
                flush=True,
            )
    return OnnxForward(folder / graph)


class TaggerModel:
    """The tagger behind the hook's predict(text) interface, on the Neural Engine or the CPU."""

    def __init__(self, folder: Path) -> None:
        graph = GRAPH_FP32 if os.environ.get("PII_TAGGER_FP32") == "1" else GRAPH_INT8
        self.forward = load_forward(folder, graph)
        self.device = self.forward.device
        self.calls = ThreadPoolExecutor(
            max(1, (os.cpu_count() or 1) // THREADS_PER_CALL)
        )
        config = json.loads((folder / "tagger.json").read_text(encoding="utf-8"))
        self.chunk_chars = int(config["chunk_chars"])
        self.thresholds = {
            label: float(cut) for label, cut in config["thresholds"].items()
        }
        self.tokenizer = Tokenizer.from_file(str(folder / "tokenizer.json"))
        self.tokenizer.no_padding()
        self.tokenizer.enable_truncation(max_length=int(config["max_tokens"]))
        self.labels = json.loads((folder / "labels.json").read_text(encoding="utf-8"))
        self.tables = bioes_tables(self.labels)
        self.outside = self.labels.index("O")
        self.persons = read_names(folder / "persons.txt.gz")
        self.places = read_names(folder / "places.txt.gz")

    def predict(self, text: str) -> list[dict[str, object]]:
        if not text:
            return []
        tagged = apply_chain(
            self.tag(text), text, self.persons, self.places, self.thresholds
        )
        model_spans = [
            {
                "start": span["start"],
                "end": span["end"],
                "label": LABEL_MAP[span["label"]],
            }
            for span in tagged
        ]
        rule_spans = [
            span for span in deterministic_spans(text) if span["label"] in RULE_LABELS
        ]
        return merge_spans(text, rule_spans + model_spans)

    def tag(self, text: str) -> list[dict]:
        """The tagger's own spans over the whole text, with its labels and scores."""
        pieces = chunk_text(text, self.chunk_chars)
        encodings = self.tokenizer.encode_batch([chunk for _, chunk in pieces])
        spans: list[dict] = []
        for (offset, _), found in zip(
            pieces, self.calls.map(self.run, encodings), strict=True
        ):
            spans += [
                {
                    **span,
                    "start": span["start"] + offset,
                    "end": span["end"] + offset,
                }
                for span in found
            ]
        return join_touching(spans)

    def run(self, encoding) -> list[dict]:
        """One window's spans, from a session call of that window alone."""
        logits = self.forward.logits(encoding.ids)
        log_probs = logits - np.logaddexp.reduce(logits, axis=-1, keepdims=True)
        return self.decode(encoding, log_probs)

    def decode(self, encoding, log_probs: np.ndarray) -> list[dict]:
        real = [
            index
            for index, (token, (start, end)) in enumerate(
                zip(encoding.tokens, encoding.offsets, strict=True)
            )
            if end > start and token != SPACE_MARK
        ]
        if not real:
            return []
        rows = log_probs[real]
        path = np.array(viterbi(rows, self.tables))
        # An O token neither opens nor closes a span, so only the others need decoding.
        tagged = np.flatnonzero(path != self.outside)
        confidence = np.exp(rows[tagged, path[tagged]]).tolist()
        return decode_spans(
            [self.labels[index] for index in path[tagged]],
            [encoding.offsets[real[index]] for index in tagged],
            confidence,
        )
