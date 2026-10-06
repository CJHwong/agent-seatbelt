"""The bilingual tagger: a 35M-parameter BIOES tagger for English, Simplified and
Traditional Chinese, then the rules it was scored with, then the deterministic secret
and card rules.

The files come from a local folder, `PII_TAGGER_DIR`, until the release is published:
the int8 ONNX graph (fp32 with `PII_TAGGER_FP32=1`), the tokenizer, the label list, the
public persons and places lists, and `tagger.json` with the window sizes and the url and
username cuts.

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
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np  # ty: ignore[unresolved-import]
import onnxruntime  # ty: ignore[unresolved-import]
from tokenizers import Tokenizer  # ty: ignore[unresolved-import]

from pii_rules import deterministic_spans, merge_spans
from pii_tagger_rules import apply_chain

GRAPH_INT8 = "model.int8.onnx"
GRAPH_FP32 = "model.onnx"
REQUIRED_FILES = (
    "tokenizer.json",
    "labels.json",
    "persons.txt.gz",
    "places.txt.gz",
    "tagger.json",
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


def asset_dir() -> Path:
    """The folder named by PII_TAGGER_DIR, with every file the backend reads."""
    configured = os.environ.get("PII_TAGGER_DIR", "")
    if not configured:
        raise RuntimeError(
            "PII_TAGGER_DIR is not set; point it at the exported tagger folder"
        )
    folder = Path(configured).expanduser()
    graph = GRAPH_FP32 if os.environ.get("PII_TAGGER_FP32") == "1" else GRAPH_INT8
    missing = [
        name for name in (graph, *REQUIRED_FILES) if not (folder / name).is_file()
    ]
    if missing:
        raise RuntimeError(f"PII_TAGGER_DIR {folder} lacks {', '.join(missing)}")
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


class TaggerModel:
    """The tagger on onnxruntime's CPU provider, behind the hook's predict(text) interface."""

    device = "cpu"

    def __init__(self, folder: Path) -> None:
        graph = GRAPH_FP32 if os.environ.get("PII_TAGGER_FP32") == "1" else GRAPH_INT8
        options = onnxruntime.SessionOptions()
        options.intra_op_num_threads = THREADS_PER_CALL
        self.session = onnxruntime.InferenceSession(
            str(folder / graph), options, providers=["CPUExecutionProvider"]
        )
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
        ids = np.array([encoding.ids], dtype=np.int64)
        logits = self.session.run(
            ["logits"], {"input_ids": ids, "attention_mask": np.ones_like(ids)}
        )[0][0].astype(np.float32)
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
