from __future__ import annotations

import hashlib
import heapq
import json
import platform
import re
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hello_slm.config import canonical_json_bytes, canonical_sha256

SPECIAL_TOKENS: dict[str, int] = {
    "<|pad|>": 0,
    "<|unk|>": 1,
    "<|bos|>": 2,
    "<|eos|>": 3,
    "<|system|>": 4,
    "<|user|>": 5,
    "<|assistant|>": 6,
    "<|end|>": 7,
}
ID_TO_SPECIAL = {value: key for key, value in SPECIAL_TOKENS.items()}
SPECIAL_PATTERN = re.compile("(" + "|".join(re.escape(token) for token in SPECIAL_TOKENS) + ")")


class TokenizerError(ValueError):
    """Raised when tokenizer training or encoding violates the restricted contract."""


@dataclass(frozen=True)
class TokenSpan:
    token_id: int
    text: str
    start: int
    end: int
    is_special: bool = False


def _split_ordinary(text: str) -> list[str]:
    pieces: list[str] = []
    index = 0
    while index < len(text):
        char = text[index]
        category = unicodedata.category(char)
        if char == "\n":
            pieces.append(char)
            index += 1
        elif char == " ":
            start = index
            while index < len(text) and text[index] == " ":
                index += 1
            pieces.append(text[start:index])
        elif category[0] in {"L", "N"} or category in {"Mn", "Mc", "Me", "Pc"}:
            start = index
            while index < len(text):
                ccat = unicodedata.category(text[index])
                if ccat[0] in {"L", "N"} or ccat in {"Mn", "Mc", "Me", "Pc"}:
                    index += 1
                else:
                    break
            pieces.append(text[start:index])
        else:
            pieces.append(char)
            index += 1
    return pieces


def pretokenize(text: str) -> list[str]:
    pieces: list[str] = []
    for part in SPECIAL_PATTERN.split(text):
        if not part:
            continue
        if part in SPECIAL_TOKENS:
            pieces.append(part)
        else:
            pieces.extend(_split_ordinary(part))
    return pieces


@dataclass
class RestrictedBPETokenizer:
    vocab: dict[str, int]
    merges: list[tuple[str, str]]
    artifact: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if {token: self.vocab.get(token) for token in SPECIAL_TOKENS} != SPECIAL_TOKENS:
            raise TokenizerError("special token IDs do not match the immutable table")
        self.id_to_token = {idx: token for token, idx in self.vocab.items()}
        self.merge_ranks = {pair: rank for rank, pair in enumerate(self.merges)}

    def _encode_piece_tokens(self, piece: str) -> list[str]:
        if piece in SPECIAL_TOKENS:
            return [piece]
        symbols = list(piece)
        while len(symbols) > 1:
            ranked_pairs = [
                (self.merge_ranks[(symbols[i], symbols[i + 1])], i)
                for i in range(len(symbols) - 1)
                if (symbols[i], symbols[i + 1]) in self.merge_ranks
            ]
            if not ranked_pairs:
                break
            _, index = min(ranked_pairs)
            symbols = symbols[:index] + [symbols[index] + symbols[index + 1]] + symbols[index + 2 :]
        return symbols

    def encode_with_spans(self, text: str, *, allow_unk: bool = False) -> list[TokenSpan]:
        spans: list[TokenSpan] = []
        cursor = 0
        for piece in pretokenize(text):
            start = text.find(piece, cursor)
            if start < 0:
                raise TokenizerError("internal pretokenizer span mismatch")
            local = start
            for token in self._encode_piece_tokens(piece):
                token_id = self.vocab.get(token)
                if token_id is None:
                    if not allow_unk:
                        raise TokenizerError(f"unknown token for text span {token!r}")
                    token_id = SPECIAL_TOKENS["<|unk|>"]
                end = local + len(token)
                spans.append(TokenSpan(token_id, token, local, end, token in SPECIAL_TOKENS))
                local = end
            cursor = start + len(piece)
        return spans

    def encode(self, text: str, *, allow_unk: bool = False) -> list[int]:
        return [span.token_id for span in self.encode_with_spans(text, allow_unk=allow_unk)]

    def decode(self, ids: list[int], *, skip_special_tokens: bool = False) -> str:
        tokens = []
        for token_id in ids:
            token = self.id_to_token.get(int(token_id), "<|unk|>")
            if skip_special_tokens and token in SPECIAL_TOKENS:
                continue
            tokens.append(token)
        return "".join(tokens)

    def to_artifact(self) -> dict[str, Any]:
        if self.artifact is None:
            raise TokenizerError("tokenizer has no artifact metadata")
        return self.artifact

    def save(self, path: str | Path) -> None:
        artifact = self.to_artifact()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_bytes(canonical_json_bytes(artifact) + b"\n")
        tmp.replace(path)


def load_tokenizer(path: str | Path) -> RestrictedBPETokenizer:
    artifact = json.loads(Path(path).read_text(encoding="utf-8"))
    vocab = {item["token"]: int(item["id"]) for item in artifact["vocabulary"]}
    merges = [tuple(item) for item in artifact["merges"]]
    expected = artifact.get("tokenizer_fingerprint")
    actual = _fingerprint_for_artifact(artifact)
    if expected != actual:
        raise TokenizerError("tokenizer_fingerprint mismatch")
    return RestrictedBPETokenizer(vocab=vocab, merges=merges, artifact=artifact)


def render_chat(conversation: Any) -> tuple[str, list[bool]]:
    messages = [
        m if isinstance(m, dict) else {"role": m.role, "content": m.content, "loss": m.loss}
        for m in conversation.messages
    ]
    text_parts: list[str] = []
    loss_mask: list[bool] = []

    def append(part: str, loss: bool) -> None:
        text_parts.append(part)
        loss_mask.extend([loss] * len(part))

    append("<|bos|>", False)
    final_assistant_loss = False
    for message in messages:
        role = message["role"]
        content = message["content"]
        content_loss = role == "assistant" and message.get("loss", True)
        final_assistant_loss = content_loss if role == "assistant" else final_assistant_loss
        append(f"<|{role}|>\n", False)
        append(content, content_loss)
        append("<|end|>", content_loss)
        append("\n", False)

    if text_parts[-1] == "\n":
        text_parts.pop()
        loss_mask.pop()
    append("<|eos|>", final_assistant_loss)
    return "".join(text_parts), loss_mask


def train_restricted_bpe(
    conversations: list[Any],
    *,
    vocab_size: int,
    min_frequency: int,
    corpus_manifest_hash: str,
    corpus_fingerprint: str,
    tokenizer_config: dict[str, Any],
) -> RestrictedBPETokenizer:
    if vocab_size < len(SPECIAL_TOKENS):
        raise TokenizerError("vocab_size cannot fit immutable special tokens")
    if vocab_size >= 50_000:
        raise TokenizerError("vocab_size must be less than 50000")

    rendered = [
        (conversation.conversation_id, render_chat(conversation)[0])
        for conversation in sorted(conversations, key=lambda item: item.conversation_id)
    ]
    words = [
        piece for _, text in rendered for piece in pretokenize(text) if piece not in SPECIAL_TOKENS
    ]
    tokenized_words = Counter(tuple(piece) for piece in words)
    vocab = dict(SPECIAL_TOKENS)

    allowed = set(str(tokenizer_config.get("allowed_characters", "")).replace("\\n", "\n"))
    observed = {symbol for word in tokenized_words for symbol in word}
    for symbol in sorted(allowed | observed):
        if symbol not in vocab and len(vocab) < vocab_size:
            vocab[symbol] = len(vocab)

    merges: list[tuple[str, str]] = []
    merge_digits = bool(tokenizer_config.get("merge_digits", True))
    bpe_state = _IncrementalBPEState(tokenized_words)
    while len(vocab) < vocab_size:
        pair = bpe_state.best_pair(
            min_frequency,
            allow_pair=None if merge_digits else _pair_keeps_digits_atomic,
        )
        if pair is None:
            break
        merged = pair[0] + pair[1]
        if merged in vocab:
            break
        vocab[merged] = len(vocab)
        merges.append(pair)
        bpe_state.apply_merge(pair, merged)

    rendered_text = "\n".join(text for _, text in rendered)
    base_artifact: dict[str, Any] = {
        "format_version": 1,
        "algorithm": "restricted_bpe",
        "implementation_version": 1,
        "special_tokens": [{"token": token, "id": idx} for token, idx in SPECIAL_TOKENS.items()],
        "normalizer": "data-spec-nfc",
        "pre_tokenizer": "specials-word-space-newline-punct",
        "vocabulary": [
            {"token": token, "id": idx}
            for token, idx in sorted(vocab.items(), key=lambda item: item[1])
        ],
        "merges": [list(pair) for pair in merges],
        "source_corpus_manifest_hash": corpus_manifest_hash,
        "canonical_corpus_fingerprint": corpus_fingerprint,
        "tokenizer_configuration_hash": canonical_sha256(tokenizer_config),
        "merge_digits": merge_digits,
        "rendered_training_text_hash": hashlib.sha256(rendered_text.encode("utf-8")).hexdigest(),
        "dependency_versions": {"python": platform.python_version()},
    }
    base_artifact["tokenizer_fingerprint"] = _fingerprint_for_artifact(base_artifact)
    return RestrictedBPETokenizer(vocab=vocab, merges=merges, artifact=base_artifact)


def _apply_merge(word: tuple[str, ...], pair: tuple[str, str], merged: str) -> tuple[str, ...]:
    output: list[str] = []
    index = 0
    while index < len(word):
        if index + 1 < len(word) and (word[index], word[index + 1]) == pair:
            output.append(merged)
            index += 2
        else:
            output.append(word[index])
            index += 1
    return tuple(output)


class _IncrementalBPEState:
    def __init__(self, weighted_words: Counter[tuple[str, ...]]) -> None:
        self.words = list(weighted_words)
        self.weights = [weighted_words[word] for word in self.words]
        self.pair_counts: Counter[tuple[str, str]] = Counter()
        self.pair_to_word_ids: defaultdict[tuple[str, str], set[int]] = defaultdict(set)
        self.heap: list[tuple[int, str, str, tuple[str, str]]] = []

        for word_id, word in enumerate(self.words):
            self._add_word_pairs(word_id, word)

    def best_pair(
        self,
        min_frequency: int,
        *,
        allow_pair: Callable[[tuple[str, str]], bool] | None = None,
    ) -> tuple[str, str] | None:
        while self.heap:
            neg_count, left, right, pair = heapq.heappop(self.heap)
            count = -neg_count
            current_count = self.pair_counts.get(pair, 0)
            if pair == (left, right) and current_count == count:
                if count < min_frequency:
                    return None
                if allow_pair is not None and not allow_pair(pair):
                    continue
                return pair
        return None

    def apply_merge(self, pair: tuple[str, str], merged: str) -> None:
        affected_word_ids = list(self.pair_to_word_ids.get(pair, ()))
        for word_id in affected_word_ids:
            old_word = self.words[word_id]
            new_word = _apply_merge(old_word, pair, merged)
            if new_word == old_word:
                continue
            self._remove_word_pairs(word_id, old_word)
            self.words[word_id] = new_word
            self._add_word_pairs(word_id, new_word)

    def _add_word_pairs(self, word_id: int, word: tuple[str, ...]) -> None:
        weight = self.weights[word_id]
        for pair, occurrences in Counter(zip(word, word[1:], strict=False)).items():
            self.pair_counts[pair] += occurrences * weight
            self.pair_to_word_ids[pair].add(word_id)
            self._push_pair(pair)

    def _remove_word_pairs(self, word_id: int, word: tuple[str, ...]) -> None:
        weight = self.weights[word_id]
        for pair, occurrences in Counter(zip(word, word[1:], strict=False)).items():
            self.pair_counts[pair] -= occurrences * weight
            if self.pair_counts[pair] <= 0:
                del self.pair_counts[pair]
            self.pair_to_word_ids[pair].discard(word_id)
            if not self.pair_to_word_ids[pair]:
                del self.pair_to_word_ids[pair]
            self._push_pair(pair)

    def _push_pair(self, pair: tuple[str, str]) -> None:
        count = self.pair_counts.get(pair, 0)
        if count > 0:
            heapq.heappush(self.heap, (-count, pair[0], pair[1], pair))


def _fingerprint_for_artifact(artifact: dict[str, Any]) -> str:
    behavior = {key: value for key, value in artifact.items() if key != "tokenizer_fingerprint"}
    return canonical_sha256(behavior)


def _pair_keeps_digits_atomic(pair: tuple[str, str]) -> bool:
    return not any(char.isascii() and char.isdigit() for char in pair[0] + pair[1])
