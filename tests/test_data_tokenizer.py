from __future__ import annotations

# ruff: noqa: E402, I001

import copy
import json
import shutil
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hello_slm.config import ConfigError, load_experiment_config, validate_parameter_cap
from hello_slm.data import DataValidationError, build_dataset_artifacts, load_and_validate_corpus
from hello_slm.tokenizer import (
    SPECIAL_TOKENS,
    load_tokenizer,
    pretokenize,
    render_chat,
    train_restricted_bpe,
)


ROOT = Path(__file__).resolve().parents[1]


def _smoke_config():
    return load_experiment_config(ROOT / "configs" / "smoke.toml", ROOT)


def _smoke_corpus():
    return load_and_validate_corpus(_smoke_config())


def _smoke_tokenizer():
    config = _smoke_config()
    corpus = load_and_validate_corpus(config)
    return train_restricted_bpe(
        corpus.by_split("train"),
        vocab_size=config.data["tokenizer"]["vocab_size"],
        min_frequency=config.data["tokenizer"]["min_frequency"],
        corpus_manifest_hash=corpus.manifest_hash,
        corpus_fingerprint=corpus.fingerprint,
        tokenizer_config=config.data["tokenizer"],
    )


def _legacy_bpe_vocab_and_merges(
    conversations,
    *,
    vocab_size: int,
    min_frequency: int,
    tokenizer_config: dict,
):
    rendered = [
        (conversation.conversation_id, render_chat(conversation)[0])
        for conversation in sorted(conversations, key=lambda item: item.conversation_id)
    ]
    words = [
        piece for _, text in rendered for piece in pretokenize(text) if piece not in SPECIAL_TOKENS
    ]
    tokenized_words = [tuple(piece) for piece in words]
    vocab = dict(SPECIAL_TOKENS)

    allowed = set(str(tokenizer_config.get("allowed_characters", "")).replace("\\n", "\n"))
    observed = {symbol for word in tokenized_words for symbol in word}
    for symbol in sorted(allowed | observed):
        if symbol not in vocab and len(vocab) < vocab_size:
            vocab[symbol] = len(vocab)

    def apply_merge(
        word: tuple[str, ...], pair: tuple[str, str], merged: str
    ) -> tuple[str, ...]:
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

    merges: list[tuple[str, str]] = []
    while len(vocab) < vocab_size:
        counts: dict[tuple[str, str], int] = {}
        for word in tokenized_words:
            for pair in zip(word, word[1:], strict=False):
                counts[pair] = counts.get(pair, 0) + 1
        candidates = [(pair, count) for pair, count in counts.items() if count >= min_frequency]
        if not candidates:
            break
        pair, _ = min(candidates, key=lambda item: (-item[1], item[0][0], item[0][1]))
        merged = pair[0] + pair[1]
        if merged in vocab:
            break
        vocab[merged] = len(vocab)
        merges.append(pair)
        tokenized_words = [apply_merge(word, pair, merged) for word in tokenized_words]

    return vocab, merges


def _conversation(conversation_id: str, user: str, assistant: str):
    return SimpleNamespace(
        conversation_id=conversation_id,
        messages=[
            {"role": "user", "content": user, "loss": True},
            {"role": "assistant", "content": assistant, "loss": True},
        ],
    )


def test_load_config_validates_schema_and_parameter_cap():
    config = _smoke_config()

    assert config.parameter_count == 123_200
    assert config.artifact_dir == ROOT / "artifacts" / "smoke"
    assert config.manifest_path == ROOT / "examples" / "corpus" / "manifest.json"

    too_large = copy.deepcopy(config.data)
    too_large["model"]["parameter_cap"] = 10
    with pytest.raises(ConfigError, match="violates cap"):
        validate_parameter_cap(too_large)


def test_corpus_validation_normalizes_and_reports_counts():
    corpus = _smoke_corpus()

    assert corpus.report["splits"]["train"]["conversations"] == 4
    assert corpus.report["splits"]["validation"]["conversations"] == 1
    assert corpus.report["splits"]["test"]["conversations"] == 1
    assert corpus.report["near_duplicate_mode"] == "exhaustive"
    assert corpus.report["near_duplicate_similarity_checked"] is True
    assert len(corpus.fingerprint) == 64
    assert (
        sorted(conversation.conversation_id for conversation in corpus.conversations)[0]
        == "hello.test.001"
    )


def test_render_chat_template_and_loss_mask_are_exact():
    conversation = _smoke_corpus().by_split("test")[0]
    rendered, mask = render_chat(conversation)

    assert rendered == (
        "<|bos|><|user|>\n"
        "How do I answer a simple greeting?<|end|>\n"
        "<|assistant|>\n"
        "Reply with a warm hello and offer one useful next step.<|end|><|eos|>"
    )
    assert len(mask) == len(rendered)
    assert sum(mask) == len("Reply with a warm hello and offer one useful next step.") + len(
        "<|end|><|eos|>"
    )


def test_tokenizer_is_deterministic_roundtrips_and_has_fixed_special_ids(tmp_path):
    config = _smoke_config()
    corpus = load_and_validate_corpus(config)
    kwargs = {
        "vocab_size": config.data["tokenizer"]["vocab_size"],
        "min_frequency": config.data["tokenizer"]["min_frequency"],
        "corpus_manifest_hash": corpus.manifest_hash,
        "corpus_fingerprint": corpus.fingerprint,
        "tokenizer_config": config.data["tokenizer"],
    }

    first = train_restricted_bpe(corpus.by_split("train"), **kwargs)
    second = train_restricted_bpe(corpus.by_split("train"), **kwargs)

    assert first.to_artifact() == second.to_artifact()
    assert {
        item["token"]: item["id"] for item in first.to_artifact()["special_tokens"]
    } == SPECIAL_TOKENS
    for conversation in corpus.conversations:
        rendered, _ = render_chat(conversation)
        ids = first.encode(rendered)
        assert SPECIAL_TOKENS["<|unk|>"] not in ids
        assert first.decode(ids) == rendered

    output = tmp_path / "tokenizer.json"
    first.save(output)
    loaded = load_tokenizer(output)
    assert loaded.encode(render_chat(corpus.by_split("validation")[0])[0])


def test_weighted_bpe_matches_legacy_duplicate_word_semantics():
    cases = [
        [
            _conversation("dup.train.003", "add add add 12 12", "12 plus 12 equals 24"),
            _conversation("dup.train.001", "add add add 12 12", "12 plus 12 equals 24"),
            _conversation("dup.train.002", "subtract subtract 9", "9 minus 4 equals 5"),
            _conversation("dup.train.004", "subtract subtract 9", "9 minus 4 equals 5"),
        ],
        [
            _conversation("overlap.train.001", "aaaa aaa aa", "aaaaaa"),
            _conversation("overlap.train.002", "aaaa aaa aa", "aaaaaa"),
            _conversation("overlap.train.003", "abab baba abba", "abababa"),
        ],
        [
            _conversation("collision.train.001", "abc ab c abc", "abcabc"),
            _conversation("collision.train.002", "a bc abc ab", "abc"),
            _conversation("collision.train.003", "bc abc a", "abcbc"),
        ],
    ]
    tokenizer_config = {
        "allowed_characters": "abcdefghijklmnopqrstuvwxyz0123456789 |+-=<>/\n",
    }

    for index, conversations in enumerate(cases):
        kwargs = {
            "vocab_size": 96,
            "min_frequency": 2,
            "corpus_manifest_hash": "a" * 64,
            "corpus_fingerprint": f"{index:064x}",
            "tokenizer_config": tokenizer_config,
        }

        tokenizer = train_restricted_bpe(conversations, **kwargs)
        legacy_vocab, legacy_merges = _legacy_bpe_vocab_and_merges(
            conversations,
            vocab_size=kwargs["vocab_size"],
            min_frequency=kwargs["min_frequency"],
            tokenizer_config=tokenizer_config,
        )

        assert tokenizer.vocab == legacy_vocab
        assert tokenizer.merges == legacy_merges


def test_tokenizer_merge_digits_defaults_to_legacy_behavior():
    conversations = [
        _conversation("digits.train.001", "12 12 12", "12 12"),
        _conversation("digits.train.002", "12 12 12", "12 12"),
    ]
    tokenizer_config = {
        "allowed_characters": "0123456789 \n",
    }
    tokenizer = train_restricted_bpe(
        conversations,
        vocab_size=48,
        min_frequency=2,
        corpus_manifest_hash="a" * 64,
        corpus_fingerprint="b" * 64,
        tokenizer_config=tokenizer_config,
    )
    legacy_vocab, legacy_merges = _legacy_bpe_vocab_and_merges(
        conversations,
        vocab_size=48,
        min_frequency=2,
        tokenizer_config=tokenizer_config,
    )

    assert tokenizer.vocab == legacy_vocab
    assert tokenizer.merges == legacy_merges
    assert ("1", "2") in tokenizer.merges
    assert "12" in tokenizer.vocab
    assert tokenizer.to_artifact()["merge_digits"] is True


def test_tokenizer_can_keep_ascii_digits_atomic():
    conversations = [
        _conversation("math.train.001", "12 + 12", "12 + 12 = 24"),
        _conversation("math.train.002", "12 + 12", "12 + 12 = 24"),
        _conversation("math.train.003", "34 + 5", "34 + 5 = 39"),
    ]
    tokenizer = train_restricted_bpe(
        conversations,
        vocab_size=96,
        min_frequency=2,
        corpus_manifest_hash="a" * 64,
        corpus_fingerprint="c" * 64,
        tokenizer_config={
            "allowed_characters": "abcdefghijklmnopqrstuvwxyz0123456789 +=\n",
            "merge_digits": False,
        },
    )

    digit_tokens = [
        token
        for token in tokenizer.vocab
        if token not in SPECIAL_TOKENS and any(char.isascii() and char.isdigit() for char in token)
    ]
    assert sorted(digit_tokens) == list("0123456789")
    assert all(
        not any(char.isascii() and char.isdigit() for char in left + right)
        for left, right in tokenizer.merges
    )
    assert tokenizer.to_artifact()["merge_digits"] is False

    rendered, _ = render_chat(conversations[0])
    spans = tokenizer.encode_with_spans(rendered)
    assert tokenizer.decode([span.token_id for span in spans]) == rendered
    assert [span.text for span in spans if span.text in {"1", "2"}].count("1") >= 4


def test_build_dataset_writes_split_tensors_with_assistant_only_labels(tmp_path):
    config = _smoke_config()
    data = copy.deepcopy(config.data)
    data["run"]["artifact_dir"] = str(tmp_path / "artifacts")
    config = replace(config, data=data)
    corpus = load_and_validate_corpus(config)
    tokenizer = train_restricted_bpe(
        corpus.by_split("train"),
        vocab_size=config.data["tokenizer"]["vocab_size"],
        min_frequency=config.data["tokenizer"]["min_frequency"],
        corpus_manifest_hash=corpus.manifest_hash,
        corpus_fingerprint=corpus.fingerprint,
        tokenizer_config=config.data["tokenizer"],
    )

    manifest = build_dataset_artifacts(config, corpus, tokenizer)

    assert set(manifest["splits"]) == {"train", "validation", "test"}
    train = torch.load(tmp_path / "artifacts" / "dataset" / "train.pt", weights_only=False)
    assert train["input_ids"].shape == (4, 128)
    assert train["labels"].shape == (4, 128)
    assert (train["labels"] != -100).sum().item() > 0
    assert (train["labels"][train["attention_mask"] == 0] == -100).all()
    for label_id in train["labels"][train["labels"] != -100].tolist():
        assert label_id not in {
            SPECIAL_TOKENS["<|pad|>"],
            SPECIAL_TOKENS["<|bos|>"],
            SPECIAL_TOKENS["<|system|>"],
            SPECIAL_TOKENS["<|user|>"],
            SPECIAL_TOKENS["<|assistant|>"],
        }

    for split in ("validation", "test"):
        payload = torch.load(
            tmp_path / "artifacts" / "dataset" / f"{split}.pt", weights_only=True
        )
        assert all(record["tokens"] == record["kept_tokens"] for record in payload["records"])
    label_values = train["labels"].flatten().tolist()
    assert SPECIAL_TOKENS["<|end|>"] in label_values
    assert SPECIAL_TOKENS["<|eos|>"] in label_values


def test_invalid_manifest_file_hash_is_rejected(tmp_path):
    manifest = json.loads((ROOT / "examples" / "corpus" / "manifest.json").read_text())
    manifest["entries"][0]["sha256"] = "0" * 64
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    config = _smoke_config()
    data = copy.deepcopy(config.data)
    data["corpus"]["manifest_path"] = str(manifest_path)
    config = replace(config, data=data)

    with pytest.raises(DataValidationError, match="sha256 mismatch"):
        load_and_validate_corpus(config)


def test_focused_profile_enforces_deterministic_conversation_splits():
    config = load_experiment_config(ROOT / "configs" / "focused-125m.toml", ROOT)

    with pytest.raises(DataValidationError, match="does not match deterministic split"):
        load_and_validate_corpus(config)


def test_disallowed_character_and_duplicate_targets_are_rejected(tmp_path):
    root = tmp_path / "repo"
    shutil.copytree(ROOT / "schemas", root / "schemas")
    source = root / "train.jsonl"
    rows = [
        {
            "schema_version": 1,
            "conversation_id": "bad.train.001",
            "source": "synthetic",
            "license": "MIT",
            "created_at": "2026-07-22T00:00:00Z",
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Same target"},
            ],
        },
        {
            "schema_version": 1,
            "conversation_id": "bad.train.002",
            "source": "synthetic",
            "license": "MIT",
            "created_at": "2026-07-22T00:00:00Z",
            "messages": [
                {"role": "user", "content": "This has @ invalid"},
                {"role": "assistant", "content": "Same target"},
            ],
        },
    ]
    source.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    manifest = {
        "format_version": 1,
        "name": "bad",
        "created_at": "2026-07-22T00:00:00Z",
        "entries": [
            {
                "path": "train.jsonl",
                "split": "train",
                "sha256": "placeholder",
                "bytes": source.stat().st_size,
                "conversation_count": 2,
                "provenance": "Self-authored synthetic conversations.",
                "license": "MIT",
                "rights_holder": "tests",
                "consent": {
                    "training_allowed": True,
                    "granted_by": "tests",
                    "grant_reference": "tests",
                },
                "allowed_use": ["tokenizer-training", "model-training"],
                "contains_personal_data": False,
                "contains_synthetic_data": True,
                "included": True,
            }
        ],
    }
    import hashlib

    manifest["entries"][0]["sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    config = _smoke_config()
    data = copy.deepcopy(config.data)
    data["corpus"]["manifest_path"] = str(manifest_path)
    config = replace(config, root=root, data=data)

    with pytest.raises(DataValidationError, match="disallowed characters|duplicate"):
        load_and_validate_corpus(config)
