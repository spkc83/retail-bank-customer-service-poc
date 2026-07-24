from __future__ import annotations

import hashlib
import json
import re
import tomllib
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from hello_slm.config import (
    ConfigError,
    ExperimentConfig,
    canonical_json_bytes,
    canonical_sha256,
    file_sha256,
    load_json,
    validate_json_schema,
)
from hello_slm.tokenizer import SPECIAL_TOKENS, RestrictedBPETokenizer, render_chat

SPLITS = ("train", "validation", "test")


class DataValidationError(ValueError):
    """Raised when corpus validation fails."""


@dataclass(frozen=True)
class Message:
    role: str
    content: str
    loss: bool


@dataclass(frozen=True)
class Conversation:
    schema_version: int
    conversation_id: str
    source: str
    license: str
    created_at: str
    messages: tuple[Message, ...]
    split: str
    manifest_path: str
    metadata: dict[str, Any] | None = None

    def canonical(self) -> dict[str, Any]:
        value: dict[str, Any] = {
            "schema_version": self.schema_version,
            "conversation_id": self.conversation_id,
            "source": self.source,
            "license": self.license,
            "created_at": self.created_at,
            "messages": [
                {"role": message.role, "content": message.content, "loss": message.loss}
                for message in self.messages
            ],
            "split": self.split,
        }
        if self.metadata:
            value["metadata"] = self.metadata
        return value


@dataclass(frozen=True)
class Corpus:
    manifest: dict[str, Any]
    manifest_hash: str
    conversations: tuple[Conversation, ...]
    fingerprint: str
    report: dict[str, Any]

    def by_split(self, split: str) -> list[Conversation]:
        return [conversation for conversation in self.conversations if conversation.split == split]


def load_and_validate_corpus(config: ExperimentConfig) -> Corpus:
    manifest_path = config.manifest_path
    manifest = load_json(manifest_path)
    validate_json_schema(manifest, config.root / "schemas" / "corpus-manifest.schema.json")
    conversation_schema_path = config.root / "schemas" / "conversation.schema.json"
    manifest_hash = file_sha256(manifest_path)
    corpus_policy = _load_corpus_policy(config)

    conversations: list[Conversation] = []
    errors: list[str] = []
    for entry in manifest["entries"]:
        if not entry["included"]:
            continue
        try:
            _validate_manifest_entry_policy(entry, corpus_policy)
            file_path = config.resolve_path(entry["path"])
            _validate_declared_file(file_path, entry)
            conversations.extend(
                _read_jsonl_file(file_path, entry, conversation_schema_path, config.data)
            )
        except (OSError, ValueError, ConfigError) as exc:
            errors.append(f"{entry['path']}: {exc}")

    rejected_inputs = _find_undeclared_inputs(config, manifest)
    near_duplicate_mode = str(config.data["corpus"]["near_duplicate_mode"])
    errors.extend(_validate_corpus_level(conversations, near_duplicate_mode))
    errors.extend(_validate_split_assignments(conversations, config.data))
    if errors:
        raise DataValidationError("; ".join(errors))

    canonical = [
        conversation.canonical()
        for conversation in sorted(conversations, key=lambda item: item.conversation_id)
    ]
    fingerprint = canonical_sha256(canonical)
    report = _build_report(
        conversations, fingerprint, rejected_inputs, near_duplicate_mode
    )
    return Corpus(
        manifest=manifest,
        manifest_hash=manifest_hash,
        conversations=tuple(conversations),
        fingerprint=fingerprint,
        report=report,
    )


def conversations_for_tokenizer(corpus: Corpus) -> list[Conversation]:
    allowed_paths = {
        entry["path"]
        for entry in corpus.manifest["entries"]
        if entry["included"] and "tokenizer-training" in entry["allowed_use"]
    }
    return [
        conversation
        for conversation in corpus.by_split("train")
        if conversation.manifest_path in allowed_paths
    ]


def build_dataset_artifacts(
    config: ExperimentConfig,
    corpus: Corpus,
    tokenizer: RestrictedBPETokenizer,
) -> dict[str, Any]:
    dataset_dir = config.artifact_dir / "dataset"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    max_seq_len = int(config.data["dataset"]["max_seq_len"])
    drop_remainder = bool(config.data["dataset"].get("drop_remainder", False))
    manifest: dict[str, Any] = {
        "format_version": 1,
        "max_seq_len": max_seq_len,
        "corpus_fingerprint": corpus.fingerprint,
        "tokenizer_fingerprint": tokenizer.to_artifact()["tokenizer_fingerprint"],
        "splits": {},
    }

    for split in SPLITS:
        tensors, records = _pack_split(
            corpus.by_split(split), tokenizer, max_seq_len, drop_remainder
        )
        output = dataset_dir / f"{split}.pt"
        payload = {"format_version": 1, "split": split, "records": records, **tensors}
        tmp = output.with_suffix(output.suffix + ".tmp")
        torch.save(payload, tmp)
        tmp.replace(output)
        manifest["splits"][split] = {
            "path": str(output),
            "examples": int(tensors["input_ids"].shape[0]),
            "loss_tokens": int((tensors["labels"] != -100).sum().item()),
            "records": len(records),
            "sha256": file_sha256(output),
        }

    manifest["dataset_fingerprint"] = canonical_sha256(manifest["splits"])
    output_manifest = dataset_dir / "manifest.json"
    output_manifest.write_bytes(canonical_json_bytes(manifest) + b"\n")
    return manifest


def _load_corpus_policy(config: ExperimentConfig) -> dict[str, Any]:
    path = config.root / "configs" / "corpus.toml"
    if not path.exists():
        return {}
    with path.open("rb") as handle:
        return tomllib.load(handle)


def _validate_manifest_entry_policy(entry: dict[str, Any], policy: dict[str, Any]) -> None:
    if not entry["consent"]["training_allowed"]:
        raise DataValidationError("training consent is not granted")
    if entry["contains_personal_data"]:
        raise DataValidationError("personal data is not allowed")
    if not entry["license"]:
        raise DataValidationError("license is required")
    allowed_licenses = policy.get("license_policy", {}).get("allowed_licenses", [])
    if allowed_licenses and entry["license"] not in allowed_licenses:
        raise DataValidationError(f"license {entry['license']!r} is not in the allowlist")
    required_use = {"train": "model-training", "validation": "validation", "test": "test"}[
        entry["split"]
    ]
    if required_use not in entry["allowed_use"]:
        raise DataValidationError(f"allowed_use must include {required_use!r}")
    source_text = " ".join(
        [entry["provenance"], entry["license"], entry["rights_holder"]]
    ).casefold()
    blocked = ["scraped", "confidential", "private", "leaked"]
    if any(word in source_text for word in blocked):
        raise DataValidationError("blocked provenance marker present")


def _validate_declared_file(path: Path, entry: dict[str, Any]) -> None:
    raw = path.read_bytes()
    if raw.startswith(b"\xef\xbb\xbf"):
        raise DataValidationError("UTF-8 BOM is not allowed")
    if raw and not raw.endswith(b"\n"):
        raise DataValidationError("JSONL files must end with a newline")
    if len(raw) != int(entry["bytes"]):
        raise DataValidationError(f"byte count mismatch: expected {entry['bytes']}, got {len(raw)}")
    actual_hash = file_sha256(path)
    if actual_hash != entry["sha256"]:
        raise DataValidationError("sha256 mismatch")
    if len(raw.splitlines()) != int(entry["conversation_count"]):
        raise DataValidationError("conversation_count mismatch")


def _read_jsonl_file(
    path: Path,
    entry: dict[str, Any],
    conversation_schema_path: Path,
    config: dict[str, Any],
) -> list[Conversation]:
    conversations: list[Conversation] = []
    allowed_chars = _allowed_characters(config)
    with path.open("r", encoding="utf-8", newline="") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.endswith("\n"):
                raise DataValidationError(f"line {line_number} does not end with newline")
            if line == "\n":
                raise DataValidationError(f"line {line_number} is blank")
            record = json.loads(line)
            validate_json_schema(record, conversation_schema_path)
            conversations.append(_normalize_record(record, entry, line_number, allowed_chars))
    return conversations


def _normalize_record(
    record: dict[str, Any],
    entry: dict[str, Any],
    line_number: int,
    allowed_chars: set[str],
) -> Conversation:
    if record["license"] != entry["license"]:
        raise DataValidationError(f"line {line_number} license does not match manifest")
    messages: list[Message] = []
    for message in record["messages"]:
        content = normalize_text(message["content"])
        if not content:
            raise DataValidationError(f"line {line_number} message normalizes to empty")
        invalid = sorted(
            {char for char in content if char not in allowed_chars or not _category_allowed(char)}
        )
        if invalid:
            raise DataValidationError(
                f"line {line_number} contains disallowed characters {invalid!r}"
            )
        role = message["role"]
        loss = bool(message.get("loss", role == "assistant"))
        messages.append(Message(role=role, content=content, loss=loss))
    _validate_roles(messages, line_number)
    return Conversation(
        schema_version=record["schema_version"],
        conversation_id=record["conversation_id"],
        source=record["source"],
        license=record["license"],
        created_at=record["created_at"],
        messages=tuple(messages),
        split=entry["split"],
        manifest_path=entry["path"],
        metadata=record.get("metadata"),
    )


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\t", " ")
    lines = [line.rstrip(" ") for line in text.split("\n")]
    text = "\n".join(lines)
    return re.sub(r"\n{3,}", "\n\n", text)


def _validate_roles(messages: list[Message], line_number: int) -> None:
    roles = [message.role for message in messages]
    if roles.count("system") > 1 or ("system" in roles and roles[0] != "system"):
        raise DataValidationError(
            f"line {line_number} system role must appear at most once and first"
        )
    expected = ["user", "assistant"]
    offset = 1 if roles and roles[0] == "system" else 0
    for index, role in enumerate(roles[offset:]):
        if role != expected[index % 2]:
            raise DataValidationError(f"line {line_number} roles must alternate user/assistant")
    if roles[-1] != "assistant" or "user" not in roles or "assistant" not in roles:
        raise DataValidationError(f"line {line_number} must include user and final assistant")
    for message in messages:
        if message.role != "assistant" and message.loss:
            raise DataValidationError(
                f"line {line_number} only assistant messages may set loss=true"
            )


def _validate_corpus_level(
    conversations: list[Conversation], near_duplicate_mode: str
) -> list[str]:
    errors: list[str] = []
    ids = [conversation.conversation_id for conversation in conversations]
    if len(set(ids)) != len(ids):
        errors.append("conversation_id values must be unique")

    seen_messages: dict[str, str] = {}
    seen_targets: dict[str, str] = {}
    near_keys: list[tuple[str, str, set[str]]] = []
    for conversation in conversations:
        messages_key = canonical_sha256(
            [{"role": msg.role, "content": msg.content} for msg in conversation.messages]
        )
        if messages_key in seen_messages:
            errors.append(f"exact duplicate conversation: {conversation.conversation_id}")
        seen_messages[messages_key] = conversation.conversation_id
        for message in conversation.messages:
            if message.role == "assistant":
                target_key = canonical_sha256(message.content)
                if target_key in seen_targets:
                    errors.append(
                        f"exact duplicate assistant target: {conversation.conversation_id}"
                    )
                seen_targets[target_key] = conversation.conversation_id
        if near_duplicate_mode == "exhaustive":
            grams = _trigram_set(
                " ".join(message.content for message in conversation.messages)
            )
            near_keys.append((conversation.conversation_id, conversation.split, grams))

    for index, (left_id, left_split, left_grams) in enumerate(near_keys):
        for right_id, right_split, right_grams in near_keys[index + 1 :]:
            similarity = _jaccard(left_grams, right_grams)
            if similarity >= 0.92:
                if left_split != right_split:
                    errors.append(f"cross-split near duplicate: {left_id} and {right_id}")
                else:
                    errors.append(f"near duplicate: {left_id} and {right_id}")
    errors.extend(_validate_train_heldout_contamination(conversations))
    return errors


def _validate_train_heldout_contamination(conversations: list[Conversation]) -> list[str]:
    errors: list[str] = []
    train = [conversation for conversation in conversations if conversation.split == "train"]
    heldout = [
        conversation
        for conversation in conversations
        if conversation.split in {"validation", "test"}
    ]
    train_messages = {
        canonical_sha256(
            [{"role": msg.role, "content": msg.content} for msg in conversation.messages]
        )
        for conversation in train
    }
    train_targets = {
        canonical_sha256(message.content)
        for conversation in train
        for message in conversation.messages
        if message.role == "assistant"
    }
    for conversation in heldout:
        message_key = canonical_sha256(
            [{"role": msg.role, "content": msg.content} for msg in conversation.messages]
        )
        if message_key in train_messages:
            errors.append(f"train-to-{conversation.split} exact conversation contamination")
        for message in conversation.messages:
            if message.role == "assistant" and canonical_sha256(message.content) in train_targets:
                errors.append(f"train-to-{conversation.split} assistant target contamination")
    return errors


def _validate_split_assignments(
    conversations: list[Conversation], config: dict[str, Any]
) -> list[str]:
    if bool(config["corpus"]["enforce_manifest_split_only"]):
        return []
    ratios = config["corpus"]["split"]
    seed = str(config["seeds"]["corpus_split"])
    train_boundary = float(ratios["train"])
    validation_boundary = train_boundary + float(ratios["validation"])
    errors: list[str] = []
    for conversation in conversations:
        digest = hashlib.sha256(
            f"{seed}\n{conversation.conversation_id}".encode()
        ).digest()
        bucket = int.from_bytes(digest[:8], "big") / 2**64
        expected = (
            "train"
            if bucket < train_boundary
            else "validation"
            if bucket < validation_boundary
            else "test"
        )
        if conversation.split != expected:
            errors.append(
                f"{conversation.conversation_id} manifest split {conversation.split!r} "
                f"does not match deterministic split {expected!r}"
            )
    return errors


def _pack_split(
    conversations: list[Conversation],
    tokenizer: RestrictedBPETokenizer,
    max_seq_len: int,
    drop_remainder: bool,
) -> tuple[dict[str, torch.Tensor], list[dict[str, Any]]]:
    rows: list[tuple[list[int], list[int], list[int]]] = []
    records: list[dict[str, Any]] = []
    for conversation in sorted(conversations, key=lambda item: item.conversation_id):
        text, char_loss_mask = render_chat(conversation)
        spans = tokenizer.encode_with_spans(text, allow_unk=False)
        if any(span.token_id == SPECIAL_TOKENS["<|unk|>"] for span in spans):
            raise DataValidationError(f"{conversation.conversation_id} encoded to <|unk|>")
        ids = [span.token_id for span in spans]
        token_loss = [any(char_loss_mask[span.start : span.end]) for span in spans]
        if len(ids) > max_seq_len:
            raise DataValidationError(
                f"{conversation.conversation_id} has {len(ids)} tokens and exceeds "
                f"max_seq_len={max_seq_len}; partial conversations are forbidden"
            )
        original_tokens = len(ids)
        input_ids = ids.copy()
        next_loss = token_loss[1:] + [False]
        labels = [
            input_ids[index + 1] if index + 1 < len(input_ids) and next_loss[index] else -100
            for index in range(len(input_ids))
        ]
        attention = [1] * len(input_ids)
        pad = max_seq_len - len(input_ids)
        input_ids.extend([SPECIAL_TOKENS["<|pad|>"]] * pad)
        labels.extend([-100] * pad)
        attention.extend([0] * pad)
        rows.append((input_ids, labels, attention))
        records.append(
            {
                "conversation_id": conversation.conversation_id,
                "split": conversation.split,
                "tokens": original_tokens,
                "kept_tokens": len(input_ids) - pad,
                "loss_tokens": sum(label != -100 for label in labels),
            }
        )
    if not rows:
        empty = torch.empty((0, max_seq_len), dtype=torch.long)
        return {
            "input_ids": empty,
            "labels": empty.clone(),
            "attention_mask": empty.clone(),
        }, records
    return (
        {
            "input_ids": torch.tensor([row[0] for row in rows], dtype=torch.long),
            "labels": torch.tensor([row[1] for row in rows], dtype=torch.long),
            "attention_mask": torch.tensor([row[2] for row in rows], dtype=torch.long),
        },
        records,
    )


def _allowed_characters(config: dict[str, Any]) -> set[str]:
    allowed = config["tokenizer"].get("allowed_characters", "")
    return set(allowed.replace("\\n", "\n"))


def _category_allowed(char: str) -> bool:
    if char == "\n":
        return True
    category = unicodedata.category(char)
    if category == "Cc" or category in {"Cf", "Co", "Cs", "Cn"}:
        return False
    if unicodedata.bidirectional(char) in {
        "RLO",
        "LRO",
        "RLE",
        "LRE",
        "PDF",
        "RLI",
        "LRI",
        "FSI",
        "PDI",
    }:
        return False
    return category in {
        "Lu",
        "Ll",
        "Lt",
        "Lm",
        "Lo",
        "Nd",
        "Mn",
        "Mc",
        "Me",
        "Pc",
        "Pd",
        "Po",
        "Ps",
        "Pe",
        "Pi",
        "Pf",
        "Sc",
        "Sk",
        "Sm",
        "Zs",
    }


def _trigram_set(text: str) -> set[str]:
    folded = " ".join(text.casefold().split())
    if len(folded) < 3:
        return {folded} if folded else set()
    return {folded[index : index + 3] for index in range(len(folded) - 2)}


def _ngram_set(text: str, n: int) -> set[tuple[str, ...]]:
    words = " ".join(text.casefold().split()).split()
    if len(words) < n:
        return set()
    return {tuple(words[index : index + n]) for index in range(len(words) - n + 1)}


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _find_undeclared_inputs(
    config: ExperimentConfig, manifest: dict[str, Any]
) -> list[dict[str, str]]:
    declared = {entry["path"] for entry in manifest["entries"]}
    declared_paths = [config.resolve_path(path) for path in declared]
    roots = {path.parent for path in declared_paths}
    rejected: list[dict[str, str]] = []
    for root in sorted(roots):
        if not root.exists():
            continue
        for path in sorted(root.rglob("*")):
            rel = (
                str(path.relative_to(config.root))
                if path.is_relative_to(config.root)
                else str(path)
            )
            if rel in declared:
                continue
            if path.is_file() and (
                path.suffix == ".jsonl" or path.name.startswith(".") or "cache" in path.parts
            ):
                rejected.append({"path": rel, "reason": "not declared in corpus manifest"})
    return rejected


def _build_report(
    conversations: list[Conversation],
    fingerprint: str,
    rejected_inputs: list[dict[str, str]],
    near_duplicate_mode: str,
) -> dict[str, Any]:
    by_split: dict[str, dict[str, int]] = {
        split: {"conversations": 0, "messages": 0, "normalized_characters": 0} for split in SPLITS
    }
    for conversation in conversations:
        bucket = by_split[conversation.split]
        bucket["conversations"] += 1
        bucket["messages"] += len(conversation.messages)
        bucket["normalized_characters"] += sum(
            len(message.content) for message in conversation.messages
        )
    return {
        "format_version": 1,
        "splits": by_split,
        "conversation_count": len(conversations),
        "near_duplicate_mode": near_duplicate_mode,
        "near_duplicate_similarity_checked": near_duplicate_mode == "exhaustive",
        "rejected_inputs": rejected_inputs,
        "canonical_corpus_fingerprint": fingerprint,
    }
