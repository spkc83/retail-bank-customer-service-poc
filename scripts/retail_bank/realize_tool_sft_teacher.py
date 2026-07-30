#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from hello_slm.config import canonical_json_bytes

TEACHER_PROMPT_VERSION = "banking-tool-sft-teacher/v1"
PRIVATE_VALUE_RE = re.compile(r"\b(?:\d[ -]?){9,}\b")
DIGIT_RE = re.compile(r"\d{2,}")
FACT_WORD_RE = re.compile(
    r"\b(?:frozen|active|closed|open|pending|completed|cancelled|canceled|failed|"
    r"checking|savings|credit|debit|replacement|dispute|transfer|transaction)\b",
    re.IGNORECASE,
)


class TeacherRealizationError(ValueError):
    """Raised when teacher realization input or output is invalid."""


@dataclass(frozen=True)
class RealizerConfig:
    input_requests: Path
    output_responses: Path
    model: str = ""
    revision: str = ""
    device: str = "cuda"
    batch_size: int = 4
    max_new_tokens: int = 220
    seed: int = 7303
    dry_run: bool = False


class TeacherBackend(Protocol):
    def realize_batch(self, rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
        """Return one JSON-compatible response row for each request row."""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Realize banking-v3 tool-use SFT teacher request JSONL locally."
    )
    parser.add_argument("--input-requests", type=Path, required=True)
    parser.add_argument("--output-responses", type=Path, required=True)
    parser.add_argument(
        "--model",
        help="Exact teacher model ID. Required unless --dry-run is used.",
    )
    parser.add_argument(
        "--revision",
        help="Exact immutable teacher model revision. Required unless --dry-run is used.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=220)
    parser.add_argument("--seed", type=int, default=7303)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    try:
        config = RealizerConfig(
            input_requests=args.input_requests,
            output_responses=args.output_responses,
            model=str(args.model or ""),
            revision=str(args.revision or ""),
            device=str(args.device),
            batch_size=int(args.batch_size),
            max_new_tokens=int(args.max_tokens),
            seed=int(args.seed),
            dry_run=bool(args.dry_run),
        )
        report = realize_teacher_requests(config)
    except (OSError, TeacherRealizationError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(json.dumps({"status": "success", **report}, sort_keys=True))
    return 0


def realize_teacher_requests(config: RealizerConfig) -> dict[str, Any]:
    if config.batch_size < 1:
        raise TeacherRealizationError("--batch-size must be at least 1")
    if config.max_new_tokens < 1:
        raise TeacherRealizationError("--max-tokens must be at least 1")
    if not config.dry_run and (not config.model or not config.revision):
        raise TeacherRealizationError(
            "--model and --revision are required unless --dry-run is used"
        )

    requests = _read_jsonl(config.input_requests)
    for row in requests:
        validate_request_row(row)

    completed = _read_existing_responses(config.output_responses)
    for row in requests:
        existing = completed.get(_required_str(row, "record_id"))
        if existing is not None:
            validate_response_row(row, existing)
    pending = [
        row
        for row in requests
        if _required_str(row, "record_id") not in completed
    ]
    backend: TeacherBackend
    if config.dry_run:
        backend = DryRunTeacherBackend(seed=config.seed)
    else:
        backend = TransformersTeacherBackend(
            model_id=config.model,
            revision=config.revision,
            device=config.device,
            max_new_tokens=config.max_new_tokens,
            seed=config.seed,
        )

    written = 0
    config.output_responses.parent.mkdir(parents=True, exist_ok=True)
    with config.output_responses.open("a", encoding="utf-8", newline="\n") as handle:
        for start in range(0, len(pending), config.batch_size):
            batch = pending[start : start + config.batch_size]
            realized = backend.realize_batch(batch)
            if len(realized) != len(batch):
                raise TeacherRealizationError("teacher backend returned the wrong batch size")
            for request_row, response_row in zip(batch, realized, strict=True):
                validated = validate_response_row(request_row, response_row)
                handle.write(json.dumps(validated, sort_keys=True) + "\n")
                handle.flush()
                written += 1

    return {
        "input_records": len(requests),
        "already_realized": len(completed),
        "written": written,
        "output_responses": str(config.output_responses),
        "model": config.model,
        "revision": config.revision,
        "dry_run": config.dry_run,
    }


class DryRunTeacherBackend:
    def __init__(self, *, seed: int) -> None:
        self._rng = random.Random(seed)

    def realize_batch(self, rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
        return [self._realize(row) for row in rows]

    def _realize(self, row: dict[str, Any]) -> dict[str, Any]:
        opener = self._rng.choice(("Please", "Can you", "Could you"))
        user = _required_str(row, "user_content").strip()
        final = _required_str(row, "final_response").strip()
        if not user.lower().startswith(("please", "can you", "could you")):
            user = f"{opener} {user[0].lower()}{user[1:]}"
        if not final.lower().startswith(("done.", "i checked", "here is")):
            final = f"I checked the banking result. {final}"
        return {
            "record_id": row["record_id"],
            "immutable_hash": row["immutable_hash"],
            "user_content": user,
            "final_response": final,
        }


class TransformersTeacherBackend:
    def __init__(
        self,
        *,
        model_id: str,
        revision: str,
        device: str,
        max_new_tokens: int,
        seed: int,
    ) -> None:
        self.model_id = model_id
        self.revision = revision
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.seed = seed
        self._tokenizer: Any | None = None
        self._model: Any | None = None

    def realize_batch(self, rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
        self._ensure_loaded()
        if self._tokenizer is None or self._model is None:
            raise TeacherRealizationError("teacher model failed to initialize")

        import torch

        prompts = [render_prompt(row) for row in rows]
        messages = [
            [{"role": "user", "content": prompt}]
            for prompt in prompts
        ]
        rendered = self._tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )
        encoded = self._tokenizer(
            rendered,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.device)
        torch.manual_seed(self.seed)
        if str(self.device).startswith("cuda"):
            torch.cuda.manual_seed_all(self.seed)
        with torch.inference_mode():
            output_ids = self._model.generate(
                **encoded,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self._pad_token_id(),
                eos_token_id=getattr(self._tokenizer, "eos_token_id", None),
            )
        input_width = int(encoded["input_ids"].shape[1])
        decoded = self._tokenizer.batch_decode(
            output_ids[:, input_width:],
            skip_special_tokens=True,
        )
        return [parse_teacher_json(text) for text in decoded]

    def _ensure_loaded(self) -> None:
        if self._model is not None and self._tokenizer is not None:
            return
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise TeacherRealizationError(
                "transformers and torch are required unless --dry-run is used"
            ) from exc

        if self.device == "cuda" and not torch.cuda.is_available():
            raise TeacherRealizationError("CUDA device requested but unavailable")
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            revision=self.revision,
            trust_remote_code=False,
        )
        if getattr(tokenizer, "pad_token_id", None) is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            revision=self.revision,
            dtype=torch.float16,
            trust_remote_code=False,
        )
        model.to(self.device)
        model.eval()
        self._tokenizer = tokenizer
        self._model = model

    def _pad_token_id(self) -> int | None:
        if self._tokenizer is None:
            return None
        pad_token_id = getattr(self._tokenizer, "pad_token_id", None)
        if pad_token_id is not None:
            return int(pad_token_id)
        eos_token_id = getattr(self._tokenizer, "eos_token_id", None)
        return int(eos_token_id) if eos_token_id is not None else None


def render_prompt(row: dict[str, Any]) -> str:
    prompt_payload = {
        "record_id": row["record_id"],
        "immutable_hash": row["immutable_hash"],
        "user_content": row["user_content"],
        "final_response": row["final_response"],
        "allowed_edits": ["user_content", "final_response"],
        "immutable_fields": row.get("immutable_fields", []),
    }
    return (
        "You rewrite synthetic retail-banking training examples.\n"
        "Return one JSON object only. Required keys: record_id, immutable_hash, "
        "user_content, final_response.\n"
        "You may rewrite only user_content and final_response. Preserve all banking facts, "
        "amounts, card digits, statuses, tool semantics, and action outcomes. Do not ask for "
        "or reveal account numbers, customer IDs, passwords, PINs, or private backend IDs. "
        "Do not add markdown.\n"
        f"Request:\n{json.dumps(prompt_payload, sort_keys=True)}"
    )


def prompt_hash() -> str:
    return f"sha256:{hashlib.sha256(TEACHER_PROMPT_VERSION.encode()).hexdigest()}"


def parse_teacher_json(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.removeprefix("```json").removeprefix("```").strip()
        stripped = stripped.removesuffix("```").strip()
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
        if match is None:
            raise TeacherRealizationError("teacher output did not contain JSON") from None
        parsed = json.loads(match.group(0))
    if not isinstance(parsed, dict):
        raise TeacherRealizationError("teacher output JSON must be an object")
    return parsed


def validate_request_row(row: dict[str, Any]) -> None:
    _required_str(row, "record_id")
    immutable_hash = _required_str(row, "immutable_hash")
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", immutable_hash):
        raise TeacherRealizationError(f"{row.get('record_id')} has invalid immutable_hash")
    _required_str(row, "user_content")
    _required_str(row, "final_response")
    allowed_edits = row.get("allowed_edits")
    if allowed_edits != ["user_content", "final_response"]:
        raise TeacherRealizationError(f"{row.get('record_id')} has unsupported allowed_edits")


def validate_response_row(
    request_row: dict[str, Any],
    response_row: dict[str, Any],
) -> dict[str, str]:
    allowed_keys = {"record_id", "immutable_hash", "user_content", "final_response"}
    extra = set(response_row) - allowed_keys
    if extra:
        raise TeacherRealizationError(
            f"{request_row.get('record_id')} teacher output changed forbidden fields: "
            f"{sorted(extra)}"
        )
    record_id = _required_str(response_row, "record_id")
    immutable_hash = _required_str(response_row, "immutable_hash")
    if record_id != request_row["record_id"]:
        raise TeacherRealizationError(f"{request_row.get('record_id')} record_id mismatch")
    if immutable_hash != request_row["immutable_hash"]:
        raise TeacherRealizationError(f"{record_id} immutable_hash mismatch")

    user_content = _clean_generated_text(response_row, "user_content")
    final_response = _clean_generated_text(response_row, "final_response")
    _validate_fact_preservation(
        record_id,
        before=_required_str(request_row, "user_content"),
        after=user_content,
        field="user_content",
    )
    _validate_fact_preservation(
        record_id,
        before=_required_str(request_row, "final_response"),
        after=final_response,
        field="final_response",
    )
    if _assistant_requests_private_data(final_response):
        raise TeacherRealizationError(f"{record_id} final_response requests private credentials")
    if _contains_private_value(user_content) or _contains_private_value(final_response):
        raise TeacherRealizationError(f"{record_id} teacher output contains private identifiers")
    return {
        "record_id": record_id,
        "immutable_hash": immutable_hash,
        "user_content": user_content,
        "final_response": final_response,
    }


def _read_existing_responses(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    completed: dict[str, dict[str, Any]] = {}
    for row in _read_jsonl(path):
        record_id = _required_str(row, "record_id")
        if record_id in completed and row != completed[record_id]:
            raise TeacherRealizationError(f"{record_id} has conflicting existing responses")
        completed[record_id] = row
    return completed


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise TeacherRealizationError(f"{path}:{line_number} JSONL row must be an object")
            rows.append(row)
    return rows


def _required_str(row: dict[str, Any], key: str) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value.strip():
        raise TeacherRealizationError(f"missing or empty string field: {key}")
    return value.strip()


def _clean_generated_text(row: dict[str, Any], key: str) -> str:
    value = _required_str(row, key)
    if "\n" in value:
        value = " ".join(part.strip() for part in value.splitlines() if part.strip())
    return value


def _validate_fact_preservation(
    record_id: str,
    *,
    before: str,
    after: str,
    field: str,
) -> None:
    required_literals = set(DIGIT_RE.findall(before)) | {
        match.group(0).lower() for match in FACT_WORD_RE.finditer(before)
    }
    after_lower = after.lower()
    missing = sorted(literal for literal in required_literals if literal.lower() not in after_lower)
    if missing:
        raise TeacherRealizationError(f"{record_id} {field} lost banking facts: {missing}")


def _contains_private_value(text: str) -> bool:
    return PRIVATE_VALUE_RE.search(text) is not None


def _assistant_requests_private_data(text: str) -> bool:
    lowered = text.lower()
    return bool(
        re.search(r"\b(?:provide|send|enter|share|tell me|confirm)\b", lowered)
        and re.search(r"\b(?:password|pin|account number|customer id|private id)\b", lowered)
    )


def response_fingerprint(rows: Iterable[dict[str, Any]]) -> str:
    return f"sha256:{hashlib.sha256(canonical_json_bytes(list(rows))).hexdigest()}"


if __name__ == "__main__":
    raise SystemExit(main())
