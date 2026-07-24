from __future__ import annotations

import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

from hello_slm.artifacts import atomic_write_json, environment_record, sha256_file
from hello_slm.config import ExperimentConfig, canonical_sha256
from hello_slm.data import Conversation, load_and_validate_corpus, normalize_text
from hello_slm.generation import _reject_disallowed_characters, _render_prompt
from hello_slm.tokenizer import SPECIAL_TOKENS, TokenizerError, load_tokenizer
from hello_slm.training import PipelineError, _resolve_precision_runtime, load_model_from_checkpoint

INTEGER_PATTERN = re.compile(r"(?<![\w.])-?\d+(?!\.\d)(?!\w)")
OPERATION_PATTERNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("addition", ("+", "sum", "total", "altogether", "combined", "add")),
    ("subtraction", ("-", "left", "remain", "difference", "subtract", "fewer")),
    ("multiplication", ("*", "x", "times", "product", "each", "groups of")),
    ("division", ("/", "divide", "divided", "quotient", "split equally", "share equally")),
)
UNSEEN_CHALLENGES = (
    {"operation": "addition", "prompt": "What is 17 + 28?", "expected_answer": 45},
    {"operation": "subtraction", "prompt": "What is 93 - 47?", "expected_answer": 46},
    {"operation": "multiplication", "prompt": "What is 12 * 8?", "expected_answer": 96},
    {"operation": "division", "prompt": "What is 144 / 12?", "expected_answer": 12},
)
ARITHMETIC_MAX_NEW_TOKENS = 64
PROFILE_SUPPORTED_OPERATIONS = {
    "arithmetic-curriculum-30m": ("add", "subtract", "divide"),
}


def evaluate_arithmetic(
    config: ExperimentConfig,
    *,
    checkpoint_path: str | Path,
    max_per_operation: int = 50,
    report_path: str | Path | None = None,
) -> dict[str, Any]:
    started = time.time()
    if max_per_operation < 1:
        raise PipelineError("max_per_operation must be positive")
    precision_runtime = _resolve_precision_runtime(config)
    device = torch.device(precision_runtime.device)
    model, checkpoint, context = load_model_from_checkpoint(config, checkpoint_path, device=device)
    tokenizer_path = config.artifact_dir / "tokenizer" / "tokenizer.json"
    tokenizer = load_tokenizer(tokenizer_path)
    corpus = load_and_validate_corpus(config)
    examples = select_arithmetic_examples(corpus.by_split("test"), max_per_operation)
    if not examples:
        raise PipelineError("test split has no arithmetic examples with expected integer answers")

    records = []
    for example in examples:
        generated = generate_arithmetic_response(
            config,
            model=model,
            tokenizer=tokenizer,
            device=device,
            prompt=example["prompt"],
        )
        predicted = extract_final_integer(generated)
        expected = int(example["expected_answer"])
        records.append(
            {
                **example,
                "generated_response": generated,
                "predicted_answer": predicted,
                "parse_failed": predicted is None,
                "correct": predicted == expected,
            }
        )

    overall, by_operation = score_arithmetic_records(records)
    supported_operations = PROFILE_SUPPORTED_OPERATIONS.get(
        str(config.data["run"]["id"]),
        tuple(sorted(by_operation)),
    )
    supported_records = [
        record for record in records if record["operation"] in supported_operations
    ]
    supported_overall, _ = score_arithmetic_records(supported_records)
    gate = gate_arithmetic_scores(
        supported_overall,
        {name: by_operation[name] for name in supported_operations if name in by_operation},
        required_operations=supported_operations,
    )
    report = {
        "command": "eval-arithmetic",
        "status": "success" if gate["passed"] else "failed",
        "profile": config.data["run"]["id"],
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "global_step": checkpoint["global_step"],
        "heldout_split": "test",
        "selection": {
            "mode": "deterministic_stratified_closed_corpus_fact_recall",
            "fact_disjoint_from_training": False,
            "claim": "paraphrase recall over the bounded restricted-corpus fact table",
            "max_per_operation": max_per_operation,
            "examples": len(examples),
            "selection_hash": canonical_sha256(
                [(item["conversation_id"], item["operation"]) for item in examples]
            ),
        },
        "overall": overall,
        "supported_operations": list(supported_operations),
        "supported_overall": supported_overall,
        "by_operation": by_operation,
        "gate": gate,
        "records": records,
        "unseen_challenges": list(UNSEEN_CHALLENGES),
        "unseen_challenges_scored": False,
        "decoding": {
            "max_new_tokens": ARITHMETIC_MAX_NEW_TOKENS,
            "temperature": 0.0,
            "top_k": 1,
            "top_p": 1.0,
            "repetition_penalty": 1.0,
        },
        "precision_runtime": precision_runtime.to_mapping(),
        "fingerprints": context,
        "environment": environment_record(),
        "duration_seconds": time.time() - started,
    }
    output = (
        Path(report_path)
        if report_path is not None
        else config.artifact_dir / "reports" / "arithmetic-exact.json"
    )
    atomic_write_json(output, report)
    return report


@torch.no_grad()
def generate_arithmetic_response(
    config: ExperimentConfig,
    *,
    model: Any,
    tokenizer: Any,
    device: torch.device,
    prompt: str,
) -> str:
    normalized = normalize_text(prompt)
    if not normalized.strip():
        raise PipelineError("empty arithmetic prompt")
    _reject_disallowed_characters(config, normalized)
    rendered = _render_prompt(config, normalized)
    try:
        prompt_ids = tokenizer.encode(rendered, allow_unk=False)
    except TokenizerError as exc:
        raise PipelineError(f"prompt cannot be encoded by restricted tokenizer: {exc}") from exc
    if len(prompt_ids) >= model.config.max_seq_len:
        raise PipelineError("rendered arithmetic prompt leaves no room for generation")
    max_new_tokens = min(
        ARITHMETIC_MAX_NEW_TOKENS,
        model.config.max_seq_len - len(prompt_ids),
    )
    output = model.generate(
        torch.tensor(prompt_ids, dtype=torch.long, device=device),
        max_new_tokens=max_new_tokens,
        temperature=0.0,
        top_k=1,
        top_p=1.0,
        repetition_penalty=1.0,
        stop_ids={SPECIAL_TOKENS["<|end|>"], SPECIAL_TOKENS["<|eos|>"]},
    )
    ids = output[0].detach().cpu().tolist()
    generated_ids = [
        token_id
        for token_id in ids[len(prompt_ids) :]
        if token_id not in {SPECIAL_TOKENS["<|end|>"], SPECIAL_TOKENS["<|eos|>"]}
    ]
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


def select_arithmetic_examples(
    conversations: list[Conversation], max_per_operation: int
) -> list[dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for conversation in conversations:
        example = conversation_to_arithmetic_example(conversation)
        if example is None:
            continue
        buckets[example["operation"]].append(example)
    selected = []
    for operation in sorted(buckets):
        selected.extend(
            sorted(buckets[operation], key=lambda item: item["conversation_id"])[:max_per_operation]
        )
    return selected


def conversation_to_arithmetic_example(conversation: Conversation) -> dict[str, Any] | None:
    user = next(
        (message.content for message in conversation.messages if message.role == "user"), None
    )
    assistant = next(
        (
            message.content
            for message in reversed(conversation.messages)
            if message.role == "assistant"
        ),
        None,
    )
    if user is None or assistant is None:
        return None
    metadata = conversation.metadata or {}
    expected_raw = metadata.get("expected_answer")
    expected = (
        int(expected_raw) if isinstance(expected_raw, int) else extract_final_integer(assistant)
    )
    if expected is None:
        return None
    operation_raw = metadata.get("operation")
    operation = (
        str(operation_raw).strip().casefold()
        if isinstance(operation_raw, str) and operation_raw.strip()
        else infer_operation(user)
    )
    return {
        "conversation_id": conversation.conversation_id,
        "operation": operation,
        "prompt": user,
        "expected_answer": expected,
    }


def extract_final_integer(text: str) -> int | None:
    matches = INTEGER_PATTERN.findall(text.replace(",", ""))
    if not matches:
        return None
    return int(matches[-1])


def infer_operation(prompt: str) -> str:
    normalized = f" {prompt.casefold()} "
    for operation, markers in OPERATION_PATTERNS:
        if any(marker in normalized for marker in markers):
            return operation
    return "unknown"


def score_arithmetic_records(
    records: list[dict[str, Any]],
) -> tuple[dict[str, int | float], dict[str, dict[str, int | float]]]:
    total = len(records)
    correct = sum(1 for record in records if record["correct"])
    parse_failures = sum(1 for record in records if record["parse_failed"])
    by_operation: dict[str, dict[str, int | float]] = {}
    for operation in sorted({str(record["operation"]) for record in records}):
        op_records = [record for record in records if record["operation"] == operation]
        op_correct = sum(1 for record in op_records if record["correct"])
        op_parse_failures = sum(1 for record in op_records if record["parse_failed"])
        by_operation[operation] = {
            "count": len(op_records),
            "correct": op_correct,
            "accuracy": op_correct / len(op_records),
            "parse_failures": op_parse_failures,
        }
    return (
        {
            "count": total,
            "correct": correct,
            "accuracy": correct / total if total else 0.0,
            "parse_failures": parse_failures,
        },
        by_operation,
    )


def gate_arithmetic_scores(
    overall: dict[str, int | float],
    by_operation: dict[str, dict[str, int | float]],
    *,
    required_operations: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    failures = []
    if float(overall["accuracy"]) < 0.90:
        failures.append("overall accuracy below 0.90")
    for operation in required_operations or ():
        if operation not in by_operation or int(by_operation[operation]["count"]) == 0:
            failures.append(f"required operation {operation} has no selected examples")
    for operation, scores in by_operation.items():
        if float(scores["accuracy"]) < 0.80:
            failures.append(f"{operation} accuracy below 0.80")
    return {
        "passed": not failures,
        "overall_accuracy_min": 0.90,
        "per_operation_accuracy_min": 0.80,
        "failures": failures,
    }
