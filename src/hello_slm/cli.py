from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from hello_slm.arithmetic_evaluation import evaluate_arithmetic
from hello_slm.evaluation import evaluate
from hello_slm.generation import generate_reply
from hello_slm.training import (
    PipelineError,
    build_dataset,
    build_tokenizer,
    load_config,
    train,
    validate_run,
)


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        result = _dispatch(args)
    except PipelineError as exc:
        print(f"hello-slm: {exc}", file=sys.stderr)
        return 2
    except (ValueError, OSError) as exc:
        print(f"hello-slm: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"hello-slm: unexpected runtime error: {exc}", file=sys.stderr)
        return 1
    if result is not None:
        print(json.dumps(result, sort_keys=True))
        if args.command == "eval-arithmetic" and result.get("status") != "success":
            return 3
    return 0


def _dispatch(args: argparse.Namespace) -> dict[str, Any] | None:
    config = load_config(args.config, getattr(args, "work_dir", None))
    if args.command == "validate":
        return validate_run(config, structural=args.structural)
    if args.command == "build-tokenizer":
        return build_tokenizer(config)
    if args.command == "build-dataset":
        return build_dataset(config)
    if args.command == "train":
        return train(config, max_steps=args.max_steps, resume=args.resume)
    if args.command == "eval":
        checkpoint = args.checkpoint or config.artifact_dir / "checkpoints" / "latest.pt"
        return evaluate(config, checkpoint_path=checkpoint)
    if args.command == "eval-arithmetic":
        checkpoint = args.checkpoint or config.artifact_dir / "checkpoints" / "latest.pt"
        return evaluate_arithmetic(
            config,
            checkpoint_path=checkpoint,
            max_per_operation=args.max_per_operation,
        )
    if args.command == "chat":
        checkpoint = args.checkpoint or config.artifact_dir / "checkpoints" / "latest.pt"
        result = generate_reply(
            config,
            checkpoint_path=checkpoint,
            prompt=args.prompt,
            as_json=args.json,
            max_new_tokens=args.max_new_tokens,
        )
        if args.json:
            return result
        print(result["response"])
        return None
    if args.command == "smoke":
        return _smoke(config)
    raise PipelineError(f"unknown command {args.command!r}")


def _smoke(config: Any) -> dict[str, Any]:
    validate = validate_run(config)
    tokenizer = build_tokenizer(config)
    dataset = build_dataset(config)
    first_train = train(config, max_steps=2)
    latest = Path(first_train["checkpoint"])
    resumed_train = train(config, max_steps=3, resume=latest)
    evaluation = evaluate(config, checkpoint_path=resumed_train["checkpoint"])
    chat = generate_reply(
        config,
        checkpoint_path=resumed_train["checkpoint"],
        prompt="Hello",
        as_json=True,
    )
    return {
        "command": "smoke",
        "status": "success",
        "validate": validate["status"],
        "tokenizer": tokenizer["status"],
        "dataset": dataset["status"],
        "first_train_step": first_train["global_step"],
        "resumed_train_step": resumed_train["global_step"],
        "heldout_loss": evaluation["heldout_loss"],
        "heldout_perplexity": evaluation["heldout_perplexity"],
        "assistant_token_accuracy": evaluation["assistant_token_accuracy"],
        "release_eligible": False,
        "chat_stop_reason": chat["metadata"]["stop_reason"],
        "artifact_dir": str(config.artifact_dir),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="hello-slm")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in (
        "validate",
        "build-tokenizer",
        "build-dataset",
        "train",
        "eval",
        "eval-arithmetic",
        "chat",
        "smoke",
    ):
        sub = subparsers.add_parser(command)
        sub.add_argument("--config", required=True)
        sub.add_argument("--work-dir")
        if command == "validate":
            sub.add_argument(
                "--structural",
                action="store_true",
                help="validate config and parameter constraints without reading the corpus",
            )
        if command == "train":
            sub.add_argument("--max-steps", type=int)
            sub.add_argument("--resume")
        if command == "eval":
            sub.add_argument("--checkpoint")
        if command == "eval-arithmetic":
            sub.add_argument("--checkpoint")
            sub.add_argument("--max-per-operation", type=int, default=50)
        if command == "chat":
            sub.add_argument("--checkpoint")
            sub.add_argument("--prompt", required=True)
            sub.add_argument("--max-new-tokens", type=int)
            sub.add_argument("--json", action="store_true")
    return parser
