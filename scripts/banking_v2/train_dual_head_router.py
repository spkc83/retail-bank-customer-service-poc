# /// script
# dependencies = [
#   "huggingface-hub>=1.4,<2",
#   "safetensors>=0.6,<1",
#   "torch>=2.9,<3",
#   "transformers>=5.13,<5.14",
# ]
# ///
"""Train, calibrate, evaluate, and publish the banking dual-head router."""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import HfApi, hf_hub_download
from safetensors.torch import load_file, save_file
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

DATASET_ID = "spkc83/retail-bank-router-training-data"
DATASET_REVISION = "96383306134a9f3331dd47cd936e65a70c585d99"
DESTINATION_ID = "spkc83/retail-bank-domain-intent-router"
BASE_MODEL_ID = "distilbert/distilbert-base-uncased"
BASE_MODEL_REVISION = "12040accade4e8a0f71eabdb258fecc2e7e948be"
SEED = 7101
MAX_LENGTH = 96
BATCH_SIZE = 64
EPOCHS = 4
LEARNING_RATE = 3e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.10
INTENT_LOSS_WEIGHT = 0.7
MINIMUM_IN_DOMAIN_RECALL = 0.98


class RouterDataset(Dataset[dict[str, Any]]):
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.rows[index]


class DualHeadRouter(nn.Module):
    def __init__(self, encoder: nn.Module, *, hidden_size: int, num_intents: int) -> None:
        super().__init__()
        self.encoder = encoder
        self.dropout = nn.Dropout(0.1)
        self.domain_head = nn.Linear(hidden_size, 2)
        self.intent_head = nn.Linear(hidden_size, num_intents)

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self.dropout(outputs.last_hidden_state[:, 0])
        return self.domain_head(pooled), self.intent_head(pooled)


def calibrate_threshold(
    probabilities: Sequence[float],
    labels: Sequence[int],
    *,
    minimum_in_domain_recall: float = MINIMUM_IN_DOMAIN_RECALL,
) -> dict[str, float]:
    if len(probabilities) != len(labels) or not labels:
        raise ValueError("probabilities and labels must be non-empty and equal length")
    candidates = [index / 200 for index in range(1, 200)]
    scored = []
    for threshold in candidates:
        counts = _domain_counts(probabilities, labels, threshold)
        recall = _safe_ratio(counts["true_positive"], counts["positive"])
        specificity = _safe_ratio(counts["true_negative"], counts["negative"])
        scored.append(
            {
                "threshold": threshold,
                "in_domain_recall": recall,
                "ood_specificity": specificity,
                "balanced_accuracy": (recall + specificity) / 2,
            }
        )
    eligible = [
        item for item in scored if item["in_domain_recall"] >= minimum_in_domain_recall
    ]
    pool = eligible or scored
    return max(
        pool,
        key=lambda item: (
            item["ood_specificity"],
            item["balanced_accuracy"],
            item["threshold"],
        ),
    )


def evaluate_predictions(
    *,
    domain_probabilities: Sequence[float],
    domain_labels: Sequence[int],
    intent_predictions: Sequence[int],
    intent_labels: Sequence[int],
    example_kinds: Sequence[str],
    threshold: float,
    num_intents: int,
) -> dict[str, Any]:
    lengths = {
        len(domain_probabilities),
        len(domain_labels),
        len(intent_predictions),
        len(intent_labels),
        len(example_kinds),
    }
    if len(lengths) != 1:
        raise ValueError("prediction fields must have equal lengths")
    counts = _domain_counts(domain_probabilities, domain_labels, threshold)
    accepted = [probability >= threshold for probability in domain_probabilities]
    intent_pairs = [
        (prediction, label)
        for prediction, label in zip(intent_predictions, intent_labels, strict=True)
        if label >= 0
    ]
    metrics: dict[str, Any] = {
        "threshold": threshold,
        "rows": len(domain_labels),
        "domain_confusion": counts,
        "in_domain_recall": _safe_ratio(counts["true_positive"], counts["positive"]),
        "ood_specificity": _safe_ratio(counts["true_negative"], counts["negative"]),
        "in_domain_false_refusal_rate": _safe_ratio(
            counts["false_negative"], counts["positive"]
        ),
        "ood_false_accept_rate": _safe_ratio(counts["false_positive"], counts["negative"]),
        "intent_macro_f1": _macro_f1(intent_pairs, num_intents=num_intents),
        "intent_rows": len(intent_pairs),
    }
    metrics["followup_false_refusal_rate"] = _subset_error_rate(
        accepted,
        domain_labels,
        example_kinds,
        target_kind="same_intent_followup",
        expected=True,
    )
    metrics["transition_ood_false_accept_rate"] = _subset_error_rate(
        accepted,
        domain_labels,
        example_kinds,
        target_kind="banking_to_ood_transition",
        expected=False,
    )
    return metrics


def release_gate_failures(metrics: dict[str, Any]) -> list[str]:
    gates = (
        ("intent_macro_f1", ">=", 0.90),
        ("in_domain_false_refusal_rate", "<=", 0.02),
        ("ood_false_accept_rate", "<=", 0.05),
        ("followup_false_refusal_rate", "<=", 0.05),
        ("transition_ood_false_accept_rate", "<=", 0.05),
    )
    failures = []
    for name, operator, threshold in gates:
        value = float(metrics[name])
        passed = value >= threshold if operator == ">=" else value <= threshold
        if not passed:
            failures.append(f"{name}={value:.6f} must be {operator} {threshold:.6f}")
    return failures


def main() -> int:
    _set_seed(SEED)
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN is required to publish the trained router")

    manifest, rows_by_split = load_governed_data(token=token)
    intent_labels = [str(label) for label in manifest["report"]["intent_labels"]]
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL_ID,
        revision=BASE_MODEL_REVISION,
        token=token,
    )
    encoder = AutoModel.from_pretrained(
        BASE_MODEL_ID,
        revision=BASE_MODEL_REVISION,
        token=token,
    )
    model = DualHeadRouter(
        encoder,
        hidden_size=int(encoder.config.hidden_size),
        num_intents=len(intent_labels),
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(
        json.dumps(
            {
                "stage": "initialized",
                "device": str(device),
                "train_rows": len(rows_by_split["train"]),
                "validation_rows": len(rows_by_split["validation"]),
                "test_rows": len(rows_by_split["test"]),
                "intent_count": len(intent_labels),
            }
        ),
        flush=True,
    )

    collate = make_collate(tokenizer)
    train_loader = DataLoader(
        RouterDataset(rows_by_split["train"]),
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate,
        num_workers=2,
        pin_memory=device.type == "cuda",
    )
    validation_loader = DataLoader(
        RouterDataset(rows_by_split["validation"]),
        batch_size=BATCH_SIZE * 2,
        shuffle=False,
        collate_fn=collate,
        num_workers=2,
        pin_memory=device.type == "cuda",
    )
    test_loader = DataLoader(
        RouterDataset(rows_by_split["test"]),
        batch_size=BATCH_SIZE * 2,
        shuffle=False,
        collate_fn=collate,
        num_workers=2,
        pin_memory=device.type == "cuda",
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    total_steps = EPOCHS * len(train_loader)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=round(total_steps * WARMUP_RATIO),
        num_training_steps=total_steps,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    history = []
    best_score = -math.inf

    with tempfile.TemporaryDirectory(prefix="retail-bank-router-") as temp_dir:
        best_path = Path(temp_dir) / "best.safetensors"
        for epoch in range(1, EPOCHS + 1):
            training_loss = train_epoch(
                model,
                train_loader,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                device=device,
                epoch=epoch,
            )
            validation_predictions = predict(model, validation_loader, device=device)
            calibration = calibrate_threshold(
                validation_predictions["domain_probabilities"],
                validation_predictions["domain_labels"],
            )
            validation_metrics = evaluate_predictions(
                **validation_predictions,
                threshold=float(calibration["threshold"]),
                num_intents=len(intent_labels),
            )
            score = (
                float(validation_metrics["intent_macro_f1"])
                + float(validation_metrics["in_domain_recall"])
                + float(validation_metrics["ood_specificity"])
            )
            epoch_result = {
                "epoch": epoch,
                "training_loss": training_loss,
                "calibration": calibration,
                "validation": validation_metrics,
                "selection_score": score,
            }
            history.append(epoch_result)
            print(json.dumps({"stage": "epoch_complete", **epoch_result}), flush=True)
            if score > best_score:
                best_score = score
                save_file(
                    {
                        name: tensor.detach().cpu().contiguous()
                        for name, tensor in model.state_dict().items()
                    },
                    best_path,
                )

        model.load_state_dict(load_file(best_path, device=str(device)), strict=True)
        best_epoch = max(history, key=lambda item: float(item["selection_score"]))
        threshold = float(best_epoch["calibration"]["threshold"])
        test_predictions = predict(model, test_loader, device=device)
        test_metrics = evaluate_predictions(
            **test_predictions,
            threshold=threshold,
            num_intents=len(intent_labels),
        )
        failures = release_gate_failures(test_metrics)
        metrics = {
            "data_revision": DATASET_REVISION,
            "base_model": BASE_MODEL_ID,
            "base_revision": BASE_MODEL_REVISION,
            "seed": SEED,
            "training": {
                "epochs": EPOCHS,
                "batch_size": BATCH_SIZE,
                "learning_rate": LEARNING_RATE,
                "weight_decay": WEIGHT_DECAY,
                "intent_loss_weight": INTENT_LOSS_WEIGHT,
            },
            "history": history,
            "selected_epoch": best_epoch["epoch"],
            "calibrated_threshold": threshold,
            "test": test_metrics,
            "release_gate_failures": failures,
            "release_eligible": not failures,
        }
        publish_artifact(
            model=model,
            tokenizer=tokenizer,
            intent_labels=intent_labels,
            data_manifest=manifest,
            metrics=metrics,
            token=token,
        )
        print(json.dumps({"stage": "completed", **metrics}, indent=2), flush=True)
        return 0 if not failures else 2


def load_governed_data(
    *,
    token: str,
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    manifest_path = Path(
        hf_hub_download(
            DATASET_ID,
            "manifest.json",
            repo_type="dataset",
            revision=DATASET_REVISION,
            token=token,
        )
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("contract") != "banking-dual-head-router-data":
        raise ValueError("unexpected router dataset manifest contract")
    rows_by_split = {}
    for entry in manifest["splits"]:
        split = str(entry["name"])
        path = Path(
            hf_hub_download(
                DATASET_ID,
                str(entry["path"]),
                repo_type="dataset",
                revision=DATASET_REVISION,
                token=token,
            )
        )
        if file_sha256(path) != entry["sha256"]:
            raise ValueError(f"{split} dataset digest mismatch")
        rows_by_split[split] = [
            json.loads(line) for line in path.open(encoding="utf-8") if line.strip()
        ]
    return manifest, rows_by_split


def make_collate(tokenizer: Any) -> Any:
    def collate(rows: list[dict[str, Any]]) -> dict[str, Any]:
        encoded = tokenizer(
            [str(row["text"]) for row in rows],
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )
        return {
            **encoded,
            "domain_labels": torch.tensor(
                [int(row["domain_label"]) for row in rows],
                dtype=torch.long,
            ),
            "intent_labels": torch.tensor(
                [int(row["intent_label"]) for row in rows],
                dtype=torch.long,
            ),
            "example_kinds": [str(row["example_kind"]) for row in rows],
        }

    return collate


def train_epoch(
    model: DualHeadRouter,
    loader: DataLoader[Any],
    *,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    epoch: int,
) -> float:
    model.train()
    total_loss = 0.0
    for step, batch in enumerate(loader, start=1):
        optimizer.zero_grad(set_to_none=True)
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        domain_labels = batch["domain_labels"].to(device, non_blocking=True)
        intent_labels = batch["intent_labels"].to(device, non_blocking=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=device.type == "cuda",
        ):
            domain_logits, intent_logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            domain_loss = nn.functional.cross_entropy(domain_logits, domain_labels)
            active_intent = intent_labels >= 0
            intent_loss = (
                nn.functional.cross_entropy(
                    intent_logits[active_intent],
                    intent_labels[active_intent],
                )
                if active_intent.any()
                else domain_loss.new_zeros(())
            )
            loss = domain_loss + (INTENT_LOSS_WEIGHT * intent_loss)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        total_loss += float(loss.detach())
        if step % 100 == 0:
            print(
                json.dumps(
                    {
                        "stage": "training",
                        "epoch": epoch,
                        "step": step,
                        "steps_in_epoch": len(loader),
                        "loss": float(loss.detach()),
                    }
                ),
                flush=True,
            )
    return total_loss / max(len(loader), 1)


def predict(
    model: DualHeadRouter,
    loader: DataLoader[Any],
    *,
    device: torch.device,
) -> dict[str, list[Any]]:
    model.eval()
    domain_probabilities: list[float] = []
    domain_labels: list[int] = []
    intent_predictions: list[int] = []
    intent_labels: list[int] = []
    example_kinds: list[str] = []
    with torch.inference_mode():
        for batch in loader:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                domain_logits, intent_logits = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
            domain_probabilities.extend(
                torch.softmax(domain_logits.float(), dim=-1)[:, 1].cpu().tolist()
            )
            domain_labels.extend(batch["domain_labels"].tolist())
            intent_predictions.extend(intent_logits.argmax(dim=-1).cpu().tolist())
            intent_labels.extend(batch["intent_labels"].tolist())
            example_kinds.extend(batch["example_kinds"])
    return {
        "domain_probabilities": domain_probabilities,
        "domain_labels": domain_labels,
        "intent_predictions": intent_predictions,
        "intent_labels": intent_labels,
        "example_kinds": example_kinds,
    }


def publish_artifact(
    *,
    model: DualHeadRouter,
    tokenizer: Any,
    intent_labels: list[str],
    data_manifest: dict[str, Any],
    metrics: dict[str, Any],
    token: str,
) -> None:
    with tempfile.TemporaryDirectory(prefix="retail-bank-router-release-") as temp_dir:
        output = Path(temp_dir)
        model.encoder.save_pretrained(output, safe_serialization=True)
        tokenizer.save_pretrained(output)
        save_file(
            {
                "domain_head.weight": model.domain_head.weight.detach().cpu().contiguous(),
                "domain_head.bias": model.domain_head.bias.detach().cpu().contiguous(),
                "intent_head.weight": model.intent_head.weight.detach().cpu().contiguous(),
                "intent_head.bias": model.intent_head.bias.detach().cpu().contiguous(),
            },
            output / "classifier_heads.safetensors",
        )
        router_config = {
            "contract": "banking-dual-head-router",
            "format_version": 1,
            "architecture": "distilbert-shared-encoder-dual-head",
            "base_model": BASE_MODEL_ID,
            "base_revision": BASE_MODEL_REVISION,
            "data_repo": DATASET_ID,
            "data_revision": DATASET_REVISION,
            "domain_labels": ["out_of_domain", "in_domain"],
            "intent_labels": intent_labels,
            "domain_threshold": metrics["calibrated_threshold"],
            "max_length": MAX_LENGTH,
            "input_format": "[CURRENT]\\n{text}\\n[PREVIOUS_USER]\\n{previous_user}",
            "fail_closed": True,
        }
        (output / "router_config.json").write_text(
            json.dumps(router_config, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (output / "metrics.json").write_text(
            json.dumps(metrics, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (output / "README.md").write_text(
            model_card(metrics=metrics, data_manifest=data_manifest),
            encoding="utf-8",
        )
        files = []
        for path in sorted(output.rglob("*")):
            if path.is_file():
                files.append(
                    {
                        "path": str(path.relative_to(output)),
                        "bytes": path.stat().st_size,
                        "sha256": file_sha256(path),
                    }
                )
        artifact_manifest = {
            "contract": "banking-dual-head-router-artifact",
            "format_version": 1,
            "source_run_id": os.environ.get("JOB_ID", "hf-job"),
            "implementation_version": os.environ.get("SOURCE_COMMIT", "unknown"),
            "data_revision": DATASET_REVISION,
            "release_eligible": metrics["release_eligible"],
            "signed": False,
            "files": files,
        }
        (output / "manifest.json").write_text(
            json.dumps(artifact_manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        api = HfApi(token=token)
        api.create_repo(
            repo_id=DESTINATION_ID,
            repo_type="model",
            private=False,
            exist_ok=True,
        )
        commit = api.upload_folder(
            folder_path=output,
            repo_id=DESTINATION_ID,
            repo_type="model",
            commit_message=(
                "Publish release-eligible dual-head router"
                if metrics["release_eligible"]
                else "Publish dual-head router candidate with failed gates"
            ),
        )
        print(json.dumps({"stage": "published", "commit": str(commit)}), flush=True)


def model_card(*, metrics: dict[str, Any], data_manifest: dict[str, Any]) -> str:
    test = metrics["test"]
    return f"""---
base_model: {BASE_MODEL_ID}
datasets:
  - {DATASET_ID}
library_name: transformers
license: apache-2.0
pipeline_tag: text-classification
tags:
  - banking
  - intent-classification
  - out-of-domain-detection
---

# Retail Bank domain-intent router

DistilBERT shared encoder with a binary supported-banking/OOD head and a
77-way Banking77 intent head. The intent loss is masked for CLINC rows.

## Held-out results

- Release eligible: `{metrics["release_eligible"]}`
- Intent macro F1: `{test["intent_macro_f1"]:.6f}`
- In-domain false-refusal rate: `{test["in_domain_false_refusal_rate"]:.6f}`
- OOD false-accept rate: `{test["ood_false_accept_rate"]:.6f}`
- Follow-up false-refusal rate: `{test["followup_false_refusal_rate"]:.6f}`
- Banking-to-OOD false-accept rate: `{test["transition_ood_false_accept_rate"]:.6f}`
- Calibrated banking threshold: `{metrics["calibrated_threshold"]:.6f}`

## Data and licenses

Classifier-only data combines PolyAI Banking77 and UCI CLINC150 under
CC-BY-4.0. Banking77 is prohibited from the generative SFT lane. The prepared
dataset contains {data_manifest["report"]["split_counts"]["train"]} train,
{data_manifest["report"]["split_counts"]["validation"]} validation, and
{data_manifest["report"]["split_counts"]["test"]} test rows.

## Safety boundary

If the artifact is unavailable or corrupt, serving must fail closed to the
standard OOD response. This model does not replace credential-input or
unsafe-output guards.
"""


def _domain_counts(
    probabilities: Sequence[float],
    labels: Sequence[int],
    threshold: float,
) -> dict[str, int]:
    true_positive = false_positive = true_negative = false_negative = 0
    for probability, label in zip(probabilities, labels, strict=True):
        predicted = probability >= threshold
        if label == 1 and predicted:
            true_positive += 1
        elif label == 1:
            false_negative += 1
        elif predicted:
            false_positive += 1
        else:
            true_negative += 1
    return {
        "true_positive": true_positive,
        "false_positive": false_positive,
        "true_negative": true_negative,
        "false_negative": false_negative,
        "positive": true_positive + false_negative,
        "negative": true_negative + false_positive,
    }


def _macro_f1(pairs: Sequence[tuple[int, int]], *, num_intents: int) -> float:
    scores = []
    for label in range(num_intents):
        true_positive = sum(prediction == label and target == label for prediction, target in pairs)
        false_positive = sum(
            prediction == label and target != label for prediction, target in pairs
        )
        false_negative = sum(
            prediction != label and target == label for prediction, target in pairs
        )
        support = true_positive + false_negative
        if support == 0:
            continue
        denominator = (2 * true_positive) + false_positive + false_negative
        scores.append((2 * true_positive) / denominator if denominator else 0.0)
    return sum(scores) / len(scores) if scores else 0.0


def _subset_error_rate(
    accepted: Sequence[bool],
    domain_labels: Sequence[int],
    example_kinds: Sequence[str],
    *,
    target_kind: str,
    expected: bool,
) -> float:
    subset = [
        prediction == expected
        for prediction, _label, kind in zip(
            accepted,
            domain_labels,
            example_kinds,
            strict=True,
        )
        if kind == target_kind
    ]
    return 1.0 - _safe_ratio(sum(subset), len(subset)) if subset else 1.0


def _safe_ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_float32_matmul_precision("high")


if __name__ == "__main__":
    raise SystemExit(main())
