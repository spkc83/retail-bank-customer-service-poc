#!/usr/bin/env python
"""Guarded banking-v2 MoE conversion/training spec generator.

This script is cloud-intended documentation plus executable guardrails. It does
not implement training, submit a Hugging Face Job, or create a Hub repository.
A future paid training path requires both ``--allow-paid-job`` and
``HELLO_SLM_ALLOW_PAID_JOB=banking-v2``.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from hello_slm.banking_moe import (
    BANKING_V2_BASE_REVISION,
    BANKING_V2_HUB_DEST,
    BANKING_V2_OOD_STOCK_RESPONSE,
    BANKING_V2_ROUTER_EVAL_REVISION,
    BankingV2Pins,
    banking_v2_qwen2_moe_config,
    banking_v2_training_summary,
    qwen2_moe_active_parameter_count,
    qwen2_moe_parameter_count,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/banking-v2-moe-9b.toml",
        help="Banking-v2 MoE config path for operator traceability.",
    )
    parser.add_argument(
        "--allow-paid-job",
        action="store_true",
        help="Enable the paid-job code path when the confirmation env var is also set.",
    )
    parser.add_argument(
        "--hub-dest",
        default=BANKING_V2_HUB_DEST,
        help="Private Hugging Face Hub destination. No repo is created by this script.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=True,
        help="Print the plan and exit. This is the default.",
    )
    return parser.parse_args()


def paid_job_confirmed(args: argparse.Namespace) -> bool:
    return bool(args.allow_paid_job and os.environ.get("HELLO_SLM_ALLOW_PAID_JOB") == "banking-v2")


def build_job_plan(config_path: str | Path, hub_dest: str) -> dict[str, Any]:
    cfg = banking_v2_qwen2_moe_config()
    pins = BankingV2Pins()
    summary = banking_v2_training_summary(cfg)
    return {
        "job_status": "executable_worker_implemented_not_launched",
        "config_path": str(config_path),
        "hub_dest": hub_dest,
        "private_hub_repo": True,
        "job": {
            "provider": "hugging-face-jobs",
            "flavor": "rtx-pro-6000",
            "timeout": "5h",
            "maximum_budget_usd": 13.75,
        },
        "pins": pins.__dict__,
        "model": {
            "total_parameters": qwen2_moe_parameter_count(cfg),
            "estimated_active_parameters": qwen2_moe_active_parameter_count(cfg),
            "config": {
                "vocab_size": cfg.vocab_size,
                "hidden_size": cfg.hidden_size,
                "num_hidden_layers": cfg.num_hidden_layers,
                "num_attention_heads": cfg.num_attention_heads,
                "num_key_value_heads": cfg.num_key_value_heads,
                "intermediate_size": cfg.intermediate_size,
                "shared_expert_intermediate_size": cfg.shared_expert_intermediate_size,
                "moe_intermediate_size": cfg.moe_intermediate_size,
                "num_experts": cfg.num_experts,
                "num_experts_per_tok": cfg.num_experts_per_tok,
                "norm_topk_prob": cfg.norm_topk_prob,
                "output_router_logits": cfg.output_router_logits,
                "router_aux_loss_coef": cfg.router_aux_loss_coef,
                "rope_parameters": cfg.rope_parameters,
                "use_sliding_window": cfg.use_sliding_window,
                "tie_word_embeddings": cfg.tie_word_embeddings,
                "qkv_bias": cfg.qkv_bias,
            },
        },
        "training": summary,
        "data_contract": {
            "generative_sft_dataset": "data/banking-v2/manifest.json",
            "generative_sft_source": "prepared_banking_v2_bitext_composition",
            "router_eval_dataset": "PolyAI/banking77",
            "router_eval_revision": BANKING_V2_ROUTER_EVAL_REVISION,
            "out_of_domain_stock_response": BANKING_V2_OOD_STOCK_RESPONSE,
        },
        "resume_validation": {
            "enabled": True,
            "checks": [
                "base_model_revision",
                "dataset_revision",
                "converted_state_manifest_sha256",
                "optimizer_scheduler_rng_state",
            ],
        },
        "launch_guard": {
            "requires_flag": "--allow-paid-job",
            "requires_env": "HELLO_SLM_ALLOW_PAID_JOB=banking-v2",
            "current_env_confirmed": os.environ.get("HELLO_SLM_ALLOW_PAID_JOB") == "banking-v2",
        },
        "cloud_command_template": [
            "Submit scripts/banking_v2/cloud_train_banking_moe.py through the "
            "Hugging Face Jobs UV API after packaging this repo and prepared corpus.",
        ],
        "approval_gated_next_actions": [
            "package and upload the local source plus prepared corpus for the ephemeral job",
            "create the private destination model repository",
            "submit the paid single-process RTX PRO 6000 job with a 5-hour timeout",
        ],
    }


def main() -> int:
    args = parse_args()
    plan = build_job_plan(args.config, args.hub_dest)
    print(json.dumps(plan, indent=2, sort_keys=True))

    if not paid_job_confirmed(args):
        print(
            "Dry-run only: paid training requires --allow-paid-job and "
            "HELLO_SLM_ALLOW_PAID_JOB=banking-v2."
        )
        return 0

    raise RuntimeError(
        "Paid cloud execution and training are not implemented in this spec-generator lane. "
        f"The pinned base revision is {BANKING_V2_BASE_REVISION}."
    )


if __name__ == "__main__":
    raise SystemExit(main())
