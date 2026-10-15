"""Arguments shared by the DPO and reward-model trainers.

The pipeline writes one JSON file per training run with these keys:

    model, data, output_dir, learning_rate, per_device_batch_size, gradient_accumulation_steps,
    num_epochs, lr_scheduler_type, warmup_ratio, max_length, gradient_checkpointing,
    deepspeed (path or null), seed, attn_implementation, bf16 (default true)

plus the trainer-specific keys documented in each module.
"""

from __future__ import annotations

import json
import sys
from typing import Any


def load_args(argv: list[str] | None = None) -> dict[str, Any]:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        raise SystemExit(f"usage: python -m {__package__}.<trainer> ARGS_JSON")
    with open(argv[0]) as f:
        return json.load(f)


def training_kwargs(a: dict[str, Any]) -> dict[str, Any]:
    """Keyword arguments common to ``transformers.TrainingArguments`` and ``trl.DPOConfig``."""
    return dict(
        output_dir=a["output_dir"],
        learning_rate=a["learning_rate"],
        per_device_train_batch_size=a["per_device_batch_size"],
        gradient_accumulation_steps=a["gradient_accumulation_steps"],
        num_train_epochs=a["num_epochs"],
        lr_scheduler_type=a["lr_scheduler_type"],
        warmup_ratio=a["warmup_ratio"],
        bf16=a.get("bf16", True),
        gradient_checkpointing=a["gradient_checkpointing"],
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=10,
        report_to="none",
        seed=a["seed"],
        deepspeed=a.get("deepspeed"),
    )


def model_kwargs(a: dict[str, Any]) -> dict[str, Any]:
    import torch

    return dict(
        dtype=torch.bfloat16 if a.get("bf16", True) else torch.float32,
        attn_implementation=a.get("attn_implementation", "sdpa"),
    )
