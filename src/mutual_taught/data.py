"""Prompt sets: partitions D_1..D_T and D_R of UltraFeedback, plus the model-selection set D_MS.

Writes ``<out_dir>/<name>.jsonl`` (rows ``{"id", "prompt"}``) for every name in
``data.partition_order`` and ``ms.jsonl``, plus ``stats.json``.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from mutual_taught.algorithms.partition import partition_prompts
from mutual_taught.config import ExperimentConfig
from mutual_taught.modeling import prompt_ids
from mutual_taught.utils.io import mark_done, write_json, write_jsonl

logger = logging.getLogger(__name__)


def unique_prompts(
    prompts: Iterable[str], tokenizer: Any, max_tokens: int, exclude: frozenset[str] = frozenset()
) -> tuple[list[str], int]:
    """Deduplicated, non-empty prompts that fit in ``max_tokens``; also returns how many were too long."""
    seen, kept, too_long = set(exclude), [], 0
    for prompt in prompts:
        prompt = prompt.strip()
        if not prompt or prompt in seen:
            continue
        seen.add(prompt)
        if len(prompt_ids(tokenizer, prompt)) > max_tokens:
            too_long += 1
            continue
        kept.append(prompt)
    return kept, too_long


def prepare_prompts(cfg: ExperimentConfig, out_dir: str | Path) -> dict:
    from datasets import load_dataset
    from transformers import AutoTokenizer

    out_dir = Path(out_dir)
    d = cfg.data
    tokenizer = AutoTokenizer.from_pretrained(cfg.models.policy)

    ms_raw = load_dataset(d.dataset, split=d.ms_split)[d.prompt_field]
    ms, ms_long = unique_prompts(ms_raw, tokenizer, d.max_prompt_tokens)
    ms = ms[: d.ms_size]

    train_raw = load_dataset(d.dataset, split=d.train_split)[d.prompt_field]
    train, train_long = unique_prompts(train_raw, tokenizer, d.max_prompt_tokens, exclude=frozenset(ms))
    parts = partition_prompts(train, d.partition_order, d.partition_size, d.split_mode, cfg.seed)

    for name, prompts in {"ms": ms, **parts}.items():
        write_jsonl(out_dir / f"{name}.jsonl", ({"id": f"{name}-{i}", "prompt": p} for i, p in enumerate(prompts)))
    stats = {
        "sizes": {"ms": len(ms), **{name: len(p) for name, p in parts.items()}},
        "train_unique_prompts": len(train),
        "dropped_too_long": {"train": train_long, "ms": ms_long},
    }
    write_json(out_dir / "stats.json", stats)
    mark_done(out_dir)
    logger.info("prompt sets written to %s: %s", out_dir, stats)
    return stats
