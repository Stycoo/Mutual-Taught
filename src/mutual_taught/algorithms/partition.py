"""Prompt partitions D_1, ..., D_T and D_R (Section 5.1, Algorithm 1 line 2)."""

from __future__ import annotations

import random
from collections.abc import Sequence
from typing import TypeVar

T = TypeVar("T")


def partition_prompts(
    items: Sequence[T],
    names: Sequence[str],
    size: int | None = None,
    mode: str = "sequential",
    seed: int = 42,
) -> dict[str, list[T]]:
    """Split ``items`` into disjoint partitions named ``names``, in that order.

    Args:
        items: the prompts.
        names: partition names, e.g. ``["pm_1", "rm", "pm_2"]``.
        size: prompts per partition; ``None`` splits everything into equal parts.
        mode: ``"sequential"`` keeps the source order, ``"random"`` shuffles first.
        seed: shuffle seed for ``mode="random"``.
    """
    pool = list(items)
    if mode == "random":
        random.Random(seed).shuffle(pool)
    elif mode != "sequential":
        raise ValueError(f"unknown split mode {mode!r}")
    if size is None:
        size = len(pool) // len(names)
    if size * len(names) > len(pool):
        raise ValueError(f"need {size} x {len(names)} prompts, only {len(pool)} available")
    return {name: pool[i * size : (i + 1) * size] for i, name in enumerate(names)}
