"""Pseudo-pair filtering for the M-step (Eq. 8, Section 4.3, Appendix F)."""

from __future__ import annotations

import statistics
from collections.abc import Sequence
from dataclasses import asdict, dataclass

FILTER_MODES = ("lqf", "hqs", "dst", "none")


@dataclass
class FilterStats:
    mode: str
    eps: float
    num_pairs: int
    num_kept: int
    num_new_preferred: int
    num_high_conf_better: int  # Delta >= eps
    num_slightly_noisy: int  # -eps < Delta < 0
    num_high_conf_noisy: int  # Delta <= -eps

    def to_dict(self) -> dict:
        return asdict(self)


def filter_pseudo_pairs(
    new_scores: Sequence[float],
    old_scores: Sequence[float],
    mode: str = "lqf",
) -> tuple[list[bool | None], FilterStats]:
    """Decide how each pseudo-pair ``(y_t, y_{t-1})`` enters reward-model training.

    ``new_scores[i] = r_{t-1}(y_t)`` and ``old_scores[i] = r_{t-1}(y_{t-1})`` for prompt i of D_R.
    The margin is ``Delta = new - old`` (Eq. 8) and the threshold ``eps_t`` is the standard
    deviation of ``old_scores`` over D_R (Section 4.3).

    Modes (Appendix F):
        lqf  Low-Quality Filtering, the paper's method: drop ``Delta <= -eps``, keep the rest
             with ``y_t`` preferred (slightly noisy pairs act as regularization).
        hqs  High-Quality Selection: keep only ``Delta >= eps``.
        dst  Direct Self-Training: keep every pair, the higher-scored response is preferred.
        none no filtering ("w/o DF" ablation, Appendix E): every pair, ``y_t`` preferred.

    Returns one decision per prompt -- True: chosen = y_t, False: chosen = y_{t-1} (dst only),
    None: drop -- and summary statistics.
    """
    if mode not in FILTER_MODES:
        raise ValueError(f"unknown filter mode {mode!r}, expected one of {FILTER_MODES}")
    if not new_scores or len(new_scores) != len(old_scores):
        raise ValueError("score lists must be non-empty and aligned")
    eps = statistics.pstdev(old_scores)
    deltas = [n - o for n, o in zip(new_scores, old_scores, strict=True)]

    decisions: list[bool | None] = []
    for d in deltas:
        if mode == "lqf":
            decisions.append(True if d > -eps else None)
        elif mode == "hqs":
            decisions.append(True if d >= eps else None)
        elif mode == "dst":
            decisions.append(None if d == 0 else d > 0)
        else:
            decisions.append(True)

    stats = FilterStats(
        mode=mode,
        eps=eps,
        num_pairs=len(deltas),
        num_kept=sum(x is not None for x in decisions),
        num_new_preferred=sum(x is True for x in decisions),
        num_high_conf_better=sum(d >= eps for d in deltas),
        num_slightly_noisy=sum(-eps < d < 0 for d in deltas),
        num_high_conf_noisy=sum(d <= -eps for d in deltas),
    )
    return decisions, stats
