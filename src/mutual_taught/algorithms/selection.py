"""Model selection for the E-step (Eq. 6-7) and the early-stop threshold tau (Appendix H)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence


def win_rate(candidate_scores: Sequence[float], previous_scores: Sequence[float]) -> float:
    """Eq. 6: fraction of D_MS prompts with ``r_{t-1}(y^k_t) > r_{t-1}(y_{t-1})`` (strict)."""
    if not candidate_scores or len(candidate_scores) != len(previous_scores):
        raise ValueError("score lists must be non-empty and aligned")
    wins = sum(c > p for c, p in zip(candidate_scores, previous_scores, strict=True))
    return wins / len(candidate_scores)


def select_checkpoint(
    win_rates: Mapping[str, float],
    order: Sequence[str],
    tau: float,
) -> tuple[str, float, bool]:
    """Eq. 7: the checkpoint with the highest win rate (earliest in ``order`` on ties).

    Returns ``(name, win_rate, halt)``; ``halt`` is True when ``max_k w^k_t < tau``, i.e. the
    iteration stops and the previous policy and reward model are kept.
    """
    if not order:
        raise ValueError("no checkpoints to select from")
    best = max(order, key=lambda k: (win_rates[k], -order.index(k)))
    return best, win_rates[best], win_rates[best] < tau
