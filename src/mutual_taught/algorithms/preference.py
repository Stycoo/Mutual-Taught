"""E-step preference pairs (data for Eq. 4) with the length control of Appendix B."""

from __future__ import annotations

from collections.abc import Sequence


def select_preference_pair(
    scores: Sequence[float],
    lengths: Sequence[int],
    length_control: bool = True,
) -> tuple[int, int] | None:
    """Choose ``(chosen, rejected)`` indices among the M responses sampled for one prompt.

    Without length control the highest-reward response is chosen and the lowest-reward one
    rejected. With length control (Appendix B) the chosen response is the *shortest* among
    those with above-average reward; the rejected one is still the lowest-reward response.

    Returns ``None`` when no informative pair exists (fewer than two responses, or no reward gap).
    """
    n = len(scores)
    if n < 2 or n != len(lengths):
        return None
    rejected = min(range(n), key=lambda i: scores[i])
    if length_control:
        mean = sum(scores) / n
        above = [i for i in range(n) if scores[i] > mean]
        if not above:
            return None
        # shortest; ties broken by higher reward
        chosen = min(above, key=lambda i: (lengths[i], -scores[i]))
    else:
        chosen = max(range(n), key=lambda i: scores[i])
    if scores[chosen] <= scores[rejected]:
        return None
    return chosen, rejected
