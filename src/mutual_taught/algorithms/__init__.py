"""Pure-Python building blocks of Mutual-Taught (no torch, no GPU).

Equation and section numbers refer to arXiv:2506.06292v2.
"""

from mutual_taught.algorithms.filtering import FILTER_MODES, FilterStats, filter_pseudo_pairs
from mutual_taught.algorithms.partition import partition_prompts
from mutual_taught.algorithms.preference import select_preference_pair
from mutual_taught.algorithms.selection import select_checkpoint, win_rate

__all__ = [
    "FILTER_MODES",
    "FilterStats",
    "filter_pseudo_pairs",
    "partition_prompts",
    "select_checkpoint",
    "select_preference_pair",
    "win_rate",
]
