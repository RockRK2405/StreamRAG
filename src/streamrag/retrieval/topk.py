"""Deterministic top-k selection: ties broken by row index, never by partition order."""

from __future__ import annotations

import numpy as np


def deterministic_topk(scores: np.ndarray, k: int, eligible: np.ndarray | None = None,
                       min_score: float | None = None, decimals: int = 6) -> list[tuple[int, float]]:
    """Return up to ``k`` (row, score) pairs sorted by score desc, then row asc.

    Scores are rounded to ``decimals`` for ordering only (stabilizes BLAS thread-count jitter); the returned
    score is the unrounded value.
    """
    if scores.size == 0 or k <= 0:
        return []
    keyed = np.round(scores.astype(np.float64), decimals)
    valid = np.ones(scores.shape[0], dtype=bool) if eligible is None else eligible.copy()
    if min_score is not None:
        valid &= keyed >= min_score
    idx = np.flatnonzero(valid)
    if idx.size == 0:
        return []
    vals = keyed[idx]
    if idx.size > k:
        kth = np.partition(vals, idx.size - k)[idx.size - k]   # k-th largest value
        keep = vals >= kth                                     # includes all ties at the boundary
        idx, vals = idx[keep], vals[keep]
    order = np.lexsort((idx, -vals))[:k]
    return [(int(idx[i]), float(scores[idx[i]])) for i in order]
