"""Statistics for the Phase 10 experiments (docs/evaluation/statistics.md).

* describe(): n, mean, median, sample standard deviation, 95 % percentile-bootstrap CI of the mean (seeded).
* paired(): system B vs A on the same samples (pairs with both values present):
    - mean / median difference, 95 % bootstrap CI of the mean difference (resampling pairs);
    - binary metric (all values 0/1): exact McNemar test (binomial test on the discordant pairs);
      continuous metric: Wilcoxon signed-rank test (zero differences dropped, two-sided; scipy);
    - effect size: Cohen's d_z (mean difference / sd of differences) and matched-pairs rank-biserial r.
Assumptions: samples are not independent draws from a population of user questions (they are implementer-written
fixture items), so p-values describe *this* item set only; with n < 20 pairs or < 6 non-zero differences no test is
run (``test: "too_few_pairs"``). No multiple-comparison correction is applied; the report reads p-values together
with effect sizes and CIs, never alone.
"""

from __future__ import annotations

import math
import statistics

import numpy as np

SEED = 20261004
B = 10000


def describe(values, seed: int = SEED) -> dict:
    v = [float(x) for x in values if x is not None and not (isinstance(x, float) and math.isnan(x))]
    if not v:
        return {"n": 0, "mean": None, "median": None, "std": None, "ci95": None}
    out = {"n": len(v), "mean": round(statistics.fmean(v), 4), "median": round(statistics.median(v), 4),
           "std": round(statistics.stdev(v), 4) if len(v) > 1 else None}
    if len(v) >= 2:
        rng = np.random.default_rng(seed)
        arr = np.asarray(v)
        means = arr[rng.integers(0, len(arr), size=(B, len(arr)))].mean(axis=1)
        out["ci95"] = [round(float(np.percentile(means, 2.5)), 4), round(float(np.percentile(means, 97.5)), 4)]
    else:
        out["ci95"] = None
    return out


def paired(a, b, seed: int = SEED, min_pairs: int = 20) -> dict:
    """a, b: dicts sample_id -> value (None = not applicable). Difference = b - a."""
    ids = [k for k in a if k in b and a[k] is not None and b[k] is not None]
    x = np.asarray([float(a[k]) for k in ids])
    y = np.asarray([float(b[k]) for k in ids])
    out: dict = {"n_pairs": len(ids)}
    if not ids:
        return {**out, "test": "no_pairs"}
    d = y - x
    out.update({"mean_a": round(float(x.mean()), 4), "mean_b": round(float(y.mean()), 4),
                "mean_diff": round(float(d.mean()), 4), "median_diff": round(float(np.median(d)), 4)})
    if len(ids) >= 2:
        rng = np.random.default_rng(seed)
        bs = d[rng.integers(0, len(d), size=(B, len(d)))].mean(axis=1)
        out["ci95_diff"] = [round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)]
    sd = float(d.std(ddof=1)) if len(d) > 1 else 0.0
    out["cohens_dz"] = round(float(d.mean()) / sd, 4) if sd > 0 else None
    nz = d[d != 0]
    if len(nz):
        ranks = _ranks(np.abs(nz))
        pos, neg = float(ranks[nz > 0].sum()), float(ranks[nz < 0].sum())
        out["rank_biserial"] = round((pos - neg) / (pos + neg), 4)
    binary = set(np.unique(np.concatenate([x, y]))) <= {0.0, 1.0}
    if len(ids) < min_pairs or len(nz) < 6:
        out["test"] = "too_few_pairs" if len(ids) < min_pairs else "too_few_nonzero_differences"
        if binary:
            out["discordant"] = [int(((x == 1) & (y == 0)).sum()), int(((x == 0) & (y == 1)).sum())]
        return out
    from scipy import stats as st
    if binary:
        b01, b10 = int(((x == 1) & (y == 0)).sum()), int(((x == 0) & (y == 1)).sum())
        out["test"] = "mcnemar_exact"
        out["discordant"] = [b01, b10]
        out["p_value"] = round(float(st.binomtest(min(b01, b10), b01 + b10, 0.5).pvalue), 6) if b01 + b10 else 1.0
    else:
        out["test"] = "wilcoxon_signed_rank"
        out["p_value"] = round(float(st.wilcoxon(nz, zero_method="wilcox", alternative="two-sided").pvalue), 6)
    return out


def _ranks(v: np.ndarray) -> np.ndarray:
    order = np.argsort(v, kind="mergesort")
    ranks = np.empty(len(v))
    i = 0
    while i < len(v):
        j = i
        while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return ranks
