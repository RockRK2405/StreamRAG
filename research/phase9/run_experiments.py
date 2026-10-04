"""Brief §50, §51, §65: leave-one-out ablations, interleaved latency test, quality-efficiency curves.

* loo      full adaptive system minus one component at a time (the component's independent contribution); same
           harness as run_benchmarks.py (session pipeline + extractive grounded answers).
* latency  retrieval stage only (generation off), fixed hybrid vs adaptive, R repetitions per case, systems
           interleaved case by case (A, B, A, B, ...) so machine-load drift affects both equally. Reported: per-turn
           median over the repetitions, then p50 / p95 over turns. Single-shot latencies of identical configurations
           differed by ~70 % between runs (results/baselines.json vs ablations.json), hence this design.
* curves   quality (Recall@10, coverage, MRR, nDCG) vs mean retrieval latency and vs retrieval calls per turn, for
           fixed hybrid / BM25 / vector at k in {1, 2, 3, 5, 10, 20} and adaptive variants (budgets, k schedules).

NOT REPORTABLE (fixture corpora, implementer labels). Usage:
  .venv/bin/python research/phase9/run_experiments.py --index-root /tmp/idx9 [--only loo|latency|curves] [--reps 5]
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import SYSTEMS, cases, mean, meta, pct, run_case, stack, summarize, write  # noqa: E402
from run_benchmarks import run as run_systems  # noqa: E402

FULL = {"adaptive_retrieval.enabled": True}
LOO = {"full": FULL,
       "-routing": {**FULL, "adaptive_retrieval.routing": False},
       "-adaptive_k": {**FULL, "adaptive_retrieval.adaptive_k": False},
       "-iterative": {**FULL, "adaptive_retrieval.iterative": False},
       "-claim_driven": {**FULL, "adaptive_retrieval.claim_driven": False},
       "-cache_and_session_reuse": {**FULL, "adaptive_retrieval.cache": False,
                                    "adaptive_retrieval.session_reuse": False},
       "-contradiction": {**FULL, "adaptive_retrieval.contradiction_retrieval": False},
       "-multi_hop": {**FULL, "adaptive_retrieval.multi_hop": False},
       "-temporal": {**FULL, "adaptive_retrieval.temporal": False},
       "-expansion": {**FULL, "adaptive_retrieval.expansion": False}}


def latency(index_root: Path, reps: int) -> dict:
    systems = {"C_fixed_hybrid": SYSTEMS["C_fixed_hybrid"], "ADAPTIVE": SYSTEMS["ADAPTIVE"]}
    per = {n: {} for n in systems}
    rows_keep = {n: [] for n in systems}
    all_cases = cases()
    for c in all_cases[:3]:                                               # warm-up
        for ov in systems.values():
            run_case(stack(c["corpus"], index_root), c, ov, generation=False)
    for c in all_cases:
        st = stack(c["corpus"], index_root)
        for rep in range(reps):
            for name, ov in (systems.items() if rep % 2 == 0 else reversed(list(systems.items()))):
                rows = run_case(st, c, ov, generation=False)
                for r in rows:
                    per[name].setdefault((r["case_id"], r["turn"]), []).append(r)
                if rep == 0:
                    rows_keep[name] += rows
    out = {}
    for name in systems:
        med = {k: statistics.median(x["retrieval_ms"] for x in v if x["retrieval_ms"] is not None)
               for k, v in per[name].items() if any(x["retrieval_ms"] is not None for x in v)}
        cpu = {k: statistics.median(x["retrieval_cpu_ms"] for x in v) for k, v in per[name].items()}
        by_cat = {}
        for (cid, t), v in per[name].items():
            cat = v[0]["category"]
            if (cid, t) in med:
                by_cat.setdefault(cat, []).append(med[(cid, t)])
        out[name] = {"retrieval_latency_ms": pct(med.values()), "retrieval_cpu_ms": pct(cpu.values()),
                     "by_category_p50": {k: pct(v)["p50"] for k, v in sorted(by_cat.items())},
                     "turns": len(med), "reps": reps}
    a, b = out["ADAPTIVE"], out["C_fixed_hybrid"]
    out["ratio_adaptive_over_fixed"] = {q: round(a["retrieval_latency_ms"][q] / b["retrieval_latency_ms"][q], 3)
                                        for q in ("p50", "p95", "mean")}
    return out


CURVE_FIXED = [(f"{m}_k{k}", {"adaptive_retrieval.enabled": False, "streaming.retrieval_mode": m,
                              "streaming.rerank": False, "multi_intent.max_candidates_per_intent": k})
               for m in ("hybrid", "bm25", "dense") for k in (1, 2, 3, 5, 10, 20)]
CURVE_ADAPTIVE = [
    ("adaptive_q1", {**FULL, "adaptive_retrieval.budget": {"max_queries": 1, "max_results": 40, "max_iterations": 1,
                                                            "max_latency_ms": 1500, "max_parallel_tasks": 2,
                                                            "max_hops": 2}}),
    ("adaptive_q2", {**FULL, "adaptive_retrieval.budget": {"max_queries": 2, "max_results": 40, "max_iterations": 2,
                                                            "max_latency_ms": 1500, "max_parallel_tasks": 2,
                                                            "max_hops": 2}}),
    ("adaptive_q3", {**FULL, "adaptive_retrieval.budget": {"max_queries": 3, "max_results": 40, "max_iterations": 3,
                                                            "max_latency_ms": 1500, "max_parallel_tasks": 2,
                                                            "max_hops": 2}}),
    ("adaptive_default_q5", FULL),
    ("adaptive_q8", {**FULL, "adaptive_retrieval.budget": {"max_queries": 8, "max_results": 80, "max_iterations": 5,
                                                            "max_latency_ms": 3000, "max_parallel_tasks": 2,
                                                            "max_hops": 3}}),
    ("adaptive_k5_start", {**FULL, "adaptive_retrieval.initial_k": {"SIMPLE": 5, "MODERATE": 5, "COMPLEX": 5,
                                                                     "MULTI_HOP": 5}}),
    ("adaptive_hybrid_fastpath", {**FULL, "adaptive_retrieval.simple_strategy": "HYBRID"}),
]


def curves(index_root: Path) -> dict:
    out = {}
    all_cases = cases()
    for name, ov in CURVE_FIXED + CURVE_ADAPTIVE:
        if "adaptive_retrieval.budget" in ov:
            from streamrag.config.settings import AdaptiveBudget
            ov = {**ov, "adaptive_retrieval.budget": AdaptiveBudget(**ov["adaptive_retrieval.budget"])}
        rows = []
        for c in all_cases:
            rows += run_case(stack(c["corpus"], index_root), c, ov, generation=False)
        s = summarize(rows)
        out[name] = {k: s[k] for k in ("recall@5", "recall@10", "coverage", "mrr@10", "ndcg@10", "returned_precision",
                                       "searches_per_turn", "embeddings_total", "chunks_retrieved_total", "avg_k")}
        out[name]["latency_mean_ms"] = s["retrieval_latency_ms"]["mean"]
        out[name]["latency_p50_ms"] = s["retrieval_latency_ms"]["p50"]
        print(name, out[name], flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index-root", type=Path, required=True)
    ap.add_argument("--only", choices=["loo", "latency", "curves"], default=None)
    ap.add_argument("--reps", type=int, default=5)
    a = ap.parse_args()
    if a.only in (None, "loo"):
        res = run_systems(LOO, a.index_root)
        write("ablations_loo.json", {"meta": meta(design="leave-one-out from the full system"),
                                     "systems": {k: {kk: vv for kk, vv in v.items() if kk != "rows"}
                                                 for k, v in res.items()}})
    if a.only in (None, "latency"):
        res = latency(a.index_root, a.reps)
        print(res)
        write("latency.json", {"meta": meta(design=f"retrieval stage only, {a.reps} interleaved repetitions per case, "
                                                   "per-turn median"), **res})
    if a.only in (None, "curves"):
        res = curves(a.index_root)
        write("curves.json", {"meta": meta(design="retrieval only, one run per case"), "points": res})


if __name__ == "__main__":
    main()
