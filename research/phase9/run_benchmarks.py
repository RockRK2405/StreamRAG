"""Brief §47-§51, §65-§67: baselines A-D vs adaptive, ablations A-G, quality / latency / operation counts.

All systems run every case of eval/dev_adaptive_retrieval through the same session pipeline (one fresh session per
case; multi-turn S-cases keep their session). Quality (Recall@K, Precision@K, MRR, nDCG, coverage) is computed
against the implementer-written section-level gold; grounding (claim support, unsupported rate, citation
precision, gold cited) is what the Phase 7 verifier decided about the extractive answer. Latency = wall time of the
retrieval stage of each turn; CPU = process CPU time in that stage. One run per case, after a warm-up.
NOT REPORTABLE (fixture corpora, implementer labels).

Usage: .venv/bin/python research/phase9/run_benchmarks.py --index-root /tmp/idx9 [--only baselines|ablations]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
from common import ABLATIONS, SYSTEMS, cases, meta, run_case, stack, summarize, write  # noqa: E402

CATS = ["simple", "exact_keyword", "semantic", "constraint", "temporal", "multi_hop", "multi_intent", "contradiction",
        "insufficient", "ambiguous", "contextual_follow_up", "repeated_question", "entity_correction",
        "constraint_change", "late_constraint", "equivalent_repeat", "cross_query_reuse", "follow_up"]


def run(systems: dict, index_root: Path, generation: bool = True) -> dict:
    out = {}
    all_cases = cases()
    for corpus in sorted({c["corpus"] for c in all_cases}):        # warm-up: models, index, NLI
        st = stack(corpus, index_root)
        run_case(st, {"case_id": "warm", "corpus": corpus, "category": "warm",
                      "turns": [{"utterance_id": "u1", "utterance_text": "What is the warm up rule?",
                                 "gold": {"citations": [], "expect": "INSUFFICIENT"}}]}, systems[next(iter(systems))],
                 generation)
    for name, ov in systems.items():
        rows = []
        for c in all_cases:
            rows += run_case(stack(c["corpus"], index_root), c, ov, generation)
        by_cat = {cat: summarize([r for r in rows if r["category"] == cat]) for cat in CATS
                  if any(r["category"] == cat for r in rows)}
        sessions = [r for r in rows if r["case_id"].startswith("S")]
        out[name] = {"overall": summarize(rows), "single_turn": summarize([r for r in rows if not
                                                                           r["case_id"].startswith("S")]),
                     "sessions": summarize(sessions), "by_category": by_cat, "rows": rows}
        o = out[name]["overall"]
        print(f"{name:20} R@5={o['recall@5']} R@10={o['recall@10']} cov={o['coverage']} P@5={o['precision@5']} "
              f"MRR={o['mrr@10']} nDCG={o['ndcg@10']} CSR={o['claim_support']} gold_cited={o['gold_cited']} "
              f"searches={o['searches_total']} emb={o['embeddings_total']} chunks={o['chunks_retrieved_total']} "
              f"avg_k={o['avg_k']} lat_p50={o['retrieval_latency_ms']['p50']} p95={o['retrieval_latency_ms']['p95']}",
              flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index-root", type=Path, required=True)
    ap.add_argument("--only", choices=["baselines", "ablations"], default=None)
    ap.add_argument("--eval", default=None, help="eval/<dir> (default dev_adaptive_retrieval)")
    a = ap.parse_args()
    if a.eval:
        common.EVAL_DIR = a.eval
    if a.only in (None, "baselines"):
        res = run(SYSTEMS, a.index_root)
        name = "baselines.json" if not a.eval else f"baselines_{a.eval}.json"
        write(name, {"meta": meta(k_fixed=5, runs="one run per case", eval_set=f"eval/{common.EVAL_DIR}"),
                     "systems": res})
    if a.only in (None, "ablations"):
        res = run(ABLATIONS, a.index_root)
        write("ablations.json", {"meta": meta(k_fixed=5, runs="one run per case", cumulative=True), "systems": res})


if __name__ == "__main__":
    main()
