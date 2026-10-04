"""Routing calibration (docs/retrieval/03): which strategy for the simple-query fast path, and the reranking mode.

Calibrated on a set that is NOT the Phase 9 evaluation set: the Phase 7 grounded dev questions (eval/dev_grounded,
gold = cited sections of the required facts) and the Phase 3 fixture retrieval items
(tests/fixtures/eval/fixture_retrieval.jsonl). Some Phase 9 B-cases reuse Phase 7 questions over the same small
corpora - disclosed overlap. Retrieval only (no generation). One run per query after a warm-up. NOT REPORTABLE.

Decision rules (fixed before running):
  simple_strategy = the cheapest strategy (mean retrieval latency) among LEXICAL / FAST_VECTOR / HYBRID whose mean
                    Recall@5 and MRR@10 on the SIMPLE-classified queries are within 0.02 of HYBRID's.
  rerank          = "policy" if reranking COMPLEX / MULTI_HOP needs raises their mean MRR@10 by >= 0.02 at a mean
                    latency cost < 50 ms, else "never" (never "always": every need would pay the cross-encoder).

Usage: .venv/bin/python research/phase9/calibrate_routing.py --index-root /tmp/idx9
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import REPO, meta, mrr, ndcg, pct, recall, stack, with_cfg, write  # noqa: E402

from streamrag.adaptive.controller import AdaptiveRequest  # noqa: E402
from streamrag.adaptive.integration import make_controller  # noqa: E402


def calibration_items() -> list[tuple[str, str, list[str]]]:
    items = []
    for p in sorted((REPO / "eval" / "dev_grounded").glob("G*.json")):
        c = json.loads(p.read_text())
        for t in c["turns"]:
            gold = sorted({f["citation"] for f in t["gold"]["required_facts"]})
            if gold:
                items.append((c["corpus"], t["utterance_text"], gold))
    for line in (REPO / "tests" / "fixtures" / "eval" / "fixture_retrieval.jsonl").read_text().splitlines():
        if line.strip():
            d = json.loads(line)
            items.append(("fixture", d["query"], d["gold"]))
    return items


def run(items, index_root, overrides: dict) -> list[dict]:
    rows = []
    for corpus, q, gold in items:
        st = with_cfg(stack(corpus, index_root), **{"adaptive_retrieval.enabled": True, **overrides})
        ctl = make_controller(st.service, st.cfg)
        t0 = time.perf_counter()
        r = ctl.run(AdaptiveRequest("c1", q, original_text=q))
        ms = (time.perf_counter() - t0) * 1000.0
        keys = list(dict.fromkeys(e.citation for e in r.evidence.items))
        rows.append({"q": q, "corpus": corpus, "complexity": r.analysis.complexity.value, "strategy": r.plan.strategy.value,
                     "recall@5": recall(keys, gold, 5), "mrr@10": mrr(keys, gold), "ndcg@10": ndcg(keys, gold),
                     "ms": round(ms, 3), "searches": r.ops.searches, "embeddings": r.ops.dense_searches,
                     "reranker_calls": r.ops.reranker_calls, "stop": r.state.stop_reason.value})
        ctl.close()
    return rows


def agg(rows) -> dict:
    return {"n": len(rows), "recall@5": round(statistics.fmean(r["recall@5"] for r in rows), 4) if rows else None,
            "mrr@10": round(statistics.fmean(r["mrr@10"] for r in rows), 4) if rows else None,
            "ndcg@10": round(statistics.fmean(r["ndcg@10"] for r in rows), 4) if rows else None,
            "latency_ms": pct(r["ms"] for r in rows), "searches": sum(r["searches"] for r in rows),
            "embeddings": sum(r["embeddings"] for r in rows), "reranker_calls": sum(r["reranker_calls"] for r in rows)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index-root", type=Path, required=True)
    a = ap.parse_args()
    items = calibration_items()
    run(items[:3], a.index_root, {})                                  # warm-up
    simple = {}
    for strat in ("HYBRID", "LEXICAL", "FAST_VECTOR"):
        rows = run(items, a.index_root, {"adaptive_retrieval.simple_strategy": strat,
                                         "adaptive_retrieval.rerank": "never"})
        simple[strat] = {"simple_only": agg([r for r in rows if r["complexity"] == "SIMPLE"]), "all": agg(rows),
                         "rows": rows}
        print(strat, simple[strat]["simple_only"], flush=True)
    h = simple["HYBRID"]["simple_only"]
    ok = [s for s in simple if simple[s]["simple_only"]["recall@5"] >= h["recall@5"] - 0.02
          and simple[s]["simple_only"]["mrr@10"] >= h["mrr@10"] - 0.02]
    chosen = min(ok, key=lambda s: simple[s]["simple_only"]["latency_ms"]["mean"])
    rer = {}
    for mode in ("never", "always"):
        rows = run(items, a.index_root, {"adaptive_retrieval.simple_strategy": chosen,
                                         "adaptive_retrieval.rerank": mode})
        rer[mode] = {"complex": agg([r for r in rows if r["complexity"] in ("COMPLEX", "MULTI_HOP")]),
                     "all": agg(rows), "rows": rows}
        print("rerank", mode, rer[mode]["complex"], flush=True)
    cx_n, cx_a = rer["never"]["complex"], rer["always"]["complex"]
    gain = (cx_a["mrr@10"] or 0) - (cx_n["mrr@10"] or 0) if cx_n["n"] else 0.0
    cost = (cx_a["latency_ms"]["mean"] or 0) - (cx_n["latency_ms"]["mean"] or 0) if cx_n["n"] else 0.0
    rerank_mode = "policy" if cx_n["n"] and gain >= 0.02 and cost < 50 else "never"
    decision = {"simple_strategy": chosen, "candidates_within_tolerance": ok, "rerank": rerank_mode,
                "rerank_complex_mrr_gain": round(gain, 4), "rerank_complex_latency_cost_ms": round(cost, 3)}
    print(decision)
    write("calibration.json", {"meta": meta(calibration_set="eval/dev_grounded + fixture_retrieval.jsonl",
                                            n_items=len(items)), "decision": decision, "simple_path": simple,
                               "rerank": rer})


if __name__ == "__main__":
    main()
