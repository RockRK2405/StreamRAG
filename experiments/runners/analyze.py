"""Builds the Phase 10 analysis artefacts from the stored experiment results (no system is run here).

Outputs
  experiments/results/tables.md              every table of PHASE_10_EVALUATION_REPORT.md (numbers copied from here)
  experiments/results/final_comparison.json  brief §69
  experiments/results/ablation_table.json    brief §70
  experiments/results/dashboard.html          static research dashboard (tables + charts)
  experiments/plots/*.svg                    brief §64 charts
  experiments/datasets/human_eval_sheet.csv  blinded sheet for the (not run) human evaluation
A missing / NOT RUN / FAILED experiment yields "NOT MEASURED" cells, never a number.

Usage: .venv/bin/python experiments/runners/analyze.py
"""

from __future__ import annotations

import csv
import html
import json
import random
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from streamrag.evaluation import plots as P  # noqa: E402

RES = REPO / "experiments" / "results"
PLOTS = REPO / "experiments" / "plots"
NM = "NOT MEASURED"
LABEL = {"naive_rag": "A naive RAG", "hybrid_rag": "B hybrid RAG", "hybrid_rerank": "C hybrid + rerank",
         "streaming_fixed": "D streaming, fixed retrieval", "adaptive_rag": "Adaptive RAG (batch)",
         "full_system": "Full system", "full_system_batch": "Full system, batch mode",
         "full_system_queryonly": "S - claim-driven", "full_system_memoryless": "S - session memory",
         "full_system_fullrestart": "S - delta retrieval", "full_system_nocancel": "S - cancellation",
         "adaptive_rag_unverified": "S_batch - verification", "adaptive_rag_nocitval": "S_batch - citation validation",
         "adaptive_rag_rerank": "S_batch + reranking", "adaptive_rag_queryonly": "query-driven (no claim slots)",
         "pipeline_fixed": "session pipeline, fixed retrieval", "adaptive_rag_nocontra": "adaptive - contradiction/temporal",
         "cancel_on": "cancellation on", "cancel_off": "cancellation off", "cancel_on_remote": "on, remote +300 ms",
         "cancel_off_remote": "off, remote +300 ms", "topk_3": "fixed k=3", "topk_5": "fixed k=5", "topk_10": "fixed k=10",
         "topk_20": "fixed k=20", "adaptive_x": "adaptive", "adaptive_x_fixedk": "adaptive, k fixed 5",
         "adaptive_x_single": "adaptive, single pass", "adaptive_x_nohop": "adaptive, no multi-hop",
         "cache_none": "no cache", "cache_session": "query / session caches", "delta_full_restart": "full re-retrieval"}


def load(exp: str) -> dict | None:
    p = RES / exp / "results.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    return d if d.get("status") == "COMPLETED" else None


def rows(exp: str) -> list[dict]:
    p = RES / exp / "samples.jsonl"
    return [json.loads(x) for x in p.read_text().splitlines()] if p.exists() else []


def m(agg: dict | None, dotted: str, stat: str = "mean"):
    if agg is None:
        return None
    if "." not in dotted:
        return agg.get(dotted)
    g, k = dotted.split(".", 1)
    v = (agg.get(g) or {}).get(k)
    return v.get(stat) if isinstance(v, dict) else v


def f(v, d: int = 3) -> str:
    if v is None:
        return NM
    if isinstance(v, float):
        return f"{v:.{d}f}"
    return str(v)


def table(head: list[str], body: list[list]) -> str:
    out = ["| " + " | ".join(head) + " |", "|" + "|".join("---" for _ in head) + "|"]
    out += ["| " + " | ".join(c if isinstance(c, str) else f(c) for c in r) + " |" for r in body]
    return "\n".join(out)


def systems(res: dict | None, split: str = "test") -> dict:
    return ((res or {}).get("splits", {}).get(split) or {}).get("systems", {})


QUALITY = [("Recall@1", "retrieval.recall@1"), ("Recall@3", "retrieval.recall@3"), ("Recall@5", "retrieval.recall@5"),
           ("Recall@10", "retrieval.recall@10"), ("Precision@5", "retrieval.precision@5"), ("MRR", "retrieval.mrr"),
           ("nDCG@10", "retrieval.ndcg@10")]
EVID = [("evidence recall", "evidence.evidence_recall"), ("evidence precision", "evidence.evidence_precision"),
        ("evidence coverage", "evidence.evidence_coverage"), ("unsupported evidence", "evidence.unsupported_evidence"),
        ("stale / contradictory evidence", "evidence.contradictory_evidence"), ("redundancy", "evidence.redundancy"),
        ("source diversity", "evidence.source_diversity")]
GEN = [("answer correct", "generation.answer_correct"), ("completeness", "generation.completeness"),
       ("forbidden value asserted", "generation.forbidden_hit"), ("insufficiency handled", "generation.insufficiency_ok"),
       ("conflict reported", "generation.conflict_reported"), ("faithfulness (verifier)", "generation.faithfulness"),
       ("groundedness (verifier)", "generation.groundedness")]
CLAIM = [("claims / answer", "claims.claims"), ("claim support (verifier)", "claims.claim_support_rate"),
         ("unsupported (verifier)", "claims.unsupported_claim_rate"),
         ("contradicted (verifier)", "claims.contradicted_claim_rate"), ("claim coverage", "claims.claim_coverage")]
CIT = [("citations / answer", "citation.citations"), ("citation precision", "citation.citation_precision"),
       ("citation recall", "citation.citation_recall"), ("citation completeness", "citation.citation_completeness"),
       ("citation entailment", "citation.citation_entailment"), ("source validity", "citation.source_validity"),
       ("position correct", "citation.position_correct")]
HAL = [("unsupported claim rate", "hallucination.unsupported_claim_rate"),
       ("hallucinated claim rate", "hallucination.hallucinated_claim_rate"),
       ("citation-less fact rate", "hallucination.citationless_fact_rate"),
       ("grounding failure (turns)", "hallucination.grounding_failure")]
EFF = [("retrieval calls / query", "efficiency.retrieval_calls"), ("embeddings / query", "efficiency.embeddings"),
       ("reranker calls / query", "efficiency.reranker_calls"), ("chunks retrieved / query", "efficiency.chunks_retrieved"),
       ("avg k", "efficiency.avg_k"), ("iterations / query", "efficiency.iterations"),
       ("expansions / query", "efficiency.expansions"), ("documents / query", "efficiency.documents_retrieved"),
       ("evidence reused / query", "efficiency.evidence_reused"), ("cache hit rate", "efficiency.cache_hit"),
       ("LLM calls / query", "efficiency.llm_calls"), ("prompt tokens / query", "efficiency.prompt_tokens"),
       ("output tokens / query", "efficiency.output_tokens")]
LATS = ["ttft", "ttfe", "ttfa", "ttva", "total", "ttfe_after_end", "ttfa_after_end", "ttva_after_end",
        "total_after_end", "generation_ms", "verification_ms"]


def metric_table(res, names: list[str], spec: list[tuple[str, str]], split: str = "test") -> str:
    sy = systems(res, split)
    return table(["metric"] + [LABEL.get(n, n) for n in names], [[lab] + [m(sy.get(n), d) for n in names]
                                                                  for lab, d in spec])


def latency_table(res, names: list[str], split: str = "test") -> str:
    sy = systems(res, split)
    body = []
    for k in LATS:
        row = [k]
        any_ = False
        for n in names:
            p = ((sy.get(n) or {}).get("latency") or {}).get(k)
            if p and p.get("n"):
                any_ = True
                row.append(f"{p['p50']:.0f} / {f(p['p90'], 0)} / {f(p['p95'], 0)} / {f(p['p99'], 0)} (n={p['n']})")
            else:
                row.append(NM)
        if any_:
            body.append(row)
    return table(["latency ms: p50 / p90 / p95 / p99"] + [LABEL.get(n, n) for n in names], body)


def comparisons(res, split: str = "test") -> str:
    comps = ((res or {}).get("comparisons") or {}).get(split) or {}
    body = []
    for name, metrics in comps.items():
        for metric, s in metrics.items():
            if s.get("n_pairs", 0) == 0:
                continue
            body.append([name, metric, s["n_pairs"], s.get("mean_a"), s.get("mean_b"), s.get("mean_diff"),
                         f"[{s['ci95_diff'][0]:.3f}, {s['ci95_diff'][1]:.3f}]" if s.get("ci95_diff") else NM,
                         s.get("test"), s.get("p_value"), s.get("cohens_dz"), s.get("rank_biserial")])
    return table(["comparison (B vs A)", "metric", "pairs", "mean A", "mean B", "diff B-A", "95% CI diff", "test",
                  "p", "d_z", "rank-biserial"], body)


def by_category(res, names: list[str], metric: str = "generation.answer_correct", field: str = "by_category",
                split: str = "test") -> str:
    sy = systems(res, split)
    cats = sorted({c for n in names for c in ((sy.get(n) or {}).get(field) or {})})
    body = [[c, (sy.get(names[0]) or {}).get(field, {}).get(c, {}).get("samples")] +
            [((sy.get(n) or {}).get(field, {}).get(c) or {}).get(metric) for n in names] for c in cats]
    return table([field.replace("by_", ""), "n"] + [LABEL.get(n, n) for n in names], body)


SPECIAL = [("CONTRADICTORY, conflicting notices: both values reported", "CONTRADICTORY", "generation.conflict_reported"),
           ("CONTRADICTORY, superseded document: current value, no stale value", "CONTRADICTORY",
            "generation.answer_correct"),
           ("TEMPORAL: stale value asserted (lower is better)", "TEMPORAL", "generation.forbidden_hit"),
           ("INSUFFICIENT_EVIDENCE: abstained without inventing a value", "INSUFFICIENT_EVIDENCE",
            "generation.insufficiency_ok"),
           ("AMBIGUOUS: abstained (descriptive only - no answer label exists)", "AMBIGUOUS", "generation.abstained")]


def special_cases(rs: list[dict], names: list[str]) -> str:
    """Per-category outcomes the generic answer_correct cannot express (mean over the turns where the label exists)."""
    body = []
    for lab, cat, metric in SPECIAL:
        g, k = metric.split(".")
        row = [lab]
        ns = []
        for n in names:
            v = [x["metrics"][g].get(k) for x in rs if x["system_variant"] == n and x["query_category"] == cat
                 and x["metrics"][g].get(k) is not None]
            ns.append(len(v))
            row.append(sum(v) / len(v) if v else None)
        body.append([row[0], max(ns) if ns else 0] + row[1:])
    return table(["outcome", "n"] + [LABEL.get(n, n) for n in names], body)


def main() -> None:
    PLOTS.mkdir(parents=True, exist_ok=True)
    md = ["# Phase 10 result tables (generated by experiments/runners/analyze.py)", "",
          "> Fixture corpora, implementer labels - NOT REPORTABLE. test = held-out transit corpus (87 turns).", ""]
    plots: dict[str, str] = {}
    e1 = load("EXP01_baselines")
    base = ["naive_rag", "hybrid_rag", "hybrid_rerank", "streaming_fixed", "adaptive_rag", "full_system"]
    md += ["## EXP01 Baseline comparison - retrieval", "", metric_table(e1, base, QUALITY), "",
           "## EXP01 - evidence", "", metric_table(e1, base, EVID), "",
           "## EXP01 - answer quality (labels, model-free unless marked)", "", metric_table(e1, base, GEN), "",
           "## EXP01 - claims", "", metric_table(e1, base, CLAIM), "", "## EXP01 - citations", "",
           metric_table(e1, base, CIT), "", "## EXP01 - hallucination", "", metric_table(e1, base, HAL), "",
           "## EXP01 - efficiency / cost", "", metric_table(e1, base, EFF), "", "## EXP01 - latency", "",
           latency_table(e1, base), "", "## EXP01 - answer correctness by query type", "", by_category(e1, base), "",
           "## EXP01 - answer correctness by difficulty", "", by_category(e1, base, field="by_difficulty"), "",
           "## EXP01 - Recall@5 by query type", "", by_category(e1, base, "retrieval.recall@5"), "",
           "## EXP01 - category-specific outcomes", "", special_cases(rows("EXP01_baselines"), base), "",
           "## EXP01 - paired comparisons", "", comparisons(e1), ""]
    sy1 = systems(e1)
    if sy1:
        plots["01_retrieval_quality"] = P.hbar("Retrieval quality: Recall@5 (test split)",
                                               [(LABEL[n], m(sy1.get(n), "retrieval.recall@5")) for n in base],
                                               subtitle="share of gold sections in the top 5 evidence items; "
                                                        f"{m(sy1.get('full_system'), 'retrieval.recall@5', 'n')} turns "
                                                        "with gold sections", vmax=1.0)
        plots["01b_answer_correct"] = P.hbar("Answer correctness (test split)",
                                             [(LABEL[n], m(sy1.get(n), "generation.answer_correct")) for n in base],
                                             subtitle="all expected key facts stated, no forbidden value; real LLM "
                                                      "qwen3:4b", vmax=1.0)
        plots["02_latency_ttva"] = P.hbar("Time to validated answer after the utterance ends, p50",
                                          [(LABEL[n], m(sy1.get(n), "latency.ttva_after_end", "p50")) for n in base],
                                          fmt="{:,.0f}", unit=" ms", subtitle="wall clock, one run per system")
        plots["06_retrieval_calls"] = P.hbar("Retrieval calls per query",
                                             [(LABEL[n], m(sy1.get(n), "efficiency.retrieval_calls")) for n in base],
                                             subtitle="searches executed (one lexical and / or dense search = one call)")
        pts = [(LABEL[n], m(sy1.get(n), "latency.ttva_after_end", "p50"), m(sy1.get(n), "generation.answer_correct"))
               for n in base]
        pts = [p for p in pts if p[1] is not None and p[2] is not None]
        plots["09_quality_latency_frontier"] = P.scatter(
            "Quality vs latency", pts, "TTVA after utterance end, p50 (ms)", "answer correctness",
            subtitle="lower-right is worse; proposed systems in orange", highlight={LABEL["adaptive_rag"], LABEL["full_system"]})
        pts2 = [(LABEL[n], m(sy1.get(n), "efficiency.retrieval_calls"), m(sy1.get(n), "generation.answer_correct"))
                for n in base]
        plots["09b_quality_compute_frontier"] = P.scatter(
            "Quality vs retrieval operations", [p for p in pts2 if p[1] is not None and p[2] is not None],
            "retrieval calls per query", "answer correctness", highlight={LABEL["adaptive_rag"], LABEL["full_system"]})
        pts3 = [(LABEL[n], m(sy1.get(n), "efficiency.llm_calls"), m(sy1.get(n), "generation.answer_correct"))
                for n in base]
        plots["09c_quality_llm_frontier"] = P.scatter(
            "Quality vs LLM calls", [p for p in pts3 if p[1] is not None and p[2] is not None],
            "LLM calls per query", "answer correctness", highlight={LABEL["adaptive_rag"], LABEL["full_system"]})
    # ---------------------------------------------------------------- per-experiment sections
    sections = [("EXP02_adaptive_topk", ["topk_3", "topk_5", "topk_10", "topk_20", "adaptive_x_fixedk", "adaptive_x"],
                 QUALITY[2:] + [("evidence precision", "evidence.evidence_precision"),
                                ("answer correct (extractive)", "generation.answer_correct")] + EFF[:5]),
                ("EXP03_claim_driven", ["adaptive_rag_queryonly", "adaptive_rag"],
                 CLAIM + GEN[:2] + HAL[:2] + [("Recall@5", "retrieval.recall@5")] + EFF[:2]),
                ("EXP04_cache", ["cache_none", "cache_session", "adaptive_x"],
                 EFF[:2] + [("cache hit rate", "efficiency.cache_hit"), ("evidence reused / query", "efficiency.evidence_reused"),
                            ("answer correct (extractive)", "generation.answer_correct"), ("Recall@5", "retrieval.recall@5")]),
                ("EXP05_delta", ["delta_full_restart", "adaptive_x"],
                 EFF[:2] + [("evidence reused / query", "efficiency.evidence_reused"),
                            ("answer correct (extractive)", "generation.answer_correct"), ("Recall@5", "retrieval.recall@5")]),
                ("EXP06_cancellation", ["cancel_off", "cancel_on", "cancel_off_remote", "cancel_on_remote"],
                 GEN[:2] + EFF[:2] + [("LLM calls / query", "efficiency.llm_calls")]),
                ("EXP07_streaming", ["full_system_batch", "full_system"], GEN[:2]),
                ("EXP08_iterative", ["adaptive_x_single", "adaptive_x"],
                 QUALITY[2:] + [("evidence coverage", "evidence.evidence_coverage"),
                                ("answer correct (extractive)", "generation.answer_correct")] + EFF[:2]),
                ("EXP09_multihop", ["topk_5", "adaptive_x_nohop", "adaptive_x"],
                 [("Recall@5", "retrieval.recall@5"), ("evidence recall", "evidence.evidence_recall"),
                  ("evidence coverage", "evidence.evidence_coverage"), ("answer correct (extractive)",
                                                                         "generation.answer_correct")] + EFF[:2]),
                ("EXP10_contradiction", ["hybrid_rerank", "pipeline_fixed", "adaptive_rag_nocontra", "adaptive_rag"],
                 GEN[:5] + [("stale / contradictory evidence", "evidence.contradictory_evidence")] + EFF[:1]),
                ("ABLATION_runtime", ["full_system", "streaming_fixed", "full_system_queryonly", "full_system_memoryless",
                                      "full_system_fullrestart", "full_system_nocancel"],
                 GEN[:3] + CLAIM[1:3] + CIT[1:2] + HAL[1:2] + QUALITY[2:3] + EFF[:2] + EFF[10:11]),
                ("ABLATION_answer_stage", ["adaptive_rag", "adaptive_rag_unverified", "adaptive_rag_nocitval",
                                           "adaptive_rag_rerank"],
                 GEN[:3] + CLAIM[:3] + CIT[1:4] + HAL + EFF[2:3] + EFF[10:11])]
    for exp, names, spec in sections:
        res = load(exp)
        md += [f"## {exp}", ""]
        if res is None:
            st = json.loads((RES / exp / "results.json").read_text()).get("status") if (RES / exp / "results.json").exists() \
                else "NOT RUN"
            md += [f"Status: {st} - {NM}", ""]
            continue
        for split in res["splits"]:
            md += [f"### split: {split}", "", metric_table(res, names, spec, split), "",
                   latency_table(res, names, split), "", comparisons(res, split), ""]
        if exp in ("EXP09_multihop", "EXP10_contradiction", "EXP04_cache"):
            md += [by_category(res, names), ""]
    # ---------------------------------------------------------------- charts from other experiments
    e7 = load("EXP07_streaming")
    if e7:
        r7 = rows("EXP07_streaming")
        for key, name in (("ttfe", "03"), ("ttfa", "04"), ("ttva", "05")):
            groups = [(LABEL[n], [x["metrics"]["latency"].get(key) for x in r7 if x["system_variant"] == n
                                  and x["metrics"]["latency"].get(key) is not None])
                      for n in ("full_system_batch", "full_system")]
            plots[f"{name}_{key}_distribution"] = P.strip(f"{key.upper()} distribution: batch vs streaming", groups,
                                                         subtitle="ms from the first transcript chunk; one dot per turn")
        curve = streaming_curve(r7)
        if curve:
            plots["11_streaming_curve"] = curve
    e4 = load("EXP04_cache")
    if e4:
        s4 = systems(e4)
        plots["07_cache_hit_rate"] = P.hbar("Cache hit rate (conversation turns)",
                                            [(LABEL[n], m(s4.get(n), "efficiency.cache_hit"))
                                             for n in ("cache_none", "cache_session", "adaptive_x")], vmax=1.0,
                                            subtitle="needs answered without any search")
    e6 = load("EXP06_cancellation")
    if e6:
        r6 = rows("EXP06_cancellation")
        names6 = ("cancel_off", "cancel_on", "cancel_off_remote", "cancel_on_remote")
        items = [(LABEL[n], waste(r6, n)[0]) for n in names6]
        plots["08_cancellation_worker_time"] = P.hbar(
            "Total worker time for the same overlapping turns (sum)", items, fmt="{:,.0f}", unit=" ms",
            subtitle="retrieval + generation + verification task execution, 15 turns; lower = less work")
        md += ["## EXP06 work done for the same inputs", "",
               "> Same sessions, same inputs. 'Total worker ms' is the fair comparison. 'Wasted' counts only tasks of "
               "queries / answers that ended cancelled or stale; without cancellation, superseded work runs to "
               "completion and is not flagged, so 'wasted' under-counts waste in the no-cancellation variants.", "",
               table(["variant", "total worker ms", "of which wasted (flagged)", "useful", "wasted generation ms",
                      "LLM calls (sum)", "cancelled tasks", "stale discarded"],
                     [[LABEL[n]] + waste(r6, n) for n in names6]), "",
               "## EXP06 final vs superseded turns", "",
               "> Added after the first EXP06 results were seen (disclosed in the report): in the overlap protocol the "
               "first question of a correction session is superseded 400 ms after it ends, so cancelling its answer is "
               "the intended behaviour. 'Final turns' = the last turn of each session (the corrected question, plus "
               "the single-turn streamed corrections).", "",
               table(["variant", "final turns", "final: answer correct", "final: TTVA after end p50 ms",
                      "final: TTVA after end max ms", "superseded first turns", "superseded: answered"],
                     [[LABEL[n]] + final_turns(r6, n) for n in names6]), ""]
    err_rows = rows("EXP01_baselines") + [x for e in ("ABLATION_runtime", "ABLATION_answer_stage")
                                          for x in rows(e) if x["system_variant"] not in base]
    err_names = list(dict.fromkeys(x["system_variant"] for x in err_rows))
    if err_rows:
        md += ["## Error categories (turns with the category; a turn can have several; test split)", "",
               error_table(err_rows, err_names), "", "## Error budget / reliability summary (brief §67)", "",
               error_budget(err_rows, err_names), ""]
    ab = load("ABLATION_runtime")
    ab2 = load("ABLATION_answer_stage")
    abl = ablation_table(ab, ab2)
    (RES / "ablation_table.json").write_text(json.dumps(abl, indent=2))
    md += ["## Final ablation table (brief §70)", "", table(
        ["system", "harness", "quality: answer correct", "grounding: claim support (verifier)", "hallucinated claim rate",
         "latency: TTVA after end p50 / p95 ms", "retrieval calls / query", "LLM calls / query",
         "reliability: failed + incomplete turns"],
        [[r["system"], r["harness"], r["quality"], r["grounding"], r["hallucinated"], r["latency"], r["retrieval_calls"],
          r["llm_calls"], r["reliability"]] for r in abl]), ""]
    if abl:
        plots["10_ablation"] = P.hbar("Ablation: answer correctness by variant",
                                      [(r["system"], r["quality"]) for r in abl], vmax=1.0,
                                      subtitle="runtime block vs full system; answer-stage block vs batch system")
    fc = final_comparison(e1)
    (RES / "final_comparison.json").write_text(json.dumps(fc, indent=2))
    md += ["## Final comparison table (brief §69)", "",
           table(["metric", "Naive RAG", "Hybrid", "Adaptive RAG", "Full system"],
                 [[r["metric"]] + [r[c] for c in ("naive_rag", "hybrid_rag", "adaptive_rag", "full_system")]
                  for r in fc]), ""]
    rob = RES / "ROBUSTNESS" / "results.json"
    if rob.exists():
        d = json.loads(rob.read_text())["summary"]
        md += ["## Robustness (brief §24)", "", table(["scenario", "runs", "recovery", "degraded success",
                                                       "incorrect answer", "propagation"],
                                                      [[k, v["runs"], v["recovery_rate"], v["degraded_success_rate"],
                                                        v["incorrect_answer_rate"], v["failure_propagation_rate"]]
                                                       for k, v in d.items()]), ""]
    vv = RES / "VERIFIER_validation" / "results.json"
    if vv.exists():
        d = json.loads(vv.read_text())
        md += ["## Claim verification accuracy (perturbations, labels by construction)", "",
               table(["n", "accuracy", "precision (SUPPORTED)", "recall (SUPPORTED)", "false acceptance"],
                     [[d["n"], d["accuracy"], d["precision_supported"], d["recall_supported"], d["false_acceptance_rate"]]]),
               "", table(["perturbation", "n", "judged SUPPORTED"], [[k, v["n"], v["judged_supported"]]
                                                                   for k, v in d["by_kind"].items()]), ""]
    for name, svg in plots.items():
        (PLOTS / f"{name}.svg").write_text(svg)
    (RES / "tables.md").write_text("\n".join(md) + "\n")
    if e1:
        human_sheet(rows("EXP01_baselines"))
        qualitative(rows("EXP01_baselines"))
    dashboard(md, plots)
    print("\n".join(md)[:3000])
    print("plots:", sorted(plots))


def waste(r6, n):
    xs = [x for x in r6 if x["system_variant"] == n and x.get("waste")]
    s = lambda k: round(sum(x["waste"].get(k) or 0 for x in xs), 1) if xs else None  # noqa: E731
    canc = sum((x.get("ops_raw") or {}).get("cancelled_tasks") or 0 for x in xs) if xs else None
    llm = sum((x.get("ops_raw") or {}).get("llm_calls") or 0 for x in xs) if xs else None
    total = round(s("wasted_exec_ms") + s("useful_exec_ms"), 1) if xs else None
    return [total, s("wasted_exec_ms"), s("useful_exec_ms"), s("wasted_generation_ms"), llm, canc,
            s("stale_discarded")]


CATS = ("RETRIEVAL_FAILURE", "QUERY_ANALYSIS_FAILURE", "ENTITY_FAILURE", "MEMORY_FAILURE", "EVIDENCE_FAILURE",
        "CLAIM_FAILURE", "GENERATION_FAILURE", "CITATION_FAILURE", "LATENCY_FAILURE", "ORCHESTRATION_FAILURE")


def error_table(rs: list[dict], names: list[str]) -> str:
    body = []
    for c in CATS + ("(no error category)",):
        row = [c]
        for n in names:
            mine = [x for x in rs if x["system_variant"] == n]
            row.append(sum(1 for x in mine if (c in x.get("error_categories", [])) or
                           (c.startswith("(") and not x.get("error_categories"))))
        body.append(row)
    return table(["category"] + [LABEL.get(n, n) for n in names], body)


def error_budget(rs: list[dict], names: list[str]) -> str:
    """Reliability summary: failures by pipeline stage, counted over turns (brief §67)."""
    spec = [("turns", lambda x: True),
            ("sample failed (exception)", lambda x: x["status"] != "ok"),
            ("no retrieval for an answerable turn", lambda x: bool(x["expected"]["evidence"])
             and not x["evidence_citations"]),
            ("retrieval failure (no gold section retrieved)", lambda x: "RETRIEVAL_FAILURE" in x["error_categories"]
             or "MEMORY_FAILURE" in x["error_categories"]),
            ("empty answer", lambda x: not (x["answer"] or "").strip()),
            ("generation fallback (LLM error / timeout -> extractive)", lambda x: "generation_fallback"
             in (x.get("error") or "")),
            ("validation failure (unsupported claim kept)", lambda x: "CLAIM_FAILURE" in x["error_categories"]),
            ("citation failure", lambda x: "CITATION_FAILURE" in x["error_categories"]),
            ("timeout: no validated answer (runtime turns)", lambda x: x["metrics"]["latency"].get("ttva") is None
             and "ttft" in x["metrics"]["latency"]),
            ("turn not completed", lambda x: (x.get("error") or "").startswith("turn_not_completed"))]
    body = [[lab] + [sum(1 for x in rs if x["system_variant"] == n and fn(x)) for n in names] for lab, fn in spec]
    return table(["reliability (turn counts)"] + [LABEL.get(n, n) for n in names], body)


def final_turns(r6: list[dict], n: str) -> list:
    data = {x["sample_id"]: x for x in (json.loads(y) for y in (REPO / "experiments" / "datasets" / "streamrag_eval_v1"
                                                                / "test.jsonl").read_text().splitlines())}
    last: dict[str, int] = {}
    for x in data.values():
        last[x["session_id"]] = max(last.get(x["session_id"], 0), x["turn_index"])
    mine = [x for x in r6 if x["system_variant"] == n]
    fin = [x for x in mine if data[x["sample_id"]]["turn_index"] == last[data[x["sample_id"]]["session_id"]]]
    sup = [x for x in mine if x not in fin]
    ac = [x["metrics"]["generation"]["answer_correct"] for x in fin if x["metrics"]["generation"]["answer_correct"]
          is not None]
    tv = sorted(x["metrics"]["latency"]["ttva_after_end"] for x in fin
                if x["metrics"]["latency"].get("ttva_after_end") is not None)
    return [len(fin), sum(ac) / len(ac) if ac else None, round(statistics.median(tv), 1) if tv else None,
            tv[-1] if tv else None, len(sup), sum(1 for x in sup if (x["answer"] or "").strip())]


def ablation_table(ab, ab2) -> list[dict]:
    out = []
    for res, names, harness in ((ab, ["full_system", "streaming_fixed", "full_system_queryonly", "full_system_memoryless",
                                      "full_system_fullrestart", "full_system_nocancel"], "streaming runtime"),
                                (ab2, ["adaptive_rag", "adaptive_rag_unverified", "adaptive_rag_nocitval",
                                       "adaptive_rag_rerank"], "batch pipeline")):
        sy = systems(res)
        for n in names:
            a = sy.get(n)
            if a is None:
                out.append({"system": LABEL[n], "harness": harness, "quality": None, "grounding": None,
                            "hallucinated": None, "latency": NM, "retrieval_calls": None, "llm_calls": None,
                            "reliability": NM})
                continue
            lat = a["latency"].get("ttva_after_end") or {}
            inc = (a.get("error_categories") or {}).get("ORCHESTRATION_FAILURE", 0)
            name = {"full_system": "Full system (S)", "adaptive_rag": "Adaptive RAG, batch (S_batch)",
                    "streaming_fixed": "S - adaptive retrieval (= baseline D)"}.get(n, LABEL[n])
            out.append({"system": name, "harness": harness,
                        "quality": m(a, "generation.answer_correct"), "grounding": m(a, "claims.claim_support_rate"),
                        "hallucinated": m(a, "hallucination.hallucinated_claim_rate"),
                        "latency": f"{f(lat.get('p50'), 0)} / {f(lat.get('p95'), 0)}" if lat else NM,
                        "retrieval_calls": m(a, "efficiency.retrieval_calls"), "llm_calls": m(a, "efficiency.llm_calls"),
                        "reliability": f"{a['failed']} failed, {inc} with orchestration failure of {a['samples']}"})
    return out


def final_comparison(e1) -> list[dict]:
    sy = systems(e1)
    cols = ("naive_rag", "hybrid_rag", "adaptive_rag", "full_system")

    def lat(n, k):
        p = ((sy.get(n) or {}).get("latency") or {}).get(k) or {}
        return f"{p['p50']:,.0f} / {f(p.get('p95'), 0)}" if p.get("n") else NM
    spec = [("Recall@5", lambda n: f(m(sy.get(n), "retrieval.recall@5"))),
            ("Recall@10", lambda n: f(m(sy.get(n), "retrieval.recall@10"))),
            ("MRR", lambda n: f(m(sy.get(n), "retrieval.mrr"))),
            ("Answer correctness", lambda n: f(m(sy.get(n), "generation.answer_correct"))),
            ("Claim support (verifier-judged)", lambda n: f(m(sy.get(n), "claims.claim_support_rate"))),
            ("Groundedness (supported and cited)", lambda n: f(m(sy.get(n), "generation.groundedness"))),
            ("Hallucinated claim rate", lambda n: f(m(sy.get(n), "hallucination.hallucinated_claim_rate"))),
            ("Citation precision", lambda n: f(m(sy.get(n), "citation.citation_precision"))),
            ("TTFE ms p50 / p95 (from first chunk)", lambda n: lat(n, "ttfe")),
            ("TTFA ms p50 / p95 (from first chunk)", lambda n: lat(n, "ttfa")),
            ("TTVA ms p50 / p95 (from first chunk)", lambda n: lat(n, "ttva")),
            ("TTFE ms p50 / p95 (after utterance end)", lambda n: lat(n, "ttfe_after_end")),
            ("TTVA ms p50 / p95 (after utterance end)", lambda n: lat(n, "ttva_after_end")),
            ("Total latency ms p50 / p95 (after utterance end)",
             lambda n: lat(n, "total_after_end") if n != "full_system" else NM),
            ("Retrieval calls / query", lambda n: f(m(sy.get(n), "efficiency.retrieval_calls"))),
            ("LLM calls / query", lambda n: f(m(sy.get(n), "efficiency.llm_calls"))),
            ("Cache hit rate", lambda n: f((sy.get(n) or {}).get("cache_hit_rate"))),
            ("Failure rate (failed samples)", lambda n: f((sy.get(n) or {}).get("failure_rate")))]
    return [{"metric": lab, **{c: fn(c) for c in cols}} for lab, fn in spec]


def streaming_curve(r7) -> str | None:
    """Fraction of expected key facts present in the latest answer text over time, for the first MULTI_INTENT
    test sample by id (fixed selection rule, chosen before the results were seen - not the best-looking case)."""
    multi = sorted({x["sample_id"] for x in r7 if x["query_category"] == "MULTI_INTENT"})
    if not multi:
        return None
    pick = multi[0]
    series = []
    for n in ("full_system_batch", "full_system"):
        trace = RES / "runs" / "traces" / f"{n}__{pick}.jsonl"
        row = next((x for x in r7 if x["sample_id"] == pick and x["system_variant"] == n), None)
        if not trace.exists() or row is None:
            continue
        evs = [json.loads(x) for x in trace.read_text().splitlines()]
        keys = [c["key"] for c in row["expected"]["claims"] if c["key"]]
        t0 = next((e["t_wall_ms"] for e in evs if e["type"] == "CHUNK_RECEIVED"), None)
        if t0 is None:
            continue
        pts = [(0.0, 0.0)]
        for e in evs:
            if e["type"] in ("ANSWER_COMPLETED", "ANSWER_COMMITTED", "ANSWER_DELTA") and e["payload"].get("text"):
                text = e["payload"]["text"].lower()
                frac = sum(all(k.lower() in text for k in ks) for ks in keys) / len(keys)
                pts.append((e["t_wall_ms"] - t0, frac))
        series.append((LABEL[n], pts))
    if not series:
        return None
    return P.steps(f"Useful answer content over time ({pick}, first MULTI_INTENT sample)", series,
                   "ms from the first transcript chunk", "share of expected facts in the latest answer text",
                   subtitle="answer text from ANSWER_COMPLETED / ANSWER_COMMITTED events; one run per variant")


def qualitative(r1: list[dict], system: str = "full_system", n_fail: int = 8, n_ok: int = 4) -> None:
    """Seeded random sample of failing and passing turns of one system (no hand-picking; brief §46)."""
    rng = random.Random(20261004)
    mine = sorted((x for x in r1 if x["system_variant"] == system), key=lambda x: x["sample_id"])
    fail = [x for x in mine if x["error_categories"]]
    ok = [x for x in mine if not x["error_categories"]]
    pick = rng.sample(fail, min(n_fail, len(fail))) + rng.sample(ok, min(n_ok, len(ok)))
    out = [f"# Qualitative sample - {system} (seeded random: {len(pick)} of {len(mine)} turns; "
           f"{len(fail)} with an error category, {len(ok)} without)", ""]
    for x in pick:
        out += [f"## {x['sample_id']} - {x['query_category']} / {x['difficulty']} - "
                f"{', '.join(x['error_categories']) or 'no error category'}", "",
                f"* query: {x['query']}", f"* expected: {x['expected']['answer']}",
                f"* answer: {(x['answer'] or '(empty)')[:600]}",
                f"* retrieved sections: {', '.join(x['evidence_citations'][:8]) or '(none)'}",
                f"* gold sections: {', '.join(x['expected']['evidence']) or '(none)'}",
                f"* trace: {trace_of(system, x['sample_id'])}", ""]
    (RES / f"qualitative_{system}.md").write_text("\n".join(out) + "\n")


def trace_of(system: str, sample_id: str) -> str:
    """Test-split sessions are named by the sample id before the turn suffix (S05.1 -> S05)."""
    p = RES / "runs" / "traces" / f"{system}__{sample_id.split('.')[0]}.jsonl"
    return str(p.relative_to(REPO)) if p.exists() else "-"


def human_sheet(r1: list[dict]) -> None:
    rng = random.Random(20261004)
    by_type: dict[str, list[str]] = {}
    for x in r1:
        by_type.setdefault(x["query_category"], []).append(x["sample_id"])
    pick = []
    for t, ids in sorted(by_type.items()):
        ids = sorted(set(ids))
        rng.shuffle(ids)
        pick += ids[:2]
    rows_ = [x for x in r1 if x["sample_id"] in pick and x["system_variant"] in ("naive_rag", "hybrid_rerank",
                                                                                    "full_system")]
    rng.shuffle(rows_)
    out = REPO / "experiments" / "datasets" / "human_eval_sheet.csv"
    with out.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["item", "query", "answer", "citations", "correctness_1_3", "relevance_1_3", "grounding_1_3",
                    "completeness_1_3", "citation_usefulness_1_3", "notes"])
        for k, x in enumerate(rows_, start=1):
            w.writerow([k, x["query"], x["answer"], "; ".join(x["citations"]), "", "", "", "", "", ""])
    key = REPO / "experiments" / "datasets" / "human_eval_key.json"
    key.write_text(json.dumps({k: {"sample_id": x["sample_id"], "system_variant": x["system_variant"]}
                               for k, x in enumerate(rows_, start=1)}, indent=2))


def dashboard(md: list[str], plots: dict[str, str]) -> None:
    """Static research dashboard: the charts plus the tables (rendered from tables.md as preformatted blocks)."""
    def md_tables(lines):
        out, buf = [], []
        for line in lines:
            if line.startswith("|"):
                buf.append(line)
                continue
            if buf:
                head = [c.strip() for c in buf[0].strip("|").split("|")]
                body = [[c.strip() for c in r.strip("|").split("|")] for r in buf[2:]]
                out.append("<div class='tw'><table><thead><tr>" + "".join(f"<th>{html.escape(c)}</th>" for c in head)
                           + "</tr></thead><tbody>" + "".join("<tr>" + "".join(f"<td>{html.escape(c)}</td>" for c in r)
                                                              + "</tr>" for r in body) + "</tbody></table></div>")
                buf = []
            if line.startswith("## "):
                out.append(f"<h2>{html.escape(line[3:])}</h2>")
            elif line.startswith("### "):
                out.append(f"<h3>{html.escape(line[4:])}</h3>")
            elif line.startswith("> "):
                out.append(f"<p class='note'>{html.escape(line[2:])}</p>")
            elif line.strip() and not line.startswith("# "):
                out.append(f"<p>{html.escape(line)}</p>")
        return "\n".join(out)
    charts = "\n".join(f"<figure>{svg}</figure>" for _, svg in sorted(plots.items()))
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport"
content="width=device-width,initial-scale=1"><title>StreamRAG Phase 10 Evaluation</title><style>
:root{{--bg:#f9f9f7;--ink:#0b0b0b;--ink2:#52514e;--line:#e1e0d9;--card:#fcfcfb}}
@media (prefers-color-scheme:dark){{:root{{--bg:#0d0d0d;--ink:#fff;--ink2:#c3c2b7;--line:#2c2c2a;--card:#1a1a19}}}}
body{{margin:0;padding:24px 16px;background:var(--bg);color:var(--ink);font:14px/1.5 system-ui,-apple-system,
"Segoe UI",sans-serif}} main{{max-width:1100px;margin:0 auto}} h1{{font-size:22px}} h2{{font-size:17px;margin-top:32px}}
.note{{color:var(--ink2)}} .tw{{overflow-x:auto;background:var(--card);border:1px solid var(--line);border-radius:8px;
margin:8px 0}} table{{border-collapse:collapse;font-size:12px;font-variant-numeric:tabular-nums}} th,td{{padding:4px 8px;
border-bottom:1px solid var(--line);text-align:left;white-space:nowrap}} figure{{margin:12px 0;overflow-x:auto}}
svg{{max-width:100%;height:auto;border-radius:8px}}</style></head><body><main>
<h1>StreamRAG - Phase 10 experimental evaluation</h1>
<p class="note">Fixture corpora, implementer labels - NOT REPORTABLE. Every number comes from experiments/results/
(generated by experiments/runners/analyze.py). "NOT MEASURED" marks values that were not measured.</p>
<h2>Charts</h2>{charts}
{md_tables(md)}
</main></body></html>"""
    (RES / "dashboard.html").write_text(page)


if __name__ == "__main__":
    main()
