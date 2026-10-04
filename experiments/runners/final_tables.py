"""Builds the Phase 11 final-benchmark tables from the stored results (no system is run here).

Inputs   FINAL_BENCHMARK_RESULTS/{heldout_v2,dev_iterations,regression_v1,robustness,resources}/...,
         demo_check*.json, environment.json, experiments/results/regression_baseline.json (Phase 10)
Outputs  FINAL_BENCHMARK_RESULTS/tables.md          every table of FINAL_BENCHMARK_RESULTS/README.md
         FINAL_BENCHMARK_RESULTS/final_summary.json  the headline numbers quoted by the final documents
         FINAL_BENCHMARK_RESULTS/regression_v1/regression_report.json
A missing or incomplete input yields "NOT MEASURED" cells, never a number.

Usage: .venv/bin/python experiments/runners/final_tables.py
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import analyze as A  # noqa: E402
from corpora import CORPORA  # noqa: E402
from streamrag.evaluation import regression as REG  # noqa: E402

FB = REPO / "FINAL_BENCHMARK_RESULTS"
NM = A.NM
A.LABEL.update({"full_system_answerability": "S + answerability flag", "adaptive_rag_noansw": "S_batch, flag off",
                "full_system_noansw": "S, flag off"})
MAIN = ["naive_rag", "hybrid_rag", "hybrid_rerank", "streaming_fixed", "adaptive_rag", "full_system",
        "full_system_batch", "full_system_answerability"]
EXTRACTIVE = ["topk_5", "adaptive_x", "cache_none", "delta_full_restart"]
HEADLINE = [("Recall@5", "retrieval.recall@5"), ("MRR", "retrieval.mrr"),
            ("evidence precision", "evidence.evidence_precision"), ("answer correct", "generation.answer_correct"),
            ("forbidden (stale) value asserted", "generation.forbidden_hit"),
            ("insufficiency handled", "generation.insufficiency_ok"),
            ("claim support (verifier)", "claims.claim_support_rate"),
            ("hallucinated claim rate", "hallucination.hallucinated_claim_rate"),
            ("citation precision (verifier)", "citation.citation_precision"),
            ("retrieval calls / turn", "efficiency.retrieval_calls"), ("embeddings / turn", "efficiency.embeddings"),
            ("LLM calls / turn", "efficiency.llm_calls")]


def load(path: Path) -> dict | None:
    if not path.exists():
        return None
    d = json.loads(path.read_text())
    return d if d.get("status", "COMPLETED") == "COMPLETED" else None


def rows(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text().splitlines()] if path.exists() else []


def dataset(name: str) -> dict[str, dict]:
    p = REPO / "experiments" / "datasets" / name / "test.jsonl"
    return {x["sample_id"]: x for x in rows(p)}


def lat(agg: dict | None, key: str, stat: str = "p50"):
    v = ((agg or {}).get("latency") or {}).get(key) or {}
    return round(v[stat], 1) if v.get(stat) is not None else None


# ------------------------------------------------------------------ gates from traces (Theme 4 guide G2-G6)
def corpus_citations(corpus_key: str) -> set[str] | None:
    try:
        from streamrag.config import load_config
        from streamrag.corpus.pipeline import build_corpus
        cfg = load_config(REPO / "configs" / "default.yaml", overrides={"paths.corpus": CORPORA[corpus_key]})
        return {c.citation for c in build_corpus(cfg).chunks}
    except Exception:  # noqa: BLE001 - reported as NOT MEASURED
        return None


def trace_gates(trace_dir: Path, system: str, data: dict[str, dict]) -> dict:
    """G2 early retrieval, G3 intent count, G6 telemetry completeness, per utterance of the runtime traces."""
    eligible = early = early_final_reused = 0
    tele_events = tele_ok = 0
    utt_total = utt_complete = 0
    mi_rows = []
    required = ("event_id", "type", "session_id", "correlation_id", "t_session_ms", "seq", "schema_version")
    by_session: dict[str, list[dict]] = {}
    for x in data.values():
        by_session.setdefault(x["session_id"], []).append(x)
    for sid, samples in by_session.items():
        p = trace_dir / f"{system}__{sid}.jsonl"
        if not p.exists():
            continue
        ev = rows(p)
        seqs = [e.get("seq") for e in ev]
        tele_events += len(ev)
        tele_ok += sum(1 for e in ev if all(e.get(k) is not None for k in required))
        if seqs != sorted(seqs):
            tele_ok = -10**9          # an ordering violation fails the gate outright
        for s in sorted(samples, key=lambda z: z["turn_index"]):
            uid = f"u{s['turn_index']}"
            mine = [e for e in ev if e.get("utterance_id") == uid]
            chunks = [e for e in mine if e["type"] == "CHUNK_RECEIVED"]
            fin = next((e for e in mine if e["type"] == "UTTERANCE_FINALIZED"), None)
            starts = [e for e in mine if e["type"] == "RETRIEVAL_STARTED"]
            done = any(e["type"] == "TURN_COMPLETED" for e in mine)
            validated = any(e["type"] in ("ANSWER_VALIDATED", "ANSWER_FINALIZED") for e in mine)
            utt_total += 1
            utt_complete += int(done and validated)
            if len(chunks) >= 2 and fin is not None and s["ground_truth_evidence"]:
                eligible += 1
                if starts and starts[0]["t_session_ms"] < fin["t_session_ms"]:
                    early += 1
                    after = [e for e in mine if e["type"] == "RETRIEVAL_DECISION"
                             and e["t_session_ms"] >= fin["t_session_ms"]]
                    if after and after[0]["payload"].get("decision") == "SKIP":
                        early_final_reused += 1
            if s["query_type"] == "MULTI_INTENT":
                tc = next((e for e in mine if e["type"] == "TURN_COMPLETED"), None)
                intents = ((tc or {}).get("payload", {}).get("intent_set") or {}).get("intents") or []
                mi_rows.append((s["sample_id"], len(s["expected_intents"]), len(intents)))
    return {"eligible": eligible, "early": early, "early_rate": early / eligible if eligible else None,
            "early_final_reused": early_final_reused, "telemetry_events": tele_events,
            "telemetry_complete_rate": (tele_ok / tele_events) if tele_events and tele_ok >= 0 else
            (0.0 if tele_events else None), "utterances": utt_total, "utterances_completed": utt_complete,
            "multi_intent": mi_rows}


def _pct(v: list[float]) -> list:
    if not v:
        return [0, None, None, None]
    v = sorted(v)
    q = lambda p: round(v[min(len(v) - 1, int(p * len(v)))], 2)  # noqa: E731
    return [len(v), round(statistics.median(v), 2), q(0.90), q(0.95)]


def stage_profile(trace_dir: Path, system: str) -> list[list]:
    """Per-stage wall time from the runtime traces (brief §20): task execution / queue wait per task type, LLM and
    extractive generation, controller decision per chunk. Verification is not listed separately: its start / end
    events are published together after it finishes, so their timestamps do not bracket it. It is inside the
    'generation' and 'draft' task times (and measured directly in the batch pipeline: latency.verification_ms)."""
    execs: dict[str, list[float]] = {}
    waits: dict[str, list[float]] = {}
    gen: dict[str, list[float]] = {}
    decide: list[float] = []
    for p in sorted(trace_dir.glob(f"{system}__*.jsonl")):
        ev = rows(p)
        last_chunk: float | None = None
        for e in ev:
            t, pl = e["type"], e["payload"]
            if t == "TASK_COMPLETED":
                k = pl.get("task_type", "?")
                execs.setdefault(k, []).append((pl.get("wall") or {}).get("exec_ms") or 0.0)
                waits.setdefault(k, []).append(pl.get("queue_wait_ms") or 0.0)
            elif t == "ANSWER_GENERATION_COMPLETED":
                gen.setdefault(pl.get("backend", "?"), []).append((pl.get("wall") or {}).get("generation_ms") or 0.0)
            elif t == "CHUNK_RECEIVED":
                last_chunk = e["t_session_ms"]
            elif t == "RETRIEVAL_DECISION" and last_chunk is not None:
                decide.append(e["t_session_ms"] - last_chunk)
                last_chunk = None
    out = [["controller: chunk received -> retrieve / wait / skip decision"] + _pct(decide)]
    out += [[f"task execution: {k}"] + _pct(v) for k, v in sorted(execs.items())]
    out += [[f"queue wait: {k}"] + _pct(v) for k, v in sorted(waits.items())]
    out += [[f"generation ({k})"] + _pct(v) for k, v in sorted(gen.items())]
    return out


def citation_check(rs: list[dict], valid: set[str] | None) -> dict:
    out = {}
    for n in dict.fromkeys(x["system_variant"] for x in rs):
        cites = [c for x in rs if x["system_variant"] == n for c in (x.get("citations") or [])]
        out[n] = {"citations": len(cites),
                  "unresolvable": None if valid is None else sum(1 for c in cites if c not in valid)}
    return out


def category_rate(rs: list[dict], system: str, cats: tuple[str, ...], metric=("generation", "answer_correct")):
    v = [x["metrics"][metric[0]].get(metric[1]) for x in rs if x["system_variant"] == system
         and x["query_category"] in cats]
    v = [y for y in v if y is not None]
    return (sum(v) / len(v), len(v)) if v else (None, 0)


# ------------------------------------------------------------------ cancellation (final vs superseded turns, v2)
def final_turns_v2(r6: list[dict], n: str, data: dict[str, dict]) -> list:
    last: dict[str, int] = {}
    for x in data.values():
        last[x["session_id"]] = max(last.get(x["session_id"], 0), x["turn_index"])
    mine = [x for x in r6 if x["system_variant"] == n]
    fin = [x for x in mine if data[x["sample_id"]]["turn_index"] == last[data[x["sample_id"]]["session_id"]]]
    sup = [x for x in mine if x not in fin]
    ac = [x["metrics"]["generation"]["answer_correct"] for x in fin
          if x["metrics"]["generation"]["answer_correct"] is not None]
    tv = sorted(x["metrics"]["latency"]["ttva_after_end"] for x in fin
                if x["metrics"]["latency"].get("ttva_after_end") is not None)
    return [len(fin), sum(ac) / len(ac) if ac else None, round(statistics.median(tv), 1) if tv else None,
            tv[-1] if tv else None, len(sup), sum(1 for x in sup if (x["answer"] or "").strip())]


# ------------------------------------------------------------------ main
def main() -> None:
    md = ["# Phase 11 final benchmark tables (generated by experiments/runners/final_tables.py)", "",
          "> Fixture corpora and implementer-written labels: TEST FIXTURE ONLY, NOT REPORTABLE as official results.",
          "> Held-out v2 = `streamrag_eval_v2` (Lakeside utility corpus, 81 turns), frozen before any Phase 11 change.",
          ""]
    summary: dict = {}
    v2 = load(FB / "heldout_v2" / "HELDOUT_V2" / "results.json")
    r2 = rows(FB / "heldout_v2" / "HELDOUT_V2" / "samples.jsonl")
    d2 = dataset("streamrag_eval_v2")
    sy2 = A.systems(v2)

    # 1. held-out main table
    md += ["## 1. Held-out v2: final system vs baselines (means; latency in ms)", "",
           A.metric_table(v2, MAIN, HEADLINE), "", "### Latency (p50 / p90 / p95 / p99 ms)", "",
           A.latency_table(v2, MAIN), "", "### Failed samples", "",
           A.table(["system"] + [A.LABEL.get(n, n) for n in MAIN],
                   [["failed / turns"] + [f"{(sy2.get(n) or {}).get('failed', NM)} / "
                                          f"{(sy2.get(n) or {}).get('samples', NM)}" for n in MAIN]]), ""]
    summary["heldout_v2"] = {n: {k: A.m(sy2.get(n), d) for k, d in HEADLINE} |
                             {"ttfe_p50": lat(sy2.get(n), "ttfe"), "ttfa_p50": lat(sy2.get(n), "ttfa"),
                              "ttva_after_end_p50": lat(sy2.get(n), "ttva_after_end"),
                              "ttva_after_end_p95": lat(sy2.get(n), "ttva_after_end", "p95"),
                              "failed": (sy2.get(n) or {}).get("failed"), "turns": (sy2.get(n) or {}).get("samples")}
                             for n in MAIN + EXTRACTIVE}
    for blk, spec in (("Evidence", A.EVID), ("Generation", A.GEN), ("Claims", A.CLAIM), ("Citations", A.CIT),
                      ("Hallucination (model-free)", A.HAL), ("Efficiency", A.EFF)):
        md += [f"### {blk}", "", A.metric_table(v2, MAIN, spec), ""]

    # 2. extractive ablation block
    md += ["## 2. Held-out v2: retrieval-side ablations (extractive answers, no LLM)", "",
           A.metric_table(v2, EXTRACTIVE, HEADLINE[:4] + HEADLINE[4:5] + HEADLINE[9:11]), ""]

    # 3. paired comparisons
    md += ["## 3. Held-out v2: paired comparisons (B vs A; McNemar exact for binary, Wilcoxon signed-rank otherwise; "
           "bootstrap 95% CI of the difference)", "", A.comparisons(v2), ""]

    # 4. categories
    md += ["## 4. Held-out v2: special cases and categories", "", A.special_cases(r2, MAIN), "",
           "### Answer correctness by category", "", A.by_category(v2, MAIN), "",
           "### Answer correctness by difficulty", "", A.by_category(v2, MAIN, field="by_difficulty"), ""]

    # 5. errors
    md += ["## 5. Held-out v2: error categories (turns with the category; a turn can have several)", "",
           A.error_table(r2, MAIN), "", "### Reliability (turn counts)", "", A.error_budget(r2, MAIN), ""]
    summary["error_categories_full_system"] = (sy2.get("full_system") or {}).get("error_categories")

    # 6. cancellation
    c2 = load(FB / "heldout_v2" / "HELDOUT_V2_CANCELLATION" / "results.json")
    rc = rows(FB / "heldout_v2" / "HELDOUT_V2_CANCELLATION" / "samples.jsonl")
    if c2 and rc:
        names = ["cancel_off", "cancel_on"]
        md += ["## 6. Held-out v2: cancellation on overlapping corrections (same inputs)", "",
               "> 'Total worker ms' is the fair comparison. 'Wasted' counts only tasks of queries / answers that ended "
               "cancelled or stale, so it under-counts waste without cancellation.", "",
               A.table(["variant", "total worker ms", "of which wasted (flagged)", "useful", "wasted generation ms",
                        "LLM calls (sum)", "cancelled tasks", "stale discarded"],
                       [[A.LABEL[n]] + A.waste(rc, n) for n in names]), "",
               A.table(["variant", "final turns", "final: answer correct", "final: TTVA after end p50 ms",
                        "final: TTVA after end max ms", "superseded first turns", "superseded: answered"],
                       [[A.LABEL[n]] + final_turns_v2(rc, n, d2) for n in names]), "",
               A.comparisons(c2), ""]
        summary["cancellation_v2"] = {n: dict(zip(["total_worker_ms", "wasted_ms", "useful_ms", "wasted_gen_ms",
                                                   "llm_calls", "cancelled_tasks", "stale_discarded"],
                                                  A.waste(rc, n))) |
                                      dict(zip(["final_turns", "final_ac", "final_ttva_p50", "final_ttva_max",
                                                "superseded", "superseded_answered"], final_turns_v2(rc, n, d2)))
                                      for n in names}
    else:
        md += ["## 6. Held-out v2: cancellation", "", NM, ""]

    # 7. Theme 4 gates
    tg = trace_gates(FB / "heldout_v2" / "runs" / "traces", "full_system", d2)
    valid = corpus_citations("utility")
    cc = citation_check(r2, valid)
    fs = sy2.get("full_system") or {}
    g5 = category_rate(r2, "full_system", ("CONTEXTUAL_FOLLOWUP", "ENTITY_CORRECTION", "STREAMING_CORRECTION",
                                           "REPEATED_QUERY"))
    g5b = category_rate(r2, "full_system_batch", ("CONTEXTUAL_FOLLOWUP", "ENTITY_CORRECTION", "STREAMING_CORRECTION",
                                                  "REPEATED_QUERY"))
    g3 = category_rate(r2, "full_system", ("MULTI_INTENT",))
    mi = tg["multi_intent"]
    gates = [
        ["G2 early retrieval (>= 80% of eligible turns)", "first retrieval starts before the utterance is finalised; "
         "eligible = needs retrieval and >= 2 chunks (dossier §G2)",
         f"{tg['early']} / {tg['eligible']} = {A.f(tg['early_rate'])}" if tg["eligible"] else NM,
         f"of these, final transcript needed no new retrieval: {tg['early_final_reused']}"],
        ["G3 multi-intent (>= 70%)", "MULTI_INTENT turns: intent count equals the labelled count; answer correct",
         (f"{sum(1 for _, e, g in mi if e == g)} / {len(mi)} intent counts; answer correct "
          f"{A.f(g3[0])} (n={g3[1]})") if mi else NM, "n is small (4 turns); see also Phase 5 multi-intent suite"],
        ["G4 grounding (>= 85% citation support, no fabricated IDs)", "verifier claim support; citations that do not "
         "resolve to a corpus section",
         f"claim support {A.f(A.m(fs, 'claims.claim_support_rate'))}; unresolvable citations "
         f"{cc.get('full_system', {}).get('unresolvable', NM)} / {cc.get('full_system', {}).get('citations', NM)}",
         "verifier = the system's own NLI model (biased instrument; Phase 10 §23)"],
        ["G5 session refinement", "answer correct on follow-up / correction / repeat turns (runtime vs batch)",
         f"{A.f(g5[0])} (n={g5[1]}) vs batch {A.f(g5b[0])}" if g5[1] else NM, ""],
        ["G6 telemetry 100%", "events carrying all required fields, in order; utterances with TURN_COMPLETED and a "
         "validated answer", f"{A.f(tg['telemetry_complete_rate'])} of {tg['telemetry_events']} events; "
         f"{tg['utterances_completed']} / {tg['utterances']} utterances", ""],
    ]
    md += ["## 7. Theme 4 gates measured on the held-out v2 traces (full system)", "",
           A.table(["gate", "definition used", "measured", "note"], gates), "",
           "### Citations that do not resolve to a section of the v2 corpus (all systems)", "",
           A.table(["system", "citations", "unresolvable"], [[A.LABEL.get(n, n), v["citations"],
                                                            NM if v["unresolvable"] is None else v["unresolvable"]]
                                                           for n, v in cc.items()]), ""]
    summary["gates_v2"] = {"G2": {k: tg[k] for k in ("eligible", "early", "early_rate", "early_final_reused")},
                           "G3": {"intent_count_match": sum(1 for _, e, g in mi if e == g), "n": len(mi),
                                  "answer_correct": g3[0]},
                           "G4": {"claim_support": A.m(fs, "claims.claim_support_rate"),
                                  "unresolvable_citations": cc.get("full_system", {}).get("unresolvable"),
                                  "citations": cc.get("full_system", {}).get("citations")},
                           "G5": {"runtime": g5[0], "batch": g5b[0], "n": g5[1]},
                           "G6": {"event_completeness": tg["telemetry_complete_rate"], "events": tg["telemetry_events"],
                                  "utterances_completed": tg["utterances_completed"], "utterances": tg["utterances"]}}

    # 8. development iterations (v1 test split = development data, and the dev split)
    base = json.loads((REPO / "experiments" / "results" / "regression_baseline.json").read_text())["systems"]
    dab = load(FB / "dev_iterations" / "DEV_ANSWERABILITY_BATCH" / "results.json")
    dfr = load(FB / "dev_iterations" / "DEV_FIXES_RUNTIME" / "results.json")
    dev_spec = [("answer correct", "generation.answer_correct"), ("insufficiency handled", "generation.insufficiency_ok"),
                ("forbidden (stale) value", "generation.forbidden_hit"), ("abstained", "generation.abstained"),
                ("Recall@5", "retrieval.recall@5"), ("LLM calls / turn", "efficiency.llm_calls")]
    md += ["## 8. Development iterations (NOT held-out: v1 test split, inspected in Phase 10, and the dev split)", "",
           "> In these runs the answerability flag defaulted to ON, so `adaptive_rag` / `full_system` = flag ON and "
           "`*_noansw` = flag OFF. After this iteration the flag was set to OFF by default.", ""]
    for res, names, split, title in ((dab, ["adaptive_rag_noansw", "adaptive_rag"], "test", "batch, v1 test split"),
                                     (dab, ["adaptive_rag_noansw", "adaptive_rag"], "dev", "batch, dev split"),
                                     (dfr, ["full_system_noansw", "full_system"], "test", "runtime, v1 test split")):
        if res:
            md += [f"### {title}", "", A.metric_table(res, names, dev_spec, split), "",
                   A.latency_table(res, names, split), "", A.comparisons(res, split), ""]
    if dfr:
        p10 = base.get("full_system", {})
        now = A.systems(dfr).get("full_system_noansw")
        md += ["### Phase 10 full system vs the Phase 11 fixes with the flag OFF (v1 test split, development data)", "",
               A.table(["metric", "Phase 10 full_system", "Phase 11 fixes, flag off"],
                       [[lab, p10.get(d), A.m(now, d)] for lab, d in dev_spec] +
                       [["TTVA after end p50 ms", p10.get("latency.ttva_after_end"), lat(now, "ttva_after_end")]]), ""]

    # 9. regression vs Phase 10 on v1
    rv = load(FB / "regression_v1" / "REGRESSION_V1" / "results.json")
    if rv:
        cur = REG.snapshot(rv, "test")
        rules = yaml.safe_load((REPO / "experiments" / "configs" / "regression_rules.yaml").read_text())["metrics"]
        rep = {s: REG.compare(v, base[s], rules) for s, v in cur.items() if s in base}
        (FB / "regression_v1" / "regression_report.json").write_text(json.dumps(rep, indent=2, default=str))
        body = [[A.LABEL.get(s, s), r["metric"], r["baseline"], r["current"], r["status"]]
                for s, rr in rep.items() for r in rr]
        md += ["## 9. Regression vs Phase 10 (v1 test split; development data, so improvements are optimistic)", "",
               "> Rules: experiments/configs/regression_rules.yaml. The evaluation instrument changed in Phase 11 "
               "(claim decomposer fix), which shifts claim-level metrics for every system.", "",
               A.table(["system", "metric", "Phase 10", "Phase 11", "status"], body), ""]
        summary["regression_v1"] = {s: {r["metric"]: [r["baseline"], r["current"], r["status"]] for r in rr}
                                    for s, rr in rep.items()}
    else:
        md += ["## 9. Regression vs Phase 10", "", NM, ""]

    # 10. robustness / resources / demo / environment
    rob = FB / "robustness" / "results.json"
    if rob.exists():
        d = json.loads(rob.read_text())["summary"]
        md += ["## 10. Robustness (fault injection, final configuration)", "",
               A.table(["scenario", "runs", "recovery", "degraded success", "incorrect answer"],
                       [[k, v["runs"], v["recovery_rate"], v["degraded_success_rate"], v["incorrect_answer_rate"]]
                        for k, v in d.items()]), ""]
        summary["robustness"] = d
    else:
        md += ["## 10. Robustness", "", NM, ""]
    res = FB / "resources" / "results.json"
    if res.exists():
        d = json.loads(res.read_text())
        md += ["## 11. Resources (this machine)", "",
               A.table(["stage", "peak RSS MB"], [[k, v] for k, v in d.get("memory_peak_rss_mb", {}).items()]), "",
               A.table(["call", "n", "CPU ms median", "wall ms median", "wall ms p90"],
                       [[k, v["n"], v["cpu_ms_median"], v["wall_ms_median"], v["wall_ms_p90"]]
                        for k, v in d.get("cpu_per_call", {}).items()]), "",
               A.table(["artefact", "disk MB"], [[k, v] for k, v in d.get("disk_mb", {}).items()]), ""]
        summary["resources"] = d
    for name in ("demo_check", "demo_check_no_llm"):
        p = FB / f"{name}.json"
        if p.exists():
            d = json.loads(p.read_text())
            sc = d.get("scenarios", [])
            body = []
            for s in sc:
                for uid, t in sorted((s.get("turns") or {}).items()):
                    mt = t.get("metrics") or {}
                    body.append([s["id"], uid, "PASS" if s.get("passed") else "FAIL", t.get("status", NM),
                                 t.get("mode") or "-", mt.get("ttfe_ms"), mt.get("ttfa_ms"), mt.get("ttva_ms"),
                                 "; ".join(s.get("failures") or []) or "-"])
            md += [f"## 12. {name} (headless run of every demo scenario; ms from the first chunk of the turn)", "",
                   f"LLM: {(d.get('summary') or {}).get('llm', NM)}", "",
                   A.table(["scenario", "turn", "check", "final status", "mode", "TTFE ms", "TTFA ms", "TTVA ms",
                            "failures"], body), ""]
            summary[name] = d.get("summary")
    prof = stage_profile(FB / "heldout_v2" / "runs" / "traces", "full_system")
    if prof and any(r[1] for r in prof):
        md += ["## 13. Performance profile: wall time per stage (full system, held-out v2 traces)", "",
               A.table(["stage", "n", "p50 ms", "p90 ms", "p95 ms"], prof), "",
               "> 'task execution: generation' = LLM call + claim verification + citation validation + repair for a "
               "final answer; 'draft' = extractive draft + verification. Verification alone is measured in the batch "
               "pipeline (§1 latency table, verification_ms).", ""]
        summary["stage_profile_v2"] = {r[0]: r[1:] for r in prof}
    env = FB / "environment.json"
    if env.exists():
        summary["environment"] = json.loads(env.read_text())

    (FB / "tables.md").write_text("\n".join(md) + "\n")
    (FB / "final_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print("wrote", FB / "tables.md", "and", FB / "final_summary.json")
    print(json.dumps(summary.get("gates_v2"), indent=1, default=str))


if __name__ == "__main__":
    main()
