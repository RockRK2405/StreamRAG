"""Streaming benchmark: early retrieval, suppression and controller ablation over BenchmarkCase sessions.

Per turn (utterance) with expectations from the case:
  eligible          = retrieval_required and >= 2 chunks          (Phase 2 definition of the G2 denominator)
  retrieved_early   = first RETRIEVAL_STARTED < UTTERANCE_FINALIZED  (stream clock)
  false_retrieval   = not retrieval_required and >= 1 retrieval
  useful_early      = the first early retrieval's top-5 shares >= 1 chunk with the final query's top-5
  final_success@5   = final query's top-5 citations intersect gold (fixture gold => NOT REPORTABLE)
Policies: end_only (A), every_chunk (B), controller (C, rules), controller_prototype (C, embedding act classifier).
"""

from __future__ import annotations

import asyncio
import csv
import json
import time
from pathlib import Path

import numpy as np

from streamrag.config.settings import StreamRagConfig, config_hash
from streamrag.models.benchmark import BenchmarkCase
from streamrag.provenance import environment
from streamrag.replay.replay import dump_trace
from streamrag.streaming import simulator as sim
from streamrag.streaming.factory import StreamingStack
from streamrag.streaming.metrics import utterance_stats
from streamrag.streaming.runner import arun_realtime, run_virtual
from streamrag.telemetry.timing import percentile_summary

POLICIES = {"end_only": ("end_only", "rules"), "every_chunk": ("every_chunk", "rules"),
            "controller": ("rules", "rules"), "controller_prototype": ("rules", "prototype")}
BANNER = "DEV SUITE ON TEST FIXTURE DOMAIN - NOT AN OFFICIAL BENCHMARK RESULT"


def load_cases(path: Path) -> list[BenchmarkCase]:
    return [BenchmarkCase.model_validate_json(f.read_text()) for f in sorted(Path(path).glob("*.json"))]


def _top5(session, query_id: str | None) -> list[str]:
    if not query_id or query_id not in session.evidence:
        return []
    return [e.evidence_id for e in session.evidence[query_id].items[:5]]


def _cites5(session, query_id: str | None) -> list[str]:
    if not query_id or query_id not in session.evidence:
        return []
    return [e.citation for e in session.evidence[query_id].items[:5]]


def evaluate_run(case: BenchmarkCase, run) -> list[dict]:
    stats = utterance_stats(run.events)
    rows = []
    for turn in case.session.turns:
        exp, st = turn.expected, stats.get(turn.utterance_id, {})
        sess = run.session
        final_q = st.get("final_query_id")
        rets = st.get("retrievals", [])
        early = [r for r in rets if st.get("finalized_ms") is not None and r["start_ms"] < st["finalized_ms"]]
        final_top = set(_top5(sess, final_q))
        useful = bool(early) and bool(set(_top5(sess, early[0]["query_id"])) & final_top)
        useless_provisional = sum(1 for r in rets if r["trigger"] == "provisional" and r["query_id"] != final_q
                                  and not (set(_top5(sess, r["query_id"])) & final_top))
        gold = {g for gi in exp.intents for g in gi.gold_evidence}
        rows.append({
            "case_id": case.case_id, "category": case.category, "utterance_id": turn.utterance_id,
            "n_chunks": len(turn.chunks), "retrieval_required": exp.retrieval_required,
            "eligible": exp.retrieval_required and len(turn.chunks) >= 2,
            "retrieval_count": st.get("retrieval_count", 0), "retrieved_early": st.get("retrieved_early", False),
            "false_retrieval": (not exp.retrieval_required) and st.get("retrieval_count", 0) > 0,
            "useful_early": useful, "useless_provisional": useless_provisional,
            "duplicate_retrievals": st.get("duplicate_retrievals", 0), "cancelled": st.get("cancelled", 0),
            "lead_time_ms": st.get("lead_time_ms"), "ttfr_ms": st.get("ttfr_ms"),
            "post_final_ms": st.get("post_final_retrieval_latency_ms"),
            "utterance_ms": (st["finalized_ms"] - st["first_chunk_ms"]) if st.get("finalized_ms") is not None and st.get("first_chunk_ms") is not None else None,
            "final_success@5": (bool(gold & set(_cites5(sess, final_q))) if gold else None),
            "final_query": next((r["query"] for r in rets if r["query_id"] == final_q), None),
            "skips": st.get("skips", {}), "decisions": st.get("decisions", {}),
            "retrieval_wall_ms": [e.payload.get("wall", {}).get("measured_ms") for e in run.events
                                  if e.type.value == "RETRIEVAL_COMPLETED" and e.utterance_id == turn.utterance_id],
        })
    return rows


def _rate(xs: list[bool]) -> float | None:
    return round(sum(1 for x in xs if x) / len(xs), 4) if xs else None


def aggregate(rows: list[dict], controller_ms: list[float], retrieval_wall_ms: list[float]) -> dict:
    elig = [r for r in rows if r["eligible"]]
    req = [r for r in rows if r["retrieval_required"]]
    non = [r for r in rows if not r["retrieval_required"]]
    gold_rows = [r for r in rows if r["final_success@5"] is not None]
    return {
        "turns": len(rows), "eligible": len(elig), "retrieval_required": len(req), "non_retrieval": len(non),
        "early_retrieval_rate": _rate([r["retrieved_early"] for r in elig]),
        "useful_early_rate": _rate([r["useful_early"] for r in elig]),
        "false_retrieval_rate": _rate([r["false_retrieval"] for r in non]),
        "suppression_rate": (round(1 - _rate([r["false_retrieval"] for r in non]), 4) if non else None),
        "missed_retrieval_rate": _rate([r["retrieval_count"] == 0 for r in req]),
        "retrievals_total": sum(r["retrieval_count"] for r in rows),
        "retrievals_per_required_utterance": percentile_summary([r["retrieval_count"] for r in req]),
        "retrievals_on_non_retrieval_turns": sum(r["retrieval_count"] for r in non),
        "duplicate_retrievals": sum(r["duplicate_retrievals"] for r in rows),
        "useless_provisional_retrievals": sum(r["useless_provisional"] for r in rows),
        "cancelled_retrievals": sum(r["cancelled"] for r in rows),
        "lead_time_ms": percentile_summary([r["lead_time_ms"] for r in elig if r["lead_time_ms"] is not None]),
        "ttfr_ms": percentile_summary([r["ttfr_ms"] for r in req if r["ttfr_ms"] is not None]),
        "post_final_retrieval_latency_ms": percentile_summary([r["post_final_ms"] for r in req if r["post_final_ms"] is not None]),
        "evidence_ready_at_end_rate": _rate([r["post_final_ms"] == 0 for r in req if r["post_final_ms"] is not None]),
        "fixture_final_success@5": _rate([r["final_success@5"] for r in gold_rows]),
        "controller_decision_wall_ms": percentile_summary(controller_ms),
        "retrieval_wall_ms": percentile_summary([x for x in retrieval_wall_ms if x is not None]),
    }


def run_streaming_benchmark(stack: StreamingStack, cases_dir: Path, policies: list[str], out_dir: Path,
                            mode: str = "virtual", command: str = "") -> dict:
    cases = load_cases(cases_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fixture = stack.bundle.manifest.is_test_fixture or any(c.is_fixture for c in cases)
    summary, all_rows = {}, []
    for pol in policies:
        strategy, act = POLICIES[pol]
        st = stack.with_policy(strategy, act)
        rows, ctl, rwall = [], [], []
        for case in cases:
            inputs = sim.from_case(case)
            if mode == "virtual":
                run = run_virtual(st.cfg, st.service, st.policy, inputs, index_hash=st.index_hash)
            else:
                run = asyncio.run(arun_realtime(st.cfg, st.service, st.policy, inputs, index_hash=st.index_hash))
            dump_trace(run.events, out_dir / "traces" / pol / f"{case.case_id}.jsonl")
            r = evaluate_run(case, run)
            for row in r:
                row["policy"] = pol
                rwall += row.pop("retrieval_wall_ms")
            rows += r
            ctl += run.session.controller_wall_ms
        summary[pol] = aggregate(rows, ctl, rwall)
        all_rows += rows
    metrics = {"REPORTABLE": not fixture, "banner": BANNER if fixture else None, "mode": mode,
               "sim_retrieval_latency_ms": stack.cfg.streaming.sim_retrieval_latency_ms if mode == "virtual" else None,
               "speed": stack.cfg.streaming.speed if mode == "realtime" else None, "policies": summary}
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True, default=str))
    with (out_dir / "per_turn.jsonl").open("w") as f:
        for r in all_rows:
            f.write(json.dumps(r, sort_keys=True, default=str) + "\n")
    cols = [k for k in all_rows[0] if not isinstance(all_rows[0][k], (dict, list))] if all_rows else []
    with (out_dir / "per_turn.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(all_rows)
    (out_dir / "run_manifest.json").write_text(json.dumps({
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "command": command, "REPORTABLE": not fixture,
        "mode": mode, "policies": policies, "cases_dir": str(cases_dir), "n_cases": len(cases),
        "config_hash": config_hash(stack.cfg), "index_content_hash": stack.index_hash,
        "embedding_model": stack.bundle.manifest.embedding_model.model_dump(mode="json") if stack.bundle.manifest.embedding_model else None,
        "environment": environment()}, indent=2, sort_keys=True, default=str))
    return metrics


def lead_time_table(rows: list[dict]) -> dict:
    lt = [r["lead_time_ms"] for r in rows if r.get("lead_time_ms") is not None]
    return {"min": float(np.min(lt)) if lt else None, "max": float(np.max(lt)) if lt else None}
