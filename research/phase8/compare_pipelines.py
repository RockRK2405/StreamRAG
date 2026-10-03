"""Brief §76: sequential pipeline vs async runtime vs full runtime, with the REAL local LLM (Ollama qwen3:4b).

Same ten streaming scenarios as the Phase 7 end-to-end test, streamed in real time (400 ms per chunk, 2.5 s between
utterances), real retrieval (bge-small ONNX + BM25), real NLI verification, real model. Fixture domain - NOT
REPORTABLE; one run per scenario and pipeline on one machine.

Pipelines:
* BASELINE - the Phase 7 pipeline as it ran before Phase 8 (Phase 4 realtime runner): retrieval only after the
  utterance ends (controller ``end_only``), one retrieval at a time, generation and verification inline on the event
  loop, no drafts, no cancellation of superseded work.
* ASYNC    - the Phase 8 runtime: early retrieval while the user speaks, lexical / dense subtasks in parallel,
  generation off the loop. No cancellation, no drafts, no semantic retrieval cache.
* FULL     - the Phase 8 runtime with every default: + cooperative cancellation, barge-in on corrections, verified
  drafts while the user speaks, the Phase 6 semantic cache (evidence reuse).

Milestones per turn (ms, from the turn's first processed chunk): first evidence, first answer (draft or final),
validated answer; plus the wait after the utterance ended (validated - utterance end), which is what a user feels.

Usage: .venv/bin/python research/phase8/compare_pipelines.py --index-root /tmp/idx8   (needs `ollama serve`)
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "phase7"))
from common import first_inputs, meta, pct, stack_for, turn_metrics, with_cfg, write  # noqa: E402

from streamrag.generation.llm import OllamaBackend  # noqa: E402
from streamrag.models.events import SessionEnd, SessionStart  # noqa: E402
from streamrag.runtime import StreamingRuntime  # noqa: E402
from streamrag.runtime.driver import drive_realtime  # noqa: E402
from streamrag.runtime.telemetry import LoopLagMonitor  # noqa: E402
from streamrag.streaming import simulator as sim  # noqa: E402
from streamrag.streaming.runner import arun_realtime  # noqa: E402

SCENARIOS = [
    ("01_simple_question", "fixture", [["How high should", "the wicks be trimmed?"]]),
    ("02_multi_intent", "grounding", [["Tell me the eligibility requirements", "and the application process",
                                       "for the permit."]]),
    ("03_incomplete_streaming_question", "fixture", [["What are the rules", "for ladders", "in the orchard?"]]),
    ("04_late_constraint", "fixture", [["What are the rules for ladders", "in the orchard?"],
                                       ["Specifically", "overnight."]]),
    ("05_entity_correction", "fixture", [["What are the rules for ladders", "in the orchard?"],
                                         ["Sorry, I meant crates", "instead of ladders."]]),
    ("06_unsupported_fact", "grounding", [["How much does it cost", "to renew a permit?"]]),
    ("07_contradictory_evidence", "grounding", [["What is the application fee", "for a new permit?"]]),
    ("08_missing_evidence", "grounding", [["What is the parking policy", "at the permit office?"]]),
    ("09_incremental_revision", "fixture", [["How are crates handled", "at harvest?"], ["Only for the", "night shift."],
                                            ["Actually, ignore", "the night shift restriction."]]),
    ("10_citation_integrity", "injection", [["Is the permit office open", "on public holidays?"]]),
]


def inputs(name: str, turns: list[list[str]]) -> list:
    evs, off = [SessionStart(session_id=name)], 0.0
    for n, chunks in enumerate(turns, start=1):
        evs += sim.stream(chunks, interval_ms=400, session_id=name, utterance_id=f"u{n}", offset_ms=off,
                          wrap_session=False)
        off += 400 * len(chunks) + 2500
    return evs + [SessionEnd(session_id=name)]


def baseline_cfg(st):
    return with_cfg(st, **{"controller.strategy": "end_only", "multi_intent.max_concurrent_retrievals": 1,
                           "generation.draft_mode": "off", "controller.cancel_superseded": "never"})


def async_cfg(st):
    return with_cfg(st, **{"runtime.cancel_running": False, "runtime.cancel_on_correction": False,
                           "controller.cancel_superseded": "never", "generation.draft_mode": "off",
                           "session.cache": False})


async def run_baseline(st, evs):
    mon = LoopLagMonitor()
    mon.start()
    t0 = time.perf_counter()
    run = await arun_realtime(st.cfg, st.service, st.policy, evs, None, st.index_hash, st.intent_stack)
    wall = time.perf_counter() - t0
    await mon.stop()
    return run.events, wall, mon.lags_ms, {}


async def run_runtime(st, evs, llm):
    rt = await StreamingRuntime(st.cfg, st, llm=llm).start()
    t0 = time.perf_counter()
    sid = await drive_realtime(rt, evs, close=True, timeout_s=180)
    wall = time.perf_counter() - t0
    summ = rt.summary()
    lags = list(rt.telemetry.loop_lag.lags_ms)
    await rt.shutdown()
    return rt.events(sid), wall, lags, summ


def measure(events, wall, lags, summ) -> dict:
    ms = turn_metrics(events, first_inputs(events))
    tasks = [e for e in events if e.type.value == "TASK_CANCELLED"]
    exec_cancel = sum((e.payload.get("wall") or {}).get("exec_ms") or 0.0 for e in tasks)
    per_turn = {}
    for uid, m in ms.items():
        wait = None if m["validated_answer"] is None or m["utterance_end"] is None else \
            round(m["validated_answer"] - m["utterance_end"], 1)
        per_turn[uid] = {**m, "wait_after_utterance_end": wait}
    return {"wall_s": round(wall, 3), "turns": per_turn,
            "retrieval_calls": sum(1 for e in events if e.type.value == "RETRIEVAL_STARTED"),
            "retrieval_cancelled": sum(1 for e in events if e.type.value == "RETRIEVAL_CANCELLED"),
            "tasks_cancelled": len(tasks), "cancelled_exec_ms": round(exec_cancel, 1),
            "queries_reused": sum(1 for e in events if e.type.value == "QUERY_REUSED"),
            "llm_calls": sum(1 for e in events if e.type.value == "LLM_CALL"),
            "drafts": sum(1 for e in events if e.type.value == "ANSWER_COMPLETED" and e.payload["status"] == "DRAFT"),
            "validated_answers": sum(1 for e in events if e.type.value == "ANSWER_FINALIZED"),
            "loop_lag_ms": {"max": round(max(lags), 1) if lags else None,
                            "p95": round(sorted(lags)[int(0.95 * (len(lags) - 1))], 1) if lags else None}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index-root", type=Path, required=True)
    ap.add_argument("--model", default="qwen3:4b")
    a = ap.parse_args()
    url = "http://127.0.0.1:11434"
    if not OllamaBackend.reachable(url, a.model):
        raise SystemExit("ollama serve with the model is required (real-LLM comparison)")
    llm = OllamaBackend(url, a.model)
    stacks = {}
    for corpus in ("fixture", "grounding", "injection"):
        stacks[corpus] = stack_for(corpus, a.index_root, backend=llm, **{"generation.backend": "ollama"})
    llm.complete([{"role": "user", "content": "Reply with {}"}], {"type": "object"})     # load the model (warm-up)
    rows = {}
    for name, corpus, turns in SCENARIOS:
        st = stacks[corpus]
        evs = inputs(name, turns)
        rows[name] = {}
        for pipe in ("BASELINE", "ASYNC", "FULL"):
            if pipe == "BASELINE":
                res = asyncio.run(run_baseline(baseline_cfg(st), evs))
            else:
                res = asyncio.run(run_runtime(async_cfg(st) if pipe == "ASYNC" else st, evs, llm))
            rows[name][pipe] = measure(*res)
            r = rows[name][pipe]
            print(name, pipe, {u: (t["first_evidence"], t["validated_answer"], t["wait_after_utterance_end"])
                               for u, t in r["turns"].items()}, "retrievals", r["retrieval_calls"],
                  "cancelled", r["tasks_cancelled"], "lag", r["loop_lag_ms"], flush=True)
    summary = {}
    for pipe in ("BASELINE", "ASYNC", "FULL"):
        turns = [t for sc in rows.values() for t in sc[pipe]["turns"].values()]
        summary[pipe] = {
            "time_to_first_evidence_ms": pct([t["first_evidence"] for t in turns]),
            "time_to_first_answer_ms": pct([t["first_answer"] for t in turns]),
            "time_to_validated_answer_ms": pct([t["validated_answer"] for t in turns]),
            "wait_after_utterance_end_ms": pct([t["wait_after_utterance_end"] for t in turns]),
            "total_latency_ms": pct([t["last_event"] for t in turns]),
            "retrieval_calls": sum(sc[pipe]["retrieval_calls"] for sc in rows.values()),
            "cancelled_tasks": sum(sc[pipe]["tasks_cancelled"] for sc in rows.values()),
            "cancelled_exec_ms": round(sum(sc[pipe]["cancelled_exec_ms"] for sc in rows.values()), 1),
            "queries_reused": sum(sc[pipe]["queries_reused"] for sc in rows.values()),
            "llm_calls": sum(sc[pipe]["llm_calls"] for sc in rows.values()),
            "drafts": sum(sc[pipe]["drafts"] for sc in rows.values()),
            "validated_answers": sum(sc[pipe]["validated_answers"] for sc in rows.values()),
            "turns": len(turns), "max_loop_lag_ms": max(sc[pipe]["loop_lag_ms"]["max"] or 0 for sc in rows.values())}
    write("comparison.json", {"meta": meta(llm=f"{a.model} via local Ollama (REAL model)",
                                           pacing="400 ms per chunk, 2.5 s between utterances (speech-like)",
                                           runs="one run per scenario and pipeline"),
                              "summary": summary, "scenarios": rows})
    print(summary)


if __name__ == "__main__":
    main()
