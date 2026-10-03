"""Phase 8 runtime measurements (brief §52-58). Fixture domain, SYNTHETIC workloads - NOT REPORTABLE.

Every number is measured on this machine by running the runtime; nothing is estimated. What is synthetic is said in
each result file:
* the LLM is ``SimulatedLLM`` (fixed latency, deterministic output) - real-model latency is measured separately in
  ``compare_pipelines.py``;
* "remote index" latency is injected (``Fault(network, delay)``) where a benchmark needs retrieval to take time;
  the retrieval work itself (bge-small ONNX, BM25, RRF) is real.

Usage: .venv/bin/python research/phase8/run_runtime_benchmarks.py --index-root /tmp/idx8 [--only concurrency,...]
"""

from __future__ import annotations

import argparse
import asyncio
import gc
import sys
import threading
import time
import tracemalloc
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (ResourceProbe, chunks, first_inputs, meta, open_fds, pct, stack_for, suite,  # noqa: E402
                    turn_metrics, with_cfg, write)

from streamrag.runtime import Fault, FaultInjector, SimulatedLLM, StreamingRuntime  # noqa: E402

LLM_MS = 800.0


# ------------------------------------------------------------------------------------------------- workload helpers
async def stream(rt, sid: str, utterances: list[str], chunk_s: float = 0.15, gap_s: float = 0.4) -> None:
    for n, text in enumerate(utterances, start=1):
        for c in chunks(text):
            rt.push_transcript_delta(sid, f"u{n}", c)
            await asyncio.sleep(chunk_s)
        rt.end_utterance(sid, f"u{n}")
        await asyncio.sleep(gap_s)


def turns_of(rt) -> list[dict]:
    out = []
    for sid, rs in rt.sessions.items():
        evs = rs.bus.events
        for uid, m in turn_metrics(evs, first_inputs(evs)).items():
            out.append({"session": sid, "utterance": uid, **m})
    return out


def summarize_turns(turns: list[dict]) -> dict:
    return {k: pct([t[k] for t in turns]) for k in ("first_event", "first_evidence", "first_answer",
                                                     "validated_answer", "last_event")}


async def run_sessions(stack, workloads: list[list[str]], llm_ms: float = LLM_MS, faults=None,
                       chunk_s: float = 0.15, stagger_s: float = 0.0) -> tuple:
    rt = await StreamingRuntime(stack.cfg, stack, llm=SimulatedLLM(latency_ms=llm_ms), faults=faults).start()
    sids = [rt.start_session(f"s{i:02d}") for i in range(len(workloads))]

    async def one(i, sid, w):
        await asyncio.sleep(i * stagger_s)
        await stream(rt, sid, w, chunk_s)
        await rt.complete_session(sid, 300)
    with ResourceProbe() as probe:
        t0 = time.perf_counter()
        await asyncio.gather(*(one(i, s, w) for i, (s, w) in enumerate(zip(sids, workloads))))
        wall = time.perf_counter() - t0
    summ = rt.summary()
    await rt.shutdown()
    return rt, wall, probe, summ


def multi_intent_workloads(n: int) -> list[list[str]]:
    cases = [c for c in suite() if c["category"] in ("D_multi_intent", "J_long_answer", "B_partially_supported")
             and c["corpus"] == "grounding"] or suite("grounding")
    texts = [t["utterance_text"] for c in cases for t in c["turns"]]
    return [[texts[i % len(texts)], texts[(i + 1) % len(texts)]] for i in range(n)]


# ------------------------------------------------------------------------------------------------- §52 concurrency
def concurrency(stacks, reps: int = 3) -> dict:
    st = stacks["grounding"]
    configs = {
        "A_sequential": dict(max_concurrent_retrievals=1, max_concurrent_cpu=1, max_concurrent_llm_calls=1,
                             split_retrieval=False),
        "B_parallel_unbounded": dict(max_concurrent_retrievals=64, max_concurrent_cpu=64, max_concurrent_llm_calls=64,
                                     split_retrieval=True),
        "C_parallel_bounded": dict(max_concurrent_retrievals=4, max_concurrent_cpu=2, max_concurrent_llm_calls=4,
                                   split_retrieval=True),
    }
    work = multi_intent_workloads(8)
    faults = lambda: FaultInjector([Fault("network", "delay", times=-1, delay_ms=40)])   # noqa: E731
    out = {}
    for name, rc in configs.items():
        runs = []
        for _ in range(reps):
            s = with_cfg(st, **{f"runtime.{k}": v for k, v in rc.items()}, **{"runtime.max_concurrent_sessions": 8,
                                                                               "runtime.cancel_on_correction": False})
            rt, wall, probe, summ = asyncio.run(run_sessions(s, work, faults=faults()))
            turns = turns_of(rt)
            failed = sum(1 for t in turns if t["validated_answer"] is None)
            runs.append({"wall_s": round(wall, 3), "turns": len(turns), "throughput_turns_per_s": round(len(turns) / wall, 3),
                         "latency_ms": summarize_turns(turns), "unanswered_turn_rate": round(failed / len(turns), 4),
                         "resources": probe.summary(), "scheduler": summ["scheduler"]["max_depth"],
                         "task_failures": summ["scheduler"]["failure_rate"], "loop_lag_ms": summ["loop_lag_ms"]})
            gc.collect()
        out[name] = {"config": rc, "runs": runs,
                     "median_wall_s": sorted(r["wall_s"] for r in runs)[len(runs) // 2],
                     "median_validated_p50_ms": sorted(r["latency_ms"]["validated_answer"]["p50"] for r in runs)[len(runs) // 2],
                     "median_validated_p95_ms": sorted(r["latency_ms"]["validated_answer"]["p95"] for r in runs)[len(runs) // 2],
                     "median_evidence_p50_ms": sorted(r["latency_ms"]["first_evidence"]["p50"] for r in runs)[len(runs) // 2],
                     "median_peak_threads": sorted(r["resources"]["peak_threads"] for r in runs)[len(runs) // 2],
                     "median_cpu_s": sorted(r["resources"]["cpu_s"] for r in runs)[len(runs) // 2]}
    return {"meta": meta(workload="8 concurrent sessions x 2 multi-intent utterances (dev suite, grounding corpus), "
                                  "3 words / 150 ms", llm=f"SimulatedLLM {LLM_MS:.0f} ms (SYNTHETIC)",
                         note="cancel_on_correction off: Phase 6 classifies some second questions of this workload as"
                              " ENTITY_CHANGE, and barge-in would cancel the first answer - a semantics effect that"
                              " differs by timing, measured separately in cancellation.json",
                         injected="40 ms remote-index latency per retrieval subtask (SYNTHETIC)", reps=reps),
            "configs": out}


# ------------------------------------------------------------------------------------------------- §53 cancellation
def cancellation(stacks, reps: int = 5) -> dict:
    st = stacks["fixture"]
    scenario = [["What are the rules for ladders in the orchard?"], ["Sorry, I meant crates instead of ladders."]]
    modes = {"no_cancellation": {"runtime.cancel_running": False, "runtime.cancel_on_correction": False,
                                 "controller.cancel_superseded": "never"},
             "cancellation": {}}
    out = {}
    for name, ov in modes.items():
        runs = []
        for rep in range(reps):
            s = with_cfg(st, **ov)
            faults = FaultInjector([Fault("dense", "delay", times=-1, delay_ms=350)])   # slow remote vector index

            async def main():
                rt = await StreamingRuntime(s.cfg, s, llm=SimulatedLLM(latency_ms=1500), faults=faults).start()
                sid = rt.start_session("s1")
                await stream(rt, sid, scenario[0], 0.12, 0.25)            # correction while u1's answer generates
                for c in chunks(scenario[1][0]):
                    rt.push_transcript_delta(sid, "u2", c)
                    await asyncio.sleep(0.12)
                rt.end_utterance(sid, "u2")
                await rt.complete_session(sid, 60)
                summ = rt.summary()
                await rt.shutdown()
                return rt, summ
            rt, summ = asyncio.run(main())
            evs = rt.events("s1")
            led = rt.sessions["s1"].session.ledger
            obsolete_q = {r.query_id for r in led.all() if r.superseded_by or r.stale or r.status == "cancelled"
                          or (r.intent_id and rt.sessions["s1"].session.mi.tracker.intents[r.intent_id].status != "ACTIVE")}
            tasks = rt.scheduler.metrics
            by_task = {e.payload["task_id"]: e.payload for e in evs if e.type.value == "TASK_SCHEDULED"}
            wasted = cancelled_tasks = 0.0
            wasted_by = {"retrieval": 0.0, "generation": 0.0, "draft": 0.0}
            for m in tasks:
                p = by_task.get(m["task_id"], {})
                obsolete = p.get("query_id") in obsolete_q or (m["task_type"] == "generation" and
                                                              p.get("answer_kind") == "final" and
                                                              m["status"] != "COMPLETED") or \
                    (m["task_type"] == "generation" and _answer_utt(evs, m["task_id"]) == "u1")
                if m["status"] == "CANCELLED":
                    cancelled_tasks += 1
                if obsolete or m["status"] == "CANCELLED":
                    kind = "generation" if m["task_type"] == "generation" else "draft" if m["task_type"] == "draft" \
                        else "retrieval"
                    wasted += m["exec_ms"] or 0.0
                    wasted_by[kind] += m["exec_ms"] or 0.0
            ms = turn_metrics(evs, first_inputs(evs))
            runs.append({"wasted_work_ms": round(wasted, 1), "wasted_by_kind_ms": {k: round(v, 1) for k, v in wasted_by.items()},
                         "cancelled_tasks": int(cancelled_tasks),
                         "retrieval_cancelled": sum(1 for e in evs if e.type.value == "RETRIEVAL_CANCELLED"),
                         "llm_calls": sum(1 for e in evs if e.type.value == "LLM_CALL"),
                         "answers_committed": [e.utterance_id for e in evs if e.type.value == "ANSWER_COMMITTED"],
                         "u2_validated_ms": ms.get("u2", {}).get("validated_answer"),
                         "total_task_exec_ms": round(sum(m["exec_ms"] or 0 for m in tasks), 1)})
        out[name] = {"runs": runs, "median_wasted_ms": sorted(r["wasted_work_ms"] for r in runs)[len(runs) // 2],
                     "median_u2_validated_ms": sorted(r["u2_validated_ms"] for r in runs if r["u2_validated_ms"])[len(runs) // 2],
                     "median_total_exec_ms": sorted(r["total_task_exec_ms"] for r in runs)[len(runs) // 2]}
    nc, c = out["no_cancellation"], out["cancellation"]
    return {"meta": meta(scenario="ladder question, then 'Sorry, I meant crates' while u1's retrieval / answer run",
                         llm="SimulatedLLM 1500 ms (SYNTHETIC)", injected="350 ms dense-index latency (SYNTHETIC)",
                         reps=reps,
                         wasted_definition="execution time of tasks that were cancelled, or whose query was superseded"
                                           " / stale, or that answered the corrected turn u1"),
            "modes": out,
            "saved_work_ms_median": round(nc["median_wasted_ms"] - c["median_wasted_ms"], 1),
            "u2_latency_change_ms_median": round(c["median_u2_validated_ms"] - nc["median_u2_validated_ms"], 1)}


def _answer_utt(evs, task_id: str) -> str | None:
    for e in evs:
        if e.type.value == "TASK_SCHEDULED" and e.payload["task_id"] == task_id:
            return e.utterance_id
    return None


# ------------------------------------------------------------------------------------------------- §54 backpressure
def backpressure(stacks, n: int = 2000) -> dict:
    st = stacks["grounding"]
    words = "What are the eligibility requirements and how are applications for the fixture permit submitted".split()
    out = {}
    variants = [(f"{q}_{burst}", unbounded, yield_every)
                for burst, yield_every in (("paced_1khz", 20), ("instant_burst", None))
                for q, unbounded in (("unbounded_no_coalescing", True), ("bounded_coalescing", False))]
    for name, unbounded, yield_every in variants:
        async def main():
            rt = StreamingRuntime(st.cfg, st, llm=SimulatedLLM(latency_ms=300), unbounded_inputs=unbounded)
            await rt.start()
            sid = rt.start_session("s1")
            rs = rt.sessions[sid]
            depth = []
            gc.collect()
            tracemalloc.start()
            t0 = time.perf_counter()
            hyp = []
            for i in range(n):                           # ASR partial hypotheses of one chunk, ~1 kHz
                hyp = words[: (i % len(words)) + 1]
                rt.push_transcript_delta(sid, "u1", " ".join(hyp), stability="partial", replaces=0 if i else None)
                depth.append(len(rs.inputs))
                if yield_every and i % yield_every == yield_every - 1:
                    await asyncio.sleep(0.001)
            t_push = time.perf_counter()
            rt.push_transcript_delta(sid, "u1", " ".join(words), stability="final", replaces=0)
            rt.end_utterance(sid, "u1")
            await rt.complete_session(sid, 300)
            t_done = time.perf_counter()
            cur, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            evs = rt.events(sid)
            summ = rt.summary()
            await rt.shutdown()
            fin = [e.payload["transcript"] for e in evs if e.type.value == "UTTERANCE_FINALIZED"]
            return {"deltas": n, "push_s": round(t_push - t0, 3), "drain_after_last_push_ms": round((t_done - t_push) * 1000, 1),
                    "total_s": round(t_done - t0, 3), "python_heap_peak_mb": round(peak / 2 ** 20, 2),
                    "max_input_queue_depth": max(depth), "chunk_events_processed": sum(1 for e in evs if e.type.value == "CHUNK_RECEIVED"),
                    "coalesced": sum(e.payload["dropped"] for e in evs if e.type.value == "TRANSCRIPT_COALESCED")
                    + sum(e.payload.get("coalesced", 0) for e in evs if e.type.value == "BACKPRESSURE_APPLIED"
                          and e.payload["action"] == "coalesced_on_full_queue"),
                    "rejected": sum(1 for e in evs if e.type.value == "BACKPRESSURE_APPLIED" and e.payload["action"] == "rejected"),
                    "controller_decisions": sum(1 for e in evs if e.type.value == "RETRIEVAL_DECISION"),
                    "retrievals": sum(1 for e in evs if e.type.value == "RETRIEVAL_STARTED"),
                    "drafts_requested": summ["sessions"]["s1"]["lane"]["drafts_requested"],
                    "final_transcript_intact": bool(fin) and fin[0] == " ".join(words),
                    "events": len(evs), "loop_lag_ms": summ["loop_lag_ms"]}
        out[name] = asyncio.run(main())
        gc.collect()
    return {"meta": meta(workload=f"{n} partial ASR hypotheses of one utterance, then the final: paced at ~1 kHz "
                                  "(the loop runs every 20 deltas) or as one instant burst (the loop cannot run)",
                         llm="SimulatedLLM 300 ms (SYNTHETIC)",
                         note="the unbounded baseline is a finite burst (no uncontrolled growth is allowed to run)"),
            "modes": out}


# ------------------------------------------------------------------------------------------------- §55 stale race
def stale(stacks, reps: int = 10) -> dict:
    st = stacks["fixture"]
    out = {}
    for name, cancel in (("no_cancellation", False), ("cancellation", True)):
        res = []
        for _ in range(reps):
            s = with_cfg(st, **{"runtime.cancel_running": cancel})
            faults = FaultInjector([Fault("network", "delay", times=2, delay_ms=1200)])   # Q1's subtasks are slow

            async def main():
                rt = await StreamingRuntime(s.cfg, s, llm=SimulatedLLM(latency_ms=200), faults=faults).start()
                sid = rt.start_session("s1")
                rt.push_transcript_delta(sid, "u1", "What are the rules for ladders in the orchard?")
                await asyncio.sleep(0.2)
                rt.end_utterance(sid, "u1")
                await asyncio.sleep(0.1)
                rt.push_transcript_delta(sid, "u2", "Sorry, I meant crates instead of ladders.")
                await asyncio.sleep(0.1)
                rt.end_utterance(sid, "u2")
                await rt.complete_session(sid, 60)
                await rt.shutdown()
                return rt
            rt = asyncio.run(main())
            evs = rt.events("s1")
            done = [e for e in evs if e.type.value == "RETRIEVAL_COMPLETED"]
            first_q = next(e.query_id for e in evs if e.type.value == "QUERY_GENERATED")
            q1_done = [e for e in done if e.query_id == first_q]
            newer_before = bool(q1_done) and any(e.seq < q1_done[0].seq and e.query_id != first_q for e in done)
            final = [e.payload["text"] for e in evs if e.type.value == "ANSWER_COMMITTED"][-1].lower()
            res.append({"q1_completed_after_newer": newer_before,
                        "stale_discarded": sum(1 for e in evs if e.type.value == "STALE_RESULT_DISCARDED"),
                        "q1_cancelled": any(e.query_id == first_q for e in evs if e.type.value == "RETRIEVAL_CANCELLED"),
                        "final_mentions_ladders": "ladder" in final, "final_mentions_crates": "crate" in final})
        out[name] = {"runs": res, "overwrites": sum(r["final_mentions_ladders"] for r in res),
                     "races_reproduced": sum(r["q1_completed_after_newer"] for r in res)}
    return {"meta": meta(scenario="Q1 slow (1.2 s injected), correction Q2 completes first, Q1 completes last",
                         llm="SimulatedLLM 200 ms (SYNTHETIC)", reps=reps), "modes": out}


# ------------------------------------------------------------------------------------------------- §56 failure injection
def failures(stacks) -> dict:
    st = stacks["fixture"]
    q = ["How high should the wicks be trimmed?"]
    scenarios = {
        "retrieval_timeout": [Fault("dense", "timeout", times=-1)],
        "vector_db_failure": [Fault("dense", "error", times=-1)],
        "lexical_failure": [Fault("lexical", "error", times=-1)],
        "network_failure_transient": [Fault("network", "transient", times=2)],
        "network_failure_persistent": [Fault("network", "error", times=-1)],
        "llm_timeout": [Fault("llm", "timeout", times=-1, delay_ms=1500)],
        "llm_error": [Fault("llm", "error", times=-1)],
        "llm_transient": [Fault("llm", "transient", times=1)],
        "validation_failure": [Fault("verification", "error", times=-1)],
    }
    out = {}
    for name, fs in scenarios.items():
        s = with_cfg(st, **{"runtime.timeouts_ms.generation": 1000, "runtime.timeouts_ms.dense": 800})

        async def main():
            rt = await StreamingRuntime(s.cfg, s, llm=SimulatedLLM(latency_ms=200), faults=FaultInjector(fs)).start()
            sid = rt.start_session("s1")
            await stream(rt, sid, q, 0.1, 0.1)
            await rt.complete_session(sid, 60)
            summ = rt.summary()
            await rt.shutdown()
            return rt, summ
        t0 = time.perf_counter()
        rt, summ = asyncio.run(main())
        evs = rt.events("s1")
        commit = [e.payload for e in evs if e.type.value == "ANSWER_COMMITTED"]
        out[name] = {"faults": [f.__dict__ for f in fs], "wall_s": round(time.perf_counter() - t0, 3),
                     "degraded_modes": [e.payload["mode"] for e in evs if e.type.value == "DEGRADED_MODE_CHANGED"],
                     "retrieval_statuses": [e.payload["status"] for e in evs if e.type.value == "RETRIEVAL_COMPLETED"],
                     "task_statuses": summ["scheduler"]["stats"], "retries": summ["retry"],
                     "errors": [e.payload.get("action") for e in evs if e.type.value == "ERROR"],
                     "answer_status": commit[-1]["status"] if commit else None,
                     "answer_mode": commit[-1]["mode"] if commit else None,
                     "answer_text": commit[-1]["text"] if commit else None,
                     "session_closed": any(e.type.value == "SESSION_CLOSED" for e in evs),
                     "injected_hits": summ["faults"]}
    return {"meta": meta(question=q[0], llm="SimulatedLLM 200 ms (SYNTHETIC)",
                         timeouts={"generation_ms": 1000, "dense_ms": 800}), "scenarios": out}


# ------------------------------------------------------------------------------------------------- §57 load
def load(stacks, sizes=(1, 2, 4, 8, 16)) -> dict:
    st = stacks["grounding"]
    out = {}
    asyncio.run(run_sessions(st, multi_intent_workloads(1)))             # warm-up (models, caches); not measured
    for llm_slots in (1, 4):
        rows = {}
        for n in sizes:
            s = with_cfg(st, **{"runtime.max_concurrent_sessions": max(sizes),
                                "runtime.max_concurrent_llm_calls": llm_slots, "runtime.cancel_on_correction": False})
            rt, wall, probe, summ = asyncio.run(run_sessions(s, multi_intent_workloads(n), stagger_s=0.05))
            turns = turns_of(rt)
            errors = sum(1 for sid in rt.sessions for e in rt.events(sid) if e.type.value == "ERROR")
            rows[str(n)] = {"sessions": n, "turns": len(turns), "wall_s": round(wall, 3),
                            "throughput_turns_per_s": round(len(turns) / wall, 3),
                            "validated_answer_ms": pct([t["validated_answer"] for t in turns]),
                            "first_evidence_ms": pct([t["first_evidence"] for t in turns]),
                            "unanswered_turns": sum(1 for t in turns if t["validated_answer"] is None),
                            "extractive_answers_overload": sum(1 for sid in rt.sessions for e in rt.events(sid)
                                                               if e.type.value == "ANSWER_COMMITTED"
                                                               and e.payload["mode"] == "extractive"),
                            "error_events": errors, "max_queue_depth": summ["scheduler"]["max_depth"],
                            "output_events_held_out_of_order": sum(s["output_order"]["held_on_arrival"]
                                                                   for s in summ["sessions"].values()),
                            "loop_lag_ms": summ["loop_lag_ms"], "resources": probe.summary()}
            gc.collect()
        out[f"llm_slots_{llm_slots}"] = rows
    return {"meta": meta(workload="N concurrent sessions x 2 multi-intent utterances (staggered 50 ms), 3 words / 150 ms",
                         llm=f"SimulatedLLM {LLM_MS:.0f} ms (SYNTHETIC)",
                         note="synthetic load on one machine and a 6-document fixture corpus: not production scale"),
            "load": out}


# ------------------------------------------------------------------------------------------------- §58 leaks
def leaks(stacks, cycles: int = 5) -> dict:
    st = stacks["grounding"]
    rows = []
    gc.collect()
    tracemalloc.start()
    for i in range(cycles):
        async def main():
            rt = await StreamingRuntime(st.cfg, st, llm=SimulatedLLM(latency_ms=100)).start()
            a, b = rt.start_session("a"), rt.start_session("b")
            await asyncio.gather(stream(rt, a, ["How are applications for the permit submitted?"], 0.03, 0.05),
                                 stream(rt, b, ["What is the application fee for a new permit?"], 0.03, 0.05))
            rt.cancel_session(b, "client_cancelled")
            out = await rt.shutdown()
            pending = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
            return out, len(pending), rt
        out, pending, rt = asyncio.run(main())
        del rt
        gc.collect()
        rows.append({"cycle": i, "threads": threading.active_count(), "open_fds": open_fds(),
                     "pending_asyncio_tasks": pending, "live_cancellation_tokens": out["live_tokens"],
                     "zombie_workers": out["zombies"],
                     "python_heap_mb": round(tracemalloc.get_traced_memory()[0] / 2 ** 20, 2)})
    tracemalloc.stop()
    return {"meta": meta(cycles=cycles, workload="runtime start, 2 sessions (one cancelled mid-run), shutdown"),
            "cycles": rows,
            "thread_growth_after_first_cycle": rows[-1]["threads"] - rows[0]["threads"],
            "fd_growth_after_first_cycle": rows[-1]["open_fds"] - rows[0]["open_fds"]}


SECTIONS = {"concurrency": concurrency, "cancellation": cancellation, "backpressure": backpressure, "stale": stale,
            "failures": failures, "load": load, "leaks": leaks}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index-root", type=Path, required=True)
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    stacks = {name: stack_for(name, a.index_root) for name in ("fixture", "grounding")}
    only = [x for x in a.only.split(",") if x] or list(SECTIONS)
    for name in only:
        t0 = time.time()
        res = SECTIONS[name](stacks)
        p = write(f"{name}.json", res)
        print(f"{name}: {time.time() - t0:.1f}s -> {p}", flush=True)


if __name__ == "__main__":
    main()
