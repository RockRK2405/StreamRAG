"""Brief §83: the thirteen final validation scenarios through the runtime, each with explicit pass checks.

Scenarios 1-5 and 9-10 use the REAL local LLM (needs `ollama serve`); 6-8 and 11-13 use the SYNTHETIC
``SimulatedLLM`` (failure / load / shutdown behaviour, not model quality). Retrieval and verification are real.
Fixture domain - NOT a benchmark.

Usage: .venv/bin/python research/phase8/final_validation.py --index-root /tmp/idx8
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import first_inputs, meta, stack_for, turn_metrics, with_cfg, write  # noqa: E402

from streamrag.generation.llm import OllamaBackend  # noqa: E402
from streamrag.runtime import Fault, FaultInjector, SimulatedLLM, StreamingRuntime  # noqa: E402
from streamrag.runtime.events import untraceable  # noqa: E402


def of(evs, *types):
    return [e for e in evs if e.type.value in types]


def finals(evs):
    return [e.payload for e in of(evs, "ANSWER_COMMITTED")]


async def session(st, turns, llm, faults=None, sid="s1", chunk_s=0.15, gap_s=0.3, between=None):
    rt = await StreamingRuntime(st.cfg, st, llm=llm, faults=faults).start()
    rt.start_session(sid)
    for n, chunks in enumerate(turns, start=1):
        for c in chunks:
            rt.push_transcript_delta(sid, f"u{n}", c)
            await asyncio.sleep(chunk_s)
        rt.end_utterance(sid, f"u{n}")
        await asyncio.sleep(between[n - 1] if between else gap_s)
    await rt.complete_session(sid, 180)
    await rt.shutdown()
    return rt, rt.events(sid)


def base_checks(evs) -> dict:
    return {"no_untraceable_events": not untraceable(evs),
            "session_terminated_cleanly": bool(of(evs, "SESSION_CLOSED", "SESSION_CANCELLED")),
            "user_stream_ordered": [e.output_seq for e in evs if e.output_seq is not None] == list(
                range(sum(1 for e in evs if e.output_seq is not None)))}


def run_all(index_root: Path, model: str) -> dict:
    real = OllamaBackend("http://127.0.0.1:11434", model) if OllamaBackend.reachable("http://127.0.0.1:11434", model) \
        else None
    if real is None:
        raise SystemExit("needs `ollama serve` with the model")
    real.complete([{"role": "user", "content": "Reply with {}"}], {"type": "object"})
    fx = stack_for("fixture", index_root, backend=real, **{"generation.backend": "ollama"})
    gr = stack_for("grounding", index_root, backend=real, **{"generation.backend": "ollama"})
    sim = lambda ms=300: SimulatedLLM(latency_ms=ms)   # noqa: E731
    res = {}

    def record(name, llm_kind, evs, checks, extra=None):
        checks = {**base_checks(evs), **checks}
        res[name] = {"llm": llm_kind, "passed": all(checks.values()), "checks": checks,
                     "final_answers": [f["text"] for f in finals(evs)], **(extra or {})}
        print(name, "PASS" if res[name]["passed"] else "FAIL", {k: v for k, v in checks.items() if not v}, flush=True)

    # 1 normal streaming question
    rt, evs = asyncio.run(session(fx, [["How high should", "the wicks be trimmed?"]], real))
    m = turn_metrics(evs, first_inputs(evs))["u1"]
    record("01_normal_streaming_question", "real", evs,
           {"early_retrieval": of(evs, "RETRIEVAL_STARTED")[0].seq < of(evs, "UTTERANCE_FINALIZED")[0].seq,
            "validated": finals(evs)[-1]["status"] == "VALIDATED_FINAL",
            "answer_correct": "4 millimetres" in finals(evs)[-1]["text"]}, {"milestones_ms": m})
    # 2 multi-intent question
    rt, evs = asyncio.run(session(gr, [["Tell me the eligibility requirements", "and the application process",
                                        "for the permit."]], real))
    record("02_multi_intent_question", "real", evs,
           {"two_sections": finals(evs)[-1]["text"].count(":\n") >= 2,
            "parallel_subtasks": max(sum(1 for e in of(evs, "TASK_STARTED") if e.t_session_ms <= t.t_session_ms)
                                     - sum(1 for e in of(evs, "TASK_COMPLETED") if e.t_session_ms <= t.t_session_ms)
                                     for t in of(evs, "TASK_STARTED")) >= 2})
    # 3 late constraint
    rt, evs = asyncio.run(session(fx, [["What are the rules for ladders", "in the orchard?"],
                                       ["Specifically", "overnight."]], real))
    record("03_late_constraint", "real", evs,
           {"constraint_detected": any(e.payload.get("change_type") == "CONSTRAINT_ADDITION"
                                       for e in of(evs, "CONTEXT_CHANGE_DETECTED")),
            "answered_twice": len(finals(evs)) == 2,
            "no_llm_call_for_refinement": sum(1 for e in of(evs, "LLM_CALL") if e.utterance_id == "u2") == 0,
            "overnight_rule_kept": "overnight" in finals(evs)[-1]["text"]})
    # 4 entity correction
    rt, evs = asyncio.run(session(fx, [["What are the rules for ladders", "in the orchard?"],
                                       ["Sorry, I meant crates", "instead of ladders."]], real, between=[0.2, 0.2]))
    record("04_entity_correction", "real", evs,
           {"correction_detected": any(e.payload.get("change_type") in ("CORRECTION", "ENTITY_CHANGE")
                                       for e in of(evs, "CONTEXT_CHANGE_DETECTED")),
            "final_about_crates": "crate" in finals(evs)[-1]["text"].lower(),
            "no_ladders_in_final": "ladder" not in finals(evs)[-1]["text"].lower()})
    # 5 concurrent retrieval (remote-index latency injected so overlap is visible)
    rt, evs = asyncio.run(session(gr, [["What are the eligibility requirements and how are applications for the "
                                        "permit submitted?"]], real,
                                  FaultInjector([Fault("network", "delay", times=-1, delay_ms=80)])))
    iv = [(e.t_session_ms, next(x.t_session_ms for x in of(evs, "TASK_COMPLETED") if x.payload["task_id"] ==
                                e.payload["task_id"])) for e in of(evs, "TASK_STARTED")
          if e.payload["task_type"] in ("lexical", "dense")]
    record("05_concurrent_retrieval", "real", evs,
           {"overlapping_subtasks": max(sum(1 for a, b in iv if a <= t < b) for t, _ in iv) >= 2,
            "bounded": rt.scheduler.pools["retrieval"].max_running <= fx.cfg.runtime.max_concurrent_retrievals})
    # 6 retrieval failure (vector index down)
    rt, evs = asyncio.run(session(fx, [["How high should the wicks be trimmed?"]], sim(),
                                  FaultInjector([Fault("dense", "error", times=-1)])))
    record("06_retrieval_failure", "simulated", evs,
           {"degraded_reported": "RETRIEVAL_DEGRADED" in [e.payload["mode"] for e in of(evs, "DEGRADED_MODE_CHANGED")],
            "still_answered": finals(evs)[-1]["status"] == "VALIDATED_FINAL"})
    # 7 retrieval timeout
    st7 = with_cfg(fx, **{"runtime.timeouts_ms.dense": 500})
    rt, evs = asyncio.run(session(st7, [["How high should the wicks be trimmed?"]], sim(),
                                  FaultInjector([Fault("dense", "timeout", times=-1)])))
    record("07_retrieval_timeout", "simulated", evs,
           {"timed_out": bool(of(evs, "TASK_TIMED_OUT")), "lexical_only": all(
               e.payload["status"] == "degraded" for e in of(evs, "RETRIEVAL_COMPLETED")),
            "still_answered": finals(evs)[-1]["status"] == "VALIDATED_FINAL", "no_zombies": rt.scheduler.zombies() == 0})
    # 8 LLM failure
    rt, evs = asyncio.run(session(fx, [["How high should the wicks be trimmed?"]], sim(),
                                  FaultInjector([Fault("llm", "error", times=-1)])))
    record("08_llm_failure", "simulated", evs,
           {"generation_degraded": "GENERATION_DEGRADED" in [e.payload["mode"] for e in of(evs, "DEGRADED_MODE_CHANGED")],
            "validated_without_model": finals(evs)[-1]["status"] == "VALIDATED_FINAL"})
    # 9 cancellation (correction while the first answer is generated)
    rt, evs = asyncio.run(session(fx, [["What are the rules for ladders", "in the orchard?"],
                                       ["Sorry, I meant crates", "instead of ladders."]], real, between=[0.3, 0.2]))
    gen_cancel = [e for e in of(evs, "TASK_CANCELLED") if e.payload["task_type"] == "generation"]
    record("09_cancellation", "real", evs,
           {"generation_cancelled": bool(gen_cancel), "only_corrected_turn_answered":
               [e.utterance_id for e in of(evs, "ANSWER_COMMITTED")] == ["u2"]},
           {"cancel_reason": gen_cancel[0].payload["error"] if gen_cancel else None})
    # 10 stale result race (cancellation off so the late result arrives)
    st10 = with_cfg(fx, **{"runtime.cancel_running": False})
    rt, evs = asyncio.run(session(st10, [["What are the rules for ladders in the orchard?"],
                                         ["Sorry, I meant crates instead of ladders."]], real,
                                  FaultInjector([Fault("network", "delay", times=2, delay_ms=1500)]),
                                  chunk_s=0.1, between=[0.1, 0.2]))
    record("10_stale_result_race", "real", evs,
           {"stale_discarded": bool(of(evs, "STALE_RESULT_DISCARDED")),
            "final_not_overwritten": "ladder" not in finals(evs)[-1]["text"].lower()})
    # 11 high-frequency transcript
    async def hf():
        rt = await StreamingRuntime(gr.cfg, gr, llm=sim()).start()
        sid = rt.start_session("s1")
        words = "What are the eligibility requirements for the permit".split()
        for i in range(500):
            rt.push_transcript_delta(sid, "u1", " ".join(words[: i % len(words) + 1]), stability="partial",
                                     replaces=0 if i else None)
        rt.push_transcript_delta(sid, "u1", " ".join(words), replaces=0)
        rt.end_utterance(sid, "u1")
        await rt.complete_session(sid, 60)
        await rt.shutdown()
        return rt, rt.events(sid)
    rt, evs = asyncio.run(hf())
    record("11_high_frequency_transcript", "simulated", evs,
           {"bounded_queue": rt.sessions["s1"].inputs.max_delta_depth <= gr.cfg.runtime.queues.input,
            "coalesced": sum(e.payload.get("coalesced", 0) for e in of(evs, "BACKPRESSURE_APPLIED")) > 400,
            "final_state_kept": of(evs, "UTTERANCE_FINALIZED")[0].payload["transcript"] ==
            "What are the eligibility requirements for the permit",
            "answered": finals(evs)[-1]["status"] == "VALIDATED_FINAL"})
    # 12 multiple concurrent sessions
    async def multi():
        rt = await StreamingRuntime(gr.cfg, gr, llm=sim()).start()
        qs = ["What are the eligibility requirements for the permit?", "How are applications for the permit submitted?",
              "What is the application fee for a new permit?", "How often must a permit be renewed?"]
        sids = [rt.start_session(f"m{i}") for i in range(len(qs))]

        async def one(sid, q):
            rt.push_transcript_delta(sid, "u1", q)
            await asyncio.sleep(0.05)
            rt.end_utterance(sid, "u1")
            await rt.complete_session(sid, 60)
        await asyncio.gather(*(one(s, q) for s, q in zip(sids, qs)))
        await rt.shutdown()
        return rt, sids
    rt, sids = asyncio.run(multi())
    evs_all = [rt.events(s) for s in sids]
    task_sets = [{e.payload["task_id"] for e in of(evs, "TASK_SCHEDULED")} for evs in evs_all]
    record("12_multiple_concurrent_sessions", "simulated", evs_all[0],
           {"all_answered": all(finals(evs) for evs in evs_all),
            "isolated_tasks": all(not (a & b) for i, a in enumerate(task_sets) for b in task_sets[i + 1:]),
            "isolated_events": all(all(e.session_id == s for e in evs) for s, evs in zip(sids, evs_all)),
            "distinct_answers": len({finals(evs)[-1]["text"] for evs in evs_all}) == len(sids)})
    # 13 graceful shutdown with active work
    async def shut():
        base = threading.active_count()
        rt = await StreamingRuntime(gr.cfg, gr, llm=sim(2000)).start()
        sid = rt.start_session("s1")
        rt.push_transcript_delta(sid, "u1", "How are applications for the permit submitted?")
        await asyncio.sleep(0.05)
        rt.end_utterance(sid, "u1")
        await asyncio.sleep(0.4)
        busy = rt.scheduler.busy()
        t0 = time.perf_counter()
        out = await rt.shutdown(grace_ms=200)
        dt = time.perf_counter() - t0
        pending = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
        return rt, rt.events(sid), busy, out, dt, pending, base
    rt, evs, busy, out, dt, pending, base = asyncio.run(shut())
    record("13_graceful_shutdown", "simulated", evs,
           {"work_was_active": busy, "session_cancelled_by_shutdown": bool(of(evs, "SESSION_CANCELLED")),
            "no_zombies": out["zombies"] == 0, "no_live_tokens": out["live_tokens"] == 0, "no_pending_tasks": not pending,
            "bounded_time": dt < 3.0, "late_start_refused": _refuses(rt)},
           {"shutdown_s": round(dt, 3)})
    return res


def _refuses(rt) -> bool:
    try:
        rt.start_session("late")
    except RuntimeError:
        return True
    return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index-root", type=Path, required=True)
    ap.add_argument("--model", default="qwen3:4b")
    a = ap.parse_args()
    res = run_all(a.index_root, a.model)
    write("final_validation.json", {"meta": meta(note="1-5, 9-10: real local LLM; 6-8, 11-13: SimulatedLLM"),
                                    "passed": sum(r["passed"] for r in res.values()), "total": len(res),
                                    "scenarios": res})
    print(f"{sum(r['passed'] for r in res.values())}/{len(res)} passed")


if __name__ == "__main__":
    main()
