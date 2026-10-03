"""Failure injection and graceful degradation (brief §39-43, §56, §59)."""

import asyncio

import pytest

from grounding_helpers import requires_nli
from runtime_helpers import commits, drive, of, realtime, rt_stack, virtual
from streamrag.runtime import Fault, FaultInjector, SimulatedLLM, StreamingRuntime

pytestmark = requires_nli
Q = ["How high should the wicks be trimmed?"]


@pytest.fixture(scope="module")
def stack(tmp_path_factory):
    return rt_stack(tmp_path_factory, "corpus")


def _modes(evs):
    return [e.payload["mode"] for e in of(evs, "DEGRADED_MODE_CHANGED")]


def test_vector_index_failure_degrades_to_lexical(stack):
    rt, evs = virtual(stack, [Q], faults=FaultInjector([Fault("dense", "error", times=-1)]))
    assert _modes(evs) == ["RETRIEVAL_DEGRADED"]
    done = of(evs, "RETRIEVAL_COMPLETED")
    assert done and all(e.payload["status"] == "degraded" for e in done)
    assert "4 millimetres" in commits(evs)[-1].payload["text"] and commits(evs)[-1].payload["status"] == "VALIDATED_FINAL"


def test_lexical_failure_degrades_to_dense(stack):
    rt, evs = virtual(stack, [Q], faults=FaultInjector([Fault("lexical", "error", times=-1)]))
    assert "RETRIEVAL_DEGRADED" in _modes(evs)
    assert commits(evs) and commits(evs)[-1].payload["status"] == "VALIDATED_FINAL"


def test_total_retrieval_failure_is_reported_not_hidden(stack):
    rt, evs = virtual(stack, [Q], faults=FaultInjector([Fault("network", "error", times=-1)]))
    done = of(evs, "RETRIEVAL_COMPLETED")
    assert done and all(e.payload["status"] == "error" for e in done)
    assert any(e.payload.get("action") == "keep_previous_evidence" for e in of(evs, "ERROR"))
    text = (commits(evs) or [None])[-1]
    assert text is None or "Not established" in text.payload["text"] or text.payload["claims"] == []
    assert of(evs, "SESSION_CLOSED")                              # the session never hangs


def test_llm_error_and_transient_failure(stack):
    rt, evs = realtime(stack, [Q], llm=SimulatedLLM(latency_ms=50),
                       faults=FaultInjector([Fault("llm", "transient", times=1)]))
    assert not _modes(evs) and commits(evs)[-1].payload["backend"] == "simulated"   # retried, full mode
    assert rt.retry.stats["retried"] >= 1
    rt, evs = realtime(stack, [Q], llm=SimulatedLLM(latency_ms=50),
                       faults=FaultInjector([Fault("llm", "error", times=-1)]))
    assert "GENERATION_DEGRADED" in _modes(evs)
    final = commits(evs)[-1].payload
    assert final["status"] == "VALIDATED_FINAL" and "4 millimetres" in final["text"]     # extractive, still verified


def test_llm_timeout_degrades(stack):
    rt, evs = realtime(stack, [Q], llm=SimulatedLLM(latency_ms=50),
                       faults=FaultInjector([Fault("llm", "timeout", times=-1, delay_ms=600)]))
    assert "GENERATION_DEGRADED" in _modes(evs) and commits(evs)[-1].payload["status"] == "VALIDATED_FINAL"


def test_validation_failure_uses_rules_only_verification(stack):
    rt, evs = realtime(stack, [Q], llm=SimulatedLLM(latency_ms=50),
                       faults=FaultInjector([Fault("verification", "error", times=-1)]))   # NLI model broken
    assert "VALIDATION_DEGRADED" in _modes(evs)
    assert any(e.payload["task_type"] == "generation" for e in of(evs, "TASK_FAILED"))
    final = commits(evs)[-1].payload
    assert final["mode"] == "rules" and final["status"] == "VALIDATED_FINAL"


def test_failures_stay_inside_their_session(stack):
    async def main():
        faults = FaultInjector([Fault("network", "error", times=-1, session_id="bad")])
        rt = await StreamingRuntime(stack.cfg, stack, faults=faults).start()
        bad, good = rt.start_session("bad"), rt.start_session("good")
        await asyncio.gather(drive(rt, bad, [Q], 0.02), drive(rt, good, [Q], 0.02))
        await asyncio.gather(rt.complete_session(bad, 30), rt.complete_session(good, 30))
        await rt.shutdown()
        return rt
    rt = asyncio.run(main())
    eb, eg = rt.events("bad"), rt.events("good")
    assert all(e.payload["status"] == "error" for e in of(eb, "RETRIEVAL_COMPLETED"))
    assert all(e.payload["status"] == "ok" for e in of(eg, "RETRIEVAL_COMPLETED"))
    assert "4 millimetres" in commits(eg)[-1].payload["text"]


def test_overloaded_llm_queue_degrades_to_extractive_answers(stack):
    """More finals than the LLM queue holds: rejected generations are answered extractively (cpu pool), never
    dropped."""
    from runtime_helpers import with_runtime
    st = with_runtime(stack, max_concurrent_llm_calls=1, **{"queues.llm": 1}, cancel_on_correction=False)

    async def main():
        rt = await StreamingRuntime(st.cfg, st, llm=SimulatedLLM(latency_ms=400)).start()
        sids = [rt.start_session(f"s{i}") for i in range(4)]
        await asyncio.gather(*(drive(rt, s, [Q], 0.01, 0.0) for s in sids))
        await asyncio.gather(*(rt.complete_session(s, 60) for s in sids))
        await rt.shutdown()
        return rt
    rt = asyncio.run(main())
    finals = [commits(rt.events(s))[-1].payload for s in rt.sessions]
    assert all(f["status"] == "VALIDATED_FINAL" for f in finals)
    assert any(f["mode"] == "extractive" for f in finals) and any(f["mode"] == "full" for f in finals)
    assert rt.scheduler.stats["rejected"] >= 1
    assert any("GENERATION_DEGRADED" in _modes(rt.events(s)) for s in rt.sessions)
