"""Realtime (asyncio) mode: retrieval runs in worker threads without freezing the transcript stream."""

import pytest

from streamrag.streaming import run_realtime, utterance_stats
from streamrag.streaming import simulator as sim

from streaming_helpers import FakeBackend, hashing_stack, with_streaming


@pytest.fixture(scope="module")
def stack(fixture_bundle):
    cfg, b = fixture_bundle
    return hashing_stack(cfg, b)


def of(events, t):
    return [e for e in events if e.type.value == t]


def test_stream_keeps_flowing_while_retrieval_runs(stack):
    backend = FakeBackend(delay_s=0.4)               # slow retrieval (400 ms) vs chunks every 100 ms
    evs = sim.stream(["how high the wicks", "should be", "trimmed in", "the lighthouse", "today"], interval_ms=100)
    r = run_realtime(stack.cfg, backend, stack.policy, evs)
    first_start = of(r.events, "RETRIEVAL_STARTED")[0]
    first_done = next(e for e in of(r.events, "RETRIEVAL_COMPLETED") if e.payload["query_id"] == first_start.payload["query_id"])
    chunks_during = [e for e in of(r.events, "CHUNK_RECEIVED") if first_start.seq < e.seq < first_done.seq]
    assert len(chunks_during) >= 2, "chunks must be processed while the first retrieval is in flight"
    assert first_done.payload["wall"]["measured_ms"] >= 350
    st = utterance_stats(r.events)["u1"]
    assert st["retrieved_early"] and st["lead_time_ms"] > 0


def test_realtime_timeout_is_structured(stack):
    cfg = with_streaming(stack.cfg, retrieval_timeout_ms=100)
    r = run_realtime(cfg, FakeBackend(delay_s=0.5), stack.policy, sim.stream(["how high the wicks"], interval_ms=100))
    done = of(r.events, "RETRIEVAL_COMPLETED")
    assert done and done[0].payload["status"] == "timeout"
    assert any(e.payload["error_class"] == "RetrieverTimeoutError" for e in of(r.events, "ERROR"))
    assert of(r.events, "SESSION_CLOSED")


def test_realtime_scheduler_orders_equal_times_like_virtual():
    """Regression: callbacks with the same timestamp must run in (priority, seq) order in realtime mode too."""
    import asyncio

    from streamrag.streaming.clock import PRIORITY_COMPLETION, PRIORITY_INPUT, PRIORITY_TIMER, RealtimeScheduler

    async def go():
        sched, order = RealtimeScheduler(asyncio.get_running_loop()), []
        for i in range(20):
            sched.call_at(30.0, PRIORITY_INPUT, order.append, ("input", i))
        sched.call_at(30.0, PRIORITY_TIMER, order.append, ("timer", 0))
        sched.call_at(30.0, PRIORITY_COMPLETION, order.append, ("completion", 0))
        sched.call_at(10.0, PRIORITY_INPUT, order.append, ("early", 0))
        await sched.drain(lambda: sched.pending > 0)
        return order

    order = asyncio.run(go())
    assert order == [("early", 0), ("completion", 0)] + [("input", i) for i in range(20)] + [("timer", 0)]


def test_realtime_utterance_end_precedes_session_end(stack):
    """Regression: SESSION_END shares the last UTTERANCE_END timestamp; the utterance must end by its own
    UTTERANCE_END (reason 'endpoint'), with no spurious duplicate-end ERROR."""
    for _ in range(3):
        r = run_realtime(stack.cfg, FakeBackend(), stack.policy, sim.stream(["how high", "the wicks"], interval_ms=50))
        fin = of(r.events, "UTTERANCE_FINALIZED")
        assert [e.payload["reason"] for e in fin] == ["endpoint"]
        assert not of(r.events, "ERROR")
