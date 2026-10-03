"""Concurrency, isolation, event-loop health, shutdown and resource leaks (brief §15-17, §58, §63-65, §71)."""

import asyncio
import threading

import pytest

from grounding_helpers import requires_nli
from runtime_helpers import commits, drive, of, realtime, rt_stack, with_runtime
from streamrag.runtime import Fault, FaultInjector, SimulatedLLM, StreamingRuntime

pytestmark = requires_nli
ELIG_PROC = ["Tell me the eligibility requirements", "and the application process", "for the fixture permit."]


@pytest.fixture(scope="module")
def stack(tmp_path_factory):
    return rt_stack(tmp_path_factory, "corpus_grounding")


def _intervals(evs, kind=("lexical", "dense")):
    start = {e.payload["task_id"]: e.t_session_ms for e in of(evs, "TASK_STARTED") if e.payload["task_type"] in kind}
    end = {e.payload["task_id"]: e.t_session_ms for e in of(evs, "TASK_COMPLETED") if e.payload["task_id"] in start}
    return [(start[t], end[t]) for t in start if t in end]


def test_independent_retrievals_run_concurrently(stack):
    faults = FaultInjector([Fault("network", "delay", times=-1, delay_ms=60)])   # a remote index: 60 ms per call
    rt, evs = realtime(stack, [["Tell me the eligibility requirements and the application process for the fixture permit."]],
                       faults=faults, step_s=0.05)
    iv = _intervals(evs)
    overlap = max(sum(1 for a, b in iv if a <= t < b) for t, _ in iv)
    assert overlap >= 2                                       # lexical + dense (and both intents) in parallel
    assert rt.scheduler.pools["retrieval"].max_running <= stack.cfg.runtime.max_concurrent_retrievals


def test_bounded_concurrency_holds_across_sessions(stack):
    st = with_runtime(stack, max_concurrent_retrievals=2)
    faults = FaultInjector([Fault("network", "delay", times=-1, delay_ms=30)])

    async def main():
        rt = await StreamingRuntime(st.cfg, st, faults=faults).start()
        sids = [rt.start_session(f"s{i}") for i in range(4)]
        await asyncio.gather(*(drive(rt, s, [ELIG_PROC], 0.02, 0.0) for s in sids))
        await asyncio.gather(*(rt.complete_session(s, 60) for s in sids))
        await rt.shutdown()
        return rt
    rt = asyncio.run(main())
    pool = rt.scheduler.pools["retrieval"]
    assert pool.max_running == 2 and pool.max_pending > 0     # queued, never more than 2 running
    assert all(commits(rt.events(s)) for s in rt.sessions)


def test_concurrent_sessions_are_isolated(stack):
    """Two sessions on one runtime (one index), different questions, interleaved in time."""
    async def main():
        rt = await StreamingRuntime(stack.cfg, stack, llm=SimulatedLLM(latency_ms=200)).start()
        a, b = rt.start_session("A"), rt.start_session("B")
        await asyncio.gather(drive(rt, a, [["What are the eligibility requirements", "for the fixture permit?"]], 0.05),
                             drive(rt, b, [["What is the application fee", "for a new fixture permit?"]], 0.05))
        await asyncio.gather(rt.complete_session(a, 30), rt.complete_session(b, 30))
        await rt.shutdown()
        return rt
    rt = asyncio.run(main())
    ea, eb = rt.events("A"), rt.events("B")
    assert all(e.session_id == "A" for e in ea) and all(e.session_id == "B" for e in eb)
    def ids(evs, key):
        return {e.payload.get(key) for e in evs if e.payload.get(key)}
    tasks_a, tasks_b = ids(ea, "task_id"), ids(eb, "task_id")
    assert tasks_a and tasks_b and not tasks_a & tasks_b
    fa, fb = commits(ea)[-1].payload["text"], commits(eb)[-1].payload["text"]
    assert "18 years" in fa and "euros" in fb and "euros" not in fa
    la, lb = rt.sessions["A"].session.ledger, rt.sessions["B"].session.ledger
    assert la is not lb and {q.utterance_id for q in la.all()} == {"u1"}
    store_a = rt.sessions["A"].session.mi.engine.store
    assert store_a is not rt.sessions["B"].session.mi.engine.store


def test_event_loop_is_never_blocked(stack):
    """Drafts (NLI verification) and finals run on worker threads: the loop keeps ticking. Each answer takes the
    (simulated) model 600 ms - had it run on the loop, the loop would have been blocked at least that long. The
    bound is half of it, robust to scheduler noise (GC, other threads), but not to a blocking call."""
    rt, evs = realtime(stack, [ELIG_PROC, ["Is proof of residence mandatory?"]], llm=SimulatedLLM(latency_ms=600),
                       step_s=0.05)
    lags = sorted(rt.telemetry.loop_lag.lags_ms)
    assert len(lags) > 50
    assert lags[-1] < 300.0, f"loop blocked for {lags[-1]:.1f} ms"
    assert lags[int(0.95 * (len(lags) - 1))] < 20.0
    assert len(commits(evs)) == 2


def test_graceful_and_forced_shutdown_release_everything(stack):
    base_threads = threading.active_count()

    async def run(force: bool):
        rt = await StreamingRuntime(stack.cfg, stack, llm=SimulatedLLM(latency_ms=1500)).start()
        sid = rt.start_session("s1")
        await drive(rt, sid, [["How are applications", "for the fixture permit submitted?"]], 0.03, 0.2)
        busy = rt.scheduler.busy()
        out = await (rt.force_shutdown() if force else rt.shutdown(grace_ms=100))
        tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
        return rt, busy, out, tasks
    for force in (False, True):
        rt, busy, out, tasks = asyncio.run(run(force))
        assert busy and out["zombies"] == 0 and out["live_tokens"] == 0 and not tasks
        evs = rt.events("s1")
        assert of(evs, "SESSION_CANCELLED") and of(evs, "SESSION_CANCELLED")[0].payload["reason"] in (
            "shutdown", "force_shutdown")
        assert not rt.scheduler.busy() and all(not p.pending and not p.running for p in rt.scheduler.pools.values())
        with pytest.raises(RuntimeError):
            rt.start_session("late")
    assert threading.active_count() <= base_threads + 2       # only the shared retrieval service's own pool


def test_no_resource_growth_over_repeated_runs(stack):
    counts = []
    for _ in range(3):
        realtime(stack, [["How are applications", "for the fixture permit submitted?"]],
                 llm=SimulatedLLM(latency_ms=50), step_s=0.02)
        counts.append(threading.active_count())
    assert counts[1] == counts[2]                             # no thread leak run over run


def test_loop_lag_monitor_detects_a_blocking_call():
    """Negative control for the loop-health test: a blocking call on the loop is visible as lag."""
    import time
    from streamrag.runtime.telemetry import LoopLagMonitor

    async def main():
        mon = LoopLagMonitor(interval_ms=5)
        mon.start()
        await asyncio.sleep(0.05)
        time.sleep(0.25)                                           # what a synchronous LLM call would do
        await asyncio.sleep(0.05)
        await mon.stop()
        return mon.lags_ms
    assert max(asyncio.run(main())) >= 200.0
