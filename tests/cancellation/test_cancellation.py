"""Cooperative cancellation (brief §18-20, §71): token tree, pending vs running work, superseded retrieval,
cancellation during generation, session reset / cancel during active tasks."""

import asyncio

import pytest

from grounding_helpers import requires_nli
from runtime_helpers import commits, of, rt_stack, virtual, with_runtime
from streamrag.runtime import Fault, FaultInjector, SimulatedLLM, StreamingRuntime
from streamrag.runtime.cancellation import CancellationManager, CancellationToken, TaskCancelled

pytestmark = requires_nli


def test_token_tree_and_callbacks():
    m = CancellationManager()
    t1, t2 = m.task_token("T1", "s"), m.task_token("T2", "s")
    fired = []
    t1.add_callback(lambda reason: fired.append(reason))
    assert m.cancel_task("T1", "superseded") and t1.is_cancelled() and not t2.is_cancelled()
    assert fired == ["superseded"] and not m.cancel_task("T1", "again")
    assert m.cancel_session("s", "reset") == 1 and t2.is_cancelled() and t2.reason == "reset"
    with pytest.raises(TaskCancelled):
        t2.raise_if_cancelled()
    late = CancellationToken(t1)                    # a child of an already-cancelled parent starts cancelled
    assert late.is_cancelled()
    m.finished("T1"), m.finished("T2")
    assert m.live_tokens == 0


@pytest.fixture(scope="module")
def stack(tmp_path_factory):
    return rt_stack(tmp_path_factory, "corpus")


def test_superseded_running_retrieval_is_cancelled(stack):
    # the first query's dense search is slow (remote index): the correction arrives while it runs
    faults = FaultInjector([Fault("dense", "delay", times=1, delay_ms=900)])
    rt, evs = virtual(stack, [["What are the rules for ladders in the orchard?",
                               "Sorry, I meant crates instead of ladders."]], faults=faults)
    cancelled = [e for e in of(evs, "RETRIEVAL_CANCELLED") if e.payload["reason"].endswith("in_flight")]
    assert cancelled, "the superseded query must be cancelled while running"
    q = cancelled[0].query_id
    rec = rt.sessions["s1"].session.ledger.get(q)
    assert rec.status == "cancelled" and rec.retrieval_status == "cancelled_running"
    assert not [e for e in of(evs, "RETRIEVAL_COMPLETED") if e.query_id == q]      # never committed
    assert any(e.payload["task_type"] == "dense" and e.payload["error"].endswith("in_flight")
               for e in of(evs, "TASK_CANCELLED"))
    final = commits(evs)[-1].payload["text"]
    assert "crate" in final.lower()


def test_correction_cancels_generation_in_flight(stack):
    async def main():
        rt = await StreamingRuntime(stack.cfg, stack, llm=SimulatedLLM(latency_ms=1500)).start()
        sid = rt.start_session("s1")
        for c in ["What are the rules", "for ladders", "in the orchard?"]:
            rt.push_transcript_delta(sid, "u1", c)
            await asyncio.sleep(0.08)
        rt.end_utterance(sid, "u1")
        await asyncio.sleep(0.4)                                    # u1's final is generating (1.5 s)
        for c in ["Sorry, I meant crates", "instead of ladders."]:
            rt.push_transcript_delta(sid, "u2", c)
            await asyncio.sleep(0.08)
        rt.end_utterance(sid, "u2")
        await rt.complete_session(sid, 30)
        await rt.shutdown()
        return rt, rt.events(sid)
    rt, evs = asyncio.run(main())
    gen_cancel = [e for e in of(evs, "TASK_CANCELLED") if e.payload["task_type"] == "generation"]
    assert gen_cancel and gen_cancel[0].payload["error"] == "correction"
    assert [c.utterance_id for c in commits(evs)] == ["u2"]       # the obsolete answer never became visible
    assert rt.sessions["s1"].lane.stats["cancelled_by_correction"] >= 1


def test_session_reset_and_cancel_during_active_tasks(stack):
    async def main():
        rt = await StreamingRuntime(stack.cfg, stack, llm=SimulatedLLM(latency_ms=2000)).start()
        sid = rt.start_session("s1")
        for c in ["What are the rules", "for ladders", "in the orchard?"]:
            rt.push_transcript_delta(sid, "u1", c)
            await asyncio.sleep(0.05)
        rt.end_utterance(sid, "u1")
        await asyncio.sleep(0.3)
        rs = rt.sessions[sid]
        busy_before = rt.scheduler.busy(lambda t: t.session_id == sid)
        rt.reset_session(sid)                                      # generation still running
        epoch = rs.state.epoch
        for c in ["How are crates handled", "at harvest?"]:
            rt.push_transcript_delta(sid, "u2", c)
            await asyncio.sleep(0.05)
        rt.end_utterance(sid, "u2")
        await rt.complete_session(sid, 30)
        sid2 = rt.start_session("s2")
        rt.push_transcript_delta(sid2, "u1", "How is the lens polished?")
        await asyncio.sleep(0.2)
        rt.cancel_session(sid2, "client_cancelled")
        n2 = len(rt.events(sid2))
        await asyncio.sleep(0.3)
        await rt.shutdown()
        return rt, rt.events(sid), busy_before, epoch, rt.events(sid2), n2
    rt, evs, busy_before, epoch, evs2, n2 = asyncio.run(main())
    assert busy_before and epoch == 1
    reset = of(evs, "SESSION_RESET")[0]
    assert reset.payload["tasks_cancelled"] >= 1
    after = evs[evs.index(reset):]
    assert [c.utterance_id for c in commits(after)] == ["u2"] and not commits(evs[:evs.index(reset)])
    assert all(e.payload.get("reason") == "epoch_changed" for e in of(after, "STALE_RESULT_DISCARDED"))
    assert of(evs2, "SESSION_CANCELLED") and len(evs2) == n2       # nothing after the cancellation
    assert rt.cancellation.live_tokens == 0


def test_cancelled_pending_work_never_runs(stack):
    st = with_runtime(stack, max_concurrent_retrievals=1)
    faults = FaultInjector([Fault("lexical", "delay", times=1, delay_ms=500)])
    rt, evs = virtual(st, [["What are the rules for ladders in the orchard?",
                            "Sorry, I meant crates instead of ladders."]], faults=faults)
    never_started = [e for e in of(evs, "TASK_CANCELLED") if e.payload["queue_wait_ms"] is None]
    started = {e.payload["task_id"] for e in of(evs, "TASK_STARTED")}
    assert never_started and not {e.payload["task_id"] for e in never_started} & started


def test_turn_superseded_by_a_refining_utterance_is_not_answered(tmp_path_factory):
    """u1's query is still running when u2 refines the same need: u1's query is cancelled and u1 is not answered
    with "no evidence" - its answer is superseded by u2's (found in the Phase 8 end-to-end demo)."""
    st = rt_stack(tmp_path_factory, "corpus_grounding")
    faults = FaultInjector([Fault("dense", "delay", times=-1, delay_ms=1200)])
    rt = StreamingRuntime(st.cfg, st, mode="virtual", faults=faults)
    sid = rt.start_session("s1")
    for i, c in enumerate(["What are the eligibility", "requirements", "for the fixture permit?"]):
        rt.push_transcript_delta(sid, "u1", c, at_ms=350 * i)
    rt.end_utterance(sid, "u1", at_ms=1050)
    for i, c in enumerate(["For international", "applicants."]):
        rt.push_transcript_delta(sid, "u2", c, at_ms=1650 + 350 * i)
    rt.end_utterance(sid, "u2", at_ms=2400)
    rt.end_session(sid, at_ms=9000)
    rt.run()
    evs = rt.events(sid)
    turns = {e.utterance_id: e.payload["answer"] for e in of(evs, "TURN_COMPLETED")}
    assert turns["u1"]["status"] == "SUPERSEDED" and turns["u1"]["superseded_by"] == "u2"
    assert [c.utterance_id for c in commits(evs)] == ["u2"]
    assert not any("do not contain an answer" in c.payload["text"] for c in commits(evs))
    assert of(evs, "RETRIEVAL_CANCELLED")
