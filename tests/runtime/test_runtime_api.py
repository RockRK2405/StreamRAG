"""StreamingRuntime API and integration (brief §60, §70-72, §55 stale-result race, §27 fast path, §66-67 security)."""

import asyncio
import inspect

import pytest

from grounding_helpers import echo, requires_nli, scripted
from runtime_helpers import commits, of, rt_stack, virtual, with_runtime
from streamrag.runtime import Fault, FaultInjector, RuntimeCapacityError, SimulatedLLM, StreamingRuntime

pytestmark = requires_nli
ELIG_PROC = ["Tell me the eligibility requirements", "and the application process", "for the fixture permit."]


@pytest.fixture(scope="module")
def stack(tmp_path_factory):
    return rt_stack(tmp_path_factory, "corpus_grounding")


@pytest.fixture(scope="module")
def orchard(tmp_path_factory):
    return rt_stack(tmp_path_factory, "corpus")


def test_session_api_capacity_and_validation(stack):
    st = with_runtime(stack, max_concurrent_sessions=2, max_chunk_chars=50)
    rt = StreamingRuntime(st.cfg, st, mode="virtual")
    a, b = rt.start_session("a"), rt.start_session("b")
    with pytest.raises(ValueError):
        rt.start_session("a")                                                     # duplicate id
    with pytest.raises(RuntimeCapacityError):
        rt.start_session("c")
    with pytest.raises(KeyError):
        rt.push_transcript_delta("zzz", "u1", "hello")
    assert not rt.push_transcript_delta(a, "u1", "x" * 51)                         # input limit
    rt.cancel_session(b)
    rt.start_session("c")                                                         # capacity freed by the cancel
    rt.run()
    rej = of(rt.events(a), "BACKPRESSURE_APPLIED")
    assert rej and rej[0].payload["reason"] == "chunk_too_long"
    params = inspect.signature(StreamingRuntime.push_transcript_delta).parameters
    assert not {"priority", "deadline", "pool", "task_type"} & set(params)        # never caller-controlled


def test_live_event_stream_is_ordered_and_terminates(stack):
    async def main():
        rt = await StreamingRuntime(stack.cfg, stack, llm=SimulatedLLM(latency_ms=100)).start()
        sid = rt.start_session("s1")
        got = []

        async def consume():
            async for ev in rt.get_events(sid):
                got.append(ev)
        consumer = asyncio.create_task(consume())
        for c in ELIG_PROC:
            rt.push_transcript_delta(sid, "u1", c)
            await asyncio.sleep(0.05)
        rt.end_utterance(sid, "u1")
        await rt.complete_session(sid, 30)
        await asyncio.wait_for(consumer, 5)
        order = rt.sessions[sid].summary()["output_order"]
        await rt.shutdown()
        return got, order
    got, order = asyncio.run(main())
    assert order["held_on_arrival"] == 0 and order["gaps_skipped"] == 0 and order["released"] == len(got)
    assert [e.output_seq for e in got] == list(range(len(got)))
    assert got[-1].type.value == "SESSION_CLOSED" and any(e.type.value == "ANSWER_COMMITTED" for e in got)


def test_end_to_end_pipeline_order(stack):
    """stream -> intent -> query -> retrieval -> evidence -> claim -> (draft) ... turn end -> generation ->
    verification -> citations -> validation -> answer stream."""
    rt, evs = virtual(stack, [ELIG_PROC], llm=scripted(echo))

    def ordered(events, stages):
        first = {}
        for i, e in enumerate(events):
            if e.type.value == "ANSWER_COMPLETED" and stages[-1] == "TURN_COMPLETED":
                first.setdefault("DRAFT_COMPLETED", i)
            first.setdefault(e.type.value, i)
        idx = [first[s] for s in stages]
        return idx == sorted(idx)
    listening = ["CHUNK_RECEIVED", "INTENT_DETECTED", "QUERY_GENERATED", "TASK_SCHEDULED", "RETRIEVAL_STARTED",
                 "RETRIEVAL_PARTIAL", "RETRIEVAL_COMPLETED", "CLAIM_CREATED", "EVIDENCE_FUSED", "DRAFT_COMPLETED",
                 "UTTERANCE_FINALIZED", "TURN_COMPLETED"]
    assert ordered(evs, listening)
    turn = of(evs, "TURN_COMPLETED")[0]
    sched = [e for e in of(evs, "TASK_SCHEDULED") if e.payload.get("answer_kind") == "final"][0]
    assert sched.seq < turn.seq                                     # requested while the turn completed
    after = evs[evs.index(sched):]
    assert ordered(after, ["TASK_SCHEDULED", "TURN_COMPLETED", "LLM_CALL", "CLAIMS_EXTRACTED", "CLAIM_VERIFIED", "CITATION_CREATED",
                           "CITATION_VALIDATED", "ANSWER_VALIDATED", "ANSWER_COMPLETED", "ANSWER_DELTA",
                           "ANSWER_COMMITTED", "SESSION_CLOSED"])
    assert turn.payload["answer"]["status"] == "PENDING" and commits(evs)[-1].payload["backend"] == "scripted"


def test_stale_result_never_overwrites_newer_state(orchard):
    """Q1 starts; Q2 (the correction) starts later and completes first; Q1 completes last. Q1 must not win."""
    st = with_runtime(orchard, cancel_running=False)                 # no cancellation: the race must be handled
    faults = FaultInjector([Fault("network", "delay", times=2, delay_ms=2500)])   # Q1's two subtasks are slow
    rt, evs = virtual(st, [["What are the rules for ladders", "in the orchard?"],
                           ["Sorry, I meant crates instead of ladders."]], faults=faults, gap=600)
    done = {e.query_id: e for e in of(evs, "RETRIEVAL_COMPLETED")}
    stale = of(evs, "STALE_RESULT_DISCARDED")
    assert stale and stale[0].payload["reason"] == "superseded"
    q1 = stale[0].payload["query_id"]
    later = [q for q, e in done.items() if q != q1 and e.seq < done[q1].seq]
    assert later, "a newer query must have completed before the stale one"
    rec = rt.sessions["s1"].session.ledger.get(q1)
    assert rec.stale_at_completion and done[q1].payload["stale"] is True
    store = rt.sessions["s1"].session.mi.engine.store
    active = [a for iid in rt.sessions["s1"].session.mi.tracker.intents
              for a in store.usable(iid) if store.assign.get((a.evidence_id, iid))]
    assert all(q1 not in str(a) for a in active)
    final = commits(evs)[-1].payload["text"].lower()
    assert "crate" in final and "ladder" not in final


def test_racing_answer_updates_commit_in_order(stack):
    st = with_runtime(stack, **{"sim_latency_ms.draft": 700.0})       # drafts slower than the provisional batches
    rt, evs = virtual(st, [ELIG_PROC], llm=scripted(echo))
    cs = commits(evs)
    vers = [e.payload["version"] for e in of(evs, "ANSWER_DELTA")]
    assert vers == sorted(vers) and cs and cs[-1].payload["status"] == "VALIDATED_FINAL"
    lane = rt.sessions["s1"].lane.stats
    assert lane["drafts_coalesced"] + lane["drafts_superseded"] >= 1    # superseded drafts never committed
    final_i = evs.index(cs[-1])
    assert not [e for e in of(evs[final_i:], "ANSWER_DELTA") if e.payload["status"] == "DRAFT"]


def test_result_arrival_order_does_not_change_the_answer(stack):
    _, fast = virtual(stack, [ELIG_PROC], llm=scripted(echo))
    _, slow = virtual(stack, [ELIG_PROC], llm=scripted(echo),
                      faults=FaultInjector([Fault("lexical", "delay", times=3, delay_ms=350)]))
    a, b = commits(fast)[-1].payload, commits(slow)[-1].payload
    assert sorted(a["claims"]) == sorted(b["claims"]) and a["text"] == b["text"]


def test_priorities_are_internal_and_documents_cannot_change_scheduling(tmp_path_factory):
    """The injection fixture tells "the assistant" to claim permits are free: retrieved text is data - it never
    reaches scheduling (priorities come only from configuration, deadlines only from the clock and the budget), the
    runtime configuration is unchanged, and the injected claim never becomes part of the answer."""
    inj = rt_stack(tmp_path_factory, "corpus_injection")
    before = inj.cfg.runtime.model_copy(deep=True)
    rt, evs = virtual(inj, [["Is the fixture permit office open on public holidays?"]])
    p = inj.cfg.runtime.priorities
    allowed = {p.final_answer, p.retrieval_final, p.retrieval_provisional, p.draft, p.validation_retrieval}
    sched = of(evs, "TASK_SCHEDULED")
    assert sched and {e.payload["priority"] for e in sched} <= allowed
    assert {e.payload["deadline_limited_by"] for e in sched if "deadline_limited_by" in e.payload} <= {
        "task_timeout", "turn_budget"}
    assert rt.rcfg == before and rt.sessions["s1"].session.cfg.runtime == before
    gen = [e for e in sched if e.payload.get("answer_kind") == "final"][0]
    assert gen.payload["task_type"] == "answer_extractive"           # extractive backend: no model, cpu pool
    assert gen.payload["priority"] == p.final_answer
    text = commits(evs)[-1].payload["text"].lower()
    assert "free" not in text and "never expire" not in text and "public holidays" in text


def test_fast_path_reuses_validated_state(orchard):
    q = ["How high should the wicks be trimmed?"]
    rt, evs = virtual(orchard, [q, q])
    u2 = [e for e in evs if e.utterance_id == "u2"]
    assert not [e for e in of(u2, "TASK_SCHEDULED") if e.payload["task_type"] in ("lexical", "dense")]
    assert of(u2, "QUERY_REUSED") or of(u2, "RETRIEVAL_SKIPPED")
    t1, t2 = commits(evs)[0].payload["text"], commits(evs)[-1].payload["text"]
    first_line = t1.split(" [")[0]
    assert first_line in t2                                          # the validated facts are reused verbatim
