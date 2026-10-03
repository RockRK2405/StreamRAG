"""Replay and event-sourced reconstruction of runtime sessions (brief §46-48)."""

import pytest

from grounding_helpers import echo, requires_nli, scripted
from runtime_helpers import commits, realtime, rt_stack, virtual
from streamrag.runtime import Fault, FaultInjector, SimulatedLLM, StreamingRuntime
from streamrag.runtime.replay import behaviour, inputs_from_runtime_trace, reconstruct, replay_runtime

pytestmark = requires_nli
ELIG_PROC = ["Tell me the eligibility requirements", "and the application process", "for the fixture permit."]


@pytest.fixture(scope="module")
def stack(tmp_path_factory):
    return rt_stack(tmp_path_factory, "corpus_grounding")


def test_virtual_trace_replays_exactly_from_the_event_log(stack):
    faults = FaultInjector([Fault("network", "transient", times=1), Fault("dense", "delay", times=1, delay_ms=500)])
    rt = StreamingRuntime(stack.cfg, stack, mode="virtual", llm=scripted(echo), faults=faults)
    sid = rt.start_session("s1")
    for i, c in enumerate(ELIG_PROC):
        rt.push_transcript_delta(sid, "u1", c, at_ms=300 * i)
    rt.push_transcript_delta(sid, "u1", "for the fixture permit, please.", replaces=2, at_ms=600)   # coalesced
    rt.end_utterance(sid, "u1", at_ms=1200)
    rt.push_transcript_delta(sid, "u2", "Is proof of residence mandatory?", at_ms=5000)
    rt.end_utterance(sid, "u2", at_ms=5400)
    rt.end_session(sid, at_ms=12000)
    rt.run()
    evs = rt.events(sid)
    assert any(e.type.value == "TRANSCRIPT_COALESCED" for e in evs) and any(e.type.value == "TASK_RETRIED" for e in evs)
    rep = replay_runtime(stack.cfg, stack, evs)
    assert rep["identical"], rep["differences"][:2]
    assert len(inputs_from_runtime_trace(evs)) >= 7


def test_reconstructed_state_matches_the_live_session(stack):
    rt, evs = virtual(stack, [ELIG_PROC, ["Is proof of residence mandatory?"]], llm=scripted(echo))
    state = reconstruct(evs)
    live = rt.sessions["s1"].session
    for rec in live.ledger.all():
        assert state["queries"][rec.query_id]["status"] == rec.status
        assert state["queries"][rec.query_id]["text"] == rec.query_text
    finals = rt.sessions["s1"].lane.finals
    assert {a["utterance_id"]: sorted(a["claims"]) for a in state["answers"].values()} == {
        u: sorted(c.claim_id for c in ga.claims) for u, ga in finals.items()}


def test_realtime_trace_replays_its_orchestration(stack):
    rt, evs = realtime(stack, [["How are applications for the fixture permit submitted?"]],
                       llm=SimulatedLLM(latency_ms=100), step_s=0.05)
    rep = replay_runtime(stack.cfg, stack, evs)
    assert rep["behaviour_identical"]
    assert behaviour(evs)["u1"]["answer_claims"] == behaviour(rep["events"])["u1"]["answer_claims"]
    assert commits(rep["events"])[-1].payload["text"] == commits(evs)[-1].payload["text"]
