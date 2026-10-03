"""Phase 6 inside the streaming session (virtual / realtime), telemetry, replay, full-restart baseline equivalence.
TEST FIXTURE corpus (fictional; hashing embedder)."""

import pytest

from streamrag.models.events import EventType as E
from streamrag.replay import ReplayEngine
from streamrag.replay.replay import behavior_signature
from streamrag.streaming import run_realtime

from session_helpers import of, p6_stack, run_turns, session_cfg, stream_turns  # noqa: F401  (fixture)
from streaming_helpers import hashing_stack

TURNS = [["What are the rules", "for ladders", "in the orchard?"], ["Specifically", "overnight."], ["Okay."],
         ["Actually, ignore", "the overnight restriction."], ["Sorry, I meant", "crates instead of ladders."]]

PHASE6_EVENTS = {"SESSION_VERSION_CREATED", "CONTEXT_CHANGE_DETECTED", "DELTA_PLAN_CREATED", "QUERY_REUSED",
                 "QUERY_SUPERSEDED", "EVIDENCE_RETAINED", "EVIDENCE_INVALIDATED", "EVIDENCE_REVALIDATED",
                 "CLAIM_CREATED", "CLAIM_INVALIDATED", "CLAIM_REVALIDATED", "ANSWER_VERSION_CREATED",
                 "ANSWER_VERSION_UPDATED"}


@pytest.fixture(scope="module")
def run(p6_stack):
    return stream_turns(p6_stack, TURNS)


def test_all_phase6_events_with_ids(run):
    seen = {e.type.value for e in run.events}
    assert PHASE6_EVENTS <= seen
    assert {t.value for t in E} >= PHASE6_EVENTS
    for e in run.events:
        t, p = e.type.value, e.payload
        if t == "CONTEXT_CHANGE_DETECTED":
            assert p["change_id"].startswith("CH") and e.utterance_id
        elif t == "DELTA_PLAN_CREATED":
            assert p["plan_id"].startswith("P") and p["change_ids"]
        elif t.startswith("EVIDENCE_") and t in PHASE6_EVENTS:
            assert p["evidence_id"] and p["intent_id"] and e.intent_id == p["intent_id"]
        elif t.startswith("CLAIM_"):
            assert p["claim_id"].startswith("C")
        elif t in ("QUERY_REUSED", "QUERY_SUPERSEDED"):
            assert e.query_id
        elif t.startswith("ANSWER_VERSION"):
            assert p["answer_id"].startswith("A") and p["frame_id"] and "diff" in p


def test_turn_payload_reports_the_targeted_update(run):
    turns = {e.utterance_id: e.payload for e in of(run.events, "TURN_COMPLETED")}
    s = {u: p["session"] for u, p in turns.items()}
    assert [c["type"] for c in s["u2"]["changes"]] == ["CONSTRAINT_ADDITION"] and s["u2"]["answer_changed"]
    assert [c["type"] for c in s["u3"]["changes"]] == ["NO_CHANGE"] and not s["u3"]["answer_changed"]
    assert [c["type"] for c in s["u4"]["changes"]] == ["CONSTRAINT_REMOVAL"]
    assert [c["type"] for c in s["u5"]["changes"]] == ["CORRECTION"]
    assert turns["u2"]["sub_queries"] == ["What are the rules for ladders in the orchard overnight"]
    assert turns["u4"]["reused_queries"] and turns["u4"]["sub_queries"] == ["What are the rules for ladders in the orchard"]
    assert [i["intent_id"] for i in turns["u2"]["unified_evidence"]["per_intent"]] == ["I1"]   # cross-turn need fused
    assert turns["u3"]["unified_evidence"] is None
    answers = [e.payload["answer_id"] for e in run.events if e.type.value.startswith("ANSWER_VERSION")]
    assert answers == ["A1", "A2", "A3", "A4"]
    first_turn = next(i for i, e in enumerate(run.events) if e.type.value == "TURN_COMPLETED")
    assert any(e.type.value == "ANSWER_VERSION_CREATED" for e in run.events[:first_turn])     # answer precedes turn


def test_mid_stream_fragment_does_not_retrieve(run):
    """'Specifically' alone (chunk 1 of u2) carries no content: no need, no query until 'overnight' arrives."""
    q = [e for e in of(run.events, "QUERY_GENERATED") if e.utterance_id == "u2"]
    assert [e.payload["query_text"] for e in q] == ["What are the rules for ladders in the orchard overnight"]
    assert q[0].payload["parent_query_id"] and q[0].payload["derived_from_change_id"]


def test_virtual_replay_is_exact_including_session_state(p6_stack, run):
    rep = ReplayEngine(p6_stack.cfg, p6_stack.service, p6_stack.policy, p6_stack.index_hash,
                       p6_stack.intent_stack).replay(run.events)
    assert rep.identical, rep.summary()
    sig = behavior_signature(run.events)
    assert sig["delta_plans"] and sig["claims_final"] and sig["evidence_final"]
    assert [x[1] for x in sig["per_utterance"]["u4"] if x[0] == "change"] == ["CONSTRAINT_REMOVAL"]


def test_realtime_trace_replays_with_identical_behaviour(fixture_bundle):
    cfg, b = fixture_bundle
    st = hashing_stack(session_cfg(cfg).model_copy(update={"streaming": cfg.streaming.model_copy(
        update={"speed": 50.0})}), b)
    from session_helpers import SessionEnd, SessionStart, sim
    evs = [SessionStart(session_id="sim")]
    off = 0.0
    for n, chunks in enumerate(TURNS[:2], 1):
        evs += sim.stream(chunks, interval_ms=400, utterance_id=f"u{n}", offset_ms=off, wrap_session=False)
        off += 400 * len(chunks) + 2500
    evs.append(SessionEnd(session_id="sim"))
    r = run_realtime(st.cfg, st.service, st.policy, evs, index_hash=st.index_hash, intent_stack=st.intent_stack)
    rep = ReplayEngine(st.cfg, st.service, st.policy, st.index_hash, st.intent_stack).replay(r.events)
    assert rep.behavior_identical, rep.behavior_differences[:3]


def test_guarded_retrieval_is_deferred_not_lost(fixture_bundle):
    cfg, b = fixture_bundle
    c = session_cfg(cfg)
    c = c.model_copy(update={"multi_intent": c.multi_intent.model_copy(update={"max_queries_per_intent": 1})})
    st = hashing_stack(c, b)
    run = stream_turns(st, [["What are the rules", "for ladders", "in the orchard?"]])
    skipped = [e for e in of(run.events, "RETRIEVAL_SKIPPED") if e.payload.get("deferred")]
    assert skipped and all(e.payload["reason"] == "intent_budget_exhausted" for e in skipped)
    turn = of(run.events, "TURN_COMPLETED")[0].payload
    assert turn["session"]["deferred"] == ["I1"]                      # honest: the final query never ran
    assert len(of(run.events, "QUERY_GENERATED")) == 1


def test_session_mode_off_keeps_phase5_behaviour(fixture_bundle):
    cfg, b = fixture_bundle
    c = session_cfg(cfg)
    c = c.model_copy(update={"session": c.session.model_copy(update={"enabled": False})})
    run = stream_turns(hashing_stack(c, b), TURNS[:2])
    assert not ({e.type.value for e in run.events} & PHASE6_EVENTS)
    assert all("session" not in e.payload for e in of(run.events, "TURN_COMPLETED"))


def test_full_restart_reaches_the_same_final_queries(p6_stack):
    turns = ["What are the rules for ladders in the orchard?", "Specifically overnight.", "Okay.",
             "Actually, ignore the overnight restriction.", "What about crates?",
             "Now explain how the telescope is recalibrated."]
    pi, ri = run_turns(p6_stack, turns)
    pr, rr = run_turns(p6_stack, turns, restart=True)
    for a, b in zip(ri, rr):
        assert [c.change_type for c in a.changes] == [c.change_type for c in b.changes]
        if b.gate.startswith("closed"):                        # "Okay.": neither pipeline does anything
            assert a.retrievals == b.retrievals == 0
            continue
        served = {k: v for k, v in b.queries.items() if v}     # the restart re-retrieves the current frame's needs
        assert served and all(a.queries[k] == v for k, v in served.items())
    assert set(served) == set(pr.last.engine.frames.active.intent_ids) == {"I3"}
    assert sum(r.retrievals for r in ri) < sum(r.retrievals for r in rr)
