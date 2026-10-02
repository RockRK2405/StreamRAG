"""Integration: stream -> controller -> retrieval -> evidence -> ledger (virtual mode, deterministic).
Covers Phase 4 brief CASES 4-10."""

import pytest

from streamrag.models import TranscriptChunk, UtteranceEnd
from streamrag.streaming import run_virtual, utterance_stats
from streamrag.streaming import simulator as sim

from streaming_helpers import FakeBackend, hashing_stack, types, with_controller, with_streaming


@pytest.fixture(scope="module")
def stack(fixture_bundle):
    cfg, b = fixture_bundle
    return hashing_stack(cfg, b)


def run(stack, inputs, cfg=None, backend=None, policy=None):
    return run_virtual(cfg or stack.cfg, backend or stack.service, policy or stack.policy, inputs, index_hash=stack.index_hash)


def of(events, t):
    return [e for e in events if e.type.value == t]


def test_end_to_end_evidence_reaches_ledger_and_turn(stack):
    r = run(stack, sim.stream(["so I wanted to ask", "how high the wicks", "should be trimmed"], interval_ms=400))
    ts = types(r.events)
    assert ts[0] == "SESSION_STARTED" and ts[-1] == "SESSION_CLOSED" and "TURN_COMPLETED" in ts
    done = of(r.events, "RETRIEVAL_COMPLETED")
    assert done and all(d.payload["evidence_ids"] for d in done)
    led = r.session.ledger
    assert all(rec.status == "completed" and rec.evidence_ids for rec in led.all())
    turn = of(r.events, "TURN_COMPLETED")[0].payload
    assert turn["final_query_id"] == led.active("u1").query_id and turn["answer"] is None
    assert turn["retrieval_events"][0]["trigger"] == "provisional"
    seq = [e.t_session_ms for e in r.events]
    assert seq == sorted(seq)                                              # timestamps never go backwards


def test_case4_query_evolves_new_version(stack):
    r = run(stack, sim.stream(["how high the wicks", "should be trimmed", "in the lighthouse"], interval_ms=500))
    qs = of(r.events, "QUERY_UPDATED")
    assert len(qs) >= 2 and qs[1].payload["supersedes"] == qs[0].payload["query_id"]
    assert qs[1].payload["relation"] == "refines" and r.session.ledger.get(qs[0].payload["query_id"]).stale


def test_case5_duplicate_chunk_not_reprocessed(stack):
    evs = sim.stream(["how high the wicks", "should be trimmed"], interval_ms=500)
    dup = TranscriptChunk(session_id="sim", utterance_id="u1", payload=evs[1].payload.model_copy(update={"timestamp_s": 0.1}))
    r = run(stack, evs[:2] + [dup] + evs[2:])
    recv = of(r.events, "CHUNK_RECEIVED")
    assert [e.payload["status"] for e in recv] == ["accepted", "duplicate", "accepted"]
    assert len(of(r.events, "TRANSCRIPT_UPDATED")) == 2                 # duplicate produced no update/decision


def test_case6_out_of_order_chunk_explicit(stack):
    evs = sim.stream(["how high", "the wicks", "should be trimmed"], interval_ms=300)
    a, b, c = (e for e in evs if isinstance(e, TranscriptChunk))
    c2 = TranscriptChunk(session_id="sim", utterance_id="u1", payload=c.payload.model_copy(update={"timestamp_s": 0.3}))
    b2 = TranscriptChunk(session_id="sim", utterance_id="u1", payload=b.payload.model_copy(update={"timestamp_s": 0.6}))
    end = next(e for e in evs if isinstance(e, UtteranceEnd))
    r = run(stack, [evs[0], a, c2, b2, end, evs[-1]])
    assert [e.payload["status"] for e in of(r.events, "CHUNK_RECEIVED")] == ["accepted", "gap", "out_of_order"]
    assert r.session.chunks.get_current_transcript("u1") == "how high the wicks should be trimmed"


def test_case7_stale_completion_keeps_evidence_with_lineage(stack):
    slow = with_streaming(stack.cfg, sim_retrieval_latency_ms=1500)
    r = run(stack, sim.stream(["how high the wicks", "should be trimmed", "in the lighthouse", "please"], interval_ms=600), cfg=slow)
    stale_done = [e for e in of(r.events, "RETRIEVAL_COMPLETED") if e.payload["stale"]]
    assert stale_done, "an early query must complete after being superseded"
    q = r.session.ledger.get(stale_done[0].payload["query_id"])
    assert q.stale and q.superseded_by and q.evidence_ids and q.stale_at_completion
    view = r.session.ledger.evidence_view("u1")
    assert any(v["stale"] for v in view) or all(set(q.evidence_ids) <= {v["evidence_id"] for v in view} for _ in [0])


def test_case8_rapid_chunks_no_retrieval_storm(stack):
    words = "how high exactly should the wicks in the lamp of the lighthouse be trimmed by the keeper".split()
    r = run(stack, sim.stream(words, interval_ms=50))
    st = utterance_stats(r.events)["u1"]
    assert st["retrieval_count"] <= stack.cfg.controller.max_retrievals_per_utterance
    assert st["decisions"].get("WAIT", 0) >= len(words) // 2 and st["duplicate_retrievals"] == 0


def test_case9_utterance_ends_right_after_retrieval_begins(stack):
    evs = sim.stream(["so I wanted to ask", "how high the wicks"], interval_ms=1000, end_gap_ms=20)
    r = run(stack, evs)
    st = utterance_stats(r.events)["u1"]
    start = of(r.events, "RETRIEVAL_STARTED")[0].t_session_ms
    fin = of(r.events, "UTTERANCE_FINALIZED")[0].t_session_ms
    assert start == 1000.0 and fin == 1020.0 and st["lead_time_ms"] == pytest.approx(20.0)
    assert st["retrieved_early"] and st["ttfr_ms"] == pytest.approx(1000.0)


def test_case10_backend_failure_is_structured(stack):
    r = run(stack, sim.stream(["how high the wicks", "should be trimmed"], interval_ms=500), backend=FakeBackend(fail_on="wicks"))
    failed = [e for e in of(r.events, "RETRIEVAL_COMPLETED") if e.payload["status"] == "error"]
    errors = [e for e in of(r.events, "ERROR") if e.payload["component"] == "async_retriever"]
    assert failed and errors and errors[0].payload["recoverable"] and errors[0].payload["action"] == "keep_previous_evidence"
    assert any(rec.status == "failed" for rec in r.session.ledger.all()) and "TURN_COMPLETED" in types(r.events)


def test_queued_superseded_query_is_cancelled(stack):
    cfg = with_controller(with_streaming(stack.cfg, sim_retrieval_latency_ms=2000), max_concurrent_retrievals=1,
                          retrieval_cooldown_ms=0)
    backend = FakeBackend()
    r = run(stack, sim.stream(["how high the wicks", "should be trimmed", "in the lighthouse"], interval_ms=500),
            cfg=cfg, backend=backend)
    cancelled = of(r.events, "RETRIEVAL_CANCELLED")
    assert cancelled and cancelled[0].payload["reason"] == "superseded_before_start"
    assert r.session.ledger.get(cancelled[0].payload["query_id"]).status == "cancelled"
    assert len(backend.calls) == len(of(r.events, "RETRIEVAL_STARTED"))   # cancelled query never ran


def test_endpoint_timeout_finalizes_without_utterance_end(stack):
    evs = [e for e in sim.stream(["how high the wicks"], interval_ms=100) if not isinstance(e, UtteranceEnd)]
    r = run(stack, evs)
    fin = of(r.events, "UTTERANCE_FINALIZED")[0]
    assert fin.payload["reason"] in {"timeout", "session_end"}


def test_suppressed_utterance_has_no_retrieval_events(stack):
    r = run(stack, sim.stream(["could you", "repeat the previous answer", "please"], interval_ms=400))
    assert not of(r.events, "RETRIEVAL_STARTED") and of(r.events, "RETRIEVAL_SKIPPED")
    assert all(e.payload["skip_kind"] in {"suppressed", "not_worthy", None} for e in of(r.events, "RETRIEVAL_SKIPPED"))


def test_all_events_schema_valid_and_required_fields(stack):
    r = run(stack, sim.stream(["how high the wicks", "should be trimmed"], interval_ms=500))
    for e in r.events:
        assert e.event_id and e.type and e.session_id and e.t_session_ms is not None
        if e.type.value not in ("SESSION_STARTED", "SESSION_CLOSED"):
            assert e.utterance_id == "u1"
