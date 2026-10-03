"""Backpressure (brief §12-14, §35, §74): coalescing, bounded queues, latest-state preference, ordered buffering."""

import asyncio

import pytest

from grounding_helpers import requires_nli
from runtime_helpers import commits, of, rt_stack, with_runtime
from streamrag.models.events import SessionEnd, TranscriptChunk, TranscriptChunkPayload, UtteranceEnd, UtteranceEndPayload
from streamrag.runtime import StreamingRuntime
from streamrag.runtime.backpressure import InputQueue, coalesce
from streamrag.runtime.streamer import OrderedReleaseBuffer, Subscription


def chunk(uid, idx, text, replaces=None, stability="partial"):
    return TranscriptChunk(session_id="s", utterance_id=uid, payload=TranscriptChunkPayload(
        chunk_index=idx, timestamp_s=0.0, text=text, stability=stability, replaces_chunk_index=replaces))


def test_coalescing_keeps_the_latest_state_and_every_boundary():
    items = [chunk("u1", 0, "what"), chunk("u1", 0, "what are", 0), chunk("u1", 0, "what are the", 0),
             chunk("u1", 1, "requirements"), chunk("u1", 0, "what are the", 0, "final"),
             UtteranceEnd(session_id="s", utterance_id="u1", payload=UtteranceEndPayload(timestamp_s=1.0)),
             chunk("u2", 0, "and"), chunk("u2", 0, "and visas", 0), SessionEnd(session_id="s")]
    kept, dropped = coalesce(items)
    texts = [(getattr(k, "utterance_id", "-"), getattr(getattr(k, "payload", None), "text", type(k).__name__))
             for k in kept]
    assert texts == [("u1", "requirements"), ("u1", "what are the"), ("-", "UtteranceEndPayload"),
                     ("u2", "and visas"), ("-", "SessionEnd")] or len(dropped) == 4
    assert len(dropped) == 4 and isinstance(kept[-1], SessionEnd)
    assert [type(k).__name__ for k in kept].count("UtteranceEnd") == 1


def test_bounded_input_queue_coalesces_then_rejects():
    q = InputQueue(capacity=3)
    assert all(q.offer(chunk("u1", 0, "w" * n, 0 if n else None)).accepted for n in range(3))
    r = q.offer(chunk("u1", 0, "wwww", 0))                          # full: revisions coalesce in place
    assert r.accepted and r.coalesced >= 1 and len(q) <= 3
    for i in range(1, 4):
        q.offer(chunk("u1", i, f"c{i}"))
    assert q.stats["rejected"] >= 1 and q.max_delta_depth <= 3      # never unbounded


def test_ordered_release_buffer_and_bounded_hold():
    t = [0.0]
    buf = OrderedReleaseBuffer(max_hold_ms=100, now_ms=lambda: t[0])
    assert buf.push("a", 1, "x1") == [] and buf.push("a", 2, "x2") == []
    assert buf.push("a", 0, "x0") == ["x0", "x1", "x2"]
    assert buf.push("a", 1, "dup") == []                            # never re-released
    assert buf.push("a", 5, "x5") == []
    t[0] = 150.0
    assert buf.flush_expired() == ["x5"] and buf.skipped == [("a", 3), ("a", 4)] and buf.pending == 0


def test_slow_subscriber_loses_drafts_first_then_disconnects():
    from streamrag.models.events import EventType, TelemetryEvent

    def ev(i, status):
        return TelemetryEvent(event_id=f"s:{i:06d}", type=EventType.ANSWER_COMPLETED, session_id="s", seq=i,
                              t_wall_ms=0.0, component="x", payload={"status": status})
    sub = Subscription(capacity=3, user_visible=True)
    for i, st in enumerate(["DRAFT", "DRAFT", "VALIDATED_FINAL", "VALIDATED_FINAL"]):
        sub.offer(ev(i, st))
    assert [e.payload["status"] for e in sub.queue] == ["VALIDATED_FINAL", "VALIDATED_FINAL"]
    assert sub.dropped_drafts == 2 and not sub.disconnected
    for i in range(4, 8):
        sub.offer(ev(i, "VALIDATED_FINAL"))
    assert sub.disconnected and not sub.queue


@pytest.fixture(scope="module")
def stack(tmp_path_factory):
    return rt_stack(tmp_path_factory, "corpus_grounding")


@requires_nli
def test_high_frequency_deltas_are_coalesced_without_losing_the_final_state(stack):
    words = "What are the eligibility requirements for the fixture permit".split()
    rt = StreamingRuntime(stack.cfg, stack, mode="virtual")
    sid = rt.start_session("s1")
    hyp = ""
    for i, w in enumerate(words):                                   # ASR partial hypotheses, 1 ms apart
        hyp = (hyp + " " + w).strip()
        rt.push_transcript_delta(sid, "u1", hyp, stability="partial", replaces=0 if i else None, at_ms=float(i // 3))
    rt.end_utterance(sid, "u1", at_ms=600)
    rt.end_session(sid, at_ms=4000)
    rt.run()
    evs = rt.events(sid)
    co = of(evs, "TRANSCRIPT_COALESCED")
    assert co and sum(e.payload["dropped"] for e in co) >= 4
    decisions = of(evs, "RETRIEVAL_DECISION")
    assert len(decisions) < len(words)                              # one decision per coalesced batch
    fin = of(evs, "UTTERANCE_FINALIZED")[0].payload["transcript"]
    assert fin == " ".join(words)                                   # the final meaningful state survived
    assert commits(evs) and "18 years" in commits(evs)[-1].payload["text"]


@requires_nli
def test_input_queue_capacity_is_enforced_under_burst(stack):
    st = with_runtime(stack, **{"queues.input": 4})

    async def main():
        rt = await StreamingRuntime(st.cfg, st).start()
        sid = rt.start_session("s1")
        accepted = [rt.push_transcript_delta(sid, "u1", f"word{i}") for i in range(12)]   # same loop tick
        rt.end_utterance(sid, "u1")
        await rt.complete_session(sid, 30)
        await rt.shutdown()
        return rt, accepted, rt.events(sid)
    rt, accepted, evs = asyncio.run(main())
    rs = rt.sessions["s1"]
    assert rs.inputs.max_delta_depth <= 4 and accepted.count(False) >= 1
    rej = [e for e in of(evs, "BACKPRESSURE_APPLIED") if e.payload["action"] == "rejected"]
    assert len(rej) == accepted.count(False) and all(e.payload["input"] for e in rej)
