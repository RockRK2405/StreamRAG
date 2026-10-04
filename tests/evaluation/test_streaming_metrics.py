"""Streaming / latency milestones on synthetic event logs with known answers."""

from types import SimpleNamespace as NS

from streamrag.evaluation.metrics import latency, streaming


def ev(t, typ, uid="u1", seq=None, **payload):
    return NS(type=NS(value=typ), utterance_id=uid, t_wall_ms=t, payload=payload, output_seq=seq)


EVENTS = [ev(1000, "CHUNK_RECEIVED"), ev(1005, "RETRIEVAL_DECISION", seq=1),
          ev(1020, "RETRIEVAL_COMPLETED", seq=2, evidence_ids=["a"]), ev(1100, "ANSWER_COMPLETED", seq=3, text="draft"),
          ev(1300, "UTTERANCE_FINALIZED", seq=4), ev(1500, "ANSWER_COMPLETED", seq=5, text="It costs 72 euros."),
          ev(1600, "ANSWER_COMMITTED", seq=6, text="It costs 72 euros."), ev(1610, "TURN_COMPLETED", seq=7)]


def test_milestones():
    m = latency.milestones(EVENTS, 1000, "u1")
    assert (m["ttft"], m["ttfe"], m["ttfa"], m["ttva"], m["total"]) == (5, 20, 100, 600, 610)
    assert m["utterance_end"] == 300 and m["ttva_after_end"] == 300 and m["ttfa_after_end"] == -200


def test_turn_stream_metrics():
    t = streaming.turn(EVENTS, 1000, "u1", [["72 euros"]])
    assert t["answer_updates"] == 3 and t["update_gaps_ms"] == [400, 100]
    assert t["time_to_useful_ms"] == 500 and t["time_to_final_ms"] == 600 and t["completed"]


def test_ordering_errors_and_stale_rate():
    bad = EVENTS[:3] + [ev(1050, "STALE_RESULT_DISCARDED", seq=2)] + EVENTS[3:]
    s = streaming.session(bad)
    assert s["ordering_errors"] == 1 and s["stale_updates"] == 1 and s["stale_update_rate"] == 1 / 3


def test_turn_evidence_reads_each_turns_fused_evidence():
    """Runtime turn evidence comes from the turn's own EVIDENCE_FUSED events (not the end-of-session ledger)."""
    from streamrag.evaluation.systems import turn_evidence

    def ev(seq, uid, typ, items=None):
        return {"seq": seq, "utterance_id": uid, "type": typ,
                "payload": {"items": [{"evidence_id": i} for i in (items or [])]}}
    evs = [ev(1, "u1", "EVIDENCE_FUSED", ["a", "b"]), ev(2, "u1", "EVIDENCE_FUSED", ["c", "a"]),
           ev(3, "u1", "ANSWER_COMMITTED"), ev(4, "u1", "EVIDENCE_FUSED", ["late"]),
           ev(5, "u2", "EVIDENCE_FUSED", ["x"]), ev(6, "u2", "ANSWER_COMMITTED")]
    assert turn_evidence(evs, "u1") == ["c", "a", "b"]      # latest fusion first, then reused items; not after commit
    assert turn_evidence(evs, "u2") == ["x"]                 # a later turn does not overwrite an earlier one
    assert turn_evidence(evs, "u3") is None                  # no fusion -> no evidence handed on
