"""EvidenceFusionEngine: provenance, cross-intent dedup, strategies, coverage, conflicts, rerank (brief §20-26).

Uses synthetic EvidenceSets (no index) so expectations are exact; values are illustrative, not measurements.
"""

import numpy as np
import pytest

from streamrag.config.settings import FusionConfig
from streamrag.fusion import EvidenceFusionEngine, IntentEvidence
from streamrag.models.evidence import Evidence, EvidenceSet, RetrievalTrace
from streamrag.models.intents import Intent

TRACE = RetrievalTrace(request_id="r", mode="hybrid", status="ok", index_version="1", corpus_hash="c",
                       index_config_hash="i")
SPAN = {"utterance_id": "u1", "start": 0, "end": 1, "text": "x"}


def intent(iid, order, priority=None):
    return Intent(intent_id=iid, session_id="s", utterance_id="u1", text=iid, resolved_text=iid, confidence=0.9,
                  order=order, priority=priority or order + 1,
                  provenance={"source": "rule", "split": "coordination", "span": SPAN})


def ev(cid, rank, doc="D", sec=None, text=None, alternates=(), score=None):
    sec = sec or cid
    return Evidence(evidence_id=cid, document_id=doc, section_id=sec, chunk_id=cid, citation=f"{doc} §{sec}",
                    text=text or f"text of {cid}", source_path=f"{doc}.txt", char_start=0, char_end=1, rank=rank,
                    score=score if score is not None else 1.0 / (60 + rank), retrieval_method="hybrid_rrf",
                    alternates=list(alternates), metadata={"token_count": 10})


def es(*items):
    return EvidenceSet(evidence_set_id="es", query="q", items=list(items), trace=TRACE)


def inp(iid, order, items, qid=None):
    return IntentEvidence(intent(iid, order), f"query {iid}", qid or f"Q{order + 1}", es(*items))


@pytest.fixture()
def engine():
    return EvidenceFusionEngine(FusionConfig(top_k=4, min_per_intent=1, section_cap_per_intent=2))


def test_provenance_on_every_item(engine):
    u = engine.fuse("s", "u1", 1, [inp("I1", 0, [ev("a", 1), ev("b", 2)]), inp("I2", 1, [ev("c", 1)])])
    for it in u.items:
        assert it.hits and all(h.intent_id and h.query_id and h.retrieval_method == "hybrid_rrf" for h in it.hits)
    assert {it.evidence_id: it.supporting_queries for it in u.items}["c"] == ["Q2"]


def test_cross_intent_duplicate_is_one_item_with_both_intents(engine):
    u = engine.fuse("s", "u1", 1, [inp("I1", 0, [ev("shared", 1), ev("a", 2)]),
                                   inp("I2", 1, [ev("shared", 2), ev("b", 1)])])
    shared = [it for it in u.items if it.evidence_id == "shared"]
    assert len(shared) == 1 and shared[0].supporting_intents == ["I1", "I2"]
    assert shared[0].supporting_queries == ["Q1", "Q2"] and shared[0].best_rank_by_intent == {"I1": 1, "I2": 2}
    assert u.dedup["cross_intent_duplicates"] == 1 and u.dedup["duplicate_items"] == 0


def test_near_duplicate_alternate_collapses_across_intents(engine):
    u = engine.fuse("s", "u1", 1, [inp("I1", 0, [ev("a", 1, alternates=["a2"])]), inp("I2", 1, [ev("a2", 1)])])
    assert [it.evidence_id for it in u.items] == ["a"]
    assert u.items[0].supporting_intents == ["I1", "I2"] and u.dedup["near_duplicates_merged"] == 1
    assert any(r.type == "DUPLICATES" and r.source == "a2" for r in u.relations)


def test_intent_aware_keeps_every_intent_covered(engine):
    """A strong intent with many hits must not take the whole budget (brief §24)."""
    strong = [ev(f"s{k}", k) for k in range(1, 9)]
    u = engine.fuse("s", "u1", 1, [inp("I1", 0, strong), inp("I2", 1, [ev("w1", 1)]), inp("I3", 2, [ev("z1", 1)])])
    assert {c.intent_id: c.covered for c in u.per_intent} == {"I1": True, "I2": True, "I3": True}
    assert len(u.items) == 4
    concat = engine.fuse("s", "u1", 1, [inp("I1", 0, strong), inp("I2", 1, [ev("w1", 1)]),
                                        inp("I3", 2, [ev("z1", 1)])], strategy="concat")
    assert [it.evidence_id for it in concat.items] == ["s1", "s2", "s3", "s4"]      # starves I2 and I3
    assert {c.intent_id: c.covered for c in concat.per_intent} == {"I1": True, "I2": False, "I3": False}


def test_concat_shows_duplicates_others_do_not(engine):
    a = [ev("x", 1), ev("y", 2)]
    b = [ev("x", 1), ev("z", 2)]
    for strat in ("concat", "global_score", "rrf", "intent_aware"):
        u = engine.fuse("s", "u1", 1, [inp("I1", 0, a), inp("I2", 1, b)], strategy=strat)
        assert (u.dedup["duplicate_items"] > 0) == (strat == "concat"), strat


def test_rrf_rewards_shared_chunks(engine):
    u = engine.fuse("s", "u1", 1, [inp("I1", 0, [ev("a", 1), ev("shared", 2)]),
                                   inp("I2", 1, [ev("b", 1), ev("shared", 2)])], strategy="rrf", top_k=1)
    assert [it.evidence_id for it in u.items] == ["shared"]


def test_section_cap_per_intent():
    eng = EvidenceFusionEngine(FusionConfig(top_k=5, min_per_intent=3, section_cap_per_intent=1))
    u = eng.fuse("s", "u1", 1, [inp("I1", 0, [ev("a", 1, sec="1"), ev("b", 2, sec="1"), ev("c", 3, sec="2")])])
    assert [it.evidence_id for it in u.items] == ["a", "c"]


def test_labels_follow_order_of_mention_then_rank(engine):
    u = engine.fuse("s", "u1", 1, [inp("I2", 1, [ev("b", 1)]), inp("I1", 0, [ev("a", 1)])])
    assert [(it.label, it.evidence_id) for it in u.items] == [("E1", "a"), ("E2", "b")]


def test_failed_and_empty_intents_are_reported_not_hidden(engine):
    failed = IntentEvidence(intent("I2", 1), "q", "Q2", None, status="retrieval_failed")
    empty = inp("I3", 2, [])
    u = engine.fuse("s", "u1", 1, [inp("I1", 0, [ev("a", 1)]), failed, empty])
    assert {c.intent_id: c.status for c in u.per_intent} == {"I1": "ok", "I2": "retrieval_failed",
                                                             "I3": "no_candidates"}


def test_numeric_conflict_is_flagged_as_potential(engine):
    a = ev("a", 1, doc="D1", text="The lamp wick height must be 4 millimetres for every keeper.")
    b = ev("b", 2, doc="D2", text="Keepers trim the lamp wick height to 6 millimetres each evening.")
    c = ev("c", 3, doc="D2", text="The beam is visible for 6 millimetres of glass.")   # same doc as b: ignored pair
    u = engine.fuse("s", "u1", 1, [inp("I1", 0, [a, b, c])])
    assert len(u.conflicts) == 1
    x = u.conflicts[0]
    assert x.status == "potential" and x.evidence_ids == ["a", "b"] and x.detail["values"] == ["4", "6"]
    assert any(r.type == "CONTRADICTS" for r in u.relations)


def test_no_conflict_for_different_units_or_same_value(engine):
    a = ev("a", 1, doc="D1", text="The signal sounds every 30 seconds in fog.")
    b = ev("b", 2, doc="D2", text="The signal sounds every 30 seconds during storms.")
    c = ev("c", 3, doc="D3", text="The signal is checked every 2 hours.")
    assert engine.fuse("s", "u1", 1, [inp("I1", 0, [a, b, c])]).conflicts == []


def test_cross_intent_dense_rerank_moves_relevant_chunk_to_other_intent():
    vec = {"a": [1, 0], "b": [0, 1]}
    eng = EvidenceFusionEngine(FusionConfig(top_k=2, min_per_intent=1, rerank="cross_intent_dense"),
                               chunk_rows={"a": 0, "b": 1}, vectors=np.array([vec["a"], vec["b"]], dtype=np.float32),
                               embed_fn=lambda texts: np.array([[0, 1] if "I2" in t else [1, 0] for t in texts],
                                                               dtype=np.float32))
    # I2 retrieved only 'a' (irrelevant to it); I1 retrieved 'b' at rank 2, which is what I2 actually asks about
    u = eng.fuse("s", "u1", 1, [inp("I1", 0, [ev("a", 1), ev("b", 2)]), inp("I2", 1, [ev("a", 1)])])
    b = next(it for it in u.items if it.evidence_id == "b")
    assert "I2" in b.selected_for and b.intent_relevance["I2"] == 1.0
    assert u.rerank == "cross_intent_dense" and "rerank" in u.timings_ms


def test_rerank_failure_falls_back_to_retrieval_order():
    eng = EvidenceFusionEngine(FusionConfig(rerank="intent_ce"))        # no cross-encoder loaded
    u = eng.fuse("s", "u1", 1, [inp("I1", 0, [ev("a", 1)])])
    assert u.rerank == "none" and u.warnings and u.warnings[0].startswith("rerank_failed")
