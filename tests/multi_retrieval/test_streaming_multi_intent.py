"""Integration: stream -> controller -> decomposition -> per-intent queries -> parallel retrieval -> fusion ->
unified evidence (brief §27-31, §37, TESTS 6-7). Virtual mode on the TEST FIXTURE index (hashing embedder)."""

import pytest

from streamrag.replay import ReplayEngine
from streamrag.streaming import run_realtime, run_virtual, utterance_stats
from streamrag.streaming import simulator as sim

from streaming_helpers import mi_stack, types


@pytest.fixture(scope="module")
def stack(fixture_bundle):
    cfg, b = fixture_bundle
    return mi_stack(cfg, b)


def run(stack, chunks, interval=400, **kw):
    return run_virtual(stack.cfg, stack.service, stack.policy, sim.stream(chunks, interval_ms=interval),
                       index_hash=stack.index_hash, intent_stack=stack.intent_stack, **kw)


def of(r, t):
    return [e for e in r.events if e.type.value == t]


INCREMENTAL = ["I need information about", "the fog signal", "and also", "the lens", "especially during", "a storm"]


def test_end_to_end_unified_evidence_with_lineage(stack):
    r = run(stack, INCREMENTAL)
    turn = of(r, "TURN_COMPLETED")[0].payload
    ues = turn["unified_evidence"]
    assert [i["intent_id"] for i in turn["intent_set"]["intents"]] == ["I1", "I2"]
    assert turn["sub_queries"] == ["the fog signal during a storm", "the lens during a storm"]
    assert ues and all(c["covered"] for c in ues["per_intent"])
    ledger = {q["query_id"]: q for node in turn["lineage"] for q in node["queries"]}
    for item in ues["items"]:                                       # intent -> query -> evidence is traceable
        for h in item["hits"]:
            assert item["evidence_id"] in ledger[h["query_id"]]["evidence_ids"]
            assert h["intent_id"] in {n["intent_id"] for n in turn["lineage"] if any(
                q["query_id"] == h["query_id"] for q in n["queries"])}


def test_6_only_new_or_changed_intents_are_retrieved(stack):
    r = run(stack, INCREMENTAL)
    batches = [(e.payload["intent_ids"], e.payload["reused_intents"]) for e in of(r, "MULTI_QUERY_STARTED")]
    assert batches == [(["I1"], []), (["I2"], ["I1"]), (["I1", "I2"], [])]     # V1, V2 (+I2 only), V3 (constraint)
    deltas = [e.payload["delta"] for e in of(r, "INTENTS_UPDATED")]
    assert [d["added"] for d in deltas] == [["I1"], ["I2"], []]
    assert deltas[2]["constraints_added"] == ["K1"]
    gen = [(e.intent_id, e.payload["query_text"]) for e in of(r, "QUERY_GENERATED")]
    assert len(gen) == len(set(gen)) == 4                                        # no duplicate retrieval
    st = utterance_stats(r.events)["u1"]
    assert st["intents"]["I1"]["retrieved_early"] and st["intents"]["I2"]["retrieved_early"]
    assert st["intent_set_versions"] == 3 and st["intent_coverage"] == 1.0


def test_parallel_dispatch_of_same_tick_intents(stack):
    r = run(stack, ["what are the rules for ladders in the orchard and how are the wicks trimmed"])
    started = of(r, "RETRIEVAL_STARTED")
    assert len(started) == 2 and started[0].t_session_ms == started[1].t_session_ms     # REQ-MI-006 overlap
    assert {e.intent_id for e in started} == {"I1", "I2"}


def test_7_supersession_in_stream_excludes_old_evidence(stack):
    r = run(stack, ["Tell me the requirements for ladders.", "Actually, I meant crates instead of ladders."], 600)
    sup = of(r, "INTENT_SUPERSEDED")
    assert [(e.payload["old"], e.payload["new"]) for e in sup] == [("I1", "I2")]
    turn = of(r, "TURN_COMPLETED")[0].payload
    assert turn["superseded_intents"] == ["I1"]
    assert [i["intent_id"] for i in turn["unified_evidence"]["per_intent"]] == ["I2"]
    stale = [q for n in turn["lineage"] if n["intent_id"] == "I1" for q in n["queries"]]
    assert stale and all(q["stale"] and q["stale_reason"] == "intent_superseded" for q in stale)   # provenance kept


def test_suppressed_turn_creates_no_intents(stack):
    r = run(stack, ["make that", "shorter"])
    assert of(r, "INTENTS_UPDATED") == [] and of(r, "QUERY_GENERATED") == []
    assert of(r, "TURN_COMPLETED")[0].payload["unified_evidence"] is None


def test_per_intent_budget_is_enforced_with_reason(fixture_bundle):
    cfg, b = fixture_bundle
    st = mi_stack(cfg, b, max_queries_per_intent=1)
    r = run(st, INCREMENTAL)
    skips = [e.payload["reason"] for e in of(r, "RETRIEVAL_SKIPPED") if e.component == "multi_query"]
    assert "intent_budget_exhausted" in skips
    assert {e.intent_id for e in of(r, "QUERY_GENERATED")} == {"I1", "I2"} and len(of(r, "QUERY_GENERATED")) == 2


def test_events_carry_intent_and_query_ids(stack):
    r = run(stack, INCREMENTAL)
    for e in r.events:
        if e.type.value in ("QUERY_GENERATED", "RETRIEVAL_STARTED", "RETRIEVAL_COMPLETED"):
            assert e.intent_id and e.query_id
        if e.type.value in ("INTENT_DETECTED", "INTENT_UPDATED"):
            assert e.intent_id
    order = types(r.events)
    assert order.index("INTENTS_UPDATED") < order.index("QUERY_GENERATED") < order.index("MULTI_QUERY_STARTED") \
        < order.index("RETRIEVAL_STARTED") < order.index("MULTI_QUERY_COMPLETED") < order.index("EVIDENCE_FUSED")


def test_multi_intent_virtual_replay_is_identical(stack):
    r = run(stack, INCREMENTAL)
    rep = ReplayEngine(stack.cfg, stack.service, stack.policy, stack.index_hash, stack.intent_stack).replay(r.events)
    assert rep.identical, rep.differences[:1]


def test_multi_intent_realtime_replays_with_identical_behaviour(stack):
    r = run_realtime(stack.cfg, stack.service, stack.policy, sim.stream(INCREMENTAL, interval_ms=150),
                     index_hash=stack.index_hash, intent_stack=stack.intent_stack)
    rep = ReplayEngine(stack.cfg, stack.service, stack.policy, stack.index_hash, stack.intent_stack).replay(r.events)
    assert rep.behavior_identical, rep.behavior_differences[:1]
    assert of(r, "TURN_COMPLETED")[0].payload["sub_queries"] == ["the fog signal during a storm",
                                                                 "the lens during a storm"]


def test_multi_intent_requires_an_intent_stack(stack):
    with pytest.raises(ValueError):
        run_virtual(stack.cfg, stack.service, stack.policy, sim.stream(["tell me about the lens"]))


def test_session_ledger_hit_reuses_evidence_instead_of_searching(stack):
    inputs = sim.stream(["how many crates can a picker fill"], interval_ms=300) [:-1] + \
        sim.stream(["how many crates can a picker fill"], 300, utterance_id="u2", offset_ms=3000,
                   wrap_session=False) + [sim.stream(["x"])[-1]]
    r = run_virtual(stack.cfg, stack.service, stack.policy, inputs, index_hash=stack.index_hash,
                    intent_stack=stack.intent_stack)
    hits = [e for e in of(r, "RETRIEVAL_SKIPPED") if e.payload.get("reason") == "ledger_hit"]
    assert len(hits) == 1 and hits[0].utterance_id == "u2"
    assert [e.utterance_id for e in of(r, "QUERY_GENERATED")] == ["u1"]
    turn2 = [e for e in of(r, "TURN_COMPLETED") if e.utterance_id == "u2"][0].payload
    assert turn2["reused_queries"] == {"I2": "Q1"} and turn2["unified_evidence"]["per_intent"][0]["covered"]


def test_correction_utterance_opens_the_gate_and_supersedes(stack):
    """Streaming dev run: 'actually I meant the lamp not the lens' was judged not retrieval-worthy by the Phase 4
    act classifier, so the correction never reached the decomposer."""
    inputs = sim.stream(["tell me about", "the lens"], interval_ms=400)[:-1] + \
        sim.stream(["actually I meant", "the lamp", "not the lens"], 400, utterance_id="u2", offset_ms=3000,
                   wrap_session=False) + [sim.stream(["x"])[-1]]
    r = run_virtual(stack.cfg, stack.service, stack.policy, inputs, index_hash=stack.index_hash,
                    intent_stack=stack.intent_stack)
    assert [(e.payload["old"], e.payload["new"]) for e in of(r, "INTENT_SUPERSEDED")] == [("I1", "I2")]
    assert [e.utterance_id for e in of(r, "QUERY_GENERATED")] == ["u1", "u2"]


def test_correction_gate_does_not_open_without_a_need_to_correct(stack):
    r = run(stack, ["actually I meant", "nothing"])
    assert of(r, "INTENTS_UPDATED") == []


def test_cross_utterance_correction_survives_quiet_ticks(stack):
    """Streaming dev run (L2): after the correction superseded I1, the next quiet tick re-decomposed the utterance,
    found no correction target any more and dropped the corrected need."""
    inputs = sim.stream(["tell me about", "the lens"], interval_ms=400)[:-1] + \
        sim.stream(["actually I meant", "the lamp", "not the lens"], 900, utterance_id="u2", offset_ms=3000,
                   wrap_session=False) + [sim.stream(["x"])[-1]]
    r = run_virtual(stack.cfg, stack.service, stack.policy, inputs, index_hash=stack.index_hash,
                    intent_stack=stack.intent_stack)
    turn2 = [e for e in of(r, "TURN_COMPLETED") if e.utterance_id == "u2"][0].payload
    assert [i["intent_id"] for i in turn2["intent_set"]["intents"]] == ["I2"]
    assert all(not e.payload["delta"]["removed"] for e in of(r, "INTENTS_UPDATED"))
    assert [e.payload["version"] for e in of(r, "INTENTS_UPDATED") if e.utterance_id == "u2"][0] == 1
