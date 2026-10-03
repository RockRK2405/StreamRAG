"""SessionState / SessionMemory / versioning / snapshot-restore / reset / archive / memory safety (Phase 6, brief §2-5,
§34-37). TEST FIXTURE corpus (fictional)."""

import json

from streamrag.session.models import SessionState, SessionStateVersion
from streamrag.session.safety import redact

from session_helpers import engine_of, p6_stack, run_turns, session_cfg  # noqa: F401  (fixture)
from streaming_helpers import hashing_stack

LADDERS = "What are the rules for ladders in the orchard?"


def test_session_state_has_all_layers(p6_stack):
    p, _ = run_turns(p6_stack, [LADDERS, "Specifically overnight."])
    st = engine_of(p).memory.get_current_state()
    assert isinstance(st, SessionState)
    assert st.current_utterance_id == "u2" and st.session_version >= 1 and st.active_frame_id == "T1"
    assert [i["intent_id"] for i in st.intent_set] == ["I1"] and st.intent_set[0]["constraints"] == ["overnight"]
    assert [c.text for c in st.constraints] == ["overnight"]
    assert st.active_queries and st.query_ledger
    assert st.evidence_store and st.claims and st.answer_state is not None and st.answer_state.answer_id == "A2"
    assert {e.text for e in st.entities} >= {"ladders"}
    assert [t.utterance_id for t in st.transcript_state] == ["u1", "u2"]
    SessionState.model_validate_json(st.model_dump_json())          # serializable contract


def test_versions_form_a_parent_chain_and_skip_no_ops(p6_stack):
    p, rs = run_turns(p6_stack, [LADDERS, "Okay.", "Specifically overnight."])
    mem = engine_of(p).memory
    vs = mem.versions
    assert all(isinstance(v, SessionStateVersion) for v in vs)
    assert [v.version_id for v in vs] == list(range(1, len(vs) + 1))
    assert all(v.parent_version == (v.version_id - 1 or None) for v in vs)
    assert {v.trigger for v in vs} <= {"interpretation", "evidence", "answer"}
    assert not [v for v in vs if v.utterance_id == "u2"]                         # "Okay." created no version
    ch = [v for v in vs if v.trigger == "interpretation" and v.utterance_id == "u3"][0]
    assert ch.changes == [c.change_id for c in rs[2].changes]
    assert mem.update_session("interpretation", 99.0) is None                    # unchanged snapshot -> no version


def test_window_compression_and_redaction(fixture_bundle):
    cfg, b = fixture_bundle
    st = hashing_stack(session_cfg(cfg, transcript_window=2), b)
    turns = [LADDERS, "My email is pat@example.org, call +1 555 010 9999.", "Okay.", "Specifically overnight."]
    p, _ = run_turns(st, turns)
    mem = engine_of(p).memory
    tr = {t.utterance_id: t for t in mem.transcript}
    assert [u for u, t in tr.items() if t.compressed] == ["u1", "u2"]
    assert tr["u1"].text == "" and len(tr["u1"].text_sha1) == 40 and tr["u1"].chars == len(LADDERS)
    assert tr["u3"].text == "Okay." and not tr["u3"].compressed
    assert tr["u2"].redactions == 2
    blob = mem.create_snapshot()
    assert "pat@example.org" not in blob and "555 010 9999" not in blob
    assert "pat@example.org" not in json.dumps(mem.archive_session().model_dump(mode="json"))


def test_redaction_patterns_keep_task_numbers():
    text, n = redact("Email a.b@x.co, token=abc123, key: sk_live_1234567890abcdefghijkl and card 4111 1111 1111 1111")
    assert n == 4 and "a.b@x.co" not in text and "abc123" not in text and "4111" not in text
    text, n = redact("At most 40 crates, stacks of 5, after 9 pm, 4 millimetres")
    assert n == 0 and text == "At most 40 crates, stacks of 5, after 9 pm, 4 millimetres"


def test_snapshot_restore_round_trip(p6_stack):
    p, _ = run_turns(p6_stack, [LADDERS, "Specifically overnight."])
    eng = engine_of(p)
    blob = eng.memory.create_snapshot()
    before = eng.memory.get_current_state().model_dump(mode="json")
    p.process("u3", "Actually, ignore the overnight restriction.", 9000.0)
    assert eng.memory.get_current_state().model_dump(mode="json") != before
    eng.memory.restore_snapshot(blob)
    assert eng.memory.get_current_state().model_dump(mode="json") == before
    # the restored session keeps working: the same late detail produces the same change again
    r = p.process("u4", "Actually, ignore the overnight restriction.", 12000.0)
    assert [c.change_type for c in r.changes] == ["CONSTRAINT_REMOVAL"]


def test_reset_session_clears_every_layer(p6_stack):
    p, _ = run_turns(p6_stack, [LADDERS, "Specifically overnight."])
    eng = engine_of(p)
    eng.memory.reset_session()
    st = eng.memory.get_current_state()
    assert st.transcript_state == [] and st.intent_set == [] and st.constraints == [] and st.frames == []
    assert st.query_ledger == [] and st.evidence_store == {} and st.claims == {} and st.answer_state is None
    assert eng.cache.entries == {} and eng.memory.versions == []
    r = p.process("u9", LADDERS, 30000.0)                                         # fresh conversation works
    assert [c.change_type for c in r.changes] == ["NEW_INTENT"] and r.retrievals == 1


def test_archive_keeps_metadata_only(p6_stack):
    p, _ = run_turns(p6_stack, [LADDERS, "Specifically overnight.", "Okay."])
    arc = engine_of(p).memory.archive_session(1.0)
    dump = json.dumps(arc.model_dump(mode="json"))
    assert arc.utterances == 3 and len(arc.transcript_sha1) == 3
    assert arc.change_types.get("CONSTRAINT_ADDITION") == 1 and arc.answer_versions == 2
    assert "ladders" not in dump.lower() and "overnight" not in dump.lower()          # no raw text survives


def test_intent_entity_and_evidence_context(p6_stack):
    p, _ = run_turns(p6_stack, [LADDERS, "Now explain how the telescope is recalibrated."])
    mem = engine_of(p).memory
    assert [i["intent_id"] for i in mem.get_intent_context()] == ["I2"]              # active frame only
    assert [i["intent_id"] for i in mem.get_intent_context("I1")] == ["I1"]
    assert all("I2" in e.intent_ids for e in mem.get_entity_context())
    ev = mem.get_relevant_evidence("I2")
    assert ev and all(e["status"] in ("ACTIVE", "RETAINED") and e["citation"] for e in ev)


def test_counters_stay_shared_after_reset_and_restore(p6_stack):
    p, _ = run_turns(p6_stack, [LADDERS])
    eng = engine_of(p)
    blob = eng.memory.create_snapshot()
    eng.memory.reset_session()
    assert eng.counters is eng.memory.counters and eng.counters == {}
    eng.memory.restore_snapshot(blob)
    assert eng.counters is eng.memory.counters and eng.counters["retrievals_planned"] == 1
    p.process("u2", "Specifically overnight.", 6000.0)
    assert eng.memory.counters["retrievals_planned"] == 2


def test_pii_never_enters_interpretation(p6_stack):
    p, rs = run_turns(p6_stack, ["What are the rules for ladders for pat@example.org in the orchard?"])
    eng = engine_of(p)
    assert "pat@example.org" not in eng.memory.create_snapshot()
    assert all("pat@example.org" not in r.query_text for r in eng.ledger.all())
