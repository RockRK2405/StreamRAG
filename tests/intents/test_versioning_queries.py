"""Intent versioning, deltas, supersession (TESTS 6, 7), query generation and validation (brief §13, §15, §28-30)."""

from intent_helpers import dec, qb, tracker  # noqa: F401  (pytest fixtures)

import json
import re

import pytest

from streamrag.intents.llm_check import LLMDecompositionCheck
from streamrag.intents.validation import validate_intent_set, validate_queries
from streamrag.models.intents import IntentSet, IntentSetDelta, SourceSpan

STEPS = ["I need information about the fog signal",
         "I need information about the fog signal and also the lens",
         "I need information about the fog signal and also the lens especially during a storm",
         "I need information about the fog signal and also the lens especially during a storm."]


def test_6_incremental_intents_and_delta(tracker):
    deltas = [tracker.update("u1", t, 300 * k)[1] for k, t in enumerate(STEPS)]
    assert deltas[0].added == ["I1"] and deltas[0].affected_intents == ["I1"]
    assert deltas[1].added == ["I2"] and deltas[1].affected_intents == ["I2"]         # only the new intent
    assert deltas[2].constraints_added == ["K1"] and deltas[2].affected_intents == ["I1", "I2"]
    assert [m.changed for m in deltas[2].modified] == [["constraints"], ["constraints"]]
    assert deltas[3] is None                                                         # nothing changed
    vs = tracker.versions("u1")
    assert [v.version for v in vs] == [1, 2, 3]
    assert [[i.intent_id for i in v.intents] for v in vs] == [["I1"], ["I1", "I2"], ["I1", "I2"]]
    assert [i.version for i in vs[-1].intents] == [2, 2]


def test_intent_ids_are_stable_while_text_grows(tracker):
    tracker.update("u1", "tell me how high the wicks", 0)
    _, delta, _ = tracker.update("u1", "tell me how high the wicks should be trimmed", 300)
    assert delta.added == [] and [m.intent_id for m in delta.modified] == ["I1"]
    assert delta.modified[0].changed == ["text"]


def test_7_correction_supersedes_and_keeps_provenance(tracker):
    tracker.update("u1", "Tell me the requirements for ladders.", 0)
    iset, delta, _ = tracker.update("u1", "Tell me the requirements for ladders. Actually, I meant crates instead of "
                                          "ladders.", 900)
    assert [(s.old, s.new) for s in delta.superseded] == [("I1", "I2")]
    assert [i.resolved_text for i in iset.intents] == ["the requirements for crates"]
    old, new = tracker.intents["I1"], tracker.intents["I2"]
    assert old.status == "SUPERSEDED" and old.superseded_by == "I2" and new.supersedes == "I1"
    assert new.lineage_root == "I1" and iset.superseded == ["I1"]


def test_correction_of_an_earlier_utterance(tracker):
    tracker.update("u1", "Tell me about the lens.", 0)
    tracker.update("u2", "And how often is it cleaned?", 1000)
    iset, delta, _ = tracker.update("u3", "Actually, I meant the lamp instead of the lens.", 2000)
    assert [(s.old, s.new) for s in delta.superseded] == [("I2", "I3")]
    assert iset.intents[0].resolved_text == "how often is the lamp cleaned"
    assert {c.span.utterance_id for c in iset.intents[0].components} == {"u2", "u3"}


def test_removed_intent_is_recorded(tracker):
    tracker.update("u1", "tell me about the lens and the", 0)
    tracker.update("u1", "tell me about the lens and the wicks", 300)
    iset, delta, _ = tracker.update("u1", "tell me about the lens", 600)       # ASR revision drops a need
    assert delta.removed == ["I2"] and tracker.intents["I2"].status == "DROPPED"
    assert [d.reason for d in iset.dropped] == ["no_longer_in_decomposition"]


# ---------------------------------------------------------------------------------------- queries
def test_query_per_intent_with_inherited_context_and_constraints(tracker, qb):
    iset, _, _ = tracker.update("u1", "For visitors, what are the rules and the schedule for the telescope?", 0)
    ks = iset.global_constraints + iset.local_constraints
    qs = [qb.build(i, ks) for i in iset.intents]
    assert [q.text for q in qs] == ["what are the rules for the telescope For visitors",
                                    "the schedule for the telescope For visitors"]
    assert [c.source for c in qs[0].components][-2:] == ["inherited", "constraint"]
    assert validate_queries(qs) == []


def test_queries_contain_only_spoken_words(tracker, qb):
    """No injected facts: every query word occurs in the session's transcripts."""
    tracker.update("u1", "What are the eligibility requirements for the observatory visits?", 0)
    tracker.update("u2", "And what about the application process, especially if it rains?", 900)
    spoken = set(re.findall(r"\w+", " ".join(tracker.transcripts().values()).lower()))
    for u in ("u1", "u2"):
        cur = tracker.current(u)
        for i in cur.intents:
            q = qb.build(i, cur.global_constraints + cur.local_constraints)
            assert set(re.findall(r"\w+", q.text.lower())) <= spoken
            for c in q.components:
                assert tracker.transcripts()[c.span.utterance_id][c.span.start:c.span.end] == c.span.text


def test_negation_and_numbers_survive_into_the_query(tracker, qb):
    iset, _, _ = tracker.update("u1", "are ladders not allowed after 5 pm and what is the crate limit", 0)
    qs = [qb.build(i, []) for i in iset.intents]
    assert "not" in qs[0].text.split() and "5" in qs[0].text.split()


def test_identical_queries_are_flagged(qb, tracker):
    iset, _, _ = tracker.update("u1", "tell me about the lens", 0)
    q = qb.build(iset.intents[0], [])
    assert [i.code for i in validate_queries([q, q])] == ["identical_query"]


# ---------------------------------------------------------------------------------------- validation
def test_rule_output_validates(tracker, qb):
    for k, t in enumerate(STEPS + ["Tell me about the lens. And how does it apply there?"]):
        iset, _, _ = tracker.update("u1" if k < 4 else "u2", t, 300 * k)
        assert validate_intent_set(iset, tracker.transcripts(), set(tracker.intents), qb.terms_fn) == []


def test_validation_catches_span_mismatch_and_duplicates(tracker, qb):
    iset, _, _ = tracker.update("u1", "tell me about the lens and the wicks", 0)
    bad = iset.intents[0].model_copy(update={"provenance": iset.intents[0].provenance.model_copy(update={
        "span": SourceSpan(utterance_id="u1", start=0, end=4, text="nope")})})
    broken = iset.model_copy(update={"intents": [bad, iset.intents[0]]})
    codes = {i.code for i in validate_intent_set(broken, tracker.transcripts(), set(tracker.intents), qb.terms_fn)}
    assert {"span_mismatch", "duplicate_intent_id", "duplicate_intent"} <= codes


def test_empty_output_is_an_issue():
    iset = IntentSet(session_id="s", utterance_id="u1", version=1, source="rule", original_text="tell me")
    assert [i.code for i in validate_intent_set(iset, {"u1": "tell me"})] == ["empty_output"]


def test_delta_empty_property():
    assert IntentSetDelta(utterance_id="u1", version=2, previous_version=1).empty


# ---------------------------------------------------------------------------------------- optional LLM check
class FakeBackend:
    name = "fake"

    def __init__(self, outputs):
        self.outputs, self.calls = list(outputs), 0

    def complete_json(self, prompt, schema, timeout_s):
        self.calls += 1
        out = self.outputs.pop(0)
        if isinstance(out, Exception):
            raise out
        return out


def test_llm_check_off_and_no_backend_keep_rules(dec):
    d = dec.decompose("I need information about the fog signal and also the lens especially during a storm", "u1")
    for chk, status in ((LLMDecompositionCheck(dec, None, "off"), "off"),
                        (LLMDecompositionCheck(dec, None, "gated"), "no_backend")):
        out, rep = chk.run(d)
        # G-c: ambiguous constraint scope; G-d: that ambiguity also lowers decomposition confidence below 0.6
        assert rep.gated_by == ["G-c", "G-d"] and rep.status == status and not rep.called and out is d


def test_llm_malformed_output_retries_once_then_falls_back(dec):
    d = dec.decompose("I need information about the fog signal and also the lens especially during a storm", "u1")
    fb = FakeBackend(["{not json", '{"intents": [{"text": 3}]}'])
    out, rep = LLMDecompositionCheck(dec, fb, "gated").run(d)
    assert fb.calls == 2 and rep.status == "fallback_rules" and out is d
    assert all(i.startswith("malformed_output") for i in rep.issues)


def test_llm_ungrounded_and_unsupported_intents_are_rejected(dec):
    d = dec.decompose("I need information about the fog signal and also the lens especially during a storm", "u1")
    raw = json.dumps({"intents": [
        {"text": "visiting hours", "source_text": "when can I visit", "type": "TIMELINE"},       # not said
        {"text": "lens", "source_text": "the lens", "type": "MAGIC"},                           # bad type
        {"text": "fog signal", "source_text": "the fog signal", "type": "OTHER"}]})              # matches rule
    out, rep = LLMDecompositionCheck(dec, FakeBackend([raw]), "gated").run(d)
    assert any(i.startswith("missing_source_span") for i in rep.issues)
    assert any(i.startswith("unsupported_type") for i in rep.issues)
    assert rep.added == [] and rep.reconciled == ["d1"] and len(out.active) == 2


def test_llm_grounded_missing_need_is_added_with_llm_provenance(dec):
    text = "tell me how high the wicks should be trimmed and then the lamp register"
    d = dec.decompose(text, "u1")
    n = len(d.active)
    raw = json.dumps({"intents": [{"text": "lens polishing", "source_text": "the lamp register", "type": "OTHER"}]})
    chk = LLMDecompositionCheck(dec, FakeBackend([raw]), "gated", low_confidence=1.01)   # force gating (G-d)
    out, rep = chk.run(d)
    assert rep.status == "ok"
    assert len(out.active) >= n
    for i in out.active:
        assert text[i.start:i.end] == i.text


@pytest.mark.parametrize("bad", ["", "[]", '{"intents": "x"}'])
def test_llm_schema_violations_fall_back(dec, bad):
    d = dec.decompose("I need information about the fog signal and also the lens especially during a storm", "u1")
    out, rep = LLMDecompositionCheck(dec, FakeBackend([bad, bad]), "gated").run(d)
    assert rep.status == "fallback_rules" and out is d


# ---------------------------------------------------------------------------------------- regressions (Phase 5 dev run)
@pytest.mark.parametrize("text,new", [
    ("how often is the lamp cleaned no wait how often is the lens polished", "how often is the lens polished"),
    ("what do workers do before leaving the tool shed scratch that where are ladders stored",
     "where are ladders stored"),
])
def test_correction_followed_by_a_question_supersedes(dec, text, new):
    """Blind dev run: the question after 'no wait' / 'scratch that' opened its own clause and the correction was lost."""
    d = dec.decompose(text, "u1")
    assert [i.resolved_text for i in d.active] == [new]
    assert [i.status for i in d.intents] == ["SUPERSEDED", "ACTIVE"]


def test_follow_up_inherits_topic_whose_span_contains_stopwords(tracker, qb):
    """Blind dev run: topic 'using a ladder' was re-searched as 'using ladder' and the follow-up lost it."""
    tracker.update("u1", "what are the requirements for using a ladder", 0)
    iset, _, _ = tracker.update("u2", "what about the time limits", 1000)
    i = iset.intents[0]
    assert i.inherited_context[0].source_span.text == "using a ladder"
    assert "ladder" in qb.build(i, []).terms


def test_correction_x_not_y_replaces_y(tracker):
    tracker.update("u1", "tell me about the lens", 0)
    iset, delta, _ = tracker.update("u2", "actually I meant the lamp not the lens", 2000)
    assert [i.resolved_text for i in iset.intents] == ["the lamp"]
    assert [(x.old, x.new) for x in delta.superseded] == [("I1", "I2")]
