"""Intent decomposition: definition, over/under-decomposition guards, types, confidence (brief TESTS 1-5, 8, 9).

Utterances are about the TEST FIXTURE corpus (fictional) or use neutral placeholders; nothing here is a benchmark.
"""

from intent_helpers import dec, qb, tracker  # noqa: F401  (pytest fixtures)

import pytest


def active(d):
    return [i.resolved_text for i in d.active]


# ---------------------------------------------------------------------------------------- critical tests
def test_1_single_need_is_one_intent(dec):
    d = dec.decompose("Tell me about the fog signal.", "u1")
    assert active(d) == ["the fog signal"]


def test_2_two_needs(dec):
    d = dec.decompose("Tell me about the fog signal and the lens.", "u1")
    assert active(d) == ["the fog signal", "the lens"]


def test_3_three_needs_with_list_commas(dec):
    d = dec.decompose("Tell me about the wicks, the lens, and the logbook.", "u1")
    assert active(d) == ["the wicks", "the lens", "the logbook"]


def test_4_general_and_specific_need_on_same_topic(dec):
    d = dec.decompose("Tell me about the tool shed and also explain the tool shed's register process.", "u1")
    assert len(d.active) == 2
    rel = [(r.type, r.source, r.target) for r in d.relations if r.type == "REFINEMENT"]
    assert rel == [("REFINEMENT", d.active[1].key, d.active[0].key)]


def test_5_constraint_is_not_an_intent(dec):
    d = dec.decompose("Tell me about the harvest limits especially for the night shift.", "u1")
    assert active(d) == ["the harvest limits"]
    assert [(k.text, k.kind, k.applies_to) for k in d.constraints] == [("for the night shift", "focus", ["d1"])]


def test_8_duplicate_need_is_merged(dec):
    d = dec.decompose("Tell me about the wicks and the wicks.", "u1")
    assert active(d) == ["the wicks"]
    assert [m[2] for m in d.merged] == ["duplicate"]


def test_9_narrative_nouns_do_not_become_intents(dec):
    d = dec.decompose("So yesterday I walked past the orchard and the tool shed and the cool room with my sister "
                      "and her dog, anyway, how many crates can a picker fill per shift?", "u1")
    assert active(d) == ["how many crates can a picker fill per shift"]
    assert [c.role for c in d.clauses] == ["CONTEXT", "REQUEST"]


# ---------------------------------------------------------------------------------------- over-decomposition
@pytest.mark.parametrize("text", [
    "What is the difference between the wick and the lens?",          # comparison: one need
    "Can ladders or crates stay in the orchard overnight?",           # alternative: one need
    "Tell me about the third and fifth week of planting.",            # 'third and fifth' is a corpus phrase
    "I need information and details about the lens.",                 # generic nouns are not needs
    "Tell me about it.",
])
def test_no_over_decomposition(dec, text):
    d = dec.decompose(text, "u1")
    assert len(d.active) <= 1


def test_tell_me_about_x_is_not_split_into_words(dec):
    d = dec.decompose("Tell me about the storm procedure.", "u1")
    assert [i.text for i in d.active] == ["the storm procedure"]
    assert d.active[0].split == "single"


# ---------------------------------------------------------------------------------------- under-decomposition
@pytest.mark.parametrize("text,expected", [
    ("Tell me eligibility requirements and application steps.", 2),
    ("What are the requirements for ladders, how long does pruning take, and are there exceptions for visitors?", 3),
    ("i want to know how high the wicks should be trimmed and how often the lamp is cleaned", 2),
    ("what do workers do before they leave the tool shed and where are ladders stored overnight", 2),
    ("the requirements for ladders how long does pruning take", 2),                 # question restart, no punct
    ("tell me about the planting schedule the water each sapling gets and the approved varieties", 3),   # ASR list
])
def test_no_under_decomposition(dec, text, expected):
    assert len(dec.decompose(text, "u1").active) == expected


def test_relative_clause_is_not_a_list_item(dec):
    d = dec.decompose("tell me the time the lamp is cleaned and the lens polish", "u1")
    assert active(d) == ["the time the lamp is cleaned", "the lens polish"]


# ---------------------------------------------------------------------------------------- other behaviour
def test_social_turn_yields_no_intent(dec):
    d = dec.decompose("hmm okay thanks", "u1")
    assert d.active == []


def test_statement_without_request_cue_falls_back_to_one_intent(dec):
    d = dec.decompose("ladder storage at night", "u1")
    assert d.source == "rule_fallback" and active(d) == ["ladder storage at night"]
    assert d.active[0].signals["explicit"] == 0.5


@pytest.mark.parametrize("text,itype", [
    ("what are the requirements for ladders", "REQUIREMENT"),
    ("how long does pruning take", "TIMELINE"),
    ("are there exceptions for visitors", "EXCEPTION"),
    ("how do keepers polish the lens", "PROCEDURAL"),
    ("how many crates can a picker fill", "FACTUAL"),
    ("what is the difference between the wick and the lens", "COMPARISON"),
    ("tell me about the fog signal", "OTHER"),
])
def test_intent_types(dec, text, itype):
    assert dec.decompose(text, "u1").active[0].intent_type == itype


def test_confidence_is_the_documented_formula(dec):
    for text in ["Tell me about the fog signal and the lens.", "ladder storage at night",
                 "Tell me about the lens. And how does it apply there?"]:
        for i in dec.decompose(text, "u1").active:
            s = i.signals
            expected = 0.35 * s["explicit"] + 0.30 * s["specificity"] + 0.20 * s["separation"] + 0.15 * s["resolution"]
            assert i.confidence == round(expected, 3)


def test_budget_drops_lowest_priority_with_reason(dec):
    d = dec.decompose("tell me about the planting schedule the water each sapling gets the approved varieties "
                      "the pruning rules and the ladder safety", "u1")
    assert len(d.active) == dec.max_intents == 4
    assert len(d.dropped) == 1 and d.dropped[0][2].startswith("max_intents_budget")
    assert sorted(i.priority for i in d.active) == [1, 2, 3, 4]


def test_order_of_mention_is_kept_separately_from_priority(dec):
    d = dec.decompose("What are the requirements for ladders, how long does pruning take, "
                      "and are there exceptions for visitors?", "u1")
    assert [i.order for i in d.active] == [0, 1, 2]
    assert sorted(i.priority for i in d.active) == [1, 2, 3]


def test_entities_and_spans_are_verbatim(dec):
    text = "Tell me about the fog signal and the tool shed."
    d = dec.decompose(text, "u1")
    for i in d.active:
        assert text[i.start:i.end] == i.text
        for s in i.segments:
            assert text[s.start:s.end] == s.text
    assert d.active[1].entities == ["tool shed"]
