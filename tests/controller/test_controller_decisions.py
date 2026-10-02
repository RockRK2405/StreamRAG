"""Controller decisions: CASE 1-3 of the Phase 4 brief, suppression, storm guards, transparency."""

import pytest

from streamrag.controller import ControllerInput, make_policy
from streamrag.controller.acts import PrototypeActClassifier, RuleActClassifier
from streamrag.ledger import QueryLedger
from streamrag.retrieval.embedders import HashingEmbedder

from streaming_helpers import with_controller


@pytest.fixture(scope="module")
def policy(fixture_bundle):
    cfg, b = fixture_bundle
    return make_policy(cfg, b)


def decide(policy, text, tick="chunk", ledger=None, now=1000.0, in_flight=0, prev_terms=None, quiet=False):
    d, q, terms = policy.decide(ControllerInput("s", "u1", text, tick, now, 0, prev_terms or [], quiet),
                                ledger or QueryLedger("s"), in_flight)
    return d


@pytest.mark.parametrize("text", ["I need", "I need information", "so I wanted to ask", "how high the wicks should be"])
def test_case1_incomplete_transcript_waits(policy, text):
    d = decide(policy, text)
    assert d.decision == "WAIT" and d.reason in {"no_content", "low_specificity", "trailing_function_word"}


@pytest.mark.parametrize("text", ["how high the wicks should be trimmed", "I need information about the fog signal",
                                  "are ladders allowed to stay in the orchard overnight"])
def test_case2_stable_retrieval_worthy_question_retrieves(policy, text):
    d = decide(policy, text)
    assert d.decision == "RETRIEVE" and d.trigger == "provisional" and d.reason == "stable_retrieval_worthy_request"
    s = d.signals
    expected_conf = round(0.4 * s["semantic_stability"] + 0.4 * s["retrieval_worthiness"] + 0.2 * s["novelty"], 3)
    assert d.confidence == pytest.approx(expected_conf, abs=1e-3)          # confidence is derived, not hand-set
    assert s["anchors"] >= 1 and not s["dangling"]


@pytest.mark.parametrize("text,reason", [
    ("okay", "backchannel"), ("thanks that helps a lot", "social_ack"), ("mm-hm okay right", "backchannel"),
    ("could you repeat the previous answer please", "presentation_restructure"),
    ("make that shorter", "presentation_restructure"), ("can you summarize the retrieved evidence for me", "presentation_restructure"),
    ("translate that into hindi", "presentation_restructure"), ("what did I ask you before", "meta_conversation"),
    ("I'm going to grab a coffee first", "not_retrieval_worthy"),
])
def test_case3_suppression_and_casual_statements_skip(policy, text, reason):
    d = decide(policy, text, tick="utterance_end")
    assert d.decision == "SKIP" and d.reason == reason and d.skip_kind in {"suppressed", "not_worthy"}


def test_transform_verb_on_corpus_topic_is_a_request(policy):
    assert decide(policy, "summarize the ladder storage rule", tick="utterance_end").decision == "RETRIEVE"


def test_information_request_without_anchor_waits_early_but_retrieves_at_end(policy):
    assert decide(policy, "who won the cricket championship yesterday").decision == "WAIT"
    d = decide(policy, "who won the cricket championship yesterday", tick="utterance_end")
    assert d.decision == "RETRIEVE" and d.trigger == "final"          # Phase 2 K4: never suppress a request


def test_redundant_query_is_skipped_with_ledger_ref(policy):
    led = QueryLedger("s")
    led.create("u1", "how high the wicks should be trimmed", "t", [], ["high", "wick", "trim"], 0, "provisional", 0,
               "chunk", "r")
    d = decide(policy, "how high the wicks should be trimmed", ledger=led, now=5000)
    assert d.decision == "SKIP" and d.skip_kind == "redundant" and d.ledger_ref == "Q1"


def test_cooldown_budget_and_in_flight_guards(fixture_bundle):
    cfg, b = fixture_bundle
    led = QueryLedger("s")
    led.create("u1", "wicks", "t", [], ["wick"], 900, "provisional", 0, "chunk", "r")
    pol = make_policy(cfg, b)
    assert decide(pol, "how high the wicks should be trimmed", ledger=led, now=1000).reason == "cooldown"
    for i in range(2):
        led.create("u1", f"q{i}", "t", [], [f"x{i}"], 0, "provisional", 0, "chunk", "r")
    assert decide(pol, "how high the wicks should be trimmed", ledger=led, now=9000).reason == "provisional_budget_exhausted"
    led.create("u1", "q9", "t", [], ["y"], 0, "final", 0, "chunk", "r")
    d = decide(pol, "how high the wicks should be trimmed", tick="utterance_end", ledger=led, now=9000)
    assert d.decision == "SKIP" and d.skip_kind == "budget"
    serial = make_policy(with_controller(cfg, allow_parallel_retrieval=False), b)
    led2 = QueryLedger("s")
    led2.create("u1", "lamp", "t", [], ["lamp"], 0, "provisional", 0, "chunk", "r")
    led2.update("Q1", status="in_flight")
    assert decide(serial, "how high the wicks should be trimmed", ledger=led2, now=9000, in_flight=1).reason == "retrieval_in_flight"


def test_signals_are_monotone_in_obvious_directions(policy):
    anchored = decide(policy, "how high the wicks should be trimmed").signals
    vague = decide(policy, "how should things be done properly").signals
    assert anchored["anchor_strength"] > vague["anchor_strength"]
    assert anchored["retrieval_worthiness"] > vague["retrieval_worthiness"]
    assert decide(policy, "the fog signal and").signals["dangling"] is True


def test_ablation_policies(fixture_bundle):
    cfg, b = fixture_bundle
    end_only = make_policy(with_controller(cfg, strategy="end_only"), b)
    assert decide(end_only, "how high the wicks should be trimmed").decision == "WAIT"
    assert decide(end_only, "make that shorter", tick="utterance_end").decision == "RETRIEVE"   # no suppression
    every = make_policy(with_controller(cfg, strategy="every_chunk"), b)
    assert decide(every, "the fog signal and").decision == "RETRIEVE"                    # retrieves on fragments


def test_prototype_classifier_interface(fixture_bundle):
    cfg, b = fixture_bundle
    emb = HashingEmbedder(b.analyzer)
    pol = make_policy(with_controller(cfg, act_classifier="prototype"), b, embed=lambda t: emb.embed(t, "query"))
    assert isinstance(pol.classifier, PrototypeActClassifier) and isinstance(pol.classifier.fallback, RuleActClassifier)
    d = decide(pol, "how high the wicks should be trimmed")
    assert d.decision in {"WAIT", "RETRIEVE", "SKIP"} and 0 <= d.signals["act_confidence"] <= 1
    with pytest.raises(ValueError):
        make_policy(with_controller(cfg, act_classifier="prototype"), b)
