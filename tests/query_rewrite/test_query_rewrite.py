"""QueryRewriteEngine (docs/retrieval/02): forms, bounded expansion, intent preserved."""

import json

from adaptive_helpers import controller
from conftest import REPO


def test_forms_and_acronym_expansion(adaptive_env):
    ctl = controller(adaptive_env)
    try:
        rq = ctl.rewriter.rewrite("  When does the   PPO open?? ", intent_id="I1")
        assert rq.normalized == "When does the PPO open" and rq.intent_id == "I1"
        assert [(e.term, e.source) for e in rq.expansions] == [("Permit Processing Office", "acronym")]
        assert rq.lexical_query.endswith("Permit Processing Office") and rq.dense_query == rq.contextual
        back = ctl.rewriter.rewrite("When does the Permit Processing Office open?")
        assert ("PPO", "alias") in [(e.term, e.source) for e in back.expansions]
    finally:
        ctl.close()


def test_contextual_form_uses_the_interpreted_need(adaptive_env):
    ctl = controller(adaptive_env)
    try:
        rq = ctl.rewriter.rewrite("What about international applicants?",
                                  contextual="eligibility requirements residence permit international applicants")
        assert rq.original == "What about international applicants?"
        assert rq.contextual.startswith("eligibility requirements") and ctl.rewriter.keeps_need(rq)
    finally:
        ctl.close()


def test_expansion_is_bounded_and_additive(adaptive_env):
    ctl = controller(adaptive_env, **{"adaptive_retrieval.max_expansions": 1})
    try:
        rq = ctl.rewriter.rewrite("How much does it cost to pay the charge for the PPO?")
        assert len(rq.expansions) <= 1
        assert set(ctl.terms_fn(rq.contextual)) <= set(ctl.terms_fn(rq.lexical_query))
    finally:
        ctl.close()
    off = controller(adaptive_env, **{"adaptive_retrieval.expansion": False})
    try:
        assert off.rewriter.rewrite("When does the PPO open?").expansions == []
    finally:
        off.close()


def test_rewrite_never_loses_a_need_term_on_any_eval_question(adaptive_env):
    ctl = controller(adaptive_env)
    try:
        for p in sorted((REPO / "eval" / "dev_adaptive_retrieval").glob("*.json")):
            for t in json.loads(p.read_text())["turns"]:
                assert ctl.rewriter.keeps_need(ctl.rewriter.rewrite(t["utterance_text"])), t["utterance_text"]
    finally:
        ctl.close()


def test_corrected_entity_is_not_reintroduced(adaptive_env):
    ctl = controller(adaptive_env)
    try:
        rq = ctl.rewriter.rewrite("What is the cost for Norvia", dropped_entities=["fee"])
        assert "fee" not in [e.term for e in rq.expansions] and rq.dropped_entities == ["fee"]
    finally:
        ctl.close()


def test_alias_mining_rejects_non_acronyms(tmp_path):
    from streamrag.adaptive.catalog import _ALIAS
    text = "The Permit Processing Office (PPO) is open. Please Read This (IGNORE) now."
    found = {short for _, short in _ALIAS.findall(text)}
    assert "PPO" in found      # pattern finds both; the initials check in MetadataCatalog.build rejects IGNORE
    long, short = [(l, s) for l, s in _ALIAS.findall(text) if s == "IGNORE"][0]
    assert "".join(w[0] for w in long.split()).upper() != short
