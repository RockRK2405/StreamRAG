"""Evaluator validation (brief §55): every metric on known examples gives the known value."""

import math

import pytest

from streamrag.evaluation.dataset import ExpectedClaim as C
from streamrag.evaluation.metrics import citation, claims, evidence, generation, hallucination, latency, retrieval


def test_perfect_retrieval_and_missing_evidence():
    gold = ["D §1", "D §2"]
    r = retrieval.compute(["D §1", "D §2", "X §1"], gold)
    assert r["recall@1"] == 0.5 and r["recall@3"] == 1.0 and r["mrr"] == 1.0 and r["ndcg@10"] == pytest.approx(1.0)
    assert r["precision@1"] == 1.0 and r["precision@3"] == pytest.approx(2 / 3)
    miss = retrieval.compute(["X §1", "Y §2"], gold)
    assert miss["recall@10"] == 0.0 and miss["mrr"] == 0.0 and miss["ndcg@10"] == 0.0
    late = retrieval.compute(["X §1", "D §2"], gold)
    assert late["mrr"] == 0.5 and late["ndcg@10"] == pytest.approx((1 / math.log2(3)) / (1 + 1 / math.log2(3)))


def test_no_gold_means_not_computed_and_any_semantics():
    assert retrieval.compute(["D §1"], [])["recall@5"] is None
    any_ = retrieval.compute(["X", "B"], ["A", "B", "C"], "any")
    assert any_["recall@1"] == 0.0 and any_["recall@3"] == 1.0 and any_["ndcg@10"] == pytest.approx(1 / math.log2(3))


def test_duplicate_keys_counted_once():
    assert retrieval.compute(["D §1", "D §1", "D §1"], ["D §1", "D §2"])["recall@3"] == 0.5


def test_evidence_metrics():
    items = [{"citation": "D §1", "document_id": "D", "text": "The fee is 55 euros."},
             {"citation": "E §1", "document_id": "E", "text": "Old fee was 40 euros."},
             {"citation": "F §1", "document_id": "F", "text": "Unrelated."}]
    ec = [C(text="fee 55", key=["55 euros"]), C(text="other", key=["99 days"])]
    m = evidence.compute(items, ["D §1"], ec, [r"\b40 euros"], "TEMPORAL")
    assert m["evidence_recall"] == 1.0 and m["evidence_precision"] == pytest.approx(1 / 3)
    assert m["evidence_coverage"] == 0.5 and m["unsupported_evidence"] == pytest.approx(2 / 3)
    assert m["contradictory_evidence"] == 1.0 and m["source_diversity"] == 1.0
    assert evidence.compute(items, [], [], [r"40"], "CONTRADICTORY")["contradictory_evidence"] is None


def test_answer_correctness_forbidden_abstention_and_conflict():
    ec = [C(text="fee", key=["55 euros"]), C(text="time", key=["21 days"])]
    good = generation.compute("The fee is 55 euros and refunds take 21 days.", ec, [r"\b40 euros"], "SUFFICIENT", [],
                              ["SUPPORTED", "SUPPORTED"], [True, False])
    assert good["answer_correct"] == 1.0 and good["completeness"] == 1.0 and good["groundedness"] == 0.5
    stale = generation.compute("The fee is 55 euros, earlier 40 euros; refunds take 21 days.", ec, [r"\b40 euros"],
                               "SUFFICIENT", [], [], [])
    assert stale["answer_correct"] == 0.0 and stale["forbidden_hit"] == 1.0
    partial = generation.compute("The fee is 55 euros.", ec, [], "SUFFICIENT", [], [], [])
    assert partial["answer_correct"] == 0.0 and partial["completeness"] == 0.5
    abst = generation.compute("The documents do not say whether buses have wifi.", [], [], "INSUFFICIENT", [], [], [])
    assert abst["insufficiency_ok"] == 1.0
    wrong = generation.compute("Yes, there is free wifi.", [], [], "INSUFFICIENT", [], [], [])
    assert wrong["insufficiency_ok"] == 0.0
    conf = generation.compute("Every 20 minutes or every 30 minutes.", [], [], "CONTRADICTORY",
                              ["20 minutes", "30 minutes"], [], [])
    assert conf["conflict_reported"] == 1.0


def _claim(factual=True, cites=(), cited_support=False, verdict="SUPPORTED"):
    return {"factual": factual, "citations": list(cites), "cited_support": cited_support, "verdict": verdict,
            "cited": bool(cites), "text": "The fee is 55 euros."}


def test_correct_and_incorrect_citations():
    ok = {"key": "D §1", "valid": True, "supports_own": True, "supports_other": False}
    wrong = {"key": "E §1", "valid": True, "supports_own": False, "supports_other": False}
    misplaced = {"key": "F §1", "valid": True, "supports_own": False, "supports_other": True}
    invalid = {"key": "Z §9", "valid": False, "supports_own": False, "supports_other": False}
    m = citation.compute([_claim(cites=[ok], cited_support=True)], [C(text="x", citation="D §1", key=["55"])])
    assert m["citation_precision"] == 1.0 and m["citation_recall"] == 1.0 and m["citation_completeness"] == 1.0
    assert m["source_validity"] == 1.0 and m["position_correct"] == 1.0
    bad = citation.compute([_claim(cites=[wrong]), _claim(cites=[misplaced]), _claim(cites=[invalid])],
                           [C(text="x", citation="D §1", key=["55"])])
    assert bad["citation_precision"] == 0.0 and bad["citation_recall"] == 0.0 and bad["citation_completeness"] == 0.0
    assert bad["source_validity"] == pytest.approx(2 / 3) and bad["position_correct"] == 0.0


def test_hallucination_terms_are_distinct():
    cl = [{"text": "The fee is 55 euros.", "factual": True, "verdict": "SUPPORTED", "cited": True},
          {"text": "The fee is 99 euros.", "factual": True, "verdict": "UNSUPPORTED", "cited": False},
          {"text": "The documents do not say.", "factual": False, "verdict": "UNSUPPORTED", "cited": False}]
    m = hallucination.compute(cl, "The application fee is 55 euros.", "What is the fee?")
    assert m["factual_claims"] == 2 and m["unsupported_claim_rate"] == 0.5
    assert m["hallucinated_claim_rate"] == 0.5 and m["citationless_fact_rate"] == 0.5 and m["grounding_failure"] == 1.0
    assert hallucination.compute([], "x", "y")["unsupported_claim_rate"] is None


def test_claim_metrics():
    m = claims.compute(["SUPPORTED", "UNSUPPORTED", "CONTRADICTED", "SUPPORTED"], "fee 55 euros",
                       [C(text="a", key=["55 euros"]), C(text="b", key=["7 days"])])
    assert m["claim_support_rate"] == 0.5 and m["unsupported_claim_rate"] == 0.25
    assert m["contradicted_claim_rate"] == 0.25 and m["claim_coverage"] == 0.5


def test_latency_percentiles_and_small_samples():
    p = latency.percentiles(list(range(1, 101)))
    assert p["p50"] == 50.5 and p["p90"] == pytest.approx(90.1) and p["p99"] == pytest.approx(99.01)
    small = latency.percentiles([1, 2, 3])
    assert small["p50"] == 2 and small["p90"] is None and small["p95"] is None and small["p99"] is None
    assert latency.percentiles([])["n"] == 0
