"""Automatic error categorisation (Phase 10; docs/evaluation/error_analysis.md).

Each evaluated turn gets zero or more categories, decided by rules over its measured metrics and labels (first
failing stage first; a turn can have several):

  ORCHESTRATION_FAILURE   the sample failed, or the turn never completed (runtime)
  LATENCY_FAILURE         validated answer later than the latency gate (``latency_gate_ms``, after utterance end)
  QUERY_ANALYSIS_FAILURE  adaptive systems: a MULTI_HOP sample not analysed MULTI_HOP, or a constraint sample whose
                          plan applied no filter / condition (needs the adaptive trace)
  ENTITY_FAILURE          ENTITY_CORRECTION / STREAMING_CORRECTION sample whose answer matches a forbidden
                          (replaced-entity) value or misses the corrected entity's facts
  MEMORY_FAILURE          CONTEXTUAL_FOLLOWUP sample whose retrieval found none of the gold evidence
  RETRIEVAL_FAILURE       other answerable samples: retrieved list contains no gold section (recall@10 = 0)
  EVIDENCE_FAILURE        gold retrieved somewhere but the evidence handed to the answer stage lacks the expected
                          facts (evidence coverage < 1), or stale / forbidden evidence handed on
  CLAIM_FAILURE           answer has unsupported factual claims (verifier-judged)
  GENERATION_FAILURE      expected facts available in the evidence but not stated, or a forbidden value stated, or
                          an INSUFFICIENT sample answered without abstaining, or a conflict not reported
  CITATION_FAILURE        a factual claim without citation, or a citation that does not support its claim
"""

from __future__ import annotations

CATEGORIES = ("RETRIEVAL_FAILURE", "QUERY_ANALYSIS_FAILURE", "ENTITY_FAILURE", "MEMORY_FAILURE", "EVIDENCE_FAILURE",
              "CLAIM_FAILURE", "GENERATION_FAILURE", "CITATION_FAILURE", "LATENCY_FAILURE", "ORCHESTRATION_FAILURE")


def categorize(sample, row: dict, latency_gate_ms: float | None = None) -> list[str]:
    m = row["metrics"]
    cats: list[str] = []
    if row["status"] != "ok" or (row.get("error") or "").startswith("turn_not_completed"):
        return ["ORCHESTRATION_FAILURE"]
    ttva = m["latency"].get("ttva_after_end")
    if latency_gate_ms is not None and ttva is not None and ttva > latency_gate_ms:
        cats.append("LATENCY_FAILURE")
    ad = row.get("adaptive") or {}
    if ad.get("complexity"):
        if sample.query_type == "MULTI_HOP" and "MULTI_HOP" not in ad["complexity"]:
            cats.append("QUERY_ANALYSIS_FAILURE")
        if sample.required_constraints and ad.get("strategies") and not any(
                s in ("FILTERED", "ITERATIVE", "SESSION_REUSE", "CACHE_REUSE") for s in ad["strategies"]):
            cats.append("QUERY_ANALYSIS_FAILURE")
    r10 = m["retrieval"].get("recall@10")
    g = m["generation"]
    if sample.query_type in ("ENTITY_CORRECTION", "STREAMING_CORRECTION") and (
            g.get("forbidden_hit") == 1.0 or (g.get("completeness") is not None and g["completeness"] < 1)):
        cats.append("ENTITY_FAILURE")
    if sample.ground_truth_evidence and r10 == 0.0:
        cats.append("MEMORY_FAILURE" if sample.query_type == "CONTEXTUAL_FOLLOWUP" else "RETRIEVAL_FAILURE")
    ev = m["evidence"]
    if (ev.get("evidence_coverage") is not None and ev["evidence_coverage"] < 1 and r10 not in (None, 0.0)) or \
            ev.get("contradictory_evidence") == 1.0:
        cats.append("EVIDENCE_FAILURE")
    h = m["hallucination"]
    if h.get("unsupported_claim_rate"):
        cats.append("CLAIM_FAILURE")
    gen_fail = (g.get("forbidden_hit") == 1.0 or g.get("insufficiency_ok") == 0.0 or g.get("conflict_reported") == 0.0
                or (g.get("completeness") is not None and g["completeness"] < 1
                    and (ev.get("evidence_coverage") or 0) >= 1))
    if gen_fail:
        cats.append("GENERATION_FAILURE")
    c = m["citation"]
    if (h.get("citationless_fact_rate") or 0) > 0 or (c.get("citation_precision") is not None
                                                       and c["citation_precision"] < 1):
        cats.append("CITATION_FAILURE")
    return list(dict.fromkeys(cats))
