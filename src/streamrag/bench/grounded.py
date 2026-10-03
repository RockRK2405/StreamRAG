"""Phase 7 dev-suite scoring (eval/dev_grounded). TEST FIXTURE DOMAIN ONLY - NOT REPORTABLE.

Two kinds of numbers, never mixed:
* **gold-based** (team-written labels): required-fact recall (a required fact counts as present when some factual
  claim of the answer entails it - entailment-model matcher, reported as such), forbidden-assertion hits (regexes for
  claims the corpus does not support), conflict presentation (every disagreeing value present), expected
  uncertainty expressed.
* **verifier-judged** (the system's own claim verifier): support rate of raw generated claims, unsupported rate of
  final claims, citation precision / coverage. Their agreement with labelled claims is measured separately
  (research/phase7: perturbation set with labels by construction, hand-labelled sample).

Free-form ablation arms (A plain LLM, B RAG, C RAG + labels, D C + validation, E D + repair) are run with
``run_free_arm``; the full system (F) and the extractive reference (X) run through the normal pipeline.
"""

from __future__ import annotations

import re
import time

from streamrag.claims.graph import sentences
from streamrag.claims.models import SUPPORTING
from streamrag.claims.textcheck import strip_markers
from streamrag.validation.policy import material

BANNER = "DEV SUITE ON TEST FIXTURE DOMAIN - NOT AN OFFICIAL BENCHMARK RESULT - NOT HELD-OUT"
ABSTAIN = re.compile(r"\b(not (specified|stated|mentioned|provided|available|given|established|included|indicated|"
                     r"described|covered)|does not (say|state|mention|specify|provide|include|contain|indicate)|"
                     r"do not (say|state|mention|specify|provide|include|contain|indicate)|no information|"
                     r"cannot be determined|unknown|not clear|unclear)\b", re.I)


def gold_recall(fact_texts: list[str], required: list[dict], nli) -> tuple[int, int, list[str]]:
    """(found, total, missing texts): a required fact is found when some answer claim - or one sentence of it - entails
    it (the entailment model misjudges multi-sentence premises: a 3-sentence claim whose first sentence *is* the gold
    fact is judged neutral)."""
    if not required:
        return 0, 0, []
    premises = list(dict.fromkeys([t for t in fact_texts] + [t[a:b] for t in fact_texts for a, b in sentences(t)]))
    missing = []
    found = 0
    for g in required:
        pairs = [(t, g["text"]) for t in premises]
        ok = any(r["label"] == "entailment" for r in nli.predict(pairs)) if pairs else False
        found += ok
        if not ok:
            missing.append(g["text"])
    return found, len(required), missing


def forbidden_hits(fact_texts: list[str], patterns: list[str]) -> list[str]:
    return [t for t in fact_texts for p in patterns if re.search(p, t, re.I)]


def conflict_presented(text: str, values: list[str]) -> bool | None:
    if not values:
        return None
    return all(v.lower() in text.lower() for v in values)


def abstains(texts: list[str]) -> bool:
    return any(ABSTAIN.search(t) for t in texts)


def run_free_arm(arm: str, question: str, pool: dict[str, str], citation_of: dict[str, str], generator, verifier,
                 mapper, retrieve_fn=None, retrieval_budget: int = 1) -> dict:
    """Ablation arms A-E. pool: evidence id -> text (the same evidence the full system used for the turn)."""
    mode = {"A": "none", "B": "evidence", "C": "evidence_labels", "D": "evidence_labels", "E": "evidence_labels"}[arm]
    ids = list(pool)
    labels = {f"E{k + 1}": e for k, e in enumerate(ids)}
    lab_of = {e: lab for lab, e in labels.items()}
    t0 = time.perf_counter()
    cand = generator.generate_free(question, [(lab_of[e], citation_of.get(e, e), pool[e]) for e in ids], mode)
    gen_ms = (time.perf_counter() - t0) * 1000.0
    t1 = time.perf_counter()
    raw, final, cits_emitted, cits_valid = [], [], 0, 0
    pool = dict(pool)
    budget = retrieval_budget
    parts = []
    for s0 in cand.sentences:                           # same extraction rules as the system: markers = citations,
        text, markers = strip_markers(s0.text)          # one claim per sentence
        for a, b in sentences(text):
            parts.append(s0.model_copy(update={"text": text[a:b], "labels": list(dict.fromkeys(s0.labels + markers))}))
    for k, s in enumerate(parts):
        cited = [labels[x] for x in s.labels if x in labels]
        v = verifier.verify(f"x{k}", s.text, cited, pool)
        abstention = bool(ABSTAIN.search(s.text)) and not v.supported
        raw.append({"text": s.text, "status": v.status, "abstention": abstention, "cited": cited,
                    "supporting": v.supporting_evidence})
        if arm in ("A", "B", "C"):
            if not abstention:
                final.append({"text": s.text, "supported": v.supported, "evidence": cited})
            if arm == "C":
                for e in cited:
                    cits_emitted += 1
                    als = [a for a in v.alignments if a.evidence_id == e]
                    cits_valid += bool(als and als[0].strength in SUPPORTING)
            continue
        if abstention:
            continue
        if v.status == "SUPPORTED" or (v.status == "CONTRADICTED" and v.supporting_evidence):
            final.append({"text": s.text, "supported": True, "evidence": v.supporting_evidence})
        elif arm == "E" and v.status == "PARTIALLY_SUPPORTED":
            for t, a in zip(v.atom_texts, v.atoms):
                if a.supported:
                    final.append({"text": t, "supported": True, "evidence": a.supporting_evidence})
        elif arm == "E" and v.status == "UNSUPPORTED" and budget > 0 and retrieve_fn and material(s.text):
            budget -= 1
            for e, t in retrieve_fn(s.text).items():
                pool[e] = t
            vv = verifier.verify(f"x{k}r", s.text, [], pool)
            if vv.supported:
                final.append({"text": s.text, "supported": True, "evidence": vv.supporting_evidence})
    if arm in ("D", "E"):
        cmap = mapper.map("arm", {f"f{k}": verifier.verify(f"f{k}", f["text"], f["evidence"], pool)
                                  for k, f in enumerate(final)}, lab_of)
        cits_emitted = cits_valid = len(cmap.citations)
    val_ms = (time.perf_counter() - t1) * 1000.0
    facts = [f for f in final]
    return {
        "arm": arm, "question": question, "raw": raw, "final": facts,
        "raw_claims": len(raw),
        "raw_supported": sum(1 for r in raw if r["status"] == "SUPPORTED" or (r["status"] == "CONTRADICTED"
                                                                             and r["supporting"])),
        "raw_unsupported": sum(1 for r in raw if r["status"] in ("UNSUPPORTED", "CONTRADICTED") and not r["supporting"]
                               and not r["abstention"]),
        "raw_partial": sum(1 for r in raw if r["status"] == "PARTIALLY_SUPPORTED"),
        "raw_abstentions": sum(1 for r in raw if r["abstention"]),
        "final_claims": len(facts), "final_unsupported": sum(1 for f in facts if not f["supported"]),
        "citations_emitted": cits_emitted, "citations_valid": cits_valid,
        "final_cited": sum(1 for f in facts if f["evidence"]) if arm in ("C", "D", "E") else 0,
        "abstained": any(r["abstention"] for r in raw), "structured_ok": cand.structured_ok,
        "llm_calls": cand.llm_calls, "prompt_tokens": cand.prompt_tokens, "output_tokens": cand.output_tokens,
        "ttft_ms": cand.ttft_ms, "generation_ms": gen_ms, "validation_ms": val_ms,
        "total_ms": gen_ms + val_ms, "text": " ".join(f["text"] for f in facts)}
