"""Answer-level metrics against the labels (model-free, deterministic).

  answer_correct     1 if every expected claim is stated (all its key strings occur in the answer) and no forbidden
                     pattern (stale / unsupported value) matches; 0 otherwise           (expected claims needed)
  completeness       expected claims stated / expected claims                            (same as claim coverage)
  forbidden_hit      1 if a forbidden pattern matches the answer                         (forbidden needed)
  abstained          the answer says the information is not available (Phase 7 ABSTAIN patterns), or the answer is
                     empty (no assertion made - an empty answer is a non-answer, not an explicit abstention)
  insufficiency_ok   INSUFFICIENT samples: abstained and no forbidden match              (expected_state needed)
  conflict_reported  CONTRADICTORY samples with conflict values: every value occurs in the answer
  faithfulness       verifier-judged claim support rate (see claims.py)
  groundedness       claims that are SUPPORTED and carry a citation / claims
Answer relevance is NOT MEASURED: it needs human or judge scores (no LLM judge is used, see the report).
"""

from __future__ import annotations

import re

from streamrag.bench.grounded import ABSTAIN


def compute(answer: str, expected_claims: list, forbidden: list[str], expected_state: str, conflict_values: list[str],
            verdicts: list[str], cited: list[bool]) -> dict:
    keyed = [c for c in expected_claims if c.key]
    low = answer.lower()
    stated = [all(k.lower() in low for k in c.key) for c in keyed]
    fhit = any(re.search(p, answer, re.I) for p in forbidden) if forbidden else False
    abst = bool(ABSTAIN.search(answer)) or not answer.strip()
    n = len(verdicts)
    return {
        "answer_correct": (float(all(stated) and not fhit)) if keyed else None,
        "completeness": (sum(stated) / len(stated)) if keyed else None,
        "forbidden_hit": float(fhit) if forbidden else None,
        "abstained": float(abst),
        "insufficiency_ok": (float(abst and not fhit)) if expected_state == "INSUFFICIENT" else None,
        "conflict_reported": (float(all(v.lower() in low for v in conflict_values)))
        if expected_state == "CONTRADICTORY" and conflict_values else None,
        "faithfulness": (sum(v == "SUPPORTED" for v in verdicts) / n) if n else None,
        "groundedness": (sum(1 for v, c in zip(verdicts, cited) if v == "SUPPORTED" and c) / n) if n else None,
    }
