"""Unsupported-claim policy (Phase 7; docs/answer/07). Deterministic: the action depends only on the verification
result, whether the claim expresses planned facts, its importance and the configured mode / budgets.

| Verification                          | Claim                                   | Action (in order)                         |
|---------------------------------------|-----------------------------------------|-------------------------------------------|
| SUPPORTED                             | any                                     | keep (citations come from the verification)|
| CONTRADICTED and also supported       | any                                     | present_conflict: both sides, both cited  |
| PARTIALLY_SUPPORTED                   | any                                     | keep_atoms: only the supported atoms stay |
| UNSUPPORTED / CONTRADICTED            | expresses planned facts                 | restore_facts: the planned evidence-derived facts verbatim |
| UNSUPPORTED                           | unplanned, material (number / date)     | retrieve (budget) -> keep if now supported, else remove |
| UNSUPPORTED / CONTRADICTED            | otherwise                               | remove                                    |

``remove`` never deletes silently: the claim stays in ``rejected`` with its reason, and the answer records that
unverifiable content was removed. Modes:
  strict   any removed factual claim triggers a revision of its section (regenerate with feedback, at most
           ``max_answer_revision_attempts``; then the extractive rendering of the section's planned facts)
  relaxed  only a missing critical fact triggers a revision; removed non-critical content is just dropped
Retrieval fallback is bounded by ``max_validation_retrievals`` per answer version.
"""

from __future__ import annotations

from typing import Literal

from streamrag.claims.models import ClaimVerification
from streamrag.claims.textcheck import numbers

Action = Literal["keep", "present_conflict", "keep_atoms", "restore_facts", "retrieve", "remove"]


def material(text: str) -> bool:
    """A claim carrying a number, amount, date or ordinal is material: wrong values mislead most."""
    return bool(numbers(text, ordinals=False))


def decide(v: ClaimVerification, text: str, planned_facts: list[str], retrievals_left: int) -> Action:
    if v.status == "SUPPORTED":
        return "keep"
    if v.status == "CONTRADICTED" and v.supporting_evidence:
        return "present_conflict"
    if v.status == "PARTIALLY_SUPPORTED":
        return "keep_atoms"
    if planned_facts:
        return "restore_facts"
    if v.status == "UNSUPPORTED" and retrievals_left > 0 and material(text):
        return "retrieve"
    return "remove"


def needs_revision(mode: str, removed_importance: list[str], missing_critical: bool) -> bool:
    if missing_critical:
        return True
    return mode == "strict" and bool(removed_importance)
