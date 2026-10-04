"""Claim extraction from generated output (Phase 7; docs/answer/03). The brief's "ClaimExtractor" for generated
answers (Phase 6's ``claims.graph.ClaimExtractor`` selects claims from *evidence*; this one reads *generator output*).

Structured output makes extraction deterministic: each generated sentence is one candidate claim (an output item that
packs several sentences is split; its facts and labels apply to each part). Per sentence:

* inline markers the model wrote into the text ("... per shift. (E1)") are removed from the claim and treated as
  cited labels (a marker is a citation, never part of the statement)
* codes / acronyms the model mangled only in punctuation or case ("IEL:TS") get the spelling of the planned facts
  ("IELTS"; Phase 11, ``textcheck.canonical_codes``) - letters and digits never change
* kind: ``connective`` when it has no content term and <= 4 words ("In addition:"), else ``fact``
* facts: the planned-fact ids it lists that exist in its section (unknown ids are dropped and recorded)
* cited evidence: the evidence of its valid labels plus the evidence of its valid facts; unknown labels are
  recorded as invalid (L0) - they never become citations
* importance: the highest importance of its facts; without facts ``important`` if it carries a number / date
  (material), else ``supplementary``

Atomicity is handled by the verifier: a sentence that is not supported as a whole is decomposed into atoms there.
"""

from __future__ import annotations

import re
from collections.abc import Callable

from pydantic import Field

from streamrag.claims.graph import sentences
from streamrag.claims.models import Importance
from streamrag.claims.textcheck import canonical_codes, strip_markers
from streamrag.generation.models import AnswerPlan, CandidateAnswer
from streamrag.models.base import Contract
from streamrag.validation.policy import material

_ORDER = {"critical": 0, "important": 1, "supplementary": 2}


class ExtractedClaim(Contract):
    claim_key: str
    section_id: str
    intent_id: str
    text: str = Field(min_length=1)
    kind: str = "fact"
    facts: list[str] = []
    fact_keys: list[str] = []
    cited_evidence: list[str] = []
    invalid_labels: list[str] = []
    unknown_facts: list[str] = []
    importance: Importance = "important"
    origin: str = "llm"


class GeneratedClaimExtractor:
    def __init__(self, terms_fn: Callable[[str], list[str]]) -> None:
        self.terms_fn = terms_fn

    def extract(self, cand: CandidateAnswer, plan: AnswerPlan) -> list[ExtractedClaim]:
        out = []
        reference = [f.text for ps in plan.sections for f in ps.facts]
        for n, s in enumerate(cand.sentences):
            text, markers = strip_markers(s.text)
            text = canonical_codes(text, reference)
            parts = [text[a:b] for a, b in sentences(text)] or ([text] if text else [])
            for k, part in enumerate(parts):
                key = f"{s.section_id}#{n}" + (f".{k}" if len(parts) > 1 else "")
                out.append(self._claim(s, part, markers, key, plan))
        return out

    def _claim(self, s, text: str, markers: list[str], key: str, plan: AnswerPlan) -> ExtractedClaim:
        sec = {f.plan_claim_id: f for ps in plan.sections if ps.section_id == s.section_id for f in ps.facts}
        words = re.findall(r"\w+", text)
        kind = "connective" if len(words) <= 4 and not self.terms_fn(text) else "fact"
        labels = list(dict.fromkeys(s.labels + [m for m in markers if m.startswith("E")]))
        fs = [f for f in dict.fromkeys(s.facts + [m for m in markers if m.startswith("F")]) if f in sec]
        unknown = [f for f in s.facts if f not in sec]
        ev = [plan.labels[lab] for lab in labels if lab in plan.labels]
        ev += [e for f in fs for e in sec[f].evidence_ids if e not in ev]
        invalid = [lab for lab in labels if lab not in plan.labels]
        if fs:
            imp = min((sec[f].importance for f in fs), key=lambda x: _ORDER[x])
        else:
            imp = "important" if material(text) else "supplementary"
        return ExtractedClaim(
            claim_key=key, section_id=s.section_id, intent_id=s.intent_id, text=text,
            kind=kind, facts=fs, fact_keys=[fact_key(sec[f]) for f in fs], cited_evidence=ev,
            invalid_labels=invalid, unknown_facts=unknown, importance=imp, origin=s.origin)


def fact_key(f) -> str:
    """Stable identity of a planned fact across plans: its source claim + its text."""
    return f"{f.phase6_claim_id}|{f.text}"
