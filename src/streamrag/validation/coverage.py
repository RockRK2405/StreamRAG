"""AnswerCoverageValidator and evidence-coverage maps (Phase 7; docs/answer/06).

An important active intent (every need of the frame's answer) is *covered* when its section contains at least one
supported factual claim, and *explicitly handled* when it contains an uncertainty statement instead (no evidence /
constraint not covered / conflict). A need that is neither is a coverage failure and blocks finalization: the system
never silently declares success.

Traceability maps returned with the report: claim -> evidence, intent -> claims, section -> intent.
"""

from __future__ import annotations

from pydantic import Field

from streamrag.models.base import Contract


class CoverageReport(Contract):
    intents: list[str] = []
    covered: list[str] = []                          # >= 1 supported factual claim
    uncertain_only: list[str] = []                   # handled explicitly with uncertainty, no facts
    failures: list[str] = []                         # neither: blocks finalization
    intent_coverage: float | None = None             # (covered + uncertain_only) / intents
    fact_coverage: float | None = None               # covered / intents
    claim_to_evidence: dict[str, list[str]] = {}
    intent_to_claims: dict[str, list[str]] = {}
    section_to_intent: dict[str, str] = {}
    critical_facts_missing: dict[str, list[str]] = Field(default_factory=dict)   # section -> planned critical ids


class AnswerCoverageValidator:
    def validate(self, sections, claims: dict, critical_planned: dict[str, list[str]],
                 expressed_facts: dict[str, set[str]]) -> CoverageReport:
        """sections: GroundedSection list; claims: claim id -> AnswerClaim; critical_planned: section -> critical
        planned-fact ids; expressed_facts: section -> planned-fact ids expressed by supported claims."""
        intents, covered, uncertain, failures = [], [], [], []
        c2e, i2c, s2i, missing = {}, {}, {}, {}
        for s in sections:
            intents.append(s.intent_id)
            s2i[s.section_id] = s.intent_id
            facts = [claims[c] for c in s.claim_ids if claims[c].kind in ("fact", "conflict")]
            unc = [claims[c] for c in s.claim_ids if claims[c].kind == "uncertainty"]
            i2c[s.intent_id] = [c.claim_id for c in facts]
            for c in facts:
                c2e[c.claim_id] = list(c.evidence_ids)
            if facts:
                covered.append(s.intent_id)
            elif unc:
                uncertain.append(s.intent_id)
            else:
                failures.append(s.intent_id)
            miss = [f for f in critical_planned.get(s.section_id, []) if f not in expressed_facts.get(s.section_id, set())]
            if miss:
                missing[s.section_id] = miss
        n = len(intents)
        return CoverageReport(intents=intents, covered=covered, uncertain_only=uncertain, failures=failures,
                              intent_coverage=round((len(covered) + len(uncertain)) / n, 4) if n else None,
                              fact_coverage=round(len(covered) / n, 4) if n else None, claim_to_evidence=c2e,
                              intent_to_claims=i2c, section_to_intent=s2i, critical_facts_missing=missing)
