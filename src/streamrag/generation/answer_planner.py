"""AnswerPlanner: organise verified, evidence-derived content into an answer structure (Phase 7; docs/answer/01).

It never introduces information: its input is a ``ClaimPlan`` (evidence-derived claims) and its output only selects,
orders and groups those claims. Decisions:

* structure      one section per need, in order of mention (multi-intent answers stay separated by intent)
* detail         ``detailed``: every planned claim up to ``max_claims_per_section``; ``concise``: critical and
                 important claims only (supplementary ones are listed in ``omitted_facts``)
* order          planned order (importance first is *not* applied: the planner keeps the order of mention so that
                 atoms of one sentence stay together)
* citations      ``per_claim``: every factual sentence carries the citations of its own supporting evidence
* uncertainty    the section's gaps (no evidence / constraint not covered / conflict) are passed to the renderer,
                 which states them deterministically; the generator is told not to write about them
* reuse          sections the Phase 6 diff marks unchanged are not regenerated (``regenerate=False``)
"""

from __future__ import annotations

from streamrag.claims.models import ClaimPlan
from streamrag.generation.models import AnswerPlan, AnswerPlanSection

_RANK = {"critical": 0, "important": 1, "supplementary": 2}


class AnswerPlanner:
    def __init__(self, detail: str = "detailed", max_claims_per_section: int = 6) -> None:
        self.detail, self.cap = detail, max_claims_per_section
        self._n = 0

    def plan(self, cp: ClaimPlan, label_of: dict[str, str]) -> AnswerPlan:
        """``label_of``: evidence id -> label shown to the generator (E1..En, stable within the plan)."""
        self._n += 1
        sections = []
        for ip in sorted(cp.intents, key=lambda i: i.order):
            facts = list(ip.claims)
            if self.detail == "concise":
                keep = [f for f in facts if f.importance != "supplementary"] or facts[:1]
            else:
                keep = facts
            if len(keep) > self.cap:                    # keep the most important, preserving their order
                chosen = {f.plan_claim_id for f in sorted(keep, key=lambda f: (_RANK[f.importance], f.order))[:self.cap]}
                keep = [f for f in keep if f.plan_claim_id in chosen]
            kept = {f.plan_claim_id for f in keep}
            sections.append(AnswerPlanSection(
                section_id=ip.section_id, intent_id=ip.intent_id, title=ip.title, order=ip.order,
                constraints=ip.constraints, facts=keep, omitted_facts=[f.plan_claim_id for f in facts
                                                                       if f.plan_claim_id not in kept],
                uncertainties=ip.gaps, regenerate=ip.needs_regeneration))
        labels = {lab: eid for eid, lab in label_of.items()}
        return AnswerPlan(answer_plan_id=f"AP{self._n}", claim_plan_id=cp.plan_id, frame_id=cp.frame_id,
                          detail=self.detail, sections=sections, labels=labels, conflicts=cp.conflicts)
