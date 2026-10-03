"""ClaimPlanner: which evidence-derived claims should the answer express? (Phase 7; docs/answer/01)

Input: the Phase 6 answer state of a topic frame (an ``AnswerVersion``: one section per active need with its usable
extractive claims, evidence, active constraints and uncertainty items), the claim graph, the evidence store and the
intent tracker. In brief terms: UnifiedEvidenceSet = the usable evidence of each need, IntentSet = the frame's active
needs, SessionState = the Phase 6 memory.

Output: a ``ClaimPlan``. Claims are **derived from evidence, never generated**: each planned claim is a verbatim
evidence sentence, or an atom of it produced by the ClaimDecomposer and accepted only if the entailment check says
the source sentence entails it (otherwise the whole sentence is kept).

Importance (deterministic, recorded in ``importance_basis``):
  critical       claims the need's active constraints are covered by (Phase 6 SUPPORTED with constraints), or,
                 without constraints, the need's highest-ranked claim
  important      the other SUPPORTED claims
  supplementary  PARTIALLY_SUPPORTED claims (true, but do not address an active constraint)
Ordering: section order (order of mention), then claim rank (Phase 6 relevance, evidence rank), atoms in sentence
order. Dependencies: an atom that borrowed its subject from an earlier atom of the same sentence depends on it.
Gaps: no evidence / constraint not covered / evidence conflict, copied from the Phase 6 section uncertainty; plus
``value_not_stated`` when the need
asks for a value of a kind ("how long" -> duration, "how much" / "fee" -> amount,
"when" -> date/time, ... - lexicon) and no planned fact both states a value of that kind and shares >= 2 content
terms with the question (similar evidence is not an answer: the answer must say that the value is not stated).
"""

from __future__ import annotations

import re

from streamrag.claims.aligner import ClaimEvidenceAligner
from streamrag.claims.decomposer import ClaimDecomposer
from streamrag.claims.models import ClaimPlan, IntentGap, IntentPlan, PlannedClaim
from streamrag.claims.textcheck import counts_noun, has_value_of, instruction_like


class ClaimPlanner:
    def __init__(self, decomposer: ClaimDecomposer, aligner: ClaimEvidenceAligner, is_boilerplate=None) -> None:
        self.decomposer, self.aligner = decomposer, aligner
        self.is_boilerplate = is_boilerplate or (lambda s: False)
        self._n = 0
        self._atom_cache: dict[str, list[str]] = {}

    def _atoms(self, sentence: str) -> list[str]:
        if sentence in self._atom_cache:
            return self._atom_cache[sentence]
        atoms = self.decomposer.decompose(sentence)
        if len(atoms) > 1:
            al = [self.aligner.align("atom", a, [("src", sentence)])[0] for a in atoms]
            if not all(x.strength in ("STRONG", "MODERATE") for x in al):
                atoms = [sentence]                      # the split is not entailed by the source: keep it whole
        self._atom_cache[sentence] = atoms
        return atoms

    def _counted(self, question: str) -> set[str]:
        """Terms of the (at most 2-word) noun after "how many" ("how many working days" -> {work, day})."""
        m = re.search(r"how many\s+(\w+(?:\s+\w+)?)", question.lower())
        return set(self.aligner.terms_fn(m.group(1))) if m else set()

    def plan(self, answer, graph, store, tracker, conflicts: dict | None = None) -> ClaimPlan:
        self._n += 1
        intents = []
        n_fact = 0
        for sec in answer.sections:
            it = tracker.intents.get(sec.intent_id)
            ks = [k.text for k in tracker.constraints_for(it)] if it is not None else list(sec.constraints)
            claims = [graph.claims[c] for c in sec.claim_ids if c in graph.claims]
            supported = [c for c in claims if c.status == "SUPPORTED"]
            planned: list[PlannedClaim] = []
            for c in claims:
                if c.source is None or self.is_boilerplate(c.text) or instruction_like(c.text):
                    continue                              # template text / injected instructions are not facts
                if c.status == "PARTIALLY_SUPPORTED":
                    imp, basis = "supplementary", "does_not_address_active_constraint"
                elif ks:
                    imp, basis = "critical", "covers_active_constraints"
                elif supported and c.claim_id == supported[0].claim_id:
                    imp, basis = "critical", "highest_ranked_claim_of_need"
                else:
                    imp, basis = "important", "supported_claim_of_need"
                atoms = self._atoms(c.text)
                prev: list[str] = []
                for a in atoms:
                    n_fact += 1
                    pid = f"F{n_fact}"
                    planned.append(PlannedClaim(
                        plan_claim_id=pid, intent_id=sec.intent_id, section_id=sec.section_id, text=a,
                        source=c.source, evidence_ids=list(c.evidence_ids), phase6_claim_id=c.claim_id,
                        atom_of=c.claim_id if len(atoms) > 1 else None, importance=imp, importance_basis=basis,
                        order=len(planned), depends_on=list(prev[:1]) if len(atoms) > 1 else [],
                        entailed_by_source=True))
                    prev.append(pid)
            gaps = [IntentGap(intent_id=sec.intent_id, kind=u.kind, aspect=u.aspect)
                    for u in sec.uncertainty if u.kind in ("no_evidence", "constraint_not_covered", "conflict")]
            asked = it.text if it is not None else sec.title
            cues = " ".join(it.type_cues) if it is not None else ""
            kind = self.decomposer.lx.asks_value(f"{cues} {asked}")
            if planned and kind:
                qt = set(self.aligner.terms_fn(asked))
                counted = self._counted(asked) if kind == "count" else set()
                # a value that names its own property ("18 years old", "40 crates" for "how many crates") only needs
                # the subject in common; a bare value ("2 years", "every week") needs >= 2 shared topic terms
                own = kind == "age" or bool(counted)
                need = min(1 if own else 2, len(qt))
                answering = [f for f in planned if has_value_of(f.text, kind)
                             and len(qt & set(self.aligner.terms_fn(f.text))) >= need
                             and (not counted or counts_noun(f.text, counted, self.aligner.terms_fn))]
                if not answering:
                    gaps.append(IntentGap(intent_id=sec.intent_id, kind="value_not_stated", aspect=asked))
            intents.append(IntentPlan(intent_id=sec.intent_id, section_id=sec.section_id, title=sec.title,
                                      order=sec.order, constraints=ks, claims=planned, gaps=gaps,
                                      needs_regeneration=sec.needs_regeneration))
        by_p6 = {}
        for ip in intents:
            for pc in ip.claims:
                by_p6.setdefault(pc.phase6_claim_id, pc.plan_claim_id)
        pairs = []
        for _, lst in (conflicts or {}).items():
            for a, b, _unit in lst:
                if a in by_p6 and b in by_p6:
                    pairs.append((by_p6[a], by_p6[b]))
        return ClaimPlan(plan_id=f"CP{self._n}", frame_id=answer.topic_id, answer_id=answer.answer_id,
                         intents=intents, conflicts=pairs)
