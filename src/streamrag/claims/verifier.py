"""ClaimVerifier: is a claim supported by the evidence, by which evidence, and is anything contradicting it?
(Phase 7; docs/answer/04)

Answers the brief's five questions per claim:

1. supported?            >= 1 evidence item with STRONG or MODERATE alignment (entailment + numbers present)
2. which evidence?       the supporting evidence ids (citations are built from these, not from the generator's labels)
3. entailed?             an entailing premise exists (similarity alone - WEAK - never counts)
4. sufficient?           entailed and every number / date of the claim occurs in the entailing premise
5. contradicted?         any CONTRADICTORY alignment in the evidence pool (all evidence of the section, not only
                         the cited items)

Labels the generator cited that are not in the evidence set are invalid (L0) and ignored. Support from evidence the
generator did not cite is accepted and reported as ``support_outside_citations`` (citation repair, spec §16.4).

A statement that is not supported as a whole is decomposed (claims/decomposer.py) and each atom is verified:

  all atoms supported            -> SUPPORTED (support composed from several premises)
  some atoms supported           -> PARTIALLY_SUPPORTED (the policy keeps only the supported atoms)
  no atom supported, one contradicted -> CONTRADICTED
  otherwise                      -> UNSUPPORTED

Supported + contradicted by different evidence -> CONTRADICTED with both lists filled: an evidence conflict that the
answer must present as such (never resolved arbitrarily).
"""

from __future__ import annotations

from streamrag.claims.aligner import ClaimEvidenceAligner
from streamrag.claims.decomposer import ClaimDecomposer
from streamrag.claims.models import SUPPORTING, ClaimVerification


class ClaimVerifier:
    def __init__(self, aligner: ClaimEvidenceAligner, decomposer: ClaimDecomposer) -> None:
        self.aligner, self.decomposer = aligner, decomposer
        self.verified = 0

    def verify(self, claim_id: str, text: str, cited: list[str], pool: dict[str, str],
               decompose: bool = True) -> ClaimVerification:
        self.verified += 1
        invalid = [c for c in cited if c not in pool]
        cited_ok = [c for c in cited if c in pool]
        v = self._single(claim_id, text, cited_ok, pool)
        v = v.model_copy(update={"invalid_labels": invalid})
        if v.status in ("SUPPORTED", "CONTRADICTED") or not decompose:
            return v
        atoms = self.decomposer.decompose(text)
        if len(atoms) < 2:
            return v
        av = [self._single(f"{claim_id}.{k + 1}", a, cited_ok, pool) for k, a in enumerate(atoms)]
        sup = [a for a in av if a.supported]
        contra = [a for a in av if a.status == "CONTRADICTED" and not a.supported]
        if sup and len(sup) == len(av):
            status, reason = "SUPPORTED", "all_atoms_supported"
        elif sup:
            status, reason = "PARTIALLY_SUPPORTED", f"{len(sup)}_of_{len(av)}_atoms_supported"
        elif contra:
            status, reason = "CONTRADICTED", "atom_contradicted"
        else:
            status, reason = "UNSUPPORTED", "no_atom_supported"
        supporting = list(dict.fromkeys(e for a in sup for e in a.supporting_evidence))
        return v.model_copy(update={
            "status": status, "supported": status == "SUPPORTED", "supporting_evidence": supporting,
            "entailed": status == "SUPPORTED", "sufficient": status == "SUPPORTED",
            "contradicting_evidence": list(dict.fromkeys(e for a in av for e in a.contradicting_evidence)),
            "support_outside_citations": list(dict.fromkeys(e for a in sup for e in a.support_outside_citations)),
            "atoms": av, "atom_texts": atoms, "reasons": v.reasons + ["decomposed", reason]})

    def _single(self, claim_id: str, text: str, cited: list[str], pool: dict[str, str]) -> ClaimVerification:
        al = self.aligner.align(claim_id, text, list(pool.items()))
        supporting = [a.evidence_id for a in al if a.strength in SUPPORTING]
        supporting.sort(key=lambda e: (e not in cited, [a.strength for a in al if a.evidence_id == e][0] != "STRONG"))
        contradicting = [a.evidence_id for a in al if a.strength == "CONTRADICTORY"]
        reasons = []
        if supporting and contradicting:
            status, reasons = "CONTRADICTED", ["evidence_conflict"]
        elif supporting:
            status = "SUPPORTED"
            reasons = ["entailed"] + (["citation_repaired"] if not set(supporting) & set(cited) and cited else [])
        elif contradicting:
            status, reasons = "CONTRADICTED", ["contradicted_by_evidence"]
        else:
            weak = [a for a in al if a.strength == "WEAK"]
            status = "UNSUPPORTED"
            reasons = ["similar_but_not_entailed"] if weak else ["no_related_evidence"]
            if any(a.basis == "numbers_not_in_evidence" for a in weak):
                reasons.append("numbers_not_in_evidence")
        return ClaimVerification(
            claim_id=claim_id, status=status, supported=bool(supporting), supporting_evidence=supporting,
            entailed=bool(supporting), sufficient=bool(supporting), contradicting_evidence=contradicting,
            cited_evidence=cited, support_outside_citations=[e for e in supporting if e not in cited],
            alignments=al, reasons=reasons, verifier=self.aligner.name)
