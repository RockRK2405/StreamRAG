"""ClaimRepairer: unsupported claim -> supported form, qualified statement or rejection (Phase 7; docs/answer/07).

Repairs never create facts:

* ``keep_atoms``      the verified atoms of a partially supported statement, each its own claim
                      ("X requires A and B and takes 30 days" -> "X requires A.", "X requires B."; 30 days dropped)
* ``restore_facts``   the planned, evidence-derived facts a distorted sentence claimed to express, verbatim
* ``llm_rewrite``     (optional, ``generation.llm_repair``) one LLM call asking for a rewrite limited to the given
                      evidence sentences; the result is verified like any other claim and discarded if unsupported
* ``qualify``         a deterministic uncertainty sentence ("... is not established by the retrieved documents"),
                      typed ``uncertainty`` and never rendered as a fact
* ``conflict``        two verbatim evidence sentences that disagree, each cited by its own evidence (no resolution)

The repaired texts are returned to the engine, which verifies them again before they can enter the answer.
"""

from __future__ import annotations

from streamrag.claims.graph import sentences
from streamrag.claims.models import SUPPORTING, ClaimVerification, IntentGap

_GAP_TEMPLATES = {
    "no_evidence": "The retrieved documents do not contain an answer to “{aspect}”.",
    "value_not_stated": "The retrieved documents do not state the value asked for in “{aspect}”.",
    "constraint_not_covered": "The retrieved documents do not say how this applies {aspect}.",
    "conflict": "The retrieved documents disagree about {aspect}.",
    "unsupported_removed": "Some details about {aspect} could not be verified in the retrieved documents and were "
                           "left out.",
}


def gap_sentence(g: IntentGap) -> str:
    aspect = " ".join(g.aspect.split()).rstrip(".?")
    if g.kind == "constraint_not_covered" and aspect and not aspect.lower().startswith(("for ", "to ", "when ",
                                                                                      "during ", "in ", "at ")):
        aspect = f"to {aspect}"
    return _GAP_TEMPLATES[g.kind].format(aspect=aspect or "this point")


def premise_text(v: ClaimVerification, evidence_id: str, evidence_text: str) -> str:
    """The sentence (or window) of ``evidence_id`` that the verification aligned with the claim."""
    for a in v.alignments + [x for atom in v.atoms for x in atom.alignments]:
        if a.evidence_id == evidence_id and a.premise_span:
            s, e = a.premise_span
            return evidence_text[s:e]
    sents = sentences(evidence_text)
    return evidence_text[sents[0][0]:sents[0][1]] if sents else evidence_text


class ClaimRepairer:
    def __init__(self, llm=None) -> None:
        self.llm = llm
        self.llm_calls = 0

    @staticmethod
    def keep_atoms(v: ClaimVerification) -> list[tuple[str, ClaimVerification]]:
        return [(t, a) for t, a in zip(v.atom_texts, v.atoms) if a.supported and a.status == "SUPPORTED"]

    @staticmethod
    def conflict_pair(v: ClaimVerification, evidence_text: dict[str, str]) -> list[tuple[str, str]]:
        """[(verbatim sentence, evidence id)] for one supporting and one contradicting evidence item."""
        sup = next((e for e in v.supporting_evidence), None)
        con = next((e for e in v.contradicting_evidence if e != sup), None)
        if sup is None or con is None:
            return []
        out = [(premise_text(v, sup, evidence_text[sup]), sup)]
        for a in v.alignments:
            if a.evidence_id == con and a.premise_span:
                s, e = a.premise_span
                out.append((evidence_text[con][s:e], con))
                break
        return out if len(out) == 2 else []

    def llm_rewrite(self, claim: str, v: ClaimVerification, evidence_text: dict[str, str]) -> str | None:
        """One bounded LLM call: rewrite the claim using only the closest evidence sentences (or nothing)."""
        if self.llm is None:
            return None
        from streamrag.generation import prompts
        ev = []
        for a in sorted(v.alignments, key=lambda a: a.strength not in SUPPORTING + ("WEAK",))[:3]:
            if a.evidence_id in evidence_text:
                ev.append((a.evidence_id, premise_text(v, a.evidence_id, evidence_text[a.evidence_id])))
        msgs = [{"role": "system", "content": prompts.SYSTEM_FACTS},
                {"role": "user", "content": "Rewrite the STATEMENT so that it says only what the EVIDENCE states. "
                                            "If the evidence does not support any part of it, return no sentences. "
                                            'Use section id "R".\nSTATEMENT: ' + prompts.quote(claim, 400) +
                                            f"\nEVIDENCE {prompts.OPEN}\n" +
                                            "\n".join(f"[{eid}] {prompts.quote(t, 600)}" for eid, t in ev) +
                                            f"\n{prompts.CLOSE}"}]
        r = self.llm.complete(msgs, prompts.SCHEMA)
        self.llm_calls += 1
        if not r.ok:
            return None
        try:
            from streamrag.generation.generator import parse_output
            out = parse_output(r.text)
        except Exception:                                   # noqa: BLE001 - any malformed output: no repair
            return None
        texts = [x.text.strip() for s in out.sections for x in s.sentences if x.text.strip()]
        return " ".join(texts) or None
