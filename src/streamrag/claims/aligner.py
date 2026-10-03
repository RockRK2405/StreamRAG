"""ClaimEvidenceAligner: how strongly does each evidence item support a claim? (Phase 7; docs/answer/04)

Support is decided by **entailment**, never by similarity (brief §10): a chunk that is about the question but does
not state the asked fact is at most WEAK.

For every candidate evidence item the premises tried are: each sentence, each window of two adjacent sentences, and
the whole chunk. Pairs are only scored for sentences sharing at least one content term or number with the claim
(plus the whole chunk), which keeps the cost proportional to related text.

  STRONG         a single sentence entails the claim, and every number in the claim occurs in that sentence
  MODERATE       only a two-sentence window or the whole chunk entails it (numbers likewise)
  CONTRADICTORY  no premise entails it and a single sentence that shares >= half of the claim's content terms
                 contradicts it (entailment models label "same topic, different statement" pairs as contradiction;
                 requiring the same proposition - shared content - and a sentence premise removes that artifact)
  WEAK           no entailment, no contradiction, but >= half of the claim's content terms occur (similar, not support)
  NONE           otherwise

Entailment comes from the NLI cross-encoder (``mode="nli"``). In ``mode="rules"`` (no model; ablation and model-less
runs) entailment is approximated by: all content terms and numbers of the claim occur in the premise and the
negation polarity matches; contradiction by: same content but different numbers or opposite polarity. The rules
mode is strictly weaker (it accepts no paraphrase) and is reported separately.
"""

from __future__ import annotations

from collections.abc import Callable

from streamrag.claims.graph import sentences
from streamrag.claims.models import EvidenceAlignment
from streamrag.claims.textcheck import instruction_like, negated, numbers


class ClaimEvidenceAligner:
    def __init__(self, terms_fn: Callable[[str], list[str]], nli=None, mode: str = "nli") -> None:
        if mode == "nli" and nli is None:
            raise ValueError("mode='nli' needs an NliModel (or use mode='rules')")
        self.terms_fn, self.nli, self.mode = terms_fn, nli, mode
        self.pairs_scored = 0

    @property
    def name(self) -> str:
        return f"nli:{self.nli.name}+rules" if self.mode == "nli" else "rules"

    def _premises(self, text: str, claim_terms: set[str], claim_nums: set[str]) -> list[tuple[str, int, int, str]]:
        # instruction-like sentences (prompt-injection markers) are never premises: an LLM that obeys an injected
        # instruction finds no support for what it was told to say
        sents = [(s, e) for s, e in sentences(text) if not instruction_like(text[s:e])]
        out = []
        related = []
        for k, (s, e) in enumerate(sents):
            st = text[s:e]
            if claim_terms & set(self.terms_fn(st)) or claim_nums & numbers(st):
                out.append((st, s, e, "sentence"))
                related.append(k)
        for k in sorted(set(related) | {r - 1 for r in related if r > 0}):
            if k + 1 < len(sents):
                s, e = sents[k][0], sents[k + 1][1]
                out.append((text[s:e], s, e, "window"))
        if sents and (len(sents) > 2 or not out):
            masked = text if len(sents) == len(sentences(text)) else " ".join(text[s:e] for s, e in sents)
            out.append((masked, 0, len(text), "chunk"))
        return out

    def align(self, claim_id: str, claim_text: str, evidence: list[tuple[str, str]]) -> list[EvidenceAlignment]:
        """evidence: [(evidence_id, text)] -> one alignment per evidence item."""
        ct = set(self.terms_fn(claim_text))
        cn = numbers(claim_text)
        per_ev = [(eid, self._premises(text, ct, cn)) for eid, text in evidence]
        verdicts: dict[tuple[str, str], dict] = {}
        if self.mode == "nli":
            pairs = [(p[0], claim_text) for _, prem in per_ev for p in prem]
            for pair, r in zip(pairs, self.nli.predict(pairs)):
                verdicts[pair] = r
            self.pairs_scored += len(pairs)
        out = []
        for eid, prem in per_ev:
            best = None
            for p_text, s, e, kind in prem:
                v = verdicts.get((p_text, claim_text)) if self.mode == "nli" else self._rules(p_text, claim_text,
                                                                                            ct, cn)
                nums_ok = cn <= numbers(p_text)
                if v["label"] == "entailment" and nums_ok:
                    strength = "STRONG" if kind == "sentence" else "MODERATE"
                    rank = 3 if kind == "sentence" else 2
                elif v["label"] == "contradiction" and kind == "sentence" and \
                        (not ct or len(ct & set(self.terms_fn(p_text))) / len(ct) >= 0.5):
                    strength, rank = "CONTRADICTORY", 1
                else:
                    strength, rank = None, 0
                if strength and (best is None or rank > best[0]):
                    best = (rank, strength, kind, s, e, v, nums_ok)
            if best is not None and best[1] != "CONTRADICTORY":
                _, strength, kind, s, e, v, _ = best
                out.append(EvidenceAlignment(claim_id=claim_id, evidence_id=eid, strength=strength,
                                             basis=f"{'entailed' if self.mode == 'nli' else 'rules_entailed'}_by_{kind}",
                                             premise=kind, premise_span=(s, e), signals=_sig(v)))
                continue
            if best is not None:                           # contradiction found, no entailment anywhere
                _, strength, kind, s, e, v, _ = best
                out.append(EvidenceAlignment(claim_id=claim_id, evidence_id=eid, strength="CONTRADICTORY",
                                             basis=f"{'contradicted' if self.mode == 'nli' else 'rules_contradicted'}"
                                                   f"_by_{kind}", premise=kind, premise_span=(s, e), signals=_sig(v)))
                continue
            text = dict(evidence)[eid]
            overlap = len(ct & set(self.terms_fn(text))) / len(ct) if ct else 0.0
            unmatched_numbers = bool(cn) and not cn <= numbers(text)
            strength = "WEAK" if overlap >= 0.5 else "NONE"
            basis = "similar_not_entailed" if strength == "WEAK" else "unrelated"
            if unmatched_numbers and overlap >= 0.5:
                basis = "numbers_not_in_evidence"
            out.append(EvidenceAlignment(claim_id=claim_id, evidence_id=eid, strength=strength, basis=basis,
                                         signals={"term_overlap": round(overlap, 4)}))
        return out

    def _rules(self, premise: str, claim: str, ct: set[str], cn: set[str]) -> dict:
        pt = set(self.terms_fn(premise))
        pn = numbers(premise)
        same_polarity = negated(premise) == negated(claim)
        if ct and ct <= pt and cn <= pn and same_polarity:
            return {"label": "entailment"}
        if ct and len(ct & pt) / len(ct) >= 0.8 and ((cn and pn and not cn <= pn) or not same_polarity):
            return {"label": "contradiction"}
        return {"label": "neutral"}


def _sig(v: dict) -> dict[str, float]:
    return {k: float(x) for k, x in v.items() if k in ("entailment", "contradiction", "neutral")}
