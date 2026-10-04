"""The common measuring instrument applied to every system's final answer (Phase 10).

Given the answer's claims (sentence text + the evidence ids it cites) and the evidence the system handed to its
answer stage, it judges each claim with the Phase 7 claim verifier (entailment model nli-deberta-v3-xsmall + rules,
alignment against *every* evidence item) and derives claim, citation and hallucination inputs. Every system - naive
RAG with its own citation labels, the Phase 7 grounded answers, the runtime's committed answers - is measured with
this same instrument and the same evidence texts (the index chunks).

Caveat (reported with every verifier-judged number): the proposed system uses the same verifier to filter its own
claims, so its verifier-judged support is favoured by construction; model-free metrics (claim coverage, answer
correctness, hallucinated values) carry no such bias.
"""

from __future__ import annotations

import re

from streamrag.bench.grounded import ABSTAIN
from streamrag.claims.graph import sentences
from streamrag.claims.textcheck import strip_markers

_VALUE = re.compile(r"\d")


class AnswerInstrument:
    def __init__(self, verifier, chunks: dict, terms_fn) -> None:
        self.verifier, self.chunks, self.terms_fn = verifier, chunks, terms_fn

    def text_of(self, chunk_id: str) -> str:
        c = self.chunks.get(chunk_id)
        return c.text if c is not None else ""

    def factual(self, text: str) -> bool:
        if ABSTAIN.search(text):
            return False
        return bool(_VALUE.search(text)) or len(set(self.terms_fn(text))) >= 3

    def judge(self, claims: list[tuple[str, list[str]]], evidence_ids: list[str]) -> list[dict]:
        """claims: [(text, cited chunk ids)] -> per claim: text, factual, verdict, cited, citations[...]"""
        handed = list(dict.fromkeys(evidence_ids))
        pool_ids = list(dict.fromkeys(handed + [c for _, cs in claims for c in cs if c in self.chunks]))
        pool = {i: self.text_of(i) for i in pool_ids}
        out = []
        verdicts = []
        for k, (text, cited) in enumerate(claims):
            ok_cited = [c for c in cited if c in pool]
            v = self.verifier.verify(f"eval{k}", text, ok_cited, pool) if pool else None
            sup = set(v.supporting_evidence) if v is not None else set()
            verdict = "SUPPORTED" if v is not None and v.supported else (v.status if v is not None else "UNSUPPORTED")
            verdicts.append((sup, verdict))
            out.append({"text": text, "factual": self.factual(text), "verdict": verdict,
                        "cited": bool(cited), "cited_ids": list(cited), "supporting": sorted(sup)})
        for k, cl in enumerate(out):
            cits = []
            for c in cl["cited_ids"]:
                own = c in verdicts[k][0]
                other = any(c in verdicts[j][0] for j in range(len(out)) if j != k)
                cits.append({"key": (self.chunks[c].citation if c in self.chunks else c), "chunk_id": c,
                             "valid": c in handed, "supports_own": own, "supports_other": other and not own})
            cl["citations"] = cits
            cl["cited_support"] = bool(set(cl["cited_ids"]) & verdicts[k][0])
        return out

    @staticmethod
    def split_free(cand_sentences, labels: dict[str, str]) -> list[tuple[str, list[str]]]:
        """Free-form generator output (naive baselines): one claim per sentence; inline markers / labels -> ids."""
        out = []
        for s0 in cand_sentences:
            text, markers = strip_markers(s0.text)
            labs = list(dict.fromkeys(list(s0.labels) + markers))
            for a, b in sentences(text):
                out.append((text[a:b].strip(), [labels[x] for x in labs if x in labels]))
        return [(t, c) for t, c in out if t]

    @staticmethod
    def split_grounded(ga) -> list[tuple[str, list[str]]]:
        """Phase 7 GroundedAnswer: its fact / conflict claims with the evidence of their citations."""
        if ga is None:
            return []
        by_id = {c.citation_id: c for c in ga.citations.citations}
        out = []
        for c in ga.claims:
            if c.kind not in ("fact", "conflict"):
                continue
            ids = [by_id[x].evidence_id for x in c.citation_ids if x in by_id and by_id[x].status == "valid"]
            out.append((c.text, list(dict.fromkeys(ids))))
        return out
