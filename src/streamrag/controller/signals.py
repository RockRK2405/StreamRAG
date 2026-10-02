"""Controller signals. Every signal is a plain, documented function of the transcript, the corpus vocabulary
and the ledger; formulas are listed in docs/streaming/03_retrieval_controller.md.

* anchor_strength  = max IDF of query terms present in the corpus vocabulary / IDF of a term in exactly one chunk
* semantic_stability = 0.35*complete + 0.20*min(1, content/3) + 0.20*has_anchor + 0.15*request_form + 0.10*persistence
* retrieval_worthiness = base(act) + 0.25*anchor_strength + 0.10*min(anchors,3)/3 + 0.10*explicit_question
* novelty = 1 - max term-Jaccard(query, any earlier session query)
"""

from __future__ import annotations

import math
import re

from streamrag.config.settings import ControllerConfig
from streamrag.controller.acts import ActResult
from streamrag.controller.lexicon import Lexicon, tokenize
from streamrag.controller.query_builder import BuiltQuery
from streamrag.ledger.ledger import QueryLedger, jaccard
from streamrag.retrieval.bm25 import BM25Index
from streamrag.retrieval.text import Analyzer

ACT_BASE = {"INFO_REQUEST": 0.55, "UNKNOWN": 0.2}
_END_PUNCT = re.compile(r"[.?!]\s*$")


class SignalComputer:
    def __init__(self, cfg: ControllerConfig, lex: Lexicon, analyzer: Analyzer, bm25: BM25Index) -> None:
        self.cfg, self.lex, self.analyzer, self.bm25 = cfg, lex, analyzer, bm25
        n = max(bm25.n_docs, 1)
        self.anchor_norm = math.log1p((n - 1 + 0.5) / 1.5) or 1.0   # IDF of a term found in exactly one chunk

    def terms(self, text: str) -> list[str]:
        return list(dict.fromkeys(self.analyzer.tokens(text)))

    def anchors(self, terms: list[str]) -> list[tuple[str, float]]:
        out = []
        for t in terms:
            idf = self.bm25.term_idf(t)
            if idf is not None and idf >= self.cfg.anchor_idf_floor:
                out.append((t, idf))
        return out

    def is_anchor_word(self, word: str) -> bool:
        return bool(self.anchors(self.terms(word)))

    def compute(self, transcript: str, query: BuiltQuery, act: ActResult, prev_terms: list[str], quiet: bool,
                final: bool, ledger: QueryLedger) -> dict:
        terms = self.terms(query.text)
        anchors = self.anchors(terms)
        toks = tokenize(transcript)
        last = toks[-1].lower if toks else ""
        boundary = bool(_END_PUNCT.search(transcript))
        trailing_comma = transcript.rstrip().endswith(",")
        dangling = (not boundary) and (last in self.lex.dangling or trailing_comma) and not final
        anchor_strength = min(1.0, max((idf for _, idf in anchors), default=0.0) / self.anchor_norm)
        request_form = act.act == "INFO_REQUEST"
        persistence = 1.0 if (quiet or boundary or final) else (jaccard(set(terms), set(prev_terms)) if prev_terms else 0.0)
        stability = (0.35 * (0.0 if dangling else 1.0) + 0.20 * min(1.0, len(terms) / 3.0)
                     + 0.20 * (1.0 if anchors else 0.0) + 0.15 * (1.0 if request_form else 0.5) + 0.10 * persistence)
        explicit_q = "question_mark" in act.cues or "wh_word" in act.cues
        worthiness = (ACT_BASE.get(act.act, 0.0) + 0.25 * anchor_strength
                      + 0.10 * min(len(anchors), 3) / 3.0 + (0.10 if explicit_q else 0.0))
        best, sim = ledger.most_similar(terms)
        return {
            "act": act.act, "act_confidence": round(act.confidence, 3), "act_cues": ",".join(act.cues),
            "content_terms": len(terms), "anchors": len(anchors),
            "anchor_terms": ",".join(t for t, _ in anchors[:8]),
            "anchor_strength": round(anchor_strength, 3), "dangling": dangling, "boundary": boundary,
            "persistence": round(persistence, 3), "quiet": quiet,
            "semantic_stability": round(stability, 3), "retrieval_worthiness": round(min(1.0, worthiness), 3),
            "novelty": round(1.0 - sim, 3), "most_similar_query": best.query_id if best else None,
            "_terms": terms,
        }
