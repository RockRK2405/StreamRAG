"""QueryRewriteEngine (docs/retrieval/02_query_rewriting.md).

Four forms, one need:
  normalized   Unicode NFKC, whitespace collapsed, trailing punctuation removed
  contextual   the Phase 5/6 query of the need (anaphora resolved, inherited context, active constraints); when no
               interpreted need is given: the normalized text
  expanded     contextual + bounded additive expansions: acronym <-> long form mined from the corpus and validated
               (catalog), then the configured synonym groups for query words unknown to the index first
  retrieval    lexical_query = expanded (exact identifiers kept verbatim); dense_query = contextual (natural
               language; expansions add noise to an embedding)

Invariant (tested): every content term of ``contextual`` occurs in ``lexical_query`` and ``dense_query`` - a rewrite
never removes or replaces what the user asked about, and never touches the intent. Entities removed by a correction
(``dropped_entities``) are not re-introduced by expansions.
"""

from __future__ import annotations

import re
import unicodedata

from streamrag.adaptive.catalog import MetadataCatalog
from streamrag.adaptive.lexicon import RetrievalLexicon
from streamrag.adaptive.models import Expansion, RewrittenQuery


class QueryRewriteEngine:
    def __init__(self, catalog: MetadataCatalog, lexicon: RetrievalLexicon, terms_fn, vocab, cfg) -> None:
        self.catalog, self.lx, self.terms_fn, self.vocab = catalog, lexicon, terms_fn, vocab
        self.cfg = cfg.adaptive_retrieval

    @staticmethod
    def normalize(text: str) -> str:
        t = unicodedata.normalize("NFKC", text)
        t = re.sub(r"\s+", " ", t).strip()
        return t.rstrip("?.!;, ").strip() or t

    def expansions(self, text: str, dropped: set[str]) -> list[Expansion]:
        if not self.cfg.expansion or self.cfg.max_expansions == 0:
            return []
        terms = self.terms_fn(text)
        tset = set(terms)
        out: list[Expansion] = []
        seen = set(tset) | dropped
        for t in terms:                                         # acronym -> long form
            long = self.catalog.acronyms.get(t)
            if long and not set(self.terms_fn(long)) <= seen:
                out.append(Expansion(term=long, source="acronym", for_term=t, reason="acronym_defined_in_corpus"))
                seen |= set(self.terms_fn(long))
        for lt, short in self.catalog.long_forms.items():       # long form -> acronym
            st = self.terms_fn(short)
            if lt and set(lt) <= tset and st and st[0] not in seen:
                out.append(Expansion(term=short, source="alias", for_term=" ".join(lt), reason="long_form_in_query"))
                seen |= set(st)
        # synonyms: words unknown to the index first (the lexical search cannot match them), then known words
        order = sorted(terms, key=lambda t: (t in self.vocab, terms.index(t)))
        for t in order:
            for group in self.lx.synonyms:
                gterms = {g: self.terms_fn(g) for g in group}
                if not any(v == [t] for v in gterms.values()):
                    continue
                for g, gt in gterms.items():
                    if gt and not set(gt) <= seen and all(x in self.vocab for x in gt) and not set(gt) & dropped:
                        out.append(Expansion(term=g, source="synonym", for_term=t,
                                             reason="query_word_unknown_to_index" if t not in self.vocab
                                             else "synonym_group"))
                        seen |= set(gt)
                        break
        return out[: self.cfg.max_expansions]

    def rewrite(self, text: str, intent_id: str | None = None, contextual: str | None = None,
                dropped_entities: list[str] | None = None) -> RewrittenQuery:
        norm = self.normalize(text)
        ctx = self.normalize(contextual) if contextual else norm
        dropped = {t for e in (dropped_entities or []) for t in self.terms_fn(e)}
        exps = self.expansions(ctx, dropped)
        expanded = " ".join([ctx] + [e.term for e in exps])
        return RewrittenQuery(intent_id=intent_id, original=text, normalized=norm, contextual=ctx, expanded=expanded,
                              lexical_query=expanded, dense_query=ctx, expansions=exps,
                              dropped_entities=sorted(dropped))

    def keeps_need(self, rq: RewrittenQuery) -> bool:
        """The rewrite invariant: no content term of the contextual query is lost."""
        base = set(self.terms_fn(rq.contextual))
        return base <= set(self.terms_fn(rq.lexical_query)) and base <= set(self.terms_fn(rq.dense_query))
