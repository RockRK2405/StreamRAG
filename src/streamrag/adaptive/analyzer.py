"""QueryComplexityAnalyzer (docs/retrieval/01_query_analysis.md).

Every signal is a measurable property of the query, the interpreted need (Phase 5/6 ``Intent`` + constraints) or
corpus *statistics* (IDF, term co-occurrence) - never of retrieved text. The complexity class is decided by
documented rules, recorded in ``reasons``:

  MULTI_HOP  a rare entity term (a name said with a capital letter mid-sentence, or an identifier) never co-occurs
             in any chunk with the rest of the question (entity_aspect_gap): the answer has to be chained through
             another statement. (Lower-case transcripts lose this cue; see docs/retrieval/07 §limitations.)
  COMPLEX    several needs in the utterance, a comparison, or >= 2 constraints / filters, or an explicit date together
             with a constraint.
  MODERATE   one constraint / metadata filter, a temporal cue, context taken from earlier turns, a vocabulary
             mismatch (query words unknown to the index), an ambiguous need, or a long question (> 8 content terms).
  SIMPLE     everything else: one need, a few content terms, nothing to filter, nothing to chain.
"""

from __future__ import annotations

import calendar
import re
from datetime import date

from streamrag.adaptive.catalog import MetadataCatalog
from streamrag.adaptive.lexicon import QUESTION_WORDS, RetrievalLexicon
from streamrag.adaptive.models import QueryAnalysis, QueryComplexity, TemporalSpec

_ID = re.compile(r"\b(?=[A-Za-z0-9-]*\d)(?=[A-Za-z0-9-]*[A-Za-z])[A-Za-z0-9]+(?:-[A-Za-z0-9]+)+\b|\b[A-Z]{2,6}\b|§\s*\d+")
_ISO = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")
_MONTHS = {m.lower(): i for i, m in enumerate(calendar.month_name) if m}
_MONTH_YEAR = re.compile(r"\b(" + "|".join(_MONTHS) + r")\s+(\d{4})\b", re.I)
_YEAR = re.compile(r"\b(19\d{2}|20\d{2})\b")
_COMMON_CAPS = {"I", "OK", "AM", "PM", "US", "TV"}


def reference_date(cfg_value: str | None) -> str:
    return cfg_value or date.today().isoformat()


class QueryComplexityAnalyzer:
    def __init__(self, service, catalog: MetadataCatalog, lexicon: RetrievalLexicon, claim_lexicon, cfg) -> None:
        self.service, self.catalog, self.lx, self.clx = service, catalog, lexicon, claim_lexicon
        self.cfg = cfg.adaptive_retrieval
        self.bm25, self.an = service.bundle.bm25, service.bundle.analyzer

    def terms(self, text: str) -> list[str]:
        return list(dict.fromkeys(self.an.tokens(text)))

    def temporal(self, text: str) -> TemporalSpec:
        ref = reference_date(self.cfg.reference_date)
        m = _ISO.search(text)
        if m:
            return TemporalSpec(kind="as_of", valid_at=m.group(0), cue=m.group(0), explicit=True)
        m = _MONTH_YEAR.search(text)
        if m:
            mo, y = _MONTHS[m.group(1).lower()], int(m.group(2))
            return TemporalSpec(kind="as_of", valid_at=f"{y:04d}-{mo:02d}-01", cue=m.group(0), explicit=True)
        m = _YEAR.search(text)
        if m:
            return TemporalSpec(kind="as_of", valid_at=f"{m.group(1)}-01-01", cue=m.group(0), explicit=True)
        cur = self.lx.has_phrase(text, self.lx.current)
        if cur:
            return TemporalSpec(kind="current", valid_at=ref, cue=cur, explicit=False)
        past = self.lx.has_phrase(text, self.lx.past)
        if past:
            return TemporalSpec(kind="past", cue=past)
        return TemporalSpec()

    @staticmethod
    def period_end(t: TemporalSpec) -> str | None:
        """End of the period an explicit month / year names (None for a single day)."""
        if t.kind != "as_of" or not t.cue:
            return None
        if _ISO.fullmatch(t.cue):
            return None
        m = _MONTH_YEAR.fullmatch(t.cue)
        if m:
            mo, y = _MONTHS[m.group(1).lower()], int(m.group(2))
            return f"{y:04d}-{mo:02d}-{calendar.monthrange(y, mo)[1]:02d}"
        return f"{t.cue}-12-31" if _YEAR.fullmatch(t.cue) else None

    def _user_stated(self, cand: dict[str, list[str]], text: str, cterms: set[str]) -> dict[str, list[str]]:
        """Keep a metadata value only if the user stated it as a constraint: inside a parsed constraint, or preceded
        by a constraint marker within three words. A document declaring a value can never create a filter by
        itself (docs/retrieval/03 §security)."""
        words = re.findall(r"[a-z0-9][a-z0-9-]*", text.lower())
        stems = [self.an.tokens(w) for w in words]
        out: dict[str, list[str]] = {}
        for f, vals in cand.items():
            keep = []
            for v in vals:
                vt = self.catalog.value_terms[(f, v)]
                if set(vt) <= cterms:
                    keep.append(v)
                    continue
                for i, st in enumerate(stems):
                    if st and st[0] == vt[0] and any(w in self.lx.constraint_markers for w in words[max(0, i - 3):i]):
                        keep.append(v)
                        break
            if keep:
                out[f] = keep
        return out

    def question_terms(self, text: str) -> set[str]:
        """Analyzed question words and the words of an asked-value phrase ("how high" -> high, "how long" -> long)."""
        out = {t for w in QUESTION_WORDS for t in self.an.tokens(w)}
        if self.clx is not None:
            words = re.findall(r"[a-z]+", text.lower())
            for _, p in self.clx.value_questions:                 # "how high", "what time" - not "fee"
                if len(p) >= 2 and any(tuple(words[i:i + len(p)]) == p for i in range(len(words))):
                    out |= {t for w in p for t in self.an.tokens(w)}
        return out

    def analyze(self, query_id: str, text: str, intent=None, constraints=(), n_intents: int = 1,
                cue_text: str = "") -> QueryAnalysis:
        reasons: list[str] = []
        terms = self.terms(text)
        qterms = self.question_terms(f"{text} {cue_text}")
        vocab = self.bm25.vocab
        exact = [m.group(0) for m in _ID.finditer(text) if m.group(0) not in _COMMON_CAPS]
        known_ids = [x for x in exact if all(t in vocab for t in self.an.tokens(x))]
        id_terms = {t for x in known_ids for t in self.an.tokens(x)}
        idf = {t: self.bm25.term_idf(t) for t in terms}
        temporal = self.temporal(text)
        tcue = set(self.an.tokens(temporal.cue)) if temporal.cue else set()
        content = [t for t in terms if t not in qterms and t not in tcue and not t.isdigit()] or \
            [t for t in terms if not t.isdigit()]
        oov = [t for t in content if idf[t] is None]
        rare = [t for t in content if idf[t] is not None and idf[t] >= self.cfg.rare_idf]
        ctexts = [c.text for c in constraints if getattr(c, "status", "active") == "active"]
        cterms = {t for c in ctexts for t in self.terms(c)}
        filters = self._user_stated(self.catalog.filters_for(set(terms) | cterms), text, cterms)
        value_kind = self.clx.asks_value(text) if self.clx is not None else None
        comparison = bool(self.lx.has_phrase(text, self.lx.comparison_cues)) or (
            intent is not None and intent.intent_type == "COMPARISON")
        context_dep = intent is not None and bool(intent.inherited_context)
        ambiguous = len(content) <= 1 or (intent is not None and bool(intent.unresolved_references))
        # entity_aspect_gap: an entity-like rare term (a name said with a capital letter mid-sentence, or an
        # identifier) that shares no chunk with the rest of the question (known content terms)
        caps = {t for w in re.findall(r"(?<!^)(?<![.?!]\s)\b[A-Z][a-z]+\b", text.strip()) for t in self.an.tokens(w)}
        known = [t for t in content if idf[t] is not None]
        bridge = []
        for t in rare:
            rows = self.bm25.rows_with(t)
            if not rows or not (t in caps or t in id_terms):
                continue
            aspect = [o for o in known if o != t and o not in cterms and o not in id_terms]
            if aspect and self.bm25.cooccurrence([t], aspect) == 0 and any(self.bm25.rows_with(o) for o in aspect):
                bridge.append(t)
        n_c = len(content)
        if bridge:
            complexity = QueryComplexity.MULTI_HOP
            reasons.append(f"entity_aspect_gap:{','.join(bridge)}")
        elif n_intents >= 2 or comparison or len(ctexts) + len(filters) >= 2 or (temporal.explicit and (ctexts or filters)):
            complexity = QueryComplexity.COMPLEX
            reasons += [r for r, ok in (("multiple_needs", n_intents >= 2), ("comparison", comparison),
                                        ("several_constraints", len(ctexts) + len(filters) >= 2),
                                        ("date_and_constraint", temporal.explicit and bool(ctexts or filters))) if ok]
        else:
            mod = [r for r, ok in (("constraint", bool(ctexts)), ("metadata_filter", bool(filters)),
                                   ("temporal", temporal.kind != "none"), ("context_dependent", context_dep),
                                   ("vocabulary_mismatch", bool(n_c) and len(oov) / n_c >= self.cfg.semantic_oov_ratio),
                                   ("ambiguous", ambiguous), ("long_question", n_c > 8)) if ok]
            complexity = QueryComplexity.MODERATE if mod else QueryComplexity.SIMPLE
            reasons += mod or ["single_need_no_constraints"]
        entities = list(intent.entities) if intent is not None and intent.entities else (exact + rare)
        return QueryAnalysis(
            query_id=query_id, text=text, complexity=complexity, reasons=reasons, terms=terms,
            question_terms=sorted(qterms & set(terms)), exact_ids=known_ids, rare_terms=rare, oov_terms=oov,
            entities=entities, constraint_texts=ctexts, metadata_filters=filters, temporal=temporal,
            value_kind=value_kind, n_intents=max(1, n_intents), context_dependent=context_dep, ambiguous=ambiguous,
            comparison=comparison, bridge_terms=bridge,
            signals={"n_terms": len(terms), "n_content_terms": n_c,
                     "oov_ratio": round(len(oov) / n_c, 3) if n_c else 0.0,
                     "max_idf": round(max((v for v in idf.values() if v is not None), default=0.0), 3),
                     "n_exact_ids": len(known_ids), "n_constraints": len(ctexts), "n_filters": len(filters),
                     "temporal": temporal.kind, "value_kind": value_kind})
