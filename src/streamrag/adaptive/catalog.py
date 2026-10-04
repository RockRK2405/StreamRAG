"""Corpus catalog for adaptive retrieval: document metadata, validity, versions, authority, aliases, titles.

Everything here is derived from the corpus, which is UNTRUSTED (docs/retrieval/03 §security):

* metadata can *narrow* a search (filters on whitelisted fields from trusted config, validity windows) and rank
  sources (authority weights come from trusted config, the document only says which status / type it has);
* a ``supersedes`` claim is honoured only when the superseded document agrees (its status is superseded / expired /
  archived, or its validity ends before the newer document starts) - a single document cannot demote another;
* acronym aliases mined from text ("Customer Service Desk (CSD)") are accepted only when the acronym matches the
  initials of the long form, and can only *add* bounded lexical expansion terms;
* nothing from the corpus selects a strategy, a budget, a k or a prompt.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field

_ALIAS = re.compile(r"((?:[A-Z][\w'-]*\s+(?:(?:of|for|and|the|on|in)\s+)?){1,7})\(([A-Z][A-Za-z0-9]{1,9})\)")
_SMALL = {"of", "for", "and", "the", "on", "in"}
_DEMOTED = {"superseded", "expired", "archived", "withdrawn", "replaced"}


@dataclass
class DocInfo:
    doc_id: str
    title: str
    meta: dict
    start: str | None = None
    end: str | None = None
    status: str | None = None
    content_sig: str = ""                                   # hash of the document's chunk texts


@dataclass
class MetadataCatalog:
    docs: dict[str, DocInfo]
    filter_values: dict[str, dict[str, set[str]]]          # field -> value -> doc ids
    value_terms: dict[tuple[str, str], tuple[str, ...]]     # (field, value) -> analyzed terms
    acronyms: dict[str, str]                                # analyzed acronym term -> long form ("csd" -> ...)
    long_forms: dict[tuple[str, ...], str]                  # analyzed long-form terms -> acronym
    title_terms: dict[str, list[str]]                       # doc id -> analyzed title terms
    superseded: dict[str, str] = field(default_factory=dict)   # doc id -> doc id that supersedes it (agreed)
    authority_cfg: dict[str, dict[str, float]] = field(default_factory=dict)

    @classmethod
    def build(cls, service, cfg, terms_fn) -> "MetadataCatalog":
        ac = cfg.adaptive_retrieval
        chunks = service.bundle.chunks
        titles, content = {}, {}
        for c in chunks:
            titles.setdefault(c.document_id, c.title or c.document_id)
            content.setdefault(c.document_id, hashlib.sha1())
            content[c.document_id].update(c.text.encode())
        docs: dict[str, DocInfo] = {}
        for d, meta in {**{d: {} for d in titles}, **service.doc_meta}.items():
            m = dict(meta)
            docs[d] = DocInfo(d, titles.get(d, d), m, _date(m.get("effective_date")), _date(m.get("valid_until")),
                              str(m["status"]).lower() if m.get("status") is not None else None,
                              content[d].hexdigest()[:12] if d in content else "")
        filt: dict[str, dict[str, set[str]]] = {}
        vterms: dict[tuple[str, str], tuple[str, ...]] = {}
        for d, info in docs.items():
            for f in ac.filter_fields:
                v = info.meta.get(f)
                if v is None:
                    continue
                for val in (x.strip().lower() for x in str(v).split(",")):
                    toks = tuple(terms_fn(val))
                    if val == "all" or len(val) < 3 or not toks or len(toks) > 3:
                        continue                                   # unusable as a filter value
                    filt.setdefault(f, {}).setdefault(val, set()).add(d)
                    vterms[(f, val)] = toks
        acr: dict[str, str] = {}
        longs: dict[tuple[str, ...], str] = {}
        for c in chunks:
            for long, short in _ALIAS.findall(c.text):
                words = long.split()
                while words and words[0].lower() in _SMALL | {"a", "an"}:
                    words = words[1:]
                initials = "".join(w[0] for w in words if w.lower() not in _SMALL).upper()
                st = terms_fn(short)
                if initials != short.upper() or len(st) != 1:
                    continue                                       # not an acronym definition
                long = " ".join(words)
                acr.setdefault(st[0], long)
                longs.setdefault(tuple(terms_fn(long)), short)
        sup = {}
        for d, info in docs.items():
            old = info.meta.get("supersedes")
            if old is None:
                continue
            for o in (x.strip() for x in str(old).split(",")):
                prev = docs.get(o)
                if prev is None or o == d:
                    continue
                agrees = (prev.status in _DEMOTED) or (prev.end is not None and info.start is not None
                                                       and prev.end < info.start)
                if agrees:
                    sup[o] = d
        return cls(docs, filt, vterms, acr, longs, {d: terms_fn(i.title) for d, i in docs.items()}, sup,
                   dict(ac.authority))

    # ------------------------------------------------------------------ filters
    def filters_for(self, terms: set[str]) -> dict[str, list[str]]:
        """Whitelisted metadata values whose analyzed words all occur in ``terms`` (a constraint said in the query)."""
        out: dict[str, list[str]] = {}
        for f, vals in self.filter_values.items():
            hit = sorted(v for v in vals if set(self.value_terms[(f, v)]) <= terms)
            if hit:
                out[f] = hit
        return out

    # ------------------------------------------------------------------ validity / authority
    def valid_at(self, doc_id: str, at: str | None, to: str | None = None) -> bool | None:
        """True / False when the document carries validity dates, None when it does not."""
        info = self.docs.get(doc_id)
        if info is None or at is None or (info.start is None and info.end is None):
            return None
        if info.start is not None and info.start > (to or at):
            return False
        if info.end is not None and info.end < at:
            return False
        return True

    def authority(self, doc_id: str) -> float:
        info = self.docs.get(doc_id)
        if info is None:
            return 1.0
        st = self.authority_cfg.get("status", {}).get(info.status or "", 1.0)
        dt = self.authority_cfg.get("doc_type", {}).get(str(info.meta.get("doc_type", "")).lower(), 1.0)
        if doc_id in self.superseded:
            st = min(st, self.authority_cfg.get("status", {}).get("superseded", 0.2))
        return round(st * dt, 4)

    def signature(self, doc_id: str) -> str:
        """Source version signature (cache validity): any change of content / version / status / validity changes it."""
        info = self.docs.get(doc_id)
        if info is None:
            return "missing"
        m = info.meta
        return "|".join([info.content_sig] + [str(m.get(k, "")) for k in ("version", "status", "effective_date",
                                                                           "valid_until", "supersedes")])

    def title_doc(self, phrase_terms: list[str]) -> str | None:
        """The single document whose title contains every analyzed term of a referenced phrase (or None)."""
        if len(phrase_terms) < 2:
            return None
        hits = [d for d in self.docs if set(phrase_terms) <= set(self.title_terms.get(d, []))]
        return hits[0] if len(hits) == 1 else None


def _date(v) -> str | None:
    if v is None:
        return None
    s = str(v)
    return s[:10] if re.match(r"^\d{4}-\d{2}-\d{2}", s) else None
