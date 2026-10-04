"""Cache-aware retrieval (docs/retrieval/08_cache_strategy.md).

Three caches, one validity rule: a cached item is reused only if everything it depended on is unchanged.

  AdaptiveQueryCache    key = sha1(analyzed terms of the contextual query | filters | validity date | claim-driven
                        flag). The entry records its validity signature: entity terms, constraint terms, the source
                        signature (content hash + version / status / validity metadata) of every evidence document,
                        the date it was valid for. The index hash is deliberately *not* in the key: an index update
                        invalidates only entries whose documents changed (fine-grained), checked on every get.
  session evidence      Phase 6 EvidenceStore (usable = ACTIVE / RETAINED for an active need) - read through the
                        ``SessionView`` the integration passes in; never mutated here
  ValidatedClaimCache   claims verified by the Phase 7 verifier in this session: claim terms -> supporting evidence
                        ids; reused as evidence only while those ids are still usable

Invalidation (each logged with a reason; RETRIEVAL_INVALIDATED):
  entity_changed          entries whose entity terms include a term a correction / entity change removed
  constraint_changed      entries whose constraint terms include a retracted / replaced constraint
  source_version_changed  an evidence document's version / status / validity signature differs (checked on get)
  temporal_validity       an evidence document is no longer valid at the entry's date (checked on get)
  corpus_changed          explicit full flush (``invalidate_corpus``: a new corpus snapshot replaces the old one)
  stale_evidence          an evidence id is no longer usable in the session (Phase 6 lifecycle)
"""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from dataclasses import dataclass, field

from streamrag.models.base import canonical_dumps


@dataclass
class CacheEntry:
    key: str
    query: str
    intent_id: str | None
    entity_terms: set[str]
    constraint_terms: set[str]
    valid_at: str | None
    doc_signatures: dict[str, str]
    evidence: list                               # Evidence objects (final evidence of the run)
    status: str                                  # sufficiency status when cached
    created_ms: float
    hits: int = 0


@dataclass
class ValidatedClaim:
    text: str
    terms: set[str]
    evidence_ids: list[str]
    verdict: str


@dataclass
class AdaptiveQueryCache:
    index_hash: str
    max_entries: int = 256
    enabled: bool = True
    entries: OrderedDict = field(default_factory=OrderedDict)
    stats: dict = field(default_factory=lambda: {"hits": 0, "misses": 0, "puts": 0, "invalidated": 0,
                                                 "evicted": 0})
    log: list[dict] = field(default_factory=list)

    def key(self, terms: list[str], filters: dict | None, valid_at: str | None, claim_driven: bool) -> str:
        raw = canonical_dumps({"t": sorted(set(terms)), "f": filters, "v": valid_at, "c": claim_driven})
        return hashlib.sha1(raw.encode()).hexdigest()[:16]

    def get(self, key: str, catalog, usable=None, now_ms: float = 0.0) -> tuple[CacheEntry | None, dict | None]:
        """(entry, None) on a valid hit; (None, invalidation) when the entry existed but is no longer valid."""
        if not self.enabled:
            return None, None
        e = self.entries.get(key)
        if e is None:
            self.stats["misses"] += 1
            return None, None
        reason = None
        for d, sig in e.doc_signatures.items():
            if catalog.signature(d) != sig:
                reason = f"source_version_changed:{d}"
                break
            if e.valid_at is not None and catalog.valid_at(d, e.valid_at) is False:
                reason = f"temporal_validity:{d}"
                break
        if reason is None and usable is not None:
            bad = [x.evidence_id for x in e.evidence if not usable(x.evidence_id)]
            if bad:
                reason = f"stale_evidence:{','.join(bad[:3])}"
        if reason is not None:
            inv = self._drop([key], reason.split(":", 1)[0], now_ms, detail=reason)
            self.stats["misses"] += 1
            return None, inv
        e.hits += 1
        self.entries.move_to_end(key)
        self.stats["hits"] += 1
        return e, None

    def put(self, entry: CacheEntry) -> None:
        if not self.enabled:
            return
        self.entries[entry.key] = entry
        self.entries.move_to_end(entry.key)
        self.stats["puts"] += 1
        while len(self.entries) > self.max_entries:
            self.entries.popitem(last=False)
            self.stats["evicted"] += 1

    def _drop(self, keys: list[str], reason: str, now_ms: float, detail: str = "") -> dict:
        for k in keys:
            self.entries.pop(k, None)
        self.stats["invalidated"] += len(keys)
        rec = {"reason": reason, "keys": keys, "n": len(keys), "detail": detail, "at_ms": now_ms}
        self.log.append(rec)
        return rec

    def invalidate_entities(self, terms: set[str], now_ms: float = 0.0) -> dict | None:
        keys = [k for k, e in self.entries.items() if e.entity_terms & terms]
        return self._drop(keys, "entity_changed", now_ms, detail=",".join(sorted(terms))) if keys else None

    def invalidate_constraints(self, terms: set[str], now_ms: float = 0.0) -> dict | None:
        keys = [k for k, e in self.entries.items() if e.constraint_terms & terms]
        return self._drop(keys, "constraint_changed", now_ms, detail=",".join(sorted(terms))) if keys else None

    def invalidate_evidence(self, ids: set[str], now_ms: float = 0.0) -> dict | None:
        keys = [k for k, e in self.entries.items() if {x.evidence_id for x in e.evidence} & ids]
        return self._drop(keys, "stale_evidence", now_ms, detail=",".join(sorted(ids)[:5])) if keys else None

    def invalidate_corpus(self, new_hash: str, now_ms: float = 0.0) -> dict | None:
        if new_hash == self.index_hash:
            return None
        self.index_hash = new_hash
        return self._drop(list(self.entries), "corpus_changed", now_ms, detail=new_hash)


@dataclass
class ValidatedClaimCache:
    claims: list[ValidatedClaim] = field(default_factory=list)

    def add(self, text: str, terms: set[str], evidence_ids: list[str], verdict: str) -> None:
        if verdict != "SUPPORTED" or not evidence_ids:
            return
        if any(c.text == text and c.evidence_ids == evidence_ids for c in self.claims):
            return
        self.claims.append(ValidatedClaim(text, terms, list(evidence_ids), verdict))

    def evidence_for(self, terms: set[str], theta: float, usable) -> list[str]:
        """Evidence ids of verified claims covering >= theta of ``terms`` whose evidence is still usable."""
        out: list[str] = []
        for c in self.claims:
            if terms and len(terms & c.terms) / len(terms) >= theta and all(usable(i) for i in c.evidence_ids):
                out += [i for i in c.evidence_ids if i not in out]
        return out

    def invalidate_evidence(self, ids: set[str]) -> int:
        n = len(self.claims)
        self.claims = [c for c in self.claims if not set(c.evidence_ids) & ids]
        return n - len(self.claims)
