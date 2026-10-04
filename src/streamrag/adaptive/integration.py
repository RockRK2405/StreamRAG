"""Session integration of the adaptive controller (docs/architecture/13 §integration).

``SessionAdaptiveRetriever`` is what the Phase 6 synchronous pipeline and the Phase 8 runtime use when
``adaptive_retrieval.enabled``. It owns one controller (and so one query cache) per session and connects it to:

* the interpreted need (Phase 5/6 ``Intent`` + active constraints) and its Phase 6 delta query (``req.text``);
* the session evidence (Phase 6 ``EvidenceStore``: usable = ACTIVE / RETAINED for an active need) - the controller's
  SESSION_REUSE / delta requirements read it through a ``SessionView`` snapshot (never mutate it);
* Phase 6 context changes -> cache invalidation with reasons (entity changed, constraint changed, stale evidence);
* Phase 7 verified claims -> ValidatedClaimCache (claim reuse while the evidence stays usable).
"""

from __future__ import annotations

import threading

from streamrag.adaptive.cache import AdaptiveQueryCache, ValidatedClaimCache
from streamrag.adaptive.catalog import MetadataCatalog
from streamrag.adaptive.controller import AdaptiveRequest, AdaptiveRetrievalController, SessionView
from streamrag.adaptive.lexicon import RetrievalLexicon

_LOCK = threading.Lock()


def shared_catalog(service, cfg) -> MetadataCatalog:
    """One catalog per (service, adaptive config): built from the index once, shared by every session."""
    an = service.bundle.analyzer
    key = (cfg.adaptive_retrieval.model_dump_json(include={"filter_fields", "authority"}),)
    with _LOCK:
        store = service.__dict__.setdefault("_adaptive_catalogs", {})
        cat = store.get(key)
        if cat is None:
            cat = store[key] = MetadataCatalog.build(service, cfg, lambda t: list(dict.fromkeys(an.tokens(t))))
        return cat


def claim_lexicon():
    from streamrag.answer_state.engine import LEXICON
    from streamrag.claims.decomposer import ClaimLexicon
    return ClaimLexicon.load(LEXICON)


def make_controller(service, cfg, **kw) -> AdaptiveRetrievalController:
    ac = cfg.adaptive_retrieval
    lex = kw.pop("lexicon", None) or RetrievalLexicon.load(ac.lexicon)
    cache = kw.pop("cache", None) or AdaptiveQueryCache(service.bundle.manifest.content_hash, ac.cache_max_entries,
                                                        ac.cache)
    return AdaptiveRetrievalController(service, cfg, claim_lexicon(), catalog=shared_catalog(service, cfg),
                                       lexicon=lex, cache=cache, **kw)


class SessionAdaptiveRetriever:
    def __init__(self, service, cfg, emit=None, **controller_kw) -> None:
        self.cfg = cfg
        self.ctl = make_controller(service, cfg, **controller_kw)
        self.claims = ValidatedClaimCache()
        self.emit = emit or (lambda *a, **k: None)
        self.results: dict[str, object] = {}          # query id -> AdaptiveResult
        self.invalidations: list[dict] = []
        self._seen: set[str] = set()                  # Phase 6 change ids already applied to the cache

    def close(self) -> None:
        self.ctl.close()

    # ------------------------------------------------------------------ session snapshot
    @staticmethod
    def view(engine, evidence_sets: dict, tracker) -> SessionView:
        """Usable evidence of the session's active needs (snapshot; Evidence objects from completed queries)."""
        if engine is None:
            return SessionView()
        objs = {}
        for es in evidence_sets.values():
            if es is None:
                continue
            for e in es.items:
                objs.setdefault(e.evidence_id, e)
        usable_ids = set()
        for it in tracker.active_session_intents():
            usable_ids |= {a.evidence_id for a in engine.store.usable(it.intent_id)}
        pool = {i: objs[i] for i in usable_ids if i in objs}
        return SessionView(pool=pool, usable=usable_ids.__contains__)

    def request(self, query_id: str, action, tracker, view: SessionView, changes=(), latency_budget_ms=None,
                checkpoint=None, cancelled=None, now_ms: float = 0.0, search_hook=None) -> AdaptiveRequest:
        it = tracker.intents.get(action.intent_id)
        dropped = [t for ch in changes for d in ch.diffs if d.intent_id in (action.intent_id, *(ch.superseded_intents))
                   for t in d.terms_removed]
        n = 1
        if it is not None:
            n = sum(1 for x in tracker.active_session_intents() if x.utterance_id == it.utterance_id)
        view.claims = self.claims
        return AdaptiveRequest(query_id=query_id, text=action.query.text, original_text=it.text if it else None,
                               intent=it, constraints=tracker.constraints_for(it) if it is not None else [],
                               n_intents=n, dropped_entities=dropped, session=view,
                               latency_budget_ms=latency_budget_ms, checkpoint=checkpoint, cancelled=cancelled,
                               now_ms=now_ms, search_hook=search_hook)

    def run(self, req: AdaptiveRequest):
        res = self.ctl.run(req)
        self.results[req.query_id] = res
        return res

    def bridge_terms(self, res) -> list[str]:
        """Analyzed words of the bridge targets of completed hops (for the claim stage's sentence relevance)."""
        out: list[str] = []
        for h in res.hops:
            if h.bridge and "->" in h.bridge and h.status == "COMPLETED" and not h.bridge.startswith("reference"):
                out += [t for t in self.ctl.terms_fn(h.bridge.split("->", 1)[1]) if t not in out]
        return out[:6]

    # ------------------------------------------------------------------ invalidation / claim reuse
    def on_changes(self, changes, tracker, now_ms: float = 0.0) -> list[dict]:
        """Phase 6 changes -> adaptive cache invalidation (entity / constraint), logged with the change id."""
        out = []
        terms_fn = self.ctl.terms_fn
        for ch in changes:
            if ch.change_type in ("ENTITY_CHANGE", "CORRECTION", "QUESTION_CHANGE"):
                removed = {t for d in ch.diffs for t in d.terms_removed}
                for d in ch.diffs:
                    if d.topic_before and d.topic_before != d.topic_after:
                        removed |= set(terms_fn(d.topic_before)) - set(terms_fn(d.topic_after or ""))
                inv = self.ctl.cache.invalidate_entities(removed, now_ms) if removed else None
                if inv:
                    out.append({**inv, "change_id": ch.change_id, "change_type": ch.change_type})
            if ch.change_type == "CONSTRAINT_REMOVAL" and ch.removed_constraints:
                removed = {t for c in ch.removed_constraints for t in terms_fn(c)}
                inv = self.ctl.cache.invalidate_constraints(removed, now_ms)
                if inv:
                    out.append({**inv, "change_id": ch.change_id, "change_type": ch.change_type})
        self.invalidations += out
        return out

    def sync_changes(self, changes, tracker, now_ms: float = 0.0) -> list[dict]:
        """Apply the Phase 6 changes not seen yet (idempotent)."""
        new = [c for c in changes if c.change_id not in self._seen]
        self._seen |= {c.change_id for c in new}
        return self.on_changes(new, tracker, now_ms) if new else []

    def on_stale_evidence(self, ids: set[str], now_ms: float = 0.0) -> list[dict]:
        out = []
        inv = self.ctl.cache.invalidate_evidence(ids, now_ms)
        if inv:
            out.append(inv)
        n = self.claims.invalidate_evidence(ids)
        if n:
            out.append({"reason": "stale_evidence", "n": n, "keys": [], "detail": "validated_claims",
                        "at_ms": now_ms})
        self.invalidations += out
        return out

    def on_grounded(self, ga) -> int:
        """Phase 7 verified claims -> ValidatedClaimCache."""
        if ga is None:
            return 0
        n = len(self.claims.claims)
        terms_fn = self.ctl.terms_fn
        for c in ga.claims:
            v = ga.verifications.get(c.claim_id)
            if v is not None and v.status == "SUPPORTED":
                self.claims.add(c.text, set(terms_fn(c.text)), list(v.supporting_evidence), v.status)
        return len(self.claims.claims) - n
