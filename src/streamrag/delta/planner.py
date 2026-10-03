"""Delta query generation, semantic retrieval cache and the DeltaPlanner (Phase 6; docs/session/04).

DeltaQueryGenerator: a need's next query is derived from its *previous* query plus the semantic delta - constraint
components of retracted constraints are dropped, components of new constraints appended, the need's own words
replaced only if they changed. Nothing else of the session is rebuilt. (A test checks the result has the same terms
as a from-scratch build.)

SemanticCache: key = sha1(index content hash | retrieval-options hash | sorted analyzed terms of the query).
Analyzed terms are what retrieval actually depends on lexically, so reordered words hit, while any change of
entity, constraint or need changes the key (no incorrect reuse). Invalidation rules (docs/session/04):
  index / corpus snapshot changed   -> every key changes (hash in key) + explicit invalidate("index_changed")
  retrieval configuration changed   -> every key changes (options hash in key) + invalidate("config_changed")
  source chunk no longer available  -> entries whose evidence includes it are dropped (invalidate_evidence)
  entity / constraint / question change -> different terms -> different key (no rule needed, never over-invalidates)

DeltaPlanner: for every changed or new need decide retrieve | reuse_active (same semantic key as its active query)
| cache_hit (an earlier completed query of the session had this key: e.g. a retracted constraint returns the need
to an earlier state); supersede the previous query of changed needs and every query of superseded needs; carry the
evidence actions of the validity manager into the plan.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable

from streamrag.context.models import ContextChange
from streamrag.delta.evidence import EvidenceStore, EvidenceValidityManager
from streamrag.delta.models import DeltaPlan, EvidenceAction, QueryAction
from streamrag.intents.query_builder import IntentQueryBuilder
from streamrag.intents.tracker import IntentTracker
from streamrag.ledger.ledger import QueryLedger
from streamrag.models.intents import Constraint, Intent, IntentQuery, QueryComponent


class DeltaQueryGenerator:
    def __init__(self, qb: IntentQueryBuilder) -> None:
        self.qb = qb

    def generate(self, intent: Intent, constraints: list[Constraint], prev: IntentQuery | None) -> tuple[IntentQuery, str]:
        if prev is None or prev.intent_id != intent.intent_id:
            return self.qb.build(intent, constraints), "full"
        active = {c.constraint_id: c for c in constraints if intent.intent_id in c.applies_to}
        core_changed = prev.intent_version != intent.version and [
            c.text for c in prev.components if c.source != "constraint"] != self._core_texts(intent)
        comps = ([c for c in prev.components if c.source != "constraint"] if not core_changed
                 else self.qb.build(intent, []).components)
        terms = list(dict.fromkeys(self.qb.terms_fn(" ".join(c.text for c in comps))))
        kept = [c for c in prev.components if c.source == "constraint" and c.ref in active]
        for c in kept:
            terms += [t for t in self.qb.terms_fn(c.text) if t not in terms]
        comps = comps + kept
        have = {c.ref for c in kept}
        for cid, k in active.items():
            if cid in have:
                continue
            new = [t for t in self.qb.terms_fn(k.text) if t not in terms]
            if new:
                comps.append(QueryComponent(text=k.text, source="constraint", ref=cid, span=k.source_span))
                terms += new
        text = " ".join(c.text for c in comps).strip()
        return IntentQuery(intent_id=intent.intent_id, intent_version=intent.version, utterance_id=intent.utterance_id,
                           text=text, components=comps, terms=list(dict.fromkeys(terms))), "delta"

    def _core_texts(self, intent: Intent) -> list[str]:
        return [c.text for c in self.qb.build(intent, []).components]


class SemanticCache:
    def __init__(self, index_hash: str, options_hash: str, enabled: bool = True) -> None:
        self.index_hash, self.options_hash, self.enabled = index_hash, options_hash, enabled
        self.entries: dict[str, str] = {}                # key -> completed query id
        self.evidence: dict[str, list[str]] = {}         # key -> evidence ids of that query
        self.stats = {"hits": 0, "misses": 0, "puts": 0, "invalidated": 0}
        self.invalidations: list[dict] = []

    def key(self, terms: list[str]) -> str:
        raw = "|".join([self.index_hash, self.options_hash, " ".join(sorted(set(terms)))])
        return hashlib.sha1(raw.encode()).hexdigest()[:16]

    def get(self, key: str) -> str | None:
        if not self.enabled:
            return None
        q = self.entries.get(key)
        self.stats["hits" if q else "misses"] += 1
        return q

    def put(self, key: str, query_id: str, evidence_ids: list[str]) -> None:
        if self.enabled and key not in self.entries:
            self.entries[key] = query_id
            self.evidence[key] = list(evidence_ids)
            self.stats["puts"] += 1

    def invalidate(self, reason: str) -> int:
        n = len(self.entries)
        self.entries.clear()
        self.evidence.clear()
        self.stats["invalidated"] += n
        self.invalidations.append({"reason": reason, "entries": n})
        return n

    def invalidate_evidence(self, evidence_ids: set[str], reason: str = "source_unavailable") -> int:
        bad = [k for k, ev in self.evidence.items() if evidence_ids & set(ev)]
        for k in bad:
            self.entries.pop(k, None)
            self.evidence.pop(k, None)
        self.stats["invalidated"] += len(bad)
        self.invalidations.append({"reason": reason, "entries": len(bad)})
        return len(bad)


def options_hash(options) -> str:
    return hashlib.sha1(options.canonical_json().encode()).hexdigest()[:12]


class DeltaPlanner:
    def __init__(self, tracker: IntentTracker, ledger: QueryLedger, store: EvidenceStore,
                 validity: EvidenceValidityManager, cache: SemanticCache, deltaq: DeltaQueryGenerator,
                 last_query: dict[str, IntentQuery], delta_scope: str = "corpus") -> None:
        self.tracker, self.ledger, self.store, self.validity = tracker, ledger, store, validity
        self.cache, self.deltaq, self.last_query, self.delta_scope = cache, deltaq, last_query, delta_scope
        self._n = 0

    def plan(self, uid: str, changes: list[ContextChange], now_ms: float, full_restart: bool = False,
             restart_targets: list[str] | None = None) -> tuple[DeltaPlan, list[EvidenceAction]]:
        tr = self.tracker
        self._n += 1
        ev_actions: list[EvidenceAction] = []
        for ch in changes:
            ev_actions += self.validity.apply(ch, lambda k: tr.constraints[k].text if k in tr.constraints else "",
                                              lambda i: tr.intents[i].topic if i in tr.intents else None, now_ms)
        targets: list[tuple[str, str]] = []                   # (intent id, change id)
        superseded: list[str] = []
        for ch in changes:
            superseded += ch.superseded_intents
            for i in ch.new_intents + (ch.affected_intents if ch.change_type not in ("CORRECTION", "INTENT_REMOVAL")
                                       else []):
                if i not in [t for t, _ in targets]:
                    targets.append((i, ch.change_id))
        if full_restart:                                      # baseline: every active need of the frame again
            ids = restart_targets if restart_targets is not None else [i.intent_id for i in tr.active_session_intents()]
            targets = [(i, changes[0].change_id if changes else None) for i in ids]
        plan = DeltaPlan(plan_id=f"P{self._n}", utterance_id=uid, change_ids=[c.change_id for c in changes],
                         affected_intents=sorted({i for c in changes for i in c.affected_intents}),
                         new_intents=sorted({i for c in changes for i in c.new_intents}), full_restart=full_restart)
        corr_parent: dict[str, str] = {}                     # corrected need -> active query of the need it replaces
        for ch in changes:
            if ch.change_type == "CORRECTION" and ch.superseded_intents and ch.new_intents:
                q = self.ledger.active_for_intent(ch.superseded_intents[0])
                if q is not None:
                    corr_parent[ch.new_intents[0]] = q.query_id
        for old in superseded:
            q = self.ledger.active_for_intent(old)
            if q is not None and q.query_id not in plan.queries_to_supersede:
                plan.queries_to_supersede.append(q.query_id)
        for iid, chid in targets:
            it = tr.intents.get(iid)
            if it is None or it.status != "ACTIVE":
                continue
            ks = tr.constraints_for(it)
            prev = None if full_restart else self.last_query.get(iid)
            query, mode = self.deltaq.generate(it, ks, prev)
            key = self.cache.key(query.terms)
            active = None if full_restart else self.ledger.active_for_intent(iid)
            base = dict(intent_id=iid, intent_version=it.version, query=query, semantic_key=key, change_id=chid)
            parent = active.query_id if active else (None if full_restart else corr_parent.get(iid))
            if active is not None and active.semantic_key == key and active.status != "failed":
                plan.queries_to_reuse.append(QueryAction(action="reuse_active", reason="unchanged_semantics",
                                                         reused_query_id=active.query_id, **base))
                continue
            hit = None if full_restart else self.cache.get(key)
            if hit is not None:
                plan.queries_to_reuse.append(QueryAction(action="cache_hit", reason="semantic_cache_hit",
                                                         reused_query_id=hit, parent_query_id=parent,
                                                         supersedes_query_id=parent, **base))
                if active is not None and active.query_id not in plan.queries_to_supersede:
                    plan.queries_to_supersede.append(active.query_id)
                continue
            kind = "changed" if active is not None else ("corrected" if parent else "new")
            plan.queries_to_create.append(QueryAction(
                action="retrieve", reason=f"{mode}_query_for_{kind}_need", parent_query_id=parent,
                supersedes_query_id=parent, scope=self.delta_scope, **base))
            if active is not None and active.query_id not in plan.queries_to_supersede:
                plan.queries_to_supersede.append(active.query_id)
        for a in ev_actions:
            tag = f"{a.evidence_id}@{a.intent_id}"
            {"RETAIN": plan.evidence_to_retain, "REVALIDATE": plan.evidence_to_revalidate}.get(
                a.decision, plan.evidence_to_discard).append(tag)
        return plan, ev_actions
