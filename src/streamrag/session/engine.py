"""AdaptiveSessionEngine: change -> impact analysis -> targeted update (Phase 6; docs/session/, ADR-016).

    interpretation update (IntentTracker delta)
        -> frame assignment + ContextChangeDetector                 (CONTEXT_CHANGE_DETECTED)
        -> DeltaPlanner: evidence lifecycle + query actions          (EVIDENCE_*, DELTA_PLAN_CREATED, QUERY_SUPERSEDED)
        -> affected claims -> PENDING_VALIDATION / SUPERSEDED         (CLAIM_INVALIDATED)
        -> SessionStateVersion                                       (SESSION_VERSION_CREATED)
    the *driver* executes the plan (retrieve | reuse_active | cache_hit) and reports results:
    evidence result for a need
        -> evidence store (+ confirm / stale revalidation-required evidence)       (EVIDENCE_REVALIDATED)
        -> claims for the need's current version: register, select, revalidate    (CLAIM_CREATED / _REVALIDATED / _INVALIDATED)
        -> SessionStateVersion
    end of turn -> AnswerStateManager.update_answer_state                       (ANSWER_VERSION_CREATED / _UPDATED)

Drivers: the streaming MultiIntentCoordinator (virtual / realtime) and the synchronous pipelines in
``session/pipeline.py`` (benchmarks, demo). The engine never retrieves by itself.

``full_restart=True`` turns the same engine into the *full restart baseline*: every change re-plans every active
need with no reuse, no cache and no evidence retention.
"""

from __future__ import annotations

import time
from collections.abc import Callable

from streamrag.answers.manager import AnswerStateManager
from streamrag.claims.graph import ClaimExtractor, ClaimGraph, ClaimRevalidator
from streamrag.context.detector import ContextChangeDetector, FrameManager
from streamrag.context.models import ContextChange
from streamrag.delta.evidence import EvidenceStore, EvidenceValidityManager
from streamrag.delta.models import DeltaPlan, EvidenceAction, QueryAction
from streamrag.delta.planner import DeltaPlanner, DeltaQueryGenerator, SemanticCache, options_hash
from streamrag.intents.decomposer import Decomposition
from streamrag.intents.query_builder import IntentQueryBuilder
from streamrag.intents.tracker import IntentTracker
from streamrag.ledger.ledger import QueryLedger
from streamrag.models.answers import AnswerVersion
from streamrag.models.events import EventType as E
from streamrag.models.evidence import EvidenceSet
from streamrag.models.intents import Intent, IntentQuery, IntentSet, IntentSetDelta
from streamrag.models.retrieval import RetrievalOptions
from streamrag.session.memory import SessionMemory

Emit = Callable[..., object]


def _noop(*_a, **_k):
    return None


class AdaptiveSessionEngine:
    def __init__(self, session_id: str, cfg, tracker: IntentTracker, ledger: QueryLedger, qb: IntentQueryBuilder,
                 index_hash: str, options: RetrievalOptions, emit: Emit | None = None,
                 full_restart: bool = False, config_hash: str = "") -> None:
        self.session_id, self.cfg, self.full_restart = session_id, cfg, full_restart
        sc = cfg.session
        self.tracker, self.ledger, self.qb = tracker, ledger, qb
        self.terms_fn = qb.terms_fn
        self.emit = emit or _noop
        self.frames = FrameManager(sc.frame_overlap_min)
        self.detector = ContextChangeDetector(tracker)
        self.store = EvidenceStore()
        self.validity = EvidenceValidityManager(self.store, self.terms_fn)
        self.cache = SemanticCache(index_hash, options_hash(options), sc.cache and not full_restart)
        self.last_query: dict[str, IntentQuery] = {}
        self.planner = DeltaPlanner(tracker, ledger, self.store, self.validity, self.cache, DeltaQueryGenerator(qb),
                                    self.last_query, sc.delta_scope)
        self.graph = ClaimGraph()
        self.extractor = ClaimExtractor(self.terms_fn, sc.claims_per_intent, sc.claim_min_relevance,
                                        tracker.dec.idf_fn)
        self.revalidator = ClaimRevalidator(self.terms_fn)
        self.answers = AnswerStateManager(tracker, self.frames, self.graph, self.store)
        self.memory = SessionMemory(session_id, tracker, ledger, self.frames, self.store, self.cache, self.graph,
                                    self.answers, sc.transcript_window, sc.redact_pii,
                                    {"config_hash": config_hash, "index_content_hash": index_hash,
                                     "delta_scope": sc.delta_scope, "cache": sc.cache, "full_restart": full_restart})
        self.known: dict[str, Intent] = {}
        # Phase 9: terms an adaptive multi-hop retrieval linked to the need ("Ruritania" -> "Zone 2"): the bridge
        # target's words also make a sentence relevant to the need (bounded: <= 6 words per hop, sanitised)
        self.retrieval_terms: dict[str, list[str]] = {}
        self.pre_change_status: dict[str, str] = {}          # claim status before the pending change(s)
        self.pending_changes: list[ContextChange] = []
        self.pending_queries: list[str] = []
        self.plans: list[DeltaPlan] = []
        self.changes: list[ContextChange] = []
        self.timings: dict[str, list[float]] = {k: [] for k in (
            "change_detection", "delta_planning", "claim_revalidation", "answer_update", "evidence_update")}
        self.memory.extras["engine"] = (self._export_state, self._import_state)
        self.counters = self.memory.counters
        for k in ("retrievals_planned", "queries_reused_active", "cache_hits", "evidence_retained",
                  "evidence_revalidation_required", "evidence_superseded", "claims_revalidated", "claims_unchanged",
                  "claims_created", "no_change_updates"):
            self.counters[k] = 0

    # ------------------------------------------------------------------ snapshot (SessionMemory extras)
    def _export_state(self) -> dict:
        return {"known": [it.model_dump(mode="json") for it in self.known.values()],
                "last_query": {k: q.model_dump(mode="json") for k, q in self.last_query.items()},
                "detector_n": self.detector._n, "planner_n": self.planner._n,
                "pre_change_status": dict(self.pre_change_status),
                "pending_changes": [c.model_dump(mode="json") for c in self.pending_changes],
                "pending_queries": list(self.pending_queries),
                "changes": [c.model_dump(mode="json") for c in self.changes],
                "plans": [p.model_dump(mode="json") for p in self.plans]}

    def _import_state(self, d: dict | None) -> None:
        """Restore (or, with None, reset) the engine's own state: the interpretation it last planned from, the
        previous query of each need (delta query base), id counters and the change / plan logs."""
        d = d or {}
        self.known = {x["intent_id"]: Intent.model_validate(x) for x in d.get("known", [])}
        self.last_query.clear()                                  # shared with the planner: mutate in place
        self.last_query.update({k: IntentQuery.model_validate(v) for k, v in d.get("last_query", {}).items()})
        self.detector._n, self.planner._n = int(d.get("detector_n", 0)), int(d.get("planner_n", 0))
        self.pre_change_status = dict(d.get("pre_change_status", {}))
        self.pending_changes = [ContextChange.model_validate(x) for x in d.get("pending_changes", [])]
        self.pending_queries = list(d.get("pending_queries", []))
        self.changes = [ContextChange.model_validate(x) for x in d.get("changes", [])]
        self.plans = [DeltaPlan.model_validate(x) for x in d.get("plans", [])]

    # ------------------------------------------------------------------ helpers
    def _bump(self, k: str, n: int = 1) -> None:
        self.counters[k] = self.counters.get(k, 0) + n

    def _version(self, trigger: str, now: float, uid: str | None, changes: list[str], summary: str):
        v = self.memory.update_session(trigger, now, uid, changes, summary)
        if v is not None:
            self.emit(E.SESSION_VERSION_CREATED, "session_memory",
                      {"version_id": v.version_id, "parent_version": v.parent_version, "trigger": trigger,
                       "changes": changes, "summary": summary, "snapshot": v.state_snapshot.model_dump(mode="json")}, uid)
        return v

    def _remember(self) -> None:
        for i, it in self.tracker.intents.items():
            self.known[i] = it

    # ------------------------------------------------------------------ transcript layer
    def observe_utterance(self, uid: str, text: str, final: bool) -> None:
        self.memory.observe_utterance(uid, text, final)

    def no_change(self, uid: str, reason: str, now: float) -> ContextChange:
        """An update that changes nothing retrieval-relevant (e.g. 'okay.'): recorded, nothing re-run."""
        ch = ContextChange(change_id=self.detector._id(), change_type="NO_CHANGE", utterance_id=uid, cue=reason,
                           confidence=1.0, session_version_from=self.memory.version, at_ms=now)
        self.changes.append(ch)
        self._bump("no_change_updates")
        self.emit(E.CONTEXT_CHANGE_DETECTED, "context_change_detector", ch.model_dump(mode="json"), uid)
        return ch

    # ------------------------------------------------------------------ interpretation -> plan
    def interpret(self, uid: str, iset: IntentSet | None, delta: IntentSetDelta | None, d: Decomposition | None,
                  now: float) -> DeltaPlan | None:
        t0 = time.perf_counter()
        frame_action = self.frames.assign(uid, delta, d, self.tracker) if delta is not None else "none"
        changes = self.detector.detect(uid, delta, d, self.known, frame_action,
                                       lambda i: (q.query_id if (q := self.ledger.active_for_intent(i)) else None),
                                       self.memory.version, now)
        self.timings["change_detection"].append((time.perf_counter() - t0) * 1000.0)
        self.memory.observe_entities(uid)
        if not changes:
            self._remember()
            return None
        for ch in changes:
            self.changes.append(ch)
            self.emit(E.CONTEXT_CHANGE_DETECTED, "context_change_detector", ch.model_dump(mode="json"), uid)
        t1 = time.perf_counter()
        fr = self.frames.active
        restart_targets = [i for i in fr.intent_ids if self.tracker.intents.get(i) is not None
                           and self.tracker.intents[i].status == "ACTIVE"] if (fr and self.full_restart) else None
        plan, ev_actions = self.planner.plan(uid, changes, now, full_restart=self.full_restart,
                                             restart_targets=restart_targets)
        self.timings["delta_planning"].append((time.perf_counter() - t1) * 1000.0)
        self._emit_evidence(ev_actions, uid)
        # affected claims: the needs that changed, the needs that were superseded, and claims on changed evidence
        t2 = time.perf_counter()
        # (a claim's validity depends on the (evidence, its own need) assignment: claims of other needs that cite
        # the same chunk are not affected; claims already STALE / SUPERSEDED stay so unless re-selected later)
        affected_intents = {i for c in changes for i in c.affected_intents + c.superseded_intents}
        live = lambda cid: self.graph.claims[cid].status not in ("STALE", "SUPERSEDED")  # noqa: E731
        affected = sorted({cid for i in affected_intents for cid in self.graph.claims_of(i) if live(cid)}
                          | {cid for a in ev_actions if a.to_status != a.from_status
                             for cid in self.graph.linked_to(a.evidence_id)
                             if self.graph.claims[cid].intent_id == a.intent_id and live(cid)}, key=_num)
        plan.claims_to_revalidate = affected
        plan.claims_unaffected = [c for c in sorted(self.graph.claims, key=_num) if c not in affected]
        for cid in affected:
            self.pre_change_status.setdefault(cid, self.graph.claims[cid].status)
            c = self.graph.claims[cid]
            it = self.tracker.intents.get(c.intent_id or "")
            if it is not None and it.status == "ACTIVE" and cid in self.graph.selected.get(it.intent_id, []):
                t = self.graph.set_status(cid, "PENDING_VALIDATION", "affected_by_change", changes[0].change_id, now)
                if t is not None:
                    self.emit(E.CLAIM_INVALIDATED, "claim_revalidator", t.model_dump(mode="json"), uid)
        others = [cid for cid in affected if self.graph.claims[cid].status != "PENDING_VALIDATION"]
        for t in self.revalidator.validate(others, self.graph, self.store, self.tracker, now, changes[0].change_id):
            self.emit(E.CLAIM_INVALIDATED, "claim_revalidator", t.model_dump(mode="json"), uid)
        self.timings["claim_revalidation"].append((time.perf_counter() - t2) * 1000.0)
        for q in plan.queries_to_supersede:
            self.emit(E.QUERY_SUPERSEDED, "delta_planner", {"query_id": q, "plan_id": plan.plan_id}, uid, query_id=q)
        self.plans.append(plan)
        self.pending_changes += changes
        self._bump("retrievals_planned", len(plan.queries_to_create))
        self.emit(E.DELTA_PLAN_CREATED, "delta_planner", plan.model_dump(mode="json"), uid)
        self._remember()
        self._version("interpretation", now, uid, [c.change_id for c in changes],
                      "; ".join(f"{c.change_type}({','.join(c.affected_intents + c.new_intents)})" for c in changes))
        return plan

    def _emit_evidence(self, actions: list[EvidenceAction], uid: str) -> None:
        for a in actions:
            if a.to_status == a.from_status:
                continue
            if a.decision == "RETAIN":
                self._bump("evidence_retained")
                self.emit(E.EVIDENCE_RETAINED, "evidence_validity", a.model_dump(mode="json"), uid, intent_id=a.intent_id)
            else:
                self._bump("evidence_revalidation_required" if a.decision == "REVALIDATE" else "evidence_superseded")
                self.emit(E.EVIDENCE_INVALIDATED, "evidence_validity", a.model_dump(mode="json"), uid,
                          intent_id=a.intent_id)

    # ------------------------------------------------------------------ plan execution callbacks (driver)
    def query_dispatched(self, action: QueryAction, query_id: str) -> None:
        self.last_query[action.intent_id] = action.query
        self.pending_queries.append(query_id)

    def query_reused(self, action: QueryAction, uid: str, now: float, record_id: str | None = None,
                     es: EvidenceSet | None = None) -> None:
        self.last_query[action.intent_id] = action.query
        self._bump("cache_hits" if action.action == "cache_hit" else "queries_reused_active")
        self.emit(E.QUERY_REUSED, "delta_planner", {
            "intent_id": action.intent_id, "action": action.action, "reason": action.reason,
            "reused_query_id": action.reused_query_id, "query_id": record_id, "query_text": action.query.text,
            "semantic_key": action.semantic_key, "change_id": action.change_id}, uid, intent_id=action.intent_id,
            query_id=record_id or action.reused_query_id)
        if es is not None:          # re-confirm evidence awaiting revalidation and revalidate the need's claims
            self.on_result(action.intent_id, record_id or action.reused_query_id, es, now, uid, reused=True)

    def on_result(self, intent_id: str, query_id: str, es: EvidenceSet, now: float, uid: str | None = None,
                  reused: bool = False) -> None:
        it = self.tracker.intents.get(intent_id)
        if it is None or it.status != "ACTIVE":
            return                                                 # result of a superseded need: lineage only
        t0 = time.perf_counter()
        before = {k: a.status for k, a in self.store.assign.items() if k[1] == intent_id}
        react = self.store.add_results(intent_id, it.version, query_id, es, now, "cache_hit" if reused else "retrieved")
        for eid, t in react:                          # retrieved again for the current need: ACTIVE again
            self.emit(E.EVIDENCE_REVALIDATED, "evidence_validity", EvidenceAction(
                evidence_id=eid, intent_id=intent_id, decision="RETAIN", from_status=t.from_status,
                to_status=t.to_status, rule=t.rule).model_dump(mode="json"), uid, intent_id=intent_id)
        confirms = self.validity.confirm_after_retrieval(intent_id, {e.evidence_id for e in es.items}, query_id, now)
        for a in confirms:
            if a.to_status != a.from_status:
                self.emit(E.EVIDENCE_REVALIDATED, "evidence_validity", a.model_dump(mode="json"), uid, intent_id=intent_id)
        if not reused:
            self._drop_unconfirmed_partial_evidence(intent_id, query_id, es, now, uid)
        q = self.last_query.get(intent_id)
        if q is not None and not reused:
            self.cache.put(self.cache.key(q.terms), query_id, [e.evidence_id for e in es.items])
        self.timings["evidence_update"].append((time.perf_counter() - t0) * 1000.0)
        t1 = time.perf_counter()
        own_terms = list(q.terms if q else self.terms_fn(it.resolved_text))
        bridge = [t for t in self.retrieval_terms.get(intent_id, []) if t not in own_terms]   # Phase 9 hop targets
        cands = self.extractor.select(own_terms + bridge, self.store, intent_id,
                                      self.terms_fn(it.topic) if it.topic else None)
        if bridge:      # a sentence relevant only through the bridge words is about another entity of the class
            cands = [c for c in cands if set(self.terms_fn(c.text)) & set(own_terms)]
        frame = self.frames.frame_of(intent_id)
        prev_sel = list(self.graph.selected.get(intent_id, []))
        new_sel = []
        for c in cands:
            claim, created = self.graph.register(it, frame.frame_id if frame else None, c, now)
            new_sel.append(claim.claim_id)
            if created:
                self._bump("claims_created")
                self.emit(E.CLAIM_CREATED, "claim_extractor", claim.model_dump(mode="json"), uid, intent_id=intent_id)
        self.graph.selected[intent_id] = new_sel
        # targeted: the need's newly selected and previously selected claims, plus its claims on evidence whose
        # assignment to this need changed status
        changed_ev = {e for (e, i), a in self.store.assign.items()
                      if i == intent_id and (e, i) in before and before[(e, i)] != a.status}
        to_check = sorted(set(new_sel) | set(prev_sel)
                          | {c for e in changed_ev for c in self.graph.linked_to(e)
                             if self.graph.claims[c].intent_id == intent_id}, key=_num)
        transitions = self.revalidator.validate(to_check, self.graph, self.store, self.tracker, now)
        self.answers.conflicts[intent_id] = self.revalidator.conflicts(
            [c for c in new_sel if self.graph.claims[c].status in ("SUPPORTED", "PARTIALLY_SUPPORTED")],
            self.graph, self.store)
        self._bump("claims_revalidated", len(to_check))
        for t in transitions:
            kind = E.CLAIM_REVALIDATED if t.to_status in ("SUPPORTED", "PARTIALLY_SUPPORTED") else E.CLAIM_INVALIDATED
            self.emit(kind, "claim_revalidator", t.model_dump(mode="json"), uid, intent_id=intent_id)
        self.timings["claim_revalidation"].append((time.perf_counter() - t1) * 1000.0)
        self._version("evidence", now, uid, [], f"evidence for {intent_id} via {query_id}")

    def _drop_unconfirmed_partial_evidence(self, intent_id: str, query_id: str, es: EvidenceSet, now: float,
                                           uid: str | None) -> None:
        """Streaming: when the need's current query returns, evidence that only *superseded* queries of the same
        utterance retrieved for the need (queries built on an earlier, partial transcript) and that the current
        query did not return again is not confirmed by what the user finally said -> STALE
        (not_confirmed_by_refined_query). Without this, "What did a single ride cost" (current fare) followed by
        "... in 2025" kept the current-fare evidence and its claim next to the 2025 one (Phase 10 error analysis,
        early commitment). Evidence from earlier utterances keeps its Phase 6 lifecycle (RETAINED across
        refinements); batch processing issues one query per need and utterance and is unaffected."""
        if uid is None or self.ledger is None:
            return
        try:
            rec = self.ledger.get(query_id)
        except KeyError:
            return
        if rec.utterance_id != uid or rec.stale:
            return
        returned = {e.evidence_id for e in es.items}
        for a in self.store.usable(intent_id):
            if a.evidence_id in returned or not a.query_ids:
                continue
            srcs = []
            for q in a.query_ids:
                try:
                    srcs.append(self.ledger.get(q))
                except KeyError:
                    srcs.append(None)
            if all(r is not None and r.utterance_id == uid and r.stale and r.query_id != query_id for r in srcs):
                before = a.status
                if self.store.transition(a.evidence_id, intent_id, "STALE", "not_confirmed_by_refined_query", None,
                                         now, query_id) is not None:
                    self._bump("evidence_superseded")
                    self.emit(E.EVIDENCE_INVALIDATED, "evidence_validity", EvidenceAction(
                        evidence_id=a.evidence_id, intent_id=intent_id, decision="SUPERSEDE", from_status=before,
                        to_status="STALE", rule="not_confirmed_by_refined_query").model_dump(mode="json"), uid,
                        intent_id=intent_id)

    # ------------------------------------------------------------------ answer
    def commit_answer(self, uid: str, now: float) -> AnswerVersion | None:
        t0 = time.perf_counter()
        v = self.answers.update_answer_state(uid, self.memory.version, now, self.pending_changes, self.pending_queries,
                                             full_rerun=self.full_restart)
        self.timings["answer_update"].append((time.perf_counter() - t0) * 1000.0)
        outcome = {c: (self.pre_change_status[c], self.graph.claims[c].status) for c in self.pre_change_status}
        self._bump("claims_unchanged", sum(1 for a, b in outcome.values() if a == b))
        self.last_revalidation = outcome
        self.pre_change_status = {}
        if v is None:
            self.pending_changes, self.pending_queries = [], []
            return None
        self.pending_changes, self.pending_queries = [], []
        kind = E.ANSWER_VERSION_CREATED if v.parent_version is None else E.ANSWER_VERSION_UPDATED
        self.emit(kind, "answer_state", {
            "answer_id": v.answer_id, "version": v.version, "frame_id": v.topic_id, "kind": v.kind,
            "supersedes_answer_id": v.supersedes_answer_id, "session_version": v.session_version,
            "sections": [{"section_id": s.section_id, "intent_id": s.intent_id, "status": s.status,
                          "needs_regeneration": s.needs_regeneration, "claim_ids": s.claim_ids,
                          "uncertainty": [u.model_dump(mode="json") for u in s.uncertainty]} for s in v.sections],
            "claim_ids": v.claim_ids, "evidence_ids": v.evidence_ids, "citations": v.citations,
            "diff": v.diff.model_dump(mode="json"), "change_summary": v.change_summary,
            "delta_queries": v.delta_queries}, uid)
        self._version("answer", now, uid, [], f"answer {v.answer_id} ({v.kind})")
        return v


def _num(x: str) -> int:
    digits = "".join(ch for ch in x if ch.isdigit())
    return int(digits) if digits else 0
