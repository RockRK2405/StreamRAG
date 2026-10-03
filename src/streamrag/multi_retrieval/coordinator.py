"""MultiIntentCoordinator: streaming multi-intent retrieval inside a StreamingSession (docs/multi_intent/05, 08-09).

Per controller tick of an utterance:

  RETRIEVAL_DECISION (Phase 4 controller = the gate: suppression, stability, retrieval-worthiness)
     gate open  -> IntentTracker.update(transcript)  -> INTENTS_UPDATED (+ INTENT_DETECTED / _UPDATED / _SUPERSEDED)
                -> per active intent: IntentQueryBuilder -> unchanged query? reuse evidence (no retrieval)
                                                         -> guards: per-intent budget, utterance budget, cooldown
                -> QUERY_GENERATED (ledger record with intent lineage) -> MULTI_QUERY_STARTED -> executor jobs
     gate closed -> nothing (suppressed / not yet stable / not retrieval-worthy)
  each RETRIEVAL_COMPLETED -> when the batch is done: MULTI_QUERY_COMPLETED -> provisional fusion
  turn completion -> final fusion (+ optional rerank: RERANK_STARTED / RERANK_COMPLETED)
                  -> EVIDENCE_DEDUPLICATED, EVIDENCE_FUSED (UnifiedEvidenceSet) -> TURN_COMPLETED payload

Delta retrieval: only intents whose query text changed (added, or modified by text/constraint/context) are
retrieved again; unchanged intents keep their evidence. Superseded/removed intents: queued queries are cancelled,
completed evidence stays in the ledger with ``stale_reason`` but is excluded from fusion.

Gate semantics: the controller's storm guards (utterance cooldown/budget, novelty) are designed for one query per
utterance; in multi-intent mode they are replaced by per-intent guards, so a decision WAIT{cooldown,
provisional_budget_exhausted, retrieval_in_flight} or SKIP{redundant, budget} still opens the gate.

Phase 6 (``session.enabled``; docs/session/, ADR-016): an ``AdaptiveSessionEngine`` owns the session state. Every
gate-open tick: tracker update -> engine.interpret (frames, context changes, delta plan, evidence lifecycle, claim
invalidation) -> the coordinator executes the plan: ``retrieve`` actions under the same per-intent guards (a guarded
action is deferred and re-dispatched at the next tick while its need version is current), ``reuse_active`` actions
re-use the active query, ``cache_hit`` actions create a ``reused`` ledger record. Completed results of a need's
active query go to engine.on_result (evidence store, claims, revalidation); the answer version is committed at turn
completion (ANSWER_VERSION_* precede TURN_COMPLETED). The gate also opens for late-arriving details
(context/cues.py). Turn intents = the utterance's own active needs + the earlier needs the turn changed.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from streamrag.context.cues import late_detail_gate
from streamrag.context.detector import net_change_types
from streamrag.fusion.engine import EvidenceFusionEngine, IntentEvidence
from streamrag.intents.decomposer import IntentDecomposer
from streamrag.intents.query_builder import IntentQueryBuilder
from streamrag.intents.tracker import IntentTracker
from streamrag.models.events import EventType as E
from streamrag.streaming.executor import Job

GATE_OPEN_WAIT = frozenset({"cooldown", "provisional_budget_exhausted", "retrieval_in_flight"})
GATE_OPEN_SKIP = frozenset({"redundant", "budget"})
# A correction ("actually I meant the lamp, not the lens") is not request-shaped, so the Phase 4 act classifier may
# call it not (yet) retrieval-worthy; when there is an active need to correct, such decisions still open the gate.
CORRECTION_GATE_REASONS = frozenset({"not_retrieval_worthy", "not_yet_retrieval_worthy", "low_specificity",
                                     "awaiting_stability"})


def gate_open(decision) -> bool:
    if decision.decision == "RETRIEVE":
        return True
    if decision.decision == "WAIT":
        return decision.reason in GATE_OPEN_WAIT
    return decision.skip_kind in GATE_OPEN_SKIP


@dataclass
class IntentStack:
    """Everything the coordinator needs besides the session (built once per index)."""

    decomposer: IntentDecomposer
    query_builder: IntentQueryBuilder
    fusion: EvidenceFusionEngine
    index_hash: str = ""                                    # Phase 6 semantic cache key component


@dataclass
class _Batch:
    batch_id: str
    utterance_id: str
    pending: set[str]
    query_ids: list[str]
    started_ms: float
    wall_ms: dict[str, float] = field(default_factory=dict)


class MultiIntentCoordinator:
    def __init__(self, session, stack: IntentStack) -> None:
        self.s = session
        self.cfg = session.cfg.multi_intent
        self.stack = stack
        self.tracker = IntentTracker(session.session_id, stack.decomposer, self.cfg)
        self.batches: dict[str, _Batch] = {}
        self.batch_of: dict[str, str] = {}
        self.unified: dict[str, object] = {}            # utterance -> latest UnifiedEvidenceSet
        self.reuse: dict[str, str] = {}                 # intent -> earlier session query whose evidence it reuses
        self.decompose_wall_ms: list[float] = []
        self.query_wall_ms: list[float] = []
        self.fusion_wall_ms: list[float] = []
        self.rerank_wall_ms: list[float] = []
        self._n_batch = 0
        self.engine = None
        if session.cfg.session.enabled:
            from streamrag.session.engine import AdaptiveSessionEngine
            self.engine = AdaptiveSessionEngine(session.session_id, session.cfg, self.tracker, session.ledger,
                                                stack.query_builder, stack.index_hash, session.options,
                                                emit=session.emit, config_hash=str(session.meta.get("config_hash", "")))
        self.deferred: dict[str, object] = {}           # intent -> guarded QueryAction awaiting dispatch
        self.turn_intents: dict[str, list[str]] = {}    # utterance -> needs the turn created or changed
        self.planned: set[str] = set()                  # utterances with an engine plan
        self.answers: dict[str, object] = {}            # utterance -> AnswerVersion committed at its turn end

    # ------------------------------------------------------------------ controller tick
    def on_decision(self, uid: str, tick: str, decision, now: float, trigger_chunk: int | None) -> None:
        s = self.s
        final = tick == "utterance_end"
        if self.engine is not None and final:
            self.engine.observe_utterance(uid, s.chunks.get_current_transcript(uid), True)
        if not gate_open(decision) and not self._correction_gate(uid, decision):
            if self.engine is not None and final and uid not in self.planned:
                self.engine.no_change(uid, f"closed:{decision.reason}", now)
            return
        transcript = s.chunks.get_current_transcript(uid)
        if self.engine is not None and s.cfg.session.redact_pii:      # Phase 6: PII never enters session memory
            from streamrag.session.safety import redact
            transcript = redact(transcript)[0]
        t0 = time.perf_counter()
        iset, delta, d = self.tracker.update(uid, transcript, now, [r.terms for r in s.ledger.all()], final=final)
        dec_ms = (time.perf_counter() - t0) * 1000.0
        self.decompose_wall_ms.append(dec_ms)
        if delta is not None:
            self._emit_intent_events(uid, iset, delta, d, dec_ms)
        if self.engine is not None:
            plan = self.engine.interpret(uid, iset, delta, d, now)
            if plan is not None:
                self.planned.add(uid)
                ids = self.turn_intents.setdefault(uid, [])
                ids += [i for i in plan.new_intents + plan.affected_intents if i not in ids]
            elif final and uid not in self.planned:
                self.engine.no_change(uid, "no_interpretation_change", now)
            self._execute_plan(uid, plan, tick, now, trigger_chunk, final)
            return
        self._dispatch(uid, iset, tick, now, trigger_chunk, final)

    def _correction_gate(self, uid: str, decision) -> bool:
        if self.engine is not None:                                # Phase 6: any late-arriving detail
            return late_detail_gate(decision, bool(self.tracker.active_session_intents()),
                                    self.s.chunks.get_current_transcript(uid), self.stack.decomposer.lx,
                                    self.stack.decomposer.terms_fn)
        if decision.reason not in CORRECTION_GATE_REASONS:
            return False                                           # suppressed acts never open the gate
        if not any(it.status == "ACTIVE" for it in self.tracker.intents.values()):
            return False
        from streamrag.intents.text import match_phrase, tokenize
        lows = [t.lower for t in tokenize(self.s.chunks.get_current_transcript(uid)) if not t.punct]
        lx = self.stack.decomposer.lx
        return any(match_phrase(lows, i, lx.correction_phrases) for i in range(len(lows)))

    def _emit_intent_events(self, uid, iset, delta, d, dec_ms) -> None:
        s = self.s
        rep = self.tracker.llm_reports.get(uid)
        s.emit(E.INTENTS_UPDATED, "intent_decomposer", {
            "version": iset.version, "source": iset.source, "intent_count": len(iset.intents),
            "intents": [{"intent_id": i.intent_id, "version": i.version, "text": i.text, "type": i.intent_type,
                         "confidence": i.confidence, "priority": i.priority} for i in iset.intents],
            "global_constraints": [c.model_dump(mode="json") for c in iset.global_constraints],
            "local_constraints": [c.model_dump(mode="json") for c in iset.local_constraints],
            "relationships": [r.model_dump(mode="json") for r in iset.relationships],
            "decomposition_confidence": iset.decomposition_confidence,
            "delta": delta.model_dump(mode="json"),
            "merged": [m.model_dump(mode="json") for m in iset.merged],
            "dropped": [x.model_dump(mode="json") for x in iset.dropped],
            "llm_check": None if rep is None else {"status": rep.status, "gated_by": rep.gated_by,
                                                   "attempts": rep.attempts, "issues": rep.issues},
            "wall": {"decompose_ms": round(dec_ms, 4)}}, uid)
        for iid in delta.added:
            it = self.tracker.intents[iid]
            s.emit(E.INTENT_DETECTED, "intent_decomposer", {"intent": it.model_dump(mode="json")}, uid, intent_id=iid)
        for m in delta.modified:
            it = self.tracker.intents[m.intent_id]
            s.emit(E.INTENT_UPDATED, "intent_decomposer", {"from_version": m.from_version, "to_version": m.to_version,
                                                           "changed": m.changed, "status": it.status,
                                                           "intent": it.model_dump(mode="json")}, uid,
                   intent_id=m.intent_id)
        for sup in delta.superseded:
            s.emit(E.INTENT_SUPERSEDED, "intent_decomposer", {"old": sup.old, "new": sup.new, "cue": sup.cue,
                                                              "stale_queries": s.ledger.mark_intent_stale(
                                                                  sup.old, "intent_superseded")},
                   uid, intent_id=sup.old)
            self._cancel_queued(sup.old, uid, f"intent_superseded_by_{sup.new}")
        for iid in delta.removed:
            s.emit(E.INTENT_UPDATED, "intent_decomposer", {"status": "DROPPED", "changed": [],
                                                           "reason": "no_longer_in_decomposition",
                                                           "stale_queries": s.ledger.mark_intent_stale(
                                                               iid, "intent_removed")}, uid, intent_id=iid)
            self._cancel_queued(iid, uid, "intent_removed")

    def _cancel_queued(self, intent_id: str, uid: str, reason: str) -> None:
        s = self.s
        for r in s.ledger.for_intent(intent_id):
            if r.status == "queued" and s.executor.cancel_queued(r.query_id):
                s.ledger.update(r.query_id, status="cancelled", retrieval_status="cancelled")
                s.emit(E.RETRIEVAL_CANCELLED, "query_ledger", {"query_id": r.query_id, "reason": reason}, uid,
                       intent_id=intent_id)
                self._batch_done(r.query_id)

    # ------------------------------------------------------------------ dispatch (delta retrieval)
    def _dispatch(self, uid, iset, tick, now, trigger_chunk, final) -> None:
        s, cfg = self.s, self.cfg
        ks = iset.global_constraints + iset.local_constraints
        plan = []
        for it in sorted(iset.intents, key=lambda i: i.priority):
            t0 = time.perf_counter()
            q = self.stack.query_builder.build(it, ks)
            self.query_wall_ms.append((time.perf_counter() - t0) * 1000.0)
            last = s.ledger.active(uid, it.intent_id)
            if last is not None and last.query_text == q.text:
                continue                                           # unchanged: reuse its evidence
            if last is None:                                       # REQ-MI-005: session ledger dedup
                hit = self._ledger_hit(q.terms, uid)
                if hit is not None:
                    if self.reuse.get(it.intent_id) != hit.query_id:
                        self.reuse[it.intent_id] = hit.query_id
                        s.emit(E.RETRIEVAL_SKIPPED, "multi_query", {"reason": "ledger_hit", "skip_kind": "redundant",
                                                                    "query_text": q.text, "ledger_ref": hit.query_id,
                                                                    "tick": tick}, uid, intent_id=it.intent_id)
                    continue
            n_int = sum(1 for r in s.ledger.for_intent(it.intent_id) if r.status != "cancelled")
            n_utt = sum(1 for r in s.ledger.for_utterance(uid) if r.status != "cancelled") + len(plan)
            reason = None
            if n_int >= cfg.max_queries_per_intent:
                reason = "intent_budget_exhausted"
            elif n_utt >= cfg.max_queries_per_utterance:
                reason = "utterance_budget_exhausted"
            elif not final and last is not None and now - last.created_at_ms < cfg.intent_cooldown_ms:
                reason = "intent_cooldown"
            if reason is not None:
                s.emit(E.RETRIEVAL_SKIPPED, "multi_query", {"reason": reason, "skip_kind": "budget",
                                                            "query_text": q.text, "ledger_ref": last.query_id
                                                            if last else None, "tick": tick}, uid,
                       intent_id=it.intent_id)
                continue
            plan.append((it, q, last))
        if not plan:
            return
        self._n_batch += 1
        bid = f"B{self._n_batch}"
        qids = []
        for it, q, last in plan:
            spans = [(c.span.start, c.span.end) for c in q.components if c.span.utterance_id == uid]
            rec, prev = s.ledger.create(uid, q.text, s.chunks.get_current_transcript(uid), spans, q.terms, now,
                                        "final" if final else "provisional", trigger_chunk, tick,
                                        "intent_added" if last is None else "intent_modified",
                                        intent_id=it.intent_id, intent_version=it.version, batch_id=bid)
            self.tracker.record_query(it.intent_id, rec.query_id)
            s.emit(E.QUERY_GENERATED, "intent_query_builder", {
                "query_id": rec.query_id, "intent_version": it.version, "query_text": q.text, "trigger": rec.trigger,
                "components": [c.model_dump(mode="json") for c in q.components], "relation": rec.relation,
                "supersedes": rec.supersedes, "lineage_root": rec.lineage_root, "batch_id": bid,
                "superseded_status": prev.status if prev else None}, uid, intent_id=it.intent_id,
                query_id=rec.query_id)
            if prev is not None and prev.status == "queued" and s.cfg.controller.cancel_superseded == "queued_only":
                if s.executor.cancel_queued(prev.query_id):
                    s.ledger.update(prev.query_id, status="cancelled", retrieval_status="cancelled")
                    s.emit(E.RETRIEVAL_CANCELLED, "query_ledger", {"query_id": prev.query_id,
                                                                   "reason": "superseded_before_start",
                                                                   "superseded_by": rec.query_id}, uid,
                           intent_id=it.intent_id)
                    self._batch_done(prev.query_id)
            qids.append(rec.query_id)
        self.batches[bid] = _Batch(bid, uid, set(qids), qids, now)
        for q in qids:
            self.batch_of[q] = bid
        s.emit(E.MULTI_QUERY_STARTED, "multi_query", {
            "batch_id": bid, "query_ids": qids, "intent_ids": [it.intent_id for it, _, _ in plan],
            "dispatch": "parallel" if s.executor.max_concurrency > 1 else "sequential",
            "max_concurrency": s.executor.max_concurrency, "reused_intents": [
                i.intent_id for i in iset.intents if i.intent_id not in {it.intent_id for it, _, _ in plan}]}, uid)
        for q in qids:
            rec = s.ledger.get(q)
            s.ledger.update(q, status="queued", retrieval_queued_at_ms=s.sched.now_ms())
            s.executor.submit(Job(q, rec.query_text, s.options, s._on_start, s._on_done))

    # ------------------------------------------------------------------ Phase 6: execute an engine delta plan
    def _execute_plan(self, uid, plan, tick, now, trigger_chunk, final) -> None:
        s, cfg, eng = self.s, self.cfg, self.engine
        creates = list(plan.queries_to_create) if plan is not None else []
        planned = {a.intent_id for a in creates + (plan.queries_to_reuse if plan is not None else [])}
        for iid, a in list(self.deferred.items()):            # guarded earlier, still the current need version
            it = self.tracker.intents.get(iid)
            if it is None or it.status != "ACTIVE" or it.version != a.intent_version or iid in planned:
                self.deferred.pop(iid)
                continue
            creates.append(a)
            self.deferred.pop(iid)
        for a in (plan.queries_to_reuse if plan is not None else []):
            if a.action == "cache_hit":
                src = s.ledger.get(a.reused_query_id)
                rec, _ = s.ledger.create(uid, a.query.text, s.chunks.get_current_transcript(uid), [], a.query.terms,
                                         now, "final" if final else "provisional", trigger_chunk, tick, a.reason,
                                         intent_id=a.intent_id, intent_version=a.intent_version,
                                         parent_query_id=a.parent_query_id, derived_from_change_id=a.change_id,
                                         semantic_key=a.semantic_key, status="reused", reused_from=a.reused_query_id)
                s.ledger.update(rec.query_id, evidence_ids=list(src.evidence_ids), retrieval_status="cache_hit",
                                evidence_set_id=src.evidence_set_id, citations=list(src.citations))
                self.tracker.record_query(a.intent_id, rec.query_id)
                eng.query_reused(a, uid, now, rec.query_id, s.evidence.get(a.reused_query_id))
            else:
                act = s.ledger.get(a.reused_query_id)
                eng.query_reused(a, uid, now, None, s.evidence.get(a.reused_query_id)
                                 if act is not None and act.status == "completed" else None)
        go = []
        for a in creates:                                     # budgets are per (need, utterance), as in Phase 5
            n_int = sum(1 for r in s.ledger.for_intent(a.intent_id)
                        if r.utterance_id == uid and r.status not in ("cancelled", "reused"))
            n_utt = sum(1 for r in s.ledger.for_utterance(uid) if r.status not in ("cancelled", "reused")) + len(go)
            last = s.ledger.active_for_intent(a.intent_id)
            reason = None
            if n_int >= cfg.max_queries_per_intent:
                reason = "intent_budget_exhausted"
            elif n_utt >= cfg.max_queries_per_utterance:
                reason = "utterance_budget_exhausted"
            elif not final and last is not None and last.utterance_id == uid \
                    and now - last.created_at_ms < cfg.intent_cooldown_ms:
                reason = "intent_cooldown"
            if reason is not None:
                self.deferred[a.intent_id] = a
                s.emit(E.RETRIEVAL_SKIPPED, "multi_query", {"reason": reason, "skip_kind": "budget",
                                                            "query_text": a.query.text, "deferred": True,
                                                            "ledger_ref": last.query_id if last else None,
                                                            "tick": tick}, uid, intent_id=a.intent_id)
                continue
            go.append(a)
        if not go:
            return
        self._n_batch += 1
        bid = f"B{self._n_batch}"
        qids = []
        for a in go:
            spans = [(c.span.start, c.span.end) for c in a.query.components if c.span.utterance_id == uid]
            rec, prev = s.ledger.create(uid, a.query.text, s.chunks.get_current_transcript(uid), spans, a.query.terms,
                                        now, "final" if final else "provisional", trigger_chunk, tick, a.reason,
                                        intent_id=a.intent_id, intent_version=a.intent_version, batch_id=bid,
                                        parent_query_id=a.parent_query_id, derived_from_change_id=a.change_id,
                                        semantic_key=a.semantic_key)
            self.tracker.record_query(a.intent_id, rec.query_id)
            eng.query_dispatched(a, rec.query_id)
            s.emit(E.QUERY_GENERATED, "delta_query_generator", {
                "query_id": rec.query_id, "intent_version": a.intent_version, "query_text": a.query.text,
                "trigger": rec.trigger, "components": [c.model_dump(mode="json") for c in a.query.components],
                "relation": rec.relation, "supersedes": rec.supersedes, "lineage_root": rec.lineage_root,
                "batch_id": bid, "parent_query_id": a.parent_query_id, "derived_from_change_id": a.change_id,
                "semantic_key": a.semantic_key, "action_reason": a.reason,
                "superseded_status": prev.status if prev else None}, uid, intent_id=a.intent_id,
                query_id=rec.query_id)
            if prev is not None and prev.status == "queued" and s.cfg.controller.cancel_superseded == "queued_only":
                if s.executor.cancel_queued(prev.query_id):
                    s.ledger.update(prev.query_id, status="cancelled", retrieval_status="cancelled")
                    s.emit(E.RETRIEVAL_CANCELLED, "query_ledger", {"query_id": prev.query_id,
                                                                   "reason": "superseded_before_start",
                                                                   "superseded_by": rec.query_id}, uid,
                           intent_id=a.intent_id)
                    self._batch_done(prev.query_id)
            qids.append(rec.query_id)
        self.batches[bid] = _Batch(bid, uid, set(qids), qids, now)
        for q in qids:
            self.batch_of[q] = bid
        s.emit(E.MULTI_QUERY_STARTED, "multi_query", {
            "batch_id": bid, "query_ids": qids, "intent_ids": [a.intent_id for a in go],
            "dispatch": "parallel" if s.executor.max_concurrency > 1 else "sequential",
            "max_concurrency": s.executor.max_concurrency, "plan_id": plan.plan_id if plan is not None else None,
            "reused_intents": [a.intent_id for a in (plan.queries_to_reuse if plan is not None else [])]}, uid)
        for q in qids:
            rec = s.ledger.get(q)
            s.ledger.update(q, status="queued", retrieval_queued_at_ms=s.sched.now_ms())
            s.executor.submit(Job(q, rec.query_text, s.options, s._on_start, s._on_done))

    def _ledger_hit(self, terms: list[str], uid: str):
        """A completed query of an *earlier* utterance with term-Jaccard >= duplicate_jaccard (Phase 2 §9.5)."""
        t = set(terms)
        best, best_j = None, 0.0
        for r in self.s.ledger.all():
            if r.utterance_id == uid or r.status != "completed" or not r.terms:
                continue
            j = len(t & set(r.terms)) / len(t | set(r.terms))
            if j >= self.cfg.duplicate_jaccard and j > best_j:
                best, best_j = r, j
        return best

    # ------------------------------------------------------------------ completion
    def on_retrieval_done(self, qid: str, wall_ms: float) -> None:
        bid = self.batch_of.get(qid)
        if bid is None:
            return
        if self.engine is not None:
            s = self.s
            rec, es = s.ledger.get(qid), s.evidence.get(qid)
            cur = s.ledger.active_for_intent(rec.intent_id) if rec.intent_id else None
            if es is not None and rec.status == "completed" and cur is not None and cur.query_id == qid:
                self.engine.on_result(rec.intent_id, qid, es, s.sched.logical_now_ms(), rec.utterance_id)
        self.batches[bid].wall_ms[qid] = round(wall_ms, 3)
        self._batch_done(qid)

    def _batch_done(self, qid: str) -> None:
        bid = self.batch_of.get(qid)
        if bid is None:
            return
        b = self.batches[bid]
        if qid not in b.pending:
            return
        b.pending.discard(qid)
        if b.pending:
            return
        s = self.s
        recs = [s.ledger.get(q) for q in b.query_ids]
        starts = [r.retrieval_started_at_ms for r in recs if r.retrieval_started_at_ms is not None]
        ends = [r.retrieval_completed_at_ms for r in recs if r.retrieval_completed_at_ms is not None]
        s.emit(E.MULTI_QUERY_COMPLETED, "multi_query", {
            "batch_id": bid, "query_ids": b.query_ids, "statuses": {r.query_id: r.status for r in recs},
            "makespan_ms": round(max(ends) - min(starts), 3) if starts and ends else None,
            "critical_path_ms": round(max((r.retrieval_completed_at_ms - r.retrieval_started_at_ms) for r in recs
                                          if r.retrieval_started_at_ms is not None
                                          and r.retrieval_completed_at_ms is not None), 3) if ends else None,
            "wall": {"per_query_ms": b.wall_ms, "sum_ms": round(sum(b.wall_ms.values()), 3)}}, b.utterance_id)
        if not s.chunks.is_finalized(b.utterance_id):       # after finalization the final fusion follows
            self.fuse(b.utterance_id, final=False)

    # ------------------------------------------------------------------ fusion
    def _turn_needs(self, uid: str):
        """Phase 6: the utterance's own active needs + the earlier needs this turn changed (cross-turn)."""
        own = list(self.tracker.active_intents(uid))
        ids = {i.intent_id for i in own}
        extra = [self.tracker.intents[i] for i in self.turn_intents.get(uid, []) if i not in ids
                 and i in self.tracker.intents and self.tracker.intents[i].status == "ACTIVE"]
        return own + extra

    def _session_inputs(self, uid: str) -> list[IntentEvidence]:
        s = self.s
        out = []
        for it in self._turn_needs(uid):
            active = s.ledger.active_for_intent(it.intent_id)
            done = s.ledger.latest_completed(it.intent_id)
            src = done
            if done is not None and done.status == "reused" and done.reused_from:
                src = s.ledger.get(done.reused_from)
            es = s.evidence.get(src.query_id) if src else None
            status = "ok"
            if es is None:
                status = "retrieval_failed" if (active is not None and active.status == "failed") else "pending"
            ref = done or active
            out.append(IntentEvidence(intent=it, query_text=ref.query_text if ref else it.resolved_text,
                                      query_id=ref.query_id if ref else None, evidence=es, status=status,
                                      stale=bool(done is not None and active is not None
                                                 and done.query_id != active.query_id)))
        return out

    def inputs(self, uid: str) -> list[IntentEvidence]:
        s = self.s
        if self.engine is not None:
            return self._session_inputs(uid)
        out = []
        for it in self.tracker.active_intents(uid):
            active = s.ledger.active(uid, it.intent_id)
            done = s.ledger.latest_completed(it.intent_id)
            if done is None and it.intent_id in self.reuse:       # ledger hit: evidence of the earlier query
                done = s.ledger.get(self.reuse[it.intent_id])
            es = s.evidence.get(done.query_id) if done else None
            status = "ok"
            if es is None:
                status = "retrieval_failed" if (active is not None and active.status == "failed") else "pending"
            out.append(IntentEvidence(intent=it, query_text=(done or active).query_text if (done or active) else
                                      it.resolved_text, query_id=done.query_id if done else
                                      (active.query_id if active else None), evidence=es, status=status,
                                      stale=bool(done is not None and active is not None
                                                 and done.query_id != active.query_id)))
        return out

    def fuse(self, uid: str, final: bool):
        s = self.s
        inputs = self.inputs(uid)
        if not inputs:
            return None
        iset = self.tracker.current(uid)
        rerank = s.cfg.fusion.rerank if final else "none"
        if rerank != "none":
            s.emit(E.RERANK_STARTED, "fusion", {"mode": rerank, "intents": [x.intent.intent_id for x in inputs],
                                               "candidates": sum(len(x.evidence.items) for x in inputs if x.evidence)},
                   uid)
        t0 = time.perf_counter()
        ues = self.stack.fusion.fuse(s.session_id, uid, iset.version if iset else 0, inputs, rerank=rerank)
        wall = (time.perf_counter() - t0) * 1000.0
        self.fusion_wall_ms.append(wall)
        if rerank != "none":
            self.rerank_wall_ms.append(ues.timings_ms.get("rerank", 0.0))
            s.emit(E.RERANK_COMPLETED, "fusion", {"mode": ues.rerank, "warnings": ues.warnings,
                                                 "wall": {"rerank_ms": ues.timings_ms.get("rerank")}}, uid)
        s.emit(E.EVIDENCE_DEDUPLICATED, "fusion", {**ues.dedup, "final": final}, uid)
        s.emit(E.EVIDENCE_FUSED, "fusion", {
            "final": final, "unified_set_id": ues.unified_set_id, "intent_set_version": ues.intent_set_version,
            "strategy": ues.strategy, "rerank": ues.rerank, "top_k": ues.top_k,
            "items": [{"label": i.label, "evidence_id": i.evidence_id, "citation": i.citation,
                       "supporting_intents": i.supporting_intents, "supporting_queries": i.supporting_queries,
                       "selected_for": i.selected_for} for i in ues.items],
            "per_intent": [c.model_dump(mode="json") for c in ues.per_intent],
            "conflicts": [c.model_dump(mode="json") for c in ues.conflicts],
            "intent_coverage": round(sum(c.covered for c in ues.per_intent) / len(ues.per_intent), 3),
            "token_count": ues.token_count, "wall": {"fusion_ms": round(wall, 4), "stages_ms": ues.timings_ms}}, uid)
        self.unified[uid] = ues
        return ues

    def turn_payload(self, uid: str) -> dict:
        if self.engine is not None:
            return self._session_turn_payload(uid)
        s = self.s
        ues = self.fuse(uid, final=True)
        iset = self.tracker.current(uid)
        active = [s.ledger.active(uid, i.intent_id) or (s.ledger.get(self.reuse[i.intent_id])
                                                        if i.intent_id in self.reuse else None)
                  for i in (iset.intents if iset else [])]
        return {
            "sub_queries": [a.query_text for a in active if a is not None],
            "final_query_ids": {i.intent_id: a.query_id for i, a in zip(iset.intents if iset else [], active)
                                if a is not None},
            "reused_queries": {i: q for i, q in self.reuse.items() if i in {x.intent_id for x in
                                                                             (iset.intents if iset else [])}},
            "intent_set": iset.model_dump(mode="json") if iset else None,
            "intent_set_versions": len(self.tracker.versions(uid)),
            "superseded_intents": [i for i, it in self.tracker.intents.items()
                                   if it.utterance_id == uid and it.status == "SUPERSEDED"],
            "lineage": s.ledger.lineage_tree(uid),
            "unified_evidence": ues.model_dump(mode="json", exclude={"timings_ms"}) if ues is not None else None,
            "citations": [i.citation for i in ues.items] if ues is not None else [],
            "wall": {"fusion_timings_ms": ues.timings_ms if ues is not None else None,
                     "decompose_ms_total": round(sum(self.decompose_wall_ms), 4)},
        }

    def _session_turn_payload(self, uid: str) -> dict:
        s, eng = self.s, self.engine
        ues = self.fuse(uid, final=True)
        answer = eng.commit_answer(uid, s.sched.logical_now_ms())     # ANSWER_VERSION_* precede TURN_COMPLETED
        if answer is not None:
            self.answers[uid] = answer
        iset = self.tracker.current(uid)
        needs = self._turn_needs(uid)
        active = {it.intent_id: s.ledger.active_for_intent(it.intent_id) for it in needs}
        changes = [c for c in eng.changes if c.utterance_id == uid]
        plans = [p for p in eng.plans if p.utterance_id == uid]
        return {
            "sub_queries": [a.query_text for a in active.values() if a is not None],
            "final_query_ids": {i: a.query_id for i, a in active.items() if a is not None},
            "reused_queries": {a.intent_id: a.reused_query_id for p in plans for a in p.queries_to_reuse},
            "intent_set": iset.model_dump(mode="json") if iset else None,
            "intent_set_versions": len(self.tracker.versions(uid)),
            "superseded_intents": [i for i, it in self.tracker.intents.items()
                                   if it.utterance_id == uid and it.status == "SUPERSEDED"],
            "lineage": s.ledger.lineage_tree(uid),
            "unified_evidence": ues.model_dump(mode="json", exclude={"timings_ms"}) if ues is not None else None,
            "citations": [i.citation for i in ues.items] if ues is not None else [],
            "session": {
                "session_version": eng.memory.version,
                "turn_intents": [it.intent_id for it in needs],
                "changes": [{"change_id": c.change_id, "type": c.change_type, "affected": c.affected_intents,
                             "new": c.new_intents, "added": c.added_constraints, "removed": c.removed_constraints,
                             "confidence": c.confidence} for c in changes],
                "net_change_types": net_change_types(changes),
                "plans": [p.plan_id for p in plans],
                "deferred": sorted(self.deferred),
                "answer_id": answer.answer_id if answer is not None else (
                    eng.answers.get_current_answer_state().answer_id if eng.answers.get_current_answer_state()
                    else None),
                "answer_changed": answer is not None,
                "counters": dict(eng.counters)},
            "wall": {"fusion_timings_ms": ues.timings_ms if ues is not None else None,
                     "decompose_ms_total": round(sum(self.decompose_wall_ms), 4)},
        }
