"""Answer lane: grounded answers off the event loop (docs/runtime/02, 09).

Per session at most one answer task runs at a time (the engine's own state - versions, drafts, reuse base - belongs
to the lane), matching the Phase 2 statechart: one answering turn at a time, commits serialized.

* **Drafts** (extractive, verified) are requested after provisional evidence batches. Only the latest pending
  draft is kept (latest-state preference); a final request drops pending drafts and cancels a running one.
* **Finals** are generated in request order. A correction cancels the previous turn's in-flight final
  (``cancel_on_correction``, spec §7.3 barge-in): its LLM stream is closed at the next chunk.
* **Snapshot isolation.** The worker gets a snapshot of exactly what the engine reads (evidence store, claims,
  intents and constraints, conflicts), taken on the loop when the task is created, so the session can keep
  processing input while the answer is generated. Fallback retrievals during validation go to the snapshot and are
  re-applied to the live store at commit.
* **Atomic commit.** The engine's events are buffered in the worker and, with the fallback evidence, applied on the
  loop in one step - only if the request is still relevant (not cancelled, the latest draft). Otherwise the engine
  is rolled back to its checkpoint and nothing becomes visible.
* **Degradation** (deterministic, never silent): a final that fails or times out is redone extractively
  (GENERATION_DEGRADED); a validation failure of the entailment model is redone with rules-only verification
  (VALIDATION_DEGRADED). If that fails too, the previous validated answer stays current and the turn reports it.
"""

from __future__ import annotations

import copy
import time
from collections import deque
from dataclasses import dataclass, field

from streamrag.answer_state.engine import GroundingContext
from streamrag.generation.llm import CALL_CONTEXT, CallContext
from streamrag.models.events import EventType as E
from streamrag.runtime.tasks import TaskStatus, TaskType

CORRECTIONS = {"CORRECTION", "ENTITY_CHANGE", "QUESTION_CHANGE"}     # Phase 6 change types: a different request


class _TrackerView:
    """What the claim planner reads from the intent tracker, frozen at snapshot time."""

    def __init__(self, tracker) -> None:
        self.intents = copy.deepcopy(dict(tracker.intents))
        self._constraints = {iid: copy.deepcopy(tracker.constraints_for(it)) for iid, it in tracker.intents.items()}

    def constraints_for(self, intent) -> list:
        return self._constraints.get(intent.intent_id, [])


class _GraphView:
    def __init__(self, graph) -> None:
        self.claims = copy.deepcopy(dict(graph.claims))


@dataclass
class _Request:
    n: int
    uid: str
    p6: object
    draft: bool
    t_ms: float
    mode: str = "full"                    # full | extractive (generation degraded) | rules (validation degraded)
    task_id: str | None = None
    cancelled: str | None = None
    vr: list = field(default_factory=list)


class AnswerLane:
    def __init__(self, rs, coordinator) -> None:
        self.rs, self.coord = rs, coordinator
        self.engine = coordinator.grounding
        self.queue: deque[_Request] = deque()
        self.current: _Request | None = None
        self.n = 0
        self.finals: dict[str, object] = {}
        self._last_final = None
        self.stats = {"drafts_requested": 0, "drafts_coalesced": 0, "drafts_superseded": 0, "finals": 0,
                      "cancelled_by_correction": 0, "degraded": 0, "snapshot_ms": []}

    def busy(self) -> bool:
        return self.current is not None or bool(self.queue)

    def last_final(self):
        return self._last_final

    # ------------------------------------------------------------------ requests (event loop)
    def request(self, uid: str, p6, draft: bool) -> dict | None:
        self.n += 1
        req = _Request(self.n, uid, p6, draft, self.rs.session.sched.logical_now_ms())
        if draft:
            self.stats["drafts_requested"] += 1
            if any(not r.draft for r in self.queue) or (self.current is not None and not self.current.draft):
                self.stats["drafts_superseded"] += 1
                return None
            before = len(self.queue)
            self.queue = deque(r for r in self.queue if not r.draft)
            self.stats["drafts_coalesced"] += before - len(self.queue)
            self.queue.append(req)
        else:
            by = self.rs.superseding_turn(uid)
            if by is not None:
                # the turn's need is being retrieved again for a later utterance that refines it (its own query was
                # cancelled as superseded): answering now would claim "no evidence" - the later turn answers instead
                self.stats["finals_superseded"] = self.stats.get("finals_superseded", 0) + 1
                return {"status": "SUPERSEDED", "superseded_by": by, "utterance_id": uid}
            self.stats["finals"] += 1
            dropped = [r for r in self.queue if r.draft]
            self.queue = deque(r for r in self.queue if not r.draft)
            self.stats["drafts_superseded"] += len(dropped)
            cur = self.current
            if cur is not None and cur.draft:
                self._cancel(cur, "superseded_by_final")
            elif cur is not None and self.rs.rcfg.cancel_on_correction and self._is_correction(uid):
                self.stats["cancelled_by_correction"] += 1
                self._cancel(cur, "correction")
            self.queue.append(req)
        self._pump()
        return None if draft else {"status": "PENDING", "request": f"AR{req.n}", "utterance_id": uid}

    def _is_correction(self, uid: str) -> bool:
        eng = self.coord.engine
        return any(c.utterance_id == uid and c.change_type in CORRECTIONS for c in eng.changes)

    def _cancel(self, req: _Request, reason: str) -> None:
        req.cancelled = reason
        if req.task_id is not None:
            self.rs.scheduler.cancel(req.task_id, reason)

    # ------------------------------------------------------------------ execution
    def _pump(self) -> None:
        if self.current is not None or not self.queue:
            return
        req = self.queue.popleft()
        self.current = req
        t0 = time.perf_counter()
        ctx = self._snapshot(req)
        self.stats["snapshot_ms"].append((time.perf_counter() - t0) * 1000.0)
        cp = self.engine.checkpoint()
        rs = self.rs
        no_model = req.mode == "extractive" or self.engine.generator.backend is None
        # a final that needs no model call runs on the cpu pool: an overloaded LLM queue cannot block it
        ttype = TaskType.DRAFT if req.draft else TaskType.ANSWER_EXTRACTIVE if no_model else TaskType.GENERATION
        prio = rs.rcfg.priorities.draft if req.draft else rs.rcfg.priorities.final_answer
        meta = {"answer_kind": "draft" if req.draft else "final", "mode": req.mode, "request": f"AR{req.n}"}
        task = rs.new_task(ttype, prio, req.uid, rs.bus.last_event_id, meta=meta,
                           deadline_parent=rs.turn_deadline(req.uid))
        req.task_id = task.task_id
        rs.scheduler.submit(task, self._work(req, ctx, cp, task.deadline_ms),
                            lambda r: self._done(req, cp, r))

    def _snapshot(self, req: _Request) -> GroundingContext:
        coord, rs = self.coord, self.rs
        eng = coord.engine
        store = copy.deepcopy(eng.store)
        options = rs.session.options.model_copy(update={"top_k": 3})
        tracker = coord.tracker
        versions = {iid: it.version for iid, it in tracker.intents.items()}

        def vr(intent_id: str, query: str) -> list[str]:
            if intent_id not in versions:
                return []
            before = {a.evidence_id for a in store.usable(intent_id)}
            try:
                es = rs.service.retrieve(query, options)
            except Exception:                             # noqa: BLE001 - fallback retrieval is best effort
                return []
            qid = f"VR{req.n}.{len(req.vr) + 1}"
            store.add_results(intent_id, versions[intent_id], qid, es, req.t_ms, "validation_retrieval")
            req.vr.append((intent_id, versions[intent_id], qid, es))
            return [e.evidence_id for e in es.items if e.evidence_id not in before]

        return GroundingContext(_GraphView(eng.graph), store, _TrackerView(tracker),
                                copy.deepcopy(eng.answers.conflicts), req.uid, vr)

    def _work(self, req: _Request, ctx: GroundingContext, cp, deadline_ms: float | None):
        engine, rs = self.engine, self.rs
        clock = rs.runtime.clock

        def run(wctx):
            buf: list[tuple[tuple, dict]] = []
            orig_emit, orig_backend = engine.emit, engine.generator.backend
            engine.emit = lambda *a, **k: buf.append((a, k))
            engine.cancel_check = wctx.checkpoint
            remaining = None if deadline_ms is None else (deadline_ms - clock.now_ms()) / 1000.0
            tok = CALL_CONTEXT.set(CallContext(wctx.token.is_cancelled, None if remaining is None
                                               or wctx.virtual else time.monotonic() + remaining))
            aligner_mode, cons_nli, nli = engine.aligner.mode, engine.consistency.nli, engine.aligner.nli
            try:
                if req.mode == "extractive":
                    engine.generator.backend = None
                if req.mode == "rules":
                    engine.aligner.mode, engine.consistency.nli = "rules", None
                elif rs.faults is not None and nli is not None:
                    engine.aligner.nli = _FaultyNli(nli, rs.faults, wctx, rs.session_id)
                ga = engine.answer(req.p6, ctx, req.t_ms, draft=req.draft)
                return ga, buf
            except BaseException:
                engine.restore(cp)
                raise
            finally:
                engine.emit, engine.generator.backend, engine.cancel_check = orig_emit, orig_backend, None
                engine.aligner.mode, engine.consistency.nli, engine.aligner.nli = aligner_mode, cons_nli, nli
                CALL_CONTEXT.reset(tok)
        return run

    def _relevant(self, req: _Request) -> bool:
        if req.cancelled is not None:
            return False
        if req.draft:                                     # a draft is only worth showing if nothing newer exists
            return req.n == self.n and not self.rs.session.chunks.is_finalized(req.uid)
        return True

    def _done(self, req: _Request, cp, r) -> None:
        rs = self.rs
        self.current = None
        if r.ok:
            ga, buf = r.result

            def apply():
                with rs.bus.dispatch(cause=rs.task_event(r.task_id), task_id=r.task_id):
                    for a, k in buf:
                        self.coord.s.emit(*a, **k)
                    store = self.coord.engine.store
                    for intent_id, version, qid, es in req.vr:
                        store.add_results(intent_id, version, qid, es, req.t_ms, "validation_retrieval")
                    self._delta(ga)
                    if not req.draft and ga.fallback:     # the engine wrote extractively: never silent
                        rs.set_degraded("GENERATION_DEGRADED", f"{ga.answer_id}: {ga.fallback}", req.uid)
                    if not req.draft:
                        self.finals[req.uid] = ga
                        self.coord.grounded[req.uid] = ga
                        self._last_final = ga
                        if rs.adaptive is not None:             # Phase 9: verified claims -> claim cache
                            rs.adaptive.on_grounded(ga)
                        rs.emit(E.ANSWER_COMMITTED, "answer_lane",
                                {"answer_id": ga.answer_id, "version": ga.version, "status": ga.status,
                                 "partial": ga.partial, "mode": req.mode, "backend": ga.backend,
                                 "text": ga.text, "claims": [c.claim_id for c in ga.claims],
                                 "citations": ga.citations.keys(), "task_id": r.task_id,
                                 "wall": {"exec_ms": r.metadata.get("exec_ms"),
                                          "queue_wait_ms": r.metadata.get("queue_wait_ms")}}, req.uid)

            if not rs.state.commit(r, apply, relevant=lambda: self._relevant(req),
                                   describe={"answer_kind": "draft" if req.draft else "final",
                                             "request": f"AR{req.n}", "reason_detail": req.cancelled},
                                   utterance_id=req.uid, transactional=True):
                self.engine.restore(cp)
        elif r.status == TaskStatus.CANCELLED or req.cancelled is not None:
            self.engine.restore(cp)
        elif not req.draft:
            self.engine.restore(cp)
            self._degrade(req, r)
        else:
            self.engine.restore(cp)
        self._pump()
        rs.session._maybe_close()

    def _degrade(self, req: _Request, r) -> None:
        """Deterministic fallback chain for a final that failed: full -> extractive / rules -> keep last answer."""
        rs = self.rs
        err = r.error or r.status.value
        if req.mode == "full":
            nli_failed = "verification" in err or "NLI" in err
            mode = "rules" if nli_failed else "extractive"
            rs.set_degraded("VALIDATION_DEGRADED" if nli_failed else "GENERATION_DEGRADED",
                            f"answer {r.task_id} {r.status.value}: {err}", req.uid)
            self.stats["degraded"] += 1
            self.queue.appendleft(_Request(req.n, req.uid, req.p6, False, req.t_ms, mode=mode))
            return
        rs.emit(E.ERROR, "answer_lane", {"component": "answer_lane", "error_class": r.status.value,
                                         "recoverable": True, "action": "keep_previous_validated_answer",
                                         "detail": err, "task_id": r.task_id}, req.uid)

    def _delta(self, ga) -> None:
        d = ga.diff
        self.coord.s.emit(E.ANSWER_DELTA, "answer_streamer", {
            "answer_id": ga.answer_id, "version": ga.version, "status": ga.status, "partial": ga.partial,
            "parent_answer_id": ga.parent_answer_id,
            "added": [{"claim_id": c, "text": ga.claim(c).text} for c in d.added if ga.claim(c)],
            "removed": list(d.removed), "modified": [list(m) for m in d.modified], "unchanged": len(d.unchanged),
            "sections_regenerated": list(d.sections_regenerated), "sections_reused": list(d.sections_reused)},
            ga.utterance_id)


class _FaultyNli:
    """Fault-injection wrapper of the entailment model (target ``verification``)."""

    def __init__(self, inner, faults, wctx, session_id) -> None:
        self.inner, self.faults, self.wctx, self.session_id = inner, faults, wctx, session_id
        self.name = inner.name

    def predict(self, pairs):
        try:
            self.faults.hit("verification", self.wctx, self.session_id)
        except Exception as exc:                          # noqa: BLE001
            raise RuntimeError(f"verification (NLI) failure: {exc}") from exc
        return self.inner.predict(pairs)

    def __getattr__(self, item):
        return getattr(self.inner, item)
