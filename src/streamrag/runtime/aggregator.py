"""Concurrent retrieval with partial results (docs/runtime/04, 09).

``RuntimeRetrievalExecutor`` implements the executor interface the Phase 4-6 ``StreamingSession`` already uses
(submit / cancel_queued / busy), so the session logic is unchanged - only where and how retrieval runs changes:

* a hybrid query becomes two concurrent subtasks on the retrieval pool, **lexical** (BM25) and **dense** (embedding
  + vector search), then one **assemble** task (RRF fusion, dedup, optional rerank) on the cpu pool;
* each subtask result is published as RETRIEVAL_PARTIAL as soon as it arrives (``StreamingEvidenceAggregator``);
* a failed dense subtask degrades the query to lexical-only (status ``degraded``, DEGRADED_MODE_CHANGED), a failed
  lexical subtask to dense-only; both failing reports the retrieval failed - the session keeps earlier evidence;
* the assembled evidence is committed through the ``StateCoordinator``: only if the query is still the current
  version of its need (otherwise STALE_RESULT_DISCARDED - the ledger records it, active state is untouched);
* a superseded query's subtasks are cancelled: pending ones never run, running ones stop at their next checkpoint
  (``cancel_running``; cooperative).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from streamrag.errors import ModelNotAvailableError, RetrieverTimeoutError
from streamrag.models.events import EventType as E
from streamrag.models.retrieval import RetrievalRequest
from streamrag.runtime.tasks import TaskStatus, TaskType


@dataclass
class _QueryJob:
    job: object
    plan: object
    kinds: list[str]
    t0: float
    task_ids: dict[str, str] = field(default_factory=dict)
    parts: dict[str, object] = field(default_factory=dict)
    errors: dict[str, str] = field(default_factory=dict)
    exc: dict[str, BaseException] = field(default_factory=dict)
    statuses: dict[str, TaskStatus] = field(default_factory=dict)
    started: bool = False
    assemble_task: str | None = None
    cancel_reason: str | None = None
    wall_ms: float = 0.0
    done: bool = False


class StreamingEvidenceAggregator:
    """Merges the partial results of each query as they arrive; evidence assembly starts once every part is in."""

    def __init__(self) -> None:
        self.queries: dict[str, _QueryJob] = {}

    def add(self, qid: str, kind: str, status: TaskStatus, part=None, error: str | None = None,
            exc: BaseException | None = None) -> bool:
        q = self.queries[qid]
        q.statuses[kind] = status
        if part is not None:
            q.parts[kind] = part
        if error is not None:
            q.errors[kind] = error
        if exc is not None:
            q.exc[kind] = exc
        return all(k in q.statuses for k in q.kinds)

    @staticmethod
    def preview(rs, part, n: int = 3) -> list[str]:
        chunks = rs.service.bundle.chunks
        return [chunks[row].citation for row, _ in part.hits[:n]]


class RuntimeRetrievalExecutor:
    def __init__(self, rs) -> None:
        self.rs = rs                                      # the RuntimeSession
        self.backend = rs.service
        self.max_concurrency = rs.rcfg.max_concurrent_retrievals
        self.agg = StreamingEvidenceAggregator()

    # ------------------------------------------------------------------ executor interface (StreamingSession)
    def submit(self, job) -> None:
        rs = self.rs
        rec0 = rs.session.ledger.get(job.query_id)
        if rs.adaptive is not None and rec0 is not None and rec0.intent_id is not None:
            self._submit_adaptive(job, rec0)
            return
        try:
            plan = rs.service.plan(RetrievalRequest(query=job.query_text, options=job.options))
        except Exception as exc:                          # noqa: BLE001 - invalid query: reported, not raised
            job.on_start(job.query_id)
            job.on_done(job.query_id, None, exc, 0.0)
            return
        split = rs.rcfg.split_retrieval and plan.options["mode"] == "hybrid"
        kinds = ["lexical", "dense"] if split else ["retrieval"]
        q = self.agg.queries[job.query_id] = _QueryJob(job, plan, kinds, time.perf_counter())
        rec = rs.session.ledger.get(job.query_id)
        priority = rs.retrieval_priority(rec)
        cause = rs.bus.anchor("query", job.query_id)
        for kind in kinds:
            ttype = {"lexical": TaskType.LEXICAL, "dense": TaskType.DENSE}.get(kind, TaskType.RETRIEVAL)
            task = rs.new_task(ttype, priority, rec.utterance_id, cause,
                               idempotency_key=f"{rs.session_id}:{rs.state.epoch}:{kind}:{plan.digest}",
                               meta={"query_id": job.query_id, "intent_id": rec.intent_id, "kind": kind})
            q.task_ids[kind] = task.task_id
            rs.scheduler.submit(task, self._work(kind, plan), lambda r, k=kind, qid=job.query_id: self._part_done(qid, k, r))

    # ------------------------------------------------------------------ Phase 9: adaptive retrieval task
    def _submit_adaptive(self, job, rec) -> None:
        """One RETRIEVAL task runs the adaptive controller for the need (docs/architecture/13 §runtime): the session
        snapshot is taken here, on the loop; the controller runs on a retrieval worker with the task's cancellation
        checkpoint and the remaining deadline as its latency budget; its buffered events are emitted on the loop
        when the result is committed through the StateCoordinator (stale results are discarded as before)."""
        from types import SimpleNamespace
        rs = self.rs
        ad, mi = rs.adaptive, rs.session.mi
        now = rs.runtime.clock.now_ms()
        for inv in ad.sync_changes(mi.engine.changes, mi.tracker, now):
            rs.emit(E.RETRIEVAL_INVALIDATED, "adaptive_retrieval", inv, rec.utterance_id)
        view = ad.view(mi.engine, dict(rs.session.evidence), mi.tracker)
        changes = [c for c in mi.engine.changes if c.utterance_id == rec.utterance_id]
        action = SimpleNamespace(intent_id=rec.intent_id, query=SimpleNamespace(text=job.query_text))
        q = self.agg.queries[job.query_id] = _QueryJob(job, None, ["adaptive"], time.perf_counter())
        cause = rs.bus.anchor("query", job.query_id)
        task = rs.new_task(TaskType.RETRIEVAL, rs.retrieval_priority(rec), rec.utterance_id, cause,
                           idempotency_key=f"{rs.session_id}:{rs.state.epoch}:adaptive:{job.query_id}",
                           meta={"query_id": job.query_id, "intent_id": rec.intent_id, "kind": "adaptive"})
        q.task_ids["adaptive"] = task.task_id
        budget = None if task.deadline_ms is None else max(1.0, task.deadline_ms - now)
        faults, sid = rs.faults, rs.session_id

        def run(wctx):
            if faults is not None:
                faults.hit("network", wctx, sid)
            wctx.checkpoint()
            # injected dense faults apply to every dense search of the controller (as to every dense subtask of the
            # fixed path); the wait is cancellation-aware
            hook = None if faults is None else (lambda spec: faults.hit("dense", wctx, sid)
                                                if "dense" in spec.retrievers else None)
            req = ad.request(job.query_id, action, mi.tracker, view, changes, latency_budget_ms=budget,
                             checkpoint=wctx.checkpoint, now_ms=now, search_hook=hook)
            return ad.run(req)

        rs.scheduler.submit(task, run, lambda r, qid=job.query_id: self._adaptive_done(qid, r))

    def _adaptive_done(self, qid: str, r) -> None:
        rs = self.rs
        q = self.agg.queries.get(qid)
        if q is None or q.done and q.cancel_reason == "superseded_before_start":
            return
        q.wall_ms += float(r.metadata.get("exec_ms") or 0.0)
        q.statuses["adaptive"] = r.status
        if q.cancel_reason is not None or r.status == TaskStatus.CANCELLED:
            self._finish_cancelled(qid, q)
            return
        if not r.ok:
            self._fail(qid, q, r.error or r.status.value)
            return
        res = r.result
        rec = rs.session.ledger.get(qid)
        with rs.bus.dispatch(cause=rs.task_event(r.task_id), task_id=r.task_id):
            for t, payload in res.events:
                rs.emit(E[t], "adaptive_retrieval", payload, rec.utterance_id, intent_id=rec.intent_id,
                        query_id=qid)
        if res.state.stop_reason is not None and res.state.stop_reason.value == "ERROR" and not res.evidence.items:
            self._fail(qid, q, next((s["error"] for s in res.searches if s["error"]), "adaptive_error"))
            return
        bt = rs.adaptive.bridge_terms(res)
        if bt and rs.query_current(qid):
            rs.session.mi.engine.retrieval_terms[rec.intent_id] = bt
        self._commit(qid, q, r, res.evidence)

    def cancel_queued(self, qid: str) -> bool:
        """True only if none of the query's subtasks had started (Phase 4 'never ran' semantics)."""
        q = self.agg.queries.get(qid)
        if q is None or q.started or q.done:
            return False
        q.cancel_reason = "superseded_before_start"
        for tid in list(q.task_ids.values()):
            self.rs.scheduler.cancel(tid, "superseded_before_start")
        q.done = True
        return True

    def cancel_running(self, qid: str, reason: str) -> bool:
        q = self.agg.queries.get(qid)
        if q is None or q.done or not self.rs.rcfg.cancel_running:
            return False
        q.cancel_reason = reason
        hit = False
        for tid in list(q.task_ids.values()) + ([q.assemble_task] if q.assemble_task else []):
            hit = self.rs.scheduler.cancel(tid, reason) or hit
        return hit

    def is_queued(self, qid: str) -> bool:
        q = self.agg.queries.get(qid)
        return q is not None and not q.started and not q.done

    def busy(self) -> bool:
        return any(not q.done for q in self.agg.queries.values())

    # ------------------------------------------------------------------ workers (run on worker threads)
    def _work(self, kind: str, plan):
        rs = self.rs
        faults, sid, svc = rs.faults, rs.session_id, rs.service

        def run(wctx):
            if faults is not None:
                faults.hit("network", wctx, sid)
                faults.hit(kind if kind != "retrieval" else "dense", wctx, sid)
            wctx.checkpoint()
            if kind == "lexical":
                return svc.search_lexical(plan)
            if kind == "dense":
                return svc.search_dense(plan, checkpoint=wctx.checkpoint, inline=True)
            return svc.retrieve(plan.request)
        return run

    # ------------------------------------------------------------------ results (event loop)
    def on_task_started(self, qid: str) -> None:
        q = self.agg.queries.get(qid)
        if q is not None and not q.started:
            q.started = True
            q.job.on_start(qid)

    def _part_done(self, qid: str, kind: str, r) -> None:
        rs = self.rs
        q = self.agg.queries.get(qid)
        if q is None or q.done and q.cancel_reason == "superseded_before_start":
            return
        q.wall_ms += float(r.metadata.get("exec_ms") or 0.0)
        exc = None
        if r.status == TaskStatus.FAILED and r.error and r.error.startswith(("ModelNotAvailableError",
                                                                             "RetrieverTimeoutError")):
            exc = (ModelNotAvailableError if r.error.startswith("ModelNotAvailableError")
                   else RetrieverTimeoutError)(r.error.split(": ", 1)[-1])
        all_in = self.agg.add(qid, kind, r.status, r.result if r.ok else None, None if r.ok else
                              (r.error or r.status.value), exc)
        if r.ok and kind != "retrieval":
            rs.emit(E.RETRIEVAL_PARTIAL, "evidence_aggregator",
                    {"query_id": qid, "task_id": r.task_id, "kind": kind, "hits": len(r.result.hits),
                     "top": StreamingEvidenceAggregator.preview(rs, r.result),
                     "parts_done": sorted(k for k in q.statuses), "parts_total": len(q.kinds),
                     "wall": {"exec_ms": r.metadata.get("exec_ms")}}, rs.utt_of(qid),
                    query_id=qid, causation_id=rs.task_event(r.task_id))
        if not all_in:
            return
        if q.cancel_reason is not None or any(s == TaskStatus.CANCELLED for s in q.statuses.values()):
            self._finish_cancelled(qid, q)
            return
        if kind == "retrieval":
            if r.ok:
                self._commit(qid, q, r, r.result)
            else:
                self._fail(qid, q, r.error or r.status.value)
            return
        ok = {k for k, s in q.statuses.items() if s == TaskStatus.COMPLETED}
        if not ok:
            self._fail(qid, q, "; ".join(f"{k}: {e}" for k, e in sorted(q.errors.items())))
            return
        if "dense" not in ok:
            rs.set_degraded("RETRIEVAL_DEGRADED", f"dense failed for {qid}: {q.errors.get('dense')}",
                            rs.utt_of(qid))
        if "lexical" not in ok:
            rs.set_degraded("RETRIEVAL_DEGRADED", f"lexical failed for {qid}: {q.errors.get('lexical')}",
                            rs.utt_of(qid))
        self._assemble(qid, q)

    def _assemble(self, qid: str, q: _QueryJob) -> None:
        rs = self.rs
        rec = rs.session.ledger.get(qid)
        lexical, dense = q.parts.get("lexical"), q.parts.get("dense")
        dense_exc = q.exc.get("dense") or (None if dense is not None else
                                           ModelNotAvailableError(q.errors.get("dense", "dense failed")))
        stage_ms = (time.perf_counter() - q.t0) * 1000.0
        plan = q.plan
        lex_failed = lexical is None

        def run(wctx):
            wctx.checkpoint()
            es = rs.service.assemble(plan, lexical, dense, dense_exc, stage_ms=stage_ms)
            if lex_failed:
                es.trace.warnings.append("lexical_failed_dense_only")
                es.trace.status = "degraded"
            return es

        task = rs.new_task(TaskType.ASSEMBLE, rs.retrieval_priority(rec), rec.utterance_id,
                           rs.task_event(q.task_ids.get("dense") or q.task_ids.get("lexical")),
                           meta={"query_id": qid, "intent_id": rec.intent_id, "kind": "assemble"})
        q.assemble_task = task.task_id
        rs.scheduler.submit(task, run, lambda r: self._assembled(qid, q, r))

    def _assembled(self, qid: str, q: _QueryJob, r) -> None:
        if r.status == TaskStatus.CANCELLED or q.cancel_reason is not None:
            self._finish_cancelled(qid, q)
        elif r.ok:
            self._commit(qid, q, r, r.result)
        else:
            self._fail(qid, q, r.error or r.status.value)

    def _commit(self, qid: str, q: _QueryJob, r, es) -> None:
        rs = self.rs
        q.done = True
        rec = rs.session.ledger.get(qid)
        wall = (time.perf_counter() - q.t0) * 1000.0

        def apply():
            with rs.bus.dispatch(cause=rs.task_event(r.task_id), task_id=r.task_id):
                q.job.on_done(qid, es, None, wall)

        applied = rs.state.commit(r, apply, relevant=lambda: rs.query_current(qid),
                                  describe={"query_id": qid, "superseded_by": rec.superseded_by},
                                  utterance_id=rec.utterance_id)
        if not applied and r.epoch == rs.state.epoch:
            # reconcile: the ledger records the completed (stale) query; fusion / claims ignore stale queries
            with rs.bus.dispatch(cause=rs.task_event(r.task_id), task_id=r.task_id):
                q.job.on_done(qid, es, None, wall)

    def _fail(self, qid: str, q: _QueryJob, error: str) -> None:
        q.done = True
        self.rs.set_degraded("RETRIEVAL_DEGRADED", f"retrieval failed for {qid}: {error}", self.rs.utt_of(qid))
        wall = (time.perf_counter() - q.t0) * 1000.0
        if not q.started:
            q.started = True
            q.job.on_start(qid)
        q.job.on_done(qid, None, RetrieverTimeoutError(error) if "TIMED_OUT" in error or "deadline" in error
                      else RuntimeError(error), wall)

    def _finish_cancelled(self, qid: str, q: _QueryJob) -> None:
        if q.done and q.cancel_reason == "superseded_before_start":
            return
        q.done = True
        if self.rs.session.ledger.get(qid).status in ("queued", "in_flight"):
            self.rs.session.on_cancelled(qid, q.wall_ms, q.cancel_reason or "cancelled")
