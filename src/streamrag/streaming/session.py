"""StreamingSession: one session's streaming logic (single-threaded; spec §7 L1 LISTENING -> FINALIZING).

Flow per accepted chunk:
  CHUNK_RECEIVED -> TRANSCRIPT_UPDATED -> RETRIEVAL_DECISION -> [QUERY_UPDATED -> (RETRIEVAL_CANCELLED of a
  queued superseded query) -> RETRIEVAL_STARTED ... RETRIEVAL_COMPLETED]  |  [RETRIEVAL_SKIPPED]
Timers: a *quiet* tick after ``stability_quiet_ms`` without new words re-runs the controller; an *endpoint*
timeout finalizes the utterance if no UTTERANCE_END arrives. On UTTERANCE_END: UTTERANCE_FINALIZED, a final
controller tick, and TURN_COMPLETED once no retrieval for the utterance is queued/in flight.

Phase 4 scope: single active query, retrieval output ends at EvidenceSet. Phase 5: with ``multi_intent.enabled``
the controller decision is a gate and a ``MultiIntentCoordinator`` decomposes the utterance into intents, retrieves
per intent (delta retrieval) and fuses the evidence into a UnifiedEvidenceSet. No answer generation.
"""

from __future__ import annotations

import time

from streamrag.config.settings import StreamRagConfig
from streamrag.controller.models import ControllerInput
from streamrag.ledger.ledger import QueryLedger
from streamrag.models.events import EventType as E
from streamrag.models.events import SessionEnd, SessionStart, TranscriptChunk, UtteranceEnd
from streamrag.models.evidence import EvidenceSet
from streamrag.models.retrieval import RetrievalOptions
from streamrag.streaming.chunk_manager import TranscriptChunkManager
from streamrag.streaming.clock import PRIORITY_TIMER
from streamrag.streaming.events import EventBus
from streamrag.streaming.executor import Job
from streamrag.streaming.metrics import utterance_stats


class StreamingSession:
    def __init__(self, session_id: str, cfg: StreamRagConfig, policy, scheduler, executor, bus: EventBus,
                 meta: dict | None = None, intent_stack=None) -> None:
        self.session_id, self.cfg, self.policy = session_id, cfg, policy
        self.sched, self.executor, self.bus = scheduler, executor, bus
        self.meta = meta or {}
        self.chunks = TranscriptChunkManager(session_id)
        self.ledger = QueryLedger(session_id)
        self.evidence: dict[str, EvidenceSet] = {}
        self.prev_terms: dict[str, list[str]] = {}
        self.timer_token: dict[str, int] = {}
        self.pending_turn: set[str] = set()
        self.closing = False
        self.closed = False
        self.started = False
        self.controller_wall_ms: list[float] = []
        self.chunk_wall_ms: list[float] = []
        self.options = RetrievalOptions(mode=cfg.streaming.retrieval_mode, top_k=cfg.streaming.top_k,
                                        rerank=cfg.streaming.rerank)
        self.mi = None
        if cfg.multi_intent.enabled:
            if intent_stack is None:
                raise ValueError("multi_intent.enabled needs an IntentStack (decomposer, query builder, fusion)")
            from streamrag.multi_retrieval.coordinator import MultiIntentCoordinator
            self.mi = MultiIntentCoordinator(self, intent_stack)
            self.options = self.options.model_copy(update={"top_k": cfg.multi_intent.max_candidates_per_intent})

    # ------------------------------------------------------------------ helpers
    def _start_ms(self, uid: str | None) -> float | None:
        return self.chunks.utterances[uid].start_ms if uid in self.chunks.utterances else None

    def emit(self, type_: E, component: str, payload: dict, uid: str | None = None, intent_id: str | None = None,
             query_id: str | None = None):
        if query_id is None and isinstance(payload.get("query_id"), str):
            query_id = payload["query_id"]
        return self.bus.emit(type_, component, payload, uid, self._start_ms(uid), intent_id, query_id)

    def _error(self, component: str, error_class: str, detail: str, action: str, uid: str | None = None,
               recoverable: bool = True, input_event=None) -> None:
        payload = {"component": component, "error_class": error_class, "recoverable": recoverable, "action": action,
                   "detail": detail}
        if input_event is not None:   # inputs that only produced an error stay replayable
            payload["input_event"] = input_event.model_dump(mode="json")
        self.emit(E.ERROR, component, payload, uid)

    # ------------------------------------------------------------------ input dispatch
    def handle_input(self, ev) -> None:
        if isinstance(ev, SessionStart):
            self._session_started(ev)
        elif isinstance(ev, TranscriptChunk):
            self._on_chunk(ev)
        elif isinstance(ev, UtteranceEnd):
            self._on_utterance_end(ev)
        elif isinstance(ev, SessionEnd):
            self._on_session_end(ev)

    def _session_started(self, ev: SessionStart | None) -> None:
        if self.started:
            return
        self.started = True
        payload = {"input": ev.payload.model_dump(mode="json") if ev else None, "policy": self.policy.name,
                   "mode": self.sched.mode, "multi_intent": self.mi is not None,
                   "session_mode": bool(self.mi is not None and self.mi.engine is not None), **self.meta}
        self.emit(E.SESSION_STARTED, "session", payload)

    def _on_chunk(self, ev: TranscriptChunk) -> None:
        t0 = time.perf_counter()
        self._session_started(None)
        uid, p = ev.utterance_id, ev.payload
        open_uid = self.chunks.open_utterance()
        if open_uid is not None and open_uid != uid and not self.chunks.is_finalized(open_uid):
            self.finalize(open_uid, "implicit_new_utterance", source="implicit")
        res = self.chunks.append_chunk(ev, self.sched.now_ms())
        self.emit(E.CHUNK_RECEIVED, "chunk_manager",
                  {"input": p.model_dump(mode="json"), "status": res.status, "missing_indices": res.missing}, uid)
        if res.status.startswith("rejected"):
            self._error("chunk_manager", "ChunkRejected", f"chunk {p.chunk_index} {res.status}", "ignored", uid)
            self.chunk_wall_ms.append((time.perf_counter() - t0) * 1000.0)
            return
        if not res.changed:
            self.chunk_wall_ms.append((time.perf_counter() - t0) * 1000.0)
            return
        self.emit(E.TRANSCRIPT_UPDATED, "chunk_manager",
                  {"transcript": res.transcript, "n_chunks": len(self.chunks.state(uid).chunks),
                   "has_gaps": bool(res.missing), "revisions": self.chunks.state(uid).revisions}, uid)
        self.decide(uid, "chunk", trigger_chunk=p.chunk_index, has_gaps=bool(res.missing))
        token = self.timer_token.get(uid, 0) + 1
        self.timer_token[uid] = token
        now = self.sched.logical_now_ms()
        self.sched.call_at(now + self.cfg.controller.stability_quiet_ms, PRIORITY_TIMER, self._on_timer, "quiet", uid, token)
        self.sched.call_at(now + self.cfg.controller.endpoint_timeout_ms, PRIORITY_TIMER, self._on_timer, "endpoint", uid, token)
        self.chunk_wall_ms.append((time.perf_counter() - t0) * 1000.0)

    def _on_timer(self, kind: str, uid: str, token: int) -> None:
        if self.timer_token.get(uid) != token or self.chunks.is_finalized(uid):
            return   # superseded by a newer chunk, or utterance already ended
        if kind == "quiet":
            self.decide(uid, "stability_timer", quiet=True)
        else:
            self.finalize(uid, "timeout", source="timeout")

    def _on_utterance_end(self, ev: UtteranceEnd) -> None:
        uid = ev.utterance_id
        if uid not in self.chunks.utterances:
            self.chunks.ensure_utterance(uid, self.sched.now_ms() - ev.payload.timestamp_s * 1000.0)
        if self.chunks.is_finalized(uid):
            self._error("chunk_manager", "DuplicateUtteranceEnd", f"utterance {uid} already finalized", "ignored", uid,
                        input_event=ev)
            return
        self.finalize(uid, ev.payload.reason, source="input", end_payload=ev.payload.model_dump(mode="json"))

    def _on_session_end(self, ev: SessionEnd) -> None:
        for uid in list(self.chunks.order):
            if not self.chunks.is_finalized(uid):
                self.finalize(uid, "session_end", source="session_end")
        self.closing = True
        self.chunks.close()
        self._session_end_payload = ev.payload.model_dump(mode="json")
        self._maybe_close()

    # ------------------------------------------------------------------ finalize / complete
    def finalize(self, uid: str, reason: str, source: str, end_payload: dict | None = None) -> None:
        now = self.sched.now_ms()
        if not self.chunks.finalize_utterance(uid, now, reason):
            return
        st = self.chunks.state(uid)
        self.emit(E.UTTERANCE_FINALIZED, "chunk_manager",
                  {"reason": reason, "source": source, "input": end_payload, "transcript": self.chunks.get_current_transcript(uid),
                   "n_chunks": len(st.chunks), "missing_indices": self.chunks.missing_indices(uid)}, uid)
        self.decide(uid, "utterance_end")
        self.pending_turn.add(uid)
        self._maybe_complete(uid)

    def _maybe_complete(self, uid: str) -> None:
        if uid not in self.pending_turn:
            return
        if any(r.status in ("queued", "in_flight") for r in self.ledger.for_utterance(uid)):
            return
        self.pending_turn.discard(uid)
        if self.mi is not None:
            mi_payload = self.mi.turn_payload(uid)                 # final fusion events precede TURN_COMPLETED
            stats = utterance_stats(self.bus.events).get(uid, {})
            self.emit(E.TURN_COMPLETED, "session", {
                "retrieval_events": [{"timestamp_s": r["start_s"], "query": r["query"], "trigger": r["trigger"],
                                      "query_id": r["query_id"], "intent_id": r.get("intent_id")}
                                     for r in stats.get("retrievals", []) if r["start_s"] is not None],
                **{k: v for k, v in mi_payload.items() if k != "answer"},
                "metrics": {k: v for k, v in stats.items() if k not in ("retrievals", "decisions_list")},
                "answer": mi_payload.get("answer"),
            }, uid)
            self._maybe_close()
            return
        active = self.ledger.active(uid)
        ev_set = self.evidence.get(active.query_id) if active else None
        stats = utterance_stats(self.bus.events).get(uid, {})
        self.emit(E.TURN_COMPLETED, "session", {
            "retrieval_events": [{"timestamp_s": r["start_s"], "query": r["query"], "trigger": r["trigger"],
                                  "query_id": r["query_id"]} for r in stats.get("retrievals", []) if r["start_s"] is not None],
            "sub_queries": [active.query_text] if active else [],
            "final_query_id": active.query_id if active else None,
            "citations": [e.citation for e in ev_set.items] if ev_set is not None else [],
            "evidence": self.ledger.evidence_view(uid),
            "metrics": {k: v for k, v in stats.items() if k not in ("retrievals", "decisions_list")},
            "answer": None,
        }, uid)
        self._maybe_close()

    def _maybe_close(self) -> None:
        if self.closing and not self.closed and not self.pending_turn and not self.executor.busy():
            self.closed = True
            self.emit(E.SESSION_CLOSED, "session", {"input": getattr(self, "_session_end_payload", None),
                                                    "utterances": len(self.chunks.order),
                                                    "queries": len(self.ledger.all()),
                                                    "ledger": self.ledger.summary()})

    # ------------------------------------------------------------------ controller
    def decide(self, uid: str, tick: str, trigger_chunk: int | None = None, quiet: bool = False,
               has_gaps: bool = False) -> None:
        now = self.sched.logical_now_ms()   # stream time: decisions depend on the input, not on processing delay
        inp = ControllerInput(self.session_id, uid, self.chunks.get_current_transcript(uid), tick, now, trigger_chunk,
                              self.prev_terms.get(uid, []), quiet, has_gaps)
        t0 = time.perf_counter()
        decision, query, terms = self.policy.decide(inp, self.ledger, len(self.ledger.in_flight()))
        ctl_ms = (time.perf_counter() - t0) * 1000.0
        self.controller_wall_ms.append(ctl_ms)
        payload = decision.model_dump(mode="json")
        payload["wall"] = {"controller_ms": round(ctl_ms, 4)}
        self.emit(E.RETRIEVAL_DECISION, "retrieval_controller", payload, uid)
        self.prev_terms[uid] = terms
        if self.mi is not None:
            if decision.decision == "SKIP":
                self.emit(E.RETRIEVAL_SKIPPED, "retrieval_controller",
                          {"reason": decision.reason, "skip_kind": decision.skip_kind, "ledger_ref": decision.ledger_ref,
                           "tick": tick}, uid)
            self.mi.on_decision(uid, tick, decision, now, trigger_chunk)
            return
        if decision.decision == "SKIP":
            self.emit(E.RETRIEVAL_SKIPPED, "retrieval_controller",
                      {"reason": decision.reason, "skip_kind": decision.skip_kind, "ledger_ref": decision.ledger_ref,
                       "tick": tick}, uid)
            return
        if decision.decision != "RETRIEVE":
            return
        rec, prev = self.ledger.create(uid, query.text, inp.transcript, query.spans, terms, now, decision.trigger,
                                       trigger_chunk, tick, decision.reason)
        self.emit(E.QUERY_UPDATED, "query_ledger",
                  {"query_id": rec.query_id, "version": rec.version, "query_text": rec.query_text,
                   "supersedes": rec.supersedes, "relation": rec.relation, "lineage_root": rec.lineage_root,
                   "trigger": rec.trigger, "trigger_chunk": trigger_chunk, "tick": tick,
                   "source_spans": [list(s) for s in rec.source_spans], "removed_tokens": query.removed,
                   "superseded_status": prev.status if prev else None}, uid)
        if prev is not None and prev.status == "queued" and self.cfg.controller.cancel_superseded == "queued_only":
            if self.executor.cancel_queued(prev.query_id):
                self.ledger.update(prev.query_id, status="cancelled", retrieval_status="cancelled")
                self.emit(E.RETRIEVAL_CANCELLED, "query_ledger",
                          {"query_id": prev.query_id, "reason": "superseded_before_start", "superseded_by": rec.query_id}, uid)
        self.ledger.update(rec.query_id, status="queued", retrieval_queued_at_ms=self.sched.now_ms())   # wall: queue wait
        self.executor.submit(Job(rec.query_id, rec.query_text, self.options, self._on_start, self._on_done))

    # ------------------------------------------------------------------ retrieval callbacks
    def _on_start(self, qid: str) -> None:
        now = self.sched.now_ms()
        rec = self.ledger.update(qid, status="in_flight", retrieval_started_at_ms=now)
        self.emit(E.RETRIEVAL_STARTED, "async_retriever",
                  {"query_id": qid, "query": rec.query_text, "trigger": rec.trigger,
                   "queue_wait_ms": round(now - (rec.retrieval_queued_at_ms or now), 3),
                   "retrieval_mode": self.options.mode}, rec.utterance_id, intent_id=rec.intent_id)

    def _on_done(self, qid: str, es: EvidenceSet | None, err: Exception | None, wall_ms: float) -> None:
        now = self.sched.now_ms()
        rec = self.ledger.get(qid)
        uid = rec.utterance_id
        latency = round(now - (rec.retrieval_started_at_ms or now), 3)
        if err is not None or es is None:
            status = "timeout" if err.__class__.__name__ == "RetrieverTimeoutError" else "error"
            self.ledger.update(qid, status="failed", retrieval_status=status, retrieval_completed_at_ms=now,
                               stale_at_completion=rec.stale, error=f"{err.__class__.__name__}: {err}")
            self.emit(E.RETRIEVAL_COMPLETED, "async_retriever",
                      {"query_id": qid, "status": status, "stale": rec.stale, "superseded_by": rec.superseded_by,
                       "evidence_ids": [], "latency_ms": latency, "wall": {"measured_ms": round(wall_ms, 3)}}, uid,
                      intent_id=rec.intent_id)
            self._error("async_retriever", err.__class__.__name__, str(err), "keep_previous_evidence", uid)
        else:
            self.evidence[qid] = es
            ids = [e.evidence_id for e in es.items]
            self.ledger.update(qid, status="completed", retrieval_status=es.trace.status, retrieval_completed_at_ms=now,
                               stale_at_completion=rec.stale, evidence_set_id=es.evidence_set_id, evidence_ids=ids,
                               citations=[e.citation for e in es.items])
            self.emit(E.RETRIEVAL_COMPLETED, "async_retriever",
                      {"query_id": qid, "status": es.trace.status, "stale": rec.stale, "superseded_by": rec.superseded_by,
                       "evidence_set_id": es.evidence_set_id, "evidence_ids": ids,
                       "citations": [e.citation for e in es.items[:5]], "warnings": es.trace.warnings,
                       "latency_ms": latency,
                       "wall": {"measured_ms": round(wall_ms, 3), "stages_ms": es.trace.timings_ms}}, uid,
                      intent_id=rec.intent_id)
        if self.mi is not None:
            self.mi.on_retrieval_done(qid, wall_ms)
        self._maybe_complete(uid)
        self._maybe_close()
