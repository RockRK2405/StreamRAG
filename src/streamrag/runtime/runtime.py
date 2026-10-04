"""StreamingRuntime: the Phase 8 asynchronous orchestration layer (docs/runtime/02, ADR-018).

One runtime serves many sessions on one asyncio event loop (``mode="realtime"``) or on a deterministic virtual
clock (``mode="virtual"``: same code, modelled task latencies, exactly replayable).

Per session (``RuntimeSession``) - an actor that owns all of the session's state:
    input queue (bounded, coalescing) -> session lane (Phase 4-7 StreamingSession logic, on the loop)
        -> retrieval subtasks (RuntimeRetrievalExecutor) and answer tasks (AnswerLane) on the shared scheduler
        -> results committed back through the StateCoordinator (version / relevance checked)
        -> event log (RuntimeEventBus) -> AnswerStreamer (ordered, bounded subscribers)

Shared (read-only or bounded): the index + retrieval service, the NLI model, the LLM backend, the TaskScheduler with
its worker pools, the cancellation / retry / timeout managers, telemetry.
"""

from __future__ import annotations

import asyncio
import dataclasses
import time
from collections.abc import AsyncIterator
from pathlib import Path

from streamrag.config.settings import StreamRagConfig, config_hash
from streamrag.generation.llm import CALL_CONTEXT
from streamrag.models.events import EventType as E
from streamrag.models.events import (SessionEnd, SessionEndPayload, SessionStart, TranscriptChunk,
                                     TranscriptChunkPayload, UtteranceEnd, UtteranceEndPayload)
from streamrag.runtime.aggregator import RuntimeRetrievalExecutor
from streamrag.runtime.answers import CORRECTIONS, AnswerLane
from streamrag.runtime.backpressure import InputQueue, UnboundedInputQueue
from streamrag.runtime.cancellation import CancellationManager
from streamrag.runtime.events import RuntimeEventBus
from streamrag.runtime.faults import FaultInjector, FaultyLLM
from streamrag.runtime.retry import RetryManager
from streamrag.runtime.scheduler import TaskScheduler, ThreadRunner, VirtualRunner
from streamrag.runtime.state import StateCoordinator
from streamrag.runtime.streamer import AnswerStreamer
from streamrag.runtime.tasks import Task, TaskType
from streamrag.runtime.telemetry import TelemetryManager, turn_milestones
from streamrag.runtime.timeouts import TimeoutManager
from streamrag.streaming.clock import PRIORITY_INPUT, RealtimeScheduler, VirtualScheduler
from streamrag.streaming.session import StreamingSession


class RuntimeCapacityError(RuntimeError):
    """max_concurrent_sessions reached."""


class RuntimeClosedError(RuntimeError):
    """The runtime is shutting down / shut down."""


class RetryingLLM:
    """Retries transient LLM failures (connection, timeout, rate limit) with the runtime's bounded backoff; the
    wait honours the call's cancellation token and deadline. Deterministic failures are returned as they are."""

    def __init__(self, inner, retry: RetryManager, virtual: bool) -> None:
        self.inner, self.retry, self.virtual = inner, retry, virtual
        self.name, self.model = inner.name, getattr(inner, "model", "unknown")

    def complete(self, messages, schema=None):
        attempt = 0
        while True:
            r = self.inner.complete(messages, schema)
            if r.ok or not self.retry.should_retry(r.error, attempt):
                if attempt:
                    r.meta = {**r.meta, "attempts": attempt + 1}
                return r
            delay = self.retry.delay_ms(attempt) / 1000.0
            attempt += 1
            if self.virtual:
                continue
            ctx = CALL_CONTEXT.get()
            end = time.monotonic() + delay
            while time.monotonic() < end:
                if ctx.is_cancelled():
                    return r
                time.sleep(min(0.005, max(0.0, end - time.monotonic())))


class SessionClock:
    """The runtime clock as one session sees it: a timer runs inside a dispatch of the session's event bus, caused
    by the event that armed it, and is dropped if the session was reset or cancelled meanwhile."""

    def __init__(self, base, rs: "RuntimeSession") -> None:
        self.base, self.rs = base, rs
        self.mode = base.mode

    def now_ms(self) -> float:
        return self.base.now_ms()

    def logical_now_ms(self) -> float:
        return self.base.logical_now_ms()

    def call_at(self, t_ms: float, priority: int, fn, *args) -> None:
        rs, epoch, cause = self.rs, self.rs.state.epoch, self.rs.bus.last_event_id

        def run() -> None:
            if not rs.alive or rs.state.epoch != epoch:
                return
            with rs.bus.dispatch(cause):
                fn(*args)
        self.base.call_at(t_ms, priority, run)


class RuntimeSession:
    def __init__(self, runtime: "StreamingRuntime", session_id: str, trace_path: Path | None = None) -> None:
        self.runtime, self.session_id = runtime, session_id
        self.rcfg = runtime.rcfg
        self.cfg = runtime.session_cfg
        self.service = runtime.stack.service
        self.scheduler = runtime.scheduler
        self.faults = runtime.faults
        self.state = StateCoordinator(session_id)
        self.bus = RuntimeEventBus(session_id, runtime.clock.now_ms, trace_path, version_fn=lambda: self.state.version)
        self.state.emit = self.emit
        self.streamer = AnswerStreamer(self.rcfg.queues.output, self.rcfg.output_max_hold_ms, runtime.clock.now_ms)
        self.bus.subscribers.append(self.streamer.on_event)
        self.bus.subscribers.append(self._watch)
        self.clock = SessionClock(runtime.clock, self)
        self.inputs = UnboundedInputQueue() if runtime.unbounded_inputs else InputQueue(self.rcfg.queues.input)
        self.degraded: dict[str, str] = {}
        self.first_input_wall: dict[str, float] = {}
        self.utt_t0: dict[str, float] = {}
        self.chunk_counter: dict[str, int] = {}
        self.n_inputs = 0
        self.alive = True
        self.cancelled_reason: str | None = None
        self._drain_armed = False
        self._task_events: dict[str, str] = {}
        self._arrivals: dict[int, float] = {}
        self.lane: AnswerLane | None = None
        self._build()

    # ------------------------------------------------------------------ construction / reset
    def _build(self) -> None:
        rt = self.runtime
        self.adaptive = None                    # Phase 9: set below when adaptive retrieval is enabled
        self.executor = RuntimeRetrievalExecutor(self)
        stack = None
        if self.cfg.multi_intent.enabled:
            stack = rt.stack.intent_stack
            if stack.grounding is not None:
                stack = dataclasses.replace(stack, grounding=stack.grounding.with_backend(rt.llm_for(self.session_id)))
        meta = {"config_hash": config_hash(self.cfg), "index_content_hash": rt.stack.index_hash,
                "runtime": {"mode": rt.mode, "epoch": self.state.epoch, "faults": rt.faults.describe()
                            if rt.faults else []}}
        self.session = StreamingSession(self.session_id, self.cfg, rt.stack.policy, self.clock, self.executor,
                                        self.bus, meta, stack)
        self.session.busy_hooks.append(lambda: len(self.inputs) > 0)
        mi = self.session.mi
        if mi is not None and mi.grounding is not None:
            self.lane = AnswerLane(self, mi)
            mi.answer_lane = self.lane
            self.session.busy_hooks.append(self.lane.busy)
        if mi is not None and mi.engine is not None:
            mem = mi.engine.memory
            self.state.checkpoint = mem.create_snapshot
            self.state.restore = mem.restore_snapshot
            if self.cfg.adaptive_retrieval.enabled:
                from streamrag.adaptive.integration import SessionAdaptiveRetriever
                self.adaptive = SessionAdaptiveRetriever(self.service, self.cfg, parallel=False, inline_dense=True)

    # ------------------------------------------------------------------ helpers used by executor / lane
    def emit(self, type_, component: str, payload: dict, uid: str | None = None, intent_id: str | None = None,
             query_id: str | None = None, causation_id: str | None = None):
        start = self.session._start_ms(uid) if uid is not None else None
        return self.bus.emit(type_, component, payload, uid, start, intent_id, query_id, causation_id=causation_id)

    def utt_of(self, qid: str) -> str:
        return self.session.ledger.get(qid).utterance_id

    def task_event(self, task_id: str | None) -> str | None:
        return self._task_events.get(task_id) if task_id else None

    def turn_deadline(self, uid: str | None) -> float | None:
        return self.runtime.timeouts.turn_deadline(self.utt_t0[uid]) if uid in self.utt_t0 else None

    def new_task(self, ttype: TaskType, priority: int, uid: str | None, causation: str | None,
                 idempotency_key: str | None = None, meta: dict | None = None,
                 deadline_parent: float | None = None) -> Task:
        now = self.runtime.clock.now_ms()
        parent = deadline_parent if deadline_parent is not None else self.turn_deadline(uid)
        dl = self.runtime.timeouts.deadline_for(ttype, now, parent)
        return Task(task_id=self.scheduler.new_task_id(), task_type=ttype, session_id=self.session_id,
                    priority=priority, state_version=self.state.version, epoch=self.state.epoch, created_at_ms=now,
                    correlation_id=f"{self.session_id}/{uid or 'session'}", causation_id=causation,
                    deadline_ms=dl.at_ms, idempotency_key=idempotency_key,
                    meta={**(meta or {}), "utterance_id": uid, "deadline_limited_by": dl.limited_by})

    def retrieval_priority(self, rec) -> int:
        """HIGH for a finalized utterance or a need without evidence yet, MEDIUM for early (provisional) retrieval
        of a need that already has evidence; aging in the scheduler prevents starvation (docs/runtime/03)."""
        p = self.rcfg.priorities
        final = self.session.chunks.is_finalized(rec.utterance_id)
        has_ev = False
        mi = self.session.mi
        if rec.intent_id is not None and mi is not None and mi.engine is not None:
            has_ev = bool(mi.engine.store.usable(rec.intent_id))
        elif rec.intent_id is not None:
            has_ev = any(r.status == "completed" for r in self.session.ledger.for_intent(rec.intent_id))
        return p.retrieval_final if (final or not has_ev) else p.retrieval_provisional

    def superseding_turn(self, uid: str) -> str | None:
        """A later utterance whose (pending) query now carries one of this turn's needs, if any."""
        mi = self.session.mi
        if mi is None:
            return None
        for it in mi._turn_needs(uid):
            act = self.session.ledger.active_for_intent(it.intent_id)
            if act is not None and act.utterance_id != uid and act.status in ("queued", "in_flight"):
                return act.utterance_id
        return None

    def query_current(self, qid: str) -> bool:
        rec = self.session.ledger.get(qid)
        return not rec.stale and rec.superseded_by is None and rec.status not in ("cancelled",)

    def set_degraded(self, mode: str, reason: str, uid: str | None) -> None:
        first = mode not in self.degraded
        self.degraded[mode] = reason
        if first:
            self.emit(E.DEGRADED_MODE_CHANGED, "runtime", {"mode": mode, "active": sorted(self.degraded),
                                                           "reason": reason}, uid)

    def on_task_event(self, task: Task, kind: str, payload: dict) -> None:
        if not self.alive:
            return
        if kind == "callback_error":                     # a result handler raised: reported, never silent
            self.emit(E.ERROR, "task_scheduler", {"component": "task_scheduler", "error_class": "ResultHandlerError",
                                                  "recoverable": False, "action": "reported", "task_id": task.task_id,
                                                  "detail": payload.get("error")})
            self.runtime.telemetry.counters["callback_errors"] += 1
            return
        etype = {"scheduled": E.TASK_SCHEDULED, "started": E.TASK_STARTED, "completed": E.TASK_COMPLETED,
                 "failed": E.TASK_FAILED, "cancelled": E.TASK_CANCELLED, "timed_out": E.TASK_TIMED_OUT,
                 "retried": E.TASK_RETRIED, "rejected": E.TASK_REJECTED, "superseded": E.TASK_CANCELLED}.get(kind)
        if etype is None:
            return
        cause = task.causation_id if kind == "scheduled" else self._task_events.get(task.task_id)
        uid = task.meta.get("utterance_id")
        ev = self.emit(etype, "task_scheduler", {**task.summary(), **payload}, uid if uid in self.session.chunks.utterances else None,
                       query_id=task.meta.get("query_id"), causation_id=cause)
        self._task_events[task.task_id] = ev.event_id
        if kind == "started" and task.meta.get("query_id") is not None and task.task_type != TaskType.ASSEMBLE:
            with self.bus.dispatch(cause=ev.event_id, task_id=task.task_id):
                self.executor.on_task_started(task.meta["query_id"])
        if kind in ("timed_out",) and task.task_type in (TaskType.LEXICAL, TaskType.DENSE, TaskType.RETRIEVAL):
            self.emit(E.ERROR, "task_scheduler", {"component": "task_scheduler", "error_class": "TaskTimedOut",
                                                  "recoverable": True, "action": "continue_with_available_evidence",
                                                  "detail": f"{task.task_type.value} {task.task_id}",
                                                  "task_id": task.task_id}, uid if uid in self.session.chunks.utterances else None)

    def _watch(self, ev) -> None:
        """Mid-stream barge-in: a correction cancels the previous turn's in-flight answer immediately."""
        if ev.type == E.CONTEXT_CHANGE_DETECTED and self.lane is not None and self.rcfg.cancel_on_correction \
                and ev.payload.get("change_type") in CORRECTIONS:
            cur = self.lane.current
            if cur is not None and cur.uid != ev.utterance_id and cur.cancelled is None:
                self.lane.stats["cancelled_by_correction"] += 1
                self.lane._cancel(cur, "correction")

    # ------------------------------------------------------------------ input lane
    def offer(self, ev) -> bool:
        if not self.alive:
            return False
        self.n_inputs += 1
        if self.n_inputs > self.rcfg.max_inputs_per_session:
            self._reject(ev, "max_inputs_per_session")
            return False
        if isinstance(ev, SessionEnd):
            self._arrivals[id(ev)] = self.runtime.clock.now_ms()
        res = self.inputs.offer(ev)
        if not res.accepted:
            self._reject(ev, "input_queue_full")
            return False
        if res.coalesced:
            self.emit(E.BACKPRESSURE_APPLIED, "backpressure", {"action": "coalesced_on_full_queue",
                                                               "coalesced": res.coalesced, "depth": res.depth},
                      getattr(ev, "utterance_id", None) if getattr(ev, "utterance_id", None) in
                      self.session.chunks.utterances else None)
        if not self._drain_armed:
            self._drain_armed = True
            self.runtime.clock.call_at(self.runtime.clock.now_ms() + self.rcfg.coalescing_window_ms, PRIORITY_INPUT,
                                       self._drain)
        return True

    def _reject(self, ev, reason: str) -> None:
        self.runtime.telemetry.counters[f"rejected:{reason}"] += 1
        self.emit(E.BACKPRESSURE_APPLIED, "backpressure", {"action": "rejected", "reason": reason,
                                                           "depth": len(self.inputs),
                                                           "input_type": getattr(ev, "type", None),
                                                           "input": ev.model_dump(mode="json") if ev is not None
                                                           else None,
                                                           "arrival_ms": round(self.runtime.clock.now_ms(), 3)})

    def _drain(self) -> None:
        self._drain_armed = False
        if not self.alive:
            return
        batch, dropped = self.inputs.take_batch()
        if dropped:
            self.emit(E.TRANSCRIPT_COALESCED, "backpressure",
                      {"dropped": len(dropped), "kept": len(batch),
                       "dropped_chunks": [[d.utterance_id, d.payload.chunk_index] for d in dropped],
                       "dropped_inputs": [d.model_dump(mode="json") for d in dropped]},
                      dropped[-1].utterance_id if dropped[-1].utterance_id in self.session.chunks.utterances else None)
        for i, ev in enumerate(batch):
            if isinstance(ev, SessionEnd):
                self.emit(E.SESSION_UPDATED, "runtime", {"input": "SESSION_END",
                                                         "payload": ev.payload.model_dump(mode="json"),
                                                         "arrival_ms": round(self._arrivals.pop(id(ev), 0.0), 3)})
            nxt = batch[i + 1] if i + 1 < len(batch) else None
            decide = not (isinstance(ev, TranscriptChunk) and isinstance(nxt, TranscriptChunk)
                          and nxt.utterance_id == ev.utterance_id)
            with self.bus.dispatch():
                self.session.handle_input(ev, decide)
        if batch:
            self.state.bump()
        self.session._maybe_close()

    # ------------------------------------------------------------------ cancel / reset
    def cancel(self, reason: str) -> None:
        if not self.alive:
            return
        n = len(self.scheduler.cancel_where(lambda t: t.session_id == self.session_id, reason))
        self.runtime.cancellation.cancel_session(self.session_id, reason)
        self.emit(E.SESSION_CANCELLED, "runtime", {"reason": reason, "tasks_cancelled": n})
        self.alive = False
        self.cancelled_reason = reason
        self.bus.close()
        self.streamer.close()

    def reset(self) -> None:
        n = len(self.scheduler.cancel_where(lambda t: t.session_id == self.session_id, "session_reset"))
        self.runtime.cancellation.cancel_session(self.session_id, "session_reset")
        self.scheduler.forget_session(self.session_id)
        self.state.new_epoch()
        self.inputs.items.clear()
        self.degraded.clear()
        self.utt_t0.clear()
        self.chunk_counter.clear()
        self.emit(E.SESSION_RESET, "runtime", {"epoch": self.state.epoch, "tasks_cancelled": n})
        self._build()

    @property
    def closed(self) -> bool:
        return self.session.closed or not self.alive

    def summary(self) -> dict:
        return {"session_id": self.session_id, "events": len(self.bus.events), "state_version": self.state.version,
                "epoch": self.state.epoch, "closed": self.closed, "cancelled": self.cancelled_reason,
                "degraded": dict(self.degraded), "stale_discarded": dict(self.state.discarded),
                "rollbacks": self.state.rollbacks, "inputs": dict(self.inputs.stats),
                "max_input_depth": self.inputs.max_depth,
                "output_order": {"released": self.streamer.released, "held_on_arrival": self.streamer.held_on_arrival,
                                 "max_held": self.streamer.order.max_held, "gaps_skipped": len(self.streamer.order.skipped)},
                "lane": None if self.lane is None else {k: v for k, v in self.lane.stats.items() if k != "snapshot_ms"},
                "milestones": turn_milestones(self.bus.events, self.first_input_wall)}


class StreamingRuntime:
    def __init__(self, cfg: StreamRagConfig, stack, mode: str = "realtime", llm=None,
                 faults: FaultInjector | None = None, trace_dir: Path | None = None,
                 unbounded_inputs: bool = False) -> None:
        if mode not in ("realtime", "virtual"):
            raise ValueError("mode must be realtime or virtual")
        self.cfg, self.stack, self.mode = cfg, stack, mode
        self.rcfg = cfg.runtime
        ctl = cfg.controller.model_copy(update={"cancel_superseded": "cooperative" if self.rcfg.cancel_running
                                                else cfg.controller.cancel_superseded})
        self.session_cfg = cfg.model_copy(update={"controller": ctl})
        self.faults = faults
        self.trace_dir = trace_dir
        self.unbounded_inputs = unbounded_inputs
        self.cancellation = CancellationManager()
        self.retry = RetryManager(self.rcfg.retry)
        self.timeouts = TimeoutManager(self.rcfg)
        self.telemetry = TelemetryManager()
        self.sessions: dict[str, RuntimeSession] = {}
        self._llm_override = llm
        self.accepting = True
        self.started = False
        self._n_sessions = 0
        self.clock = None
        self.scheduler: TaskScheduler | None = None
        if mode == "virtual":
            self.clock = VirtualScheduler()
            self._make_scheduler(VirtualRunner(self.clock, self.rcfg.sim_latency_ms))
            self.started = True

    # ------------------------------------------------------------------ lifecycle
    def _make_scheduler(self, runner) -> None:
        self.runner = runner
        self.scheduler = TaskScheduler(self.rcfg, self.clock, runner, self.cancellation, self.retry,
                                       on_event=self._route_task_event)

    async def start(self) -> "StreamingRuntime":
        if self.mode != "realtime" or self.started:
            return self
        loop = asyncio.get_running_loop()
        self.clock = RealtimeScheduler(loop)
        self._make_scheduler(ThreadRunner(loop, self.clock, {"retrieval": self.rcfg.max_concurrent_retrievals,
                                                            "cpu": self.rcfg.max_concurrent_cpu,
                                                            "llm": self.rcfg.max_concurrent_llm_calls}))
        self.telemetry.loop_lag.start()
        self.started = True
        return self

    async def __aenter__(self) -> "StreamingRuntime":
        return await self.start()

    async def __aexit__(self, *exc) -> None:
        await self.shutdown()

    def llm_for(self, session_id: str):
        base = self._llm_override
        if base is None:
            base = self.stack.grounding.backend if self.cfg.generation.enabled else None
        if base is None:
            return None
        if self.faults is not None:
            base = FaultyLLM(base, self.faults, session_id)
        return RetryingLLM(base, self.retry, self.mode == "virtual")

    def _route_task_event(self, task: Task, kind: str, payload: dict) -> None:
        rs = self.sessions.get(task.session_id)
        if rs is not None and rs.state.epoch == task.epoch:
            rs.on_task_event(task, kind, payload)

    # ------------------------------------------------------------------ API
    def start_session(self, session_id: str | None = None) -> str:
        if not self.started:
            raise RuntimeClosedError("call `await runtime.start()` first (realtime mode)")
        if not self.accepting:
            raise RuntimeClosedError("runtime is shutting down")
        if session_id is not None and session_id in self.sessions and not self.sessions[session_id].closed:
            raise ValueError(f"session {session_id} is already active")
        active = sum(1 for s in self.sessions.values() if not s.closed)
        if active >= self.rcfg.max_concurrent_sessions:
            self.telemetry.counters["sessions_rejected"] += 1
            raise RuntimeCapacityError(f"max_concurrent_sessions={self.rcfg.max_concurrent_sessions} reached")
        self._n_sessions += 1
        sid = session_id or f"rt{self._n_sessions:04d}"
        trace = self.trace_dir / f"{sid}.jsonl" if self.trace_dir is not None else None
        rs = self.sessions[sid] = RuntimeSession(self, sid, trace)
        rs.offer(SessionStart(session_id=sid))
        return sid

    def _session(self, session_id: str) -> RuntimeSession:
        rs = self.sessions.get(session_id)
        if rs is None:
            raise KeyError(f"unknown session {session_id}")
        return rs

    def push(self, event, at_ms: float | None = None) -> bool:
        """Raw input event (TranscriptChunk / UtteranceEnd / SessionEnd). Virtual mode: ``at_ms`` = arrival time."""
        rs = self._session(event.session_id)
        if self.mode == "virtual" and at_ms is not None and at_ms > self.clock.now_ms():
            self.clock.call_at(at_ms, PRIORITY_INPUT, self._accept, rs, event)
            return True
        return self._accept(rs, event)

    def _accept(self, rs: RuntimeSession, event) -> bool:
        uid = getattr(event, "utterance_id", None)
        if uid is not None and uid not in rs.first_input_wall:
            rs.first_input_wall[uid] = round((time.perf_counter() - rs.bus._t0) * 1000.0, 3)
            rs.utt_t0[uid] = self.clock.now_ms()
        return rs.offer(event)

    def push_transcript_delta(self, session_id: str, utterance_id: str, text: str, *, stability: str = "final",
                              replaces: int | None = None, at_ms: float | None = None) -> bool:
        """A transcript delta (appended text), or a revision of an earlier chunk (``replaces``: an ASR partial
        hypothesis that a newer one replaces). Priority, routing and deadlines are never taken from the caller."""
        rs = self._session(session_id)
        if len(text) > self.rcfg.max_chunk_chars:
            rs._reject(None, "chunk_too_long")
            return False
        if self.mode == "virtual" and at_ms is not None and at_ms > self.clock.now_ms():
            self.clock.call_at(at_ms, PRIORITY_INPUT, self._delta, rs, utterance_id, text, stability, replaces)
            return True
        return self._delta(rs, utterance_id, text, stability, replaces)

    def _delta(self, rs: RuntimeSession, uid: str, text: str, stability: str, replaces: int | None) -> bool:
        if uid not in rs.utt_t0:
            rs.first_input_wall[uid] = round((time.perf_counter() - rs.bus._t0) * 1000.0, 3)
            rs.utt_t0[uid] = self.clock.now_ms()
        if replaces is None:
            idx = rs.chunk_counter.get(uid, -1) + 1
            rs.chunk_counter[uid] = idx
        else:
            idx = rs.chunk_counter.get(uid, replaces)
        ts = max(0.0, (self.clock.now_ms() - rs.utt_t0[uid]) / 1000.0)
        # utterance_offset_s + timestamp_s = arrival time on the runtime clock (replayable from the event log)
        ev = TranscriptChunk(session_id=rs.session_id, utterance_id=uid,
                             payload=TranscriptChunkPayload(chunk_index=idx, timestamp_s=round(ts, 4), text=text,
                                                            stability=stability, replaces_chunk_index=replaces,
                                                            utterance_offset_s=round(rs.utt_t0[uid] / 1000.0, 6)))
        return rs.offer(ev)

    def end_utterance(self, session_id: str, utterance_id: str, reason: str = "endpoint",
                      at_ms: float | None = None) -> bool:
        rs = self._session(session_id)
        if self.mode == "virtual" and at_ms is not None and at_ms > self.clock.now_ms():
            self.clock.call_at(at_ms, PRIORITY_INPUT, self._end, rs, utterance_id, reason)
            return True
        return self._end(rs, utterance_id, reason)

    def _end(self, rs: RuntimeSession, uid: str, reason: str) -> bool:
        t0 = rs.utt_t0.get(uid, self.clock.now_ms())
        ts = max(0.0, (self.clock.now_ms() - t0) / 1000.0)
        return rs.offer(UtteranceEnd(session_id=rs.session_id, utterance_id=uid,
                                     payload=UtteranceEndPayload(timestamp_s=round(ts, 4), reason=reason)))

    def end_session(self, session_id: str, at_ms: float | None = None) -> bool:
        rs = self._session(session_id)
        ev = SessionEnd(session_id=session_id, payload=SessionEndPayload(reason="client_closed"))
        if self.mode == "virtual" and at_ms is not None and at_ms > self.clock.now_ms():
            self.clock.call_at(at_ms, PRIORITY_INPUT, rs.offer, ev)
            return True
        return rs.offer(ev)

    def events(self, session_id: str, user_visible: bool = False) -> list:
        evs = self._session(session_id).bus.events
        return [e for e in evs if e.output_seq is not None] if user_visible else list(evs)

    async def get_events(self, session_id: str, user_visible: bool = True) -> AsyncIterator:
        """Live event stream of a session (bounded subscription; ends when the session closes)."""
        rs = self._session(session_id)
        sub = rs.streamer.subscribe(user_visible)
        sub.waiter = asyncio.Event()
        while True:
            while sub.queue:
                yield sub.queue.popleft()
            if sub.disconnected or rs.streamer.closed or rs.closed:
                while sub.queue:
                    yield sub.queue.popleft()
                return
            sub.waiter.clear()
            await sub.waiter.wait()

    def cancel_session(self, session_id: str, reason: str = "client_cancelled") -> None:
        self._session(session_id).cancel(reason)

    def reset_session(self, session_id: str) -> None:
        self._session(session_id).reset()

    async def complete_session(self, session_id: str, timeout_s: float = 120.0) -> dict:
        self.end_session(session_id)
        await self.wait_closed(session_id, timeout_s)
        return self._session(session_id).summary()

    async def wait_closed(self, session_id: str, timeout_s: float = 120.0) -> None:
        rs = self._session(session_id)
        end = time.monotonic() + timeout_s
        while not rs.closed:
            if time.monotonic() > end:
                raise TimeoutError(f"session {session_id} did not close within {timeout_s}s")
            await asyncio.sleep(0.005)

    async def wait_idle(self, timeout_s: float = 120.0) -> None:
        end = time.monotonic() + timeout_s
        while self.scheduler.busy() or any(len(s.inputs) or s._drain_armed for s in self.sessions.values()):
            if time.monotonic() > end:
                raise TimeoutError("runtime did not become idle")
            await asyncio.sleep(0.005)

    def run(self) -> None:
        """Virtual mode: process every scheduled input, task and timer (deterministic)."""
        if self.mode != "virtual":
            raise RuntimeError("run() is for virtual mode; use the async API in realtime mode")
        self.clock.run()

    # ------------------------------------------------------------------ shutdown
    async def shutdown(self, grace_ms: float | None = None) -> dict:
        """Graceful: stop accepting, let active work finish (<= grace), cancel the rest, wait for workers, flush."""
        if not self.started:
            return {}
        self.accepting = False
        grace = (self.rcfg.shutdown_grace_ms if grace_ms is None else grace_ms) / 1000.0
        end = time.monotonic() + grace
        while (self.scheduler.busy() or any(len(s.inputs) for s in self.sessions.values())) and time.monotonic() < end:
            await asyncio.sleep(0.005)
        self.scheduler.stop_accepting()
        return await self._close(cancel_reason="shutdown")

    async def force_shutdown(self) -> dict:
        """Immediate: no grace period - everything is cancelled now; workers stop at their next checkpoint (waited
        for at most 1 s, then the pools are closed without waiting)."""
        if not self.started:
            return {}
        self.accepting = False
        self.scheduler.stop_accepting()
        return await self._close(cancel_reason="force_shutdown", wait_workers=False)

    async def _close(self, cancel_reason: str, wait_workers: bool = True) -> dict:
        cancelled = 0
        for rs in self.sessions.values():
            if rs.alive and not rs.session.closed:
                cancelled += 1
                rs.cancel(cancel_reason)
            elif rs.alive:
                rs.emit(E.RUNTIME_SHUTDOWN, "runtime", {"reason": cancel_reason})
                rs.alive = False
                rs.bus.close()
                rs.streamer.close()
        self.scheduler.cancel_where(lambda t: True, cancel_reason)
        if self.mode == "realtime":
            # workers stop at their next checkpoint: wait for them (bounded - 5 s graceful, 1 s forced)
            end = time.monotonic() + (5.0 if wait_workers else 1.0)
            while any(p.running for p in self.scheduler.pools.values()) and time.monotonic() < end:
                await asyncio.sleep(0.005)
            await asyncio.to_thread(self.runner.close, wait_workers)
            for _ in range(3):                         # let the last completion callbacks run
                await asyncio.sleep(0)
            await self.telemetry.loop_lag.stop()
        self.started = False
        return {"sessions_cancelled": cancelled, "live_tokens": self.cancellation.live_tokens,
                "zombies": self.scheduler.zombies()}

    # ------------------------------------------------------------------ reporting
    def summary(self) -> dict:
        return {"mode": self.mode, "sessions": {k: s.summary() for k, s in self.sessions.items()},
                "scheduler": self.telemetry.scheduler_summary(self.scheduler),
                "retry": dict(self.retry.stats), "cancellation": dict(self.cancellation.stats),
                "counters": dict(self.telemetry.counters),
                "faults": dict(self.faults.hits) if self.faults else {},
                "loop_lag_ms": self.telemetry.loop_lag.lags_ms and {
                    "max": round(max(self.telemetry.loop_lag.lags_ms), 3),
                    "p95": round(sorted(self.telemetry.loop_lag.lags_ms)[int(0.95 * (len(self.telemetry.loop_lag.lags_ms) - 1))], 3),
                    "samples": len(self.telemetry.loop_lag.lags_ms)}}
