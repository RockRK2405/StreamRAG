"""Run a stream of input events through a StreamingSession in virtual (deterministic) or realtime mode."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path

from streamrag.config.settings import StreamRagConfig, config_hash
from streamrag.models.events import SessionEnd, SessionStart, TelemetryEvent, TranscriptChunk, UtteranceEnd
from streamrag.streaming.clock import PRIORITY_INPUT, RealtimeScheduler, VirtualScheduler
from streamrag.streaming.events import EventBus
from streamrag.streaming.executor import AsyncExecutor, RetrievalBackend, VirtualExecutor
from streamrag.streaming.session import StreamingSession


@dataclass
class StreamRun:
    session: StreamingSession
    events: list[TelemetryEvent]
    mode: str


def _session_id(inputs: list) -> str:
    return inputs[0].session_id if inputs else "sim"


def input_times(inputs: list) -> list[tuple[float, object]]:
    """Arrival time (ms) of every input on the stream clock; a SessionEnd is appended if missing."""
    out: list[tuple[float, object]] = []
    offsets: dict[str, float] = {}
    t_last = 0.0
    for ev in inputs:
        if isinstance(ev, TranscriptChunk):
            off = ev.payload.utterance_offset_s or 0.0
            offsets[ev.utterance_id] = off
            t = (off + ev.payload.timestamp_s) * 1000.0
        elif isinstance(ev, UtteranceEnd):
            t = (offsets.get(ev.utterance_id, 0.0) + ev.payload.timestamp_s) * 1000.0
        elif isinstance(ev, SessionStart):
            t = 0.0
        else:
            t = t_last
        t_last = max(t_last, t)
        out.append((t, ev))
    if not any(isinstance(e, SessionEnd) for e in inputs):
        out.append((t_last, SessionEnd(session_id=_session_id(inputs))))
    return out


def _schedule(sched, session: StreamingSession, inputs: list) -> None:
    for t, ev in input_times(inputs):
        sched.call_at(t, PRIORITY_INPUT, session.handle_input, ev)


def _meta(cfg: StreamRagConfig, index_hash: str | None) -> dict:
    return {"config_hash": config_hash(cfg), "index_content_hash": index_hash,
            "sim_retrieval_latency_ms": cfg.streaming.sim_retrieval_latency_ms}


def _slots(cfg: StreamRagConfig) -> int:
    return cfg.multi_intent.max_concurrent_retrievals if cfg.multi_intent.enabled else cfg.controller.max_concurrent_retrievals


def run_virtual(cfg: StreamRagConfig, backend: RetrievalBackend, policy, inputs: list, trace_path: Path | None = None,
                index_hash: str | None = None, intent_stack=None) -> StreamRun:
    sched = VirtualScheduler()
    bus = EventBus(_session_id(inputs), sched.now_ms, trace_path)
    ex = VirtualExecutor(sched, backend, cfg.streaming.sim_retrieval_latency_ms, _slots(cfg))
    session = StreamingSession(_session_id(inputs), cfg, policy, sched, ex, bus, _meta(cfg, index_hash), intent_stack)
    _schedule(sched, session, inputs)
    sched.run()
    bus.close()
    return StreamRun(session, bus.events, "virtual")


async def arun_realtime(cfg: StreamRagConfig, backend: RetrievalBackend, policy, inputs: list,
                        trace_path: Path | None = None, index_hash: str | None = None, intent_stack=None) -> StreamRun:
    loop = asyncio.get_running_loop()
    sched = RealtimeScheduler(loop, cfg.streaming.speed)
    bus = EventBus(_session_id(inputs), sched.now_ms, trace_path)
    ex = AsyncExecutor(sched, backend, cfg.streaming.retrieval_timeout_ms, _slots(cfg))
    session = StreamingSession(_session_id(inputs), cfg, policy, sched, ex, bus, _meta(cfg, index_hash), intent_stack)
    _schedule(sched, session, inputs)
    await sched.drain(lambda: ex.busy() or not session.closed)
    bus.close()
    return StreamRun(session, bus.events, "realtime")


def run_realtime(cfg: StreamRagConfig, backend: RetrievalBackend, policy, inputs: list, trace_path: Path | None = None,
                 index_hash: str | None = None, intent_stack=None) -> StreamRun:
    return asyncio.run(arun_realtime(cfg, backend, policy, inputs, trace_path, index_hash, intent_stack))
