"""In-process event bus. Every streaming event is a validated ``TelemetryEvent`` (spec §6.1 envelope):
deterministic ``event_id`` (session:seq), ``t_session_ms`` (session stream clock), ``t_stream_s`` (utterance-relative)
and ``t_wall_ms`` (monotonic wall clock). Non-deterministic measurements live under ``payload["wall"]`` so a
replay can compare everything else exactly (``canonical_for_replay``).
"""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from streamrag.models.events import EventType, TelemetryEvent


class EventBus:
    def __init__(self, session_id: str, now_ms: Callable[[], float], trace_path: Path | None = None) -> None:
        self.session_id = session_id
        self.now_ms = now_ms
        self.events: list[TelemetryEvent] = []
        self.subscribers: list[Callable[[TelemetryEvent], None]] = []
        self.emit_wall_ms: list[float] = []          # emission cost (profiling)
        self._t0 = time.perf_counter()
        self._fh = None
        if trace_path is not None:
            trace_path.parent.mkdir(parents=True, exist_ok=True)
            self._fh = trace_path.open("w", encoding="utf-8")

    def emit(self, type_: EventType, component: str, payload: dict[str, Any], utterance_id: str | None = None,
             utterance_start_ms: float | None = None, intent_id: str | None = None,
             query_id: str | None = None) -> TelemetryEvent:
        t = time.perf_counter()
        now = round(float(self.now_ms()), 3)
        t_stream = round((now - utterance_start_ms) / 1000.0, 4) if utterance_start_ms is not None else None
        seq = len(self.events)
        ev = TelemetryEvent(event_id=f"{self.session_id}:{seq:06d}", type=type_, session_id=self.session_id,
                            utterance_id=utterance_id, intent_id=intent_id, query_id=query_id, seq=seq, t_session_ms=max(now, 0.0), t_stream_s=t_stream,
                            t_wall_ms=round((t - self._t0) * 1000.0, 3), component=component, payload=payload)
        self.events.append(ev)
        if self._fh is not None:
            self._fh.write(ev.canonical_json() + "\n")
        for fn in self.subscribers:
            fn(ev)
        self.emit_wall_ms.append((time.perf_counter() - t) * 1000.0)
        return ev

    def close(self) -> None:
        if self._fh is not None and not self._fh.closed:
            self._fh.flush()
            self._fh.close()


def _strip_wall(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _strip_wall(v) for k, v in obj.items() if k != "wall"}
    if isinstance(obj, list):
        return [_strip_wall(v) for v in obj]
    return obj


def canonical_for_replay(ev: TelemetryEvent | dict) -> dict:
    d = ev.model_dump(mode="json") if isinstance(ev, TelemetryEvent) else dict(ev)
    d.pop("t_wall_ms", None)
    return _strip_wall(d)
