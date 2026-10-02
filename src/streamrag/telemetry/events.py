"""JSONL telemetry sink. Write-only by design: the pipeline never reads telemetry back (spec REQ-SESS-003).

Phase 3 uses it for retrieval/benchmark traces; the async, session-scoped bus arrives in Phase 4.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from streamrag.models.events import EventType, TelemetryEvent


class JsonlEventSink:
    def __init__(self, path: Path, session_id: str = "run") -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("a", encoding="utf-8")
        self._seq = 0
        self._t0 = time.perf_counter()
        self.session_id = session_id

    def emit(self, type_: EventType, component: str, payload: dict[str, Any],
             utterance_id: str | None = None, t_stream_s: float | None = None) -> TelemetryEvent:
        ev = TelemetryEvent(event_id=f"{self.session_id}:{self._seq:06d}", type=type_, session_id=self.session_id,
                            utterance_id=utterance_id, seq=self._seq, t_stream_s=t_stream_s,
                            t_wall_ms=round((time.perf_counter() - self._t0) * 1000.0, 3),
                            component=component, payload=payload)
        self._fh.write(ev.canonical_json() + "\n")
        self._seq += 1
        return ev

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.flush()
            self._fh.close()

    def __enter__(self) -> "JsonlEventSink":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
