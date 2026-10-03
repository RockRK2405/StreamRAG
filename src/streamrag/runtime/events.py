"""Runtime event bus (docs/runtime/01): the Phase 4 ``EventBus`` plus trace relations.

Every event is a ``TelemetryEvent`` (same deterministic ``session:seq`` ids) with, in addition:

* ``correlation_id`` - the turn it belongs to (``<session>/<utterance>``; session-level events: ``<session>/session``);
* ``causation_id`` - the event whose handling produced it. Runtime-generated events name their cause explicitly
  (RETRIEVAL_STARTED <- the query's QUERY_GENERATED; an answer's events <- the generation TASK_COMPLETED). Pipeline
  events take the cause of the dispatch they are emitted in (an input chunk, a timer, a task completion);
* ``parent_event_id`` - the trace-tree parent by entity lineage: session -> utterance -> intent -> query -> task ->
  evidence -> claim -> answer (``trace_tree``);
* ``state_version`` - the session state version at emission;
* ``output_seq`` - contiguous order of the user-visible stream (only on user-visible events).

All emission for a session happens on the event loop (worker results are committed there), so ``seq`` is a total
order of the session's state transitions.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from streamrag.models.events import EventType, TelemetryEvent
from streamrag.streaming.events import EventBus

USER_VISIBLE = frozenset({
    EventType.SESSION_STARTED, EventType.UTTERANCE_FINALIZED, EventType.ANSWER_STARTED,
    EventType.ANSWER_SECTION_STARTED, EventType.ANSWER_CLAIM_READY, EventType.ANSWER_CITATION_READY,
    EventType.ANSWER_SECTION_COMPLETED, EventType.ANSWER_VALIDATED, EventType.ANSWER_REVISED,
    EventType.ANSWER_COMPLETED, EventType.ANSWER_DELTA, EventType.ANSWER_FINALIZED, EventType.ANSWER_COMMITTED,
    EventType.TURN_COMPLETED, EventType.DEGRADED_MODE_CHANGED, EventType.SESSION_CANCELLED,
    EventType.SESSION_RESET, EventType.SESSION_CLOSED, EventType.ERROR,
})

# events that create the anchor of an entity (others referencing the entity hang below it)
_CREATES = {"intent": {EventType.INTENT_DETECTED}, "query": {EventType.QUERY_UPDATED, EventType.QUERY_GENERATED},
            "task": {EventType.TASK_SCHEDULED}}


class RuntimeEventBus(EventBus):
    def __init__(self, session_id: str, now_ms: Callable[[], float], trace_path: Path | None = None,
                 version_fn: Callable[[], int] | None = None) -> None:
        super().__init__(session_id, now_ms, trace_path)
        self.version_fn = version_fn or (lambda: 0)
        self._dispatch: list[list] = []                  # [explicit cause, first event of the dispatch, task]
        self._anchors: dict[tuple[str, str], str] = {}
        self._root: str | None = None
        self.output_seq = 0
        self.closed = False

    # ------------------------------------------------------------------ dispatch context
    @contextmanager
    def dispatch(self, cause: str | None = None, task_id: str | None = None) -> Iterator[None]:
        self._dispatch.append([cause, None, task_id])
        try:
            yield
        finally:
            self._dispatch.pop()

    @property
    def last_event_id(self) -> str | None:
        return self.events[-1].event_id if self.events else None

    def anchor(self, kind: str, key: str | None) -> str | None:
        return self._anchors.get((kind, key)) if key is not None else None

    # ------------------------------------------------------------------ emission
    def emit(self, type_: EventType, component: str, payload: dict[str, Any], utterance_id: str | None = None,
             utterance_start_ms: float | None = None, intent_id: str | None = None, query_id: str | None = None,
             causation_id: str | None = None) -> TelemetryEvent:
        t = time.perf_counter()
        now = round(float(self.now_ms()), 3)
        t_stream = round((now - utterance_start_ms) / 1000.0, 4) if utterance_start_ms is not None else None
        seq = len(self.events)
        eid = f"{self.session_id}:{seq:06d}"
        ctx = self._dispatch[-1] if self._dispatch else None
        if causation_id is None and ctx is not None:
            causation_id = ctx[0] if ctx[0] is not None else ctx[1]
        elif causation_id is None and not self._dispatch:
            causation_id = None
        parent = self._parent(type_, payload, utterance_id, intent_id, query_id, ctx)
        ev = TelemetryEvent(event_id=eid, type=type_, session_id=self.session_id, utterance_id=utterance_id,
                            intent_id=intent_id, query_id=query_id, seq=seq, t_session_ms=max(now, 0.0),
                            t_stream_s=t_stream, t_wall_ms=round((t - self._t0) * 1000.0, 3), component=component,
                            payload=payload,
                            correlation_id=f"{self.session_id}/{utterance_id or 'session'}",
                            causation_id=causation_id, parent_event_id=parent, state_version=self.version_fn(),
                            output_seq=self._next_output() if type_ in USER_VISIBLE else None)
        if ctx is not None and ctx[1] is None:
            ctx[1] = eid
        self._register(ev)
        self.events.append(ev)
        if self._fh is not None:
            self._fh.write(ev.canonical_json() + "\n")
        for fn in list(self.subscribers):
            fn(ev)
        self.emit_wall_ms.append((time.perf_counter() - t) * 1000.0)
        return ev

    def _next_output(self) -> int:
        n = self.output_seq
        self.output_seq += 1
        return n

    def _parent(self, type_, payload, uid, iid, qid, ctx) -> str | None:
        tid = payload.get("task_id") if isinstance(payload.get("task_id"), str) else None
        aid = payload.get("answer_id") if isinstance(payload.get("answer_id"), str) else None
        creates = {k for k, types in _CREATES.items() if type_ in types}
        order = [("task", tid if "task" not in creates else None), ("answer", aid),
                 ("task", ctx[2] if ctx is not None else None),
                 ("query", qid if "query" not in creates else None),
                 ("intent", iid if "intent" not in creates else None), ("utterance", uid)]
        for kind, key in order:
            a = self.anchor(kind, key)
            if a is not None:
                return a
        return self._root

    def _register(self, ev: TelemetryEvent) -> None:
        if self._root is None:
            self._root = ev.event_id
        p = ev.payload
        keys = [("utterance", ev.utterance_id), ("intent", ev.intent_id), ("query", ev.query_id),
                ("task", p.get("task_id") if ev.type == EventType.TASK_SCHEDULED else None),
                ("answer", p.get("answer_id") if isinstance(p.get("answer_id"), str) else None)]
        for kind, key in keys:
            if key is not None and (kind, key) not in self._anchors:
                self._anchors[(kind, key)] = ev.event_id

    def close(self) -> None:
        self.closed = True
        super().close()


def trace_tree(events: list[TelemetryEvent]) -> dict[str, list[str]]:
    """parent_event_id -> children (in order)."""
    tree: dict[str, list[str]] = {}
    for e in events:
        if e.parent_event_id is not None:
            tree.setdefault(e.parent_event_id, []).append(e.event_id)
    return tree


def untraceable(events: list[TelemetryEvent]) -> list[str]:
    """Events (other than the session root) with neither a parent nor a cause, or whose references are dangling."""
    ids = {e.event_id for e in events}
    out = []
    for i, e in enumerate(events):
        refs = [r for r in (e.parent_event_id, e.causation_id) if r is not None]
        if i > 0 and not refs:
            out.append(e.event_id)
        elif any(r not in ids for r in refs) or e.correlation_id is None:
            out.append(e.event_id)
    return out


def path_to_root(events: list[TelemetryEvent], event_id: str) -> list[TelemetryEvent]:
    by_id = {e.event_id: e for e in events}
    out, cur = [], by_id.get(event_id)
    while cur is not None:
        out.append(cur)
        cur = by_id.get(cur.parent_event_id) if cur.parent_event_id else None
    return out
