"""AnswerStreamer: the ordered, user-visible output stream (docs/runtime/01, 02).

Work completes in any order; what the user sees must not. Two mechanisms:

1. **Structural order.** Every session emits on the event loop, and answer results are committed by the answer lane
   one at a time in request order, so the session log is a total order. User-visible events carry a contiguous
   ``output_seq``.
2. **Ordered release buffer** (defence in depth, and the mechanism for consumers that receive events from several
   producers): items carry ``(stream, seq)``; item N+1 that arrives before N is held until N arrives, or until it was
   held longer than ``max_hold_ms`` - then the gap is skipped and reported, so buffering is never indefinite.

Subscribers get a bounded queue each. A subscriber that falls behind first loses intermediate DRAFT answer events
(a newer draft or the final supersedes them); if it still overflows it is disconnected with an error marker and
must resynchronise from the session log - the producer never blocks and memory stays bounded.
"""

from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field

from streamrag.models.events import EventType, TelemetryEvent
from streamrag.runtime.events import USER_VISIBLE


@dataclass
class _Held:
    seq: int
    item: object
    since_ms: float


@dataclass
class OrderedReleaseBuffer:
    max_hold_ms: float
    now_ms: Callable[[], float]
    next_seq: dict[str, int] = field(default_factory=dict)
    held: dict[str, dict[int, _Held]] = field(default_factory=dict)
    skipped: list[tuple[str, int]] = field(default_factory=list)
    max_held: int = 0

    def push(self, stream: str, seq: int, item) -> list:
        """Returns the items now releasable, in order."""
        nxt = self.next_seq.setdefault(stream, 0)
        if seq < nxt:
            return []                                       # duplicate / already skipped: never re-released
        self.held.setdefault(stream, {})[seq] = _Held(seq, item, self.now_ms())
        self.max_held = max(self.max_held, sum(len(h) for h in self.held.values()))
        return self._drain(stream)

    def _drain(self, stream: str) -> list:
        out, h = [], self.held.get(stream, {})
        while self.next_seq[stream] in h:
            out.append(h.pop(self.next_seq[stream]).item)
            self.next_seq[stream] += 1
        return out

    def flush_expired(self) -> list:
        """Skip gaps whose successor was held too long (missing items are reported, not waited for forever)."""
        out, now = [], self.now_ms()
        for stream, h in self.held.items():
            while h and min(x.since_ms for x in h.values()) + self.max_hold_ms <= now:
                first = min(h)
                for s in range(self.next_seq[stream], first):
                    self.skipped.append((stream, s))
                self.next_seq[stream] = first
                out.extend(self._drain(stream))
        return out

    @property
    def pending(self) -> int:
        return sum(len(h) for h in self.held.values())


class Subscription:
    def __init__(self, capacity: int, user_visible: bool) -> None:
        self.queue: deque[TelemetryEvent] = deque()
        self.capacity, self.user_visible = capacity, user_visible
        self.dropped_drafts = 0
        self.disconnected = False
        self.waiter: asyncio.Event | None = None

    def offer(self, ev: TelemetryEvent) -> None:
        if self.disconnected or (self.user_visible and ev.type not in USER_VISIBLE):
            return
        if len(self.queue) >= self.capacity:
            before = len(self.queue)
            self.queue = deque(e for e in self.queue if e.payload.get("status") != "DRAFT")
            self.dropped_drafts += before - len(self.queue)
            if len(self.queue) >= self.capacity:
                self.disconnected = True
                self.queue.clear()
                self._wake()
                return
        self.queue.append(ev)
        self._wake()

    def _wake(self) -> None:
        if self.waiter is not None:
            self.waiter.set()


class AnswerStreamer:
    """Fans a session's events out to subscribers (bounded). User-visible events pass the ordered release buffer
    (by ``output_seq``) before fan-out; telemetry-only events go straight through in log order."""

    def __init__(self, capacity: int, max_hold_ms: float = 2000.0, now_ms=None) -> None:
        self.capacity = capacity
        self.subs: list[Subscription] = []
        self.closed = False
        self.order = OrderedReleaseBuffer(max_hold_ms, now_ms or (lambda: 0.0))
        self.released = 0
        self.held_on_arrival = 0

    def subscribe(self, user_visible: bool = True) -> Subscription:
        s = Subscription(self.capacity, user_visible)
        self.subs.append(s)
        return s

    def on_event(self, ev: TelemetryEvent) -> None:
        if ev.output_seq is None:
            out = [ev]
        else:
            out = self.order.push(ev.session_id, ev.output_seq, ev) + self.order.flush_expired()
            if not out:
                self.held_on_arrival += 1
            self.released += len(out)
        for e in out:
            for s in self.subs:
                s.offer(e)
        if ev.type in (EventType.SESSION_CLOSED, EventType.SESSION_CANCELLED, EventType.RUNTIME_SHUTDOWN):
            self.close()

    def close(self) -> None:
        self.closed = True
        for s in self.subs:
            s._wake()
