"""Schedulers. The streaming session logic is the same in both modes; only time and execution differ.

* ``VirtualScheduler`` — deterministic discrete-event simulation. Callbacks run in (time, priority, seq) order;
  the clock jumps to each event. Same input + config + code => identical event sequence (replayable).
* ``RealtimeScheduler`` — asyncio event loop with the wall clock (optionally sped up). Used to measure real
  latencies and to prove the stream is not blocked while retrievals run in worker threads.

Priorities for simultaneous events: retrieval completion (0) < input (1) < timer (2).

Two clocks: ``now_ms()`` is when something actually happens (used for event timestamps and latency telemetry);
``logical_now_ms()`` is the stream time a callback was scheduled for (used for controller decisions and their
timers), so realtime decisions depend on the input stream, not on processing delays. In virtual mode they coincide.
"""

from __future__ import annotations

import asyncio
import heapq
from collections.abc import Callable

PRIORITY_COMPLETION, PRIORITY_INPUT, PRIORITY_TIMER = 0, 1, 2


class VirtualScheduler:
    mode = "virtual"

    def __init__(self) -> None:
        self._heap: list[tuple[float, int, int, Callable, tuple]] = []
        self._seq = 0
        self._now = 0.0

    def now_ms(self) -> float:
        return self._now

    def logical_now_ms(self) -> float:
        return self._now

    def call_at(self, t_ms: float, priority: int, fn: Callable, *args) -> None:
        heapq.heappush(self._heap, (max(float(t_ms), self._now), priority, self._seq, fn, args))
        self._seq += 1

    def run(self) -> None:
        while self._heap:
            t, _, _, fn, args = heapq.heappop(self._heap)
            self._now = t
            fn(*args)


class RealtimeScheduler:
    """Same (time, priority, seq) ordering as ``VirtualScheduler``, dispatched on the wall clock: one loop timer is
    armed at the absolute loop time of the earliest entry. (Arming one ``call_later`` per callback with a delay derived
    from a fresh ``now`` let a later-scheduled callback with the same timestamp fire first — e.g. SESSION_END before
    the UTTERANCE_END it was meant to follow.)"""

    mode = "realtime"

    def __init__(self, loop: asyncio.AbstractEventLoop, speed: float = 1.0) -> None:
        self.loop, self.speed = loop, speed
        self._t0 = loop.time()
        self._heap: list[tuple[float, int, int, Callable, tuple]] = []
        self._seq = 0
        self._timer: asyncio.TimerHandle | None = None
        self._timer_t: float | None = None
        self._dispatching = False
        self._logical: float | None = None

    @property
    def pending(self) -> int:
        return len(self._heap)

    def now_ms(self) -> float:
        return (self.loop.time() - self._t0) * 1000.0 * self.speed

    def logical_now_ms(self) -> float:
        """Scheduled time of the callback being dispatched; wall time outside the scheduler (e.g. retrieval
        completions delivered by asyncio tasks)."""
        return self._logical if self._logical is not None else self.now_ms()

    def call_at(self, t_ms: float, priority: int, fn: Callable, *args) -> None:
        heapq.heappush(self._heap, (float(t_ms), priority, self._seq, fn, args))
        self._seq += 1
        if not self._dispatching:
            self._arm()

    def _arm(self) -> None:
        if not self._heap:
            return
        t = self._heap[0][0]
        if self._timer is not None:
            if self._timer_t is not None and self._timer_t <= t:
                return
            self._timer.cancel()
        self._timer_t = t
        self._timer = self.loop.call_at(self._t0 + t / 1000.0 / self.speed, self._dispatch)

    def _dispatch(self) -> None:
        due = max(self._timer_t or 0.0, self.now_ms())
        self._timer = self._timer_t = None
        self._dispatching = True
        try:
            while self._heap and self._heap[0][0] <= due:
                t, _, _, fn, args = heapq.heappop(self._heap)
                self._logical = t
                fn(*args)
        finally:
            self._dispatching = False
            self._logical = None
            self._arm()

    async def drain(self, busy: Callable[[], bool], poll_s: float = 0.002) -> None:
        """Wait until ``busy()`` is false. The caller's predicate covers pending inputs (session not closed until
        SESSION_END, which is dispatched after every input with an earlier or equal timestamp) and in-flight
        retrievals; timers still pending after the session closed are no-ops (they check finalized state) and are
        discarded with the loop."""
        while busy():
            await asyncio.sleep(poll_s)
