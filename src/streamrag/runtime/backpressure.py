"""Backpressure: bounded queues, transcript-delta coalescing, load shedding (docs/runtime/08).

Strategy (in order):
1. **Adaptive batching** - a session's input queue is drained in one lane step: all pending deltas are applied to
   the transcript, and the expensive work (controller decision, decomposition, retrieval) runs once for the batch.
   With ``coalescing_window_ms = 0`` this only happens when deltas arrive faster than they are processed.
2. **Coalescing** - an intermediate transcript state that a later pending delta replaces (an ASR partial
   hypothesis revised before it was processed) is dropped. Utterance / session ends and the latest state of every
   chunk are never dropped, so the final meaningful transcript is always the one processed.
3. **Bounded queues** - every queue has a capacity. A full input queue first coalesces in place; if still full the
   delta is rejected with a BACKPRESSURE_APPLIED event (never silent, never unbounded). Control inputs (utterance /
   session ends, at most one per utterance) are never rejected. A full task queue sheds the
   least important pending task (drafts / analytics) before rejecting new work.
"""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass

from streamrag.models.events import SessionEnd, TranscriptChunk, UtteranceEnd


def _target(ev) -> tuple[str, int] | None:
    if isinstance(ev, TranscriptChunk):
        p = ev.payload
        return ev.utterance_id, (p.replaces_chunk_index if p.replaces_chunk_index is not None else p.chunk_index)
    return None


def coalesce(items: list) -> tuple[list, list]:
    """(kept, dropped): a chunk is dropped when a later pending chunk of the same utterance targets the same chunk
    index (it replaces it) and nothing that must see the earlier state (an utterance end) lies between them."""
    last_for: dict[tuple[str, int], int] = {}
    barrier_after: dict[str, int] = {}
    for i, ev in enumerate(items):
        t = _target(ev)
        if t is not None:
            last_for[t] = i
        elif isinstance(ev, (UtteranceEnd, SessionEnd)):
            barrier_after[getattr(ev, "utterance_id", "*")] = i
    kept, dropped = [], []
    for i, ev in enumerate(items):
        t = _target(ev)
        if t is not None and last_for[t] != i:
            b = barrier_after.get(t[0], -1)
            if not (i < b < last_for[t]):            # never coalesce across the utterance end
                dropped.append(ev)
                continue
        kept.append(ev)
    return kept, dropped


@dataclass
class OfferResult:
    accepted: bool
    coalesced: int = 0
    depth: int = 0


class InputQueue:
    """Bounded per-session input queue (latest-state preference under pressure)."""

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.items: deque = deque()
        self.max_depth = 0
        self.max_delta_depth = 0                          # transcript deltas only (the bounded part)
        self.stats: Counter = Counter()

    def _deltas(self) -> int:
        return sum(1 for x in self.items if isinstance(x, TranscriptChunk))

    def offer(self, ev) -> OfferResult:
        if not isinstance(ev, TranscriptChunk):
            # control inputs (utterance / session ends) are never dropped: they are what makes the final state
            # meaningful, and there is at most one per utterance
            self.items.append(ev)
            self.max_depth = max(self.max_depth, len(self.items))
            self.stats["accepted"] += 1
            return OfferResult(True, 0, len(self.items))
        if self._deltas() >= self.capacity:
            kept, dropped = coalesce(list(self.items) + [ev])
            if sum(1 for x in kept if isinstance(x, TranscriptChunk)) <= self.capacity:
                self.items = deque(kept)
                self.stats["coalesced_on_full"] += len(dropped)
                self.max_depth = max(self.max_depth, len(self.items))
                self.max_delta_depth = max(self.max_delta_depth, self._deltas())
                return OfferResult(True, len(dropped), len(self.items))
            self.stats["rejected"] += 1
            return OfferResult(False, 0, len(self.items))
        self.items.append(ev)
        self.max_depth = max(self.max_depth, len(self.items))
        self.max_delta_depth = max(self.max_delta_depth, self._deltas())
        self.stats["accepted"] += 1
        return OfferResult(True, 0, len(self.items))

    def take_batch(self) -> tuple[list, list]:
        items = list(self.items)
        self.items.clear()
        kept, dropped = coalesce(items)
        self.stats["coalesced"] += len(dropped)
        return kept, dropped

    def __len__(self) -> int:
        return len(self.items)


class UnboundedInputQueue(InputQueue):
    """Baseline for the backpressure benchmark only: no capacity, no coalescing (every delta processed)."""

    def __init__(self) -> None:
        super().__init__(capacity=10 ** 12)

    def take_batch(self) -> tuple[list, list]:
        items = list(self.items)
        self.items.clear()
        return items, []
