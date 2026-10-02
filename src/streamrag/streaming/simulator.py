"""Deterministic transcript simulator: chunk lists -> timed input events (spec §5), plus JSONL input files.

* ``stream(chunks, interval_ms=200)``  fixed interval; chunk i arrives at i*interval (first at 0, as in the guide)
* ``timed(chunks, words_per_second)``  chunk arrives when its last word has been spoken (seeded jitter)
* ``from_case(BenchmarkCase)``          the case's own chunk timestamps (multi-turn sessions supported)

Real audio can be added later by producing the same TRANSCRIPT_CHUNK / UTTERANCE_END events from an ASR.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from pydantic import TypeAdapter

from streamrag.models.benchmark import BenchmarkCase
from streamrag.models.events import InputEvent, SessionEnd, SessionStart, TranscriptChunk, UtteranceEnd

_ADAPTER = TypeAdapter(InputEvent)


def _utterance_events(session_id: str, utterance_id: str, chunks: list[str], times_s: list[float], end_s: float,
                      offset_s: float) -> list:
    evs: list = [TranscriptChunk(session_id=session_id, utterance_id=utterance_id,
                                 payload={"chunk_index": i, "timestamp_s": round(t, 4), "text": c,
                                          "utterance_offset_s": round(offset_s, 4)})
                 for i, (c, t) in enumerate(zip(chunks, times_s))]
    evs.append(UtteranceEnd(session_id=session_id, utterance_id=utterance_id,
                            payload={"timestamp_s": round(end_s, 4), "last_chunk_index": len(chunks) - 1}))
    return evs


def stream(chunks: list[str], interval_ms: float = 200, end_gap_ms: float = 500, session_id: str = "sim",
           utterance_id: str = "u1", offset_ms: float = 0, wrap_session: bool = True) -> list:
    times = [i * interval_ms / 1000.0 for i in range(len(chunks))]
    evs = _utterance_events(session_id, utterance_id, chunks, times, times[-1] + end_gap_ms / 1000.0, offset_ms / 1000.0)
    return ([SessionStart(session_id=session_id)] + evs + [SessionEnd(session_id=session_id)]) if wrap_session else evs


def timed(chunks: list[str], words_per_second: float = 2.6, end_gap_ms: float = 500, jitter: float = 0.0,
          seed: int = 0, session_id: str = "sim", utterance_id: str = "u1", offset_ms: float = 0,
          wrap_session: bool = True) -> list:
    rng = random.Random(seed)
    t, times = 0.0, []
    for c in chunks:
        dur = max(1, len(c.split())) / words_per_second
        t += dur * (1.0 + rng.uniform(-jitter, jitter))
        times.append(t)
    evs = _utterance_events(session_id, utterance_id, chunks, times, times[-1] + end_gap_ms / 1000.0, offset_ms / 1000.0)
    return ([SessionStart(session_id=session_id)] + evs + [SessionEnd(session_id=session_id)]) if wrap_session else evs


def from_case(case: BenchmarkCase, gap_between_turns_ms: float = 1500) -> list:
    sid = case.session.session_id
    evs: list = [SessionStart(session_id=sid)]
    offset = 0.0
    for turn in case.session.turns:
        chunks = [c.text for c in turn.chunks]
        times = [c.timestamp_s for c in turn.chunks]
        evs += _utterance_events(sid, turn.utterance_id, chunks, times, turn.utterance_end_s, offset)
        offset += turn.utterance_end_s + gap_between_turns_ms / 1000.0
    evs.append(SessionEnd(session_id=sid))
    return evs


def event_time_ms(ev) -> float | None:
    """Session-clock arrival time of an input event (None = 'immediately after the previous event')."""
    if isinstance(ev, TranscriptChunk):
        return ((ev.payload.utterance_offset_s or 0.0) + ev.payload.timestamp_s) * 1000.0
    return None


def write_inputs(path: Path, events: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for e in events:
            f.write(e.canonical_json() + "\n")


def read_inputs(path: Path) -> list:
    return [_ADAPTER.validate_json(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def parse_inline(text: str, sep: str = "|") -> list[str]:
    return [c.strip() for c in text.split(sep) if c.strip()]


def dumps_inputs(events: list) -> str:
    return "\n".join(json.dumps(e.model_dump(mode="json"), sort_keys=True) for e in events)
