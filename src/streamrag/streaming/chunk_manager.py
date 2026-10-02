"""TranscriptChunkManager: validated, idempotent accumulation of transcript chunks per utterance.

Statuses returned by ``append_chunk`` (never silently corrupts the transcript):
  accepted            next expected chunk
  duplicate           same index + same text -> ignored
  revision            same index + different text, or ``replaces_chunk_index`` -> replaced (ASR revision)
  out_of_order        index below the highest seen, filling a gap -> inserted in index order
  gap                 index skips ahead -> accepted, missing indices recorded (``missing_indices``)
  empty               empty/whitespace text -> recorded (keeps index bookkeeping) but no transcript change
  rejected_finalized  chunk for an utterance that has already ended -> rejected (explicit error upstream)
  rejected_closed     session closed -> rejected

The transcript is always rebuilt from chunks in ``chunk_index`` order.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

from streamrag.models.events import TranscriptChunk

Status = Literal["accepted", "duplicate", "revision", "out_of_order", "gap", "empty", "rejected_finalized",
                 "rejected_closed"]
_NO_SPACE_BEFORE = re.compile(r"^[,.;:?!)\]'’]")


@dataclass
class ChunkRecord:
    session_id: str
    utterance_id: str
    chunk_index: int
    received_at_ms: float      # session stream clock when the chunk was received
    timestamp_s: float         # input timestamp (utterance-relative)
    text: str
    stability: str
    is_final: bool             # ASR hypothesis is final (not "partial"); utterance end is a separate event


@dataclass
class UtteranceState:
    utterance_id: str
    start_ms: float
    chunks: dict[int, ChunkRecord] = field(default_factory=dict)
    max_index: int = -1
    finalized: bool = False
    finalized_at_ms: float | None = None
    end_reason: str | None = None
    first_received_ms: float | None = None
    last_received_ms: float | None = None
    revisions: int = 0


@dataclass
class AppendResult:
    status: Status
    changed: bool
    missing: list[int]
    transcript: str


class TranscriptChunkManager:
    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        self.utterances: dict[str, UtteranceState] = {}
        self.order: list[str] = []
        self.closed = False

    # ------------------------------------------------------------------ queries
    def open_utterance(self) -> str | None:
        for u in reversed(self.order):
            if not self.utterances[u].finalized:
                return u
        return None

    def state(self, utterance_id: str) -> UtteranceState:
        return self.utterances[utterance_id]

    def is_finalized(self, utterance_id: str) -> bool:
        return utterance_id in self.utterances and self.utterances[utterance_id].finalized

    def missing_indices(self, utterance_id: str) -> list[int]:
        st = self.utterances[utterance_id]
        return [i for i in range(st.max_index + 1) if i not in st.chunks]

    def get_current_transcript(self, utterance_id: str | None = None) -> str:
        uid = utterance_id or (self.order[-1] if self.order else None)
        if uid is None:
            return ""
        out = ""
        for i in sorted(self.utterances[uid].chunks):
            t = self.utterances[uid].chunks[i].text.strip()
            if not t:
                continue
            out = t if not out else (out + t if _NO_SPACE_BEFORE.match(t) else out + " " + t)
        return out

    def get_recent_chunks(self, n: int = 5, utterance_id: str | None = None) -> list[ChunkRecord]:
        uid = utterance_id or (self.order[-1] if self.order else None)
        if uid is None:
            return []
        recs = sorted(self.utterances[uid].chunks.values(), key=lambda r: r.received_at_ms)
        return recs[-n:]

    # ------------------------------------------------------------------ mutation
    def append_chunk(self, ev: TranscriptChunk, now_ms: float) -> AppendResult:
        p, uid = ev.payload, ev.utterance_id
        if self.closed:
            return AppendResult("rejected_closed", False, [], "")
        if uid not in self.utterances:
            start = p.utterance_offset_s * 1000.0 if p.utterance_offset_s is not None else now_ms - p.timestamp_s * 1000.0
            self.utterances[uid] = UtteranceState(uid, start_ms=max(start, 0.0))
            self.order.append(uid)
        st = self.utterances[uid]
        if st.finalized:
            return AppendResult("rejected_finalized", False, self.missing_indices(uid), self.get_current_transcript(uid))
        rec = ChunkRecord(self.session_id, uid, p.chunk_index, now_ms, p.timestamp_s, p.text, p.stability,
                          p.stability == "final")
        st.first_received_ms = now_ms if st.first_received_ms is None else st.first_received_ms
        st.last_received_ms = now_ms
        target = p.replaces_chunk_index if p.replaces_chunk_index is not None else p.chunk_index
        existing = st.chunks.get(target)
        if existing is not None:
            if existing.text == p.text and p.replaces_chunk_index is None:
                return AppendResult("duplicate", False, self.missing_indices(uid), self.get_current_transcript(uid))
            st.chunks[target] = rec
            st.revisions += 1
            return AppendResult("revision", True, self.missing_indices(uid), self.get_current_transcript(uid))
        before_max = st.max_index
        st.chunks[target] = rec
        st.max_index = max(st.max_index, target)
        if not p.text.strip():
            return AppendResult("empty", False, self.missing_indices(uid), self.get_current_transcript(uid))
        if target < before_max:
            status: Status = "out_of_order"
        elif target > before_max + 1:
            status = "gap"
        else:
            status = "accepted"
        return AppendResult(status, True, self.missing_indices(uid), self.get_current_transcript(uid))

    def finalize_utterance(self, utterance_id: str, now_ms: float, reason: str) -> bool:
        st = self.utterances.get(utterance_id)
        if st is None or st.finalized:
            return False
        st.finalized, st.finalized_at_ms, st.end_reason = True, now_ms, reason
        return True

    def ensure_utterance(self, utterance_id: str, start_ms: float) -> None:
        if utterance_id not in self.utterances:
            self.utterances[utterance_id] = UtteranceState(utterance_id, start_ms=start_ms)
            self.order.append(utterance_id)

    def reset(self) -> None:
        self.utterances.clear()
        self.order.clear()
        self.closed = False

    def close(self) -> None:
        self.closed = True
