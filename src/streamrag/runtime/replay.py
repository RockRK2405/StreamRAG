"""Replay and event-sourced reconstruction of runtime sessions (docs/runtime/11).

* ``inputs_from_runtime_trace`` recovers every input of a session with its arrival time from the event log alone:
  chunks carry ``utterance_offset_s + timestamp_s`` (arrival on the runtime clock), coalesced and rejected inputs are
  logged with their payloads, utterance ends with their utterance-relative time, the session end with SESSION_UPDATED.
* ``replay_runtime`` re-runs those inputs through a **virtual-clock** runtime with the recorded LLM outputs (LLM_CALL)
  and the recorded fault configuration. A virtual-mode trace replays exactly (every field except wall-clock
  measurements). A realtime trace replays its orchestration decisions on a deterministic clock: the decisions that
  depend on wall timing (which deltas were coalesced, when a cancellation reached a worker, whether a deadline was
  hit) can differ - ``behaviour`` compares the timing-independent outcome instead.
* ``reconstruct`` rebuilds the session state (queries, evidence, claims, answers) purely from events.
"""

from __future__ import annotations

from streamrag.generation.llm import RecordedBackend
from streamrag.models.events import SessionEnd, SessionEndPayload, TelemetryEvent, TranscriptChunk, UtteranceEnd
from streamrag.runtime.faults import Fault, FaultInjector
from streamrag.streaming.events import canonical_for_replay


def _chunk_arrival(d: dict) -> float:
    p = d["payload"]
    return round(((p.get("utterance_offset_s") or 0.0) + p["timestamp_s"]) * 1000.0, 3)


def inputs_from_runtime_trace(events: list[TelemetryEvent]) -> list[tuple[float, object]]:
    out: list[tuple[float, int, object]] = []
    offsets: dict[str, float] = {}
    sid = events[0].session_id

    def chunk(d: dict) -> None:
        ev = TranscriptChunk.model_validate(d)
        offsets.setdefault(ev.utterance_id, (ev.payload.utterance_offset_s or 0.0) * 1000.0)
        out.append((_chunk_arrival(d), len(out), ev))

    for e in events:
        t, p = e.type.value, e.payload
        if t == "CHUNK_RECEIVED" and p.get("input") is not None:
            chunk({"session_id": sid, "utterance_id": e.utterance_id, "payload": p["input"]})
        elif t == "TRANSCRIPT_COALESCED":
            for d in p.get("dropped_inputs", []):
                chunk(d)
        elif t == "BACKPRESSURE_APPLIED" and p.get("action") == "rejected" and p.get("input"):
            d = p["input"]
            if d.get("type") == "TRANSCRIPT_CHUNK":
                out.append((p["arrival_ms"], len(out), TranscriptChunk.model_validate(d)))
        elif t == "UTTERANCE_FINALIZED" and p.get("source") == "input" and p.get("input") is not None:
            ue = UtteranceEnd(session_id=sid, utterance_id=e.utterance_id, payload=p["input"])
            out.append((round(offsets.get(e.utterance_id, 0.0) + ue.payload.timestamp_s * 1000.0, 3), len(out), ue))
        elif t == "SESSION_UPDATED" and p.get("input") == "SESSION_END":
            out.append((p["arrival_ms"], len(out),
                        SessionEnd(session_id=sid, payload=SessionEndPayload(**p.get("payload", {})))))
    out.sort(key=lambda x: (x[0], x[1]))
    return [(arrival, ev) for arrival, _, ev in out]


def _faults(events: list[TelemetryEvent]) -> FaultInjector | None:
    for e in events:
        if e.type.value == "SESSION_STARTED":
            specs = (e.payload.get("runtime") or {}).get("faults") or []
            return FaultInjector([Fault(**f) for f in specs]) if specs else None
    return None


def replay_runtime(cfg, stack, events: list[TelemetryEvent], max_diffs: int = 20) -> dict:
    from streamrag.runtime.runtime import StreamingRuntime
    sid = events[0].session_id
    records = [e.payload for e in events if e.type.value == "LLM_CALL"]
    rt = StreamingRuntime(cfg, stack, mode="virtual", llm=RecordedBackend(records) if records else None,
                          faults=_faults(events))
    rt.start_session(sid)
    for t, ev in inputs_from_runtime_trace(events):
        rt.push(ev, at_ms=t)
    rt.run()
    replayed = rt.events(sid)
    a = [canonical_for_replay(e) for e in events]
    b = [canonical_for_replay(e) for e in replayed]
    diffs = []
    for i in range(max(len(a), len(b))):
        x, y = (a[i] if i < len(a) else None), (b[i] if i < len(b) else None)
        if x != y:
            diffs.append({"index": i, "original": x, "replayed": y})
            if len(diffs) >= max_diffs:
                break
    return {"identical": not diffs and len(a) == len(b), "n_original": len(a), "n_replayed": len(b),
            "differences": diffs, "behaviour_identical": behaviour(events) == behaviour(replayed),
            "events": replayed}


def behaviour(events: list[TelemetryEvent]) -> dict:
    """Timing-independent outcome: per utterance the final transcript, the queries issued (text, final status),
    and the validated answer (claims) - what a user and the ledger see, not when."""
    out: dict = {}
    q = reconstruct(events)
    for e in events:
        if e.type.value == "UTTERANCE_FINALIZED":
            out.setdefault(e.utterance_id, {})["transcript"] = e.payload.get("transcript")
    for qid, r in q["queries"].items():
        out.setdefault(r["utterance_id"], {}).setdefault("queries", []).append((r["text"], r["status"]))
    for aid, a in q["answers"].items():
        if a["status"] == "VALIDATED_FINAL":
            out.setdefault(a["utterance_id"], {})["answer_claims"] = sorted(a["claims"])
    for v in out.values():
        v["queries"] = sorted(v.get("queries", []))
    return out


def reconstruct(events: list[TelemetryEvent]) -> dict:
    """Session state rebuilt from events only (event sourcing)."""
    queries: dict[str, dict] = {}
    evidence: dict[str, list[str]] = {}
    claims: dict[str, str] = {}
    answers: dict[str, dict] = {}
    for e in events:
        t, p = e.type.value, e.payload
        if t in ("QUERY_GENERATED", "QUERY_UPDATED"):
            queries[p["query_id"]] = {"text": p.get("query_text"), "intent_id": e.intent_id,
                                      "utterance_id": e.utterance_id, "status": "queued", "superseded_by": None}
        elif t == "RETRIEVAL_STARTED" and p.get("query_id") in queries:
            queries[p["query_id"]]["status"] = "in_flight"
        elif t == "RETRIEVAL_COMPLETED" and p.get("query_id") in queries:
            ok = p.get("status") in ("ok", "degraded", "empty", "partial")
            queries[p["query_id"]]["status"] = "completed" if ok else "failed"
            if ok and not p.get("stale"):
                evidence[p["query_id"]] = list(p.get("evidence_ids", []))
        elif t == "RETRIEVAL_CANCELLED" and p.get("query_id") in queries:
            queries[p["query_id"]]["status"] = "cancelled"
        elif t == "QUERY_SUPERSEDED" and p.get("query_id") in queries:
            queries[p["query_id"]]["superseded_by"] = p.get("superseded_by")
        elif t in ("CLAIM_CREATED", "CLAIM_INVALIDATED", "CLAIM_REVALIDATED"):
            claims[p["claim_id"]] = p.get("to_status") or p.get("status") or "SUPPORTED"
        elif t == "ANSWER_COMMITTED":
            answers[p["answer_id"]] = {"status": p["status"], "version": p["version"], "claims": p.get("claims", []),
                                       "utterance_id": e.utterance_id, "text": p.get("text")}
    return {"queries": queries, "evidence": evidence, "claims": claims, "answers": answers}
