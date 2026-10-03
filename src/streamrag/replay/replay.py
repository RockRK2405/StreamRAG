"""ReplayEngine: saved event trace -> reconstructed inputs -> re-run -> compare (deterministic debugging).

A trace is self-describing: CHUNK_RECEIVED carries the original chunk payload (including duplicates and
rejected chunks), UTTERANCE_FINALIZED carries the original UTTERANCE_END payload when ``source == "input"``,
and SESSION_STARTED / SESSION_CLOSED carry the session start/end payloads. System-generated events
(timeouts, implicit finalization, decisions, retrievals) are *not* inputs — they are what the replay must
reproduce.

Exactness: virtual-mode traces replay identically (all fields except wall-clock ones: ``t_wall_ms`` and
``payload.wall``). Realtime traces replay the same inputs in virtual mode, so timestamps (real arrival jitter,
measured latencies) and the ``mode`` / config hash necessarily differ; for them the meaningful check is
``behavior_identical``: same controller decisions, query versions and finalizations per utterance, in order, and
the same retrieval outcomes (query, status, evidence ids). Every exact difference is still listed.

Phase 6 session traces: virtual replay is exact for the session events too (context changes, delta plans, evidence /
claim transitions, session versions, answer versions all derive from logical time). The behaviour signature adds, per
utterance in order, the context changes and the committed answer version (claims compared by text + status, since
claim ids follow result arrival order), and, order-free, the delta plans and the final evidence / claim lifecycle.

Phase 7 traces: LLM outputs are part of the trace (LLM_CALL: request hash + output). Replay feeds them back through a
``RecordedBackend`` instead of calling the model, so a session that used an LLM replays exactly as well.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from streamrag.models.events import SessionEnd, SessionStart, TelemetryEvent, TranscriptChunk, UtteranceEnd
from streamrag.streaming.events import canonical_for_replay
from streamrag.streaming.runner import run_virtual


def read_trace(path: Path) -> list[TelemetryEvent]:
    return [TelemetryEvent.model_validate_json(line) for line in Path(path).read_text(encoding="utf-8").splitlines()
            if line.strip()]


def inputs_from_trace(events: list[TelemetryEvent]) -> list:
    sid = events[0].session_id if events else "sim"
    out: list = []
    for ev in events:
        t, p = ev.type.value, ev.payload
        if t == "SESSION_STARTED" and p.get("input") is not None:
            out.append(SessionStart(session_id=sid, payload=p["input"]))
        elif t == "CHUNK_RECEIVED":
            out.append(TranscriptChunk(session_id=sid, utterance_id=ev.utterance_id, payload=p["input"]))
        elif t == "UTTERANCE_FINALIZED" and p.get("source") == "input":
            out.append(UtteranceEnd(session_id=sid, utterance_id=ev.utterance_id, payload=p["input"]))
        elif t == "ERROR" and p.get("input_event") is not None:
            ie = p["input_event"]
            cls = {"UTTERANCE_END": UtteranceEnd, "TRANSCRIPT_CHUNK": TranscriptChunk, "SESSION_END": SessionEnd,
                   "SESSION_START": SessionStart}[ie["type"]]
            out.append(cls.model_validate(ie))
        elif t == "SESSION_CLOSED" and p.get("input") is not None:
            out.append(SessionEnd(session_id=sid, payload=p["input"]))
    return out


def behavior_signature(events: list[TelemetryEvent]) -> dict:
    """Timing-free behaviour of a run. Per utterance, the ordered decisions, query versions and finalization; plus
    the sorted retrieval outcomes (their interleaving with decisions depends on real completion times)."""
    ordered: dict[str, list] = {}
    outcomes: list = []
    plans: list = []
    claim_text: dict[str, str] = {}
    claim_final: dict[str, str] = {}
    ev_final: dict[str, str] = {}
    for e in events:
        t, p, u = e.type.value, e.payload, e.utterance_id
        if t == "RETRIEVAL_DECISION":
            ordered.setdefault(u, []).append(["decision", p.get("decision"), p.get("reason"), p.get("query_text")])
        elif t in ("QUERY_UPDATED", "QUERY_GENERATED"):
            ordered.setdefault(u, []).append(["query", p.get("query_id"), e.intent_id, p.get("query_text"),
                                              p.get("relation")])
        elif t == "INTENTS_UPDATED":
            ordered.setdefault(u, []).append(["intents", p.get("version"), [i["intent_id"] for i in p["intents"]],
                                              [i["text"] for i in p["intents"]]])
        elif t == "INTENT_SUPERSEDED":
            ordered.setdefault(u, []).append(["superseded", p.get("old"), p.get("new")])
        elif t == "EVIDENCE_FUSED" and p.get("final"):
            ordered.setdefault(u, []).append(["fused", [i["evidence_id"] for i in p["items"]]])
        elif t == "UTTERANCE_FINALIZED":
            ordered.setdefault(u, []).append(["finalized", p.get("reason"), p.get("transcript")])
        elif t == "RETRIEVAL_COMPLETED":
            outcomes.append([u, p.get("query_id"), p.get("status"), list(p.get("evidence_ids") or [])])
        elif t == "RETRIEVAL_CANCELLED":
            outcomes.append([u, p.get("query_id"), "cancelled", []])
        elif t == "CONTEXT_CHANGE_DETECTED":
            ordered.setdefault(u, []).append(["change", p.get("change_type"), p.get("affected_intents"),
                                              p.get("new_intents"), p.get("added_constraints"),
                                              p.get("removed_constraints")])
        elif t == "DELTA_PLAN_CREATED":
            plans.append([u, sorted(q["query"]["text"] for q in p["queries_to_create"]),
                          sorted([q["intent_id"], q["action"]] for q in p["queries_to_reuse"])])
        elif t == "CLAIM_CREATED":
            claim_text[p["claim_id"]] = p["text"]
            claim_final[p["claim_id"]] = p["status"]
        elif t in ("CLAIM_INVALIDATED", "CLAIM_REVALIDATED"):
            claim_final[p["claim_id"]] = p["to_status"]
        elif t in ("EVIDENCE_RETAINED", "EVIDENCE_INVALIDATED", "EVIDENCE_REVALIDATED"):
            ev_final[f"{p['evidence_id']}@{p['intent_id']}"] = p["to_status"]
        elif t in ("ANSWER_VERSION_CREATED", "ANSWER_VERSION_UPDATED"):
            ordered.setdefault(u, []).append(["answer", p.get("kind"), sorted(
                [s["intent_id"], s["status"], len(s["claim_ids"])] for s in p["sections"])])
    sig = {"per_utterance": ordered, "retrieval_outcomes": sorted(outcomes, key=json.dumps)}
    if plans or claim_text:
        sig["delta_plans"] = sorted(plans, key=json.dumps)
        sig["claims_final"] = sorted([claim_text.get(c, c), s] for c, s in claim_final.items())
        sig["evidence_final"] = dict(sorted(ev_final.items()))
    return sig


@dataclass
class ReplayReport:
    identical: bool
    n_original: int
    n_replayed: int
    differences: list[dict] = field(default_factory=list)
    replayed: list[TelemetryEvent] = field(default_factory=list)
    source_mode: str | None = None
    behavior_identical: bool = False
    behavior_differences: list[dict] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """Pass criterion: exact for virtual traces, behaviour-level for realtime traces."""
        return self.identical or (self.source_mode == "realtime" and self.behavior_identical)

    def summary(self) -> dict:
        return {"identical": self.identical, "behavior_identical": self.behavior_identical, "ok": self.ok,
                "n_original": self.n_original, "n_replayed": self.n_replayed,
                "n_differences": len(self.differences), "first_difference": self.differences[0] if self.differences else None,
                "behavior_differences": self.behavior_differences[:5], "source_mode": self.source_mode}


class ReplayEngine:
    def __init__(self, cfg, backend, policy, index_hash: str | None = None, intent_stack=None) -> None:
        self.cfg, self.backend, self.policy, self.index_hash = cfg, backend, policy, index_hash
        self.intent_stack = intent_stack

    def replay(self, original: list[TelemetryEvent], max_diffs: int = 20) -> ReplayReport:
        stack = self.intent_stack
        llm = [e.payload for e in original if e.type.value == "LLM_CALL"]
        if llm and stack is not None and getattr(stack, "grounding", None) is not None:
            # Phase 7: the LLM is not re-run; its recorded outputs are replayed (keyed by request hash)
            from dataclasses import replace

            from streamrag.generation.llm import RecordedBackend
            stack = replace(stack, grounding=stack.grounding.with_backend(RecordedBackend(llm)))
        run = run_virtual(self.cfg, self.backend, self.policy, inputs_from_trace(original), index_hash=self.index_hash,
                          intent_stack=stack)
        a = [canonical_for_replay(e) for e in original]
        b = [canonical_for_replay(e) for e in run.events]
        diffs = []
        for i in range(max(len(a), len(b))):
            x = a[i] if i < len(a) else None
            y = b[i] if i < len(b) else None
            if x != y:
                diffs.append({"index": i, "original": x, "replayed": y})
                if len(diffs) >= max_diffs:
                    break
        mode = next((e.payload.get("mode") for e in original if e.type.value == "SESSION_STARTED"), None)
        sa, sb = behavior_signature(original), behavior_signature(run.events)
        bdiffs = []
        for u in sorted(set(sa["per_utterance"]) | set(sb["per_utterance"]), key=str):
            x, y = sa["per_utterance"].get(u), sb["per_utterance"].get(u)
            if x != y:
                bdiffs.append({"utterance_id": u, "original": x, "replayed": y})
        if sa["retrieval_outcomes"] != sb["retrieval_outcomes"]:
            bdiffs.append({"retrieval_outcomes": {"original": sa["retrieval_outcomes"], "replayed": sb["retrieval_outcomes"]}})
        return ReplayReport(not diffs and len(a) == len(b), len(a), len(b), diffs, run.events, mode,
                            behavior_identical=not bdiffs, behavior_differences=bdiffs)

    def replay_file(self, path: Path) -> ReplayReport:
        return self.replay(read_trace(path))


def dump_trace(events: list[TelemetryEvent], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(e.canonical_json() for e in events) + "\n", encoding="utf-8")


def diff_to_text(report: ReplayReport) -> str:
    return json.dumps(report.summary(), indent=2, default=str)
