"""SessionMemory: the four memory layers + session state versions (Phase 6; docs/session/02).

The layers are separate objects; SessionMemory is the façade that reads them (get_* methods), versions them
(update_session -> SessionStateVersion), snapshots / restores them and resets / archives the session.

What counts as useful state (everything else is not stored):
  transcript   last ``transcript_window`` utterances, redacted; older ones -> hash only (compressed, also in the
               interpretation layer: the tracker keeps the needs and spans of an old utterance, not its text)
  semantic     frames, needs and their versions, constraints with lifecycle, entities (text, terms, spans, needs)
  retrieval    query records (lineage), semantic cache entries, evidence records + per-need lifecycle
  answers      claims (+ links, transitions) and answer versions
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Callable

from streamrag.claims.graph import ClaimGraph
from streamrag.context.detector import FrameManager
from streamrag.delta.evidence import EvidenceStore
from streamrag.delta.planner import SemanticCache
from streamrag.intents.tracker import IntentTracker
from streamrag.ledger.ledger import QueryLedger
from streamrag.answers.manager import AnswerStateManager
from streamrag.session.models import (
    EntityRecord,
    SessionArchive,
    SessionSnapshot,
    SessionState,
    SessionStateVersion,
    TranscriptEntry,
)
from streamrag.session.safety import redact


class SessionMemory:
    def __init__(self, session_id: str, tracker: IntentTracker, ledger: QueryLedger, frames: FrameManager,
                 store: EvidenceStore, cache: SemanticCache, graph: ClaimGraph, answers: AnswerStateManager,
                 window: int = 6, redact_pii: bool = True, config: dict | None = None) -> None:
        self.session_id = session_id
        self.tracker, self.ledger, self.frames, self.store = tracker, ledger, frames, store
        self.cache, self.graph, self.answers = cache, graph, answers
        self.window, self.redact_pii = window, redact_pii
        self.config = dict(config or {})
        # state owned by the session's driver (engine): name -> (export, import), JSON-able; part of every snapshot
        self.extras: dict[str, tuple[Callable[[], object], Callable[[object], None]]] = {}
        self.initialize_session()

    # ------------------------------------------------------------------ lifecycle
    def initialize_session(self) -> None:
        self.transcript: list[TranscriptEntry] = []
        self.entities: dict[str, EntityRecord] = {}
        self.versions: list[SessionStateVersion] = []
        self.current_utterance_id: str | None = None
        if not hasattr(self, "counters"):
            self.counters: dict[str, float] = {}
        self.counters.clear()                         # shared with the engine: always mutated in place

    def reset_session(self) -> None:
        """Clear all active state (every layer); the components keep working for a fresh conversation."""
        self.tracker.__init__(self.tracker.session_id, self.tracker.dec, self.tracker.cfg)
        self.ledger.__init__(self.ledger.session_id)
        self.frames.frames, self.frames._n, self.frames._placed = [], 0, {}
        self.store.__init__()
        self.cache.invalidate("session_reset")
        self.graph.__init__()
        self.answers.__init__(self.tracker, self.frames, self.graph, self.store)
        for _, put in self.extras.values():
            put(None)
        self.initialize_session()

    def archive_session(self, now_ms: float = 0.0) -> SessionArchive:
        changes: dict[str, int] = {}
        for v in self.versions:
            for c in v.summary.split(";"):
                kind = c.strip().split("(")[0].split(" ")[0]
                if kind and kind.isupper():
                    changes[kind] = changes.get(kind, 0) + 1
        q: dict[str, int] = {}
        for r in self.ledger.all():
            q[r.status] = q.get(r.status, 0) + 1
        return SessionArchive(
            session_id=self.session_id, archived_at_ms=now_ms, config_hash=str(self.config.get("config_hash", "")),
            index_content_hash=str(self.config.get("index_content_hash", "")), utterances=len(self.transcript),
            transcript_sha1=[t.text_sha1 for t in self.transcript], session_versions=len(self.versions),
            change_types=dict(sorted(changes.items())), queries=dict(sorted(q.items())),
            evidence_status=self.store.counts(), claim_status=self.graph.status_counts(),
            answer_versions=len(self.answers.versions), counters=dict(self.counters))

    # ------------------------------------------------------------------ layer A: transcript
    def observe_utterance(self, utterance_id: str, text: str, final: bool) -> TranscriptEntry:
        clean, n = redact(text) if self.redact_pii else (text, 0)
        e = TranscriptEntry(utterance_id=utterance_id, text=clean, text_sha1=hashlib.sha1(text.encode()).hexdigest(),
                            chars=len(text), final=final, redactions=n)
        self.transcript = [t for t in self.transcript if t.utterance_id != utterance_id] + [e]
        self.current_utterance_id = utterance_id
        keep = {t.utterance_id for t in self.transcript[-self.window:]}
        for t in self.transcript:
            if t.utterance_id not in keep:
                self.tracker.compress_transcript(t.utterance_id)
        self.transcript = [t if t.utterance_id in keep or t.compressed else
                           t.model_copy(update={"text": "", "compressed": True}) for t in self.transcript]
        return e

    # ------------------------------------------------------------------ layer B: semantic
    def observe_entities(self, utterance_id: str) -> None:
        for it in self.tracker.intents.values():
            if it.utterance_id != utterance_id or it.status != "ACTIVE":
                continue
            for ent in it.entities:
                terms = list(self.tracker.dec.terms_fn(ent))
                if not terms:
                    continue
                key = " ".join(sorted(terms))
                rec = self.entities.get(key)
                if rec is None:
                    self.entities[key] = EntityRecord(text=ent, terms=terms, first_seen=utterance_id,
                                                      last_seen=utterance_id, intent_ids=[it.intent_id])
                else:
                    self.entities[key] = rec.model_copy(update={
                        "last_seen": utterance_id,
                        "intent_ids": rec.intent_ids + ([it.intent_id] if it.intent_id not in rec.intent_ids else [])})

    def get_intent_context(self, intent_id: str | None = None) -> list[dict]:
        frame = self.frames.active
        ids = [intent_id] if intent_id else (frame.intent_ids if frame else [])
        out = []
        for i in ids:
            it = self.tracker.intents.get(i)
            if it is None or it.status != "ACTIVE":
                continue
            out.append({"intent_id": i, "version": it.version, "text": it.resolved_text, "type": it.intent_type,
                        "constraints": [k.text for k in self.tracker.constraints_for(it)],
                        "queries": [r.query_id for r in self.ledger.for_intent(i)]})
        return out

    def get_entity_context(self, frame_only: bool = True) -> list[EntityRecord]:
        frame = self.frames.active
        if not frame_only or frame is None:
            return list(self.entities.values())
        ids = set(frame.intent_ids)
        return [e for e in self.entities.values() if ids & set(e.intent_ids)]

    # ------------------------------------------------------------------ layer C: retrieval
    def get_relevant_evidence(self, intent_id: str) -> list[dict]:
        return [{"evidence_id": a.evidence_id, "status": a.status, "citation": self.store.records[a.evidence_id].citation,
                 "best_rank": a.best_rank} for a in self.store.usable(intent_id)]

    # ------------------------------------------------------------------ versions
    @property
    def version(self) -> int:
        return self.versions[-1].version_id if self.versions else 0

    def snapshot(self) -> SessionSnapshot:
        fr = self.frames.active
        return SessionSnapshot(
            frame_id=fr.frame_id if fr else None, frames={f.frame_id: f.status for f in self.frames.frames},
            intents={i: f"v{it.version}:{it.status}" for i, it in sorted(self.tracker.intents.items())},
            constraints={k: c.status for k, c in sorted(self.tracker.constraints.items())},
            queries={r.query_id: r.status + (":stale" if r.stale else "") for r in self.ledger.all()},
            evidence={f"{e}@{i}": a.status for (e, i), a in sorted(self.store.assign.items())},
            claims={c: cl.status for c, cl in sorted(self.graph.claims.items())},
            answer_id=self.answers.versions[-1].answer_id if self.answers.versions else None)

    def update_session(self, trigger: str, now_ms: float, utterance_id: str | None = None,
                       changes: list[str] | None = None, summary: str = "") -> SessionStateVersion | None:
        """Record a new version if the state snapshot changed (versions are never created for no-ops)."""
        snap = self.snapshot()
        if self.versions and self.versions[-1].state_snapshot == snap and trigger != "reset":
            return None
        v = SessionStateVersion(version_id=self.version + 1, parent_version=self.version or None, created_at_ms=now_ms,
                                utterance_id=utterance_id, trigger=trigger, changes=list(changes or []),
                                summary=summary, state_snapshot=snap)
        self.versions.append(v)
        return v

    def get_current_state(self) -> SessionState:
        fr = self.frames.active
        active_q = [r.query_id for r in self.ledger.all() if not r.stale and r.status not in ("cancelled",)]
        return SessionState(
            session_id=self.session_id, current_utterance_id=self.current_utterance_id, session_version=self.version,
            transcript_state=list(self.transcript), frames=list(self.frames.frames),
            active_frame_id=fr.frame_id if fr else None, intent_set=self.get_intent_context(),
            constraints=[c for c in self.tracker.constraints.values() if c.status == "active"],
            entities=self.get_entity_context(), query_ledger=self.ledger.summary(), active_queries=active_q,
            superseded_queries=[r.query_id for r in self.ledger.all() if r.stale],
            evidence_store=self.store.counts(), claims=self.graph.status_counts(),
            answer_state=self.answers.get_current_answer_state(), telemetry_state=dict(self.counters),
            configuration={k: v for k, v in self.config.items() if isinstance(v, (str, int, float, bool))})

    # ------------------------------------------------------------------ snapshot / restore
    def create_snapshot(self) -> str:
        """Serializable snapshot of every layer (JSON); restore_snapshot() returns the memory to it."""
        def dump(m):
            return m.model_dump(mode="json")
        blob = {
            "transcript": [dump(t) for t in self.transcript], "entities": {k: dump(v) for k, v in self.entities.items()},
            "versions": [dump(v) for v in self.versions], "current_utterance_id": self.current_utterance_id,
            "counters": self.counters, "frames": [dump(f) for f in self.frames.frames], "frames_n": self.frames._n,
            "tracker": self.tracker.export_state(),
            "ledger": [dump(r) for r in self.ledger.all()],
            "evidence_records": {k: dump(v) for k, v in self.store.records.items()},
            "evidence_assign": [dump(a) for a in self.store.assign.values()],
            "cache": {"entries": self.cache.entries, "evidence": self.cache.evidence},
            "claims": {k: dump(v) for k, v in self.graph.claims.items()},
            "links": [dump(v) for v in self.graph.links.values()], "selected": self.graph.selected,
            "answers": [dump(v) for v in self.answers.versions], "status_at": self.answers.status_at,
            "conflicts": {k: [list(x) for x in v] for k, v in self.answers.conflicts.items()},
            "extras": {k: get() for k, (get, _) in self.extras.items()},
        }
        return json.dumps(blob)                     # insertion order is state (ids iterate in creation order)

    def restore_snapshot(self, blob: str) -> None:
        from streamrag.context.models import TopicFrame
        from streamrag.delta.models import EvidenceAssignment, EvidenceRecord
        from streamrag.ledger.models import QueryRecord
        from streamrag.models.answers import AnswerVersion, Claim, ClaimEvidenceLink
        d = json.loads(blob)
        self.transcript = [TranscriptEntry.model_validate(t) for t in d["transcript"]]
        self.entities = {k: EntityRecord.model_validate(v) for k, v in d["entities"].items()}
        self.versions = [SessionStateVersion.model_validate(v) for v in d["versions"]]
        self.current_utterance_id = d["current_utterance_id"]
        self.counters.clear()
        self.counters.update(d["counters"])
        self.frames.frames = [TopicFrame.model_validate(f) for f in d["frames"]]
        self.frames._n, self.frames._placed = int(d["frames_n"]), {}       # placements matter within an utterance only
        self.tracker.import_state(d["tracker"])
        self.ledger._records = {r["query_id"]: QueryRecord.model_validate(r) for r in d["ledger"]}
        self.ledger._order = [r["query_id"] for r in d["ledger"]]
        self.store.records = {k: EvidenceRecord.model_validate(v) for k, v in d["evidence_records"].items()}
        self.store.assign = {(a["evidence_id"], a["intent_id"]): EvidenceAssignment.model_validate(a)
                             for a in d["evidence_assign"]}
        self.cache.entries, self.cache.evidence = dict(d["cache"]["entries"]), copy.deepcopy(d["cache"]["evidence"])
        self.graph.claims = {k: Claim.model_validate(v) for k, v in d["claims"].items()}
        self.graph.links = {(x["claim_id"], x["evidence_id"]): ClaimEvidenceLink.model_validate(x) for x in d["links"]}
        self.graph.selected = {k: list(v) for k, v in d["selected"].items()}
        self.answers.versions = [AnswerVersion.model_validate(v) for v in d["answers"]]
        self.answers.status_at = {k: dict(v) for k, v in d["status_at"].items()}
        self.answers.conflicts = {k: [tuple(x) for x in v] for k, v in d["conflicts"].items()}
        for k, (_, put) in self.extras.items():
            put(d["extras"].get(k))
        # derived indexes
        self.graph._key = {}
        for cid, c in self.graph.claims.items():
            it = self.tracker.intents.get(c.intent_id or "")
            if c.source is not None and it is not None:
                self.graph._key[(it.lineage_root or it.intent_id, c.source.evidence_id, c.source.char_start,
                                 c.source.char_end)] = cid
