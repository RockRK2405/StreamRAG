"""ContextChangeDetector and topic frames (Phase 6; docs/session/03, 10).

Input: the previous interpretation (the engine's last-known intent versions) + the tracker's new interpretation
(``IntentSetDelta`` of the update, the decomposition, the constraint registry). Output: typed ``ContextChange``s.
The comparison is semantic - analyzed term sets, topic, facet and constraint ids of each need - never raw strings:
"requirements for X" -> "requirements for X for international applicants" is a CONSTRAINT_ADDITION on the same
need, not a new need.

Change confidence (computed, never authored) = the lowest confidence among the interpretation elements the change
rests on: the decomposer's confidence of each new / modified need and the scope_confidence of each added
constraint (removals and supersessions that follow an explicit marker count 1.0).

Topic frames: new needs join the active frame when they are related to it (FOLLOW_UP / DEPENDENT / REFINEMENT
relation, an elliptical follow-up decision, or >= ``frame_overlap_min`` shared topic terms); otherwise the active
frame goes dormant and a new frame opens (context isolation: nothing of the old frame is applied to it). A dormant
frame whose topic terms the new needs share is reactivated, and so is a dormant frame whose need the update changed
(a late detail aimed at an earlier topic: "Back to X, what about Z?").
"""

from __future__ import annotations

from collections.abc import Callable

from streamrag.context.models import ContextChange, SemanticDiff, TopicFrame
from streamrag.intents.decomposer import Decomposition
from streamrag.intents.tracker import IntentTracker
from streamrag.models.intents import Intent, IntentSetDelta


class FrameManager:
    def __init__(self, overlap_min: int = 1) -> None:
        self.overlap_min = overlap_min
        self.frames: list[TopicFrame] = []
        self._n = 0                                           # frame ids are never reused (pruned frames included)

    @property
    def active(self) -> TopicFrame | None:
        return next((f for f in self.frames if f.status == "active"), None)

    def frame_of(self, intent_id: str) -> TopicFrame | None:
        return next((f for f in self.frames if intent_id in f.intent_ids), None)

    def _replace(self, frame: TopicFrame) -> None:
        self.frames = [frame if f.frame_id == frame.frame_id else f for f in self.frames]

    def _prune(self, uid: str, tracker: IntentTracker) -> None:
        """A frame opened in this utterance whose needs all disappeared again (a mid-stream fragment that turned
        out to be a late detail) is dropped and the frame it displaced becomes active again."""
        act = self.active
        if act is None or act.opened_in != uid or act.last_answer_version is not None:
            return
        if any(tracker.intents.get(i) is not None and tracker.intents[i].status == "ACTIVE" for i in act.intent_ids):
            return
        self.frames = [f for f in self.frames if f.frame_id != act.frame_id]
        dormant = [f for f in self.frames if f.status == "dormant"]
        if dormant:
            self._replace(dormant[-1].model_copy(update={"status": "active"}))

    def assign(self, uid: str, delta: IntentSetDelta, d: Decomposition, tracker: IntentTracker) -> str:
        """Place the update's new needs; returns 'same' | 'new_frame' | 'none'."""
        if delta.removed:
            self._prune(uid, tracker)
        new = [i for i in delta.added if i in tracker.intents]
        corr_new = [s.new for s in delta.superseded if s.new and s.new not in new]
        if not new and not corr_new:
            back = [f for m in delta.modified if (f := self.frame_of(m.intent_id)) is not None and f.status == "dormant"]
            if back:
                act = self.active
                if act is not None:
                    self._replace(act.model_copy(update={"status": "dormant"}))
                self._replace(self.frame_of(back[-1].intent_ids[0]).model_copy(update={"status": "active"}))
                return "reactivated"
            return "none" if not (delta.modified or delta.removed or delta.superseded) else "same"
        act = self.active
        topic = {i: set(_topic_terms(tracker, i)) for i in new + corr_new}
        if act is None:
            self._open(uid, new + corr_new, topic)
            return "new_frame"
        rel_targets = {r.target for i in new for r in _relations(tracker, uid) if r.source == i}
        corrects_frame = any(s.old in act.intent_ids for s in delta.superseded)
        related = bool(corr_new) or corrects_frame or bool(rel_targets & set(act.intent_ids)) or bool(delta.cross_turn) \
            or any(f.get("decision") in ("parallel", "constraint") for f in d.follow_ups) \
            or any(len(t & set(act.topic_terms)) >= max(1, self.overlap_min) for t in topic.values())
        if related:
            self._replace(act.model_copy(update={
                "intent_ids": act.intent_ids + [i for i in new + corr_new if i not in act.intent_ids],
                "topic_terms": sorted(set(act.topic_terms).union(*topic.values()))}))
            return "same"
        dormant = [f for f in self.frames if f.status == "dormant"
                   and any(len(t & set(f.topic_terms)) >= max(1, self.overlap_min) for t in topic.values())]
        self._replace(act.model_copy(update={"status": "dormant"}))
        if dormant:                                           # back to an earlier topic
            f = dormant[-1]
            self._replace(f.model_copy(update={"status": "active",
                                               "intent_ids": f.intent_ids + [i for i in new if i not in f.intent_ids],
                                               "topic_terms": sorted(set(f.topic_terms).union(*topic.values()))}))
            return "reactivated"
        self._open(uid, new + corr_new, topic)
        return "new_frame"

    def _open(self, uid: str, intents: list[str], topic: dict[str, set[str]]) -> None:
        self._n += 1
        self.frames.append(TopicFrame(frame_id=f"T{self._n}", intent_ids=list(intents), status="active",
                                      opened_in=uid, topic_terms=sorted(set().union(*topic.values()))))


def _topic_terms(tracker: IntentTracker, iid: str) -> list[str]:
    it = tracker.intents[iid]
    if it.topic:
        return list(tracker.dec.terms_fn(it.topic))
    return list(tracker.terms.get(iid, []))


def _relations(tracker: IntentTracker, uid: str):
    cur = tracker.current(uid)
    return cur.relationships if cur else []


def semantic_diff(old: Intent | None, new: Intent | None, tracker: IntentTracker) -> SemanticDiff:
    terms = tracker.dec.terms_fn

    def tset(i: Intent | None) -> set[str]:
        return set(terms(i.resolved_text + " " + " ".join(c.text for c in i.inherited_context))) if i else set()
    a, b = tset(old), tset(new)
    ref = new or old
    return SemanticDiff(
        intent_id=ref.intent_id, from_version=old.version if old else None, to_version=new.version if new else None,
        terms_added=sorted(b - a), terms_removed=sorted(a - b), topic_before=old.topic if old else None,
        topic_after=new.topic if new else None, aspect_before=old.aspect if old else None,
        aspect_after=new.aspect if new else None,
        constraints_added=sorted(set(new.constraint_ids if new else []) - set(old.constraint_ids if old else [])),
        constraints_removed=sorted(set(old.constraint_ids if old else []) - set(new.constraint_ids if new else [])))


class ContextChangeDetector:
    def __init__(self, tracker: IntentTracker) -> None:
        self.tracker = tracker
        self._n = 0

    def _id(self) -> str:
        self._n += 1
        return f"CH{self._n}"

    def detect(self, uid: str, delta: IntentSetDelta | None, d: Decomposition | None, known: dict[str, Intent],
               frame_action: str, active_query: Callable[[str], str | None], session_version: int,
               now_ms: float) -> list[ContextChange]:
        tr = self.tracker
        if delta is None or delta.empty:
            return []
        out: list[ContextChange] = []
        corr_new = set()
        rels = _relations(tr, uid)
        for sup in delta.superseded:                                         # explicit corrections
            old, new = known.get(sup.old) or tr.intents.get(sup.old), tr.intents.get(sup.new)
            diff = semantic_diff(old, new, tr)
            corr_new.add(sup.new)
            entity = bool(old and new and old.topic and new.topic
                          and not set(tr.dec.terms_fn(old.topic)) & set(tr.dec.terms_fn(new.topic)))
            out.append(self._mk("CORRECTION", uid, affected=[sup.old], new_intents=[sup.new] if new else [],
                                superseded=[sup.old], relation="correction", frame_action=frame_action, diffs=[diff],
                                cue=f"{sup.cue}{'; entity_replacement' if entity else ''}",
                                conf=[(f"intent:{sup.new}", new.confidence)] if new else [], aq=[active_query(sup.old)],
                                sv=session_version, at=now_ms))
        for iid in delta.added:
            if iid in corr_new or iid not in tr.intents:
                continue
            it = tr.intents[iid]
            follow = any(r.source == iid and r.type in ("FOLLOW_UP", "DEPENDENT") for r in rels) \
                or any(f.get("decision") == "parallel" for f in (d.follow_ups if d else []))
            out.append(self._mk("NEW_INTENT", uid, new_intents=[iid], relation="follow_up" if follow else "independent",
                                frame_action=frame_action, diffs=[semantic_diff(None, it, tr)],
                                cue="follow_up" if follow else "new_need", conf=[(f"intent:{iid}", it.confidence)],
                                sv=session_version, at=now_ms))
        for m in delta.modified:
            old, new = known.get(m.intent_id), tr.intents.get(m.intent_id)
            if new is None:
                continue
            diff = semantic_diff(old, new, tr)
            aq = [active_query(m.intent_id)]
            if diff.constraints_added:
                ks = [tr.constraints[k] for k in diff.constraints_added if k in tr.constraints]
                out.append(self._mk("CONSTRAINT_ADDITION", uid, affected=[m.intent_id], relation="same_need",
                                    frame_action=frame_action, added=[k.text for k in ks], diffs=[diff],
                                    cue=",".join(sorted({k.scope_reason for k in ks})),
                                    conf=[(f"scope:{k.constraint_id}", k.scope_confidence) for k in ks], aq=aq,
                                    sv=session_version, at=now_ms))
            if diff.constraints_removed:
                ks = [tr.constraints[k] for k in diff.constraints_removed if k in tr.constraints]
                out.append(self._mk("CONSTRAINT_REMOVAL", uid, affected=[m.intent_id], relation="same_need",
                                    frame_action=frame_action, removed=[k.text for k in ks], diffs=[diff],
                                    cue="retracted" if all(not k.replaces for k in ks) else "updated", aq=aq,
                                    sv=session_version, at=now_ms))
            if "text" in m.changed or "context" in m.changed:
                kind = self._text_change(old, new, diff)
                out.append(self._mk(kind, uid, affected=[m.intent_id], relation="same_need", frame_action=frame_action,
                                    diffs=[diff], cue=",".join(m.changed), conf=[(f"intent:{m.intent_id}", new.confidence)],
                                    aq=aq, sv=session_version, at=now_ms))
        for iid in delta.removed:
            out.append(self._mk("INTENT_REMOVAL", uid, affected=[iid], relation="none", frame_action=frame_action,
                                diffs=[semantic_diff(known.get(iid), None, tr) if known.get(iid) else
                                       SemanticDiff(intent_id=iid)], cue="no_longer_in_decomposition",
                                aq=[active_query(iid)], sv=session_version, at=now_ms))
        return out

    def _text_change(self, old: Intent | None, new: Intent, diff: SemanticDiff) -> str:
        if old is None or not diff.terms_removed:
            return "REFINEMENT"                                   # terms only added: more specific
        terms = self.tracker.dec.terms_fn
        if old.topic and new.topic and not set(terms(old.topic)) & set(terms(new.topic)):
            return "ENTITY_CHANGE"
        if (old.aspect or "") != (new.aspect or "") or old.intent_type != new.intent_type:
            return "QUESTION_CHANGE"
        return "REFINEMENT"

    def _mk(self, kind, uid, affected=(), new_intents=(), superseded=(), added=(), removed=(), relation="none",
            frame_action="none", diffs=(), cue="", conf=(), aq=(), sv=0, at=0.0) -> ContextChange:
        signals = {k: round(float(v), 3) for k, v in conf}
        return ContextChange(change_id=self._id(), change_type=kind, utterance_id=uid, affected_intents=list(affected),
                             new_intents=list(new_intents), superseded_intents=list(superseded),
                             added_constraints=list(added), removed_constraints=list(removed),
                             affected_queries=[q for q in aq if q], relation=relation, frame_action=frame_action,
                             diffs=list(diffs), cue=cue, confidence=round(min(signals.values(), default=1.0), 3),
                             confidence_signals=signals, session_version_from=sv, at_ms=at)


def net_change_types(changes: list[ContextChange]) -> list[str]:
    """Turn-level view of the changes one streamed utterance produced provisionally (docs/session/03).

    While an utterance streams, its interpretation is revised tick by tick; the turn's *net* effect is:
      * changes to a need created in this turn are part of its creation (only NEW_INTENT / CORRECTION remain);
      * a need created and removed again in the turn (a fragment) leaves no trace;
      * a constraint added and removed again in the turn cancels (a provisional value that was revised).
    The individual provisional changes stay in the event trace; this only summarises them.
    """
    created = {i for c in changes if c.change_type in ("NEW_INTENT", "CORRECTION") for i in c.new_intents}
    removed_intents = {i for c in changes if c.change_type == "INTENT_REMOVAL" for i in c.affected_intents}
    k_added = {k for c in changes if c.change_type == "CONSTRAINT_ADDITION" for d in c.diffs for k in d.constraints_added}
    k_removed = {k for c in changes if c.change_type == "CONSTRAINT_REMOVAL" for d in c.diffs
                 for k in d.constraints_removed}
    out = []
    for c in changes:
        t = c.change_type
        if t == "NEW_INTENT" and set(c.new_intents) <= removed_intents:
            continue
        if t == "INTENT_REMOVAL" and set(c.affected_intents) <= created:
            continue
        if t in ("REFINEMENT", "ENTITY_CHANGE", "QUESTION_CHANGE", "CONSTRAINT_ADDITION", "CONSTRAINT_REMOVAL") \
                and c.affected_intents and set(c.affected_intents) <= created:
            continue
        if t == "CONSTRAINT_ADDITION" and not ({k for d in c.diffs for k in d.constraints_added} - k_removed):
            continue
        if t == "CONSTRAINT_REMOVAL" and not ({k for d in c.diffs for k in d.constraints_removed} - k_added):
            continue
        out.append(t)
    return out or (["NO_CHANGE"] if changes else [])
