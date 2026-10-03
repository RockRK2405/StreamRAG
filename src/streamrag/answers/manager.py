"""Answer state: versioned, sectioned, claim-level (Phase 6; docs/session/08). Rendering text is Phase 7.

An answer is the structured state of one topic frame: one ``AnswerSection`` per active need (in order of mention),
each listing the claims currently usable for that need (SUPPORTED / PARTIALLY_SUPPORTED, selected for the need's
current version), their evidence, the need's active constraints and its uncertainty items.

A new ``AnswerVersion`` is committed only when something changed (first state of a frame, or a non-empty diff).
The diff lists kept (unchanged) / modified (same claim, new status or need version) / added / retracted claims and
added / removed / changed / unchanged sections. Sections whose status is new or changed carry
``needs_regeneration = true``: the exact unit Phase 7 must re-render; unchanged sections are reused verbatim
(minimal regeneration, spec §14.2 step 7).
"""

from __future__ import annotations

from streamrag.claims.graph import ClaimGraph
from streamrag.context.detector import FrameManager
from streamrag.context.models import ContextChange, TopicFrame
from streamrag.delta.evidence import EvidenceStore
from streamrag.intents.tracker import IntentTracker
from streamrag.models.answers import AnswerDiff, AnswerSection, AnswerVersion, ClaimChange, UncertaintyItem

SHOWN = ("SUPPORTED", "PARTIALLY_SUPPORTED")


def _id_num(x: str) -> int:
    digits = "".join(ch for ch in x if ch.isdigit())
    return int(digits) if digits else 0


class AnswerStateManager:
    def __init__(self, tracker: IntentTracker, frames: FrameManager, graph: ClaimGraph, store: EvidenceStore) -> None:
        self.tracker, self.frames, self.graph, self.store = tracker, frames, graph, store
        self.versions: list[AnswerVersion] = []
        self.conflicts: dict[str, list[tuple[str, str, str]]] = {}     # intent -> potential conflicts
        self.status_at: dict[str, dict[str, str]] = {}                  # answer id -> claim statuses when committed

    # ------------------------------------------------------------------ API (brief §25)
    def get_current_answer_state(self, frame_id: str | None = None) -> AnswerVersion | None:
        fid = frame_id or (self.frames.active.frame_id if self.frames.active else None)
        own = [v for v in self.versions if v.topic_id == fid]
        return own[-1] if own else None

    def create_answer_state(self, utterance_id: str, session_version: int, now_ms: float,
                            changes: list[ContextChange] | None = None, delta_queries: list[str] | None = None,
                            full_rerun: bool = False) -> AnswerVersion | None:
        """Initial state of the active frame (first version)."""
        return self.update_answer_state(utterance_id, session_version, now_ms, changes, delta_queries, full_rerun)

    def update_answer_state(self, utterance_id: str, session_version: int, now_ms: float,
                            changes: list[ContextChange] | None = None, delta_queries: list[str] | None = None,
                            full_rerun: bool = False) -> AnswerVersion | None:
        """Commit a new version if the active frame's answer state changed; returns it (or None)."""
        frame = self.frames.active
        if frame is None:
            return None
        prev = self.get_current_answer_state(frame.frame_id)
        sections = self._sections(frame, prev)
        cand = self._version(frame, sections, prev, utterance_id, session_version, now_ms)
        diff = self.compare_answer_versions(prev, cand)
        if prev is not None and not _diff_nonempty(diff):
            return None
        cand = cand.model_copy(update={"diff": diff, "delta_queries": list(delta_queries or []),
                                       "full_rerun": full_rerun,
                                       "change_summary": _summary(changes or [], diff, sections)})
        self.versions.append(cand)
        self.status_at[cand.answer_id] = {c: self.graph.claims[c].status for c in cand.claim_ids}
        self.frames._replace(frame.model_copy(update={"last_answer_version": cand.version}))
        for cid in cand.claim_ids:                                     # introduced_in / modified_in bookkeeping
            c = self.graph.claims[cid]
            upd = {}
            if c.introduced_in is None:
                upd["introduced_in"] = cand.version
            elif any(m.claim_id == cid for m in diff.modified):
                upd["modified_in"] = c.modified_in + [cand.version]
            if upd:
                self.graph.claims[cid] = c.model_copy(update=upd)
        return cand

    def compare_answer_versions(self, a: AnswerVersion | None, b: AnswerVersion) -> AnswerDiff:
        prev_claims = {c: self._claim_sig(c, a) for c in (a.claim_ids if a else [])}
        new_claims = {c: self._claim_sig(c, b) for c in b.claim_ids}
        kept = [c for c in b.claim_ids if c in prev_claims and prev_claims[c] == new_claims[c]]
        modified = [ClaimChange(claim_id=c, from_text=self.graph.claims[c].text, to_text=self.graph.claims[c].text,
                                from_citations=prev_claims[c][2], to_citations=new_claims[c][2],
                                from_status=prev_claims[c][0], to_status=new_claims[c][0])
                    for c in b.claim_ids if c in prev_claims and prev_claims[c] != new_claims[c]]
        added = [c for c in b.claim_ids if c not in prev_claims]
        retracted = [c for c in (a.claim_ids if a else []) if c not in new_claims]
        ps = {s.section_id: s for s in (a.sections if a else [])}
        ns = {s.section_id: s for s in b.sections}
        sec_changed = [s for s in ns if s in ps and self._sec_sig(ns[s]) != self._sec_sig(ps[s])]
        return AnswerDiff(
            kept=kept, modified=modified, added=added, retracted=retracted,
            citations_added=[c for c in b.citations if c not in (a.citations if a else [])],
            citations_removed=[c for c in (a.citations if a else []) if c not in b.citations],
            evidence_added=[e for e in b.evidence_ids if e not in (a.evidence_ids if a else [])],
            evidence_removed=[e for e in (a.evidence_ids if a else []) if e not in b.evidence_ids],
            sections_added=[s for s in ns if s not in ps], sections_removed=[s for s in ps if s not in ns],
            sections_changed=sec_changed, sections_unchanged=[s for s in ns if s in ps and s not in sec_changed],
            uncertainty_introduced=[f"{u.intent_id}:{u.kind}" for s in b.sections for u in s.uncertainty
                                    if f"{u.intent_id}:{u.kind}" not in _unc(a)],
            uncertainty_resolved=[x for x in _unc(a) if x not in _unc(b)])

    # ------------------------------------------------------------------ internals
    def _claim_sig(self, cid: str, v: AnswerVersion | None):
        """(status, text, evidence) of a claim as of version ``v`` (committed) or now (candidate). A new version of
        the need alone does not modify a claim: it is modified only if its validity or support changed."""
        c = self.graph.claims[cid]
        status = self.status_at.get(v.answer_id, {}).get(cid, c.status) if v is not None else c.status
        return (status, c.text, list(c.evidence_ids))

    def _sec_sig(self, s: AnswerSection):
        return (s.intent_version, tuple(s.claim_ids), tuple(s.constraints),
                tuple((u.kind, u.aspect) for u in s.uncertainty))

    def _sections(self, frame: TopicFrame, prev: AnswerVersion | None) -> list[AnswerSection]:
        out = []
        active = [i for i in frame.intent_ids if i in self.tracker.intents and self.tracker.intents[i].status == "ACTIVE"]
        for order, iid in enumerate(sorted(active, key=_id_num)):
            it = self.tracker.intents[iid]
            cids = [c for c in self.graph.selected.get(iid, []) if self.graph.claims[c].status in SHOWN]
            cids.sort(key=lambda c: (-self.graph.claims[c].confidence, _id_num(c)))
            ks = self.tracker.constraints_for(it)
            unc = []
            if not cids:
                unc.append(UncertaintyItem(intent_id=iid, kind="no_evidence", aspect=it.resolved_text))
            elif ks and all(self.graph.claims[c].status == "PARTIALLY_SUPPORTED" for c in cids):
                unc.append(UncertaintyItem(intent_id=iid, kind="constraint_not_covered",
                                           aspect="; ".join(k.text for k in ks)))
            if self.conflicts.get(iid):
                unc.append(UncertaintyItem(intent_id=iid, kind="conflict",
                                           aspect=",".join(sorted({f"{a}/{b}" for a, b, _ in self.conflicts[iid]}))))
            evid = list(dict.fromkeys(e for c in cids for e in self.graph.claims[c].evidence_ids))
            out.append(AnswerSection(section_id=f"S-{it.lineage_root or iid}", intent_id=iid, intent_version=it.version,
                                     title=it.resolved_text, order=order, claim_ids=cids, evidence_ids=evid,
                                     constraints=[k.text for k in ks], uncertainty=unc))
        ps = {s.section_id: s for s in (prev.sections if prev else [])}
        final = []
        for s in out:
            old = ps.get(s.section_id)
            if old is None:
                final.append(s.model_copy(update={"status": "new", "needs_regeneration": True}))
            elif self._sec_sig(old) != self._sec_sig(s) or self._status_changed(old, s, prev):
                final.append(s.model_copy(update={"status": "changed", "needs_regeneration": True}))
            else:
                final.append(s.model_copy(update={"status": "unchanged", "needs_regeneration": False}))
        return final

    def _status_changed(self, old: AnswerSection, new: AnswerSection, prev: AnswerVersion | None) -> bool:
        if prev is None:
            return False
        before = self.status_at.get(prev.answer_id, {})
        return any(before.get(c) not in (None, self.graph.claims[c].status) for c in new.claim_ids)

    def _version(self, frame: TopicFrame, sections: list[AnswerSection], prev: AnswerVersion | None,
                 utterance_id: str, session_version: int, now_ms: float) -> AnswerVersion:
        claim_ids = [c for s in sections for c in s.claim_ids]
        evidence = list(dict.fromkeys(e for s in sections for e in s.evidence_ids))
        cites = list(dict.fromkeys(self.store.records[e].citation for e in evidence))
        slots = {k.constraint_id: k.text for s in sections for k in self.tracker.constraints_for(
            self.tracker.intents[s.intent_id])}
        n = (prev.version + 1) if prev else 1
        return AnswerVersion(
            answer_id=f"A{len(self.versions) + 1}", version=n, parent_version=prev.version if prev else None,
            supersedes_answer_id=prev.answer_id if prev else None, topic_id=frame.frame_id, utterance_id=utterance_id,
            session_version=session_version, kind="initial" if prev is None else "refinement", created_at_ms=now_ms,
            sections=sections, claim_ids=claim_ids, evidence_ids=evidence, citations=cites,
            uncertainty=[u for s in sections for u in s.uncertainty], frame_slots=slots)


def _unc(v: AnswerVersion | None) -> list[str]:
    return [f"{u.intent_id}:{u.kind}" for s in (v.sections if v else []) for u in s.uncertainty]


def _diff_nonempty(d: AnswerDiff) -> bool:
    return bool(d.modified or d.added or d.retracted or d.sections_added or d.sections_removed or d.sections_changed
                or d.uncertainty_introduced or d.uncertainty_resolved)


def _summary(changes: list[ContextChange], d: AnswerDiff, sections: list[AnswerSection]) -> str:
    ch = "; ".join(f"{c.change_type}{'(' + ','.join(c.affected_intents + c.new_intents) + ')' if c.affected_intents or c.new_intents else ''}"
                   + (f" +{c.added_constraints}" if c.added_constraints else "")
                   + (f" -{c.removed_constraints}" if c.removed_constraints else "") for c in changes)
    regen = [s.section_id for s in sections if s.needs_regeneration]
    return (f"{ch or 'evidence update'} | claims: {len(d.kept)} kept, {len(d.modified)} modified, {len(d.added)} added,"
            f" {len(d.retracted)} retracted | sections to regenerate: {', '.join(regen) or 'none'}")
