"""Evidence store with an explicit lifecycle, and the rules that decide validity after a context change
(Phase 6; docs/session/05).

Evidence is never deleted. Applicability is tracked per (evidence, need) as an ``EvidenceAssignment`` whose status
moves through ACTIVE / RETAINED / REVALIDATION_REQUIRED / STALE / SUPERSEDED / INVALID, each transition with the
rule that caused it. A need's claims and answer section may only use ACTIVE or RETAINED evidence.

Validity rules (analyzed terms of the evidence text vs the change; documented in docs/session/05):

  CONSTRAINT_ADDITION  (constraint terms T, head term h = last term of T)
     evidence contains all of T                 -> ACTIVE                 constraint_covered
     evidence contains h but not all of T       -> REVALIDATION_REQUIRED  constraint_dimension_other_value
     evidence contains none of T                -> RETAINED               general_evidence_still_applicable
  CONSTRAINT_REMOVAL   (removed constraint terms T)
     evidence contains all of T                 -> REVALIDATION_REQUIRED  specific_to_removed_constraint
     otherwise                                  -> RETAINED               general_evidence_still_applicable
  REFINEMENT           (added terms A; topic terms N of the refined need)
     evidence contains some of A                -> ACTIVE                 covers_refinement
     A adds topic terms (the need now names what it is about, e.g. "the rules" -> "the rules for X")
       and the evidence contains none of N      -> REVALIDATION_REQUIRED  does_not_cover_refined_topic
     otherwise                                  -> RETAINED               general_evidence_still_applicable
  QUESTION_CHANGE                               -> REVALIDATION_REQUIRED  question_changed
  ENTITY_CHANGE        (new topic terms N)
     evidence contains some of N                -> REVALIDATION_REQUIRED  mentions_new_entity
     otherwise                                  -> SUPERSEDED             specific_to_replaced_entity
  CORRECTION           old need -> SUPERSEDED (intent_superseded); evidence that mentions the new need's topic is
                       carried to the new need as REVALIDATION_REQUIRED (carried_to_correction)
  INTENT_REMOVAL                                -> STALE                  intent_removed
  after the delta retrieval of a need: REVALIDATION_REQUIRED evidence retrieved again -> ACTIVE
                       (confirmed_by_delta_retrieval); not retrieved again -> STALE (not_confirmed_by_delta_retrieval)
  source no longer in the index                 -> INVALID                source_unavailable
"""

from __future__ import annotations

from collections.abc import Callable

from streamrag.context.models import ContextChange
from streamrag.delta.models import (
    USABLE_EVIDENCE,
    EvidenceAction,
    EvidenceAssignment,
    EvidenceRecord,
    EvidenceStatus,
    EvidenceTransition,
)
from streamrag.models.evidence import EvidenceSet

DECISION = {"ACTIVE": "RETAIN", "RETAINED": "RETAIN", "REVALIDATION_REQUIRED": "REVALIDATE", "SUPERSEDED": "SUPERSEDE",
            "STALE": "SUPERSEDE", "INVALID": "DISCARD"}


class EvidenceStore:
    def __init__(self) -> None:
        self.records: dict[str, EvidenceRecord] = {}
        self.assign: dict[tuple[str, str], EvidenceAssignment] = {}

    def add_results(self, intent_id: str, intent_version: int, query_id: str, es: EvidenceSet, now_ms: float,
                    rule: str = "retrieved") -> list[tuple[str, EvidenceTransition]]:
        """Assign retrieved evidence to the need; returns the re-activations (evidence that was STALE / SUPERSEDED /
        RETAINED for this need and was retrieved again)."""
        out = []
        for e in es.items:
            if e.evidence_id not in self.records:
                self.records[e.evidence_id] = EvidenceRecord(
                    evidence_id=e.evidence_id, document_id=e.document_id, section_id=e.section_id, citation=e.citation,
                    text=e.text, first_seen_query=query_id, first_seen_ms=now_ms)
            key = (e.evidence_id, intent_id)
            a = self.assign.get(key)
            if a is None:
                t = EvidenceTransition(from_status=None, to_status="ACTIVE", rule=rule, query_id=query_id, at_ms=now_ms)
                self.assign[key] = EvidenceAssignment(evidence_id=e.evidence_id, intent_id=intent_id, status="ACTIVE",
                                                      intent_version=intent_version, query_ids=[query_id],
                                                      best_rank=e.rank, history=[t])
            else:
                upd = {"query_ids": a.query_ids + ([query_id] if query_id not in a.query_ids else []),
                       "best_rank": min(a.best_rank, e.rank), "intent_version": intent_version}
                self.assign[key] = a.model_copy(update=upd)
                if a.status == "REVALIDATION_REQUIRED":
                    continue                     # decided by confirm_after_retrieval (confirmed_by_delta_retrieval)
                t = self.transition(e.evidence_id, intent_id, "ACTIVE", f"re_{rule}", None, now_ms, query_id)
                if t is not None:
                    out.append((e.evidence_id, t))
        return out

    def transition(self, evidence_id: str, intent_id: str, to: EvidenceStatus, rule: str, change_id: str | None,
                   now_ms: float, query_id: str | None = None) -> EvidenceTransition | None:
        a = self.assign[(evidence_id, intent_id)]
        if a.status == to:
            return None
        t = EvidenceTransition(from_status=a.status, to_status=to, rule=rule, change_id=change_id, query_id=query_id,
                               at_ms=now_ms)
        self.assign[(evidence_id, intent_id)] = a.model_copy(update={"status": to, "history": a.history + [t]})
        return t

    def for_intent(self, intent_id: str, statuses: tuple[str, ...] | None = None) -> list[EvidenceAssignment]:
        out = [a for (e, i), a in self.assign.items() if i == intent_id and (statuses is None or a.status in statuses)]
        return sorted(out, key=lambda a: (a.best_rank, a.evidence_id))

    def usable(self, intent_id: str) -> list[EvidenceAssignment]:
        return self.for_intent(intent_id, USABLE_EVIDENCE)

    def text(self, evidence_id: str) -> str:
        return self.records[evidence_id].text

    def counts(self) -> dict[str, int]:
        c: dict[str, int] = {}
        for a in self.assign.values():
            c[a.status] = c.get(a.status, 0) + 1
        return dict(sorted(c.items()))


class EvidenceValidityManager:
    def __init__(self, store: EvidenceStore, terms_fn: Callable[[str], list[str]]) -> None:
        self.store, self.terms_fn = store, terms_fn
        self._cache: dict[str, set[str]] = {}

    def _terms(self, evidence_id: str) -> set[str]:
        if evidence_id not in self._cache:
            self._cache[evidence_id] = set(self.terms_fn(self.store.text(evidence_id)))
        return self._cache[evidence_id]

    def _set(self, out: list[EvidenceAction], a: EvidenceAssignment, to: str, rule: str, change_id: str | None,
             now_ms: float) -> None:
        before = a.status
        t = self.store.transition(a.evidence_id, a.intent_id, to, rule, change_id, now_ms)
        out.append(EvidenceAction(evidence_id=a.evidence_id, intent_id=a.intent_id, decision=DECISION[to],
                                  from_status=before, to_status=to, rule=rule if t is not None else f"{rule} (unchanged)"))

    def apply(self, change: ContextChange, constraint_text: Callable[[str], str], topic_of: Callable[[str], str | None],
              now_ms: float) -> list[EvidenceAction]:
        out: list[EvidenceAction] = []
        ct = change.change_type
        cid = change.change_id
        for diff in change.diffs:
            iid = diff.intent_id
            usable = self.store.usable(iid)
            if ct == "CONSTRAINT_ADDITION":
                for k in diff.constraints_added:
                    tk = self.terms_fn(constraint_text(k))
                    if not tk:
                        continue
                    head = tk[-1]
                    for a in usable:
                        et = self._terms(a.evidence_id)
                        if set(tk) <= et:
                            self._set(out, a, "ACTIVE", "constraint_covered", cid, now_ms)
                        elif head in et:
                            self._set(out, a, "REVALIDATION_REQUIRED", "constraint_dimension_other_value", cid, now_ms)
                        else:
                            self._set(out, a, "RETAINED", "general_evidence_still_applicable", cid, now_ms)
                    usable = self.store.usable(iid)
            elif ct == "CONSTRAINT_REMOVAL":
                for k in diff.constraints_removed:
                    tk = set(self.terms_fn(constraint_text(k)))
                    for a in usable:
                        if tk and tk <= self._terms(a.evidence_id):
                            self._set(out, a, "REVALIDATION_REQUIRED", "specific_to_removed_constraint", cid, now_ms)
                        else:
                            self._set(out, a, "RETAINED", "general_evidence_still_applicable", cid, now_ms)
                    usable = self.store.usable(iid)
            elif ct == "REFINEMENT":
                added = set(diff.terms_added)
                topic = set(self.terms_fn(diff.topic_after or ""))
                for a in usable:
                    et = self._terms(a.evidence_id)
                    if added & et:
                        self._set(out, a, "ACTIVE", "covers_refinement", cid, now_ms)
                    elif added & topic and not topic & et:
                        self._set(out, a, "REVALIDATION_REQUIRED", "does_not_cover_refined_topic", cid, now_ms)
                    else:
                        self._set(out, a, "RETAINED", "general_evidence_still_applicable", cid, now_ms)
            elif ct == "QUESTION_CHANGE":
                for a in usable:
                    self._set(out, a, "REVALIDATION_REQUIRED", "question_changed", cid, now_ms)
            elif ct == "ENTITY_CHANGE":
                new_topic = set(self.terms_fn(diff.topic_after or ""))
                for a in usable:
                    if new_topic & self._terms(a.evidence_id):
                        self._set(out, a, "REVALIDATION_REQUIRED", "mentions_new_entity", cid, now_ms)
                    else:
                        self._set(out, a, "SUPERSEDED", "specific_to_replaced_entity", cid, now_ms)
            elif ct == "INTENT_REMOVAL":
                for a in self.store.for_intent(iid, ("ACTIVE", "RETAINED", "REVALIDATION_REQUIRED")):
                    self._set(out, a, "STALE", "intent_removed", cid, now_ms)
        if ct == "CORRECTION":
            new_iid = change.new_intents[0] if change.new_intents else None
            new_topic = set(self.terms_fn(topic_of(new_iid) or "")) if new_iid else set()
            for old in change.superseded_intents:
                for a in self.store.for_intent(old, ("ACTIVE", "RETAINED", "REVALIDATION_REQUIRED")):
                    carry = bool(new_topic & self._terms(a.evidence_id))
                    self._set(out, a, "SUPERSEDED", "intent_superseded", cid, now_ms)
                    if carry and new_iid and (a.evidence_id, new_iid) not in self.store.assign:
                        t = EvidenceTransition(from_status=None, to_status="REVALIDATION_REQUIRED",
                                               rule="carried_to_correction", change_id=cid, at_ms=now_ms)
                        self.store.assign[(a.evidence_id, new_iid)] = EvidenceAssignment(
                            evidence_id=a.evidence_id, intent_id=new_iid, status="REVALIDATION_REQUIRED",
                            intent_version=1, query_ids=list(a.query_ids), best_rank=a.best_rank, history=[t])
                        out.append(EvidenceAction(evidence_id=a.evidence_id, intent_id=new_iid, decision="REVALIDATE",
                                                  from_status=None, to_status="REVALIDATION_REQUIRED",
                                                  rule="carried_to_correction"))
        return out

    def confirm_after_retrieval(self, intent_id: str, retrieved: set[str], query_id: str, now_ms: float,
                                change_id: str | None = None) -> list[EvidenceAction]:
        out: list[EvidenceAction] = []
        for a in self.store.for_intent(intent_id, ("REVALIDATION_REQUIRED",)):
            if a.evidence_id in retrieved:
                self._set(out, a, "ACTIVE", "confirmed_by_delta_retrieval", change_id, now_ms)
            else:
                self._set(out, a, "STALE", "not_confirmed_by_delta_retrieval", change_id, now_ms)
        return out

    def check_sources(self, available: set[str], now_ms: float) -> list[EvidenceAction]:
        out: list[EvidenceAction] = []
        for a in list(self.store.assign.values()):
            if a.evidence_id not in available and a.status != "INVALID":
                self._set(out, a, "INVALID", "source_unavailable", None, now_ms)
        return out
