"""Claims: extraction, the claim-evidence graph and targeted revalidation (Phase 6; docs/session/06-07).

Extraction (no LLM; Phase 7 owns generation): a claim is a *verbatim sentence* of a usable evidence item of a need,
chosen by IDF-weighted relevance = sum of corpus IDF of the query terms the sentence contains / sum of IDF of the
in-corpus query terms (need words + inherited context + constraints). Up to ``claims_per_intent`` sentences per need
version, each with relevance >= ``claim_min_relevance`` and either sharing >= min(2, |query terms|) terms with the
query or containing an anchor term: a term of the need's topic (what the question is about) when the need has one,
else the query's most corpus-specific terms (max IDF, ties included). One shared word that is neither (such as a
place name that occurs everywhere) is not enough to make a sentence a claim about the need.
Because the claim text is a substring of the evidence text (``ClaimSource`` span), SUPPORTS is justified by
construction - no claim is invented.

Claim ids are stable per (need lineage, evidence id, sentence span): a constraint change re-uses the same claims
and only re-evaluates them; a correction creates a new need lineage and therefore new claims.

Revalidation (only claims of affected needs, or claims linked to evidence whose status changed):

  need superseded (correction)                          -> SUPERSEDED    intent_superseded
  need removed / out of the current selection           -> STALE         intent_removed | not_selected_for_version
  no linked evidence usable (ACTIVE/RETAINED):
       some linked evidence REVALIDATION_REQUIRED       -> PENDING_VALIDATION  awaiting_evidence_revalidation
       otherwise                                         -> UNSUPPORTED   evidence_<status>   (link: UNSUPPORTED)
  evidence usable, but the claim does not contain the
       terms of an active constraint of the need        -> PARTIALLY_SUPPORTED does_not_address_constraint
                                                            (link: PARTIALLY_SUPPORTS)
  otherwise                                             -> SUPPORTED     verbatim_in_valid_evidence (link: SUPPORTS)

  Conflict: two selected claims of one need from different documents stating different numbers for the same unit
  around a shared content word -> CONTRADICTS links both ways (potential; the claims keep their status, the answer
  section records uncertainty kind=conflict). Non-numeric contradictions are not detected.

A claim is never left SUPPORTED when its evidence is no longer usable: every evidence transition triggers
revalidation of the claims linked to it (tested).
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass

from streamrag.delta.evidence import EvidenceStore
from streamrag.delta.models import USABLE_EVIDENCE
from streamrag.intents.tracker import IntentTracker
from streamrag.models.answers import Claim, ClaimEvidenceLink, ClaimSource, ClaimStatus, ClaimTransition
from streamrag.models.intents import Intent

_SENT = re.compile(r"[^.!?\n]+(?:[.!?]+|$)")
_LIST_PREFIX = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+")
_UNIT = re.compile(r"(\d+(?:[.,]\d+)?)\s*(%|percent|millimet(?:re|er)s?|mm|centimet(?:re|er)s?|cm|met(?:re|er)s?|"
                   r"kilomet(?:re|er)s?|km|grams?|kilograms?|kg|litres?|liters?|seconds?|minutes?|hours?|days?|"
                   r"weeks?|months?|years?|people|persons?|euros?|dollars?|rupees?)\b", re.I)


def sentences(text: str) -> list[tuple[int, int]]:
    """(start, end) spans of sentences / list items, trimmed of whitespace and list markers."""
    out = []
    for m in _SENT.finditer(text):
        s, e = m.start(), m.end()
        seg = text[s:e]
        lead = len(seg) - len(seg.lstrip())
        s += lead
        pm = _LIST_PREFIX.match(text[s:e])
        if pm:
            s += pm.end()
        e = s + len(text[s:e].rstrip())
        if e - s >= 3 and any(ch.isalpha() for ch in text[s:e]):
            out.append((s, e))
    return out


@dataclass
class ClaimCandidate:
    evidence_id: str
    start: int
    end: int
    text: str
    relevance: float
    rank: int


class ClaimExtractor:
    def __init__(self, terms_fn: Callable[[str], list[str]], per_intent: int = 4, min_relevance: float = 0.2,
                 idf_fn: Callable[[str], float | None] | None = None) -> None:
        self.terms_fn, self.k, self.min_rel = terms_fn, per_intent, min_relevance
        self.idf_fn = idf_fn or (lambda t: 1.0)

    def select(self, query_terms: list[str], store: EvidenceStore, intent_id: str,
               topic_terms: list[str] | None = None) -> list[ClaimCandidate]:
        weights = {t: w for t in dict.fromkeys(query_terms) if (w := self.idf_fn(t)) is not None and w > 0}
        if not weights:
            return []
        total = sum(weights.values())
        anchors = {t for t in (topic_terms or []) if t in weights}
        if not anchors:
            top = max(weights.values())
            anchors = {t for t, w in weights.items() if w == top}
        cands = []
        for a in store.usable(intent_id):
            text = store.text(a.evidence_id)
            for s, e in sentences(text):
                st = set(self.terms_fn(text[s:e]))
                shared = [t for t in weights if t in st]
                rel = sum(weights[t] for t in shared) / total
                if rel >= self.min_rel and (len(shared) >= min(2, len(weights)) or anchors & set(shared)):
                    cands.append(ClaimCandidate(a.evidence_id, s, e, text[s:e], round(rel, 4), a.best_rank))
        cands.sort(key=lambda c: (-c.relevance, c.rank, c.evidence_id, c.start))
        return cands[: self.k]


class ClaimGraph:
    def __init__(self) -> None:
        self.claims: dict[str, Claim] = {}
        self.links: dict[tuple[str, str], ClaimEvidenceLink] = {}
        self.transitions: list[ClaimTransition] = []
        self.selected: dict[str, list[str]] = {}          # intent id -> claim ids selected for its current version
        self._key: dict[tuple[str, str, int, int], str] = {}

    def register(self, intent: Intent, frame_id: str | None, cand: ClaimCandidate, now_ms: float,
                 introduced_by_constraint: str | None = None) -> tuple[Claim, bool]:
        key = (intent.lineage_root or intent.intent_id, cand.evidence_id, cand.start, cand.end)
        cid = self._key.get(key)
        if cid is not None:
            c = self.claims[cid].model_copy(update={"intent_version": intent.version, "confidence": cand.relevance,
                                                    "confidence_signals": {"relevance": cand.relevance},
                                                    "frame_id": frame_id, "updated_at_ms": now_ms})
            self.claims[cid] = c
            return c, False
        cid = f"C{len(self.claims) + 1}"
        self._key[key] = cid
        c = Claim(claim_id=cid, text=cand.text, intent_id=intent.intent_id, intent_version=intent.version,
                  frame_id=frame_id, evidence_ids=[cand.evidence_id],
                  source=ClaimSource(evidence_id=cand.evidence_id, char_start=cand.start, char_end=cand.end),
                  status="PENDING_VALIDATION", status_reason="registered", confidence=cand.relevance,
                  confidence_signals={"relevance": cand.relevance}, introduced_by_constraint=introduced_by_constraint,
                  created_at_ms=now_ms, updated_at_ms=now_ms)
        self.claims[cid] = c
        self.links[(cid, cand.evidence_id)] = ClaimEvidenceLink(claim_id=cid, evidence_id=cand.evidence_id,
                                                               relation="SUPPORTS", basis="verbatim_sentence")
        self.transitions.append(ClaimTransition(claim_id=cid, from_status=None, to_status="PENDING_VALIDATION",
                                                reason="registered", at_ms=now_ms))
        return c, True

    def set_status(self, cid: str, status: ClaimStatus, reason: str, change_id: str | None,
                   now_ms: float) -> ClaimTransition | None:
        c = self.claims[cid]
        if c.status == status and c.status_reason == reason:
            return None
        t = ClaimTransition(claim_id=cid, from_status=c.status, to_status=status, reason=reason, change_id=change_id,
                            at_ms=now_ms)
        self.claims[cid] = c.model_copy(update={"status": status, "status_reason": reason, "updated_at_ms": now_ms})
        self.transitions.append(t)
        return t

    def set_link(self, cid: str, eid: str, relation: str, basis: str) -> None:
        old = self.links.get((cid, eid))
        if old is None or old.relation != relation or old.basis != basis:
            self.links[(cid, eid)] = ClaimEvidenceLink(claim_id=cid, evidence_id=eid, relation=relation, basis=basis)

    def claims_of(self, intent_id: str) -> list[str]:
        return [c.claim_id for c in self.claims.values() if c.intent_id == intent_id]

    def linked_to(self, evidence_id: str) -> list[str]:
        return sorted({cid for (cid, eid) in self.links if eid == evidence_id})

    def status_counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for c in self.claims.values():
            out[c.status] = out.get(c.status, 0) + 1
        return dict(sorted(out.items()))


class ClaimRevalidator:
    def __init__(self, terms_fn: Callable[[str], list[str]]) -> None:
        self.terms_fn = terms_fn

    def validate(self, claim_ids: list[str], graph: ClaimGraph, store: EvidenceStore, tracker: IntentTracker,
                 now_ms: float, change_id: str | None = None) -> list[ClaimTransition]:
        out: list[ClaimTransition] = []
        for cid in claim_ids:
            c = graph.claims[cid]
            it = tracker.intents.get(c.intent_id or "")
            if it is None or it.status == "SUPERSEDED":
                st, why = "SUPERSEDED", "intent_superseded"
            elif it.status != "ACTIVE":
                st, why = "STALE", "intent_removed"
            elif cid not in graph.selected.get(it.intent_id, []):
                st, why = "STALE", "not_selected_for_version"
            else:
                st, why = self._evidence_verdict(c, it, graph, store, tracker)
            t = graph.set_status(cid, st, why, change_id, now_ms)
            if t is not None:
                out.append(t)
        return out

    def _evidence_verdict(self, c: Claim, it: Intent, graph: ClaimGraph, store: EvidenceStore,
                          tracker: IntentTracker) -> tuple[str, str]:
        statuses = {eid: store.assign[(eid, it.intent_id)].status for eid in c.evidence_ids
                    if (eid, it.intent_id) in store.assign}
        usable = [e for e, s in statuses.items() if s in USABLE_EVIDENCE]
        if not usable:
            for eid, s in statuses.items():
                graph.set_link(c.claim_id, eid, "UNSUPPORTED", f"evidence_{s.lower()}")
            if any(s == "REVALIDATION_REQUIRED" for s in statuses.values()):
                return "PENDING_VALIDATION", "awaiting_evidence_revalidation"
            worst = sorted(statuses.values())[0].lower() if statuses else "missing"
            return "UNSUPPORTED", f"evidence_{worst}"
        ct = set(self.terms_fn(c.text))
        uncovered = [k.constraint_id for k in tracker.constraints_for(it) if not set(self.terms_fn(k.text)) <= ct]
        for eid in usable:
            if uncovered:
                graph.set_link(c.claim_id, eid, "PARTIALLY_SUPPORTS", "does_not_address_constraint:" + ",".join(uncovered))
            else:
                graph.set_link(c.claim_id, eid, "SUPPORTS", "verbatim_sentence")
        if uncovered:
            return "PARTIALLY_SUPPORTED", "does_not_address_constraint:" + ",".join(uncovered)
        return "SUPPORTED", "verbatim_in_valid_evidence"

    def conflicts(self, claim_ids: list[str], graph: ClaimGraph, store: EvidenceStore) -> list[tuple[str, str, str]]:
        """Potential numeric conflicts among claims (different documents); adds CONTRADICTS links both ways."""
        facts = []
        for cid in claim_ids:
            c = graph.claims[cid]
            doc = store.records[c.evidence_ids[0]].document_id
            low = c.text.lower()
            for m in _UNIT.finditer(low):
                ctx = set(self.terms_fn(low)) - set(self.terms_fn(m.group(0)))
                facts.append((cid, doc, m.group(2).rstrip("s"), m.group(1), ctx))
        out = []
        for n, (a, da, ua, va, ca) in enumerate(facts):
            for b, db, ub, vb, cb in facts[n + 1:]:
                if a != b and da != db and ua == ub and va != vb and ca & cb:
                    graph.set_link(a, graph.claims[b].evidence_ids[0], "CONTRADICTS", f"potential_numeric_conflict:{ua}")
                    graph.set_link(b, graph.claims[a].evidence_ids[0], "CONTRADICTS", f"potential_numeric_conflict:{ua}")
                    out.append((a, b, ua))
        return out
