"""QueryLedger: the session's record of every retrieval query version (spec §13 ``ledger``).

* Each RETRIEVE decision creates a new version (Q1, Q2, ...). Phase 4 has a single *active* query per
  utterance: a new version supersedes the previous one (``relation`` = ``refines`` when the old terms are mostly
  contained in the new ones, else ``replaces``).
* Superseded queries are marked ``stale`` but are never deleted; if their retrieval completes later, the
  evidence is kept with stale lineage (``evidence_view`` returns it after the active query's evidence).
* Novelty checks use term-set Jaccard across the whole session (repeat questions reuse earlier evidence).
* Phase 5: with ``intent_id`` set, supersession is scoped to (utterance, intent): each intent has its own active
  query and lineage, so a new version of I2's query never makes I1's evidence stale. When an intent itself is
  superseded or removed, its queries are marked stale with ``stale_reason`` (evidence retained for provenance).
"""

from __future__ import annotations

from streamrag.ledger.models import QueryRecord


def jaccard(a: set[str], b: set[str]) -> float:
    return len(a & b) / len(a | b) if (a or b) else 1.0


def containment(old: set[str], new: set[str]) -> float:
    return len(old & new) / len(old) if old else 0.0


class QueryLedger:
    REFINES_CONTAINMENT = 0.6   # >= 60% of the old query's terms survive => the new version refines it

    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        self._records: dict[str, QueryRecord] = {}
        self._order: list[str] = []

    # ------------------------------------------------------------------ creation / lineage
    def create(self, utterance_id: str, query_text: str, transcript: str, spans: list[tuple[int, int]],
               terms: list[str], now_ms: float, trigger: str, trigger_chunk: int | None, tick: str,
               reason: str, intent_id: str | None = None, intent_version: int | None = None,
               batch_id: str | None = None, parent_query_id: str | None = None,
               derived_from_change_id: str | None = None, semantic_key: str | None = None,
               status: str = "pending", reused_from: str | None = None) -> tuple[QueryRecord, QueryRecord | None]:
        # Phase 6: an intent's lineage continues across utterances (a late detail in u2 refines I1 of u1)
        prev = self.active_for_intent(intent_id) if intent_id is not None else self.active(utterance_id)
        qid = f"Q{len(self._order) + 1}"
        version = (sum(1 for r in self._records.values() if r.intent_id == intent_id) if intent_id is not None
                   else sum(1 for r in self._records.values()
                            if r.utterance_id == utterance_id and r.intent_id is None)) + 1
        relation, supersedes, root = "initial", None, qid
        if prev is not None:
            relation = "refines" if containment(set(prev.terms), set(terms)) >= self.REFINES_CONTAINMENT else "replaces"
            supersedes, root = prev.query_id, prev.lineage_root
            self._records[prev.query_id] = prev.model_copy(update={"superseded_by": qid, "stale": True,
                                                                   "stale_reason": "superseded_query"})
        rec = QueryRecord(query_id=qid, session_id=self.session_id, utterance_id=utterance_id, version=version,
                          query_text=query_text, transcript_snapshot=transcript, source_spans=spans, terms=terms,
                          created_at_ms=now_ms, trigger=trigger, trigger_chunk=trigger_chunk, tick=tick,
                          decision_reason=reason, supersedes=supersedes, relation=relation, lineage_root=root,
                          intent_id=intent_id, intent_version=intent_version, batch_id=batch_id,
                          parent_query_id=parent_query_id if parent_query_id is not None else supersedes,
                          derived_from_change_id=derived_from_change_id, semantic_key=semantic_key, status=status,
                          reused_from=reused_from)
        self._records[qid] = rec
        self._order.append(qid)
        return rec, (self._records[prev.query_id] if prev is not None else None)

    def update(self, query_id: str, **fields) -> QueryRecord:
        rec = self._records[query_id].model_copy(update=fields)
        self._records[query_id] = rec
        return rec

    # ------------------------------------------------------------------ queries
    def get(self, query_id: str) -> QueryRecord:
        return self._records[query_id]

    def all(self) -> list[QueryRecord]:
        return [self._records[q] for q in self._order]

    def for_utterance(self, utterance_id: str) -> list[QueryRecord]:
        return [r for r in self.all() if r.utterance_id == utterance_id]

    def for_intent(self, intent_id: str) -> list[QueryRecord]:
        return [r for r in self.all() if r.intent_id == intent_id]

    def active(self, utterance_id: str, intent_id: str | None = None) -> QueryRecord | None:
        """Latest non-cancelled query of the utterance (Phase 4) or of one intent of it (Phase 5)."""
        for r in reversed(self.for_utterance(utterance_id)):
            if r.status != "cancelled" and r.intent_id == intent_id:
                return r
        return None

    def active_for_intent(self, intent_id: str) -> QueryRecord | None:
        """Latest non-cancelled query of an intent, in any utterance (Phase 6 cross-turn lineage)."""
        for r in reversed(self.for_intent(intent_id)):
            if r.status != "cancelled":
                return r
        return None

    def latest_completed(self, intent_id: str) -> QueryRecord | None:
        """Newest query of the intent whose retrieval completed with evidence (falls back across stale versions)."""
        for r in reversed(self.for_intent(intent_id)):
            if r.status in ("completed", "reused"):
                return r
        return None

    def mark_intent_stale(self, intent_id: str, reason: str) -> list[str]:
        """Intent superseded/removed: its queries stay (provenance) but are stale."""
        changed = []
        for r in self.for_intent(intent_id):
            if not r.stale or r.stale_reason != reason:
                self._records[r.query_id] = r.model_copy(update={"stale": True, "stale_reason": reason})
                changed.append(r.query_id)
        return changed

    def lineage_tree(self, utterance_id: str) -> list[dict]:
        """Intent -> query versions -> evidence ids (docs/multi_intent/05)."""
        tree: dict[str | None, dict] = {}
        for r in self.for_utterance(utterance_id):
            node = tree.setdefault(r.intent_id, {"intent_id": r.intent_id, "queries": []})
            node["queries"].append({"query_id": r.query_id, "version": r.version, "query_text": r.query_text,
                                    "status": r.status, "stale": r.stale, "stale_reason": r.stale_reason,
                                    "relation": r.relation, "supersedes": r.supersedes,
                                    "evidence_ids": list(r.evidence_ids)})
        return list(tree.values())

    def most_similar(self, terms: list[str]) -> tuple[QueryRecord | None, float]:
        best, best_sim = None, 0.0
        t = set(terms)
        for r in self.all():
            if r.status in ("cancelled", "failed"):
                continue
            sim = jaccard(t, set(r.terms))
            if sim > best_sim:
                best, best_sim = r, sim
        return best, best_sim

    def in_flight(self) -> list[QueryRecord]:
        return [r for r in self.all() if r.status in ("queued", "in_flight")]

    def retrieval_count(self, utterance_id: str) -> int:
        return sum(1 for r in self.for_utterance(utterance_id) if r.status != "cancelled")

    def last_issued_ms(self, utterance_id: str) -> float | None:
        """Stream time at which the utterance's latest non-cancelled query was issued (the RETRIEVE decision).
        The cooldown counts from here, not from the retrieval start, which includes execution/queue delay."""
        issued = [r.created_at_ms for r in self.for_utterance(utterance_id) if r.status != "cancelled"]
        return max(issued) if issued else None

    def evidence_view(self, utterance_id: str) -> list[dict]:
        """Evidence associated with the utterance: active query first, then stale lineage (newest first),
        deduplicated by evidence_id; each item lists every query that retrieved it."""
        view: dict[str, dict] = {}
        recs = list(reversed(self.for_utterance(utterance_id)))
        for r in recs:
            for eid in r.evidence_ids:
                item = view.setdefault(eid, {"evidence_id": eid, "query_ids": [], "stale": True})
                item["query_ids"].append(r.query_id)
                if not r.stale:
                    item["stale"] = False
        return sorted(view.values(), key=lambda x: (x["stale"], 0))   # active first; stable within groups

    def summary(self) -> list[dict]:
        return [r.model_dump(mode="json", include={"query_id", "utterance_id", "intent_id", "version", "query_text",
                                                   "status", "stale", "stale_reason", "supersedes", "superseded_by",
                                                   "relation", "retrieval_status", "evidence_ids"})
                for r in self.all()]
