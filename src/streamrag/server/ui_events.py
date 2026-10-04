"""Runtime events -> safe, user-facing UI events (brief: show progress, never internal reasoning).

The runtime's event log is rich (tasks, ledgers, claim lifecycles). The demo UI only receives:
transcript updates, short stage statuses ("Searching relevant sources..."), detected needs, the retrieval plan chosen
per need (strategy label), evidence citations, context changes (correction / refinement), cancelled / reused work,
answer drafts and validated answers (claims + citations), degraded-mode notices, safe error messages and per-turn
latency milestones (TTFE / TTFA / TTVA, measured from the first transcript chunk of the turn).
No prompt, model output beyond the verified answer, or document instruction is forwarded."""

from __future__ import annotations

STRATEGY_LABEL = {
    "LEXICAL": "Fast path: keyword lookup",
    "FAST_VECTOR": "Fast path: small vector search",
    "SEMANTIC": "Semantic search (meaning, not keywords)",
    "HYBRID": "Hybrid search (keywords + meaning)",
    "FILTERED": "Filtered search (applicant type / date / version)",
    "MULTI_HOP": "Multi-hop: follows a reference into a second document",
    "ITERATIVE": "Iterative: searches again for missing facts",
    "CACHE_REUSE": "Reused a validated cached result",
    "SESSION_REUSE": "Reused evidence from this conversation",
}
CHANGE_TEXT = {
    "CORRECTION": "Correction detected - updating the question",
    "ENTITY_CHANGE": "Different subject detected - updating the question",
    "REFINEMENT": "More detail added - refining the search",
    "CONSTRAINT_ADDITION": "New condition added - refining the search",
    "CONSTRAINT_REMOVAL": "Condition removed - widening the search",
    "QUESTION_CHANGE": "Question changed - updating the search",
    "INTENT_REMOVAL": "A request was withdrawn",
}
STAGES = ("query", "understanding", "retrieval", "evidence", "generating", "verifying", "final")


class UiMapper:
    """Stateful per session: dedupes statuses and tracks per-turn milestones."""

    def __init__(self, final_answer=None) -> None:
        self.final_answer = final_answer            # uid -> GroundedAnswer (validated finals) or None
        self.t0: dict[str, float] = {}
        self.marks: dict[str, dict[str, float]] = {}
        self.stage: dict[str, str] = {}

    def _stage(self, uid: str, stage: str, text: str) -> list[dict]:
        if self.stage.get(uid) == stage:
            return []
        self.stage[uid] = stage
        return [{"type": "status", "utterance": uid, "stage": stage, "text": text}]

    def _mark(self, uid: str, name: str, t: float) -> list[dict]:
        m = self.marks.setdefault(uid, {})
        if name in m or uid not in self.t0:
            return []
        m[name] = round(t - self.t0[uid], 1)
        return [{"type": "metrics", "utterance": uid, **{f"{k}_ms": v for k, v in m.items()}}]

    def map(self, ev) -> list[dict]:
        t, p, uid = ev.type.value, ev.payload, ev.utterance_id
        out: list[dict] = []
        if uid is None:
            if t == "DEGRADED_MODE_CHANGED":
                return [{"type": "notice", "level": "warning", "text": _degraded_text(p)}]
            return []
        base = {"utterance": uid, "trace": ev.correlation_id}
        if t == "CHUNK_RECEIVED":
            self.t0.setdefault(uid, ev.t_wall_ms)
            out += self._stage(uid, "query", "Listening...")
        elif t == "TRANSCRIPT_UPDATED":
            out.append({"type": "transcript", **base, "text": p.get("transcript", "")})
        elif t == "INTENTS_UPDATED":
            items = [{"id": i.get("intent_id"), "text": i.get("text"), "kind": i.get("type")}
                     for i in p.get("intents") or []]
            out += self._stage(uid, "understanding", "Understanding the question...")
            out.append({"type": "intents", **base, "items": items})
        elif t == "CONTEXT_CHANGE_DETECTED" and p.get("change_type") in CHANGE_TEXT:
            out.append({"type": "change", **base, "change_type": p["change_type"],
                        "text": CHANGE_TEXT[p["change_type"]]})
        elif t == "RETRIEVAL_POLICY_SELECTED":
            st = p.get("strategy")
            out.append({"type": "plan", **base, "intent": ev.intent_id, "strategy": st,
                        "label": STRATEGY_LABEL.get(st, str(st).title()), "complexity": p.get("complexity"),
                        "retrievers": p.get("retrievers") or [], "top_k": p.get("top_k"),
                        "filtered": bool(p.get("filters"))})
        elif t == "RETRIEVAL_STARTED":
            out += self._stage(uid, "retrieval", "Searching relevant sources...")
        elif t == "RETRIEVAL_COMPLETED" and (p.get("citations") or p.get("n_results")):
            out += self._mark(uid, "ttfe", ev.t_wall_ms)
            out += self._stage(uid, "evidence", "Checking evidence...")
        elif t == "EVIDENCE_FUSED":
            cites = list(dict.fromkeys(i.get("citation") for i in p.get("items") or [] if i.get("citation")))
            out.append({"type": "evidence", **base, "citations": cites})
        elif t in ("QUERY_SUPERSEDED",):
            out.append({"type": "cancelled", **base, "what": "search", "text": "Dropped an outdated search"})
        elif t == "TASK_CANCELLED" and p.get("task_type") in ("generation", "draft", "answer_extractive", "dense",
                                                              "lexical", "retrieval"):
            what = "answer" if p.get("task_type") in ("generation", "draft", "answer_extractive") else "search"
            out.append({"type": "cancelled", **base, "what": what,
                        "text": f"Cancelled outdated {'answer generation' if what == 'answer' else 'search work'}"})
        elif t in ("QUERY_REUSED", "CACHE_HIT"):
            out.append({"type": "reuse", **base, "text": "Reused evidence from earlier in the conversation"})
        elif t == "EVIDENCE_RETAINED":
            out.append({"type": "reuse", **base, "text": "Kept still-valid evidence"})
        elif t == "ANSWER_GENERATION_STARTED":
            out += self._stage(uid, "generating", "Writing the answer...")
        elif t in ("CLAIM_VERIFICATION_STARTED", "CLAIM_VERIFIED"):
            out += self._stage(uid, "verifying", "Checking every statement against the sources...")
        elif t == "CITATION_VALIDATED":
            out += self._stage(uid, "verifying", "Verifying citations...")
        elif t == "ANSWER_COMPLETED" and p.get("text"):
            out += self._mark(uid, "ttfa", ev.t_wall_ms)
            if p.get("status") == "DRAFT":
                out.append({"type": "answer", **base, "status": "DRAFT", "text": p["text"], "claims": [],
                            "version": p.get("version")})
        elif t == "ANSWER_COMMITTED":
            out += self._mark(uid, "ttfa", ev.t_wall_ms) + self._mark(uid, "ttva", ev.t_wall_ms)
            out += self._stage(uid, "final", "Verified answer")
            out.append({"type": "answer", **base, "status": p.get("status"), "text": p.get("text", ""),
                        "claims": self._claims(uid), "version": p.get("version"), "mode": p.get("mode"),
                        "backend": p.get("backend")})
        elif t == "DEGRADED_MODE_CHANGED":
            out.append({"type": "notice", **base, "level": "warning", "text": _degraded_text(p)})
        elif t == "ERROR":
            out.append({"type": "notice", **base, "level": "error",
                        "text": "A step failed; the system kept the last verified answer."
                        if p.get("recoverable", True) else "A step failed."})
        elif t == "TURN_COMPLETED":
            out.append({"type": "turn_done", **base})
        return out

    def _claims(self, uid: str) -> list[dict]:
        ga = self.final_answer(uid) if self.final_answer else None
        if ga is None:
            return []
        by_id = {c.citation_id: c for c in ga.citations.citations}
        out = []
        for sec in ga.sections:
            for cid in sec.claim_ids:
                c = ga.claim(cid)
                if c is None:
                    continue
                keys = [by_id[x].display_metadata.get("key", "") for x in c.citation_ids
                        if x in by_id and by_id[x].status == "valid"]
                out.append({"section": sec.title, "text": c.text, "kind": c.kind, "citations": list(dict.fromkeys(keys))})
        return out


def _degraded_text(p: dict) -> str:
    mode = p.get("mode") or ""
    if mode.startswith("GENERATION"):
        return "Language model unavailable or too slow - showing verified extractive answers"
    if mode.startswith("VALIDATION"):
        return "Entailment checker unavailable - using rule-based verification"
    if "RETRIEVAL" in mode or "LEXICAL" in mode or "DENSE" in mode:
        return "Part of the search is unavailable - continuing with the remaining index"
    return "Running in a degraded mode"
