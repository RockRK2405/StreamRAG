"""IntentTracker: stable intent ids, IntentSet versions and deltas across streaming updates (docs/multi_intent/08-09).

Each update re-decomposes the utterance's *current* transcript (decomposition is cheap and deterministic) and
reconciles the draft with the previous version:

  draft intent  ~ previous active intent   (term containment >= refine_containment or Jaccard >= match_jaccard)
      unchanged text/constraints/context  -> same intent, same version (no new retrieval)
      changed                             -> same intent_id, version + 1  (MODIFIED -> delta retrieval)
      superseded by a correction          -> SUPERSEDED (provenance kept)
  draft without a match                  -> new intent id (ADDED -> retrieval)
  previous intent without a match        -> REMOVED (status DROPPED, reason recorded)

A new ``IntentSet`` version is produced only when the delta is non-empty. Constraint ids are stable by normalized
text within the utterance.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from streamrag.config.settings import MultiIntentConfig
from streamrag.intents.decomposer import (
    Decomposition,
    DecompositionContext,
    DraftConstraint,
    DraftIntent,
    IntentDecomposer,
    PriorIntent,
    Segment,
)
from streamrag.models.intents import (
    Constraint,
    DropRecord,
    InheritedContext,
    Intent,
    IntentChange,
    IntentProvenance,
    IntentRelationship,
    IntentSet,
    IntentSetDelta,
    MergeRecord,
    QueryComponent,
    SourceSpan,
    Supersession,
)


@dataclass
class _UtteranceState:
    transcript: str = ""
    sets: list[IntentSet] = field(default_factory=list)
    key_to_id: dict[str, str] = field(default_factory=dict)
    constraint_ids: dict[str, str] = field(default_factory=dict)   # normalized text -> K id
    last: Decomposition | None = None


class IntentTracker:
    def __init__(self, session_id: str, decomposer: IntentDecomposer, cfg: MultiIntentConfig) -> None:
        self.session_id, self.dec, self.cfg = session_id, decomposer, cfg
        self.utterances: dict[str, _UtteranceState] = {}
        self.order: list[str] = []
        self.intents: dict[str, Intent] = {}            # every intent ever created (incl. superseded)
        self.terms: dict[str, list[str]] = {}
        self._n_intent = 0
        self._n_constraint = 0
        self.llm_check = None                           # LLMDecompositionCheck (optional, gated, final only)
        self.llm_reports: dict[str, object] = {}

    # ------------------------------------------------------------------ views
    def transcripts(self) -> dict[str, str]:
        return {u: s.transcript for u, s in self.utterances.items()}

    def current(self, utterance_id: str) -> IntentSet | None:
        st = self.utterances.get(utterance_id)
        return st.sets[-1] if st and st.sets else None

    def versions(self, utterance_id: str) -> list[IntentSet]:
        st = self.utterances.get(utterance_id)
        return list(st.sets) if st else []

    def active_intents(self, utterance_id: str) -> list[Intent]:
        cur = self.current(utterance_id)
        return [self.intents[i.intent_id] for i in cur.intents if self.intents[i.intent_id].status == "ACTIVE"] \
            if cur else []

    def context_for(self, utterance_id: str, prior_query_terms: list[list[str]] | None = None) -> DecompositionContext:
        """``recent`` = active intents of the most recent earlier utterance that has any (anaphora / ellipsis
        inherit only from there: carrying every earlier topic forward would blindly copy context). ``prior`` = all
        earlier active intents (a correction may name an older need explicitly)."""
        prior: list[PriorIntent] = []
        recent: list[PriorIntent] = []
        mine = {i for i, it in self.intents.items() if it.utterance_id == utterance_id}
        for u in self.order:
            if u == utterance_id:
                continue
            acts = self.active_intents(u)
            # intents this utterance's own correction superseded stay correction targets (prior only, never
            # anaphora antecedents): re-decomposing the growing utterance must find the same target again
            mine_sup = [it for it in self.intents.values() if it.utterance_id == u and it.status == "SUPERSEDED"
                        and it.superseded_by in mine]
            for it in mine_sup:
                prior.append(self._prior(it, u))
            if acts:
                recent = []
                for it in acts:
                    p = self._prior(it, u)
                    prior.append(p)
                    recent.append(p)
        return DecompositionContext(prior=prior, recent=recent, prior_query_terms=prior_query_terms or [])

    def _prior(self, it: Intent, u: str) -> PriorIntent:
        tr = self.utterances[u].transcript
        topic = None
        if it.topic is not None and it.topic_span is not None:
            sp = it.topic_span
            topic = (it.topic, sp.start, sp.end, sp.utterance_id)
        aspect_span = None
        if it.aspect:
            k = tr.lower().find(it.aspect.lower())
            aspect_span = (k, k + len(it.aspect)) if k >= 0 else None
        segs = [Segment(c.text, c.span.utterance_id, c.span.start, c.span.end,
                        "intent" if c.source == "intent" else "inherited", c.ref) for c in it.components]
        return PriorIntent(it.intent_id, u, it.text, self.terms.get(it.intent_id, []), topic, it.aspect, aspect_span,
                           segs)

    # ------------------------------------------------------------------ update
    def update(self, utterance_id: str, transcript: str, now_ms: float,
               prior_query_terms: list[list[str]] | None = None,
               final: bool = False) -> tuple[IntentSet, IntentSetDelta | None, Decomposition]:
        st = self.utterances.get(utterance_id)
        if st is None:
            st = self.utterances[utterance_id] = _UtteranceState()
            self.order.append(utterance_id)
        st.transcript = transcript
        d = self.dec.decompose(transcript, utterance_id, self.context_for(utterance_id, prior_query_terms))
        if final and self.llm_check is not None and utterance_id not in self.llm_reports:   # <= 1 call/utterance
            d, rep = self.llm_check.run(d)
            self.llm_reports[utterance_id] = rep
        st.last = d
        prev = st.sets[-1] if st.sets else None
        prev_active = [self.intents[i.intent_id] for i in prev.intents] if prev else []
        superseded_here = [it for it in self.intents.values()
                           if it.utterance_id == utterance_id and it.status == "SUPERSEDED"]
        mapping = self._match(d, prev_active, superseded_here)         # draft key -> existing intent id
        delta = IntentSetDelta(utterance_id=utterance_id, version=(prev.version + 1) if prev else 1,
                               previous_version=prev.version if prev else 0)
        # (an empty first decomposition - e.g. "actually I meant" before its content - creates no version)
        key_to_id: dict[str, str] = {}
        for di in d.intents:                                           # ids first (relations/constraints need them)
            if di.key in mapping:
                key_to_id[di.key] = mapping[di.key]
            elif di.status in ("ACTIVE", "SUPERSEDED"):
                self._n_intent += 1
                key_to_id[di.key] = f"I{self._n_intent}"
        constraints = self._constraints(d.constraints, st, key_to_id, utterance_id, transcript)
        old_k = {c.constraint_id for c in (prev.global_constraints + prev.local_constraints)} if prev else set()
        new_k = {c.constraint_id for c in constraints}
        delta.constraints_added = sorted(new_k - old_k, key=_id_num)
        delta.constraints_removed = sorted(old_k - new_k, key=_id_num)
        matched_prev = set(mapping.values())
        for di in d.intents:
            iid = key_to_id.get(di.key)
            if iid is None:
                continue
            existing = self.intents.get(iid)
            if di.status == "ACTIVE":
                new = self._to_intent(di, iid, utterance_id, transcript, key_to_id, constraints, existing, now_ms, d)
                if existing is None:
                    delta.added.append(iid)
                else:
                    changed = _changes(existing, new)
                    if changed:
                        new = new.model_copy(update={"version": existing.version + 1})
                        delta.modified.append(IntentChange(intent_id=iid, from_version=existing.version,
                                                           to_version=existing.version + 1, changed=changed))
                    else:
                        new = new.model_copy(update={"version": existing.version,
                                                     "created_at_ms": existing.created_at_ms,
                                                     "updated_at_ms": existing.updated_at_ms})
                    new = new.model_copy(update={"query_ids": existing.query_ids})
                self.intents[iid] = new
                self.terms[iid] = list(di.terms)
            elif existing is None and di.status == "SUPERSEDED":    # said and corrected within one update
                by = key_to_id.get(di.superseded_by) if di.superseded_by else None
                new = self._to_intent(di, iid, utterance_id, transcript, key_to_id, constraints, None, now_ms, d)
                self.intents[iid] = new.model_copy(update={"status": "SUPERSEDED", "superseded_by": by})
                self.terms[iid] = list(di.terms)
                delta.superseded.append(Supersession(old=iid, new=by or "", cue=_cue(d, di.superseded_by)))
            elif existing is not None and existing.status == "ACTIVE":
                status = "SUPERSEDED" if di.status == "SUPERSEDED" else "DROPPED"
                by = key_to_id.get(di.superseded_by) if di.superseded_by else None
                self.intents[iid] = existing.model_copy(update={"status": status, "superseded_by": by,
                                                                "updated_at_ms": now_ms})
                if status == "SUPERSEDED":
                    delta.superseded.append(Supersession(old=iid, new=by or "", cue=_cue(d, di.superseded_by)))
                else:
                    delta.removed.append(iid)
        for p in prev_active:                                          # vanished from the decomposition
            if p.intent_id not in matched_prev and self.intents[p.intent_id].status == "ACTIVE":
                self.intents[p.intent_id] = p.model_copy(update={"status": "DROPPED", "updated_at_ms": now_ms})
                delta.removed.append(p.intent_id)
        for old_id, new_key, cue in d.external_supersessions:          # correction of an earlier utterance's intent
            old = self.intents.get(old_id)
            new_id = key_to_id.get(new_key)
            if old is not None and old.status == "ACTIVE" and new_id:
                self.intents[old_id] = old.model_copy(update={"status": "SUPERSEDED", "superseded_by": new_id,
                                                              "updated_at_ms": now_ms})
                delta.superseded.append(Supersession(old=old_id, new=new_id, cue=cue))
        for di in d.intents:                                           # lineage of corrected intents
            iid = key_to_id.get(di.key)
            if iid and di.status == "ACTIVE" and di.supersedes:
                old_id = key_to_id.get(di.supersedes, di.supersedes)
                old = self.intents.get(old_id)
                root = old.lineage_root if old is not None else iid
                self.intents[iid] = self.intents[iid].model_copy(update={"supersedes": old_id, "lineage_root": root})
        delta.affected_intents = list(dict.fromkeys(delta.added + [m.intent_id for m in delta.modified]))
        active_ids = [key_to_id[di.key] for di in d.intents if di.status == "ACTIVE" and di.key in key_to_id]
        new_version = not delta.empty
        iset = IntentSet(
            session_id=self.session_id, utterance_id=utterance_id,
            version=delta.version if new_version else (prev.version if prev else 0), source=d.source,
            original_text=transcript, intents=[self.intents[i] for i in active_ids],
            global_constraints=[c for c in constraints if c.scope == "global"],
            local_constraints=[c for c in constraints if c.scope == "local"],
            relationships=self._relations(d, key_to_id), decomposition_confidence=d.confidence,
            superseded=sorted({s.old for s in delta.superseded} | set(prev.superseded if prev else []), key=_id_num),
            merged=[MergeRecord(text=t, into=key_to_id.get(k, k), reason=r) for t, k, r in d.merged],
            dropped=[DropRecord(text=t, intent_id=key_to_id.get(k) if k else None, reason=r) for t, k, r in d.dropped]
            + [DropRecord(text=self.intents[r].text, intent_id=r, reason="no_longer_in_decomposition")
               for r in delta.removed if r not in {key_to_id.get(k) for _, k, _ in d.dropped}],
            created_at_ms=now_ms if (new_version or prev is None) else prev.created_at_ms)
        st.key_to_id = key_to_id
        if new_version:
            st.sets.append(iset)
            return iset, delta, d
        if prev is None:
            return iset, None, d                                       # still nothing to track (version 0)
        st.sets[-1] = iset                                             # same version; refreshed object
        return iset, None, d

    def record_query(self, intent_id: str, query_id: str) -> None:
        it = self.intents[intent_id]
        self.intents[intent_id] = it.model_copy(update={"query_ids": it.query_ids + [query_id]})
        for st in self.utterances.values():
            if st.sets:
                cur = st.sets[-1]
                if any(i.intent_id == intent_id for i in cur.intents):
                    st.sets[-1] = cur.model_copy(update={"intents": [self.intents[i.intent_id]
                                                                     for i in cur.intents]})

    # ------------------------------------------------------------------ internals
    def _match(self, d: Decomposition, prev_active: list[Intent], superseded: list[Intent]) -> dict[str, str]:
        """Active drafts continue active intents; a superseded draft maps back to the intent it already is
        (active or already superseded), so re-decomposing a corrected utterance never mints a new id."""
        if not prev_active and not superseded:
            return {}
        pairs = []
        for di in d.intents:
            if not di.terms:
                continue
            td = set(di.terms)
            pool = prev_active + (superseded if di.status == "SUPERSEDED" else [])
            for p in pool:
                tp = set(self.terms.get(p.intent_id, []))
                if not tp:
                    continue
                jac = len(td & tp) / len(td | tp)
                cont = len(td & tp) / len(tp)
                if jac >= self.cfg.match_jaccard or cont >= self.cfg.refine_containment:
                    pairs.append((jac + cont, di.key, p.intent_id))
        out: dict[str, str] = {}
        used: set[str] = set()
        for _, k, iid in sorted(pairs, key=lambda x: (-x[0], x[1], _id_num(x[2]))):
            if k in out or iid in used:
                continue
            out[k] = iid
            used.add(iid)
        return out

    def _constraints(self, drafts: list[DraftConstraint], st: _UtteranceState, key_to_id: dict[str, str], uid: str,
                     transcript: str) -> list[Constraint]:
        out = []
        for k in drafts:
            norm = " ".join(k.text.lower().split())
            cid = st.constraint_ids.get(norm)
            if cid is None:
                self._n_constraint += 1
                cid = st.constraint_ids[norm] = f"K{self._n_constraint}"
            out.append(Constraint(
                constraint_id=cid, kind=k.kind, marker=k.marker, text=k.text, scope=k.scope,
                applies_to=[key_to_id[a] for a in k.applies_to if a in key_to_id], scope_reason=k.scope_reason,
                scope_confidence=k.scope_confidence,
                source_span=SourceSpan(utterance_id=uid, start=k.start, end=k.end, text=transcript[k.start:k.end])))
        return out

    def _relations(self, d: Decomposition, key_to_id: dict[str, str]) -> list[IntentRelationship]:
        kmap = {k.key: None for k in d.constraints}
        out = []
        for r in d.relations:
            if r.type == "CONSTRAINT_OF":
                continue                                               # constraints carry applies_to themselves
            src = key_to_id.get(r.source, r.source if r.source.startswith("I") else None)
            tgt = key_to_id.get(r.target, r.target if r.target.startswith("I") else None)
            if src and tgt and r.source not in kmap:
                out.append(IntentRelationship(type=r.type, source=src, target=tgt, cue=r.cue))
        return out

    def _to_intent(self, di: DraftIntent, iid: str, uid: str, transcript: str, key_to_id: dict[str, str],
                   constraints: list[Constraint], existing: Intent | None, now_ms: float, d: Decomposition) -> Intent:
        trs = self.transcripts()

        def span(sg: Segment) -> SourceSpan:
            tr = trs.get(sg.utterance_id, transcript if sg.utterance_id == uid else "")
            return SourceSpan(utterance_id=sg.utterance_id, start=sg.start, end=sg.end, text=tr[sg.start:sg.end])

        def ref(r: str | None) -> str | None:
            return key_to_id.get(r, r) if r else None
        components = [QueryComponent(text=sg.text, source="intent" if sg.source == "intent" else "inherited",
                                     ref=ref(sg.ref), span=span(sg)) for sg in di.segments]
        inherited = [InheritedContext(text=sg.text, reason=sg.reason, from_intent=ref(sg.ref), replaces=sg.replaces,
                                      source_span=span(sg))
                     for sg in di.segments + di.inherited if sg.source == "inherited" and sg.reason]
        kids = [c.constraint_id for c in constraints if iid in c.applies_to]
        topic = di.topic[0] if di.topic else None
        topic_span = None
        if di.topic:
            tu = di.topic[3]
            tr_t = trs.get(tu, transcript if tu == uid else "")
            topic_span = SourceSpan(utterance_id=tu, start=di.topic[1], end=di.topic[2],
                                    text=tr_t[di.topic[1]:di.topic[2]])
        return Intent(
            intent_id=iid, session_id=self.session_id, utterance_id=uid,
            version=existing.version if existing else 1, text=di.text, resolved_text=di.resolved_text,
            components=components, intent_type=di.intent_type, type_cues=di.type_cues, entities=di.entities,
            topic=topic, topic_span=topic_span, aspect=di.aspect, constraint_ids=kids, inherited_context=inherited,
            unresolved_references=di.unresolved, order=di.order, priority=di.priority,
            priority_score=di.priority_score, confidence=di.confidence, confidence_signals=di.signals,
            status="ACTIVE",
            provenance=IntentProvenance(source=di.origin if d.source != "reconciled" or di.origin == "llm"
                                        else "reconciled", split=di.split, clause_index=di.clause_index,
                                        span=SourceSpan(utterance_id=uid, start=di.start, end=di.end,
                                                        text=transcript[di.start:di.end])),
            supersedes=existing.supersedes if existing else None, lineage_root=existing.lineage_root if existing else iid,
            created_at_ms=existing.created_at_ms if existing else now_ms, updated_at_ms=now_ms,
            query_ids=existing.query_ids if existing else [])


def _cue(d: Decomposition, key: str | None) -> str:
    di = next((x for x in d.intents if x.key == key), None)
    return (di.correction_cue or "correction") if di else "correction"


def _changes(old: Intent, new: Intent) -> list[str]:
    changed = []
    if old.resolved_text != new.resolved_text or old.text != new.text:
        changed.append("text")
    if sorted(old.constraint_ids) != sorted(new.constraint_ids):
        changed.append("constraints")
    if [c.text for c in old.inherited_context] != [c.text for c in new.inherited_context]:
        changed.append("context")
    return changed


def _id_num(x: str) -> tuple[str, int]:
    head = x.rstrip("0123456789")
    tail = x[len(head):]
    return head, int(tail) if tail else 0
