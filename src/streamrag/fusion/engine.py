"""EvidenceFusionEngine: per-intent EvidenceSets -> one UnifiedEvidenceSet (docs/multi_intent/06-07; ADR-009, ADR-015).

Pipeline:  provenance hits -> cross-intent dedup (same chunk / Phase 3 near-duplicate alternates) -> optional
cross-intent rerank -> selection strategy -> labels -> coverage, relations, conflict check.

Selection strategies (all compared in research/phase5; ``intent_aware`` is the default, ADR-015):
  concat        intent lists one after another, duplicates kept (naive baseline)
  global_score  union ranked by each chunk's best retrieval score (scores are not comparable across queries)
  rrf           union ranked by sum over intents of 1/(k + rank) (rank-based, but rewards chunks many intents share)
  intent_aware  coverage floor: ``min_per_intent`` round-robin picks per intent in priority order, then fill to
                ``top_k`` by (rank within intent, intent priority); section cap per intent; a chunk picked by
                several intents is one item serving all of them (ADR-009 quota round-robin + global budget)

Cross-intent rerank (pluggable):
  none                 retrieval order
  intent_ce            cross-encoder(intent query, chunk) on each intent's own candidates
  cross_intent_dense   cosine(intent query embedding, chunk embedding) for every candidate x every intent: a chunk
                       retrieved by I1 can be ranked into I2's list when it is relevant to I2's query
  cross_intent_ce      cross-encoder for every candidate x every intent (most expensive)

Conflicts: conservative typed-value heuristic only (same unit + shared content word, different numeric value,
different documents) -> ``EvidenceConflict(status="potential")``. It does not understand claims; non-numeric
contradictions are not detected (documented limitation).
"""

from __future__ import annotations

import hashlib
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from streamrag.config.settings import FusionConfig
from streamrag.fusion.models import (
    EvidenceConflict,
    EvidenceHit,
    EvidenceRelation,
    FusedEvidence,
    IntentCoverage,
    UnifiedEvidenceSet,
)
from streamrag.models.evidence import Evidence, EvidenceSet
from streamrag.models.intents import Intent


@dataclass
class IntentEvidence:
    """What fusion receives for one intent: its query's evidence (or why there is none)."""

    intent: Intent
    query_text: str
    query_id: str | None
    evidence: EvidenceSet | None
    status: str = "ok"                     # ok | retrieval_failed | pending
    stale: bool = False                    # evidence of a superseded version of this intent's query


@dataclass
class _Cand:
    ev: Evidence
    hits: list[EvidenceHit] = field(default_factory=list)
    alternates: set[str] = field(default_factory=set)
    relevance: dict[str, float] = field(default_factory=dict)
    fused: float | None = None


_UNIT = re.compile(r"(\d+(?:[.,]\d+)?)\s*(%|percent|millimet(?:re|er)s?|mm|centimet(?:re|er)s?|cm|met(?:re|er)s?|"
                   r"kilomet(?:re|er)s?|km|grams?|kilograms?|kg|litres?|liters?|ml|seconds?|minutes?|hours?|days?|"
                   r"weeks?|months?|years?|times?|degrees?|euros?|dollars?|rupees?|inr|usd|eur)\b", re.I)
_WORD = re.compile(r"[a-z]+")
_UNIT_WORD = re.compile(_UNIT.pattern.split(r"\s*", 1)[1].rstrip(r"\b"), re.I)   # the unit alternatives alone


class EvidenceFusionEngine:
    def __init__(self, cfg: FusionConfig, chunk_rows: dict[str, int] | None = None, vectors: np.ndarray | None = None,
                 embed_fn: Callable[[list[str]], np.ndarray] | None = None, reranker=None,
                 stopwords: frozenset[str] = frozenset()) -> None:
        self.cfg, self.rows, self.vectors = cfg, chunk_rows or {}, vectors
        self.embed_fn, self.reranker, self.stop = embed_fn, reranker, stopwords

    # ------------------------------------------------------------------ public
    def fuse(self, session_id: str, utterance_id: str, intent_set_version: int, inputs: list[IntentEvidence],
             strategy: str | None = None, rerank: str | None = None, top_k: int | None = None,
             min_per_intent: int | None = None) -> UnifiedEvidenceSet:
        strategy = strategy or self.cfg.strategy
        rerank = rerank or self.cfg.rerank
        top_k = top_k or self.cfg.top_k
        floor = self.cfg.min_per_intent if min_per_intent is None else min_per_intent
        timings: dict[str, float] = {}
        warnings: list[str] = []
        t_all = time.perf_counter()
        order = {x.intent.intent_id: x.intent.order for x in inputs}
        prio = {x.intent.intent_id: x.intent.priority for x in inputs}
        intents = sorted([x.intent.intent_id for x in inputs], key=lambda i: (prio[i], order[i]))

        t = time.perf_counter()
        cands, lists, stats = self._collect(inputs)
        timings["dedup"] = _ms(t)
        if rerank != "none" and cands:
            t = time.perf_counter()
            try:
                lists = self._rerank(rerank, inputs, cands, lists)
            except Exception as exc:   # noqa: BLE001 - fall back to retrieval order (spec §18.9 failure mode)
                warnings.append(f"rerank_failed_retrieval_order_kept: {exc.__class__.__name__}: {exc}")
                rerank = "none"
            timings["rerank"] = _ms(t)
        t = time.perf_counter()
        if strategy == "concat":
            picked = self._concat(lists, intents, order, top_k)
        elif strategy == "global_score":
            picked = self._global_score(cands, lists, top_k)
        elif strategy == "rrf":
            picked = self._rrf(cands, lists, top_k)
        else:
            picked = self._intent_aware(cands, lists, intents, top_k, floor)
        timings["select"] = _ms(t)
        items = self._items(picked, cands, lists, order, keep_duplicates=strategy == "concat")
        t = time.perf_counter()
        conflicts = self._conflicts(items) if self.cfg.conflict_check else []
        timings["conflicts"] = _ms(t)
        for c in conflicts:
            for it in items:
                if it.evidence_id in c.evidence_ids:
                    it.conflict_ids.append(c.conflict_id)
        coverage = self._coverage(inputs, items, lists)
        relations = self._relations(items, conflicts)
        timings["total"] = _ms(t_all)
        digest = hashlib.sha1("|".join([session_id, utterance_id, str(intent_set_version), strategy, rerank,
                                        str(top_k)] + [i.evidence_id for i in items]).encode()).hexdigest()[:12]
        return UnifiedEvidenceSet(
            unified_set_id=f"ues-{digest}", session_id=session_id, utterance_id=utterance_id,
            intent_set_version=intent_set_version, strategy=strategy, rerank=rerank, top_k=top_k, items=items,
            per_intent=coverage, relations=relations, conflicts=conflicts,
            dedup={**stats, "items": len(items), "duplicate_items": len(items) - len({i.evidence_id for i in items})},
            token_count=sum(i.token_count for i in {i.evidence_id: i for i in items}.values()),
            timings_ms={k: round(v, 4) for k, v in timings.items()}, warnings=warnings)

    # ------------------------------------------------------------------ collection + cross-intent dedup
    def _collect(self, inputs: list[IntentEvidence]):
        cands: dict[str, _Cand] = {}
        alias: dict[str, str] = {}                 # near-duplicate chunk -> kept chunk
        lists: dict[str, list[str]] = {}
        n_hits = 0
        near = 0
        for x in inputs:
            iid = x.intent.intent_id
            lists[iid] = []
            if x.evidence is None:
                continue
            for e in x.evidence.items:
                n_hits += 1
                key = alias.get(e.chunk_id, e.chunk_id)
                if key != e.chunk_id:
                    near += 1
                c = cands.get(key)
                if c is None:
                    c = cands[key] = _Cand(e)
                    for a in e.alternates:
                        alias.setdefault(a, key)
                    c.alternates.update(e.alternates)
                c.hits.append(EvidenceHit(intent_id=iid, query_id=x.query_id, retrieval_method=e.retrieval_method,
                                          rank=e.rank, score=e.score, bm25_rank=e.bm25_rank, dense_rank=e.dense_rank,
                                          rrf_score=e.rrf_score, rerank_score=e.rerank_score, stale=x.stale))
                if key not in lists[iid]:
                    lists[iid].append(key)
        multi = sum(1 for c in cands.values() if len({h.intent_id for h in c.hits}) > 1)
        return cands, lists, {"input_hits": n_hits, "unique_chunks": len(cands), "cross_intent_duplicates": multi,
                              "near_duplicates_merged": near}

    # ------------------------------------------------------------------ rerank
    def _rerank(self, mode: str, inputs: list[IntentEvidence], cands: dict[str, _Cand], lists: dict[str, list[str]]):
        qtext = {x.intent.intent_id: x.query_text for x in inputs}
        if mode == "intent_ce":
            if self.reranker is None:
                raise RuntimeError("intent_ce needs a cross-encoder (rerank model not loaded)")
            out = {}
            for iid, lst in lists.items():
                head = lst[: self.cfg.rerank_k]
                if head:
                    s = self.reranker.score(qtext[iid], [cands[k].ev.text for k in head])
                    for k, v in zip(head, s):
                        cands[k].relevance[iid] = float(v)
                out[iid] = sorted(head, key=lambda k: -cands[k].relevance[iid]) + lst[self.cfg.rerank_k:]
            return out
        union = list(cands)
        if mode == "cross_intent_dense":
            if self.embed_fn is None or self.vectors is None:
                raise RuntimeError("cross_intent_dense needs the dense index and query embedder")
            ids = list(qtext)
            q = self.embed_fn([qtext[i] for i in ids])
            rows = [self.rows[k] for k in union]
            sims = self.vectors[rows] @ q.T                      # (candidates, intents)
            for r, k in enumerate(union):
                for c, iid in enumerate(ids):
                    cands[k].relevance[iid] = float(sims[r, c])
        elif mode == "cross_intent_ce":
            if self.reranker is None:
                raise RuntimeError("cross_intent_ce needs a cross-encoder (rerank model not loaded)")
            for iid, text in qtext.items():
                s = self.reranker.score(text, [cands[k].ev.text for k in union])
                for k, v in zip(union, s):
                    cands[k].relevance[iid] = float(v)
        else:
            raise ValueError(f"unknown rerank mode {mode}")
        # each intent's list = every candidate ranked by its relevance to that intent's query
        return {iid: sorted(union, key=lambda k: (-cands[k].relevance[iid], k)) for iid in lists}

    # ------------------------------------------------------------------ strategies (return [(chunk, intent)])
    def _concat(self, lists, intents, order, top_k):
        out = []
        for iid in sorted(intents, key=lambda i: order[i]):
            out += [(k, iid) for k in lists[iid]]
        return out[:top_k]

    def _global_score(self, cands, lists, top_k):
        def best(k):
            return max(h.score for h in cands[k].hits)
        ranked = sorted(cands, key=lambda k: (-best(k), k))[:top_k]
        return [(k, self._owner(k, cands, lists)) for k in ranked]

    def _rrf(self, cands, lists, top_k):
        score = {k: sum(1.0 / (self.cfg.rrf_k + lst.index(k) + 1) for lst in lists.values() if k in lst) for k in cands}
        ranked = sorted(cands, key=lambda k: (-round(score[k], 12), k))[:top_k]
        for k in ranked:
            cands[k].fused = score[k]
        return [(k, self._owner(k, cands, lists)) for k in ranked]

    def _intent_aware(self, cands, lists, intents, top_k, floor):
        picked: list[tuple[str, str]] = []
        chosen: set[str] = set()
        per_section: dict[tuple[str, str, str], int] = {}
        pos = {i: 0 for i in intents}
        cap = self.cfg.section_cap_per_intent

        def next_for(iid):
            lst = lists.get(iid, [])
            while pos[iid] < len(lst):
                k = lst[pos[iid]]
                pos[iid] += 1
                ev = cands[k].ev
                sec = (iid, ev.document_id, ev.section_id)
                if per_section.get(sec, 0) >= cap:
                    continue
                return k, sec
            return None, None

        def take(iid):
            while True:
                k, sec = next_for(iid)
                if k is None:
                    return False
                per_section[sec] = per_section.get(sec, 0) + 1
                if k in chosen:                          # already selected for another intent: serves both
                    picked.append((k, iid))
                    return True
                if len(chosen) >= top_k:
                    return False
                chosen.add(k)
                picked.append((k, iid))
                return True

        for _ in range(floor):                           # coverage floor, round-robin in priority order
            for iid in intents:
                take(iid)
        # fill: remaining candidates by (rank within intent, intent priority)
        rest = sorted([(lists[i].index(k), n, k, i) for n, i in enumerate(intents) for k in lists.get(i, [])
                       if (k, i) not in set(picked)], key=lambda x: (x[0], x[1]))
        for _, _, k, iid in rest:
            if k in chosen or len(chosen) >= top_k:       # fill adds new items only (support is in hits)
                continue
            ev = cands[k].ev
            sec = (iid, ev.document_id, ev.section_id)
            if per_section.get(sec, 0) >= cap:
                continue
            per_section[sec] = per_section.get(sec, 0) + 1
            chosen.add(k)
            picked.append((k, iid))
        return picked

    @staticmethod
    def _owner(k, cands, lists):
        return min(((lists[i].index(k), i) for i in lists if k in lists[i]), default=(0, cands[k].hits[0].intent_id))[1]

    # ------------------------------------------------------------------ items, coverage, relations, conflicts
    def _items(self, picked, cands, lists, order, keep_duplicates: bool) -> list[FusedEvidence]:
        sel: dict[str, list[str]] = {}
        seq: list[str] = []
        for k, iid in picked:
            if k not in sel:
                sel[k] = []
                seq.append(k)
            if iid not in sel[k]:
                sel[k].append(iid)

        def sort_key(k):
            first = min((order.get(i, 99) for i in sel[k]), default=99)
            rank = min((lists[i].index(k) for i in sel[k] if k in lists.get(i, [])), default=99)
            return first, rank
        # concat baseline keeps one item per (chunk, intent) pick, in pick order: duplicates stay visible
        keys = [k for k, _ in picked] if keep_duplicates else sorted(seq, key=sort_key)
        out = []
        for n, k in enumerate(keys, start=1):
            c = cands[k]
            e = c.ev
            supp = list(dict.fromkeys(h.intent_id for h in c.hits))
            out.append(FusedEvidence(
                label=f"E{n}", evidence_id=e.evidence_id, chunk_id=e.chunk_id, document_id=e.document_id,
                section_id=e.section_id, citation=e.citation, section_title=e.section_title, text=e.text,
                source_path=e.source_path, char_start=e.char_start, char_end=e.char_end, supporting_intents=supp,
                supporting_queries=list(dict.fromkeys(h.query_id for h in c.hits if h.query_id)),
                selected_for=sel[k], hits=c.hits,
                best_rank_by_intent={i: min(h.rank for h in c.hits if h.intent_id == i) for i in supp},
                intent_relevance={i: round(v, 6) for i, v in c.relevance.items()},
                fused_score=round(float(c.fused if c.fused is not None else max(h.score for h in c.hits)), 6),
                alternates=sorted(c.alternates), token_count=int(e.metadata.get("token_count", 0) or 0)))
        return out

    def _coverage(self, inputs, items, lists) -> list[IntentCoverage]:
        out = []
        for x in inputs:
            iid = x.intent.intent_id
            mine = [it for it in items if iid in it.supporting_intents or iid in it.selected_for]
            status = "ok"
            if x.status in ("retrieval_failed", "pending"):
                status = x.status
            elif x.evidence is not None and not x.evidence.items:
                status = "no_candidates"
            elif x.evidence is None:
                status = "pending"
            out.append(IntentCoverage(
                intent_id=iid, query_id=x.query_id, status=status,
                candidates=len(x.evidence.items) if x.evidence else 0,
                evidence_ids=list(dict.fromkeys(it.evidence_id for it in mine)),
                labels=[it.label for it in mine], covered=bool(mine),
                best_rank=min((it.best_rank_by_intent[iid] for it in mine if iid in it.best_rank_by_intent),
                              default=None)))
        return out

    def _relations(self, items: list[FusedEvidence], conflicts: list[EvidenceConflict]) -> list[EvidenceRelation]:
        rel = []
        uniq = list({i.evidence_id: i for i in items}.values())
        for it in uniq:
            for a in it.alternates:
                rel.append(EvidenceRelation(source=a, target=it.evidence_id, type="DUPLICATES",
                                            basis="near_duplicate_collapsed (Phase 3 dedup)"))
        for n, a in enumerate(uniq):
            for b in uniq[n + 1:]:
                if (a.document_id, a.section_id) == (b.document_id, b.section_id):
                    rel.append(EvidenceRelation(source=a.evidence_id, target=b.evidence_id, type="RELATED",
                                                basis="same_section"))
        for c in conflicts:
            a, b = c.evidence_ids[:2]
            rel.append(EvidenceRelation(source=a, target=b, type="CONTRADICTS", basis=f"potential:{c.kind}"))
        return rel

    def _conflicts(self, items: list[FusedEvidence]) -> list[EvidenceConflict]:
        uniq = list({i.evidence_id: i for i in items}.values())
        facts = []
        for it in uniq:
            low = it.text.lower()
            for m in _UNIT.finditer(low):
                window = low[max(0, m.start() - 60): m.end() + 60]
                words = {w for w in _WORD.findall(window) if len(w) > 3 and w not in self.stop}
                facts.append((it, _unit(m.group(2)), m.group(1).replace(",", "."), words))
        out = []
        seen = set()
        for n, (a, ua, va, wa) in enumerate(facts):
            for b, ub, vb, wb in facts[n + 1:]:
                if a.document_id == b.document_id or ua != ub or va == vb:
                    continue
                shared_intents = set(a.supporting_intents) & set(b.supporting_intents)
                shared = sorted(w for w in (wa & wb) if not _UNIT_WORD.fullmatch(w))   # the unit is not context
                if not shared_intents or not shared:
                    continue
                key = tuple(sorted((a.evidence_id, b.evidence_id))) + (ua,)
                if key in seen:
                    continue
                seen.add(key)
                out.append(EvidenceConflict(
                    conflict_id=f"X{len(out) + 1}", kind="numeric_value", intent_ids=sorted(shared_intents),
                    evidence_ids=[a.evidence_id, b.evidence_id],
                    detail={"unit": ua, "values": [va, vb], "shared_words": shared[:5]}))
        return out


def _unit(u: str) -> str:
    u = u.lower()
    for full, short in (("millimet", "mm"), ("centimet", "cm"), ("kilomet", "km"), ("met", "m"), ("kilogram", "kg"),
                        ("gram", "g"), ("lit", "l"), ("second", "s"), ("minute", "min"), ("hour", "h"),
                        ("day", "day"), ("week", "week"), ("month", "month"), ("year", "year"), ("percent", "%")):
        if u.startswith(full):
            return short
    return u


def _ms(t: float) -> float:
    return (time.perf_counter() - t) * 1000.0
