"""Phase 5 multi-intent evaluation (docs/multi_intent/, PHASE_5 report §16-19).

Intent matching (spec §22.6, fixed before scoring): similarity between a predicted intent's meaning (resolved text +
inherited context) and a gold intent's description/paraphrases = 0.5 * cosine(separate embedder) + 0.5 * content-term
F1, maximised over description and paraphrases. The separate embedder is all-MiniLM-L6-v2 (the system retrieves with
bge-small), to avoid self-preference; if it is unavailable the lexical F1 alone is used and the report says so.
One-to-one Hungarian assignment; a pair counts as a match iff its similarity >= tau_eval (0.5, pre-registered).

All metrics are computed from system outputs against gold labels. Every output produced on fixture data carries
REPORTABLE=false and the DEV-SUITE banner.
"""

from __future__ import annotations

import json
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from streamrag.controller.query_builder import QueryBuilder
from streamrag.fusion import IntentEvidence
from streamrag.intents.query_builder import IntentQueryBuilder
from streamrag.intents.tracker import IntentTracker
from streamrag.models.benchmark import MultiIntentBenchmarkCase
from streamrag.models.events import SessionEnd, SessionStart
from streamrag.models.retrieval import RetrievalOptions, RetrievalRequest
from streamrag.multi_retrieval import MultiQueryRetriever
from streamrag.streaming import simulator as sim

BANNER = "DEV SUITE ON TEST FIXTURE DOMAIN - NOT AN OFFICIAL BENCHMARK RESULT - NOT HELD-OUT"
TAU_EVAL = 0.5


def load_mi_cases(path: Path) -> list[MultiIntentBenchmarkCase]:
    return [MultiIntentBenchmarkCase.model_validate_json(p.read_text()) for p in sorted(Path(path).glob("*.json"))]


def mi_inputs(case: MultiIntentBenchmarkCase, gap_ms: float = 1500.0) -> list:
    sid = f"mi-{case.case_id}"
    evs: list = [SessionStart(session_id=sid)]
    offset = 0.0
    for u in case.utterances:
        evs += sim._utterance_events(sid, u.utterance_id, [c.text for c in u.chunks], [c.timestamp_s for c in u.chunks],
                                     u.utterance_end_s, offset)
        offset += u.utterance_end_s + gap_ms / 1000.0
    evs.append(SessionEnd(session_id=sid))
    return evs


# ---------------------------------------------------------------------------------------------- matching
class IntentMatcher:
    def __init__(self, terms_fn, embedder=None, tau: float = TAU_EVAL) -> None:
        self.terms_fn, self.embedder, self.tau = terms_fn, embedder, tau
        self._cache: dict[str, np.ndarray] = {}

    @property
    def method(self) -> str:
        return "0.5*cos(all-minilm-l6-v2)+0.5*termF1" if self.embedder is not None else "termF1"

    def _vec(self, text: str) -> np.ndarray:
        if text not in self._cache:
            self._cache[text] = self.embedder.embed([text], "query")[0]
        return self._cache[text]

    def _f1(self, a: str, b: str) -> float:
        ta, tb = set(self.terms_fn(a)), set(self.terms_fn(b))
        if not ta or not tb:
            return 0.0
        p, r = len(ta & tb) / len(ta), len(ta & tb) / len(tb)
        return 0.0 if p + r == 0 else 2 * p * r / (p + r)

    def sim(self, pred: str, gold_texts: list[str]) -> float:
        best = 0.0
        for gt in gold_texts:
            f1 = self._f1(pred, gt)
            s = 0.5 * float(self._vec(pred) @ self._vec(gt)) + 0.5 * f1 if self.embedder is not None else f1
            best = max(best, s)
        return best

    def match(self, preds: list[str], golds: list[list[str]]) -> list[tuple[int, int, float]]:
        if not preds or not golds:
            return []
        m = np.array([[self.sim(p, g) for g in golds] for p in preds])
        rows, cols = linear_sum_assignment(-m)
        return [(int(r), int(c), round(float(m[r, c]), 4)) for r, c in zip(rows, cols) if m[r, c] >= self.tau]


def meaning(intent) -> str:
    return " ".join([intent.resolved_text] + [c.text for c in intent.inherited_context
                                              if c.reason not in ("anaphora", "correction_aspect")])


# ---------------------------------------------------------------------------------------------- offline evaluation
@dataclass
class Counts:
    tp: int = 0
    fp: int = 0
    fn: int = 0

    def add(self, other: "Counts") -> None:
        self.tp, self.fp, self.fn = self.tp + other.tp, self.fp + other.fp, self.fn + other.fn

    def prf(self) -> dict:
        p = self.tp / (self.tp + self.fp) if self.tp + self.fp else None
        r = self.tp / (self.tp + self.fn) if self.tp + self.fn else None
        f = (2 * p * r / (p + r)) if (p is not None and r is not None and p + r) else (0.0 if p is not None and r is not None else None)
        return {"precision": _r(p), "recall": _r(r), "f1": _r(f), "tp": self.tp, "fp": self.fp, "fn": self.fn}


def _r(x):
    return None if x is None else round(x, 4)


def _citations(es) -> list[str]:
    return [e.citation for e in es.items] if es is not None else []


@dataclass
class UtteranceEval:
    case_id: str
    category: str
    utterance_id: str
    n_gold: int
    n_pred: int
    intents: Counts
    matches: list
    constraint: Counts
    constraint_attach_ok: int = 0
    constraint_scope_ok: int = 0
    constraint_scope_n: int = 0
    relation: Counts = field(default_factory=Counts)
    context_required: int = 0
    context_retained: int = 0
    superseded_gold: int = 0
    superseded_ok: int = 0
    lenient_g3: bool | None = None
    strict_g3: bool | None = None
    queries: dict = field(default_factory=dict)        # gold id -> query text
    per_intent_recall: dict = field(default_factory=dict)
    timings: dict = field(default_factory=dict)
    retrieval: dict = field(default_factory=dict)


def evaluate_offline(stack, cases: list[MultiIntentBenchmarkCase], matcher: IntentMatcher, ks=(3, 5, 8),
                     dispatch: str = "parallel", rerankers: dict | None = None) -> list[UtteranceEval]:
    """Final-transcript evaluation: decomposition quality, per-intent retrieval and fusion/ablation arms."""
    cfg, b = stack.cfg, stack.bundle
    terms = stack.intent_stack.query_builder.terms_fn
    qb: IntentQueryBuilder = stack.intent_stack.query_builder
    fusion = stack.intent_stack.fusion
    single_builder = QueryBuilder(stack.intent_stack.decomposer.lx.base)
    mr = MultiQueryRetriever(stack.service, RetrievalOptions(top_k=cfg.multi_intent.max_candidates_per_intent),
                             max_concurrent=cfg.multi_intent.max_concurrent_retrievals)
    out: list[UtteranceEval] = []
    for case in cases:
        tracker = IntentTracker(f"mi-{case.case_id}", stack.intent_stack.decomposer, cfg.multi_intent)
        pred_to_gold: dict[str, str] = {}
        snaps = []
        t_ms = 0.0
        for u in case.utterances:                      # pass 1: the whole session (later utterances may correct)
            t0 = time.perf_counter()
            iset, _, _ = tracker.update(u.utterance_id, u.utterance_text, t_ms, final=True)
            snaps.append((u, iset, (time.perf_counter() - t0) * 1000.0))
            t_ms += 5000.0
        for u, iset, t_dec in snaps:                   # pass 2: score each utterance in the session's final state
            gold_active = [g for g in u.expected_intents if not g.superseded]
            preds = [tracker.intents[i.intent_id] for i in iset.intents
                     if tracker.intents[i.intent_id].status == "ACTIVE"]
            ms = matcher.match([meaning(p) for p in preds], [[g.description] + g.paraphrases for g in gold_active])
            for pi, gi, _ in ms:
                pred_to_gold[preds[pi].intent_id] = gold_active[gi].gold_intent_id
            ic = Counts(tp=len(ms), fp=len(preds) - len(ms), fn=len(gold_active) - len(ms))
            ev = UtteranceEval(case.case_id, case.category, u.utterance_id, len(gold_active), len(preds), ic,
                               [(preds[p].intent_id, gold_active[g].gold_intent_id, s) for p, g, s in ms], Counts())
            if len(gold_active) >= 2:
                ev.lenient_g3 = len(ms) >= 2
                ev.strict_g3 = len(ms) == len(gold_active) and ic.fp == 0
            # superseded needs: matched against the session's superseded predictions (final state)
            sup_gold = [g for g in u.expected_intents if g.superseded]
            sup_pred = [it for it in tracker.intents.values() if it.status == "SUPERSEDED"
                        and it.utterance_id == u.utterance_id]
            ev.superseded_gold = len(sup_gold)
            if sup_gold and sup_pred:
                ev.superseded_ok = len(matcher.match([meaning(p) for p in sup_pred],
                                                     [[g.description] + g.paraphrases for g in sup_gold]))
            # constraints
            ks_pred = iset.global_constraints + iset.local_constraints
            cm = _match_constraints(terms, ks_pred, u.expected_constraints)
            ev.constraint = Counts(tp=len(cm), fp=len(ks_pred) - len(cm), fn=len(u.expected_constraints) - len(cm))
            for pk, gk in cm:
                attached = sorted(pred_to_gold.get(a, "?") for a in pk.applies_to)
                ev.constraint_attach_ok += attached == sorted(gk.applies_to)
                if len(gold_active) >= 2:
                    ev.constraint_scope_n += 1
                    ev.constraint_scope_ok += pk.scope == gk.scope
            # relationships (typed, mapped to gold ids)
            pred_rel = {(r.type, pred_to_gold.get(r.source), pred_to_gold.get(r.target)) for r in iset.relationships}
            gold_rel = {(r.type, r.source, r.target) for r in u.expected_relationships}
            ev.relation = Counts(tp=len(pred_rel & gold_rel), fp=len(pred_rel - gold_rel), fn=len(gold_rel - pred_rel))
            # queries + context retention
            t0 = time.perf_counter()
            queries = {p.intent_id: qb.build(p, ks_pred) for p in preds}
            t_q = (time.perf_counter() - t0) * 1000.0
            gid_of = {pi: gid for pi, gid, _ in ev.matches}
            for g in gold_active:
                if not g.required_query_terms:
                    continue
                ev.context_required += 1
                pi = next((p for p, gg in gid_of.items() if gg == g.gold_intent_id), None)
                if pi is not None and set(terms(" ".join(g.required_query_terms))) <= set(queries[pi].terms):
                    ev.context_retained += 1
            ev.queries = {gid_of.get(pi, f"unmatched:{pi}"): q.text for pi, q in queries.items()}
            # retrieval: per-intent + fusion arms + single-query baseline
            t0 = time.perf_counter()
            res = mr.retrieve(list(queries.values()), dispatch) if queries else None
            t_r = (time.perf_counter() - t0) * 1000.0
            by = res.by_intent() if res else {}
            for g in gold_active:
                pi = next((p for p, gg in gid_of.items() if gg == g.gold_intent_id), None)
                cites = _citations(by[pi].evidence) if pi in by else []
                ev.per_intent_recall[g.gold_intent_id] = {kk: _recall(g.gold_evidence, cites[:kk]) for kk in ks}
            inputs = [IntentEvidence(p, queries[p.intent_id].text, None, by[p.intent_id].evidence if p.intent_id in by
                                     else None) for p in preds]
            gold_sets = [g.gold_evidence for g in gold_active if g.answerable]
            arms = {}
            sq = single_builder.build(u.utterance_text)
            for kk in ks:
                if not sq.empty:
                    t1 = time.perf_counter()
                    es = stack.service.retrieve(RetrievalRequest(query=sq.text, options=RetrievalOptions(top_k=kk)))
                    arms.setdefault("A_single_query", {})[kk] = _arm(_citations(es), gold_sets,
                                                                      (time.perf_counter() - t1) * 1000.0)
                for strat in ("concat", "global_score", "rrf", "intent_aware"):
                    if inputs:
                        t1 = time.perf_counter()
                        ues = fusion.fuse(case.case_id, u.utterance_id, iset.version, inputs, strategy=strat,
                                          rerank="none", top_k=kk)
                        arms.setdefault(f"fusion_{strat}", {})[kk] = _arm(
                            [i.citation for i in ues.items], gold_sets, (time.perf_counter() - t1) * 1000.0,
                            dup=ues.dedup["duplicate_items"], n_items=len(ues.items))
                for name, (mode, eng) in (rerankers or {}).items():
                    if inputs:
                        t1 = time.perf_counter()
                        ues = eng.fuse(case.case_id, u.utterance_id, iset.version, inputs, strategy="intent_aware",
                                       rerank=mode, top_k=kk)
                        arms.setdefault(name, {})[kk] = _arm([i.citation for i in ues.items], gold_sets,
                                                             (time.perf_counter() - t1) * 1000.0,
                                                             dup=ues.dedup["duplicate_items"], n_items=len(ues.items))
            ev.retrieval = {"arms": arms, "dispatch": dispatch, "total_ms": res.total_ms if res else 0.0,
                            "critical_path_ms": res.critical_path_ms if res else 0.0, "n_queries": len(queries)}
            ev.timings = {"decompose_ms": round(t_dec, 4), "query_ms": round(t_q, 4), "retrieve_ms": round(t_r, 4)}
            out.append(ev)
    mr.close()
    return out


def _recall(gold: list[str], cites: list[str]) -> float | None:
    if not gold:
        return None
    return round(len(set(gold) & set(cites)) / len(set(gold)), 4)


def _arm(cites: list[str], gold_sets: list[list[str]], ms: float, dup: int = 0, n_items: int | None = None) -> dict:
    covered = [bool(set(gs) & set(cites)) for gs in gold_sets]
    union = set(c for gs in gold_sets for c in gs)
    return {"coverage": round(sum(covered) / len(covered), 4) if covered else None,
            "overall_recall": round(len(union & set(cites)) / len(union), 4) if union else None,
            "duplicate_rate": round(dup / n_items, 4) if n_items else 0.0, "ms": round(ms, 4)}


def _match_constraints(terms, preds, golds) -> list:
    out, used = [], set()
    for gk in golds:
        tg = set(terms(gk.text))
        best, best_s = None, 0.0
        for n, pk in enumerate(preds):
            if n in used:
                continue
            tp = set(terms(pk.text))
            s = len(tg & tp) / len(tg | tp) if (tg | tp) else 0.0
            if s > best_s:
                best, best_s = n, s
        if best is not None and best_s >= 0.5:
            used.add(best)
            out.append((preds[best], gk))
    return out


# ---------------------------------------------------------------------------------------------- aggregation
def _mean(xs):
    xs = [x for x in xs if x is not None]
    return round(statistics.fmean(xs), 4) if xs else None


def pct(xs: list[float]) -> dict:
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return {"n": 0}
    return {"n": len(xs), "p50": round(float(np.percentile(xs, 50)), 4), "p95": round(float(np.percentile(xs, 95)), 4),
            "mean": round(statistics.fmean(xs), 4), "max": round(xs[-1], 4)}


def aggregate_offline(rows: list[UtteranceEval], ks=(3, 5, 8)) -> dict:
    tot, kc, rc = Counts(), Counts(), Counts()
    for r in rows:
        tot.add(r.intents)
        kc.add(r.constraint)
        rc.add(r.relation)
    by_cat: dict[str, Counts] = {}
    for r in rows:
        by_cat.setdefault(r.category, Counts()).add(r.intents)
    compound = [r for r in rows if r.lenient_g3 is not None]
    exact_count = [r.n_pred == r.n_gold for r in rows]
    recall = {kk: _mean([v[kk] for r in rows for v in r.per_intent_recall.values()]) for kk in ks}
    arms = sorted({a for r in rows for a in r.retrieval.get("arms", {})})
    arm_tab = {a: {kk: {m: _mean([r.retrieval["arms"][a][kk][m] for r in rows if a in r.retrieval.get("arms", {})
                                  and kk in r.retrieval["arms"][a]])
                        for m in ("coverage", "overall_recall", "duplicate_rate")} for kk in ks} for a in arms}
    arm_ms = {a: pct([r.retrieval["arms"][a][ks[0]]["ms"] for r in rows if a in r.retrieval.get("arms", {})])
              for a in arms}
    ctx_req = sum(r.context_required for r in rows)
    sup = sum(r.superseded_gold for r in rows)
    return {
        "utterances": len(rows), "cases": len({r.case_id for r in rows}),
        "intents": tot.prf(), "intents_by_category": {c: v.prf() for c, v in sorted(by_cat.items())},
        "intent_count_exact_rate": _mean([1.0 if x else 0.0 for x in exact_count]),
        "g3_lenient": _mean([1.0 if r.lenient_g3 else 0.0 for r in compound]),
        "g3_strict": _mean([1.0 if r.strict_g3 else 0.0 for r in compound]), "compound_utterances": len(compound),
        "constraints": kc.prf(),
        "constraint_attachment_accuracy": round(sum(r.constraint_attach_ok for r in rows) / kc.tp, 4) if kc.tp else None,
        "constraint_scope_accuracy_multi": round(sum(r.constraint_scope_ok for r in rows)
                                                 / max(1, sum(r.constraint_scope_n for r in rows)), 4)
        if sum(r.constraint_scope_n for r in rows) else None,
        "relationships": rc.prf(),
        "context_retention": round(sum(r.context_retained for r in rows) / ctx_req, 4) if ctx_req else None,
        "context_required": ctx_req,
        "supersession_recall": round(sum(r.superseded_ok for r in rows) / sup, 4) if sup else None,
        "superseded_gold": sup,
        "per_intent_recall_at_k": recall,
        "arms": arm_tab, "arm_latency_ms_at_k0": arm_ms,
        "latency_ms": {"decompose": pct([r.timings["decompose_ms"] for r in rows]),
                       "query_generation": pct([r.timings["query_ms"] for r in rows]),
                       "multi_retrieval": pct([r.timings["retrieve_ms"] for r in rows if r.retrieval.get("n_queries")])},
    }


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str))


# ---------------------------------------------------------------------------------------------- streaming evaluation
def evaluate_streaming(stack, cases: list[MultiIntentBenchmarkCase], matcher: IntentMatcher, mode: str = "virtual",
                       trace_dir: Path | None = None) -> list[dict]:
    """Cases streamed chunk by chunk through the StreamingSession (multi-intent enabled)."""
    from streamrag.streaming import run_realtime, run_virtual, utterance_stats
    runner = run_virtual if mode == "virtual" else run_realtime
    rows = []
    for case in cases:
        trace = (trace_dir / f"{case.case_id}.jsonl") if trace_dir else None
        run = runner(stack.cfg, stack.service, stack.policy, mi_inputs(case), trace, stack.index_hash,
                     stack.intent_stack)
        stats = utterance_stats(run.events)
        mi = run.session.mi
        for u in case.utterances:
            st = stats.get(u.utterance_id, {})
            turn = next((e.payload for e in run.events if e.type.value == "TURN_COMPLETED"
                         and e.utterance_id == u.utterance_id), {})
            gold_active = [g for g in u.expected_intents if not g.superseded]
            active = mi.tracker.active_intents(u.utterance_id)
            ms = matcher.match([meaning(p) for p in active], [[g.description] + g.paraphrases for g in gold_active])
            gens = [e for e in run.events if e.type.value == "QUERY_GENERATED" and e.utterance_id == u.utterance_id]
            seen, dup = set(), 0
            for e in gens:
                key = (e.intent_id, e.payload["query_text"])
                dup += key in seen
                seen.add(key)
            ues = turn.get("unified_evidence") or {}
            cites = [i["citation"] for i in ues.get("items", [])]
            gold_sets = [g.gold_evidence for g in gold_active if g.answerable]
            eligible = len(u.chunks) >= 2
            early, ready = [], []
            for pi, gi, _ in ms:
                ist = st.get("intents", {}).get(active[pi].intent_id, {})
                if eligible:
                    early.append(bool(ist.get("retrieved_early")))
                    ready.append(bool(ist.get("retrieved_early") or ist.get("ledger_hit")))
            ledger_hits = sum(1 for e in run.events if e.type.value == "RETRIEVAL_SKIPPED"
                              and e.utterance_id == u.utterance_id and e.payload.get("reason") == "ledger_hit")
            rows.append({
                "case_id": case.case_id, "category": case.category, "utterance_id": u.utterance_id,
                "n_gold": len(gold_active), "n_pred": len(active), "matched": len(ms),
                "queries_generated": len(gens), "expected_query_count": u.expected_query_count,
                "duplicate_queries_same_intent": dup, "ledger_hits": ledger_hits,
                "intent_set_versions": st.get("intent_set_versions", 0),
                "early_matched_intents": early, "early_or_reused": ready, "eligible": eligible,
                "lead_time_ms": st.get("lead_time_ms"), "post_final_ms": st.get("post_final_retrieval_latency_ms"),
                "unified_items": len(cites), "unified_coverage": _arm(cites, gold_sets, 0.0)["coverage"],
                "unified_overall_recall": _arm(cites, gold_sets, 0.0)["overall_recall"],
                "decompose_ms": list(mi.decompose_wall_ms), "fusion_ms": list(mi.fusion_wall_ms)})
    return rows


def aggregate_streaming(rows: list[dict]) -> dict:
    early = [x for r in rows for x in r["early_matched_intents"]]
    return {
        "utterances": len(rows),
        "matched_intent_early_rate": round(sum(early) / len(early), 4) if early else None,
        "matched_intent_early_or_reused_rate": round(sum(x for r in rows for x in r["early_or_reused"]) / len(early), 4)
        if early else None,
        "matched_intents_eligible": len(early),
        "queries_per_utterance": pct([r["queries_generated"] for r in rows]),
        "queries_minus_expected": pct([r["queries_generated"] - r["expected_query_count"] for r in rows]),
        "duplicate_queries_same_intent": sum(r["duplicate_queries_same_intent"] for r in rows),
        "ledger_hits": sum(r["ledger_hits"] for r in rows),
        "intent_set_versions": pct([r["intent_set_versions"] for r in rows]),
        "unified_coverage": _mean([r["unified_coverage"] for r in rows]),
        "unified_overall_recall": _mean([r["unified_overall_recall"] for r in rows]),
        "lead_time_ms": pct([r["lead_time_ms"] for r in rows if r["lead_time_ms"] is not None]),
        "post_final_ms": pct([r["post_final_ms"] for r in rows if r["post_final_ms"] is not None]),
    }
