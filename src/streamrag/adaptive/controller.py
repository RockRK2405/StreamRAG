"""AdaptiveRetrievalController (docs/architecture/13_adaptive_retrieval.md; docs/retrieval/03-11).

    Query + Intent + SessionState + AvailableEvidence + RuntimeBudget
      -> analysis (complexity)  -> rewrite  -> claim requirements  -> policy (strategy, k, filters, rerank, budget)
      -> [cache / session reuse]  -> bounded loop:
             search(es)  -> merge (dedup, RRF across searches)  -> assess sufficiency (requirements, conflicts)
             -> stop?  (sufficient | contradiction | budget | no expected gain | no results | error | cancelled)
             -> next action: contradiction search | hop | requirement query | relax filter | broaden | expand k
      -> final evidence (requirement support first, conflict losers excluded, source diversity cap, final_k)

The controller only *searches* through the Phase 3 ``RetrievalService`` stages; it never generates text, never calls
an LLM, and treats every retrieved chunk as data: corpus text can contribute words to a hop query (bounded,
sanitised) but never selects a strategy, a budget, a k, a filter field or a prompt.

Every step is recorded: ``RetrievalDecision`` (inputs, reason, expected / actual gain, latency), ``RetrievalHop``,
the search log, and buffered events (the caller emits them on its own thread: RETRIEVAL_POLICY_SELECTED,
QUERY_REWRITTEN, CACHE_HIT / CACHE_MISS, RETRIEVAL_INVALIDATED, RETRIEVAL_STARTED / RETRIEVAL_COMPLETED (per search),
EVIDENCE_ASSESSED, RETRIEVAL_EXPANDED, HOP_CREATED, RETRIEVAL_STOPPED).
"""

from __future__ import annotations

import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable

from streamrag.adaptive.analyzer import QueryComplexityAnalyzer, reference_date
from streamrag.adaptive.cache import AdaptiveQueryCache, CacheEntry, ValidatedClaimCache
from streamrag.adaptive.catalog import MetadataCatalog
from streamrag.adaptive.lexicon import RetrievalLexicon
from streamrag.adaptive.models import (EvidenceRequirement, OperationCounts, QueryAnalysis, RetrievalBudgetState,
                                       RetrievalDecision, RetrievalHop, RetrievalPlan, RetrievalState,
                                       RetrievalStrategy as S, RewrittenQuery, StopReason, SufficiencyAssessment)
from streamrag.adaptive.policy import RetrievalPolicySelector, next_k
from streamrag.adaptive.requirements import RequirementBuilder
from streamrag.adaptive.rewrite import QueryRewriteEngine
from streamrag.adaptive.stopping import RetrievalStoppingPolicy, actual_gain, expected_gain
from streamrag.adaptive.sufficiency import EvidenceSufficiencyEvaluator
from streamrag.adaptive.values import sentences
from streamrag.claims.textcheck import instruction_like
from streamrag.errors import ModelNotAvailableError, RetrieverTimeoutError
from streamrag.models.evidence import Evidence, EvidenceSet, RetrievalTrace
from streamrag.models.retrieval import RetrievalFilters, RetrievalOptions, RetrievalRequest

_WORD = re.compile(r"[A-Za-z0-9][\w'-]*")
MAX_QUERY_CHARS = 2000         # longer input is truncated before analysis (resource exhaustion)
MAX_CONSTRAINTS = 8
_MAIN = ("initial", "fallback_lexical", "broaden_retrievers", "relax_filter", "rerank", "expand_k", "entity")


@dataclass
class SessionView:
    """What the session already holds (read-only for the controller)."""

    pool: dict[str, Evidence] = field(default_factory=dict)        # usable evidence of the session's active needs
    usable: Callable[[str], bool] | None = None                    # is an evidence id still usable?
    claims: ValidatedClaimCache | None = None


@dataclass
class AdaptiveRequest:
    query_id: str
    text: str                                    # the need's query (Phase 5/6 contextual query, or the question)
    original_text: str | None = None             # what was said (cues: temporal, value kind)
    intent: Any = None
    constraints: list = field(default_factory=list)
    n_intents: int = 1
    dropped_entities: list[str] = field(default_factory=list)
    session: SessionView | None = None
    latency_budget_ms: float | None = None       # runtime deadline (Phase 8) - the smaller budget wins
    cancelled: Callable[[], bool] | None = None
    checkpoint: Callable[[], None] | None = None   # Phase 8 cooperative cancellation (raises)
    search_hook: Callable | None = None           # per-request hook before each search (Phase 8 fault injection)
    now_ms: float = 0.0


@dataclass
class SearchSpec:
    search_id: str
    action: str
    query_lex: str
    query_dense: str
    retrievers: list[str]
    k: int
    filters: RetrievalFilters | None
    rerank: bool
    hop_id: str | None = None
    purpose: str = ""
    hook: Callable | None = None

    def signature(self) -> tuple:
        return (self.query_lex, self.query_dense, tuple(self.retrievers), self.k,
                self.filters.canonical_json() if self.filters else None)


@dataclass
class AdaptiveResult:
    query_id: str
    analysis: QueryAnalysis
    rewritten: RewrittenQuery
    plan: RetrievalPlan
    requirements: list[EvidenceRequirement]
    assessment: SufficiencyAssessment
    state: RetrievalState
    evidence: EvidenceSet
    decisions: list[RetrievalDecision] = field(default_factory=list)
    hops: list[RetrievalHop] = field(default_factory=list)
    searches: list[dict] = field(default_factory=list)
    excluded: dict[str, str] = field(default_factory=dict)
    ops: OperationCounts = field(default_factory=OperationCounts)
    events: list[tuple[str, dict]] = field(default_factory=list)
    cache: str = "off"                           # off | miss | hit | invalidated
    wall_ms: float = 0.0
    n_specs: int = 0
    search_hook: Callable | None = None

    @property
    def strategy(self) -> S:
        return self.plan.strategy


@dataclass
class _Pooled:
    ev: Evidence
    score: float
    first_round: int
    hop_id: str | None
    session: bool = False


class AdaptiveRetrievalController:
    def __init__(self, service, cfg, claim_lexicon=None, *, catalog: MetadataCatalog | None = None,
                 lexicon: RetrievalLexicon | None = None, cache: AdaptiveQueryCache | None = None,
                 inline_dense: bool = False, parallel: bool = True, search_hook: Callable | None = None,
                 clock: Callable[[], float] = time.perf_counter) -> None:
        self.svc, self.cfg, self.ac = service, cfg, cfg.adaptive_retrieval
        an = service.bundle.analyzer
        self.terms_fn = lambda t: list(dict.fromkeys(an.tokens(t)))
        self.lexicon = lexicon or RetrievalLexicon.load(self.ac.lexicon)
        self.catalog = catalog or MetadataCatalog.build(service, cfg, self.terms_fn)
        self.analyzer = QueryComplexityAnalyzer(service, self.catalog, self.lexicon, claim_lexicon, cfg)
        self.rewriter = QueryRewriteEngine(self.catalog, self.lexicon, self.terms_fn, service.bundle.bm25.vocab, cfg)
        self.builder = RequirementBuilder(self.terms_fn, self.ac.claim_driven)
        self.ref_date = reference_date(self.ac.reference_date)
        self.evaluator = EvidenceSufficiencyEvaluator(self.terms_fn, self.catalog, self.ac.requirement_coverage,
                                                      self.ref_date, service.bundle.bm25.vocab,
                                                      service.bundle.bm25.term_idf)
        dense_ok = service.bundle.dense is not None and service.embedder is not None
        self.selector = RetrievalPolicySelector(cfg, dense_ok, service.reranker is not None)
        self.stopping = RetrievalStoppingPolicy(self.ac.min_gain)
        self.cache = cache if cache is not None else AdaptiveQueryCache(service.bundle.manifest.content_hash,
                                                                        self.ac.cache_max_entries, self.ac.cache)
        self.inline_dense, self.parallel, self.search_hook, self.clock = inline_dense, parallel, search_hook, clock
        self._pool = ThreadPoolExecutor(max_workers=self.ac.budget.max_parallel_tasks,
                                        thread_name_prefix="streamrag-adaptive") if parallel else None
        self._search_ms: list[float] = []            # measured search latencies (latency-aware stopping)
        self._rerank_ms_per_cand: float | None = None
        self._n_plan = 0

    def close(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=False, cancel_futures=True)

    # ------------------------------------------------------------------ search
    def _search(self, s: SearchSpec) -> tuple[EvidenceSet | None, str | None, float]:
        mode = "hybrid" if len(s.retrievers) == 2 else ("bm25" if s.retrievers == ["bm25"] else "dense")
        opts = RetrievalOptions(mode=mode, top_k=s.k, filters=s.filters, rerank=s.rerank,
                                rerank_k=self.ac.rerank_max_candidates if s.rerank else None)
        t0 = time.perf_counter()
        try:
            if self.search_hook is not None:
                self.search_hook(s)
            if s.hook is not None:
                s.hook(s)
            q_l, q_d = s.query_lex, s.query_dense
            plan_l = self.svc.plan(RetrievalRequest(query=q_l if mode != "dense" else q_d, options=opts))
            plan_d = plan_l if q_d == plan_l.request.query else self.svc.plan(
                RetrievalRequest(query=q_d, options=opts))
            lexical = self.svc.search_lexical(plan_l) if mode != "dense" else None
            dense, derr = None, None
            if mode != "bm25":
                try:
                    dense = self.svc.search_dense(plan_d, inline=self.inline_dense)
                except (ModelNotAvailableError, RetrieverTimeoutError) as exc:
                    derr = exc
            es = self.svc.assemble(plan_d if mode != "bm25" else plan_l, lexical, dense, derr,
                                   stage_ms=(time.perf_counter() - t0) * 1000.0)
            return es, None, (time.perf_counter() - t0) * 1000.0
        except Exception as exc:    # noqa: BLE001 - reported as a failed search; the loop decides what next
            if exc.__class__.__name__ in ("TaskCancelled", "CancelledError"):
                raise
            return None, f"{exc.__class__.__name__}: {exc}", (time.perf_counter() - t0) * 1000.0

    # ------------------------------------------------------------------ main entry
    def run(self, req: AdaptiveRequest) -> AdaptiveResult:
        # resource bounds (untrusted input): query length and number of constraints the analysis looks at
        if len(req.text) > MAX_QUERY_CHARS or len(req.constraints) > MAX_CONSTRAINTS:
            req.text, req.constraints = req.text[:MAX_QUERY_CHARS], list(req.constraints)[:MAX_CONSTRAINTS]
        if req.original_text is not None and len(req.original_text) > MAX_QUERY_CHARS:
            req.original_text = req.original_text[:MAX_QUERY_CHARS]
        t0 = self.clock()
        ms = lambda: round((self.clock() - t0) * 1000.0, 3)  # noqa: E731
        events: list[tuple[str, dict]] = []
        decisions: list[RetrievalDecision] = []
        qid = req.query_id

        def decide(strategy, action, reason, inputs=None, exp=None, act=None, cost=0.0, result=None):
            d = RetrievalDecision(decision_id=f"{qid}:D{len(decisions) + 1}", query_id=qid, strategy=strategy,
                                  action=action, reason=reason, inputs=inputs or {}, expected_gain=exp,
                                  actual_gain=act, latency_cost_ms=max(0.0, cost), result=result or {},
                                  timestamp_ms=ms())
            decisions.append(d)
            return d

        cue_text = " ".join(x for x in (req.original_text, getattr(req.intent, "text", None)) if x)
        a = self.analyzer.analyze(qid, req.text, req.intent, req.constraints, req.n_intents, cue_text)
        if cue_text:
            extra = self.analyzer.temporal(cue_text)
            if a.temporal.kind == "none" and extra.kind != "none":
                a.temporal = extra
            if a.value_kind is None and self.analyzer.clx is not None:
                a.value_kind = self.analyzer.clx.asks_value(cue_text)
        rq = self.rewriter.rewrite(req.original_text or req.text, getattr(req.intent, "intent_id", None),
                                   contextual=req.text, dropped_entities=req.dropped_entities)
        events.append(("QUERY_REWRITTEN", {"query_id": qid, "normalized": rq.normalized, "contextual": rq.contextual,
                                           "lexical_query": rq.lexical_query, "dense_query": rq.dense_query,
                                           "expansions": [e.model_dump() for e in rq.expansions],
                                           "intent_unchanged": self.rewriter.keeps_need(rq)}))
        valid_at = a.temporal.valid_at if a.temporal.kind in ("as_of", "current") and self.ac.temporal else None
        # requirement validity: the explicit date / period, else "now" (reference date); "past" without a date: none
        req_at = valid_at or (self.ref_date if a.temporal.kind == "none" and self.ac.temporal else None)
        alts: dict[str, list[str]] = {}
        for x in rq.expansions:
            for t in self.terms_fn(x.for_term):
                alts.setdefault(t, []).extend(self.terms_fn(x.term))
        reqs = self.builder.build(a, rq.contextual, req_at, self.analyzer.period_end(a.temporal), alts)
        budget_ms = min(self.ac.budget.max_latency_ms, req.latency_budget_ms or float("inf"))
        strategy, why = self.selector.base_strategy(a, req.latency_budget_ms)
        filters = self.selector.filters_for(a, self.analyzer.period_end(a.temporal))
        tight = req.latency_budget_ms is not None and req.latency_budget_ms < self.ac.tight_latency_ms
        rerank = (False, "latency budget too tight") if tight else \
            self.selector.rerank_for(a, strategy, budget_ms, self._rerank_ms_per_cand)
        b = self.ac.budget
        iters = 1 if (tight or not self.ac.iterative) else b.max_iterations
        budget = RetrievalBudgetState(max_queries=b.max_queries, max_results=b.max_results,
                                      max_iterations=iters, max_latency_ms=budget_ms,
                                      max_parallel_tasks=b.max_parallel_tasks, max_hops=b.max_hops)
        self._n_plan += 1
        plan = self.selector.plan(f"{qid}:P{self._n_plan}", a, getattr(req.intent, "intent_id", None), strategy, why,
                                  filters, rerank, [r.requirement_id for r in reqs], budget_ms, iters)
        ops = OperationCounts()
        res = AdaptiveResult(qid, a, rq, plan, reqs, SufficiencyAssessment(status="INSUFFICIENT", coverage=0.0,
                                                                          quality=0.0),
                             RetrievalState(current_strategy=strategy), _empty(self.svc, qid, rq.contextual),
                             decisions=decisions, ops=ops, events=events, search_hook=req.search_hook)
        decide(strategy, "select", why, {"complexity": a.complexity.value, "reasons": a.reasons,
                                         "signals": a.signals, "filters": a.metadata_filters,
                                         "temporal": a.temporal.model_dump(), "requirements": len(reqs)},
               result={"top_k": plan.top_k, "retrievers": plan.retrievers, "rerank": plan.reranking})
        events.append(("RETRIEVAL_POLICY_SELECTED", {
            "query_id": qid, "plan_id": plan.plan_id, "strategy": strategy.value, "strategy_reason": why,
            "complexity": a.complexity.value, "complexity_reasons": a.reasons, "top_k": plan.top_k,
            "retrievers": plan.retrievers, "filters": filters.model_dump(mode="json") if filters else None,
            "reranking": plan.reranking, "rerank_reason": plan.rerank_reason,
            "requirements": [r.claim_slot for r in reqs], "budget": budget.remaining()}))

        pool: dict[str, _Pooled] = {}

        # ---- 1. cache
        key = self.cache.key(self.terms_fn(rq.contextual), filters.model_dump(mode="json") if filters else None,
                             valid_at, self.ac.claim_driven)
        usable = req.session.usable if req.session is not None else None
        if self.cache.enabled:
            entry, inv = self.cache.get(key, self.catalog, usable, req.now_ms)
            if inv is not None:
                res.cache = "invalidated"
                events.append(("RETRIEVAL_INVALIDATED", {"query_id": qid, **inv}))
                decide(strategy, "cache_invalidated", inv["detail"] or inv["reason"])
            if entry is not None:
                ops.cache_hits += 1
                res.cache = "hit"
                for e in entry.evidence:
                    pool[e.evidence_id] = _Pooled(e, 1.0 / (60 + e.rank), 0, None)
                assess, exc = self.evaluator.assess(reqs, [p.ev for p in pool.values()], a.temporal.kind)
                events.append(("CACHE_HIT", {"query_id": qid, "key": key, "cached_query": entry.query,
                                             "evidence": len(entry.evidence), "status": assess.status,
                                             "hits": entry.hits}))
                plan.strategy = S.CACHE_REUSE
                plan.strategy_reason = f"valid cached result ({entry.status}) for the same terms / filters / date"
                decide(S.CACHE_REUSE, "cache_hit", plan.strategy_reason, {"key": key}, result={"status": assess.status})
                stop = StopReason.SUFFICIENT_EVIDENCE if assess.status == "SUFFICIENT" else (
                    StopReason.CONTRADICTION if assess.status == "CONTRADICTORY" else StopReason.NO_EXPECTED_GAIN)
                return self._finish(res, pool, reqs, assess, exc, budget, stop, S.CACHE_REUSE, ms, events)
            ops.cache_misses += 1
            if res.cache == "off":
                res.cache = "miss"
            events.append(("CACHE_MISS", {"query_id": qid, "key": key}))

        # ---- 2. session evidence (delta requirements)
        assess: SufficiencyAssessment | None = None
        if self.ac.session_reuse and req.session is not None and (req.session.pool or req.session.claims):
            sp = dict(req.session.pool)
            if req.session.claims is not None and usable is not None:
                for r in reqs:
                    for eid in req.session.claims.evidence_for(set(r.terms), self.ac.requirement_coverage, usable):
                        if eid in req.session.pool:
                            sp[eid] = req.session.pool[eid]
            if sp:
                a0, exc0 = self.evaluator.assess(reqs, list(sp.values()), a.temporal.kind)
                supporting = {i for r in reqs for i in r.evidence_ids}
                for eid in supporting:
                    e = sp[eid]
                    pool[eid] = _Pooled(e, 1.0 / (60 + e.rank), 0, None, session=True)
                events.append(("EVIDENCE_ASSESSED", {"query_id": qid, "round": 0, "source": "session",
                                                     **a0.model_dump()}))
                if a0.status == "SUFFICIENT":
                    ops.session_reuse += 1
                    plan.strategy = S.SESSION_REUSE
                    plan.strategy_reason = "session evidence satisfies every requirement (no search)"
                    decide(S.SESSION_REUSE, "session_reuse", plan.strategy_reason,
                           {"session_pool": len(sp)}, result={"supporting": sorted(supporting)})
                    return self._finish(res, pool, reqs, a0, exc0, budget, StopReason.SUFFICIENT_EVIDENCE,
                                        S.SESSION_REUSE, ms, events, cache_key=key)
                if supporting:
                    assess = a0
                    decide(strategy, "session_partial", f"session evidence meets {a0.met}; searching for {a0.unmet}",
                           {"session_pool": len(sp)})

        # ---- 3. bounded loop
        k = plan.top_k
        first = [r for r in reqs if assess is None or r.requirement_id in assess.unmet] or reqs
        if assess is not None and len(first) < len(reqs):         # delta: search only for what is still missing
            specs = [self._spec(res, f"requirement:{r.requirement_id}", r.query_text or rq.contextual,
                                r.query_text or rq.dense_query, plan.retrievers, k, filters, plan.reranking)
                     for r in first[: budget.max_parallel_tasks]]
        else:
            specs = [self._spec(res, "initial", rq.lexical_query, rq.dense_query, plan.retrievers, k, filters,
                                plan.reranking)]
        executed: set[tuple] = set()
        hops: list[RetrievalHop] = res.hops
        hopped: set[str] = set()
        contradiction_done = False
        relaxed = filters is None
        broadened = len(plan.retrievers) == 2
        stalled, last_fused, no_results = 0, 0, False
        stop: StopReason | None = None
        cur_filters = filters
        round_n = 0
        exc_ids: dict[str, str] = {}
        if strategy == S.MULTI_HOP:
            hops.append(RetrievalHop(hop_id=f"{qid}:H0", query=rq.contextual, purpose="the question as asked",
                                     status="PLANNED"))
            specs[0].hop_id = f"{qid}:H0"
            ent = " ".join(w for w in rq.contextual.split() if set(self.terms_fn(w)) & set(a.bridge_terms))
            if ent and budget.max_parallel_tasks > 1:   # hop 0b: what the corpus says about the entity itself
                hops.append(RetrievalHop(hop_id=f"{qid}:H0e", parent_hop_id=f"{qid}:H0", query=ent,
                                         purpose="statements about the entity (bridge candidates)", status="PLANNED"))
                specs.append(self._spec(res, "entity", ent, ent, plan.retrievers, plan.top_k, filters, plan.reranking,
                                        purpose="entity statements", hop_id=f"{qid}:H0e"))
        while True:
            if req.checkpoint is not None:
                req.checkpoint()
            if req.cancelled is not None and req.cancelled():
                stop = StopReason.USER_CANCELLED
                ops.cancelled += 1
                break
            mean = sum(self._search_ms[-20:]) / len(self._search_ms[-20:]) if self._search_ms else 0.0
            budget.latency_used_ms = ms()
            specs = [s for s in specs if s.signature() not in executed]
            room = budget.max_queries - budget.queries_used
            specs = specs[: max(0, min(room, budget.max_parallel_tasks))]
            if not specs:
                stop = StopReason.QUERY_LIMIT if room <= 0 else StopReason.NO_EXPECTED_GAIN
                break
            if budget.latency_used_ms + mean > budget.max_latency_ms and round_n > 0:
                stop = StopReason.LATENCY_LIMIT
                break
            round_n += 1
            budget.iterations_used = round_n
            before = assess
            outs = self._run_specs(specs, res, events, ms)
            any_ok, n_items, err = False, 0, None
            for s, (es, error, lat) in zip(specs, outs):
                executed.add(s.signature())
                budget.queries_used += 1
                budget.results_used += s.k
                if es is None:
                    err = error
                    continue
                any_ok = True
                n_items += len(es.items)
                last_fused = max(last_fused, es.trace.candidates_fused) if s.action == "initial" else last_fused
                if s.action == "initial" or s.action.startswith("expand_k"):
                    last_fused = es.trace.candidates_fused
                for e in es.items:
                    p = pool.get(e.evidence_id)
                    if p is None:
                        pool[e.evidence_id] = _Pooled(e, 1.0 / (60 + e.rank), round_n, s.hop_id)
                    else:
                        p.score += 1.0 / (60 + e.rank)
                if s.hop_id:
                    h = next((h for h in hops if h.hop_id == s.hop_id), None)
                    if h is not None:
                        h.evidence_ids = [e.evidence_id for e in es.items]
                        h.status = "COMPLETED"
            budget.latency_used_ms = ms()
            assess, exc_ids = self.evaluator.assess(reqs, [p.ev for p in pool.values()], a.temporal.kind)
            gain = actual_gain(before, assess)
            for d in decisions[-len(specs):]:
                if d.action not in ("select",):
                    d.actual_gain = gain
            hop_round = any(sp.action == "hop" for sp in specs)
            # a follow-up round that gained nothing (a hop round is not stalled: its value shows one round later)
            stalled = stalled + 1 if gain <= 0 and round_n > 1 and not hop_round else 0
            no_results = not pool and not n_items
            events.append(("EVIDENCE_ASSESSED", {"query_id": qid, "round": round_n, "gain": gain,
                                                 "pool": len(pool), **assess.model_dump()}))
            if not any_ok and err is not None and specs[0].retrievers != ["bm25"] and \
                    ("ModelNotAvailable" in err or "Timeout" in err) and budget.queries_used < budget.max_queries:
                specs = [self._spec(res, "fallback_lexical", rq.lexical_query, rq.dense_query, ["bm25"], k,
                                    cur_filters, False)]
                decide(strategy, "fallback_lexical", f"search failed ({err}); lexical-only retry")
                continue
            contradiction_exhausted = contradiction_done or not self.ac.contradiction_retrieval
            stop = self.stopping.check(assess, budget, error=not any_ok and err is not None and not pool,
                                       no_results=no_results and relaxed and broadened,
                                       contradiction_exhausted=contradiction_exhausted,
                                       mean_search_ms=sum(self._search_ms[-20:]) / max(1, len(self._search_ms[-20:])),
                                       stalled_rounds=stalled)
            if stop is not None:
                break
            if assess.unmet and set(assess.unmet) <= set(assess.unattainable) and not assess.conflicts \
                    and strategy not in (S.MULTI_HOP,):
                stop = StopReason.NO_EXPECTED_GAIN
                decide(strategy, "stop", f"unmet requirements {assess.unattainable} are unattainable: their terms "
                       "are unknown to the index (no search can meet them)")
                break
            # ---- next actions (expected gain)
            cands: list[tuple[float, SearchSpec, str]] = []
            n_r = len(reqs)
            if assess.status == "CONTRADICTORY" and not contradiction_exhausted:
                contradiction_done = True
                r = next(r for r in reqs if r.status == "CONFLICT")
                docs = {self._doc_of(pool, i) for i in r.evidence_ids}
                others = sorted(set(self.catalog.docs) - docs)
                if others:
                    f = RetrievalFilters(document_ids=others, metadata=cur_filters.metadata if cur_filters else None,
                                         valid_at=cur_filters.valid_at if cur_filters else None,
                                         valid_to=cur_filters.valid_to if cur_filters else None)
                    cq = f"{r.query_text or rq.contextual} current version effective"
                    sp = self._spec(res, "contradiction_search", cq, r.query_text or rq.dense_query,
                                    ["bm25", "dense"] if broadened else plan.retrievers,
                                    next_k(k, self.ac.k_schedule) or k, f, plan.reranking,
                                    purpose=f"third source for conflicting values of {r.requirement_id}")
                    cands.append((expected_gain("contradiction_search", assess, n_r), sp,
                                  f"values conflict for {r.requirement_id} ({r.values}); search other sources"))
            if self.ac.multi_hop and strategy in (S.MULTI_HOP, S.ITERATIVE) and (assess.unmet or assess.conflicts):
                for spec, why2 in self._hops(res, a, rq, reqs, pool, hopped, hops, cur_filters, plan, valid_at,
                                             budget.max_hops):
                    cands.append((expected_gain("hop", assess, n_r), spec, why2))
            if not relaxed and assess.unmet and cur_filters is not None and cur_filters.metadata:
                relaxed = True
                cur_filters = RetrievalFilters(valid_at=cur_filters.valid_at, valid_to=cur_filters.valid_to) \
                    if cur_filters.valid_at else None
                sp = self._spec(res, "relax_filter", rq.lexical_query, rq.dense_query, plan.retrievers,
                                next_k(k, self.ac.k_schedule) or k, cur_filters, plan.reranking)
                cands.append((expected_gain("relax_filter", assess, n_r), sp,
                              "metadata filter left requirements unmet: search without it (validity kept)"))
            if not broadened and assess.unmet and self.selector.dense:
                broadened = True
                sp = self._spec(res, "broaden_retrievers", rq.lexical_query, rq.dense_query, ["bm25", "dense"],
                                next_k(k, self.ac.k_schedule) or k, cur_filters, plan.reranking)
                cands.append((expected_gain("requirement_query", assess, n_r), sp,
                              f"{plan.retrievers} left requirements unmet: hybrid"))
            rtv = ["bm25", "dense"] if broadened and self.selector.dense else plan.retrievers
            for r in reqs:
                if r.status != "UNMET" or not r.query_text or r.kind == "bridge":
                    continue                                # bridge / link slots are served by their hop
                nk = next_k(k, self.ac.k_schedule) or k
                sp = self._spec(res, f"requirement:{r.requirement_id}", r.query_text, r.query_text, rtv, nk,
                                cur_filters, plan.reranking, purpose=r.claim_slot)
                if sp.signature() in executed:
                    continue
                cands.append((expected_gain("requirement_query", assess, n_r), sp, f"unmet {r.claim_slot}"))
            if (plan.reranking is False and self.ac.rerank == "policy" and self.selector.rerank_ok and assess.unmet
                    and last_fused > k and not tight):
                sp = self._spec(res, "rerank", rq.lexical_query, rq.dense_query, rtv, k, cur_filters, True)
                cands.append((expected_gain("rerank", assess, n_r), sp,
                              f"requirements unmet with {last_fused} candidates for k={k}: cross-encoder rerank"))
            nk = next_k(k, self.ac.k_schedule) if self.ac.adaptive_k else None
            if nk is not None and assess.status != "CONTRADICTORY":
                sp = self._spec(res, f"expand_k:{nk}", rq.lexical_query, rq.dense_query, rtv, nk, cur_filters,
                                plan.reranking)
                cands.append((expected_gain("expand_k", assess, n_r, more_candidates=last_fused > k), sp,
                              f"adaptive top-k {k} -> {nk} (candidates available: {last_fused})"))
            cands = [c for c in cands if c[1].signature() not in executed]
            cands.sort(key=lambda c: -c[0])
            worth = [c for c in cands if self.stopping.worth(c[0])]
            if not worth:
                stop = StopReason.NO_EXPECTED_GAIN if assess.status != "CONTRADICTORY" else StopReason.CONTRADICTION
                decide(strategy, "stop", f"best expected gain {cands[0][0] if cands else 0.0} < min_gain "
                       f"{self.ac.min_gain}" if assess.status != "CONTRADICTORY" else "unresolved value conflict",
                       {"candidates": [(c[1].action, c[0]) for c in cands[:5]]})
                break
            specs = []
            # targeted actions (hop / contradiction / requirement query) run with their own kind only: a speculative
            # k expansion next to a hop would spend a search before the hop's value is known
            fam = worth[0][1].action.split(":")[0]
            same = [c for c in worth if c[1].action.split(":")[0] == fam]
            for g, sp, why2 in same[: budget.max_parallel_tasks]:
                specs.append(sp)
                decide(strategy, sp.action, why2, {"k": sp.k, "query": sp.query_lex,
                                                    "filters": sp.filters.model_dump(mode="json") if sp.filters
                                                    else None}, exp=g)
                events.append(("RETRIEVAL_EXPANDED", {"query_id": qid, "action": sp.action, "reason": why2,
                                                      "k": sp.k, "expected_gain": g, "search_id": sp.search_id,
                                                      "hop_id": sp.hop_id}))
                if sp.action.startswith("expand_k"):
                    k = sp.k
            if any(sp.action.startswith(("requirement", "relax", "broaden", "contradiction")) for sp in specs):
                k = max(k, max(sp.k for sp in specs))
        return self._finish(res, pool, reqs, assess or res.assessment, exc_ids, budget, stop or StopReason.ERROR,
                            strategy, ms, events, cache_key=key)

    # ------------------------------------------------------------------ helpers
    def _spec(self, res: AdaptiveResult, action: str, ql: str, qd: str, retrievers: list[str], k: int,
              filters, rerank: bool, purpose: str = "", hop_id: str | None = None) -> SearchSpec:
        res.n_specs += 1
        return SearchSpec(f"{res.query_id}:S{res.n_specs}", action, ql, qd, list(retrievers), k, filters, rerank,
                          hop_id, purpose, res.search_hook)

    def _run_specs(self, specs: list[SearchSpec], res: AdaptiveResult, events, ms) -> list:
        for s in specs:
            events.append(("RETRIEVAL_STARTED", {"query_id": res.query_id, "search_id": s.search_id,
                                                 "adaptive_search": True, "action": s.action, "query": s.query_lex,
                                                 "retrievers": s.retrievers, "k": s.k, "hop_id": s.hop_id,
                                                 "filters": s.filters.model_dump(mode="json") if s.filters else None}))
        if self._pool is not None and len(specs) > 1:
            outs = list(self._pool.map(self._search, specs))
        else:
            outs = [self._search(s) for s in specs]
        for s, (es, err, lat) in zip(specs, outs):
            self._search_ms.append(lat)
            ops = res.ops
            ops.searches += 1
            ops.lexical_searches += "bm25" in s.retrievers
            ops.dense_searches += "dense" in s.retrievers
            if es is not None:
                ops.chunks_returned += len(es.items)
                if es.trace.rerank_applied:
                    ops.reranker_calls += 1
                    ops.reranked_candidates += min(self.ac.rerank_max_candidates, es.trace.candidates_fused)
                    rr = es.trace.timings_ms.get("rerank")
                    if rr and es.trace.candidates_fused:
                        self._rerank_ms_per_cand = rr / min(self.ac.rerank_max_candidates, es.trace.candidates_fused)
            res.searches.append({"search_id": s.search_id, "action": s.action, "query": s.query_lex,
                                 "dense_query": s.query_dense, "retrievers": s.retrievers, "k": s.k,
                                 "filters": s.filters.model_dump(mode="json") if s.filters else None,
                                 "hop_id": s.hop_id, "latency_ms": round(lat, 3), "error": err,
                                 "items": [e.evidence_id for e in es.items] if es else [],
                                 "status": es.trace.status if es else "error",
                                 "candidates": es.trace.candidates_fused if es else 0})
            events.append(("RETRIEVAL_COMPLETED", {"query_id": res.query_id, "search_id": s.search_id,
                                                   "adaptive_search": True, "status": es.trace.status if es else "error",
                                                   "n_results": len(es.items) if es else 0, "error": err,
                                                   "latency_ms": round(lat, 3),
                                                   "citations": [e.citation for e in es.items][:5] if es else []}))
        return outs

    @staticmethod
    def _doc_of(pool: dict[str, _Pooled], eid: str) -> str:
        p = pool.get(eid)
        return p.ev.document_id if p else eid

    def _hops(self, res, a: QueryAnalysis, rq: RewrittenQuery, reqs, pool, hopped: set[str], hops, filters, plan,
              valid_at, max_hops: int):
        """Bridge statements in retrieved evidence: '<entity> <relation cue> <target>' (relation hop) or
        '<reference cue> <document title>' (reference hop). Corpus text supplies at most a short target phrase -
        sanitised, instruction-like text rejected - never a strategy, k, budget or filter field."""
        out = []
        if sum(1 for h in hops if h.bridge) >= max_hops:       # bridge hops used (hop 0 / entity search excluded)
            return out
        ents = a.bridge_terms or [t for t in a.rare_terms if not t.isdigit()]
        ranked = sorted(pool.values(), key=lambda p: -p.score)[:8]
        for ent in ents:
            if ent in hopped:
                continue
            for p in ranked:
                tgt = self._relation_target(p.ev.text, ent)
                if tgt is None:
                    continue
                hopped.add(ent)
                hop_q = self._replace(rq.contextual, ent, tgt)
                parent = hops[0].hop_id if hops else None
                hid = f"{res.query_id}:H{len(hops)}"
                hops.append(RetrievalHop(hop_id=hid, parent_hop_id=parent, query=hop_q, bridge=f"{ent} -> {tgt}",
                                         purpose=f"'{ent}' is linked to '{tgt}' ({p.ev.citation}); ask about it"))
                main = next((r for r in reqs if ent in r.terms and r.kind in ("fact", "value")), None)
                core = list(main.terms) if main is not None else self.terms_fn(rq.contextual)
                vk = main.value_kind if main is not None else a.value_kind
                if main is not None:
                    self.builder.link(main, ent, tgt)
                reqs.append(self.builder.bridge(f"R{len(reqs) + 1}", ent, tgt, core, valid_at, vk, hop_q))
                res.events.append(("HOP_CREATED", hops[-1].model_dump()))
                out.append((self._spec(res, "hop", hop_q, hop_q, ["bm25", "dense"] if self.selector.dense
                                       else ["bm25"], plan.top_k, filters, plan.reranking, purpose="hop", hop_id=hid),
                            f"bridge {ent} -> {tgt} found in {p.ev.citation}"))
                break
        if out:
            return out
        for p in ranked[:3]:                                    # reference hop: "see the <title>"
            doc = self._reference(p.ev.text)
            if doc is None or doc in hopped or any(q.ev.document_id == doc for q in pool.values()):
                continue
            hopped.add(doc)
            hid = f"{res.query_id}:H{len(hops)}"
            hops.append(RetrievalHop(hop_id=hid, parent_hop_id=hops[0].hop_id if hops else None, query=rq.contextual,
                                     bridge=f"reference -> {doc}",
                                     purpose=f"{p.ev.citation} refers to '{self.catalog.docs[doc].title}'"))
            res.events.append(("HOP_CREATED", hops[-1].model_dump()))
            out.append((self._spec(res, "hop", rq.lexical_query, rq.dense_query,
                                   ["bm25", "dense"] if self.selector.dense else ["bm25"], plan.top_k,
                                   RetrievalFilters(document_ids=[doc]), plan.reranking, purpose="hop", hop_id=hid),
                        f"reference to document {doc} in {p.ev.citation}"))
            break
        return out

    def _relation_target(self, text: str, ent: str) -> str | None:
        for s in sentences(text):
            if instruction_like(s):
                continue
            low = s.lower()
            for cue in self.lexicon.relation_cues:
                i = low.find(f" {cue} ")
                if i < 0:
                    continue
                if ent not in self.terms_fn(s[:i]):
                    continue
                rest = s[i + len(cue) + 2:]
                words = _WORD.findall(re.split(r"[.;:,()]", rest)[0])[:6]
                tgt = " ".join(words)
                if tgt and self.terms_fn(tgt) and len(tgt) <= 60:
                    return tgt
        return None

    def _reference(self, text: str) -> str | None:
        for s in sentences(text):
            if instruction_like(s):
                continue
            low = s.lower()
            for cue in self.lexicon.reference_cues:
                i = low.find(cue + " ")
                if i < 0:
                    continue
                phrase = re.split(r"[.;:,()]", s[i + len(cue) + 1:])[0]
                doc = self.catalog.title_doc(self.terms_fn(" ".join(_WORD.findall(phrase)[:8])))
                if doc is not None:
                    return doc
        return None

    def _replace(self, text: str, ent: str, target: str) -> str:
        words = text.split()
        out, done = [], False
        for w in words:
            if not done and ent in self.terms_fn(w):
                out.append(target)
                done = True
            else:
                out.append(w)
        return " ".join(out) if done else f"{text} {target}"

    def _finish(self, res: AdaptiveResult, pool: dict[str, _Pooled], reqs, assess: SufficiencyAssessment,
                exc_ids: dict[str, str], budget: RetrievalBudgetState, stop: StopReason, strategy: S, ms, events,
                cache_key: str | None = None) -> AdaptiveResult:
        supporting = {i for r in reqs for i in r.evidence_ids}
        res.excluded = dict(exc_ids)
        # supporting evidence, plus (when not SUFFICIENT) the other results of the question's own searches (initial,
        # k expansion, fallbacks) as context; targeted / hop / contradiction searches contribute only what supports a
        # requirement - they were run for a purpose and failed it.
        # When the evidence is SUFFICIENT only supporting items are handed on (precision for the claim stage).
        context_ok = assess.status != "SUFFICIENT"
        main = {e for s in res.searches if s["action"].split(":")[0] in _MAIN for e in s["items"]}
        cands = [p for i, p in pool.items() if i not in exc_ids and (i in supporting or (context_ok and i in main))]
        cands.sort(key=lambda p: (0 if p.ev.evidence_id in supporting else 1, -p.score,
                                  -self.catalog.authority(p.ev.document_id), p.ev.evidence_id))
        per_doc: dict[str, int] = {}
        kept: list[Evidence] = []
        alt: set[str] = set()
        for p in cands:
            e = p.ev
            if e.evidence_id in alt or per_doc.get(e.document_id, 0) >= self.ac.max_per_document:
                continue
            per_doc[e.document_id] = per_doc.get(e.document_id, 0) + 1
            alt |= set(e.alternates)
            kept.append(e)
            if len(kept) >= self.ac.final_k:
                break
        items = [e.model_copy(update={"rank": n}) for n, e in enumerate(kept, start=1)]
        base = res.evidence.trace
        last = next((s for s in reversed(res.searches) if s["status"] != "error"), None)
        trace = base.model_copy(update={"status": "ok" if items else "empty",
                                        "warnings": [f"adaptive:{strategy.value}", f"stop:{stop.value}"]
                                        + ([] if last is None else [f"last_search:{last['search_id']}"])})
        res.evidence = EvidenceSet(evidence_set_id=f"es-ad-{res.query_id}", query=res.rewritten.contextual,
                                   items=items, token_count=0, trace=trace)
        res.assessment = assess
        res.requirements = reqs
        budget.latency_used_ms = ms()
        res.state = RetrievalState(current_strategy=strategy, iteration=budget.iterations_used,
                                   queries_executed=budget.queries_used, evidence_count=len(items),
                                   evidence_quality=assess.quality, coverage=assess.coverage,
                                   latency_spent_ms=budget.latency_used_ms, budget_remaining=budget.remaining(),
                                   stop_reason=stop)
        res.wall_ms = budget.latency_used_ms
        res.decisions.append(RetrievalDecision(
            decision_id=f"{res.query_id}:D{len(res.decisions) + 1}", query_id=res.query_id, strategy=strategy,
            action="stop", reason=stop.value, inputs={"assessment": assess.status, "coverage": assess.coverage},
            result={"evidence": len(items), "excluded": len(res.excluded)}, timestamp_ms=ms()))
        events.append(("RETRIEVAL_STOPPED", {"query_id": res.query_id, "stop_reason": stop.value,
                                             "strategy": strategy.value, "assessment": assess.status,
                                             "coverage": assess.coverage, "iterations": budget.iterations_used,
                                             "searches": res.ops.searches, "evidence": len(items),
                                             "latency_ms": budget.latency_used_ms,
                                             "budget_remaining": budget.remaining(),
                                             "excluded": res.excluded}))
        if cache_key is not None and strategy not in (S.CACHE_REUSE,) and stop not in (
                StopReason.USER_CANCELLED, StopReason.ERROR, StopReason.LATENCY_LIMIT) and items:
            self.cache.put(CacheEntry(
                key=cache_key, query=res.rewritten.contextual, intent_id=res.plan.intent_id,
                entity_terms={t for x in res.analysis.entities for t in self.terms_fn(x)} | set(res.analysis.rare_terms),
                constraint_terms={t for c in res.analysis.constraint_texts for t in self.terms_fn(c)},
                valid_at=res.plan.filters.valid_at if res.plan.filters else None,
                doc_signatures={e.document_id: self.catalog.signature(e.document_id) for e in items},
                evidence=items, status=assess.status, created_ms=0.0))
        return res


def _empty(svc, qid: str, query: str) -> EvidenceSet:
    m = svc.bundle.manifest
    return EvidenceSet(evidence_set_id=f"es-ad-{qid}", query=query or "-", items=[],
                       trace=RetrievalTrace(request_id=f"ad-{qid}", mode="hybrid", status="empty",
                                            index_version=m.index_version, corpus_hash=m.corpus_version,
                                            index_config_hash=m.index_config_hash))
