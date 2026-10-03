"""Synchronous drivers of the adaptive session (Phase 6): one call per user turn (benchmarks, demo, tests).

``AdaptivePipeline``     incremental: interpretation update -> change detection -> delta plan -> execute only the
                         plan (retrieve | reuse | cache hit) -> evidence / claims -> answer version.
``FullRestartPipeline``  baseline (brief §41): every turn rebuilds a fresh session from the whole conversation
                         (full intent analysis of every turn), retrieves every active need of the frame, re-extracts
                         and re-validates every claim and regenerates the answer from scratch.

Both apply the same Phase 4 controller gate (suppressed / not-retrieval-worthy turns change nothing unless they carry
a late detail), so the comparison isolates incremental processing. Stage times are wall-clock measurements.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from streamrag.context.cues import late_detail_gate
from streamrag.context.models import ContextChange
from streamrag.controller.models import ControllerInput
from streamrag.delta.models import DeltaPlan
from streamrag.intents.tracker import IntentTracker
from streamrag.ledger.ledger import QueryLedger
from streamrag.models.answers import AnswerVersion
from streamrag.models.events import TelemetryEvent
from streamrag.models.retrieval import RetrievalOptions
from streamrag.multi_retrieval import MultiQueryRetriever
from streamrag.multi_retrieval.coordinator import gate_open
from streamrag.session.engine import AdaptiveSessionEngine
from streamrag.session.safety import redact
from streamrag.streaming.events import EventBus


@dataclass
class TurnResult:
    utterance_id: str
    gate: str
    changes: list[ContextChange] = field(default_factory=list)
    plan: DeltaPlan | None = None
    answer: AnswerVersion | None = None
    retrievals: int = 0
    reused_active: int = 0
    cache_hits: int = 0
    timings_ms: dict[str, float] = field(default_factory=dict)
    queries: dict[str, str] = field(default_factory=dict)       # intent id -> active query text after the turn


class AdaptivePipeline:
    def __init__(self, stack, cfg=None, full_restart: bool = False, session_id: str = "sync",
                 dispatch: str = "parallel") -> None:
        self.stack, self.cfg = stack, cfg or stack.cfg
        st = stack.intent_stack
        self.tracker = IntentTracker(session_id, st.decomposer, self.cfg.multi_intent)
        self.ledger = QueryLedger(session_id)
        self.options = RetrievalOptions(mode=self.cfg.streaming.retrieval_mode,
                                        top_k=self.cfg.multi_intent.max_candidates_per_intent,
                                        rerank=self.cfg.streaming.rerank)
        self._clock = 0.0
        self.bus = EventBus(session_id, lambda: self._clock)
        self.engine = AdaptiveSessionEngine(session_id, self.cfg, self.tracker, self.ledger, st.query_builder,
                                            stack.index_hash, self.options, self._emit, full_restart)
        self.retriever = MultiQueryRetriever(stack.service, self.options,
                                             max_concurrent=self.cfg.multi_intent.max_concurrent_retrievals)
        self.dispatch = dispatch
        self.evidence: dict[str, object] = {}
        self.results: list[TurnResult] = []

    def close(self) -> None:
        self.retriever.close()

    @property
    def events(self) -> list[TelemetryEvent]:
        return self.bus.events

    def _emit(self, type_, component, payload, uid=None, intent_id=None, query_id=None):
        return self.bus.emit(type_, component, payload, uid, None, intent_id, query_id)

    def _gate(self, uid: str, text: str, now: float) -> str:
        inp = ControllerInput(self.tracker.session_id, uid, text, "utterance_end", now, None, [], True, False)
        decision, _, _ = self.stack.policy.decide(inp, self.ledger, 0)
        if gate_open(decision):
            return "open"
        dec = self.stack.intent_stack.decomposer
        if late_detail_gate(decision, bool(self.tracker.active_session_intents()), text, dec.lx, dec.terms_fn):
            return "late_detail"
        return f"closed:{decision.reason}"

    def _clean(self, text: str) -> str:
        """What interpretation sees: PII-redacted text (memory safety at ingestion; docs/session/10)."""
        return redact(text)[0] if self.cfg.session.redact_pii else text

    def interpret_only(self, uid: str, text: str, now: float) -> None:
        """Interpretation without retrieval (used by the full-restart baseline to rebuild earlier turns)."""
        self._clock = now
        self.engine.observe_utterance(uid, text, True)
        if self._gate(uid, text, now).startswith("closed"):
            return
        iset, delta, d = self.tracker.update(uid, self._clean(text), now, [r.terms for r in self.ledger.all()],
                                             final=True)
        self.engine.interpret(uid, iset, delta, d, now)

    def process(self, uid: str, text: str, now: float) -> TurnResult:
        t_all = time.perf_counter()
        self._clock = now
        timings: dict[str, float] = {}
        self.engine.observe_utterance(uid, text, True)
        gate = self._gate(uid, text, now)
        res = TurnResult(uid, gate)
        if gate.startswith("closed"):
            res.changes = [self.engine.no_change(uid, gate, now)]
            timings["total"] = (time.perf_counter() - t_all) * 1000.0
            res.timings_ms = timings
            self.results.append(res)
            return res
        n_t = {k: len(v) for k, v in self.engine.timings.items()}
        t0 = time.perf_counter()
        iset, delta, d = self.tracker.update(uid, self._clean(text), now, [r.terms for r in self.ledger.all()],
                                             final=True)
        timings["interpretation"] = (time.perf_counter() - t0) * 1000.0
        plan = self.engine.interpret(uid, iset, delta, d, now)
        if plan is None:
            res.changes = [self.engine.no_change(uid, "no_interpretation_change", now)]
        else:
            res.plan = plan
            res.changes = [c for c in self.engine.changes if c.change_id in plan.change_ids]
            t1 = time.perf_counter()
            self._execute(plan, uid, now, res)
            timings["retrieval"] = (time.perf_counter() - t1) * 1000.0
        res.answer = self.engine.commit_answer(uid, now)
        for k, v in self.engine.timings.items():
            new = v[n_t.get(k, 0):]
            if new:
                timings[k] = sum(new)
        timings["total"] = (time.perf_counter() - t_all) * 1000.0
        res.timings_ms = {k: round(v, 4) for k, v in timings.items()}
        res.queries = {i.intent_id: (self.ledger.active_for_intent(i.intent_id).query_text
                                     if self.ledger.active_for_intent(i.intent_id) else "")
                       for i in self.tracker.active_session_intents()}
        self.results.append(res)
        return res

    def _execute(self, plan: DeltaPlan, uid: str, now: float, res: TurnResult) -> None:
        for ch in (c for c in self.engine.changes if c.change_id in plan.change_ids):   # as the coordinator does
            for old in ch.superseded_intents:
                self.ledger.mark_intent_stale(old, "intent_superseded")
            if ch.change_type == "INTENT_REMOVAL":
                for old in ch.affected_intents:
                    self.ledger.mark_intent_stale(old, "intent_removed")
        creates = []
        for a in plan.queries_to_create:
            spans = [(c.span.start, c.span.end) for c in a.query.components if c.span.utterance_id == uid]
            rec, _ = self.ledger.create(uid, a.query.text, self.tracker.utterances[uid].transcript, spans, a.query.terms,
                                        now, "final", None, "utterance_end", a.reason, intent_id=a.intent_id,
                                        intent_version=a.intent_version, parent_query_id=a.parent_query_id,
                                        derived_from_change_id=a.change_id, semantic_key=a.semantic_key)
            self.engine.query_dispatched(a, rec.query_id)
            creates.append((a, rec))
        for a in plan.queries_to_reuse:
            if a.action == "cache_hit":
                src = self.ledger.get(a.reused_query_id)
                rec, _ = self.ledger.create(uid, a.query.text, self.tracker.utterances[uid].transcript, [],
                                            a.query.terms, now, "final", None, "utterance_end", a.reason,
                                            intent_id=a.intent_id, intent_version=a.intent_version,
                                            parent_query_id=a.parent_query_id, derived_from_change_id=a.change_id,
                                            semantic_key=a.semantic_key, status="reused",
                                            reused_from=a.reused_query_id)
                self.ledger.update(rec.query_id, evidence_ids=list(src.evidence_ids), retrieval_status="cache_hit",
                                   evidence_set_id=src.evidence_set_id)
                self.engine.query_reused(a, uid, now, rec.query_id, self.evidence.get(a.reused_query_id))
                res.cache_hits += 1
            else:
                self.engine.query_reused(a, uid, now, None, self.evidence.get(a.reused_query_id))
                res.reused_active += 1
        if not creates:
            return
        out = self.retriever.retrieve([a.query for a, _ in creates], self.dispatch)
        by = out.by_intent()
        for a, rec in creates:
            o = by[a.intent_id]
            if o.evidence is None:
                self.ledger.update(rec.query_id, status="failed", retrieval_status="error", error=o.error)
                continue
            self.evidence[rec.query_id] = o.evidence
            self.ledger.update(rec.query_id, status="completed", retrieval_status=o.evidence.trace.status,
                               evidence_ids=[e.evidence_id for e in o.evidence.items],
                               evidence_set_id=o.evidence.evidence_set_id,
                               citations=[e.citation for e in o.evidence.items])
            self.engine.on_result(a.intent_id, rec.query_id, o.evidence, now, uid)
        res.retrievals += len(creates)


class FullRestartPipeline:
    """Baseline: every turn = a fresh session over the whole conversation so far (no reuse of any kind)."""

    def __init__(self, stack, cfg=None, session_id: str = "restart") -> None:
        self.stack, self.cfg, self.session_id = stack, cfg or stack.cfg, session_id
        self.turns: list[tuple[str, str, float]] = []
        self.last: AdaptivePipeline | None = None

    def close(self) -> None:
        if self.last is not None:
            self.last.close()

    def process(self, uid: str, text: str, now: float) -> TurnResult:
        t0 = time.perf_counter()
        self.turns.append((uid, text, now))
        if self.last is not None:
            self.last.close()
        p = AdaptivePipeline(self.stack, self.cfg, full_restart=True, session_id=self.session_id)
        for u, t, n in self.turns[:-1]:
            p.interpret_only(u, t, n)                 # full intent analysis of the whole conversation again
        res = p.process(uid, text, now)
        res.timings_ms["total"] = round((time.perf_counter() - t0) * 1000.0, 4)
        self.last = p
        return res
