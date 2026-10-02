"""Retrieval policies: the rule-first RetrievalController (default) and two ablation baselines.

RetrievalController decision procedure (spec §8.5, single active query in Phase 4):

  1. act classification          -> SKIP(suppressed) for confident PRESENTATION / SOCIAL / BACKCHANNEL / META
  2. query construction          -> WAIT(no_content) / at end SKIP(not_worthy: empty)
  3. signals (stability, worthiness, novelty, anchors)
  4. provisional ticks:  dangling -> WAIT; insufficient (content/anchor) -> WAIT; stability < min -> WAIT;
                         worthiness < min -> WAIT
     utterance end:      UNKNOWN act with worthiness < min -> SKIP(not_worthy)   (information requests are
                         always retrieved at the end: Phase 2 correction K4)
  5. novelty < threshold -> SKIP(redundant, ledger_ref)
  6. storm guards (provisional): budget (one slot reserved for the final query), cooldown, in-flight
     utterance end: absolute budget only
  7. RETRIEVE(trigger = provisional | final)

Baselines (Exp "controller ablation"): EndOnlyPolicy (A: static RAG, retrieve once at utterance end, no
suppression) and EveryChunkPolicy (B: retrieve on every chunk that changes the transcript).
"""

from __future__ import annotations

from streamrag.config.settings import ControllerConfig
from streamrag.controller.acts import SUPPRESSED_ACTS
from streamrag.controller.models import ControllerInput, RetrievalDecision
from streamrag.controller.query_builder import BuiltQuery, QueryBuilder
from streamrag.controller.signals import SignalComputer
from streamrag.ledger.ledger import QueryLedger


def _public(signals: dict) -> dict:
    return {k: v for k, v in signals.items() if not k.startswith("_")}


class RetrievalController:
    name = "rules"

    def __init__(self, cfg: ControllerConfig, classifier, builder: QueryBuilder, signals: SignalComputer) -> None:
        self.cfg, self.classifier, self.builder, self.sig = cfg, classifier, builder, signals

    def decide(self, inp: ControllerInput, ledger: QueryLedger, in_flight: int) -> tuple[RetrievalDecision, BuiltQuery, list[str]]:
        cfg = self.cfg
        final = inp.tick == "utterance_end"
        base = dict(tick=inp.tick, session_id=inp.session_id, utterance_id=inp.utterance_id,
                    trigger_chunk=inp.trigger_chunk, t_session_ms=inp.now_ms, policy=self.name)
        act = self.classifier.classify(inp.transcript)
        query = self.builder.build(inp.transcript)
        s = self.sig.compute(inp.transcript, query, act, inp.prev_tick_terms, inp.quiet, final, ledger)
        terms = s["_terms"]
        pub = _public(s)

        def mk(decision, reason, conf, **kw):
            return RetrievalDecision(decision=decision, reason=reason, reasons=kw.pop("reasons", [reason]),
                                     confidence=round(max(0.0, min(1.0, conf)), 3), signals=pub,
                                     query_text=query.text or None, **base, **kw), query, terms

        if act.act in SUPPRESSED_ACTS and act.confidence >= cfg.act_suppress_confidence:
            return mk("SKIP", SUPPRESSED_ACTS[act.act], act.confidence, skip_kind="suppressed")
        if query.empty or not terms:
            if final:
                return mk("SKIP", "empty_content", 1.0, skip_kind="not_worthy")
            return mk("WAIT", "no_content", 1.0)
        stab, worth = s["semantic_stability"], s["retrieval_worthiness"]
        if not final:
            if s["dangling"]:
                return mk("WAIT", "trailing_function_word", 1.0 - stab)
            sufficient = s["anchors"] > 0 and (s["content_terms"] >= cfg.min_content_tokens
                                               or s["anchor_strength"] >= cfg.strong_anchor_strength)
            if not sufficient:
                return mk("WAIT", "low_specificity", 1.0 - stab)
            if stab < cfg.min_stability:
                return mk("WAIT", "awaiting_stability", 1.0 - stab)
            if worth < cfg.min_worthiness:
                return mk("WAIT", "not_yet_retrieval_worthy", 1.0 - worth)
        elif act.act != "INFO_REQUEST" and worth < cfg.min_worthiness:
            return mk("SKIP", "not_retrieval_worthy", 1.0 - worth, skip_kind="not_worthy")
        if s["novelty"] < cfg.novelty_threshold:
            return mk("SKIP", "redundant_query", 1.0 - s["novelty"], skip_kind="redundant",
                      ledger_ref=s["most_similar_query"])
        n_done = ledger.retrieval_count(inp.utterance_id)
        if final:
            if n_done >= cfg.max_retrievals_per_utterance:
                return mk("SKIP", "budget_exhausted", 1.0, skip_kind="budget")
        else:
            limit = cfg.max_retrievals_per_utterance - (1 if cfg.reserve_final_retrieval else 0)
            if n_done >= limit:
                return mk("WAIT", "provisional_budget_exhausted", 1.0)
            last = ledger.last_issued_ms(inp.utterance_id)
            if last is not None and inp.now_ms - last < cfg.retrieval_cooldown_ms:
                return mk("WAIT", "cooldown", 1.0)
            if in_flight and not cfg.allow_parallel_retrieval:
                return mk("WAIT", "retrieval_in_flight", 1.0)
        conf = 0.4 * stab + 0.4 * worth + 0.2 * s["novelty"]
        return mk("RETRIEVE", "utterance_end_final_query" if final else "stable_retrieval_worthy_request", conf,
                  trigger="final" if final else "provisional")


class EndOnlyPolicy:
    """Ablation A (static RAG): wait for the end of the utterance, then always retrieve once."""

    name = "end_only"

    def __init__(self, builder: QueryBuilder, signals: SignalComputer) -> None:
        self.builder, self.sig = builder, signals

    def decide(self, inp: ControllerInput, ledger: QueryLedger, in_flight: int):
        q = self.builder.build(inp.transcript)
        terms = self.sig.terms(q.text) if q.text else []
        base = dict(tick=inp.tick, session_id=inp.session_id, utterance_id=inp.utterance_id,
                    trigger_chunk=inp.trigger_chunk, t_session_ms=inp.now_ms, policy=self.name, query_text=q.text or None)
        if inp.tick != "utterance_end":
            return RetrievalDecision(decision="WAIT", reason="policy_end_only", confidence=1.0, **base), q, terms
        if not terms:
            return RetrievalDecision(decision="SKIP", reason="empty_content", skip_kind="not_worthy",
                                     confidence=1.0, **base), q, terms
        return RetrievalDecision(decision="RETRIEVE", reason="policy_end_only", confidence=1.0, trigger="final",
                                 **base), q, terms


class EveryChunkPolicy:
    """Ablation B (naive streaming): retrieve whenever a chunk changes the transcript, no guards."""

    name = "every_chunk"

    def __init__(self, builder: QueryBuilder, signals: SignalComputer) -> None:
        self.builder, self.sig = builder, signals

    def decide(self, inp: ControllerInput, ledger: QueryLedger, in_flight: int):
        q = self.builder.build(inp.transcript)
        terms = self.sig.terms(q.text) if q.text else []
        base = dict(tick=inp.tick, session_id=inp.session_id, utterance_id=inp.utterance_id,
                    trigger_chunk=inp.trigger_chunk, t_session_ms=inp.now_ms, policy=self.name, query_text=q.text or None)
        if not terms or inp.tick == "stability_timer":
            return RetrievalDecision(decision="WAIT", reason="policy_every_chunk_no_change", confidence=1.0, **base), q, terms
        if inp.tick == "utterance_end":
            last = ledger.active(inp.utterance_id)
            if last is not None and last.query_text == q.text:
                return RetrievalDecision(decision="SKIP", reason="already_retrieved_identical", skip_kind="redundant",
                                         confidence=1.0, ledger_ref=last.query_id, **base), q, terms
        trig = "final" if inp.tick == "utterance_end" else "provisional"
        return RetrievalDecision(decision="RETRIEVE", reason="policy_every_chunk", confidence=1.0, trigger=trig,
                                 **base), q, terms
