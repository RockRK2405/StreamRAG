"""Optional gated LLM decomposition check (spec §9.6; docs/multi_intent/02 "LLM check").

The rule decomposer is always authoritative input; the LLM may only *add* needs it missed or confirm existing
ones, and its output never reaches retrieval unless it passes deterministic validation:

  1. gating      at most one call per utterance, only when a rule signal says the rules may be wrong
                 (G-a: >= 2 request cues but 1 intent; G-b: long utterance; G-c: ambiguous constraint scope;
                  G-d: low rule confidence)
  2. parsing     strict JSON against LLM_SCHEMA (pydantic); malformed -> retry once -> fall back to rules
  3. grounding   every proposed intent must quote a verbatim ``source_text`` of the transcript; unsupported
                 types, empty text, ungrounded or duplicate intents are rejected individually
  4. reconcile   matched to rule intents by term Jaccard >= tau_match: matched -> provenance 'reconciled';
                 unmatched grounded LLM intents -> new intents (provenance 'llm'); rule intents are never removed

No LLM backend is configured in this repository (ADR-007, team decision Q2 open), so ``llm_check`` defaults to
``off``. The module is exercised in tests with scripted fake backends; it has not been measured with a real model.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Literal, Protocol

from pydantic import Field, ValidationError

from streamrag.intents.decomposer import Decomposition, IntentDecomposer
from streamrag.intents.text import tokenize
from streamrag.models.base import Contract
from streamrag.models.intents import INTENT_TYPES


class LLMIntent(Contract):
    text: str = Field(min_length=1)
    source_text: str = Field(min_length=1)          # verbatim quote from the transcript
    type: str = "OTHER"


class LLMDecomposition(Contract):
    intents: list[LLMIntent] = Field(default_factory=list)


LLM_SCHEMA = LLMDecomposition.model_json_schema()


class StructuredBackend(Protocol):
    name: str

    def complete_json(self, prompt: str, schema: dict, timeout_s: float) -> str: ...


@dataclass
class LLMCheckReport:
    called: bool = False
    gated_by: list[str] = field(default_factory=list)
    status: Literal["not_gated", "off", "no_backend", "ok", "fallback_rules"] = "not_gated"
    attempts: int = 0
    issues: list[str] = field(default_factory=list)
    added: list[str] = field(default_factory=list)
    reconciled: list[str] = field(default_factory=list)
    latency_ms: float = 0.0


PROMPT = ("Split the user's request into its separate information needs. Return JSON only, matching the schema. "
          "For each need give 'text' (the need), 'source_text' (an exact quote from the transcript that states it) "
          "and 'type' (one of {types}). Do not add needs that are not stated. Transcript: {transcript!r}")


class LLMDecompositionCheck:
    def __init__(self, decomposer: IntentDecomposer, backend: StructuredBackend | None, mode: str = "off",
                 timeout_ms: int = 300, tau_match: float = 0.5, long_tokens: int = 30, low_confidence: float = 0.6):
        self.dec, self.backend, self.mode = decomposer, backend, mode
        self.timeout_s, self.tau, self.long, self.low = timeout_ms / 1000.0, tau_match, long_tokens, low_confidence

    def gating(self, d: Decomposition) -> list[str]:
        lx = self.dec.lx
        words = [t.lower for t in tokenize(d.transcript) if not t.punct]
        cues = sum(1 for w in words if w in lx.base.question_words) + \
            sum(1 for c in d.clauses if c.role == "REQUEST" and c.head)
        reasons = []
        if cues >= 2 and len(d.active) == 1:
            reasons.append("G-a")
        if len([w for w in words if self.dec._is_content(w)]) >= self.long:
            reasons.append("G-b")
        if any(k.scope_confidence < 1.0 for k in d.constraints):
            reasons.append("G-c")
        if d.active and d.confidence < self.low:
            reasons.append("G-d")
        return reasons

    def run(self, d: Decomposition) -> tuple[Decomposition, LLMCheckReport]:
        rep = LLMCheckReport(gated_by=self.gating(d))
        if not rep.gated_by:
            return d, rep
        if self.mode == "off":
            rep.status = "off"
            return d, rep
        if self.backend is None:
            rep.status = "no_backend"
            return d, rep
        t0 = time.perf_counter()
        parsed = None
        for attempt in range(2):                      # one retry
            rep.attempts = attempt + 1
            rep.called = True
            try:
                raw = self.backend.complete_json(
                    PROMPT.format(types=", ".join(INTENT_TYPES), transcript=d.transcript), LLM_SCHEMA, self.timeout_s)
                parsed = LLMDecomposition.model_validate(json.loads(raw))
                break
            except (json.JSONDecodeError, ValidationError, TypeError) as exc:
                rep.issues.append(f"malformed_output: {exc.__class__.__name__}")
            except TimeoutError:
                rep.issues.append("timeout")
                break
        rep.latency_ms = round((time.perf_counter() - t0) * 1000.0, 3)
        if parsed is None:
            rep.status = "fallback_rules"
            return d, rep
        return self._reconcile(d, parsed, rep)

    def _reconcile(self, d: Decomposition, llm: LLMDecomposition, rep: LLMCheckReport):
        tr_low = d.transcript.lower()
        accepted = []
        for li in llm.intents:
            if li.type not in INTENT_TYPES:
                rep.issues.append(f"unsupported_type: {li.type}")
                continue
            k = tr_low.find(li.source_text.lower().strip())
            if k < 0:
                rep.issues.append(f"missing_source_span: {li.source_text!r}")
                continue
            accepted.append((li, k, k + len(li.source_text.strip())))
        rule_terms = [(i, set(i.terms)) for i in d.active]
        new_spans = []
        for li, s, e in accepted:
            terms = set(self.dec.terms_fn(d.transcript[s:e]))
            if not terms:
                rep.issues.append(f"empty_intent: {li.source_text!r}")
                continue
            best = max(((len(terms & t) / len(terms | t), i) for i, t in rule_terms if t),
                       key=lambda x: x[0], default=(0.0, None))
            if best[1] is not None and best[0] >= self.tau:
                rep.reconciled.append(best[1].key)
            elif any(abs(s - a) < 1 and abs(e - b) < 1 for a, b in new_spans):
                rep.issues.append(f"duplicate_intent: {li.source_text!r}")
            else:
                new_spans.append((s, e))
        if not new_spans:
            rep.status = "ok"
            if rep.reconciled:
                d.source = "reconciled"
            return d, rep
        merged = self.dec.decompose_with_extra_spans(d.transcript, d.utterance_id, new_spans, d)
        rep.added = [i.key for i in merged.active if getattr(i, "from_llm", False)]
        rep.status = "ok"
        return merged, rep
