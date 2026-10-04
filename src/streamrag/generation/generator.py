"""GroundedAnswerGenerator and ExtractiveGenerator (Phase 7; docs/answer/02).

The generator turns an ``AnswerPlan`` into a ``CandidateAnswer``: sentences with the planned facts and evidence
labels they claim to express. Nothing it returns is trusted: every sentence goes through claim extraction and
verification afterwards.

* LLM path: structured output (JSON schema). Invalid JSON or a schema violation is re-asked at most
  ``max_structured_retries`` times; after that, or on any backend error, the section(s) fall back to the extractive
  generator (recorded in ``fallback``). Arbitrary prose is never parsed.
* Extractive path (``ExtractiveGenerator``, ADR-007 fallback, always available): each planned fact becomes one
  sentence verbatim, citing its own evidence. Cannot introduce information by construction.

Every LLM call is reported through ``on_call`` (-> LLM_CALL telemetry with the request hash and the raw output, which
deterministic replay re-uses).
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable

from pydantic import BaseModel, ValidationError

from streamrag.generation import prompts
from streamrag.generation.llm import LLMResponse
from streamrag.generation.models import AnswerPlan, CandidateAnswer, CandidateSentence


class _Sentence(BaseModel):
    text: str
    facts: list[str] = []
    evidence: list[str] = []


class _Section(BaseModel):
    section_id: str
    sentences: list[_Sentence]
    answers_need: bool | None = None          # the model's answerability judgement (None: not stated)


class _Output(BaseModel):
    sections: list[_Section]


def parse_output(text: str) -> _Output:
    return _Output.model_validate(json.loads(text))


class ExtractiveGenerator:
    name = "extractive"

    def generate(self, plan: AnswerPlan, sections: list[str] | None = None) -> CandidateAnswer:
        lab = {e: label for label, e in plan.labels.items()}
        out = []
        for s in plan.sections:
            if sections is not None and s.section_id not in sections:
                continue
            for f in s.facts:
                out.append(CandidateSentence(sentence_id="", section_id=s.section_id, intent_id=s.intent_id,
                                             text=f.text, facts=[f.plan_claim_id],
                                             labels=[lab[e] for e in f.evidence_ids if e in lab], origin="extractive"))
        return CandidateAnswer(sentences=out, backend="extractive")


class GroundedAnswerGenerator:
    def __init__(self, backend, max_structured_retries: int = 1,
                 on_call: Callable[[str, LLMResponse, int], None] | None = None, answerability: bool = True) -> None:
        self.backend = backend
        self.answerability = answerability
        self.retries = max_structured_retries
        self.extractive = ExtractiveGenerator()
        self.on_call = on_call or (lambda *_: None)

    def _call(self, purpose: str, messages: list[dict]) -> tuple[_Output | None, list[LLMResponse], str | None]:
        responses, err = [], None
        msgs = messages
        for attempt in range(self.retries + 1):
            # the answerability field exists only for grounded generation with the switch on; the free baselines
            # (generate_<mode>) keep the Phase 10 schema
            r = self.backend.complete(msgs, prompts.SCHEMA if (purpose == "generate" and self.answerability)
                                      else prompts.SCHEMA_NO_ANSWERABILITY)
            responses.append(r)
            self.on_call(purpose, r, attempt)
            if not r.ok:
                return None, responses, f"backend_error: {r.error}"
            try:
                return parse_output(r.text), responses, None
            except (ValueError, ValidationError) as e:
                err = f"schema_invalid: {str(e).splitlines()[0][:200]}"
                msgs = prompts.retry_messages(messages, r.text, err)
        return None, responses, err

    def generate(self, plan: AnswerPlan, sections: list[str] | None = None,
                 kept: dict[str, list[str]] | None = None,
                 avoid: dict[str, list[str]] | None = None, facts: dict[str, list[str]] | None = None) -> CandidateAnswer:
        """Write the plan's sections (or only ``sections``) with the LLM; ``kept``: sentences already in the answer
        for a section (refinement: the model only adds what is missing)."""
        todo = [s for s in plan.sections if (sections is None or s.section_id in sections) and s.facts]
        if facts is not None:                             # only these planned facts (refinement / revision)
            todo = [s.model_copy(update={"facts": [f for f in s.facts if f.plan_claim_id in facts.get(s.section_id, [])]})
                    for s in todo]
            todo = [s for s in todo if s.facts]
        if self.backend is None or not todo:              # extractive backend configured / nothing to write
            sub = plan.model_copy(update={"sections": todo})
            return self.extractive.generate(sub)
        label_of = {e: lab for lab, e in plan.labels.items()}
        t0 = time.perf_counter()
        out, responses, err = self._call("generate", prompts.facts_messages(todo, label_of, kept, avoid,
                                                                            self.answerability))
        gen_ms = (time.perf_counter() - t0) * 1000.0
        stats = dict(llm_calls=len(responses), prompt_tokens=sum(r.prompt_tokens or 0 for r in responses),
                     output_tokens=sum(r.output_tokens or 0 for r in responses),
                     ttft_ms=responses[0].ttft_ms if responses else None, generation_ms=gen_ms,
                     model=getattr(self.backend, "model", None))
        if out is None:
            ans = self.extractive.generate(plan.model_copy(update={"sections": todo}))
            return ans.model_copy(update={"backend": "extractive", "structured_ok": False, "fallback": err, **stats})
        by_id = {s.section_id: s for s in todo}
        sents, unanswered = [], []
        for sec in out.sections:
            ps = by_id.get(sec.section_id)
            if ps is None:
                continue                                  # a section that was not asked for: ignored
            if sec.answers_need is False and self.answerability:
                unanswered.append(ps.section_id)          # the facts are related but do not answer the need
                continue
            for x in sec.sentences:
                if not x.text.strip():
                    continue
                sents.append(CandidateSentence(sentence_id="", section_id=ps.section_id, intent_id=ps.intent_id,
                                               text=" ".join(x.text.split()), facts=list(x.facts),
                                               labels=list(x.evidence), origin="llm"))
        return CandidateAnswer(sentences=sents, backend=self.backend.name, structured_ok=True,
                               unanswered_sections=unanswered, **stats)

    def generate_free(self, question: str, evidence: list[tuple[str, str, str]], mode: str) -> CandidateAnswer:
        """Ablation arms A-C (research/phase7): plain / RAG / RAG+labels generation into one section "S1"."""
        t0 = time.perf_counter()
        out, responses, err = self._call(f"generate_{mode}", prompts.evidence_messages(question, evidence, mode))
        stats = dict(llm_calls=len(responses), prompt_tokens=sum(r.prompt_tokens or 0 for r in responses),
                     output_tokens=sum(r.output_tokens or 0 for r in responses),
                     ttft_ms=responses[0].ttft_ms if responses else None,
                     generation_ms=(time.perf_counter() - t0) * 1000.0, model=getattr(self.backend, "model", None))
        if out is None:
            return CandidateAnswer(sentences=[], backend=self.backend.name, structured_ok=False, fallback=err, **stats)
        sents = [CandidateSentence(sentence_id="", section_id="S1", intent_id="", text=" ".join(x.text.split()),
                                   facts=[], labels=list(x.evidence), origin="llm")
                 for sec in out.sections for x in sec.sentences if x.text.strip()]
        return CandidateAnswer(sentences=sents, backend=self.backend.name, structured_ok=True, **stats)
