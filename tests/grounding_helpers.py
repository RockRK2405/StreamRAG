"""Phase 7 test helpers. TEST FIXTURE ONLY: fictional corpora (tests/fixtures/corpus*, corpus_grounding,
corpus_injection) and *scripted* generator outputs written for the tests (never benchmark results).

``scripted(fn)`` builds a deterministic fake LLM: ``fn(sections)`` receives the sections the generator asked for -
{section_id: [(fact_id, labels, fact_text), ...]} parsed from the real prompt - and returns
{section_id: [(text, [fact ids], [labels]), ...]}. So a test can make the "LLM" echo the facts, distort them, add
unsupported material or cite labels that do not exist, and check what the pipeline does with it.
"""

from __future__ import annotations

import json
import re

import pytest

from conftest import FIX, make_cfg, model_available
from streamrag.generation.llm import ScriptedBackend
from streamrag.retrieval import build_index
from streamrag.session import AdaptivePipeline
from streaming_helpers import hashing_stack

requires_nli = pytest.mark.skipif(not model_available("nli-deberta-v3-xsmall"), reason="NLI model not downloaded")

_SEC = re.compile(r"^SECTION (\S+) ")
_FACT = re.compile(r"^\s+(F\d+) \[([^\]]*)\]: (\".*\")$")


def requested_sections(messages: list[dict]) -> dict[str, list[tuple[str, list[str], str]]]:
    out, cur = {}, None
    for line in messages[-1]["content"].splitlines():
        m = _SEC.match(line)
        if m:
            cur = m.group(1)
            out[cur] = []
            continue
        m = _FACT.match(line)
        if m and cur:
            out[cur].append((m.group(1), [x.strip() for x in m.group(2).split(",") if x.strip()], json.loads(m.group(3))))
    return out


def scripted(fn) -> ScriptedBackend:
    def respond(messages, schema):
        secs = requested_sections(messages)
        plan = fn(secs)
        return json.dumps({"sections": [{"section_id": sid, "sentences": [
            {"text": t, "facts": list(fs), "evidence": list(ls)} for t, fs, ls in sents]} for sid, sents in plan.items()]})
    return ScriptedBackend(respond)


def echo(secs):
    """A well-behaved generator: one sentence per fact, verbatim, with its own fact id and labels."""
    return {sid: [(t, [f], labs) for f, labs, t in facts] for sid, facts in secs.items()}


def grounding_stack(tmp_path_factory, corpus: str = "corpus_grounding", overrides: dict | None = None, **gen):
    tmp = tmp_path_factory.mktemp(f"g_{corpus}")
    ov = {"multi_intent.enabled": True, "session.enabled": True, "generation.enabled": True,
          "generation.backend": "extractive", **{f"generation.{k}": v for k, v in gen.items()}, **(overrides or {})}
    cfg = make_cfg(tmp, corpus=FIX / corpus, **ov)
    return hashing_stack(cfg, build_index(cfg))


def run_answer(stack, turns, backend=None, **cfg_gen):
    """Synchronous adaptive session with grounded answers; ``backend``: a scripted LLM (None = extractive)."""
    cfg = stack.cfg
    if cfg_gen:
        cfg = cfg.model_copy(update={"generation": cfg.generation.model_copy(update=cfg_gen)})
    res = stack.grounding if backend is None else stack.grounding.with_backend(backend)
    p = AdaptivePipeline(stack, cfg, grounding=res)
    out = [p.process(f"u{n}", t, n * 3000.0) for n, t in enumerate(turns, start=1)]
    return p, out
