"""Helpers shared by Phase 4 tests (imported via pytest pythonpath = tests)."""

from __future__ import annotations

import time

from streamrag.controller import make_policy
from streamrag.models import Evidence, EvidenceSet, RetrievalTrace
from streamrag.retrieval import RetrievalService
from streamrag.retrieval.embedders import HashingEmbedder
from streamrag.streaming.factory import StreamingStack


def hashing_stack(cfg, bundle, **controller_overrides) -> StreamingStack:
    if controller_overrides:
        cfg = cfg.model_copy(update={"controller": cfg.controller.model_copy(update=controller_overrides)})
    service = RetrievalService(bundle, cfg, embedder=HashingEmbedder(bundle.analyzer))
    return StreamingStack(cfg, bundle, service, make_policy(cfg, bundle))


def with_streaming(cfg, **kw):
    return cfg.model_copy(update={"streaming": cfg.streaming.model_copy(update=kw)})


def with_controller(cfg, **kw):
    return cfg.model_copy(update={"controller": cfg.controller.model_copy(update=kw)})


def make_es(query: str, ids: list[str]) -> EvidenceSet:
    items = [Evidence(evidence_id=i, document_id="D", section_id=str(n), chunk_id=i, citation=f"D §{n}", text="t",
                      source_path="d.txt", char_start=0, char_end=1, rank=n + 1, score=1.0 / (n + 1),
                      retrieval_method="hybrid_rrf") for n, i in enumerate(ids)]
    trace = RetrievalTrace(request_id="rq", mode="hybrid", status="ok", index_version="3.0", corpus_hash="c",
                           index_config_hash="i")
    return EvidenceSet(evidence_set_id=f"es-{query[:8]}", query=query, items=items, trace=trace)


class FakeBackend:
    """Deterministic backend: records calls, optional wall delay, optional failure on matching queries."""

    def __init__(self, delay_s: float = 0.0, fail_on: str | None = None) -> None:
        self.calls: list[str] = []
        self.delay_s, self.fail_on = delay_s, fail_on

    def retrieve(self, query, options=None):
        self.calls.append(query)
        if self.delay_s:
            time.sleep(self.delay_s)
        if self.fail_on is not None and self.fail_on in query:
            raise RuntimeError("index backend unavailable")
        return make_es(query, [f"E{len(self.calls)}a", f"E{len(self.calls)}b"])


def types(events) -> list[str]:
    return [e.type.value for e in events]


def with_multi_intent(cfg, **kw):
    """Phase 5: multi-intent streaming enabled (+ overrides of the multi_intent section)."""
    return cfg.model_copy(update={"multi_intent": cfg.multi_intent.model_copy(update={"enabled": True, **kw})})


def mi_stack(cfg, bundle, **kw) -> StreamingStack:
    st = hashing_stack(with_multi_intent(cfg, **kw), bundle)
    return st
