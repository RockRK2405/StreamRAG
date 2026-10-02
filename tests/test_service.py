"""Hybrid retrieval API, evidence traceability/serialization, rerank plumbing, failure handling."""

import time

import numpy as np
import pytest

from streamrag.errors import InvalidQueryError, ModelNotAvailableError
from streamrag.models import EvidenceSet, RetrievalFilters, RetrievalOptions
from streamrag.retrieval import RetrievalService, build_index
from streamrag.retrieval.embedders import HashingEmbedder

from conftest import MODELS, REPO, make_cfg, requires_bge, requires_ce


@pytest.fixture(scope="module")
def svc(fixture_bundle):
    cfg, b = fixture_bundle
    s = RetrievalService(b, cfg, embedder=HashingEmbedder(b.analyzer))
    yield s
    s.close()


@pytest.mark.parametrize("mode,method", [("bm25", "bm25"), ("dense", "dense"), ("hybrid", "hybrid_rrf")])
def test_modes_return_ranked_traceable_evidence(svc, mode, method):
    es = svc.retrieve("can ladders stay in the orchard overnight", RetrievalOptions(mode=mode, top_k=4))
    assert es.trace.status == "ok" and es.trace.mode == mode and 0 < len(es.items) <= 4
    assert [e.rank for e in es.items] == list(range(1, len(es.items) + 1))
    assert all(e.retrieval_method == method for e in es.items)
    docs = {d.document_id: d for d in svc.bundle.documents()}
    for e in es.items:
        assert docs[e.document_id].text[e.char_start:e.char_end] == e.text
        assert e.citation == f"{e.document_id} §{e.section_id}" and e.evidence_id == e.chunk_id


def test_hybrid_records_both_sources(svc):
    es = svc.retrieve("wick trimming height", RetrievalOptions(mode="hybrid"))
    top = es.items[0]
    assert top.citation == "fixture_lighthouse_manual §1.1"
    assert top.bm25_rank is not None and top.dense_rank is not None and top.rrf_score > 0


def test_evidence_set_serialization_roundtrip_and_deterministic_id(svc):
    a = svc.retrieve("fog signal storm")
    b = svc.retrieve("fog signal storm")
    assert a.evidence_set_id == b.evidence_set_id and a.chunk_ids() == b.chunk_ids()
    again = EvidenceSet.from_json(a.canonical_json())
    assert again == a


def test_batch_equals_single(svc):
    qs = ["fog signal storm", "crates per shift", "telescope calibration"]
    batch = svc.retrieve_batch(qs)
    single = [svc.retrieve(q) for q in qs]
    assert [b.chunk_ids() for b in batch] == [s.chunk_ids() for s in single]


def test_filters(svc):
    es = svc.retrieve("ladders", RetrievalOptions(filters=RetrievalFilters(document_ids=["Doc_07"])))
    assert es.items and all(e.document_id == "Doc_07" for e in es.items)
    es = svc.retrieve("ladders", RetrievalOptions(filters=RetrievalFilters(section_ids=["Doc_07 §2.2"])))
    assert [e.citation for e in es.items] == ["Doc_07 §2.2"]
    es = svc.retrieve("ladders", RetrievalOptions(filters=RetrievalFilters(document_ids=["nope"])))
    assert es.items == [] and "filters_matched_no_chunks" in es.trace.warnings and es.trace.status == "empty"


@pytest.mark.parametrize("q", ["", "   "])
def test_empty_query_is_explicit_error(svc, q):
    with pytest.raises(InvalidQueryError):
        svc.retrieve(q)


def test_no_result_query_is_empty_not_wrong(svc):
    es = svc.retrieve("zyxw qwvut", RetrievalOptions(mode="bm25"))
    assert es.items == [] and es.trace.status == "empty" and "no_lexical_terms_in_vocabulary" in es.trace.warnings


def test_dense_failure_degrades_hybrid_and_errors_dense(fixture_bundle):
    cfg, b = fixture_bundle
    s = RetrievalService(b, cfg, embedder=None)
    es = s.retrieve("fog signal")
    assert es.trace.status == "degraded" and any(w.startswith("dense_failed") for w in es.trace.warnings)
    assert all(e.retrieval_method == "bm25" for e in es.items)
    with pytest.raises(ModelNotAvailableError):
        s.retrieve("fog signal", RetrievalOptions(mode="dense"))
    strict = RetrievalService(b, cfg.model_copy(update={"retrieval": cfg.retrieval.model_copy(
        update={"on_dense_failure": "error"})}), embedder=None)
    with pytest.raises(ModelNotAvailableError):
        strict.retrieve("fog signal")


class SlowEmbedder(HashingEmbedder):
    def embed(self, texts, kind):
        time.sleep(0.2)
        return super().embed(texts, kind)


def test_retriever_timeout_degrades(fixture_bundle):
    cfg, b = fixture_bundle
    fast_timeout = cfg.model_copy(update={"retrieval": cfg.retrieval.model_copy(update={"dense_timeout_ms": 20})})
    s = RetrievalService(b, fast_timeout, embedder=SlowEmbedder(b.analyzer))
    es = s.retrieve("fog signal")
    assert es.trace.status == "degraded" and "RetrieverTimeoutError" in " ".join(es.trace.warnings)


class ReverseReranker:
    """Deterministic fake: prefers later candidates, to prove the reorder path."""
    name = "fake"

    def score(self, query, passages):
        return np.arange(len(passages), dtype=np.float32)


class SlowReranker(ReverseReranker):
    def score(self, query, passages):
        time.sleep(0.3)
        return super().score(query, passages)


def test_reranker_is_pluggable(fixture_bundle):
    cfg, b = fixture_bundle
    base = RetrievalService(b, cfg, embedder=HashingEmbedder(b.analyzer))
    off = base.retrieve("ladders orchard", RetrievalOptions(rerank=False, top_k=3))
    with pytest.raises(ModelNotAvailableError):
        base.retrieve("ladders orchard", RetrievalOptions(rerank=True))
    rr = RetrievalService(b, cfg, embedder=HashingEmbedder(b.analyzer), reranker=ReverseReranker())
    on = rr.retrieve("ladders orchard", RetrievalOptions(rerank=True, rerank_k=3, top_k=3))
    assert on.trace.rerank_applied and all(e.retrieval_method == "hybrid_rrf_rerank" for e in on.items)
    assert on.chunk_ids() == list(reversed(off.chunk_ids()))
    assert [e.rerank_score for e in on.items] == [2.0, 1.0, 0.0]


def test_reranker_timeout_keeps_fused_order(fixture_bundle):
    cfg, b = fixture_bundle
    c2 = cfg.model_copy(update={"rerank": cfg.rerank.model_copy(update={"timeout_ms": 20})})
    s = RetrievalService(b, c2, embedder=HashingEmbedder(b.analyzer), reranker=SlowReranker())
    es = s.retrieve("ladders orchard", RetrievalOptions(rerank=True, top_k=3))
    assert es.trace.status == "partial" and not es.trace.rerank_applied
    assert any(w.startswith("rerank_timeout") for w in es.trace.warnings)


@requires_bge
@requires_ce
def test_real_models_end_to_end(tmp_path):
    cfg = make_cfg(tmp_path, embedder="bge-small-en-v1.5")
    b = build_index(cfg)
    s = RetrievalService.from_config(cfg, index_path=b.path, load_rerank=True)
    es = s.retrieve("when is the telescope recalibrated", RetrievalOptions(rerank=True, top_k=3))
    assert es.trace.rerank_applied and es.items[0].citation == "fixture_observatory_notes §1"
    assert b.manifest.embedding_model.revision == "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
    s.close()
