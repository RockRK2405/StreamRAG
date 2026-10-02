"""Retrieval API: ``retrieve(query, options) -> EvidenceSet`` and ``retrieve_batch``.

Pipeline (ADR-001/003/009):  query -> [BM25 | dense] -> candidate union -> RRF -> dedup -> [rerank] -> top-k
-> Evidence objects. This layer knows nothing about streaming, voice, answer generation or UI, and it can only
search the loaded index (no network, no document injection API).
"""

from __future__ import annotations

import hashlib
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from pathlib import Path

import numpy as np

from streamrag.config.settings import StreamRagConfig
from streamrag.errors import (
    EmbeddingDimensionMismatchError,
    InvalidQueryError,
    ModelNotAvailableError,
    RerankerTimeoutError,
    RetrieverTimeoutError,
)
from streamrag.models.base import canonical_dumps
from streamrag.models.evidence import Evidence, EvidenceSet, RetrievalTrace
from streamrag.models.retrieval import RetrievalFilters, RetrievalHit, RetrievalOptions, RetrievalRequest, RetrievalResult
from streamrag.retrieval.dedup import deduplicate
from streamrag.retrieval.embedders import Embedder, load_embedder
from streamrag.retrieval.fusion import Candidate, rrf_fuse, single_list
from streamrag.retrieval.rerank import Reranker, load_reranker
from streamrag.retrieval.store import IndexBundle, load_index, resolve_index
from streamrag.telemetry.timing import Stopwatch

_METHOD = {("bm25", False): "bm25", ("dense", False): "dense", ("hybrid", False): "hybrid_rrf",
           ("bm25", True): "bm25_rerank", ("dense", True): "dense_rerank", ("hybrid", True): "hybrid_rrf_rerank"}


class RetrievalService:
    def __init__(self, bundle: IndexBundle, cfg: StreamRagConfig, embedder: Embedder | None = None,
                 reranker: Reranker | None = None, init_warnings: list[str] | None = None) -> None:
        self.bundle = bundle
        self.cfg = cfg
        self.embedder = embedder
        self.reranker = reranker
        self.init_warnings = list(init_warnings or [])
        if embedder is not None and bundle.dense is not None and embedder.dimension != bundle.dense.info.dimension:
            raise EmbeddingDimensionMismatchError(
                f"query embedder '{embedder.name}' dim {embedder.dimension} != index dim {bundle.dense.info.dimension}")
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="streamrag-retrieval")
        chunks = bundle.chunks
        self._doc_ids = np.array([c.document_id for c in chunks], dtype=object)
        self._citations = np.array([c.citation for c in chunks], dtype=object)
        self._section_ids = np.array([c.section_id for c in chunks], dtype=object)

    # ------------------------------------------------------------------ construction
    @classmethod
    def from_config(cls, cfg: StreamRagConfig, index_path: Path | None = None,
                    load_rerank: bool | None = None) -> "RetrievalService":
        bundle = load_index(index_path or resolve_index(cfg))
        warnings: list[str] = []
        embedder = None
        if bundle.dense is not None:
            try:
                embedder = load_embedder(bundle.dense.info.name, cfg.paths.model_registry, cfg.paths.models_dir,
                                         bundle.analyzer, cfg.dense.intra_op_threads, cfg.dense.batch_size)
            except ModelNotAvailableError as exc:
                if cfg.retrieval.on_dense_failure == "error":
                    raise
                warnings.append(f"dense_unavailable: {exc}")
        reranker = None
        if load_rerank if load_rerank is not None else cfg.rerank.enabled:
            reranker = load_reranker(cfg.rerank.model, cfg.paths.model_registry, cfg.paths.models_dir,
                                     cfg.dense.intra_op_threads, cfg.rerank.batch_size)
        return cls(bundle, cfg, embedder, reranker, warnings)

    def close(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)

    # ------------------------------------------------------------------ public API
    def retrieve(self, query: str | RetrievalRequest, options: RetrievalOptions | None = None) -> EvidenceSet:
        req = query if isinstance(query, RetrievalRequest) else RetrievalRequest(query=query, options=options or RetrievalOptions())
        return self._retrieve(req, None, 0.0)

    def retrieve_batch(self, queries: list[str | RetrievalRequest],
                       options: RetrievalOptions | None = None) -> list[EvidenceSet]:
        reqs = [q if isinstance(q, RetrievalRequest) else RetrievalRequest(query=q, options=options or RetrievalOptions())
                for q in queries]
        for r in reqs:
            self._validate(r.query)
        vecs, share = None, 0.0
        needs_dense = any((r.options.mode or self.cfg.retrieval.mode) in ("dense", "hybrid") for r in reqs)
        if needs_dense and self.embedder is not None and self.bundle.dense is not None and reqs:
            with Stopwatch() as sw:
                vecs = self.embedder.embed([r.query for r in reqs], "query")
            share = sw.ms / len(reqs)
        return [self._retrieve(r, None if vecs is None else vecs[i], share) for i, r in enumerate(reqs)]

    @staticmethod
    def to_retrieval_result(es: EvidenceSet) -> RetrievalResult:
        status = es.trace.status
        return RetrievalResult(retrieval_id=es.trace.request_id, status=status, latency_ms=es.trace.timings_ms,
                               n_candidates=es.trace.candidates_fused,
                               results=[RetrievalHit(chunk_id=e.chunk_id, lexical_rank=e.bm25_rank, dense_rank=e.dense_rank,
                                                     rrf_score=e.rrf_score, rerank_score=e.rerank_score) for e in es.items])

    # ------------------------------------------------------------------ internals
    @staticmethod
    def _validate(query: str) -> None:
        if not query or not query.strip():
            raise InvalidQueryError("query is empty")

    def _mask(self, f: RetrievalFilters | None) -> np.ndarray | None:
        if f is None or (f.document_ids is None and f.section_ids is None):
            return None
        mask = np.ones(len(self.bundle.chunks), dtype=bool)
        if f.document_ids is not None:
            mask &= np.isin(self._doc_ids, list(f.document_ids))
        if f.section_ids is not None:
            wanted = list(f.section_ids)
            mask &= np.isin(self._citations, wanted) | np.isin(self._section_ids, wanted)
        return mask

    def _with_timeout(self, fn, timeout_ms: int, exc_type):
        fut = self._executor.submit(fn)
        try:
            return fut.result(timeout=timeout_ms / 1000.0)
        except FutureTimeout as exc:
            raise exc_type(f"exceeded {timeout_ms} ms") from exc

    def _resolve(self, o: RetrievalOptions) -> dict:
        r, rr = self.cfg.retrieval, self.cfg.rerank
        return {"mode": o.mode or r.mode, "top_k": o.top_k or r.top_k, "lexical_k": o.lexical_k or r.lexical_k,
                "dense_k": o.dense_k or r.dense_k, "rrf_k": o.rrf_k or r.rrf_k,
                "rerank": rr.enabled if o.rerank is None else o.rerank, "rerank_k": o.rerank_k or rr.rerank_k,
                "dedup": self.cfg.dedup.enabled if o.dedup is None else o.dedup,
                "filters": o.filters.model_dump(mode="json") if o.filters else None}

    def _retrieve(self, req: RetrievalRequest, qvec: np.ndarray | None, embed_share_ms: float) -> EvidenceSet:
        self._validate(req.query)
        o = self._resolve(req.options)
        mode = o["mode"]
        b = self.bundle
        digest = hashlib.sha1(canonical_dumps({"q": req.query, "o": o, "i": b.manifest.content_hash}).encode()).hexdigest()[:16]
        request_id = req.request_id or f"rq-{digest}"
        warnings = list(self.init_warnings)
        status = "ok"
        timings: dict[str, float] = {}
        lex: list[tuple[int, float]] = []
        dense_hits: list[tuple[int, float]] = []
        dense_ok = False
        mask = self._mask(req.options.filters)

        with Stopwatch() as total:
            if mask is not None and not mask.any():
                warnings.append("filters_matched_no_chunks")
            else:
                if mode in ("bm25", "hybrid"):
                    with Stopwatch() as sw:
                        k = max(o["lexical_k"], o["top_k"]) if mode == "bm25" else o["lexical_k"]
                        lex = b.bm25.search(req.query, b.analyzer, k, mask)
                    timings["lexical"] = sw.ms
                    if not b.bm25.query_terms(req.query, b.analyzer):
                        warnings.append("no_lexical_terms_in_vocabulary")
                if mode in ("dense", "hybrid"):
                    try:
                        if b.dense is None or self.embedder is None:
                            raise ModelNotAvailableError("dense index or query embedder not available")
                        if qvec is None:
                            with Stopwatch() as sw:
                                qvec = self._with_timeout(lambda: self.embedder.embed([req.query], "query")[0],
                                                          self.cfg.retrieval.dense_timeout_ms, RetrieverTimeoutError)
                            timings["embed"] = sw.ms
                        else:
                            timings["embed"] = embed_share_ms
                        with Stopwatch() as sw:
                            k = max(o["dense_k"], o["top_k"]) if mode == "dense" else o["dense_k"]
                            dense_hits = b.dense.search(qvec, k, mask, self.cfg.retrieval.dense_min_similarity)
                        timings["dense"] = sw.ms
                        dense_ok = True
                    except (ModelNotAvailableError, RetrieverTimeoutError) as exc:
                        if mode == "dense" or self.cfg.retrieval.on_dense_failure == "error":
                            raise
                        warnings.append(f"dense_failed_lexical_only: {exc.__class__.__name__}: {exc}")
                        status = "degraded"

            with Stopwatch() as sw:
                if mode == "hybrid" and dense_ok:
                    cands = rrf_fuse(lex, dense_hits, o["rrf_k"])
                elif mode in ("bm25", "hybrid"):
                    cands = single_list(lex, "lexical")
                else:
                    cands = single_list(dense_hits, "dense")
            timings["fusion"] = sw.ms
            n_fused = len(cands)

            with Stopwatch() as sw:
                vectors = b.dense.matrix if b.dense is not None else None
                cfg_dedup = self.cfg.dedup.model_copy(update={"enabled": o["dedup"]})
                cands, removed = deduplicate(cands, b.chunks, vectors, cfg_dedup)
            timings["dedup"] = sw.ms

            reranked = False
            if o["rerank"] and cands:
                if self.reranker is None:
                    raise ModelNotAvailableError("rerank requested but no reranker is loaded (rerank.enabled / --rerank)")
                head = cands[: o["rerank_k"]]
                try:
                    with Stopwatch() as sw:
                        scores = self._with_timeout(
                            lambda: self.reranker.score(req.query, [b.chunks[c.row].index_text for c in head]),
                            self.cfg.rerank.timeout_ms, RerankerTimeoutError)
                    timings["rerank"] = sw.ms
                    for c, s in zip(head, scores):
                        c.rerank = float(s)
                    order = sorted(range(len(head)), key=lambda i: (-round(head[i].rerank, 6), i))
                    cands = [head[i] for i in order] + cands[o["rerank_k"]:]
                    reranked = True
                except RerankerTimeoutError as exc:
                    warnings.append(f"rerank_timeout_fused_order_kept: {exc}")
                    status = "partial"

            effective = "bm25" if mode == "hybrid" and not dense_ok else mode   # degraded hybrid is lexical-only
            items = [self._evidence(c, rank, effective, reranked and c.rerank is not None)
                     for rank, c in enumerate(cands[: o["top_k"]], start=1)]
        timings["total"] = total.ms
        if not items and status == "ok":
            status = "empty"
        trace = RetrievalTrace(request_id=request_id, mode=mode, status=status, rerank_applied=reranked,
                               candidates_lexical=len(lex), candidates_dense=len(dense_hits), candidates_fused=n_fused,
                               dedup_removed=removed,
                               timings_ms={k: round(v, 4) for k, v in timings.items()}, warnings=warnings,
                               index_version=b.manifest.index_version, corpus_hash=b.manifest.corpus_version,
                               index_config_hash=b.manifest.index_config_hash)
        return EvidenceSet(evidence_set_id=f"es-{digest}", query=req.query, items=items,
                           token_count=sum(b.chunks[c.row].token_count for c in cands[: o["top_k"]]), trace=trace)

    def _evidence(self, c: Candidate, rank: int, mode: str, reranked: bool) -> Evidence:
        ch = self.bundle.chunks[c.row]
        if reranked:
            score = c.rerank
        elif c.rrf is not None:
            score = c.rrf
        else:
            score = c.lex_score if c.lex_score is not None else c.dense_score
        chunks = self.bundle.chunks
        return Evidence(
            evidence_id=ch.chunk_id, document_id=ch.document_id, section_id=ch.section_id, chunk_id=ch.chunk_id,
            citation=ch.citation, section_title=ch.section_title, section_path=ch.section_path, text=ch.text,
            source_path=ch.source_path, char_start=ch.char_start, char_end=ch.char_end, page_start=ch.page_start,
            page_end=ch.page_end, rank=rank, score=float(score), retrieval_method=_METHOD[(mode, reranked)],
            bm25_score=c.lex_score, bm25_rank=c.lex_rank, dense_score=c.dense_score, dense_rank=c.dense_rank,
            rrf_score=c.rrf, rerank_score=c.rerank,
            alternates=[chunks[r].chunk_id for r in c.alternates], overlaps_with=[chunks[r].chunk_id for r in c.overlaps],
            metadata={"title": ch.title, "token_count": ch.token_count, "part": ch.part})
