"""BM25 lexical retrieval over a precomputed sparse weight matrix (ADR-001).

w(t, d) = idf(t) * tf * (k1 + 1) / (tf + k1 * (1 - b + b * |d| / avgdl)),  idf = ln(1 + (N - df + .5)/(df + .5))
score(q, d) = sum over unique query terms t of w(t, d). Only positive scores are returned (no-match => []).
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from streamrag.retrieval.text import Analyzer
from streamrag.retrieval.topk import deterministic_topk


class BM25Index:
    def __init__(self, vocab: dict[str, int], weights: sp.csc_matrix, doc_len: np.ndarray, k1: float, b: float) -> None:
        self.vocab = vocab
        self.weights = weights            # shape (n_docs, n_terms), float32, CSC (fast column slicing)
        self.doc_len = doc_len
        self.k1, self.b = k1, b

    @property
    def n_docs(self) -> int:
        return self.weights.shape[0]

    @property
    def idf(self) -> np.ndarray:
        """Per-term IDF (same formula as indexing), derived from the CSC column counts (= document frequency).
        Phase 4 uses it for the controller's corpus anchor-strength signal (a dictionary lookup, not a search)."""
        cached = getattr(self, "_idf", None)
        if cached is None:
            df = np.diff(self.weights.indptr).astype(np.float64)
            cached = np.log1p((self.n_docs - df + 0.5) / (df + 0.5))
            self._idf = cached
        return cached

    def rows_with(self, term: str) -> frozenset[int]:
        """Chunk rows containing an analyzed term (CSC column slice; Phase 6 corpus co-occurrence)."""
        j = self.vocab.get(term)
        if j is None:
            return frozenset()
        return frozenset(int(r) for r in self.weights.indices[self.weights.indptr[j]:self.weights.indptr[j + 1]])

    def cooccurrence(self, all_of: list[str], any_of: list[str]) -> int:
        """Number of chunks containing every term of ``all_of`` and at least one term of ``any_of``."""
        if not all_of or not any_of:
            return 0
        rows = None
        for t in all_of:
            r = self.rows_with(t)
            rows = r if rows is None else rows & r
            if not rows:
                return 0
        other = frozenset().union(*(self.rows_with(t) for t in any_of))
        return len(rows & other)

    def term_idf(self, term: str) -> float | None:
        """IDF of an *analyzed* term, or None if the term is not in the corpus vocabulary."""
        j = self.vocab.get(term)
        return None if j is None else float(self.idf[j])

    @classmethod
    def build(cls, texts: list[str], analyzer: Analyzer, k1: float, b: float) -> "BM25Index":
        docs = [Counter(analyzer.tokens(t)) for t in texts]
        vocab_terms = sorted({t for d in docs for t in d})
        vocab = {t: i for i, t in enumerate(vocab_terms)}
        rows, cols, tfs = [], [], []
        for r, d in enumerate(docs):
            for t, c in sorted(d.items()):
                rows.append(r)
                cols.append(vocab[t])
                tfs.append(c)
        n = len(texts)
        doc_len = np.array([sum(d.values()) for d in docs], dtype=np.float32)
        avgdl = float(doc_len.mean()) if n else 0.0
        rows_a, cols_a = np.array(rows, dtype=np.int64), np.array(cols, dtype=np.int64)
        tf = np.array(tfs, dtype=np.float64)
        df = np.bincount(cols_a, minlength=len(vocab)).astype(np.float64)
        idf = np.log1p((n - df + 0.5) / (df + 0.5))
        norm = k1 * (1.0 - b + b * doc_len[rows_a] / (avgdl or 1.0))
        w = (idf[cols_a] * tf * (k1 + 1.0) / (tf + norm)).astype(np.float32)
        weights = sp.csc_matrix((w, (rows_a, cols_a)), shape=(n, len(vocab)), dtype=np.float32)
        weights.sort_indices()
        return cls(vocab, weights, doc_len, k1, b)

    def query_terms(self, query: str, analyzer: Analyzer) -> list[str]:
        return sorted({t for t in analyzer.tokens(query) if t in self.vocab})

    def scores(self, query: str, analyzer: Analyzer) -> np.ndarray:
        cols = [self.vocab[t] for t in self.query_terms(query, analyzer)]
        if not cols:
            return np.zeros(self.n_docs, dtype=np.float32)
        return np.asarray(self.weights[:, cols].sum(axis=1)).ravel()

    def search(self, query: str, analyzer: Analyzer, top_k: int, mask: np.ndarray | None = None) -> list[tuple[int, float]]:
        s = self.scores(query, analyzer)
        eligible = s > 0 if mask is None else (s > 0) & mask
        return deterministic_topk(s, top_k, eligible=eligible)

    # ---------------------------------------------------------------- persistence
    def save(self, directory: Path) -> list[Path]:
        directory.mkdir(parents=True, exist_ok=True)
        sp.save_npz(directory / "weights.npz", self.weights, compressed=False)
        np.save(directory / "doc_len.npy", self.doc_len)
        (directory / "vocab.json").write_text(json.dumps(sorted(self.vocab, key=self.vocab.get), ensure_ascii=False))
        (directory / "meta.json").write_text(json.dumps({"k1": self.k1, "b": self.b, "n_docs": self.n_docs,
                                                          "n_terms": len(self.vocab)}, sort_keys=True))
        return [directory / f for f in ("weights.npz", "doc_len.npy", "vocab.json", "meta.json")]

    @classmethod
    def load(cls, directory: Path) -> "BM25Index":
        meta = json.loads((directory / "meta.json").read_text())
        terms = json.loads((directory / "vocab.json").read_text())
        weights = sp.load_npz(directory / "weights.npz").tocsc()
        doc_len = np.load(directory / "doc_len.npy")
        if weights.shape != (meta["n_docs"], meta["n_terms"]) or len(terms) != meta["n_terms"]:
            from streamrag.errors import IndexCorruptError
            raise IndexCorruptError(f"BM25 artifacts inconsistent in {directory}")
        return cls({t: i for i, t in enumerate(terms)}, weights, doc_len, meta["k1"], meta["b"])
