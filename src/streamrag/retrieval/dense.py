"""Exact dense index: float32 L2-normalized matrix, inner product = cosine (ADR-001: no vector DB).

Phase 1 measured exact search at 0.01 ms (1k) .. 1.6 ms (100k x 384), so ANN is unjustified at this scale.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from streamrag.errors import EmbeddingDimensionMismatchError, IndexCorruptError
from streamrag.models.corpus import EmbeddingInfo
from streamrag.retrieval.topk import deterministic_topk


class DenseIndex:
    def __init__(self, matrix: np.ndarray, info: EmbeddingInfo) -> None:
        if matrix.ndim != 2 or matrix.shape[1] != info.dimension:
            raise EmbeddingDimensionMismatchError(
                f"embedding matrix shape {matrix.shape} does not match dimension {info.dimension}")
        self.matrix = np.ascontiguousarray(matrix, dtype=np.float32)
        self.matrix.flags.writeable = False          # read-only: shared safely across sessions
        self.info = info

    @property
    def n_docs(self) -> int:
        return self.matrix.shape[0]

    def scores(self, qvec: np.ndarray) -> np.ndarray:
        q = np.asarray(qvec, dtype=np.float32).reshape(-1)
        if q.shape[0] != self.info.dimension:
            raise EmbeddingDimensionMismatchError(
                f"query vector dim {q.shape[0]} != index dim {self.info.dimension} (model '{self.info.name}')")
        return self.matrix @ q

    def search(self, qvec: np.ndarray, top_k: int, mask: np.ndarray | None = None,
               min_similarity: float | None = None) -> list[tuple[int, float]]:
        return deterministic_topk(self.scores(qvec), top_k, eligible=mask, min_score=min_similarity)

    def save(self, directory: Path) -> list[Path]:
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / "embeddings.npy", self.matrix)
        (directory / "meta.json").write_text(self.info.canonical_json())
        return [directory / "embeddings.npy", directory / "meta.json"]

    @classmethod
    def load(cls, directory: Path) -> "DenseIndex":
        try:
            info = EmbeddingInfo.model_validate_json((directory / "meta.json").read_text())
            matrix = np.load(directory / "embeddings.npy", allow_pickle=False)
        except (OSError, ValueError) as exc:
            raise IndexCorruptError(f"dense index unreadable in {directory}: {exc}") from exc
        return cls(matrix, info)
