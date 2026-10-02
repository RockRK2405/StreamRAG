"""Candidate union + Reciprocal Rank Fusion (ADR-001/ADR-009).

RRF(d) = sum over lists L containing d of 1 / (k + rank_L(d)), ranks 1-based. Rank-based, so BM25 and cosine
scales never need calibration — the reason it is preferred over weighted score fusion on a held-out corpus.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Candidate:
    row: int
    lex_rank: int | None = None
    lex_score: float | None = None
    dense_rank: int | None = None
    dense_score: float | None = None
    rrf: float | None = None
    rerank: float | None = None
    alternates: list[int] = field(default_factory=list)
    overlaps: list[int] = field(default_factory=list)

    def best_rank(self) -> int:
        ranks = [r for r in (self.lex_rank, self.dense_rank) if r is not None]
        return min(ranks) if ranks else 10**9


def single_list(hits: list[tuple[int, float]], source: str) -> list[Candidate]:
    out = []
    for rank, (row, score) in enumerate(hits, start=1):
        c = Candidate(row)
        if source == "lexical":
            c.lex_rank, c.lex_score = rank, score
        else:
            c.dense_rank, c.dense_score = rank, score
        out.append(c)
    return out


def rrf_fuse(lexical: list[tuple[int, float]], dense: list[tuple[int, float]], k: int) -> list[Candidate]:
    by_row: dict[int, Candidate] = {}
    for rank, (row, score) in enumerate(lexical, start=1):
        c = by_row.setdefault(row, Candidate(row))
        c.lex_rank, c.lex_score = rank, score
    for rank, (row, score) in enumerate(dense, start=1):
        c = by_row.setdefault(row, Candidate(row))
        c.dense_rank, c.dense_score = rank, score
    for c in by_row.values():
        c.rrf = sum(1.0 / (k + r) for r in (c.lex_rank, c.dense_rank) if r is not None)
    # deterministic: RRF desc, then best single-list rank, then corpus order
    return sorted(by_row.values(), key=lambda c: (-round(c.rrf or 0.0, 12), c.best_rank(), c.row))
