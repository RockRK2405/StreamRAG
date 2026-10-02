"""Deterministic, conservative evidence deduplication.

Applied to the fused candidate list (best-ranked first); the best-ranked member of a group is kept and the
others are recorded as ``alternates`` (still traceable). Rules, in order:

1. exact duplicate text (same text hash)                      -> merge
2. overlapping chunks of the same section (shared char span)  -> merge only if overlap >= overlap_merge_ratio
                                                                 of the shorter chunk, else annotate overlaps_with
3. near duplicates: cosine >= near_dup_cosine AND word-set Jaccard >= near_dup_jaccard AND identical negation
   and number tokens                                          -> merge

Rule 3's negation/number guard exists because "Ladders are permitted ..." and "Ladders are not permitted ..."
are lexically and semantically close but carry opposite meaning; they must never be merged.
"""

from __future__ import annotations

import re

import numpy as np

from streamrag.config.settings import DedupConfig
from streamrag.models.corpus import CorpusChunk
from streamrag.retrieval.fusion import Candidate

_WORD = re.compile(r"\w+")
_NEGATION = re.compile(r"\b(?:no|not|never|nor|none|cannot|without|except|unless|neither)\b|n't\b")
_NUMBER = re.compile(r"\d+(?:[.,]\d+)*")


def _protected(text: str) -> tuple[frozenset[str], frozenset[str]]:
    low = text.lower()
    return frozenset(_NEGATION.findall(low)), frozenset(_NUMBER.findall(low))


def _jaccard(a: set[str], b: set[str]) -> float:
    return len(a & b) / len(a | b) if (a or b) else 1.0


def deduplicate(cands: list[Candidate], chunks: list[CorpusChunk], vectors: np.ndarray | None,
                cfg: DedupConfig, near_dup_window: int = 50) -> tuple[list[Candidate], int]:
    if not cfg.enabled:
        return cands, 0
    kept: list[Candidate] = []
    seen_hash: dict[str, Candidate] = {}
    removed = 0
    words_cache: dict[int, set[str]] = {}

    def words(row: int) -> set[str]:
        if row not in words_cache:
            words_cache[row] = set(_WORD.findall(chunks[row].text.lower()))
        return words_cache[row]

    for c in cands:
        ch = chunks[c.row]
        # 1. exact duplicate text
        rep = seen_hash.get(ch.text_sha1)
        if rep is not None:
            rep.alternates.append(c.row)
            removed += 1
            continue
        merged = False
        for k in kept[:near_dup_window]:
            kc = chunks[k.row]
            # 2. overlapping spans within the same section
            if kc.document_id == ch.document_id and kc.section_id == ch.section_id:
                ov = min(kc.char_end, ch.char_end) - max(kc.char_start, ch.char_start)
                if ov > 0:
                    shorter = min(kc.char_end - kc.char_start, ch.char_end - ch.char_start)
                    if ov / max(shorter, 1) >= cfg.overlap_merge_ratio:
                        k.alternates.append(c.row)
                        merged = True
                        break
                    if k.row not in c.overlaps:
                        c.overlaps.append(k.row)
                        k.overlaps.append(c.row)
            # 3. near duplicates (guarded)
            if cfg.near_duplicates and vectors is not None:
                if float(vectors[k.row] @ vectors[c.row]) >= cfg.near_dup_cosine \
                        and _jaccard(words(k.row), words(c.row)) >= cfg.near_dup_jaccard \
                        and _protected(kc.text) == _protected(ch.text):
                    k.alternates.append(c.row)
                    merged = True
                    break
        if merged:
            removed += 1
            continue
        kept.append(c)
        seen_hash[ch.text_sha1] = c
    return kept, removed
