import hashlib

import numpy as np
import pytest

from streamrag.config.settings import DedupConfig
from streamrag.models.corpus import CorpusChunk
from streamrag.retrieval.dedup import deduplicate
from streamrag.retrieval.fusion import Candidate, rrf_fuse, single_list


def test_rrf_exact_values_and_union():
    fused = rrf_fuse([(10, 5.0), (11, 4.0)], [(11, 0.9), (12, 0.8)], k=60)
    by = {c.row: c for c in fused}
    assert set(by) == {10, 11, 12}
    assert by[11].rrf == pytest.approx(1 / 62 + 1 / 61)
    assert by[10].rrf == pytest.approx(1 / 61) and by[12].rrf == pytest.approx(1 / 62)
    assert [c.row for c in fused] == [11, 10, 12]


def test_rrf_tie_break_is_deterministic():
    fused = rrf_fuse([(5, 1.0)], [(3, 1.0)], k=60)        # equal RRF, equal best rank -> lower row first
    assert [c.row for c in fused] == [3, 5]


def test_rrf_k_changes_weighting():
    a = rrf_fuse([(1, 1.0), (2, 1.0)], [(2, 1.0)], k=1)
    assert a[0].row == 2


def test_single_list_keeps_order_and_scores():
    c = single_list([(4, 2.0), (1, 1.0)], "lexical")
    assert [(x.row, x.lex_rank, x.lex_score) for x in c] == [(4, 1, 2.0), (1, 2, 1.0)]


def chunk(i, text, doc="D", sec="1", start=0):
    return CorpusChunk(chunk_id=f"{doc}§{sec}#{i}", document_id=doc, section_id=sec, part=i + 1, citation=f"{doc} §{sec}",
                       title="T", section_title="S", text=text, index_text=text, source_path="d.txt", char_start=start,
                       char_end=start + len(text), position=i, position_in_section=i, token_count=len(text.split()),
                       text_sha1=hashlib.sha1(text.encode()).hexdigest())


CFG = DedupConfig()


def run(chunks, vectors=None, cfg=CFG):
    return deduplicate([Candidate(i) for i in range(len(chunks))], chunks, vectors, cfg)


def test_exact_duplicates_merged_with_alternates():
    kept, removed = run([chunk(0, "Same text here.", sec="1"), chunk(1, "Same text here.", sec="2")])
    assert removed == 1 and kept[0].alternates == [1]


def test_negation_difference_never_merged_even_with_identical_vectors():
    a = "Ladders are permitted in the orchard overnight when stored."
    b = "Ladders are not permitted in the orchard overnight when stored."
    v = np.ones((2, 4), dtype=np.float32) / 2.0
    kept, removed = run([chunk(0, a, sec="1"), chunk(1, b, sec="2")], v,
                        DedupConfig(near_dup_cosine=0.5, near_dup_jaccard=0.5))
    assert removed == 0 and len(kept) == 2


def test_number_difference_never_merged():
    v = np.ones((2, 4), dtype=np.float32) / 2.0
    kept, removed = run([chunk(0, "Pickers fill at most 40 crates.", sec="1"),
                         chunk(1, "Pickers fill at most 50 crates.", sec="2")], v,
                        DedupConfig(near_dup_cosine=0.5, near_dup_jaccard=0.5))
    assert removed == 0


def test_near_duplicate_merged_when_guards_pass():
    v = np.ones((2, 4), dtype=np.float32) / 2.0
    a = "The keeper polishes the lens with a dry cloth every single day."
    b = "The keeper polishes the lens with a dry cloth every single day!"     # same word set
    kept, removed = run([chunk(0, a, sec="1"), chunk(1, b, sec="2")], v)
    assert removed == 1 and kept[0].alternates == [1]


def test_overlap_merge_threshold():
    big = chunk(0, "x" * 100, start=0)
    mostly = chunk(1, "y" * 100, start=10)        # overlaps 90% of the shorter -> merged
    kept, removed = run([big, mostly])
    assert removed == 1
    small_overlap = chunk(1, "z" * 100, start=70)  # overlaps 30% -> kept, annotated
    kept, removed = run([big, small_overlap])
    assert removed == 0 and kept[0].overlaps == [1] and kept[1].overlaps == [0]


def test_disabled_dedup_passthrough():
    kept, removed = run([chunk(0, "a"), chunk(1, "a")], cfg=DedupConfig(enabled=False))
    assert removed == 0 and len(kept) == 2
