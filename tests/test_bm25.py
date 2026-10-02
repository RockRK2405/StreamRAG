import numpy as np

from streamrag.retrieval.bm25 import BM25Index
from streamrag.retrieval.text import Analyzer, number_words_to_digits

A = Analyzer()
DOCS = ["The lamp is cleaned every evening before sunset.",
        "Wicks are trimmed to a height of 4 millimetres.",
        "The fog signal sounds every 30 seconds in a storm.",
        "Keepers polish the lens with a dry cloth.",
        "The lamp register records each cleaning of the lamp."]
IDX = BM25Index.build(DOCS, A, 1.5, 0.75)


def top(q, k=5):
    return [r for r, _ in IDX.search(q, A, k)]


def test_exact_keyword_match():
    assert top("fog signal")[0] == 2


def test_partial_match():
    assert top("height of the wick flame")[0] == 1          # only some query terms present


def test_terminology_variation_via_stemming():
    assert top("trimming wicks")[0] == 1                      # trimming ~ trimmed, wicks ~ wick
    assert top("polishing lenses")[0] == 3


def test_number_words_normalized_on_both_sides():
    assert number_words_to_digits(["thirty", "five", "sec"]) == ["35", "sec"]
    assert top("every thirty seconds")[0] == 2


def test_empty_and_no_result_queries():
    assert IDX.search("", A, 5) == []
    assert IDX.search("the and of", A, 5) == []               # stopwords only
    assert IDX.search("zebra quantum", A, 5) == []            # no vocabulary overlap -> no evidence


def test_deterministic_tie_break_by_row():
    idx = BM25Index.build(["alpha beta", "alpha beta", "alpha beta"], A, 1.5, 0.75)
    assert [r for r, _ in idx.search("alpha", A, 3)] == [0, 1, 2]


def test_scores_positive_and_sorted():
    hits = IDX.search("lamp cleaning", A, 5)
    scores = [s for _, s in hits]
    assert scores == sorted(scores, reverse=True) and all(s > 0 for s in scores)
    assert hits[0][0] == 4                                    # "lamp" twice + "cleaning"


def test_mask_filters_rows():
    mask = np.array([False, False, False, False, True])
    assert [r for r, _ in IDX.search("lamp", A, 5, mask)] == [4]


def test_save_load_roundtrip(tmp_path):
    IDX.save(tmp_path / "bm25")
    loaded = BM25Index.load(tmp_path / "bm25")
    assert loaded.search("lamp register", A, 5) == IDX.search("lamp register", A, 5)
