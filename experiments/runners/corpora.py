"""Corpus keys used by the evaluation dataset -> fixture directories (all TEST FIXTURE ONLY)."""

CORPORA = {
    "transit": "tests/fixtures/corpus_eval_transit",      # Phase 10 test split (inspected in the Phase 10 error analysis)
    "utility": "tests/fixtures/corpus_eval_utility",      # Phase 11 held-out set v2 (frozen before Phase 11 changes)
    "adaptive": "tests/fixtures/corpus_adaptive",         # Phase 9 dev / held-out
    "fixture": "tests/fixtures/corpus",                   # Phase 3-7
    "grounding": "tests/fixtures/corpus_grounding",       # Phase 7
    "conflict": "tests/fixtures/corpus_conflict",         # Phase 7
    "injection": "tests/fixtures/corpus_injection",       # Phase 7
}
