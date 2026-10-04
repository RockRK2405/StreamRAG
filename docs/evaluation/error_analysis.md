# Error analysis (Phase 10)

`src/streamrag/evaluation/errors.py` assigns each evaluated turn zero or more categories by rules over its measured
metrics and labels: ORCHESTRATION_FAILURE, LATENCY_FAILURE, QUERY_ANALYSIS_FAILURE, ENTITY_FAILURE, MEMORY_FAILURE,
RETRIEVAL_FAILURE, EVIDENCE_FAILURE, CLAIM_FAILURE, GENERATION_FAILURE, CITATION_FAILURE (definitions in the module
docstring). Every categorised turn is written to `experiments/failure_cases/<experiment>.jsonl` with query, expected,
actual answer, retrieved evidence, judged claims, citations, trace reference and category. The categories are
diagnostic heuristics (a turn can have several; QUERY_ANALYSIS_FAILURE needs an adaptive trace), checked on synthetic
rows in the tests. Qualitative analysis in the report uses successes, failures and edge cases from these files.
