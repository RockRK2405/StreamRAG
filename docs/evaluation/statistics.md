# Statistical analysis (Phase 10)

Code: `src/streamrag/evaluation/stats.py`; tests: `tests/evaluation/test_stats_errors_regression.py`.

* **Descriptives** per system and metric: n, mean, median, sample SD, 95 % percentile-bootstrap CI of the mean
  (10,000 resamples, seed 20261004). Latency: p50 / p90 / p95 / p99 when the sample size allows (latency.md rules).
* **Paired comparisons** (same samples, both values present): mean and median difference, 95 % bootstrap CI of the
  mean difference; binary metrics → exact McNemar test on the discordant pairs; continuous metrics → two-sided Wilcoxon
  signed-rank (zero differences dropped). Effect sizes: Cohen's d_z and matched-pairs rank-biserial r.
* **Minimum sizes:** < 20 pairs or < 6 non-zero differences → no test (reported as such).
* **Assumptions and reading.** Samples are implementer-written fixture items, not a random sample of user questions:
  p-values describe this item set only. Turns of one conversation are not independent. No multiple-comparison
  correction is applied, so p-values are read together with effect sizes and CIs, never alone; "significant" is not
  used as a verdict.
* **Randomness control:** LLM temperature 0, seed 7 (Ollama), model digest recorded; retrieval and verification are
  deterministic; bootstrap seeded. Latency is wall clock on one machine and varies run to run (one run per variant
  unless stated).
