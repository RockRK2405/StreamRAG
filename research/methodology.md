# Methodology

## Development and evaluation protocol

1. **Phases 1–9 built the system** on development fixtures: `eval/dev_*` and the `tests/fixtures/corpus*` corpora.
2. **Phase 10 evaluated it on a new held-out set,** v1: Northvale transit, 87 turns, all 14 query types. It used
   baselines, ten experiments, ablations, paired statistics, a verifier validation, robustness tests and a
   reproducibility re-run (`../PHASE_10_EVALUATION_REPORT.md`).
3. **Phase 10's error analysis inspected v1,** so v1 became development data.
4. **Phase 11 wrote and froze a new held-out set,** v2: Lakeside utility, 81 turns, all 14 query types. The hash and
   time are recorded in `experiments/datasets/streamrag_eval_v2/FROZEN.sha256`. This happened **before** any Phase 11
   system change. Fixes were developed on the dev split and v1 only.
5. **The final system and every baseline are run once on v2** (`FINAL_BENCHMARK_RESULTS/heldout_v2`). The v1 re-run is
   reported separately as a regression check and is labelled optimistic.

## Measurement

* **Metrics.** Definitions are in `../docs/evaluation/metrics.md`; the code is in `src/streamrag/evaluation/metrics/`.
  * Label metrics are deterministic: answer correctness = all expected key strings stated and no forbidden (stale)
    value.
  * Verifier-judged metrics use the system's own NLI verifier, so they are biased toward verified systems. Phase 10
    §23 quantifies this.
* **Latency.** Wall-clock. TTFE / TTFA / TTVA are measured from the first transcript chunk, and also after the
  utterance ends. Percentiles need a minimum n.
* **Statistics.** Paired by turn: exact McNemar for binary metrics, Wilcoxon signed-rank for continuous ones, a
  bootstrap 95% CI of the mean difference, and the effect sizes d_z and rank-biserial. No test below 20 pairs or 6
  non-zero differences. No multiple-comparison correction, so p-values are descriptive.
* **Reproducibility.** Temperature 0, seed 7, a fixed statistics seed, static datasets. Phase 10 measured identical
  labels on re-run and about 3% latency noise.

## Honesty rules applied

* No result is reported without a stored run.
* Anything not measured is marked NOT MEASURED.
* Negative results are reported, including the final system losing to hybrid RAG on v1 in Phase 10.
* Fixture corpora are labelled NOT REPORTABLE.
* Framework bugs are disclosed: Phase 10 §29, `FINAL_BENCHMARK_RESULTS/README.md`.
