# Regression evaluation and quality gates (Phase 10)

* `experiments/runners/check_regression.py` compares a new experiment's results with the stored reference
  (`experiments/results/regression_baseline.json`, snapshot of the Phase 10 runs) using direction-aware rules with
  absolute tolerances (`experiments/configs/regression_rules.yaml`) and evaluates the quality gates
  (`experiments/configs/quality_gates.yaml`). It reports ok / improvement / regression / missing and PASS / FAIL /
  NOT MEASURED; it never blocks by itself — a regression is a signal to read with its context (dataset, model,
  hardware).
* Gate thresholds are **project-specific and provisional**: derived from the Phase 10 fixture measurements (each gate
  states its basis) and must be re-derived on the official corpus. They are not universal quality targets.
* `make_regression_baseline.py` refreshes the reference after an accepted change.
