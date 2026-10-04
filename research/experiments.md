# Experiments

| set | what | where |
|---|---|---|
| Phase 10, v1 (now development data) | baselines A–D vs adaptive and full system; experiments 1–10; runtime and answer-stage ablations; verifier validation; robustness; reproducibility | `../PHASE_10_EVALUATION_REPORT.md`, `../experiments/results/` |
| Phase 11 development iterations | each fix measured before and after on the dev split and v1 (answerability check, streaming fixes) | `../FINAL_BENCHMARK_RESULTS/dev_iterations/` |
| Phase 11 held-out, v2 | the final system and every baseline, run once; batch vs streaming; answerability ablation; extractive retrieval variants; cancellation | `../FINAL_BENCHMARK_RESULTS/heldout_v2/` |
| Phase 11 regression, v1 | the final system vs the stored Phase 10 runs | `../FINAL_BENCHMARK_RESULTS/regression_v1/` |
| Phase 11 robustness | synthetic faults on the final system | `../FINAL_BENCHMARK_RESULTS/robustness/` |
| Phase 11 demo check | ten demo scenarios with declared expectations, with and without the LLM | `../FINAL_BENCHMARK_RESULTS/demo_check*.json` |
| Phase 11 resources, environment | memory, CPU, disk, LLM footprint; versions and hashes | `../FINAL_BENCHMARK_RESULTS/resources/`, `environment.json` |

**Configurations.** `../experiments/configs/systems.yaml` (variants), `experiments.yaml` (Phase 10),
`phase11_dev.yaml`, and `final_benchmark.yaml`.

**Runner.**

```bash
experiments/runners/run_experiments.py --dataset <v1|v2> --results-dir <dir> --experiments-file <yaml>
```

Every run writes its config record, log, raw runs, scored rows, traces and failure cases.
