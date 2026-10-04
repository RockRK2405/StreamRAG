# StreamRAG experiments (Phase 10)

Everything needed to reproduce the numbers in `PHASE_10_EVALUATION_REPORT.md`. Every number in that report comes
from a file under `experiments/results/`, written by the runners below. Values that were not measured are recorded
as `NOT MEASURED` or `NOT RUN`.

> All corpora are TEST FIXTURES written by the implementer, and so are the labels. The results show how the systems
> behave on these fixtures and are **NOT REPORTABLE** as product or benchmark quality.

## Layout

| path | contents |
|---|---|
| `configs/systems.yaml` | every system variant: family (free / pipeline / runtime), generation (llm / extractive), config overrides |
| `configs/experiments.yaml` | experiments 1-10 and the two ablation blocks: systems, splits, paired comparisons, metrics |
| `configs/regression_rules.yaml`, `configs/quality_gates.yaml` | regression tolerances and project-specific gates, each with its basis |
| `datasets/build_dataset.py` | builds `datasets/streamrag_eval_v1/{test,dev}.jsonl` + `manifest.json` (hashes) |
| `datasets/human_eval_sheet.csv` | blinded sheet for the human evaluation protocol (`docs/evaluation/human_evaluation.md`); not filled - NOT RUN |
| `runners/run_experiments.py` | runs the experiments (`ExperimentRunner`, `src/streamrag/evaluation/runner.py`) |
| `runners/verifier_validation.py` | accuracy of the claim-verification instrument on perturbed claims (labels by construction) |
| `runners/robustness.py` | full system under injected faults and hostile input (Phase 8 FaultInjector; SYNTHETIC faults) |
| `runners/reproducibility.py` | second run of selected variants; compares answers, evidence, labels and latency |
| `runners/environment.py` | `results/environment.json`: packages, model files + hashes, LLM digest, hardware, OS, config hash, git |
| `runners/analyze.py` | `results/tables.md`, `results/final_comparison.json`, `results/ablation_table.json`, `results/dashboard.html`, `plots/*.svg`, human-eval sheet |
| `runners/make_regression_baseline.py`, `runners/check_regression.py` | regression reference and check (`docs/evaluation/regression.md`) |
| `results/<experiment>/` | `config_record.json`, `log.jsonl`, `results.json` (COMPLETED / FAILED / NOT RUN), `samples.jsonl`, `samples.csv` |
| `results/runs/` | raw per-turn records of each variant (`<variant>__<split>.jsonl`), the scored rows (`.scored.jsonl`), run metadata (`.meta.json`) and runtime event traces (`traces/<variant>__<sample>.jsonl`) |
| `failure_cases/<experiment>.jsonl` | every turn with an error category: query, expected, answer, evidence, claims, citations, trace path |
| `metrics/README.md` | where each metric is defined and computed |
| `plots/` | SVG charts (light and dark mode); only measured values are drawn |

## Reproduce

```bash
.venv/bin/python experiments/datasets/build_dataset.py
```
```bash
ollama serve
```
```bash
.venv/bin/python experiments/runners/run_experiments.py --index-root /tmp/idx10
```
```bash
.venv/bin/python experiments/runners/verifier_validation.py --index-root /tmp/idx10
```
```bash
.venv/bin/python experiments/runners/robustness.py --index-root /tmp/idx10
```
```bash
.venv/bin/python experiments/runners/reproducibility.py --index-root /tmp/idx10
```
```bash
.venv/bin/python experiments/runners/environment.py
```
```bash
.venv/bin/python experiments/runners/analyze.py && .venv/bin/python experiments/runners/make_regression_baseline.py
```

* **LLM:** local `qwen3:4b` through Ollama, temperature 0, seed 7, `num_ctx` 8192, at most 1024 output tokens. Without a
  reachable server, every experiment that needs the LLM is written as `NOT RUN` and no substitute numbers are used.
  `--no-llm` forces this.
* **Seeds:** LLM seed 7, bootstrap / statistics seed 20261004, and the dataset is static. The retrieval, verification
  and extractive paths are deterministic. Latency is wall clock and varies between runs; see
  `results/REPRODUCIBILITY/`.
* **Runs are shared and reused.** Each (variant, split) runs once, and every experiment that uses that variant reads the
  same stored run. `--rerun` re-executes the runs. Scoring is derived from the stored raw run, so a change to a metric
  can be re-scored without re-running the system.
* **Timing interference.** Run the runners one after another, never in parallel, because they share the CPU and the
  LLM server.
* **Held-out test split.** System behaviour was not changed using test outputs. Only evaluation-framework bugs were
  fixed, and each fix is disclosed in the report (section 29).
