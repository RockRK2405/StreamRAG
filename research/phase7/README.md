# Phase 7 research: grounded answers (dev suite, fixture domain, NOT REPORTABLE)

| Path | What it is |
|---|---|
| `build_grounded_suite.py` | writes `eval/dev_grounded/G01..G28.json` (categories A–J, gold facts, forbidden patterns, temptation tags) |
| `run_grounded_benchmarks.py` | benchmark + ablation arms A–F, X; hallucination tags; revision vs full restart; latency; verifier perturbation eval; blind labelling sheet |
| `e2e_streaming.py` | brief §64: ten scenarios through the streaming pipeline with the local LLM, exact replay check |
| `label_agreement.py` | joins the blind hand labels (`labels/claim_labels.jsonl`) with the verifier's verdicts |
| `report_tables.py` | regenerates `results/report_tables.md` (the report's tables) from the result files |
| `results_first_run/` | first full run, kept unchanged; exposed the defects listed in the report §20 (items 1–5) |
| `results_postfix_run/` | intermediate run after those fixes; the hand labels were made blind on its labelling sheet |
| `results/` | final run of record (after §20 items 6–7); `e2e/` holds the end-to-end traces and their generated README |

Requires a running `ollama serve` with `qwen3:4b` pulled and `streamrag fetch-models nli-deberta-v3-xsmall`.
The LLM runs at temperature 0 with seed 7. Arms A–E produced identical output in the post-fix and final runs.
