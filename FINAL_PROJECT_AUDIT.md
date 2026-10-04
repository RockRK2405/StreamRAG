# Final Project Audit (Phase 11, step 1)

Audit date: 2026-10-04. Repository state at audit: branch `phase10-evaluation`, HEAD `29e4e35`, clean working tree.
Every check below was run against the actual repository. This file records the state **before** Phase 11 changes.
The resolution of each finding is listed in §6.

## 1. Repository structure (actual)

| path | contents |
|---|---|
| `src/streamrag/` | the system: 27 subpackages, 188 modules, CLI `streamrag` (`cli.py`) |
| `configs/` | `default.yaml` plus lexicons (`claim_`, `controller_`, `intent_`, `retrieval_lexicon.yaml`) and `models.yaml` |
| `tests/` | 645 tests: 110 in root files and 535 across 40 area folders |
| `experiments/` | Phase 10 evaluation package: configs, dataset, runners, results, plots, failure cases |
| `eval/` | Phase 4–9 development suites and the Phase 9 held-out set |
| `research/phase1–9/` | per-phase benchmark scripts and results |
| `docs/` | `architecture/` (13), `decisions/` (ADR-001…019), `retrieval/`, `streaming/`, `multi_intent/`, `session/`, `answer/`, `runtime/`, `evaluation/`, `schemas/` |
| `models/`, `indexes/` | pinned ONNX models and built indexes (git-ignored; reproducible with `streamrag fetch-models` / `build-index`) |

Not present, although the brief or the specification assumes them:
* `app/`, `backend/`, `frontend/`: there is no HTTP server, no web UI and no `/health` endpoint.
* `Dockerfile`, `docker-compose.yml`: specification K2 / REQ-DEP-001 makes `docker compose up` the mandatory primary
  path. Milestone M8 (packaging) was never implemented.
* `.env.example` exists, but it is the Phase 3 template: it mentions only `STREAMRAG_CORPUS` and `HF_HOME`, plus an
  unused, commented-out `ANTHROPIC_API_KEY` placeholder. (A first listing in this audit missed the file because a shell
  glob failed; corrected here.)

## 2. Test run (before changes)

`.venv/bin/python -m pytest -q`: **645 passed, 0 failed, exit code 0, 54 s.**

| area | tests |
|---|---|
| retrieval / corpus | root files |
| streaming / controller / runtime | 35 + 13 + 8, plus scheduler, backpressure, cancellation, retries, timeouts, concurrency, failure injection, replay, events |
| multi-intent / fusion | 53 + 18 + 13 |
| session / context / delta / ledger | 29 + 15 + 13 + 4 |
| grounding (generation, claims, citations, validation, answer state) | 7 + 22 + 6 + 6 + 21 |
| adaptive retrieval | 27, plus routing, query analysis and rewrite, sufficiency, iterative, multi-hop, contradiction, cache, budget |
| evaluation | 24 |
| end-to-end / integration | 8 |

## 3. Static checks

| check | result |
|---|---|
| broken imports | none: all 188 modules import |
| syntax / compile | `compileall` clean (src, experiments/runners) |
| TODO / FIXME / HACK markers | none in `src/`, `configs/`, `tests/`, `experiments/runners/` |
| hard-coded secrets (API keys, tokens, passwords, cloud key patterns) | none found |
| hard-coded URLs | one: `generation.ollama_url: http://127.0.0.1:11434` (config, loopback). The LLM client refuses non-loopback URLs by design (ADR-017). |
| environment variables | only `STREAMRAG_CORPUS`; nothing else is configurable from the environment |
| `.gitignore` | ignores `.env`, `.venv/`, `/models/`, `/indexes/`, `/runs/` |
| unused subpackages | none: every subpackage is imported by the system, tests or research code |
| duplicate implementations | `answers/` (Phase 6 versioned session answers) and `answer_state/` (Phase 7 grounded answer state) look similar but are two layers. The Phase 7 state wraps the Phase 6 versioning, and both are used. Not removed. |
| dependencies | runtime: pydantic, numpy, scipy, pyyaml, snowballstemmer, tokenizers, onnxruntime, all used. Optional extras: `pdf`, `models`, `dev`, `bench-torch` (research only). `httpx` is in the venv but not declared or used. |
| package metadata | `pyproject.toml` still describes "Phase 3", version 0.3.0. The README describes "Current phase: Phase 9". Both are stale. |

## 4. Configuration consistency

* `configs/default.yaml` has `multi_intent`, `session`, `generation` and `adaptive_retrieval` all **disabled**. Every
  Phase 10 "full system" run enabled them through per-variant overrides. There is no single named configuration of
  the final pipeline that the CLI, a server, Docker and the benchmark could share.
* The LLM URL, model, ports, corpus path and index root are YAML-only. Containers need them from the environment.
* In a compose network, the LLM sidecar would be reached as `http://ollama:11434`, which the loopback rule refuses.
  This needs an explicit, documented allowlist.

## 5. Functional gaps (from Phase 10 measurements and code reading)

| # | gap | evidence | severity |
|---|---|---|---|
| G1 | **LLM timeout never falls back.** The extractive redo runs out of turn deadline, and the turn ends with no answer. | Phase 10 §25: 0 / 10 recovered (LLM *error* path: 10 / 10). `runtime/answers.py::_degrade` re-queues `mode="extractive"`. | high: demo fallback, reliability |
| G2 | **The Phase 4 retrieval decision labels "Which documents do…" as meta-conversation** and never retrieves. | Phase 10 §20 (T25, S05.1): `RETRIEVAL_DECISION SKIP reason=meta_conversation` | high: a silent wrong answer |
| G3 | **Early commitment while streaming.** Temporal and version resolution, and constraint retrieval, decided on a partial transcript survive into the validated answer. | Phase 10 §14: 7 turns lost vs batch, all partial-transcript decisions | high: quality |
| G4 | **No answerability check.** With off-topic evidence, grounded answers state adjacent facts. | Phase 10 §20: insufficiency handled in 1 / 5 | high: brief §16 |
| G5 | **Entity corrections keep the previous entity's claims** (S02.2), or end empty (S10.2). | Phase 10 §20 | medium |
| G6 | `session.full_restart` changes no runtime retrieval, so the runtime delta ablation is uninformative. | Phase 10 §12 | low (evaluation only) |
| G7 | No HTTP API, demo UI, health / readiness checks or container packaging. | §1 | high: submission |
| G8 | No single final-pipeline configuration; the environment can't configure it. | §4 | medium |
| G9 | `.env.example`, README and package metadata are stale (Phase 3 / Phase 9 wording). | §1, §3 | medium |
| G10 | The multi-hop analysis did not fire on new-domain questions (1 / 7). | Phase 10 §16 | medium (research) |

Missing tests: no HTTP / demo tests (nothing to test yet), and no end-to-end test of the uncertainty response. The
logging helpers (`telemetry/`) are covered only indirectly.

## 6. Phase 11 resolution plan (and outcome)

The protocol for system changes: the Phase 10 test split was looked at during Phase 10's error analysis, so it is no
longer blind. A **new held-out set (`streamrag_eval_v2`, new domain) is written and frozen before any system change**.
Fixes are developed on the dev split plus the Phase 10 test split, and evaluated once on v2.

Outcome of every audit item. The numbers are in `FINAL_BENCHMARK_RESULTS/README.md`.

| # | gap | resolution | measured outcome |
|---|---|---|---|
| G1 | LLM timeout never falls back | `runtime/answers.py`: the extractive redo gets its own deadline | robustness `llm_timeout`: recovery 10 / 10 (Phase 10: 0 / 10) |
| G2 | "Which documents do…" labelled meta-conversation | `controller/acts.py`: META only without corpus-anchored words outside the meta phrase | regression test; all 6 v2 "which form…" turns retrieve; demo scenario E ("Which documents do domestic applicants submit…") passes |
| G3 | early commitment while streaming | `session/engine.py` (drop evidence only superseded partial queries found), `multi_retrieval/coordinator.py` (re-validate a need when its budget is spent) | v1 (development): stale values 3 / 13 → 0 / 13. v2 (held-out): 0 / 13; streaming vs batch gap net 1 turn |
| G4 | no answerability check | LLM answerability flag implemented and measured; **off** by default | dev and v2: more correct abstentions (v2: 1 → 2 of 4) but lower correctness (v2: 0.718 → 0.662). Remains a limitation |
| G5 | entity corrections keep the previous entity's claims | the G3 rules, plus the need re-validation | v1 turns fixed. v2 follow-up / correction turns are still weak (0.61 streaming vs 0.72 batch): `LIMITATIONS.md` §2 |
| G6 | runtime delta ablation uninformative | measured instead with the extractive pipeline on v2 (`delta_full_restart`) | 1.49 vs 1.35 retrieval calls per turn (p = 0.016), same answers |
| G7 | no HTTP API, demo UI, health checks or container | `src/streamrag/server/` (HTTP + SSE, UI, `/health`, `/ready`), `Dockerfile`, `docker-compose.yml` | `tests/server/`; Docker healthy in 8 s; `demo-check` 10 / 10 with and without the LLM |
| G8 | no single final configuration; no environment configuration | `configs/profiles/final.yaml`, `load_final_config`, `STREAMRAG_*` variables, LLM host allowlist | `tests/test_config.py` |
| G9 | stale `.env.example`, README and package metadata | rewritten; version 1.0.0 | — |
| G10 | multi-hop analysis rarely fires on new domains | not changed (no held-out-safe fix) | v2 multi-hop correctness 0.571: `LIMITATIONS.md` §2 |

Additional fixes found during Phase 11 development, each with a regression test:
* claim decomposer garbling "A and B of C" lists;
* false conflicts between documents scoped to different groups;
* labelling of version conflicts.

See `docs/architecture/14_final_architecture.md` §14.4.
