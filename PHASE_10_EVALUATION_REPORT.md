# Samsung PRISM Theme 4
# Phase 10 — Experimental Evaluation

> **Status of every number in this report: TEST FIXTURE ONLY — NOT REPORTABLE.** The official Theme 4 corpus is
> still unavailable, and every corpus and label here was written by the implementer. All numbers were measured by
> the runners in `experiments/runners/` and are stored under `experiments/results/`. Each table names its source
> file. Anything that was not measured is marked **NOT MEASURED** or **NOT RUN**. One run per variant; the LLM is a
> local 4B model on a laptop.

## 1. Evaluation Objective

This phase measures whether the StreamRAG architecture built in Phases 3–9 does better than simpler designs, and
what each improvement costs. It covers:

* the streaming runtime (Phase 8);
* multi-intent session RAG with delta retrieval and evidence reuse (Phases 5–6);
* claim-verified grounded generation (Phase 7);
* adaptive retrieval (Phase 9).

The comparison baselines are naive vector RAG, hybrid RAG, hybrid RAG with reranking, and the streaming runtime with
fixed retrieval. They are compared on:

* retrieval;
* evidence;
* answer correctness;
* grounding;
* citations;
* latency;
* streaming behaviour;
* retrieval and compute efficiency;
* cancellation;
* reuse;
* robustness.

Ablations isolate each component. The aim is evidence, including evidence against the design, not a showcase for the
proposed system.

## 2. Research Questions

| id | question (brief §65) | where it is answered |
|---|---|---|
| RQ1 | Does adaptive retrieval improve the quality–efficiency tradeoff? | Exp. 1, 2, 8; §22 |
| RQ2 | Does claim-driven retrieval improve grounding? | Exp. 3; ablation |
| RQ3 | Does delta retrieval reduce redundant work? | Exp. 5; ablation |
| RQ4 | Does cancellation reduce wasted computation? | Exp. 6; ablation |
| RQ5 | Does streaming reduce perceived response latency? | Exp. 7; §21 |
| RQ6 | Does session memory reduce retrieval repetition? | Exp. 4; ablation (memoryless) |
| RQ7 | Does contradiction-aware retrieval improve reliability? | Exp. 10 |
| RQ8 | What is the latency cost of stronger verification? | ablation (answer stage); §21 |

## 3. Hypotheses

These were stated before the test split was run. Each is accepted or rejected in §30 on the measured paired
differences.

* **H1 (RQ1).** Adaptive retrieval reaches the answer quality of the fixed hybrid pipeline with fewer chunks per need
  than a large fixed k. It costs extra retrieval calls on multi-hop and conflicting questions.
* **H2 (RQ2).** Claim-driven sufficiency raises claim support and claim coverage over query-driven sufficiency.
* **H3 (RQ3).** Delta retrieval reduces retrieval calls on refinements and corrections and leaves quality unchanged.
* **H4 (RQ4).** Cooperative cancellation reduces worker time spent on superseded work when the user corrects a
  question mid-flight.
* **H5 (RQ5).** Streaming produces first evidence and first answer content earlier than batch processing. Measured
  from the end of the utterance, the validated answer also arrives earlier.
* **H6 (RQ6).** Session memory reduces repeated retrieval in multi-turn conversations, measured by cache hits and
  reused evidence.
* **H7 (RQ7).** Contradiction- and temporal-aware retrieval reduces stale values in answers and makes reporting a
  conflict more likely.
* **H8 (RQ8).** Claim verification and citation validation add measurable latency after the LLM call, but less than
  the LLM generation time.

## 4. Dataset

`experiments/datasets/streamrag_eval_v1`. Full documentation is in `docs/evaluation/dataset.md`, and the hashes are in
`manifest.json`.

| | test (held out) | dev |
|---|---|---|
| scored turns | **87** | 129 |
| sessions | 70: 50 single questions, 15 conversations of 2–3 turns, 5 streamed ASR corrections | 112 |
| corpus | `tests/fixtures/corpus_eval_transit`: 15 documents, 43 chunks, new in Phase 10 | Phase 3–9 fixtures: `corpus_adaptive`, `corpus`, `corpus_grounding`, `corpus_conflict`, `corpus_injection` |
| sha256 (first 16 hex) | `18e51582e184efd2` | `17638ae166a734d5` |
| query types | all 14: SIMPLE 15, EXACT_TERM 6, SEMANTIC 5, MULTI_INTENT 5, MULTI_CONSTRAINT 8, TEMPORAL 6, CONTEXTUAL_FOLLOWUP 5, ENTITY_CORRECTION 5, MULTI_HOP 7, AMBIGUOUS 5, CONTRADICTORY 5, INSUFFICIENT_EVIDENCE 5, REPEATED_QUERY 5, STREAMING_CORRECTION 5 | 13 (no STREAMING_CORRECTION) |
| difficulty | EASY 27, MEDIUM 42, HARD 18, VERY_HARD 0 | EASY 77, MEDIUM 46, HARD 6 |

* **Fields.** Each record has `query`, `stream` (ASR chunks with revisions), `conversation_context`,
  `expected_intents`, `required_entities`, `required_constraints`, `ground_truth_evidence` (sections),
  `expected_claims` (statement, citation and the key strings an answer must contain), `expected_answer`,
  `expected_state`, `conflict_values`, `forbidden` (stale or unsupported values an answer must not assert),
  `source_documents`, `difficulty` and `query_type`.
* **Difficulty is computed, not assigned.** It is a points rule over needs, multi-hop, constraints, temporal,
  context, conflict, insufficiency, streaming revision and lexical gap. EASY = 0, MEDIUM = 1, HARD = 2–3,
  VERY_HARD ≥ 4. No test item reaches VERY_HARD, so that level is reported as "no samples".
* **Leakage controls.** These are checked by `tests/evaluation/test_dataset_and_leakage.py`:
  * the test corpus and questions were written before any system ran on them;
  * test strings do not occur in `src/` or `configs/`;
  * no system module imports the evaluation package;
  * test documents share no ids or near-duplicate text with development documents.
* **Held-out discipline.** No system behaviour was changed after looking at test outputs (see §29).
* **The dev split is optimistic.** It is the material Phases 7–9 were built on.

## 5. Experimental Setup

| item | value |
|---|---|
| hardware | Apple M5 Pro (Mac17,9), 24 GiB unified memory, no discrete GPU |
| OS | macOS 26.5.1 (25F80) |
| runtime | CPython 3.12.0; packages, model files and hashes in `experiments/results/environment.json` |
| LLM | `qwen3:4b` via Ollama 0.35.1, local, temperature 0, seed 7, num_ctx 8192, max 1024 output tokens |
| embedder | `bge-small-en-v1.5` (ONNX, CPU) |
| lexical | BM25 (Phase 3), RRF fusion for hybrid |
| reranker | `ms-marco-MiniLM-L6-v2` cross-encoder (only where stated) |
| verifier / instrument | `nli-deberta-v3-xsmall` + rules (Phase 7 `ClaimVerifier`) |
| database | none: in-process BM25 (scipy sparse) and exact dense search (numpy) over the index files |
| code | branch `phase9-adaptive-retrieval`, HEAD `c45497d` plus the uncommitted Phase 10 tree (every `config_record.json` stores the commit and the dirty flag) |
| configuration | `configs/default.yaml` plus the per-variant overrides in `experiments/configs/systems.yaml`; adaptive reference date 2026-10-03 |

* **Streaming replay.** Every runtime variant receives each turn as three-word transcript chunks, one every 250 ms,
  followed by an utterance end. STREAMING_CORRECTION items replay their stored chunk revisions. The next turn of a
  conversation starts once the runtime is idle, plus 300 ms. The cancellation experiment overlaps turns instead (see
  Experiment 6). Latencies are wall clock on the shared laptop. Variants ran one after another, never in parallel.
* **Batch harnesses.** Baselines A–C and the batch pipelines receive the final transcript at once. Their latency
  starts at that moment, which is the same reference as "after utterance end" for the runtime variants.
* **Instrument.** Every answer from every system is scored by the same instrument, so verifier-judged metrics are
  comparable across systems:
  * answers are split into claims, each with its cited sections;
  * each claim is judged by the Phase 7 verifier against the sections the system retrieved;
  * label metrics (answer correctness, completeness, forbidden values, conflict reporting, insufficiency) are
    deterministic string checks against the expected-claim key strings.
* **Instrument bias.** The instrument is part of the proposed system's own Phase 7 stack, so verifier-judged metrics
  are biased toward systems that verify with the same model. §19 and §23 measure and discuss this.
* **No LLM judge** is used anywhere.
* **Randomness control.**
  * LLM: seed 7, temperature 0.
  * Statistics: seed 20261004.
  * Dataset: static, built deterministically.
  * Retrieval and verification: deterministic.
  * Asynchronous scheduling and wall-clock latency are not deterministic. Their run-to-run variation is measured in
    §19 (`experiments/results/REPRODUCIBILITY`).

## 6. Baselines

| id | system | retrieval | answer |
|---|---|---|---|
| A | Naive RAG (`naive_rag`) | dense top-5 | one LLM call; the model writes the answer and cites evidence labels; no verification |
| B | Hybrid RAG (`hybrid_rag`) | BM25 + dense, RRF top-5 | as A |
| C | Hybrid + reranking (`hybrid_rerank`) | hybrid top-5 + cross-encoder | as A |
| D | Streaming RAG, fixed retrieval (`streaming_fixed`) | Phase 8 runtime, Phase 4–6 retrieval decisions, multi-intent session, delta planning, fixed hybrid k = 5 per need | Phase 7 grounded answers (verified claims, validated citations), drafts during the utterance |
| S_batch | Adaptive RAG (`adaptive_rag`) | Phase 6 session pipeline + Phase 9 adaptive retrieval, given the final transcript (no streaming) | Phase 7 grounded answers |
| S | **Full system** (`full_system`) | D + Phase 9 adaptive retrieval (claim-driven sufficiency, iterative retrieval, adaptive k, multi-hop, temporal and contradiction retrieval, cache and session reuse) | Phase 7 grounded answers, streamed drafts, validated final answer |

The extractive variants in Experiments 2, 4, 5, 8 and 9 replace the LLM with the deterministic Phase 7 extractive
generator. This isolates retrieval behaviour from LLM variance and makes those experiments exactly repeatable.

## 7. Metrics

The definitions are in `docs/evaluation/metrics.md` and the code index is in `experiments/metrics/README.md`. The
metric modules were validated before use (brief §55) against hand-computed cases in `tests/evaluation/`. Those cases
cover perfect and missing retrieval, correct and incorrect citations, a stale value asserted, abstention, and the
percentile minimum-n rules.

| group | metrics |
|---|---|
| retrieval | Recall@1/3/5/10, Precision@1/3/5, MRR, nDCG@10 on gold sections; for ambiguous questions any listed section counts |
| evidence | evidence recall / precision / coverage, unsupported evidence, stale evidence present, redundancy, source diversity |
| claims | claims per answer, claim support / unsupported / contradicted rate (verifier-judged), claim coverage (labels) |
| generation | answer correctness (all expected key strings stated and no forbidden value), completeness, forbidden value asserted, insufficiency handled, conflict reported, faithfulness, groundedness |
| citation | citation precision, recall, completeness, entailment, source validity, position correctness |
| hallucination | unsupported claim rate, hallucinated claim rate (numbers or identifiers in neither the evidence nor the question), citation-less fact rate, grounding-failure turns |
| latency | TTFT, TTFE, TTFA, TTVA and total, measured from the first chunk and from the utterance end; p50 / p90 / p95 / p99. p90 needs n ≥ 10, p95 n ≥ 20 and p99 n ≥ 100; otherwise NOT MEASURED |
| streaming | answer updates, revisions, stale-update rate, time to useful and final answer, ordering errors |
| efficiency / cost | retrieval calls, embeddings, lexical searches, reranker calls, chunks retrieved, avg k, iterations, expansions, documents, cache hit, evidence reused, LLM calls, prompt and output tokens |
| cancellation | worker time on superseded work (wasted vs useful, from task `exec_ms`), cancelled tasks, stale results discarded |
| robustness | recovery, degraded success, incorrect answer, failure propagation |

**Not measured, with reasons:**

* answer relevance and human-judged quality: human study not run, no LLM judge;
* GPU memory: no discrete GPU;
* currency cost: no priced API;
* VERY_HARD results: no samples.

## 8. Experiment 1 — Baseline Comparison

**Question.** How do naive, hybrid and hybrid + rerank RAG, streaming without adaptivity, adaptive RAG and the full
system compare on the same 87 held-out turns?

**Data.** `experiments/results/EXP01_baselines/results.json`; tables in `experiments/results/tables.md`.

**Turns with each label:**
* gold sections: 82 turns;
* answer key strings: 74;
* forbidden (stale) values: 13;
* INSUFFICIENT: 5.

| metric (test) | A naive | B hybrid | C hybrid + rerank | D streaming, fixed | S_batch adaptive | **S full system** |
|---|---|---|---|---|---|---|
| Recall@1 | 0.695 | 0.640 | 0.780 | 0.634 | 0.774 | 0.762 |
| Recall@5 | 0.976 | **0.994** | **0.994** | 0.963 | 0.915 | 0.908 |
| MRR | 0.867 | 0.836 | **0.925** | 0.823 | 0.900 | 0.891 |
| evidence precision | 0.246 | 0.256 | 0.254 | 0.240 | **0.698** | 0.669 |
| evidence items handed to the answer stage | 5.0 | 5.0 | 5.0 | 5.1 | 2.3 | 2.3 |
| **answer correctness** (labels) | 0.851 | **0.892** | 0.878 | 0.703 | 0.797 | 0.730 |
| completeness | 0.899 | **0.926** | 0.912 | 0.824 | 0.824 | 0.804 |
| stale / forbidden value asserted (n = 13, lower is better) | 0.154 | 0.077 | 0.077 | 0.538 | **0.000** | 0.231 |
| insufficiency handled (n = 5) | **0.600** | 0.400 | 0.400 | 0.200 | 0.200 | 0.200 |
| hallucinated claim rate (model-free) | 0.018 | 0.012 | **0.005** | 0.016 | 0.012 | 0.014 |
| claim support (verifier-judged, biased §23) | 0.824 | 0.861 | 0.868 | 1.000 | 1.000 | 1.000 |
| citation precision (verifier-judged) | 0.825 | 0.843 | 0.842 | 1.000 | 1.000 | 1.000 |
| TTVA after utterance end, p50 / p95 ms | 1,784 / 4,450 | 1,844 / 3,657 | 1,916 / 4,434 | 2,130 / 6,714 | **1,621** / 5,763 | 1,690 / 4,833 |
| retrieval calls / query | 1.00 | 1.00 | 1.00 | 1.52 | 1.12 | 1.39 |
| LLM calls / query | 1.00 | 1.00 | 1.00 | 1.47 | 1.29 | 1.28 |

**Paired tests** (exact McNemar for binary metrics, Wilcoxon signed-rank for continuous ones):
* **Full system vs naive RAG, answer correctness: 0.730 vs 0.851.**
  * The full system lost 13 turns and gained 4 (p = 0.049).
  * 95% bootstrap CI of the difference: [−0.230, −0.014].
* **Adaptive RAG vs hybrid + rerank.**
  * Recall@5: −0.079 (CI [−0.140, −0.024], p = 0.012).
  * Answer correctness: −0.081, 10 lost and 4 gained (p = 0.18, not significant).
* **Hybrid + rerank vs hybrid.**
  * MRR: +0.089 (CI [0.046, 0.136], p < 0.001).
  * TTVA: +220 ms at the mean (p < 0.001).
* **Full system vs streaming baseline D.**
  * MRR: +0.069 (p = 0.021).
  * TTVA after utterance end: −784 ms at the mean (CI [−1,150, −451], p < 0.001).
  * Retrieval calls: −0.13 per query (p = 0.048).
  * Answer correctness: +0.027 (p = 0.73, not significant).

**Interpretation**
* On this 43-chunk corpus, a stateless hybrid top-5 retriever followed by one LLM call is the most accurate system by
  the labels.
* The proposed stack wins on stale-value avoidance, but only in batch mode (0 of 13 vs 1 of 13 for B and C). It also
  wins on evidence precision (0.70 vs 0.25), on verifier-judged support and on latency after the utterance ends.
* Its answer correctness is lower. §20 analyses the causes:
  * retrieval-decision skips;
  * early commitment while streaming;
  * multi-hop and constraint analysis;
  * no answerability check.

## 9. Experiment 2 — Adaptive Top-k

Extractive answers, so retrieval effects are isolated from the LLM. Sources:
`experiments/results/EXP02_adaptive_topk/results.json` and `experiments/plots/09b_quality_compute_frontier.svg`.

| test (n = 87) | k = 3 | k = 5 | k = 10 | k = 20 | adaptive, k fixed at 5 | **adaptive** |
|---|---|---|---|---|---|---|
| Recall@5 | 0.890 | **0.963** | **0.963** | **0.963** | 0.921 | 0.915 |
| MRR | 0.807 | 0.815 | 0.815 | 0.815 | **0.906** | 0.900 |
| nDCG@10 | 0.815 | 0.849 | 0.849 | 0.849 | **0.897** | 0.890 |
| evidence precision | 0.355 | 0.236 | 0.122 | 0.062 | 0.681 | **0.698** |
| answer correctness (extractive) | 0.689 | 0.703 | 0.703 | 0.703 | 0.770 | **0.784** |
| retrieval calls / query | 1.02 | 1.02 | 1.02 | 1.02 | 1.12 | 1.12 |
| embeddings / query | 1.02 | 1.02 | 1.02 | 1.02 | **0.62** | **0.62** |
| chunks retrieved / query | 3.1 | 5.1 | 10.2 | 20.5 | 5.7 | 4.7 |
| avg k | 3 | 5 | 10 | 20 | 5.2 | 4.1 |

**Dev split (n = 129).**

| dev metric | adaptive | k = 5 | k = 10 | k = 20 |
|---|---|---|---|---|
| Recall@5 | 0.966 | 0.951 | 0.951 | 0.951 |
| MRR | 0.958 | 0.894 | 0.896 | 0.896 |
| evidence precision | 0.791 | 0.239 | 0.141 | 0.094 |
| retrieval calls / query | 1.34 | 1.07 | 1.07 | 1.07 |

**Paired tests.**
* **Adaptive vs k = 5, test.**
  * Recall@5: −0.049 (p = 0.031).
  * MRR: +0.085 (p = 0.006).
  * Retrieval calls: +0.09 per query (p = 0.035).
  * Answer correctness: +0.081, 10 gained and 4 lost (p = 0.18, not significant).
* **Adaptive vs k = 5, dev.** MRR: +0.064 (p = 0.008).
* **k = 10 and k = 20 add nothing over k = 5 on test.** They retrieve two and four times the chunks and lower
  precision.

**Reading.**
* Adaptive k ranks better and hands on far less noise: precision triples.
* It loses some recall on the held-out corpus: gold sections it filters out.
* It uses more searches but fewer embeddings, because a lexical fast path skips the dense search.

## 10. Experiment 3 — Claim-Driven Retrieval

The real LLM is used here. Sources: `EXP03_claim_driven`, and `ABLATION_runtime` for the runtime arm.

| test | query-driven sufficiency | **claim-driven** (S_batch) | S − claim-driven (runtime) | S (runtime) |
|---|---|---|---|---|
| answer correctness | 0.743 | **0.797** | 0.703 | 0.730 |
| claim coverage / completeness | 0.804 | **0.824** | 0.804 | 0.804 |
| stale value asserted (n = 13) | — | 0.000 | 0.385 | 0.231 |
| Recall@5 | 0.896 | 0.915 | 0.908 | 0.908 |
| claim support (verifier) | 1.000 | 1.000 | 1.000 | 1.000 |
| hallucinated claim rate | 0.012 | 0.012 | 0.014 | 0.014 |
| retrieval calls / query | 1.10 | 1.12 | 1.40 | 1.39 |

* **Batch pipeline.** Claim-driven sufficiency gained 4 turns and lost 0 on answer correctness. That is below the 6
  discordant pairs the exact test needs, so no p-value is reported. Its cost was +0.01 retrieval calls per query
  (p = 0.67).
* **Runtime.** It gained 2 turns and lost 0, and stale values fell from 5 of 13 to 3 of 13.
* **Claim support** is 1.000 in every arm. Both arms verify with the same verifier as the instrument, so this metric
  cannot show a grounding difference here (§23).

## 11. Experiment 4 — Cache

Extractive answers on the conversation sessions only (CONTEXTUAL_FOLLOWUP, ENTITY_CORRECTION and REPEATED_QUERY,
whole sessions): 32 test turns and 30 dev turns. Source: `EXP04_cache`.

| | no cache | query + session caches | + validated evidence / claim reuse (adaptive) |
|---|---|---|---|
| cache hit rate, test / dev | 0.000 / 0.000 | 0.062 / 0.100 | **0.094 / 0.200** |
| retrieval calls / query, test / dev | 1.06 / 1.37 | 1.00 / 1.27 | **0.97 / 1.17** |
| embeddings / query, test / dev | 0.50 / 0.90 | 0.50 / 0.87 | **0.47 / 0.77** |
| answer correctness, test | **0.781** | **0.781** | 0.750 |
| Recall@5, test | **0.922** | **0.922** | 0.891 |
| TTVA after end p50, test (ms) | 9 | 9 | 9 |

* **Work saved.** On dev, full reuse cut retrieval calls by 0.20 per turn against no cache (p = 0.031) and hit the
  cache on 6 of 30 turns (exact McNemar p = 0.031).
* **Quality cost on test.** Evidence and claim reuse cost one turn of answer correctness. In S01.2, a contextual
  follow-up about the student discount was answered from the reused fare evidence of the previous turn. Too few
  discordant pairs for a test.
* **Latency.** With extractive answers the latency saved is about 1 ms. Measured, not claimed as a saving.

## 12. Experiment 5 — Delta Retrieval vs Full Re-retrieval

Batch pipeline, extractive answers, same conversation sessions as Experiment 4. Source: `EXP05_delta`.

| | full re-retrieval each turn | **delta retrieval** (adaptive) |
|---|---|---|
| retrieval calls / query, test / dev | 1.22 / 1.50 | **0.97 / 1.17** |
| embeddings / query, test / dev | 0.53 / 0.93 | **0.47 / 0.77** |
| evidence reused / query, test / dev | 0 / 0 | 0.53 / 0.47 |
| answer correctness, test | **0.812** | 0.750 |
| Recall@5, test | **0.922** | 0.891 |
| TTVA after end p50, test (ms) | 16 | 9 |

* **Work saved (H3).** Delta retrieval removes 0.25 retrieval calls per turn on test (CI [0.06, 0.47], p = 0.031)
  and 0.33 on dev (p = 0.016). This is the redundant work H3 predicted.
* **Quality cost.** It lost 2 turns of answer correctness on test (2 lost, 0 gained; too few for a test):
  * S01.2: retained fare evidence answered a student-discount follow-up;
  * S08.2: after "I meant the monthly pass", the delta plan answered "Not established", while full re-retrieval
    found the price.
* **Runtime ablation is uninformative.** In the streaming runtime, `session.full_restart` changed no turn's retrieval
  calls or answer (§18). The switch reaches the engine, since it disables the semantic cache (5 → 0 `QUERY_REUSED`
  events), but the runtime's delta plans came out identical. The runtime ablation therefore says nothing about delta
  retrieval. This is listed for Phase 11.

## 13. Experiment 6 — Cancellation

Overlapping turns: the next utterance starts 400 ms after the previous one ends. Sessions: the 5 ENTITY_CORRECTION
sessions and the 5 streamed ASR corrections, 15 turns in all. Full system with the real LLM. Two variants add a
SYNTHETIC 300 ms per dense search, simulating a remote index. Source: `EXP06_cancellation`.

**Same inputs, work done (sum over 15 turns)**

| variant | total worker ms | LLM calls | cancelled tasks | flagged wasted ms |
|---|---|---|---|---|
| cancellation off | 28,439 | 20 | 0 | 123 |
| **cancellation on** | **19,914** | **10** | 5 | 3,866 |
| off, remote +300 ms | 32,772 | 20 | 0 | 1,981 |
| **on, remote +300 ms** | **21,532** | **10** | 7 | 5,217 |

**Final vs superseded turns**

| variant | final turns: answer correct | final turns: TTVA after end p50 / max ms | superseded first questions answered |
|---|---|---|---|
| cancellation off | 5 / 10 | 2,459 / 3,808 | 5 / 5 |
| **cancellation on** | 5 / 10 | **1,379** / 2,877 | 0 / 5 |
| off, remote | 5 / 10 | 2,443 / 4,102 | 5 / 5 |
| **on, remote** | 5 / 10 | **1,551** / 2,616 | 0 / 5 |

* **Less work, faster corrected answers.** For the same inputs, cancellation reduced total worker time by 30%
  (local) and 34% (remote), and halved LLM calls. The answer to the corrected question arrived about 1.1 s sooner at
  the median. Final-answer correctness was unchanged.
* **Scoring caveat.** The configured all-turn answer correctness drops from 0.667 to 0.333. The 5 superseded first
  questions get no answer, which is the intended behaviour. The final-turn view was added **after** the first results
  were seen, and both views are reported.
* **Wasted-work caveat.** "Flagged wasted ms" under-counts waste without cancellation: superseded work runs to
  completion and is never flagged. Total worker time is the fair comparison.
* **Statistics.** n = 15 is below the 20-pair minimum, so no p-values.
* **Sequential conversations.** Without overlapping turns, cancellation changes nothing measurable. The runtime
  ablation "S − cancellation" equals S on all 87 turns.

## 14. Experiment 7 — Streaming vs Batch

Same runtime, same retrieval stack. Batch mode retrieves at the end of the utterance and makes no drafts. Source:
`EXP07_streaming`, plots `03`–`05` and `11`.

| test | batch | **streaming (S)** |
|---|---|---|
| TTFE from first chunk, p50 / p95 ms | 766 / 1,273 | **261 / 767** |
| TTFA (first answer content) from first chunk, p50 / p95 ms | 2,191 / 6,491 | **271 / 1,632** |
| TTVA (validated answer) from first chunk, p50 / p95 ms | **2,191** / 6,492 | 2,367 / **5,561** |
| TTVA after utterance end, p50 / p95 ms | **1,496** / 5,703 | 1,690 / **4,833** |
| time to an answer containing all expected facts, p50 ms | 2,222 (n = 65) | **273** (n = 63) |
| answer updates / turn; revisions / turn | 3.0; 0.14 | 5.1; 0.39 |
| stale updates; ordering errors | 0; 0 | 0; 0 |
| **answer correctness** | **0.824** | 0.730 |

**Paired results**

| metric | streaming − batch | 95% CI | test |
|---|---|---|---|
| TTFE | −442 ms | [−496, −389] | p < 0.001 |
| TTFA | −2,177 ms | [−2,519, −1,861] | p < 0.001 |
| TTVA | +124 ms | [17, 227] | p < 0.001 |
| answer correctness | 7 turns lost, 0 gained | — | exact McNemar p = 0.016 |

**Reading.**
* Streaming moves first evidence and first answer content much earlier. A draft containing all expected facts is
  visible at a median of 273 ms after the first chunk, often before the user stops speaking.
* The *validated* answer is not faster at the median. Its tail is shorter.
* Streaming costs answer quality. All 7 lost turns are decisions made on a partial transcript (§20):
  * T25, S05.1: "Which documents do…" is classified as meta-conversation and never retrieved;
  * T24: retrieval on "What does a monthly pass cost" before "for a student" arrives;
  * T26, T28, T45: temporal and version resolution done before the date or "now" arrives;
  * SC1: a claim about the pre-correction entity survives the ASR revision.
  Batch mode in the same runtime answers all 7 correctly.

## 15. Experiment 8 — Iterative Retrieval

Extractive answers. Source: `EXP08_iterative`.

| | single pass | **iterative** |
|---|---|---|
| Recall@5, test / dev | 0.902 / 0.952 | **0.915 / 0.966** |
| MRR, test / dev | 0.900 / 0.945 | 0.900 / **0.958** |
| evidence coverage, test | 0.899 | **0.919** |
| answer correctness, test | 0.770 | **0.784** |
| retrieval calls / query, test / dev | **1.02 / 1.11** | 1.12 / 1.34 |
| TTVA after end p50, test (ms) | 9 | 9 |

* **Cost.** Iteration costs +0.09 retrieval calls per query on test (p = 0.008) and +0.23 on dev (p < 0.001).
* **Gain.** One more correct test turn (1 gained, 0 lost) and small recall gains. The quality gain is too small to
  test.

## 16. Experiment 9 — Multi-hop Retrieval

Extractive answers on MULTI_HOP turns: 7 test, 5 dev. Source: `EXP09_multihop`.

| | fixed k = 5 | adaptive without multi-hop | adaptive with multi-hop |
|---|---|---|---|
| Recall@5, test | **1.000** | 0.714 | 0.714 |
| answer correctness, test | **0.857** | 0.429 | 0.429 |
| retrieval calls / query, test | **1.14** | 1.43 | 1.57 |
| Recall@5, dev (development material) | 0.300 | 0.300 | **0.900** |
| retrieval calls / query, dev | **1.00** | 2.00 | 3.00 |

* **Held-out test: a negative result.**
  * Multi-hop retrieval did not help, and adaptive retrieval was *worse* than fixed k = 5.
  * With 43 chunks, the top 5 already contains both hops. Adaptive filtering then drops the second hop (T35 keeps
    only `TCO §1`).
  * The hop mechanism fired on 1 of 7 test items (T31, `millbrook -> Zone A`) and changed no outcome. The other 6
    were analysed as SIMPLE or COMPLEX, not MULTI_HOP.
* **Dev split.** Multi-hop works on the dev material it was developed on (Recall@5 0.30 → 0.90).
* **n is 7 and 5**, so this is anecdotal. There are no tests.

## 17. Experiment 10 — Contradiction Resolution

Real LLM, CONTRADICTORY and TEMPORAL turns (11). Source: `EXP10_contradiction`.

| | C hybrid + rerank | session pipeline, fixed retrieval | adaptive − contradiction / temporal | **adaptive (S_batch)** |
|---|---|---|---|---|
| answer correctness (n = 8 labelled) | 6 / 8 | 3 / 8 | 6 / 8 | **8 / 8** |
| stale value asserted (n = 5) | 1 / 5 | 5 / 5 | 2 / 5 | **0 / 5** |
| conflicting notices: both values reported (n = 3) | 3 / 3 | 3 / 3 | 3 / 3 | 3 / 3 |
| stale evidence handed on (n = 3) | 3 / 3 | 3 / 3 | 2 / 3 | **0 / 3** |
| retrieval calls / query | 1.00 | 1.00 | 1.09 | 1.09 |

* **Batch mode.** Contradiction- and temporal-aware retrieval removed every stale value and answered all labelled
  turns correctly. It cost +0.09 searches per query.
* **Streaming does not hold this.** The full system asserts a stale value in 2 of the 3 TEMPORAL turns that carry
  stale-value labels (T26, T28), versus 0 in batch mode (§14, §20).
* **n is 8 or fewer**, so there are no tests.

## 18. Ablation Study

Each variant removes one component. Sources: `ABLATION_runtime` (streaming runtime, n = 87) and
`ABLATION_answer_stage` (batch pipeline, n = 87). The table is `experiments/results/ablation_table.json` and the plot
is `experiments/plots/10_ablation.svg`.

| system | quality: answer correct | grounding: stale value asserted (n = 13) / hallucinated rate | latency: TTVA after end p50 / p95 ms | retrieval cost: calls / embeddings / LLM calls per query | reliability: failed / no validated answer |
|---|---|---|---|---|---|
| **Full system (S)** | 0.730 | 0.231 / 0.014 | 1,690 / 4,833 | 1.39 / 0.77 / 1.28 | 0 / 2 |
| S − adaptive retrieval (= D) | 0.703 | 0.538 / 0.016 | 2,130 / 6,714 | 1.52 / 1.52 / 1.47 | 0 / 2 |
| S − claim-driven retrieval | 0.703 | 0.385 / 0.014 | 1,631 / 4,571 | 1.40 / 0.81 / 1.31 | 0 / 2 |
| S − session memory (fresh runtime per turn) | 0.716 | 0.231 / 0.015 | 1,648 / 5,323 | 1.37 / 0.72 / 1.26 | 0 / 7 |
| S − delta retrieval | 0.730 | 0.231 / 0.014 | 1,632 / 4,610 | 1.39 / 0.77 / 1.28 | 0 / 2 |
| S − cancellation | 0.730 | 0.231 / 0.014 | 1,601 / 4,536 | 1.39 / 0.77 / 1.28 | 0 / 2 |
| **Adaptive RAG, batch (S_batch)** | 0.797 | 0.000 / 0.012 | 1,621 / 5,763 | 1.12 / 0.62 / 1.29 | 0 / 0 |
| S_batch − evidence verification | **0.865** | 0.000 / 0.010; unsupported claims 0.153 | 1,519 / 5,621 | 1.12 / 0.62 / 1.00 | 0 / 0 |
| S_batch − citation validation | 0.770 | 0.000 / 0.000; 13 empty answers | 1,557 / 5,547 | 1.12 / 0.62 / 1.00 | 0 / 0 |
| S_batch + reranking | 0.797 | 0.000 / 0.014 | 1,691 / 5,591 | 1.12 / 0.62 / 1.25 | 0 / 0 |

* **Adaptive retrieval** is the largest runtime contribution. Against S − adaptive:
  * stale values fall from 7 of 13 to 3 of 13;
  * embeddings per query halve;
  * TTVA after end is −784 ms at the mean (p < 0.001).
  Answer correctness rises only 0.027 (not significant).
* **Claim-driven retrieval:** +2 correct turns and −2 stale values (too few pairs for a test).
* **Session memory** (removed, every turn starts a fresh runtime):
  * 5 more turns with no validated answer (7 vs 2);
  * Recall@5 0.860 vs 0.908;
  * 2 turns lost, 1 gained.
* **Delta retrieval and cancellation** show no effect in sequential runtime conversations. For delta, see §12. For
  cancellation, Experiment 6 measures the effect with overlapping turns.
* **Evidence verification lowers measured correctness by 0.068.** Without it, 6 turns are gained and 1 lost
  (McNemar p = 0.125). The verifier drops true claims it cannot entail (§23). Without it, 15.3% of claims are
  unsupported (p < 0.001) and citation precision falls to 0.879. This is a quality-versus-safety tradeoff, not a free
  gain.
* **Citation validation** has no significant correctness effect (6 lost, 4 gained, p = 0.75). It removes 13 answers
  entirely (empty), because claims with wrong LLM labels fail verification and are dropped.
* **Reranking** in the adaptive pipeline changes nothing in quality. It adds about 34 ms per search.

## 19. Statistical Analysis

* **Unit and n.** The unit is a turn. Test n = 87, with 74 to 82 labelled per metric. Dev n = 129. Every table
  reports the means. `results.json` also stores the median, SD, bootstrap 95% CI, and p50 / p90 / p95 / p99.
* **Tests.** All comparisons are paired on the same turns:
  * binary metrics: exact McNemar (two-sided binomial on discordant pairs);
  * continuous metrics: Wilcoxon signed-rank (two-sided, zero differences dropped);
  * plus a bootstrap 95% CI of the mean difference (10,000 resamples, seed 20261004).
* **Effect sizes.** Cohen's d_z and matched-pairs rank-biserial.
* **Minimum sample.** No test below 20 pairs or 6 non-zero differences. Such cells read `too_few_*`, with p NOT
  MEASURED.
* **Assumptions.**
  * Turns are treated as independent, although 32 test turns belong to 15 conversations (clustered).
  * Wilcoxon assumes symmetric differences. Latency differences are skewed, so the mean CI and the Wilcoxon p-value
    can disagree. Example: S − claim-driven TTVA has a mean difference of −27 ms with CI [−158, 103], yet a
    Wilcoxon p of 0.003, because there is a consistent small per-turn shift plus large outliers. Both are reported.
  * No multiple-comparison correction across the 287 comparison rows (`results.json` → `comparisons`). Read p-values
    as descriptive. Effects with p < 0.01 and a CI excluding 0 are the robust ones:
    * adaptive vs k = 5: MRR;
    * reranking: MRR;
    * S vs D: TTVA;
    * streaming vs batch: TTFE / TTFA;
    * verification: unsupported claims;
    * iteration: retrieval calls.
* **Reproducibility (measured).** `experiments/results/REPRODUCIBILITY/results.json` re-ran four variants with
  identical configuration.

| variant | identical answer text | identical evidence | identical correctness labels | run-to-run TTVA, median abs. difference | median relative |
|---|---|---|---|---|---|
| adaptive_x (extractive) | 87 / 87 | 87 / 87 | 87 / 87 | 1.1 ms | 14% of ~9 ms |
| naive_rag (LLM) | 87 / 87 | 87 / 87 | 87 / 87 | 72 ms | 3.8% |
| adaptive_rag (LLM) | 86 / 87 | 87 / 87 | 87 / 87 | 39 ms | 2.6% |
| full_system (runtime + LLM) | 86 / 87 | 87 / 87 | 87 / 87 | 45 ms | 2.6% |

* **What the reproducibility runs show.**
  * Quality numbers are reproducible exactly. One LLM answer text (T19) differs between runs without changing its
    label.
  * Latency varies by a few percent at the median and by up to about 1 s per turn (max).
  * The full system's second run used the corrected evidence code directly and matched the trace re-derived first
    run on every turn. This validates the repair in §29.

## 20. Error Analysis

**Error categories.** These are rule-assigned per turn (`docs/evaluation/error_analysis.md`). The failure store is
`experiments/failure_cases/<experiment>.jsonl` and holds every categorised turn with its query, expected answer,
answer, evidence, claims, citations and trace path. Counts on the test split (a turn can have several categories):

| category | A naive | B hybrid | C rerank | D streaming fixed | S_batch | **S** |
|---|---|---|---|---|---|---|
| RETRIEVAL_FAILURE | 0 | 0 | 0 | 2 | 4 | 4 |
| QUERY_ANALYSIS_FAILURE (adaptive systems only) | — | — | — | — | 10 | 10 |
| ENTITY_FAILURE | 2 | 2 | 1 | 7 | 4 | 5 |
| MEMORY_FAILURE | 0 | 0 | 0 | 0 | 1 | 1 |
| EVIDENCE_FAILURE | 10 | 9 | 9 | 9 | 5 | 9 |
| CLAIM_FAILURE | 15 | 15 | 13 | 0 | 0 | 0 |
| GENERATION_FAILURE | 11 | 10 | 11 | 24 | 12 | 16 |
| CITATION_FAILURE | 16 | 17 | 16 | 0 | 0 | 0 |
| LATENCY_FAILURE / ORCHESTRATION_FAILURE | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| turns with no category | 56 | 57 | 56 | 59 | 61 | 56 |

**Reading the categories**
* Categories flag process deviations as well as wrong answers. The full system has 31 categorised turns but 20 wrong
  answers among its 74 labelled turns.
* QUERY_ANALYSIS_FAILURE needs the adaptive trace, so baselines cannot receive it.
* No LATENCY_FAILURE gate was configured, so that category is never assigned.

**Failure patterns.** These are traced in the event logs. The qualitative sample (`experiments/results/
qualitative_full_system.md`) is a seeded random draw of 8 failing and 4 passing turns; nothing was hand-picked.

1. **The retrieval decision skips real questions** (QUERY_ANALYSIS, recorded as RETRIEVAL_FAILURE).
   * The Phase 4 rule-based decision labels the partial transcript "Which documents do…" as meta-conversation, with
     confidence 0.85, and never retrieves (T25, S05.1).
   * Turns with no retrieval: 2 in D and S, 3 in S_batch, 7 when session memory is removed. The runtime in batch mode
     answers T25 and S05.1 correctly.
2. **Early commitment while streaming** (GENERATION / ENTITY).
   * Retrieval and drafts run on partial transcripts. "What did a single ride cost" is retrieved before "in 2025"
     arrives.
   * The committed answer then reports a conflict instead of resolving it by date (T26, T28, T45).
   * "What does a monthly pass cost" is retrieved before "for a student" (T24).
   * A claim about the pre-correction entity survives the ASR revision (SC1).
   * Together with pattern 1, these are exactly the 7 turns streaming loses against batch mode (§14).
3. **Entity corrections in conversations keep or lose the wrong claims.**
   * "Sorry, I meant seniors, not students" keeps the students' claims next to the seniors' (S02.2).
   * "I meant folding bicycles" ends with an empty validated answer (S10.2).
   * ENTITY_FAILURE: S 5, D 7, baselines 1–2. The stateless baselines simply answer the new sentence.
4. **No answerability check.**
   * Grounded systems answer adjacent facts to unanswerable questions: "free wifi?" gets the senior concession (T46),
     "pay by credit card?" gets penalty amounts (T48). Insufficiency handled: 1 of 5 for every grounded system.
   * Naive RAG scores 3 of 5, but two of those are empty answers counted as abstentions (§7).
5. **Multi-hop and constraints.**
   * 6 of 7 test multi-hop questions are not analysed as MULTI_HOP.
   * Adaptive filtering drops the second-hop section (T35 keeps `TCO §1` and drops `TCO §2`, "closed on Sundays").
6. **Baselines: unverified claims and citations.**
   * A–C keep unsupported claims in 13–15 turns and mis-cite in 16–17 turns. The proposed stack removes these by
     construction.
   * Their errors are mostly *assertive*: a stale fare stated as current, e.g. naive RAG asserts a stale value in 2
     of 13 labelled turns.

## 21. Latency Analysis

Distributions: `experiments/plots/03_ttfe_distribution.svg`, `04_ttfa_distribution.svg`, `05_ttva_distribution.svg`,
`02_latency_ttva.svg`.

**Where time goes.** The local LLM dominates every validated answer.

| | generation, p50 / p95 ms | first token ms | rest of the path |
|---|---|---|---|
| naive RAG | 1,780 / 4,446 | 151 (LLM TTFT) | retrieval 3 ms (dense); 61 ms (hybrid + rerank) |
| S_batch | 1,606 / 5,736 | — | retrieval 13 ms; verification p50 2 ms / p95 102 ms |

**Validated answer after the end of the utterance**, p50 / p95 ms:

| system | p50 | p95 |
|---|---|---|
| S_batch | 1,621 | 5,763 |
| S | 1,690 | 4,833 |
| naive | 1,784 | 4,450 |
| hybrid | 1,844 | 3,657 |
| hybrid + rerank | 1,916 | 4,434 |
| D | 2,130 | 6,714 |

* All systems sit within a few hundred ms at the median. The proposed systems have longer p95 tails than hybrid,
  because of repair LLM calls (1.28 LLM calls per query).
* **Streaming changes what the user sees first, not when the validated answer arrives.**
  * TTFE p50 is 261 ms and TTFA p50 is 271 ms from the first chunk. The utterances last about 1 s.
  * TTFA after the utterance end has a negative median (−243 ms): first answer content appears before the user
    stops speaking.
* p99 is NOT MEASURED everywhere: n < 100.
* Run-to-run latency noise is about 3% at the median (§19). Ablation latency differences below about 100 ms at the
  median are within run-to-run noise (median abs. difference 39–72 ms per turn).

## 22. Retrieval Efficiency

| test | retrieval calls / q | embeddings / q | chunks retrieved / q | evidence handed on | evidence precision | source |
|---|---|---|---|---|---|---|
| fixed k = 5 (extractive) | 1.02 | 1.02 | 5.1 | 5.1 | 0.236 | EXP02 |
| fixed k = 20 (extractive) | 1.02 | 1.02 | 20.5 | 19.8 | 0.062 | EXP02 |
| adaptive (extractive) | 1.12 | **0.62** | 4.7 | 2.3 | **0.698** | EXP02 |
| D streaming fixed (LLM) | 1.52 | 1.52 | — | 5.1 | 0.240 | EXP01 |
| S full system (LLM) | 1.39 | 0.77 | — | 2.3 | 0.669 | EXP01 |
| S_batch (LLM) | 1.12 | 0.62 | — | 2.3 | 0.698 | EXP01 |

* **Adaptive retrieval spends more searches but fewer embeddings.** The lexical fast path skips dense search on
  exact-term questions. It also hands the answer stage about 55% fewer, much more precise evidence items.
* **Streaming re-retrieves as the transcript grows.** S makes 1.39 calls per turn against 1.12 in batch, and D makes
  1.52.
* **Reranker calls are 0** for every runtime variant. The policy is `rerank: never`, set from the Phase 9 reranker
  evaluation.
* Cache, delta and iteration costs are in §11, §12 and §15.

## 23. Grounding Analysis

* **The instrument and its measured accuracy.** Claim support is judged by the Phase 7 verifier
  (`nli-deberta-v3-xsmall` + rules). On 172 claims with labels known by construction
  (`experiments/results/VERIFIER_validation`):

| measure | value |
|---|---|
| accuracy | 0.942 |
| precision / recall of SUPPORTED | 0.896 / 0.896 |
| false acceptance | 0.040 |

| perturbation | judged SUPPORTED |
|---|---|
| number swap | 0 / 38 |
| entity swap | 2 / 46 |
| negation | 3 / 40 (all "X is not classified as Zone Y") |
| original (true) claims | 43 / 48 |

  The verifier rejects 5 true claims, mostly past-tense temporal facts such as "In 2025 a single ride cost 2.40
  euros". This explains part of the correctness lost to verification (§18).
* **The bias.** Systems that release only verified claims score 1.000 on claim support, citation precision and
  citation entailment by construction, because the instrument is their own verifier. These metrics separate verified
  from unverified systems: A–C score 0.82–0.87 on claim support. They cannot rank verified systems against each other.
* **Model-free grounding.** The hallucinated claim rate counts numbers or identifiers that appear in neither the
  evidence nor the question. It is low everywhere, 0.005–0.018, and no difference is significant.
* **Stale-value assertion** is the grounding failure that separates systems: S_batch 0 / 13, S 3 / 13, D 7 / 13, B and
  C 1 / 13, A 2 / 13.
* **Answer relevance and human-judged grounding: NOT MEASURED.** The human protocol and its blinded sheet are prepared
  (84 items, `experiments/datasets/human_eval_sheet.csv`), but the study was not run. No LLM judge was used.

## 24. Citation Analysis

| test | A naive | B hybrid | C rerank | S_batch | S |
|---|---|---|---|---|---|
| citations / answer | 1.95 | 1.89 | 1.90 | 1.98 | 2.06 |
| citation precision (cited section entails its claim) | 0.825 | 0.843 | 0.842 | 1.000 | 1.000 |
| citation recall (claims with a supporting citation) | 0.835 | 0.849 | 0.862 | 1.000 | 1.000 |
| citation completeness (expected claims stated and cited) | 0.892 | **0.932** | 0.919 | 0.818 | 0.797 |
| source validity (cited section was retrieved for the turn) | 1.000 | 1.000 | 1.000 | 0.988 | 0.985 |
| position correctness | 0.993 | 0.993 | 0.986 | 1.000 | 1.000 |

* **Validated citations are always entailing** (by construction), and the baselines mis-cite about 16% of claims.
* **Completeness is lower** for the proposed systems, because fewer expected claims are stated at all.
* **Source validity below 1** for S and S_batch: some citations point to sections retained from the previous turn,
  which were not part of the turn's own evidence (S02.2, S04.2).
* **Ablation:** without citation validation, citation precision stays at 1.000, because the verifier still filters
  claims. 13 answers become empty, though (§18).

## 25. Robustness

**Setup.** `experiments/runners/robustness.py` drives the full system (runtime + adaptive retrieval + qwen3:4b) on 10
test questions per scenario: T01, T04, T08, T13, T16, T21, T27, T31, T44, T17. Faults are injected with the Phase 8
`FaultInjector` and are **SYNTHETIC**. Source: `experiments/results/ROBUSTNESS/results.json`.

| scenario | runs | completed with a committed answer | answer correct | degraded mode declared | incorrect answer | outcome |
|---|---|---|---|---|---|---|
| every retrieval fails | 10 | 10 | 0 | 10 | 0 | explicit "Not established …" abstention every time; nothing invented |
| dense index unavailable | 10 | 10 | 10 | 0 | 0 | lexical evidence suffices |
| transient network errors (2 retries) | 10 | 10 | 10 | 0 | 0 | retried transparently |
| **every LLM call times out** | 10 | **0** | 0 | 10 | 0 | **GENERATION_DEGRADED and TIMED_OUT reported, but no fallback answer is produced** |
| every LLM call errors | 10 | 10 | 10 | 10 | 0 | extractive fallback (`ollama->extractive`) |
| verifier (NLI) fails | 10 | 10 | 10 | 10 | 0 | rules-only verification |
| malformed input: empty chunk, 5,000-character chunk, control characters, injection text before the question | 10 | 10 | 10 | 0 | 0 | junk and injection ignored |
| rapid updates: one word every 20 ms + 2 ASR revisions | 10 | 10 | 9 | 0 | 1 | T21 keeps an extra seniors claim |
| failure propagation: session A with every retrieval failing, session B clean, side by side | 5 | 5 / 5 for B | B identical to its clean run 5 / 5 | — | 0 | propagation rate 0.0 |

**Negative result.** When the LLM times out, the runtime declares the degraded mode, records the timeout and
completes the turn, but commits no answer, after about 30 s of deadlines. With an LLM *error*, the extractive
fallback recovers every turn. The timeout path does not reach that fallback. This is listed for Phase 11.

**Not measured:** real network partitions, real database outages, memory pressure, and concurrent load beyond two
sessions.

## 26. Resource Usage

**Setup.** `experiments/runners/resources.py` measured this machine (Apple M5 Pro, 24 GiB unified memory, macOS
26.5.1). Source: `experiments/results/RESOURCES/results.json` and `environment.json`.

**Memory (peak RSS of the evaluation process)**

| step | peak RSS |
|---|---|
| Python + imports | 40 MB |
| + index, BM25, `bge-small` embedder, NLI verifier | 1,012 MB |
| + cross-encoder reranker | 1,181 MB |
| + verification instrument | 1,275 MB |

These are process peaks, so the increments approximate each component's footprint.

**CPU per call** (30 test questions; ONNX uses several threads, so CPU time exceeds wall time)

| call | CPU median / p90 ms | wall median / p90 ms |
|---|---|---|
| dense top-5 | 23 / 27 | 3.3 / 3.8 |
| hybrid top-5 | 25 / 28 | 3.6 / 4.0 |
| hybrid top-5 + rerank | 278 / 297 | 32 / 34 |
| verify one claim against 5 sections | 87 / 240 | 22 / 60 |

**In-process CPU per runtime turn** (mean; LLM server excluded)

| variant | CPU per turn |
|---|---|
| S | 160 ms |
| D | 492 ms |
| S in batch mode | 133 ms |

**Disk.** The test-corpus index is 0.13 MB. Models:

| model | size |
|---|---|
| `nli-deberta-v3-xsmall` | 279 MB |
| `bge-small-en-v1.5` | 128 MB |
| `ms-marco-MiniLM-L6-v2` | 87 MB (qint8: 23 MB) |
| `all-MiniLM-L6-v2` | 87 MB (qint8: 22 MB) |

Weight hashes are in `environment.json`.

**LLM server.**
* `qwen3:4b` (Q4_K_M, digest `359d7dd4…`), Ollama 0.35.1.
* `ollama ps`: 3.9 GB resident, "100% GPU" (Apple Metal, unified memory), context 8,192.
* Server process RSS: 31 MB.
* Model runner process RSS: NOT MEASURED (not matched by the probe).

**Other resources.**
* GPU memory: NOT MEASURED (no discrete GPU).
* Network: local HTTP to Ollama only, one request per LLM call (§27).
* Database requests: none, because retrieval is in process.

## 27. Cost / Operation Analysis

No priced API is used: the LLM is local, and embedding, reranking and NLI run in process. **Currency cost is NOT
MEASURED.** Operation counts per query on the test split (`EXP01_baselines`, `ABLATION_*`):

| system | retrieval calls | embeddings | reranker calls | LLM calls | prompt tokens | output tokens |
|---|---|---|---|---|---|---|
| A naive RAG | 1.00 | 1.00 | 0 | 1.00 | 323 | 148 |
| B hybrid RAG | 1.00 | 1.00 | 0 | 1.00 | 323 | 142 |
| C hybrid + rerank | 1.00 | 1.00 | 1.00 | 1.00 | 328 | 145 |
| D streaming, fixed | 1.52 | 1.52 | 0 | 1.47 | 537 | 187 |
| S_batch adaptive RAG | 1.12 | 0.62 | 0 | 1.29 | 421 | 141 |
| **S full system** | 1.39 | 0.77 | 0 | 1.28 | 423 | 141 |
| S_batch − verification | 1.12 | 0.62 | 0 | 1.00 | — | — |

* **The full system costs about 1.3 times the LLM calls and prompt tokens of a single-shot baseline.** The extra comes
  from repair and regeneration calls and from streaming drafts. Output tokens are unchanged.
* **Adaptive retrieval removes about half the embeddings of fixed streaming** (0.77 vs 1.52) and cuts LLM calls from
  1.47 to 1.28.
* **Cancellation halves LLM calls when users correct themselves mid-flight** (Experiment 6: 20 → 10 calls for the
  same 15 turns).
* **Network and database requests.** The only network calls are local HTTP calls to Ollama, one per LLM call. There
  are no database requests: the index is in process.
* **Converting to currency** for a hosted deployment would need a priced model and a tokens × price formula. That is
  not done here.

## 28. Negative Results

Each item is measured; the sections hold the numbers.

1. **The full system is not the most accurate system.**
   * Answer correctness: S 0.730 vs B hybrid 0.892 and A naive 0.851. Against naive: 13 turns lost, 4 gained,
     p = 0.049.
   * Recall@5: 0.908 vs 0.994 (§8).
2. **Streaming lowers answer quality.**
   * Against batch mode in the same runtime: 7 turns lost, 0 gained, p = 0.016.
   * The validated answer is not faster at the median (+124 ms at the mean) (§14).
3. **Evidence verification lowers measured answer correctness.**
   * 0.797 with verification vs 0.865 without (6 vs 1 discordant turns, p = 0.125).
   * The verifier rejects true temporal claims; recall of SUPPORTED is 0.896 (§18, §23).
4. **Multi-hop retrieval did not help on the held-out corpus.**
   * Adaptive with multi-hop: 3 / 7 correct; fixed k = 5: 6 / 7. The hop fired on 1 of 7 items (§16).
5. **Adaptive k lowers recall on the held-out corpus.** Recall@5 0.915 vs 0.963 at fixed k = 5 (p = 0.031), although
   ranking and precision improve (§9).
6. **Reuse has a quality cost.**
   * Cache reuse lost 1 turn (S01.2).
   * Delta retrieval lost 2 turns.
   * Both gained 0 (§11, §12).
7. **The runtime ablations of delta retrieval and cancellation show no effect** on sequential conversations. The
   `session.full_restart` switch did not change runtime retrieval (§12, §18).
8. **No grounded system handles unanswerable questions well:** 1 / 5. Without an answerability check they answer
   adjacent facts (§20).
9. **Reranking adds about 34–56 ms per search with no answer-quality gain**, both in hybrid baseline C (vs B) and in
   the adaptive pipeline. It does improve MRR for the hybrid baseline (+0.089).
10. **Removing session memory costs little quality** (−1 net turn), but leaves 5 more turns without a validated
    answer.
11. **LLM timeouts are not recovered.** With every LLM call timing out, 0 / 10 turns get an answer. The degraded mode
    is declared, but there is no extractive fallback on this path. LLM *errors* do fall back, 10 / 10 (§25).

## 29. Limitations

**Data**
* Every corpus is a small fictional fixture: 43 chunks in the test corpus. Retrieval is easy at this scale, because
  k = 5 covers 12 % of the whole index, so retrieval differences between systems are compressed.
* One person wrote the system, the documents, the questions and the labels. There is no second annotator and no
  agreement statistic.
* Questions are typed, templated English. ASR revisions are scripted, not produced by a recogniser.
* Category sizes are 5–8 turns, so per-category numbers are anecdotal. VERY_HARD has no samples.

**Instrument**
* Answer correctness uses literal key strings, so a correct paraphrase can count as wrong. No LLM judge or human
  study backs the label metrics.
* Verifier-judged metrics (claim support, faithfulness, groundedness, citation entailment) use the proposed system's
  own Phase 7 verifier. Systems that already filter their claims with this verifier score close to 1 on these
  metrics by construction. Use the label metrics and the hallucination metric for cross-system grounding comparisons.
  §23 quantifies this bias.

**Measurement**
* Each variant was run once on one laptop with a 4B local LLM.
* Latency includes Ollama queueing on a shared machine. The run-to-run spread is measured in §19, but not the spread
  across machines.
* The cancellation experiment uses a SYNTHETIC remote-index delay (300 ms per dense search) for half of its variants.
* Robustness faults are SYNTHETIC (FaultInjector).
* Not measured: GPU memory, network, and database servers. The LLM runs locally and the indexes are in process.

**Statistics**
* Tests are paired, two-sided and not corrected for multiple comparisons. Read the p-values as descriptive.
* Bootstrap CIs resample turns, which ignores the clustering of turns within conversations (15 multi-turn sessions).

**Framework changes made during Phase 10**

These are evaluation-framework fixes and disclosures only. No system behaviour was changed after test outputs were
seen.

1. **Runtime turn evidence (the largest fix).**
   * The runtime harness read each turn's evidence from the session ledger *after the whole session*. Earlier turns
     of a conversation therefore got the later turn's evidence, or none. Answers were unaffected, but retrieval,
     evidence and citation-validity metrics of runtime variants were wrong. The bug was found while checking the
     first EXP01 results: S08.1 cited `FARES-2026 §1` while its recorded evidence was empty.
   * The fix: evidence is now read from the turn's own `EVIDENCE_FUSED` events (`systems.turn_evidence`). The
     stored runs were repaired from their event traces (`experiments/runners/rederive_runtime_evidence.py`;
     originals kept as `*.pre_evidence_fix.jsonl`; 6–19 turns changed per variant) and re-scored.
   * A first version of the fix also counted draft-time evidence the final answer never used. It was tightened
     before re-scoring.
   * The reproducibility re-run used the fixed code directly and matched the repaired runs on every turn (§19).
2. **Experiment 6 final-turn view.** It was added after the first EXP06 results were seen, because the overlap
   protocol makes the superseded first questions unanswered by design. Both views are reported.
3. **Experiment 6 wasted work.** Total worker time is reported as the fair measure, because the "wasted" counter
   cannot flag superseded work that ran to completion.
4. **Flaky Phase 8 cancellation test.** A fixed sleep was replaced with a wait for the running LLM task.
5. **Unintended reranking.** Fixed systems silently reranked because the shared stacks load the reranker. Variants
   now default to `streaming.rerank: false` unless they enable it.
6. **LLM routing.** Runtime extractive variants received the LLM. Now only `generation: llm` variants receive it.
7. **Sample ids.** Dev and test ids collided. Dev ids are now prefixed; the test file is byte-identical.
8. **Leakage check.** It raised a false near-duplicate alarm by comparing front matter. It now compares body text
   only.
9. **Failure store.** Records did not carry their trace path, and the runner test wrote into the repository. The
   store now follows the results directory. Memoryless runtime turns now get their own trace file; before this,
   later turns overwrote the session's trace.
10. **Metric documentation.** The abstention metric's docstring did not say that an empty answer counts as
    abstaining. The behaviour was unchanged; only the documentation was corrected.
11. **Charts.** Layout fixes, and distribution percentiles now use the same definition as the tables.
12. **New switch.** `session.full_restart` was added as a configuration switch for the delta ablation (default
    `false`, no behaviour change).
13. **Run reuse outside the repository.** Run reuse crashed when the results directory was outside the repository,
    as in the runner test's temporary directory. Paths are now logged relative to the repository when possible.
14. **Anti-hardcoding guard scope.** The "benchmark-specific branch" pattern scan (`tests/test_no_hardcoding.py`) now
    skips the measurement harness `src/streamrag/evaluation/`, which must read the dataset label field
    `expected_answer`. The harness is still scanned for embedded evaluation strings and guide vocabulary, and no
    system module may import it.

## 30. Research Findings

### Final comparison table

Brief §69. Test split, n = 87. Source: `experiments/results/final_comparison.json`.

| metric | Naive RAG | Hybrid | Adaptive RAG (batch) | Full system |
|---|---|---|---|---|
| Recall@5 | 0.976 | 0.994 | 0.915 | 0.908 |
| Recall@10 | 0.976 | 0.994 | 0.915 | 0.908 |
| MRR | 0.867 | 0.836 | 0.900 | 0.891 |
| Answer correctness (labels) | 0.851 | 0.892 | 0.797 | 0.730 |
| Claim support (verifier-judged; biased, §23) | 0.824 | 0.861 | 1.000 | 1.000 |
| Groundedness (supported and cited; verifier-judged) | 0.824 | 0.861 | 1.000 | 1.000 |
| Hallucinated claim rate (model-free) | 0.018 | 0.012 | 0.012 | 0.014 |
| Citation quality: precision (verifier-judged) | 0.825 | 0.843 | 1.000 | 1.000 |
| TTFE from first chunk, p50 / p95 ms | NOT MEASURED | NOT MEASURED | NOT MEASURED | 261 / 767 |
| TTFA from first chunk, p50 / p95 ms | NOT MEASURED | NOT MEASURED | NOT MEASURED | 271 / 1,632 |
| TTVA from first chunk, p50 / p95 ms | NOT MEASURED | NOT MEASURED | NOT MEASURED | 2,367 / 5,561 |
| TTFE after utterance end, p50 / p95 ms | 3 / 4 | 5 / 8 | 13 / 29 | NOT MEASURED (evidence arrives before the end) |
| TTVA after utterance end, p50 / p95 ms | 1,784 / 4,450 | 1,844 / 3,657 | 1,621 / 5,763 | 1,690 / 4,833 |
| Total latency after utterance end, p50 / p95 ms | 1,784 / 4,450 | 1,844 / 3,657 | 1,621 / 5,763 | NOT MEASURED |
| Retrieval calls / query | 1.00 | 1.00 | 1.12 | 1.39 |
| Cache hit rate | 0.000 | 0.000 | 0.034 | 0.023 |
| Failure rate (failed samples) | 0.000 | 0.000 | 0.000 | 0.000 |

**Notes on the NOT MEASURED cells.**
* The batch systems have no transcript stream, so first-chunk latencies do not apply to them.
* The full system's "total" includes waiting for session completion, so it is not comparable and is left as NOT
  MEASURED.

### Research questions

| RQ | answer (fixture data, test split unless stated) | hypothesis |
|---|---|---|
| **RQ1** Does adaptive retrieval improve the quality–efficiency tradeoff? | **Partly.** Ranking improves (MRR +0.085, p = 0.006) and evidence precision triples. Embeddings fall by 39% (extractive) and 49% (runtime). Searches rise (+0.09 per query, p = 0.035), and Recall@5 falls (−0.049, p = 0.031). Extractive answer correctness rises from 0.703 to 0.784 (not significant). With the LLM, the adaptive systems are less accurate than hybrid top-5 on this small corpus. | H1 partly supported: precision and embeddings yes, recall no |
| **RQ2** Does claim-driven retrieval improve grounding? | **Weak evidence, not significant.** +4 / −0 correct turns in batch, +2 / −0 in the runtime, and 2 fewer stale values in the runtime. Verifier-judged claim support cannot discriminate: both arms score 1.000. | H2 not established (too few discordant pairs) |
| **RQ3** Does delta retrieval reduce redundant work? | **Yes, in the batch pipeline.** −0.25 retrieval calls per conversation turn on test (p = 0.031) and −0.33 on dev (p = 0.016), with 0.5 evidence items reused per turn. The cost is 2 / −0 turns of answer correctness. The runtime ablation shows no effect. | H3 supported for work; the "quality unchanged" part was not met |
| **RQ4** Does cancellation reduce wasted computation? | **Yes, when users correct themselves mid-flight.** Same inputs: −30% total worker time and −50% LLM calls, and the corrected answer arrives about 1.1 s sooner at the median. Final-answer correctness is unchanged (5 / 10 both). Superseded questions go unanswered by design. No effect on sequential turns. n = 15, so no tests. | H4 supported (descriptive) |
| **RQ5** Does streaming reduce perceived response latency? | **Yes for first evidence and first answer content, no for the validated answer.** TTFE −442 ms and TTFA −2,177 ms (p < 0.001). A draft with all expected facts appears at a median of 273 ms from the first chunk, against 2,222 ms in batch. TTVA is +124 ms (p < 0.001), and answer correctness is lower (−7 turns, p = 0.016). | H5: first part supported; "validated answer earlier" rejected |
| **RQ6** Does session memory reduce retrieval repetition? | **A little.** Cache hits rise from 0 to 9.4% (test) and 20% (dev) of conversation turns. Dev retrieval calls fall by 0.20 per turn (p = 0.031). Without memory, the runtime has 5 more unanswered turns. Reuse cost 1 turn of correctness. | H6 supported (small effect) |
| **RQ7** Does contradiction-aware retrieval improve reliability? | **Yes in batch mode.** Stale values 0 / 5 vs 2 / 5 without it and 5 / 5 with fixed retrieval; correctness 8 / 8 vs 6 / 8. It is lost in streaming: S asserts stale values in 2 / 3 TEMPORAL turns. n ≤ 8, so no tests. | H7 supported in batch mode (descriptive) |
| **RQ8** What is the latency cost of stronger verification? | **Small in time, real in quality.** Verification itself costs 2 ms at p50 and 102 ms at p95 per answer. The verified path's repair calls (+0.29 LLM calls per query) make TTVA +55 ms at the mean (CI [21, 89], p = 0.004), about 3% of TTVA. It also removes true claims: answer correctness −0.068 (n.s.). | H8 supported for latency; the quality cost was not hypothesised |

### What the evidence supports

**Confirmed (measured, robust):**
* streaming produces first evidence and first answer content seconds earlier;
* adaptive retrieval ranks better and hands far less noise to the answer stage with fewer embeddings;
* claim verification with validated citations removes unsupported claims and mis-citations: 15.3% and 12% of claims
  without it, p < 0.001;
* contradiction- and temporal-aware retrieval removes stale values in batch mode;
* cancellation halves LLM work when users correct themselves;
* the whole evaluation is reproducible: identical labels on re-run, about 3% latency noise.

**Not confirmed:**
* the full system's answer accuracy is not better than simple baselines on this corpus;
* streaming does not deliver the validated answer sooner and costs accuracy through early commitment;
* multi-hop retrieval did not generalise to the held-out corpus.

**Most promising direction:** keep streaming for drafts and evidence, but re-decide retrieval and resolution on the
final transcript before validation. In this data, that removes every one of the 7 streaming-only failures, since
batch mode answers them correctly.

## 31. Engineering Decisions

* **One stored run per variant, shared by every experiment.** The same full-system run feeds Experiments 1, 6 and 7
  and the ablation, so all tables agree. Scoring is separate from running: stored raw runs are re-scored without
  re-running the system.
* **One instrument for every system.** Answers are split into claims, judged against the system's own retrieved
  evidence, and checked against label key strings. Label metrics are model-free and deterministic. The
  verifier-judged metrics are kept, but their bias is reported.
* **Extractive variants for retrieval-only questions** (Experiments 2, 4, 5, 8 and 9). They remove LLM variance, so
  retrieval effects are measured exactly and quickly.
* **Latency reported two ways:** from the first chunk (streaming view) and from the utterance end (comparable with
  batch systems). TTVA counts only the committed, validated answer.
* **Wasted work measured from `TASK_*` events.** Executed milliseconds of tasks whose query or answer was cancelled,
  superseded or stale are counted as waste. Waste is never estimated.
* **A minimum n for every percentile and test.** p90 needs n ≥ 10, p95 n ≥ 20 and p99 n ≥ 100. A test needs 20 pairs
  and 6 non-zero differences. Below that the result is NOT MEASURED.
* **No LLM judge and no currency.** Cost is reported as operation and token counts only.
* **Dependency-free SVG charts**, so the package needs no plotting stack. Charts follow the validated data-viz palette
  in light and dark mode, and each mark has a tooltip.

## 32. Phase 11 Requirements

These are prerequisites and inputs for Phase 11. Nothing here was implemented.

1. **Official corpus.** Obtain the Theme 4 corpus (still NOT AVAILABLE) and re-run
   `experiments/runners/run_experiments.py` on it with an independently annotated test split. All Phase 10 numbers
   are fixture results.
2. **Independent labels.** A second annotator for the test split, the human study in
   `docs/evaluation/human_evaluation.md` (sheet ready), and agreement statistics. Then validate or replace the
   key-string correctness check.
3. **Fixes indicated by the error analysis.** For Phase 11 to decide; none applied in Phase 10, because the test split
   is held out:
   * re-decide retrieval on the final transcript before validation (the 7 streaming-only failures);
   * fix the Phase 4 meta-conversation rule for "Which documents…";
   * add an answerability / insufficiency check;
   * drop pre-correction claims after entity corrections;
   * multi-hop detection on new domains;
   * recalibrate the verifier on temporal claims;
   * an extractive fallback when generation times out (§25).
4. **Wiring checks.** Verify that `session.full_restart` reaches the runtime's planner, then re-run the runtime delta
   ablation. Re-run the runtime experiments with traces kept per turn for memoryless runs.
5. **New measurements, then fresh test data.** Any fix must be measured on the dev split and then on a *new* held-out
   set. The Phase 10 test split has now been looked at and is no longer blind for those fixes.
6. **Statistical power.** Per-category conclusions need at least 20 items per category, which is about 4× the current
   size, and conversation-level bootstrap.
7. **Deployment and cost.** If a hosted LLM or deployment target is chosen: price per token and target hardware.
   Re-derive the latency gates and quality gates (`experiments/configs/quality_gates.yaml`) there.
8. **Regression gate.** Every change runs `experiments/runners/check_regression.py` against
   `experiments/results/regression_baseline.json`.

## Appendix A — Final quality gate (brief §73)

| gate | status | evidence |
|---|---|---|
| Evaluation dataset exists | ✅ | `experiments/datasets/streamrag_eval_v1` (test 87 / dev 129 turns), `docs/evaluation/dataset.md` |
| Query categories exist | ✅ | all 14 in test, 13 in dev |
| Baselines exist | ✅ | A naive, B hybrid, C hybrid + rerank, D streaming fixed (`experiments/configs/systems.yaml`) |
| Proposed system is evaluated | ✅ | `full_system` and `adaptive_rag`, Experiments 1–10 and the ablations |
| Retrieval / evidence / claim / generation / citation metrics exist | ✅ | `src/streamrag/evaluation/metrics/`, validated in `tests/evaluation/` |
| Latency / streaming / efficiency metrics exist | ✅ | the same modules; §21, §14, §22 |
| Robustness metrics exist | ✅ | `experiments/results/ROBUSTNESS`, §25 |
| Ablation framework exists | ✅ | `ABLATION_runtime`, `ABLATION_answer_stage`, §18 |
| Experiment runner exists | ✅ | `ExperimentRunner`, `experiments/runners/run_experiments.py` |
| Results are machine-readable | ✅ | `results.json`, `samples.jsonl`, `samples.csv` per experiment; raw runs and traces |
| Failure cases are stored | ✅ | `experiments/failure_cases/*.jsonl`, `experiments/results/qualitative_full_system.md` |
| Regression evaluation exists | ✅ | `check_regression.py`, `regression_rules.yaml`, `quality_gates.yaml`, `regression_baseline.json` |
| Reproducibility information exists | ✅ | `config_record.json` per experiment, `environment.json`, `REPRODUCIBILITY/`, `experiments/README.md` |
| Visualizations exist where meaningful | ✅ | 14 SVGs in `experiments/plots/` (brief §64 list + 3 frontiers + streaming curve), `experiments/results/dashboard.html` |
| No fabricated results exist | ✅ | every number traces to a stored result; NOT MEASURED wherever no measurement exists |
| All tests pass | ✅ | 645 passed (`.venv/bin/python -m pytest -q`), exit code 0 |
| Documentation exists | ✅ | `docs/evaluation/*.md`, `experiments/README.md`, `experiments/metrics/README.md`, this report |

## Appendix B — Final validation (brief §74)

| # | run | result | stored at |
|---|---|---|---|
| 1 | full test suite | 645 passed, 0 failed | — |
| 2 | baseline benchmark | COMPLETED, 6 systems × 87 turns, 0 failed samples | `results/EXP01_baselines` |
| 3 | adaptive benchmark | COMPLETED (test + dev) | `results/EXP02_adaptive_topk`, `EXP08_iterative`, `EXP09_multihop` |
| 4 | ablation experiments | COMPLETED, 10 variants × 87 turns, 0 failed samples | `results/ABLATION_runtime`, `ABLATION_answer_stage` |
| 5 | streaming benchmark | COMPLETED | `results/EXP07_streaming`, `runs/traces/` |
| 6 | cancellation benchmark | COMPLETED, 4 variants × 15 turns | `results/EXP06_cancellation` |
| 7 | cache benchmark | COMPLETED (test + dev) | `results/EXP04_cache`, `EXP05_delta` |
| 8 | robustness benchmark | COMPLETED, 8 scenarios × 10 + propagation × 5 | `results/ROBUSTNESS` |
| 9 | failure injection | COMPLETED (inside 8; SYNTHETIC faults); LLM-timeout scenario unrecovered | `results/ROBUSTNESS` |
| 10 | end-to-end benchmark | COMPLETED: the full system over all 14 query types, transcript stream → intent → query → adaptive retrieval → evidence → claims → generation → validation → citations → streamed answer | `results/EXP01_baselines` (`full_system`), traces |
| — | verifier validation | COMPLETED (172 perturbed claims) | `results/VERIFIER_validation` |
| — | reproducibility re-run | COMPLETED (4 variants) | `results/REPRODUCIBILITY` |
| — | resources / environment | COMPLETED | `results/RESOURCES`, `results/environment.json` |
| — | human evaluation | NOT RUN (protocol and blinded sheet prepared) | `docs/evaluation/human_evaluation.md`, `experiments/datasets/human_eval_sheet.csv` |

**Reproduce:** see `experiments/README.md`. The commands, in order:
1. `build_dataset.py`
2. `ollama serve`
3. `run_experiments.py --index-root …`
4. `verifier_validation.py`
5. `robustness.py`
6. `reproducibility.py`
7. `resources.py`
8. `environment.py`
9. `analyze.py`
10. `make_regression_baseline.py`

Runs made before the evidence fix need `rederive_runtime_evidence.py` before step 9. Runs made with the current code
do not.

