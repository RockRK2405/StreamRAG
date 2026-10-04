# Final Benchmark Results (Phase 11)

> **TEST FIXTURE ONLY — NOT REPORTABLE as official results.** The corpora are fictional and the labels were written
> by the implementer. The official Theme 4 corpus was not available.

* Every number here is generated from the stored files in this folder by `experiments/runners/final_tables.py`. The
  full tables, with every metric, CI and p-value, are in `tables.md`; the headline numbers are in
  `final_summary.json`.
* Phase 10 results (`experiments/results/`) were not overwritten.

## 0. Protocol and environment

| item | value |
|---|---|
| held-out set | `streamrag_eval_v2`: Lakeside utility services, 14 documents, **81 turns** (49 single, 14 sessions, 4 streamed corrections), all 14 query types. Written and hash-frozen **before** any Phase 11 change (`experiments/datasets/streamrag_eval_v2/FROZEN.sha256`). Run **once**. |
| development data | the dev split, and the v1 test split (Northvale transit, 87 turns), which was inspected in the Phase 10 error analysis |
| systems | 12 on v2. Baselines: A naive dense top-5; B hybrid; C hybrid + cross-encoder; D streaming with fixed retrieval. Then adaptive RAG (batch), the **full system**, its batch mode, and the answerability-flag variant. Extractive retrieval ablations: fixed top-5, adaptive, no cache, full re-retrieval. Cancellation on / off. |
| final system | `configs/default.yaml` + `configs/profiles/final.yaml` (multi-intent, session, adaptive retrieval, grounded generation; reranker off; answerability flag off) |
| LLM | `qwen3:4b` (Q4_K_M) via local Ollama, temperature 0, seed 7 |
| models | bge-small-en-v1.5 (ONNX), nli-deberta-v3-xsmall (ONNX); the cross-encoder only for baseline C |
| machine | Apple M5 Pro (Mac17,9), 15 CPU cores, 24 GB, macOS 26.5.1, Python 3.12.0 (`environment.json`) |
| statistics | paired by turn: exact McNemar (binary), Wilcoxon signed-rank (continuous), bootstrap 95% CI of the difference; no test below 20 pairs; no multiple-comparison correction |
| code | the v2 runs used the Phase 11 working tree, committed as `d130ad0`; only the UI static files changed after the runs started. One later change (§9) is shown not to affect any stored answer. |

## 1. Held-out v2: final system vs baselines

Means over turns; latency p50 in ms.

| metric | A naive | B hybrid | C hybrid + rerank | D streaming fixed | adaptive (batch) | **full system** | full system, batch | + answerability flag |
|---|---|---|---|---|---|---|---|---|
| Recall@5 | **0.994** | 0.974 | 0.961 | 0.981 | 0.935 | 0.883 | 0.935 | 0.883 |
| MRR | 0.845 | 0.853 | 0.883 | 0.878 | **0.900** | 0.862 | **0.900** | 0.862 |
| evidence precision | 0.247 | 0.244 | 0.242 | 0.242 | 0.551 | **0.552** | 0.551 | **0.552** |
| answer correct | 0.761 | 0.747 | **0.803** | 0.620 | 0.732 | 0.718 | 0.732 | 0.662 |
| stale / forbidden value (n = 13) | 0.231 | 0.308 | 0.154 | 0.538 | **0** | **0** | **0** | **0** |
| insufficiency handled (n = 4) | **0.50** | 0.25 | 0.25 | 0.25 | 0.25 | 0.25 | 0.25 | **0.50** |
| hallucinated-value claim rate | 0.070 | 0.093 | 0.062 | **0** | **0** | **0** | **0** | **0** |
| claim support (verifier, biased) | 0.835 | 0.790 | 0.825 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| TTFE from first word | after end | after end | after end | 256 | after end | **264** | 767 | 264 |
| TTFA from first word | after end | after end | after end | 285 | after end | **271** | 2,477 | 271 |
| TTVA after utterance end | **1,411** | 1,440 | 1,489 | 2,275 | 1,693 | 1,643 | 1,659 | 1,648 |
| TTVA after end, p95 | **3,191** | 3,851 | 4,198 | 3,874 | 4,189 | 3,989 | 3,827 | 3,614 |
| retrieval calls / turn | **1.00** | **1.00** | **1.00** | 1.77 | 1.35 | 1.86 | 1.35 | 1.86 |
| embeddings / turn | 1.00 | 1.00 | 1.00 | 1.77 | **0.83** | 1.06 | **0.83** | 1.06 |
| LLM calls / turn | **1.00** | **1.00** | **1.00** | 1.37 | 1.28 | 1.27 | 1.28 | 1.40 |
| failed turns | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

**Paired comparisons, full system vs each baseline** (p-values in `tables.md` §3):

* **Answer correctness: no significant difference.**
  * vs naive: −0.042, p = 0.68;
  * vs hybrid: −0.028, p = 0.84;
  * vs hybrid + rerank: −0.085, p = 0.24;
  * vs D: +0.099, p = 0.14.
* **Grounding: significantly better.**
  * Hallucinated-value rate vs A / B / C: −0.072 / −0.096 / −0.061 (p = 0.008 / 0.001 / 0.008).
  * Verifier claim support +0.16 to +0.22 (p < 0.001; biased instrument).
* **Evidence precision: +0.31 vs every baseline** (p < 0.001).
* **Recall@5 is lower.**
  * vs naive: −0.110 (p = 0.004);
  * vs hybrid: −0.091 (p = 0.03).
* **Latency.**
  * TTFA vs batch mode: −2,266 ms (mean, p < 0.001).
  * TTVA after the utterance end: not significantly different from naive RAG (+153 ms mean, p = 0.58). It is 66 ms
    *shorter* than batch mode (mean, p < 0.001).
* **Work.**
  * Streaming costs +0.86 retrieval calls and +0.27 LLM calls per turn vs naive RAG (p < 0.001).
  * The answerability flag costs +0.12 LLM calls.

**Special cases** (n is small; descriptive):

| case | naive | hybrid | hybrid + rerank | D | full system |
|---|---|---|---|---|---|
| conflicting notices: both values reported (n = 2) | 2 / 2 | 2 / 2 | 2 / 2 | 2 / 2 | 2 / 2 |
| superseded document: current value, no stale value (n = 2) | 2 / 2 | 1 / 2 | 0 / 2 | 0 / 2 | 2 / 2 |
| temporal: stale value asserted (n = 2, lower is better) | 1 / 2 | 1 / 2 | 1 / 2 | 2 / 2 | **0 / 2** |
| unanswerable: abstained without inventing (n = 4) | 2 / 4 | 1 / 4 | 1 / 4 | 1 / 4 | 1 / 4 |

## 2. Ablations on v2

| change | effect | test |
|---|---|---|
| streaming → batch mode of the same pipeline | TTFE p50 264 → 767 ms; TTFA p50 271 → 2,477 ms; TTVA after end p50 1,643 vs 1,659 ms; answer correctness 0.718 vs 0.732 (batch alone right on 2 turns, streaming alone on 1); Recall@5 0.883 vs 0.935 | TTFA p < 0.001 |
| adaptive → fixed top-5 (extractive) | evidence precision 0.551 → 0.240 (p < 0.001); stale values 0 → 7 / 13; embeddings 0.83 → 1.06 (p = 0.007); retrieval calls 1.35 → 1.06 (p = 0.006); Recall@5 0.935 → 0.981 (p = 0.22); correctness 0.718 → 0.662 (p = 0.39) | |
| caches → none (extractive) | retrieval calls 1.35 → 1.42 per turn (p = 0.031); cache hits 7.4% → 0; quality identical | |
| delta → full re-retrieval (extractive) | retrieval calls 1.35 → 1.49 (p = 0.016); quality identical | |
| answerability flag on | insufficiency 1 → 2 of 4; correctness 0.718 → 0.662 (5 vs 1 discordant, p = 0.22); TTVA after end +103 ms (mean, p < 0.001); LLM calls +0.12 | flag stays **off** |

**Cancellation on overlapping corrections** (HELDOUT_V2_CANCELLATION; the same 14 inputs for both variants;
descriptive):

| | total worker ms | LLM calls | cancelled tasks | final turns correct | final turns: TTVA after end p50 / max | superseded first turns answered |
|---|---|---|---|---|---|---|
| cancellation off | 21,755 | 15 | 0 | 5 / 9 | 1,656 / 2,818 ms | 5 / 5 |
| cancellation on | **15,972 (−27%)** | **9** | 5 | 5 / 9 | **1,509 / 2,165 ms** | 0 / 5 (by design) |

## 3. Theme 4 gates (from the full system's v2 traces)

| gate | definition used | measured | status |
|---|---|---|---|
| G1 reproducibility | `docker compose up`, keyless replay | image builds; healthy in 8 s; `demo-check` 10 / 10 in the container without an LLM; temperature 0, seed 7; Phase 10 measured identical labels on re-run | met |
| G2 early retrieval ≥ 80% | first retrieval starts before the utterance is finalised; eligible = needs retrieval and ≥ 2 chunks | **76 / 77 = 98.7%**. In 39 of those turns the final transcript needed no new search. | met |
| G3 multi-intent ≥ 70% | intent count = labelled count on MULTI_INTENT turns | 4 / 4; answer correct 4 / 4 (small n) | met (n = 4) |
| G4 grounding ≥ 85%, no fabricated IDs | verifier claim support; citations not resolving to a corpus section | claim support 1.000 (biased instrument); **0 of 173** citations unresolvable; model-free hallucinated-value rate 0 | met |
| G5 session refinement | correctness on follow-up / correction / repeat turns | 0.611 streaming vs 0.722 batch (n = 18) | **weak**: see §4 |
| G6 telemetry 100% | events with all required fields, in order; utterances completed with a validated answer | 10,562 / 10,562 events; 81 / 81 utterances | met |

Guide requirements beyond the gates:
* a baseline: four, plus batch and ablation variants;
* ≥ 3 edge-case failures: §4;
* ≥ 2 ablations: §2, plus the Phase 10 ablations.

## 4. Error analysis on v2 (full system)

Error categories (turns; a turn can have several):

| category | turns |
|---|---|
| GENERATION | 17 |
| QUERY_ANALYSIS | 13 |
| RETRIEVAL | 6 |
| ENTITY | 4 |
| EVIDENCE | 2 |
| MEMORY | 2 |
| CLAIM / CITATION | 0 |
| no error | 50 of 81 |

These are the main failure patterns. v2 is held out, so none of them was fixed.

1. **Elliptical follow-ups are rewritten with the wrong slot.**
   * VS04.2, "And in the south?", was searched as "When are bins collected in Service Area North".
   * Batch mode fails the same way.
2. **Entity corrections lose the original question.**
   * VS06.2, "Sorry, I meant Fernhill, not Oakridge", was searched as "Fernhill". The answer gives the district, not
     the collection day.
   * VS07.2 failed the same way.
3. **Streaming-only session failures.**
   * VS01.2, "And for households on income support?": the previous turn's claim was kept, with "not established" for
     the new need. Batch mode answered (40%).
   * VS10.2, "I meant garden waste, not household bins": no answer. Batch mode answered.
   * These are the two turns behind the G5 gap.
4. **Adjacent fact instead of the asked value, or of "not in the sources".**
   * 3 of 4 unanswerable questions got an adjacent true fact.
   * VS03.2, "Do I get that back?", got the relief fact, not the refund rule.
5. **Multi-constraint and multi-hop recall.**
   * Multi-constraint: 3 / 8 correct (naive 8 / 8). Of the 5 failures, 2 are retrieval misses (gold section not in
    the evidence) and 3 are generation misses (evidence present, key fact not stated).
   * Multi-hop: 4 / 7 (hybrid + rerank 6 / 7). VSC1, "How long does a decision take for pensioners?", retrieved the
     connection document.

Per-category correctness for every system is in `tables.md` §4. The failure records (system, turn, expected, answer,
categories, trace path) are in `failure_cases/HELDOUT_V2.jsonl`.

## 5. Regression vs Phase 10 (v1 test split; development data, so improvements are optimistic)

The full system, compared with Phase 10 under the rules in `experiments/configs/regression_rules.yaml`:

| metric | Phase 10 | Phase 11 | status |
|---|---|---|---|
| answer correct | 0.730 | **0.797** | improvement |
| stale / forbidden value | 0.231 | **0.000** | improvement |
| Recall@5 / MRR | 0.908 / 0.891 | 0.933 / 0.916 | ok |
| evidence precision | 0.669 | 0.716 | improvement |
| hallucinated-value rate | 0.014 | 0.010 | ok |
| TTVA after end: mean / p50 / p95 | 2,052 / 1,690 / 4,833 ms | **1,744 / 1,261 / 3,531 ms** | ok / improvement |
| retrieval calls / turn | 1.391 | 1.414 | "regression" (tolerance 0); explained below |
| LLM calls / turn | 1.276 | 1.253 | improvement |
| streaming vs batch mode: turns lost | 7 | **2** | |

* **The retrieval-call increase is intended.**
  * T25 and S05.1 ("Which documents do…") were skipped as meta-conversation in Phase 10 and now retrieve: +3 calls.
    Both turned from wrong to right.
  * S05.2 now reuses evidence: −1 call.
  * Net +2 calls over 87 turns.
* **Adaptive RAG (batch):**
  * correctness 0.797 → 0.824;
  * TTVA after end p95 5,763 → 4,090 ms;
  * stale values 0 → 0;
  * retrieval calls 1.115 → 1.138, also from the meta-question fix.

**Disclosure: one regression run was repeated.**
* The first full-system v1 run measured a TTVA after the end of 3,161 ms mean and 7,320 ms p95, a regression.
  Investigation of the Ollama server's own request log showed this:
  * from 16:59 to 17:09, the server's median time per `/api/chat` request rose from about 1.1 s to 2.4–4.2 s;
  * the rise covered the whole of that run (median 2.07 s vs 1.06 s in the development run one hour earlier);
  * the requests were the same, with fewer LLM calls per turn (1.25);
  * no other job from this session was running. The cause outside the system is not established.
* The run was repeated under normal server timing (median 1.09 s per request): mean 1,744 ms, p95 3,531 ms. Quality
  was identical (0.797).
* Both runs are kept. The first is in `regression_v1/first_run_slow_llm_window/`, with its own `results.json` and
  `regression_report.json`. The table above is the repeated run.
* The held-out v2 runs finished before the slowdown, at normal server timing (per-minute medians 1.05–1.86 s).

## 6. Robustness (fault injection on the final configuration, 10 runs each)

| fault | recovered | degraded but correct | incorrect answer |
|---|---|---|---|
| retrieval failure | 10 / 10 | 10 / 10 | 0 |
| vector index failure | 10 / 10 | — | 0 |
| network transient | 10 / 10 | — | 0 |
| **LLM timeout** | **10 / 10** (Phase 10: 0 / 10) | 10 / 10 | 0 |
| LLM error | 10 / 10 | 10 / 10 | 0 |
| verifier failure | 10 / 10 | 10 / 10 | 0 |
| malformed query | 10 / 10 | — | 0 |
| rapid updates | 10 / 10 | — | 0 |

"Degraded" means a declared degraded mode: lexical-only, extractive or rules-only.

## 7. Demo check (`streamrag demo-check`, final code)

* **With the LLM** (`demo_check.json`): **10 / 10** scenarios pass their declared expectations, with every turn
  VALIDATED_FINAL.
  * TTFE 7–14 ms from the first chunk where the first chunk is already a question (A, B, C, D2, E, F1, F2).
  * Verified answer 2.1–5.9 s from the first chunk, including the time the scripted speech takes.
* **Without the LLM** (`demo_check_no_llm.json`): **10 / 10**, with extractive verified answers.
* **Docker** (Phase 11 deployment check): healthy in 8 s; 10 / 10 without an LLM in the container; 3 / 3 checked
  scenarios with the host's LLM.
* **Earlier run.** `demo_check_before_ielts_fix.json` is the LLM run before the §9 change. It also passed 10 / 10, but
  two answers showed the model's misspelling "IEL:TS".

## 8. Performance profile (§20)

Wall time per stage, from the full system's v2 traces (`tables.md` §13) and component micro-benchmarks
(`resources/results.json`):

| stage | p50 | p95 | note |
|---|---|---|---|
| controller decision per chunk | 0.9 ms | 1.4 ms | rule-based, no model |
| retrieval task (adaptive plan, incl. BM25 / dense) | 9.5 ms | 20.3 ms | hybrid top-5 alone: 3.6 ms wall, 24.7 ms CPU |
| draft (extractive + verification) | 3.7 ms | 21.4 ms | |
| LLM generation (qwen3:4b, Metal) | 1,491 ms | 2,974 ms | **dominant cost** |
| final answer task (LLM + verification + citations + repair) | 1,637 ms | 3,965 ms | |
| claim verification, 1 claim vs 5 sections | 19 ms | 58 ms (p90) | batch pipeline per answer: 4 ms p50, 49 ms p95 |
| cross-encoder rerank of top-5 (off) | 29 ms wall | 262 ms CPU | |
| queue wait, any pool | < 0.15 ms | | no queueing at one session |

**Memory.**
* Process with the index, BM25, embedder and NLI: 1,020 MB peak RSS.
* LLM: 3.9 GB resident (Ollama, Metal unified memory).
* Models on disk: bge-small 128 MB, NLI 279 MB.

**Latency before / after Phase 11** (v1, the same machine, normal LLM timing):
* TTVA after the end, p50: 1,690 → 1,261 ms. p95: 4,833 → 3,531 ms.
* The cause is not isolated. LLM calls per turn fell only slightly (1.276 → 1.253). The Phase 10 session's LLM server
  timing was not logged, so part of the difference may be environmental (§5 shows how much that matters).
* No latency optimisation is claimed beyond this measurement. TTFE / TTFA were already under 300 ms in Phase 10.

## 9. Development iterations (before → after, development data only)

| change | data | before | after | decision |
|---|---|---|---|---|
| streaming fixes (G1–G3, G5), flag off | v1, runtime | correctness 0.730, stale 3 / 13, TTVA p50 1,690 ms | 0.797, 0 / 13, 1,254 ms | kept |
| answerability flag | v1, batch | correctness 0.824, insufficiency 1 / 5 | flag on: 0.784, 2 / 5 | **off** |
| answerability flag | dev, batch | insufficiency 6 / 14 | flag on: 7 / 14 | **off** |
| answerability flag | v1, runtime | 0.797 | flag on: 0.757 | **off** |

**Post-benchmark change** (after the v2 runs), found in the demo check:
* `qwen3:4b` wrote "IEL:TS" for "IELTS", and verification accepted it as entailed.
* Generated codes and acronyms now get the evidence spelling when they differ only in punctuation
  (`claims/textcheck.py::canonical_codes`, with a test).
* Letters and digits never change.
* A scan of all 5,196 stored benchmark answers (v1, v2, dev; every system) found **no** such token. The change
  therefore cannot alter any reported answer.

## 10. Negative results and disclosures (summary)

* The full system is **not** more accurate than the baselines on v2: 0.718 vs 0.747–0.803, all n.s.
* Its Recall@5 is the lowest of all systems on v2 (0.883).
* Its verified answer is not faster than naive RAG's after the user stops.
* Follow-up and correction turns are weak in the new domain (G5).
* Only 1 of 4 unanswerable questions was recognised.
* The reranked hybrid baseline was the strongest on v2 answer correctness. The reranker stays off, because re-tuning on
  v2 would break the held-out protocol.
* Verifier-judged metrics are biased in favour of the verified systems.
* One environment-affected run was repeated (§5). One post-benchmark code change (§9).

## Reproduce

Every step needs `ollama serve` with `qwen3:4b`, except the tables. Use a fresh results directory, so the stored
results stay untouched. First the held-out benchmark:

```bash
.venv/bin/python experiments/runners/run_experiments.py --index-root /tmp/streamrag_idx --dataset streamrag_eval_v2 --results-dir runs/final_benchmark --experiments-file experiments/configs/final_benchmark.yaml --only HELDOUT_V2,HELDOUT_V2_CANCELLATION
```

Then the regression check on v1:

```bash
.venv/bin/python experiments/runners/run_experiments.py --index-root /tmp/streamrag_idx --dataset streamrag_eval_v1 --results-dir runs/regression_v1 --experiments-file experiments/configs/final_benchmark.yaml --only REGRESSION_V1
```

Then robustness and the demo check:

```bash
.venv/bin/python experiments/runners/robustness.py --index-root /tmp/streamrag_idx --out runs/robustness
```

```bash
.venv/bin/streamrag demo-check
```

Finally, rebuild the tables from the stored results in this folder:

```bash
.venv/bin/python experiments/runners/final_tables.py
```
