# Samsung PRISM Theme 4
# Phase 9 — Adaptive Retrieval Intelligence

| | |
|---|---|
| Date | 2026-10-04 |
| Status | Phase 9 complete. **OFFICIAL_CORPUS_STATUS = NOT_AVAILABLE** (unchanged). Not committed (commit on request). Adaptive retrieval is **off by default** (`adaptive_retrieval.enabled: false`). |
| Code | New package `src/streamrag/adaptive/` (14 modules, about 2,400 lines): models, lexicon, values, catalog, analyzer, rewrite, requirements, sufficiency, policy, stopping, cache, controller, integration. Extensions: document metadata (`corpus/metadata.py`, `CorpusDocument.metadata`, whitelist `corpus.metadata_fields`); `RetrievalFilters.metadata / valid_at / valid_to` and `RetrievalService` metadata masks; `AdaptivePipeline._execute_adaptive` (Phase 6); `RuntimeRetrievalExecutor._submit_adaptive` (Phase 8); `AdaptiveSessionEngine.retrieval_terms`; Phase 7 time values; 9 new event types; config `adaptive_retrieval`, `configs/retrieval_lexicon.yaml`. |
| Tests | **621 passing, 0 skipped**: 515 from Phases 3–8 and 106 new — `tests/adaptive_retrieval` 29 (the 18 cases of brief §58, security, measurement plumbing), `query_analysis` 10, `query_rewrite` 6, `routing` 12, `evidence_sufficiency` 8, `iterative_retrieval` 7, `multi_hop` 3, `cache` 8, `contradiction` 6, `budget` 7, `streaming_integration` 8, the isolation scan for `adaptive` (+1) and a Phase 7 regression test (+1, §24). No earlier test was changed. |
| Data | Fixture corpus `tests/fixtures/corpus_adaptive` (16 fictional documents with front-matter metadata, 46 chunks) + the Phase 3/7 fixture corpora. Dev set `eval/dev_adaptive_retrieval` (54 single-turn cases + 8 sessions = 72 turns); held-out set `eval/heldout_adaptive_retrieval` (20 + 2 sessions, written after the code freeze, run once); calibration set = Phase 7 `eval/dev_grounded` + Phase 3 fixture retrieval items. **All labels written by the implementer.** |
| Measurements | Real retrieval (bge-small ONNX, BM25, RRF, ms-marco cross-encoder), real NLI verifier, extractive answers for the system comparisons, **real local LLM (qwen3:4b, Ollama, temperature 0)** for the grounded-quality comparison (§20) and the final end-to-end test (§17). Injected remote-index latency in one H5 condition is labelled SYNTHETIC. Results: `research/phase9/results/`, tables `research/phase9/results/tables.md`. |

**Read this first: what the numbers mean.**

1. **Fixture domain, implementer labels, tiny corpora (6–60 chunks).** Nothing here is reportable as Theme 4 performance. Differences of one or two turns are within noise.
2. **The dev set is not blind.** While building the controller I ran the dev set and fixed the generic defects it exposed (each fix is listed in §24); the dev numbers are therefore optimistic. The routing defaults were calibrated on a *different* question set (§5), and a held-out set written after the freeze was run once (§18).
3. **Latency is milliseconds on a laptop with a local index.** Single-run latencies of identical configurations differed by up to ~70 % between runs, so the latency test (§21) uses 5 interleaved repetitions per turn.
4. **Hypotheses were tested, not assumed** (§2). H1 is **not supported** as stated.

**Quality gate (brief §72):**

| Gate (brief §72) | Status | Where |
|---|---|---|
| Query complexity analyzer exists | ✅ explainable classes, reasons | §4, docs/retrieval/01 |
| Retrieval policy selector exists; multiple strategies; selection logged | ✅ 9 strategies, `strategy_reason`, RETRIEVAL_POLICY_SELECTED | §5 |
| Query rewriting; context-aware rewriting | ✅ normalized / contextual / expanded / retrieval-specific; Phase 6 contextual need | §6 |
| Constraint-aware retrieval; temporal retrieval | ✅ metadata filters (user-stated), validity windows; publication ≠ effective date | §5, §9, §14 |
| Adaptive top-k | ✅ 3/5 → 10 → 20 | §7 |
| Evidence sufficiency evaluator; claim-driven retrieval | ✅ slots, 4 states | §8, §9 |
| Iterative retrieval; multi-hop where required | ✅ bounded loop; hops with `RetrievalHop` | §10, §11 |
| Stopping policy; retrieval budgets | ✅ 9 stop reasons; 6 budgets, Phase 8 deadline / cancellation | §15, §16 |
| Cache reuse; cache invalidation | ✅ (pipeline benefit small, H4) | §12, §13 |
| Evidence deduplication | ✅ chunk-id + alternates across searches, per-source cap | §10 |
| Contradiction-aware retrieval | ✅ detection, resolution, targeted third-source search | §14 |
| Streaming integration; delta retrieval preserved | ✅ Phase 6 pipeline + Phase 8 runtime (tests); e2e §17 / §20 | §17 |
| Baselines; ablation studies | ✅ A–D + adaptive; cumulative A–G + leave-one-out | §18, §19 |
| Actual measurements; no fake results | ✅ every number from `research/phase9/results/` (tables generated); NOT MEASURED stated where it applies | §18–23 |
| Tests pass | ✅ 621 / 621 | header |
| Documentation exists | ✅ docs/retrieval/01–11 (Phase 9 series), docs/architecture/13 (Mermaid, validated), ADR-019 | — |

## 1. Objective

Decide per need — not per system — *whether* to retrieve, *how* (strategy, retrievers, filters, k, reranking), *how much* (iterations, expansions, hops) and *when to stop*, so that the answer gets sufficient evidence with the least retrieval work. Research question of the phase: can the system choose the minimum retrieval computation that still yields a sufficiently grounded answer?

What Phase 9 is not: no new retriever or model, no LLM in the retrieval loop, no change to the Phase 7 verifier or the Phase 8 scheduler semantics.

## 2. Research Hypotheses

Defined before the measurements (brief §46) and tested on the dev set; verdicts use the measured numbers of §18–23.

| # | Hypothesis (brief §46) | Test | Result | Verdict |
|---|---|---|---|---|
| H1 | Adaptive retrieval reduces latency compared with always-on hybrid retrieval while maintaining answer quality | §21 interleaved latency test; §18 / §20 quality | retrieval-stage latency p50 **+15 %** (6.56 vs 5.70 ms), p95 **+151 %** (16.6 vs 6.6 ms); faster only on simple / exact / semantic needs; median CPU −79 %. Whole turn with the real LLM: p50 **−26 %** (1.85 vs 2.50 s), p95 −32 % (shorter prompts). Quality maintained (claim support 0.924 vs 0.930) or better (MRR 0.953 vs 0.869, gold cited 0.915 vs 0.900) | **Not supported for retrieval latency**; supported for end-to-end turn latency with the local LLM (one run) and for quality |
| H2 | Claim-driven retrieval improves evidence support | ablation C → D and leave-one-out; real-LLM C vs D (§20) | coverage 0.882 → 0.919, MRR 0.848 → 0.902, returned precision 0.559 → 0.683, gold cited 0.838 → 0.858 (+5 searches); LOO: removing it lowers coverage 0.968 → 0.931, gold cited 0.930 → 0.903; extractive claim support 1.000 in both (not discriminating); real-LLM C → D: claim support 0.932 → 0.929 (equal within noise), gold cited 0.823 → 0.843 | **Supported** on evidence-level metrics (fixture) |
| H3 | Adaptive top-k reduces unnecessary retrieval | LOO `adaptive_k`; vs fixed k=5 | at equal quality the adaptive schedule retrieves 420 instead of 504 chunks (−17 %), avg k 4.70 vs 5.87; against fixed hybrid k=5 the *system* retrieves 18 % more chunks (420 vs 357, extra rounds) but hands on ~3× more precise evidence (returned precision 0.773 vs 0.250) | **Partly supported**: fewer chunks than starting every need at k=5, not fewer than one fixed k=5 search |
| H4 | Cache-aware retrieval reduces repeated retrieval work | ablation D → E (sessions), LOO cache | sessions: 25 → 23 searches, 19 → 17 embeddings; whole set 94 → 92 searches; exact repeats were already absorbed by the Phase 6 semantic cache (2 hits in every system); adaptive query-cache hits in the pipeline: 0; SESSION_REUSE: 2 needs answered with 0 searches | **Supported, small** (2 of 25 session searches) |
| H5 | Cancellation + adaptive retrieval reduces wasted computation | §22 runtime, 2 × 2 configurations, local and +150 ms remote (SYNTHETIC) | wasted retrieval time vs fixed + cancellation: −70 % local (398 vs 1360 ms), −34 % remote (4956 vs 7478 ms) — all from adaptive retrieval; cancellation cancelled 0–1 tasks per configuration and saved ≈ 0 | **Supported for adaptive retrieval, not for cancellation** in this workload |

## 3. Existing Retrieval Limitations

Before Phase 9 every need ran the same pipeline (Phase 3, used unchanged by Phases 4–8): BM25 top-50 ‖ dense top-50 → RRF → dedup → [rerank] → top-k (5 per need in the session pipeline). Measured consequences on the Phase 9 dev set (fixed hybrid, `research/phase9/results/baselines.json`):

* **Same cost for every need.** One embedding + one BM25 search per need, whether the question names an identifier (BM25 alone suffices) or needs two documents chained.
* **Low evidence precision.** 5 chunks per need regardless of how many are relevant: returned-set precision 0.250 (one relevant chunk in four handed on).
* **No notion of "enough".** A need whose answer is not in the corpus gets the same 5 chunks as one that is answered; a need that needs a second document never gets it (multi-hop coverage 0.500).
* **No constraints or time.** "for domestic applicants" or "in December 2025" are just words in the query; metadata did not exist in the index; superseded documents rank like current ones.
* **No conflict handling at retrieval time.** Conflicting sources were only flagged after the fact (Phase 5 fusion, Phase 7 answer).
* **Reuse only by identical term sets** (Phase 6 semantic cache), without validity checks.

## 4. Query Complexity

`QueryComplexityAnalyzer` (docs/retrieval/01). Signals come from the need (Phase 5/6 intent, constraints, number of needs) and corpus *statistics* (IDF, co-occurrence, metadata catalog) — never from retrieved text. Classes and their rules:

| Class | Rule (stored in `reasons`) | Dev-set needs |
|---|---|---|
| MULTI_HOP | `entity_aspect_gap`: a capitalised / identifier rare term never co-occurs with the rest of the question | 5 |
| COMPLEX | several needs, comparison, ≥ 2 constraints, date + constraint | 9 |
| MODERATE | constraint, filter, temporal cue, context dependence, vocabulary mismatch, ambiguity, > 8 content terms | 19 |
| SIMPLE | otherwise | 39 |

Two analyzer defects found during development and fixed generically: question / value-phrase words ("how **high**", "**rules**") had triggered false MULTI_HOP gaps; a lower-case word rare in this corpus ("charge") had too. Both are now excluded (content terms; entity cue = capitalisation or identifier).

## 5. Retrieval Policy Engine

`RetrievalPolicySelector` = router (docs/retrieval/03). Nine strategies; each plan stores `strategy_reason`, retrievers, k, filters, reranking + reason, max_iterations, latency budget, evidence threshold, stop condition; RETRIEVAL_POLICY_SELECTED logs it. Routing order: cache → session evidence → MULTI_HOP → exact identifiers (LEXICAL) → user-stated constraint / explicit date (FILTERED) → vocabulary mismatch (SEMANTIC) → COMPLEX (ITERATIVE) → SIMPLE fast path → HYBRID; latency-aware override below 150 ms remaining (LEXICAL, one round); dense unavailable → LEXICAL.

**Strategies actually needed (brief: "do not implement all strategies blindly").** All nine exist, but FAST_VECTOR is never selected by default: the calibration found it no better than BM25 on simple needs and slower (it remains a config option). Routing decisions on the dev set (per need, ADAPTIVE):

| Strategy | Needs (of 72 runs) |
|---|---|
| LEXICAL (simple-need fast path and exact identifiers) | 39 |
| FILTERED | 13 |
| HYBRID | 6 |
| MULTI_HOP | 5 |
| ITERATIVE | 5 |
| SEMANTIC | 2 |
| SESSION_REUSE | 2 |
| CACHE_REUSE | 0 (exact repeats were already caught by the Phase 6 semantic cache before the controller ran) |
| FAST_VECTOR | 0 (not selected by the calibrated policy) |

**Calibration (on a different question set).** Rules fixed before running (`research/phase9/calibrate_routing.py`):

| Simple needs (n = 24) | Recall@5 | MRR@10 | nDCG@10 | mean latency ms | embeddings |
|---|---|---|---|---|---|
| HYBRID | 0.958 | 0.979 | 0.952 | 8.0 | 27 |
| LEXICAL | 0.958 | 0.979 | 0.952 | 4.4 | 3 |
| FAST_VECTOR | 0.958 | 0.979 | 0.952 | 7.4 | 27 |

Decision: `simple_strategy = LEXICAL` (equal quality, cheapest). Reranking: the pre-set rule looks only at COMPLEX / MULTI_HOP calibration needs and there were none (the Phase 7 questions are single needs), so it could not justify reranking and the default stays `never`. For the record, on *all* 28 calibration items reranking raised MRR 0.930 → 0.958 at 5.2 → 23.5 ms mean latency; the policy mode is measured separately in §18.

## 6. Query Rewriting

`QueryRewriteEngine` (docs/retrieval/02): original → normalized → contextual (the Phase 5/6 need query: resolved anaphora, inherited context, active constraints — "What about international applicants?" after an eligibility question becomes eligibility + international) → expanded (≤ 3 additive expansions: corpus-defined acronyms validated by initials, configured synonym groups, only words the index knows) → lexical query (expanded) / dense query (contextual). Invariant checked on every dev question: no content term of the need is lost; corrected entities are never re-introduced. Expansions also become accepted equivalents in the sufficiency check. Leave-one-out effect of expansion: removing it lowers coverage 0.968 → 0.949 and MRR 0.953 → 0.922 on the dev set (§19).

## 7. Adaptive Top-K

Initial k by complexity (SIMPLE 3, else 5); 5 → 10 → 20 only while slots are unmet and the last search had more candidates than k (docs/retrieval/04). The handed-on evidence is what supports a slot (plus context when not sufficient), not a fixed k. Average k per search and chunks retrieved: average k per search 4.70 (fixed: 5); removing adaptive k (start at 5, no schedule) retrieves 504 instead of 420 chunks (+20 %) at equal coverage (0.968) and MRR (0.951 vs 0.953) (§19).

## 8. Claim-Driven Retrieval

Intent → claim slots → evidence requirements (docs/retrieval/05): a *value* slot when the question asks a value kind (the evidence must state a value of that kind, not just mention the topic), a *condition* slot per constraint, comparison items, and bridge / link slots from hops. "How much does it cost to renew a permit?" is not met by "The application fee … is 55 euros" (anchor term *renew* missing) nor by "Permits must be renewed every 2 years" (no amount): the adaptive system reports the gap instead of retrieving more of the same. Measured contribution (H2): cumulative ablation C → D (+ claim-driven): coverage 0.882 → 0.919, MRR 0.848 → 0.902, nDCG 0.829 → 0.884, returned-set precision 0.559 → 0.683, gold sections cited by the answer 0.838 → 0.858, at +5 searches over 72 turns; leave-one-out from the full system: coverage 0.968 → 0.931, MRR 0.953 → 0.934, gold cited 0.930 → 0.903. Real-LLM claim support: §20.

## 9. Evidence Sufficiency

`EvidenceSufficiencyEvaluator`: SUFFICIENT / PARTIAL / INSUFFICIENT / CONTRADICTORY from the slots (term coverage ≥ 0.6, anchor and key terms, value presence, condition, validity, applicability; acronym / synonym / stem-variant equivalents). It is a lexical gate for retrieval, not a verifier. Agreement of the final evidence state with the dev labels: 57 of 69 adaptive runs end in the labelled state (dev); the disagreements are mostly SUFFICIENT-labelled needs judged INSUFFICIENT because the evidence paraphrases the question (§24). Final states on the dev set:

| Final state | Needs |
|---|---|
| SUFFICIENT | 54 |
| INSUFFICIENT | 13 |
| CONTRADICTORY | 3 |
| PARTIAL | 2 |

## 10. Iterative Retrieval

Bounded loop (docs/retrieval/06): search → merge (dedup, RRF across searches) → assess → stop? → best next action family by expected gain (contradiction search, hop, filter relaxation, broadening, requirement query, rerank, k expansion); targeted actions run alone; identical searches never repeat; with session evidence only the unmet slots are searched (delta requirements). Leave-one-out effect of iteration: without it (one round) coverage drops 0.968 → 0.946 and gold cited 0.930 → 0.893, with 17 fewer searches (75 vs 92) (§19).

## 11. Multi-Hop Retrieval

Hop 0 (the question) ‖ hop 0e (the entity alone) → bridge statement "<entity> is classified as <target>" or "see the <document title>" in the retrieved text → hop n with the target substituted (or the referenced document) → link + bridge slots (docs/retrieval/07). `RetrievalHop` records parent, query, purpose, evidence, status, bridge. Example (dev A28, trace in tests/multi_hop): hop 0e finds *ANNEX-C §1* "Zemland is classified as Group B" → hop 1 "What documents does an applicant from Group B need" → *GROUP-RULES §2*. Bridge targets extend the Phase 6 claim selection, so the answer can state both facts. Multi-hop coverage: fixed hybrid 0.500 vs adaptive 1.000; leave-one-out: without hops coverage 0.968 → 0.939 and gold cited 0.930 → 0.878.

## 12. Cache Reuse

Query cache (terms + filters + validity date; per-document source signatures), session evidence (Phase 6 store snapshot → SESSION_REUSE or delta requirements), validated-claim cache (Phase 7 SUPPORTED claims) and the unchanged Phase 6 semantic cache (docs/retrieval/08). Measured in the session pipeline (dev, 18 session turns): exact or equivalent repeats were caught by the Phase 6 semantic cache *before* the adaptive controller ran (2 hits, in every system), so the adaptive query cache had 0 hits in the pipeline runs (it does hit in direct controller use - `tests/cache`, case 9 / 17). Cross-query reuse added 2 SESSION_REUSE needs (a follow-up whose slots the session evidence already met: 0 searches). Turns served without any search: 5 / 72 adaptive vs 3 / 72 fixed. Effect on work (H4): §2, §22.

## 13. Cache Invalidation

Reasons and triggers: `entity_changed` (Phase 6 correction / entity change), `constraint_changed` (constraint removal), `source_version_changed` (content / version / status / validity signature of a document), `temporal_validity`, `stale_evidence` (Phase 6 lifecycle), `corpus_changed` (explicit flush). Each emits RETRIEVAL_INVALIDATED with the reason and change id; tested in `tests/cache` (§62: entity B never reuses entity A's evidence) and `tests/streaming_integration` (a correction "Sorry, I meant Norvia, not Zemland" invalidates the Zemland entries; the new hop is norvia → Group A). Entity invalidation is conservative (it may drop entries sharing a removed word; never keeps a stale one).

## 14. Contradiction-Aware Retrieval

Values of the asked kind are compared across documents (not within one, not across different populations, not for one-word needs); resolution by validity window, *agreed* supersession (a document cannot demote another alone; "past" questions keep the old version) and authority ≥ 0.3; losers are only chunks contradicting every kept value; an unresolved clash triggers one targeted search for a third source (other documents, "current version effective"), then stops with CONTRADICTION and hands both sides on (docs/retrieval/09). Dev results: all 4 contradiction turns end CONTRADICTORY (PPO opening hours 9:00 vs 8:30; the Phase 7 fee notices 40 vs 55 euros; dome wind limits 25 vs 30 km/h — the last two have no metadata, so nothing can resolve them) and the answer reports both sides in 4 / 4 (fixed hybrid: also 4 / 4: the Phase 7 answer already reports conflicts). Resolved conflicts: the current fee (2024 vs 2026 edition) by validity window; "previous fee" keeps the 2024 edition (`past_version`). Cost: 1 extra (targeted) search per contradiction; leave-one-out shows no quality change and 3 fewer searches without it — on this data the third-source search never found a resolving document (there is none).

## 15. Retrieval Budgets

`RetrievalBudget`: max_queries 5, max_results 40, max_iterations 3, max_latency 1500 ms, max_parallel_tasks 2, max_hops 2; integrated with Phase 8 (task deadline → latency budget, cancellation token checked between searches, injected faults per dense search; docs/retrieval/11). Budget-sensitivity (quality vs number of allowed searches, §22): allowing 1 search per need gives coverage 0.924 / MRR 0.939; 2 searches 0.961 / 0.946; 3 searches 0.968 / 0.953; the default 5 and a larger budget (8 searches, 5 rounds) give the same as 3 — the stopping policy rarely uses more than 3.

## 16. Stopping Policy

Nine stop reasons (docs/retrieval/10), one per run. Marginal value measured after each round (Δcoverage + 0.5·Δquality); expected gain = unmet share × documented action prior; no optimality claimed. Stop reasons on the dev set:

| Stop reason | Needs |
|---|---|
| SUFFICIENT_EVIDENCE | 54 |
| NO_EXPECTED_GAIN | 14 |
| CONTRADICTION | 3 |
| ITERATION_LIMIT | 1 |
| QUERY_LIMIT / LATENCY_LIMIT / NO_RESULTS / ERROR / USER_CANCELLED | 0 (exercised in tests/budget, tests/adaptive_retrieval) |

## 17. Streaming Integration

* **Phase 6 session pipeline** (`AdaptivePipeline`): a turn's delta plan is executed by one controller run per changed / new need; unchanged needs are reused as before (`reuse_active`), Phase 6 cache hits stay; corrections / constraint removals invalidate adaptive cache entries with reasons; Phase 7 verified claims feed the claim cache.
* **Phase 8 runtime**: a need is one RETRIEVAL task (instead of lexical / dense / assemble subtasks) running the controller with the task's cancellation checkpoint and remaining deadline; its events are buffered and emitted on the loop at commit (a superseded run's events are never emitted). Tested: virtual-mode multi-hop turn, word-by-word streaming (bounded searches), and a late detail that cancels the running adaptive task (`superseded_in_flight`).
* Delta retrieval is preserved: a late constraint ("For international applicants.") re-plans only the affected need (FILTERED), a repeated question is not retrieved again, a follow-up whose slots the session evidence already meets is answered with SESSION_REUSE (0 searches).
* H5 (cancellation × adaptive) is measured in §22.

**Final end-to-end test (brief §73)** — `research/phase9/e2e_final.py`, Phase 8 runtime in realtime mode, chunks streamed every 250 ms, adaptive retrieval on, real retrieval / NLI and the **real local LLM** (qwen3:4b), 12 scenarios / 17 turns, expectations written before the run (`results/e2e_final/README.md`, event logs `*.jsonl`): **19 / 24 checks passed**; all 17 turns completed with a validated final answer from the LLM (20 LLM calls, 80 claims verified, 18 rejected, 36 citations validated, 603 streamed answer events, 0 untraceable of 2,424 events).

| # | Scenario | Final strategy | Checks |
|---|---|---|---|
| 1 | simple question | LEXICAL (a provisional partial query had used HYBRID) | 1 / 2 — the check inspected the *first* plan, which belonged to a partial transcript (mis-specified check, reported as failed) |
| 2 | multi-intent | one plan per need | 2 / 2 |
| 3 | contextual follow-up | FILTERED international | 1 / 2 — the international-specific section was not cited (§24) |
| 4 | exact entity | LEXICAL | 2 / 2 |
| 5 | temporal ("December 2025") | FILTERED by period, 2024 edition cited (40 euros) | 2 / 2 |
| 6 | insufficient evidence | INSUFFICIENT, gap stated | 2 / 2 |
| 7 | contradictory evidence | CONTRADICTION, both opening hours reported | 2 / 2 |
| 8 | multi-hop | zemland → Group B, both facts cited | 2 / 2 |
| 9 | cached question | repeat re-retrieved | 1 / 2 — in streaming the repeat was interpreted as a different need (§24) |
| 10 | changed entity | entity_changed invalidation, norvia → Group A | 2 / 2 |
| 11 | changed constraint | FILTERED {domestic, international} | 0 / 2 (§24) |
| 12 | streaming correction (ASR revision) | FILTERED international | 2 / 2 |

## 18. Baseline Comparison

Every system runs the same session pipeline (intent decomposition, delta planning, evidence lifecycle, extractive grounded answer + NLI verification); only retrieval differs. Fixed systems use k = 5 per need (the pipeline default). Dev set: 72 turns (68 with gold sections), one run per case; latencies here are single-run (see §21 for the repeated test).

**Dev set** (`results/baselines.json`):

| System | R@5 | R@10 | coverage | P@5 | returned prec. | MRR@10 | nDCG@10 | gold cited by answer | searches | embeddings | reranker calls | chunks |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A vector-only k=5 | 0.944 | 0.944 | 0.944 | 0.250 | 0.249 | 0.888 | 0.889 | 0.890 | 72 | 72 | 0 | 357 |
| B BM25-only k=5 | 0.909 | 0.914 | 0.914 | 0.235 | 0.282 | 0.799 | 0.818 | 0.878 | 72 | 0 | 0 | 332 |
| C fixed hybrid k=5 | 0.944 | 0.944 | 0.944 | 0.250 | 0.250 | 0.869 | 0.861 | 0.886 | 72 | 72 | 0 | 357 |
| D hybrid + rerank k=5 | **0.971** | **0.971** | **0.971** | **0.262** | 0.261 | 0.912 | 0.921 | 0.905 | 72 | 72 | 72 | 357 |
| **Adaptive** | 0.961 | 0.968 | 0.968 | 0.256 | **0.773** | **0.953** | **0.944** | **0.930** | 92 | 50 | 0 | 420 |
| Adaptive, rerank = policy | 0.961 | 0.968 | 0.968 | 0.256 | 0.767 | 0.951 | 0.942 | 0.930 | 92 | 50 | 30 | 420 |

**Held-out set** (`results/baselines_heldout_adaptive_retrieval.json`, 23 turns, written after the freeze, run once):

| System | R@5 | coverage | MRR@10 | nDCG@10 | returned prec. | gold cited | searches | embeddings | chunks |
|---|---|---|---|---|---|---|---|---|---|
| A vector-only | 0.932 | 0.932 | 0.888 | 0.892 | 0.218 | 0.886 | 23 | 23 | 115 |
| B BM25-only | **0.977** | **0.977** | 0.924 | **0.939** | 0.250 | **0.955** | 23 | 0 | 112 |
| C fixed hybrid | 0.909 | 0.909 | 0.886 | 0.893 | 0.209 | 0.886 | 23 | 23 | 115 |
| D hybrid + rerank | 0.932 | 0.932 | 0.932 | 0.915 | 0.218 | 0.909 | 23 | 23 | 115 |
| **Adaptive** | 0.932 | 0.932 | **0.977** | 0.937 | **0.806** | 0.932 | 32 | 17 | 129 |
| Adaptive, rerank = policy | 0.932 | 0.932 | 1.000 | 0.947 | 0.806 | 0.932 | 32 | 17 | 129 |

Reading: on both sets the adaptive policy ranks the needed sections first far more often than fixed hybrid (MRR +0.08 / +0.09) and hands on evidence that is about three times more precise (only what supports a claim slot), while computing fewer embeddings (−31 % / −26 %) but running more searches (+28 % / +39 %, mostly cheap BM25 follow-ups). Coverage: above fixed hybrid on both sets, below hybrid + rerank on the dev set, and below plain BM25 on the held-out set (BM25 alone is very strong on this small, lexically clean corpus). Hybrid did not beat its parts here: on the dev set it equals vector-only, on the held-out set it is below both — "hybrid is better" is not supported on these fixtures (brief §7).

**Reranker evaluation (§28–29):** always reranking (D vs C) raised dev coverage 0.944 → 0.971 and MRR 0.869 → 0.912 at 5.6 → 30.4 ms p50 retrieval latency (single run) and 72 cross-encoder calls; the adaptive rerank *policy* reranked 30 needs and changed nothing measurable on the dev set (MRR 0.953 → 0.951) while raising p95 latency 15 → 77 ms; on the held-out set it raised MRR 0.977 → 1.000. Decision kept: `rerank: never` by default (calibration rule, §5).

## 19. Ablation Studies

**Cumulative (brief §51, A → G; dev set, `results/ablations.json`):**

| Ablation | coverage | MRR@10 | nDCG@10 | returned prec. | gold cited | searches | embeddings | chunks | avg k |
|---|---|---|---|---|---|---|---|---|---|
| A fixed retrieval (hybrid k=5) | 0.944 | 0.869 | 0.861 | 0.250 | 0.886 | 72 | 72 | 357 | 5.00 |
| B + adaptive strategy selection | 0.904 | 0.866 | 0.844 | 0.551 | 0.853 | 72 | 33 | 338 | 5.00 |
| C + adaptive top-k (and iteration) | 0.882 | 0.848 | 0.829 | 0.559 | 0.838 | 80 | 41 | 354 | 4.53 |
| D + claim-driven retrieval | 0.919 | 0.902 | 0.884 | 0.683 | 0.858 | 85 | 46 | 414 | 4.97 |
| E + cache reuse | 0.919 | 0.895 | 0.878 | 0.685 | 0.858 | 83 | 44 | 404 | 4.96 |
| F + contradiction handling | 0.919 | 0.895 | 0.878 | 0.685 | 0.858 | 86 | 44 | 417 | 4.97 |
| G full (+ multi-hop, temporal, expansion) | **0.968** | **0.953** | **0.944** | **0.773** | **0.930** | 92 | 50 | 420 | 4.70 |

Strategy selection alone (B) halves embeddings but *loses* coverage (the fast path without a sufficiency check stops at BM25); adaptive k without claim slots (C) loses more (the single all-terms requirement is a poor gate); claim-driven slots (D) recover most of it; cache and contradiction handling change little on single-turn data; the components added in G (multi-hop, temporal validity, expansion) give the largest jump on this dev set — which contains temporal and multi-hop categories by design.

**Leave-one-out from the full system** (each component's independent contribution, `results/ablations_loo.json`):

| Removed | coverage | MRR@10 | gold cited | searches | embeddings | chunks |
|---|---|---|---|---|---|---|
| — (full) | 0.968 | 0.953 | 0.930 | 92 | 50 | 420 |
| routing (always hybrid) | 0.931 | 0.942 | 0.886 | 87 | 87 | 430 |
| adaptive k | 0.968 | 0.951 | 0.930 | 92 | 50 | 504 |
| iteration (one round) | 0.946 | 0.941 | 0.893 | 75 | 36 | 287 |
| claim-driven slots | 0.931 | 0.934 | 0.903 | 83 | 44 | 335 |
| cache + session reuse | 0.968 | 0.961 | 0.930 | 94 | 52 | 430 |
| contradiction search | 0.968 | 0.953 | 0.930 | 89 | 50 | 407 |
| multi-hop | 0.939 | 0.944 | 0.878 | 87 | 45 | 420 |
| temporal validity | 0.968 | 0.922 | 0.930 | 92 | 50 | 422 |
| expansion | 0.949 | 0.922 | 0.910 | 91 | 49 | 415 |

Every component except the cache and the contradiction search has a measurable quality effect on this set; those two only change work (2 and 3 searches). Routing is what saves embeddings (87 → 50).

## 20. Retrieval Quality

Retrieval quality is in §18 (Recall@K, Precision@K, MRR, nDCG, coverage, returned-set precision). Grounded-answer quality (brief §66: claim support, citation accuracy, retrieval recall):

* **Extractive answers** (all systems, dev): claim support 1.000, unsupported 0.000, citation precision 1.000 everywhere — the extractive generator copies verified sentences, so these do not discriminate. What does: the share of gold sections the final answer cites — fixed hybrid 0.886, adaptive 0.930 (held-out 0.886 vs 0.932); gaps reported for unanswerable questions 2/4 vs 3/4; conflicts reported 4/4 in both.
* **Real local LLM (qwen3:4b)**, `results/llm_quality.json`:

| System (72 turns, qwen3:4b, 0 fallback turns) | claim support | unsupported | citation precision | gold sections cited | coverage | MRR@10 | unanswerable reported | conflicts reported | LLM calls | searches | turn latency p50 / p95 ms |
|---|---|---|---|---|---|---|---|---|---|---|---|
| C fixed hybrid | 0.930 | 0.070 | 1.000 | 0.900 | 0.944 | 0.869 | 2 / 4 | 4 / 4 | 84 | 72 | 2502 / 5415 |
| **Adaptive** | 0.924 | 0.076 | 1.000 | **0.915** | **0.968** | **0.953** | **3 / 4** | 4 / 4 | **75** | 92 | **1846 / 3687** |
| ablation C (adaptive k, no claim slots) | 0.932 | 0.068 | 1.000 | 0.823 | 0.882 | 0.848 | 3 / 4 | 4 / 4 | 75 | 80 | 2147 / 4781 |
| ablation D (+ claim-driven) | 0.929 | 0.071 | 1.000 | 0.843 | 0.919 | 0.902 | 3 / 4 | 4 / 4 | 74 | 85 | 1798 / 3898 |

Claim support (verifier-judged share of the model's claims that are supported before the unsupported-claim policy) is equal within noise (0.930 vs 0.924); every released citation is valid in all systems; the adaptive answers cite more of the gold sections and report one more unanswerable question. **End-to-end turn latency with the real model is lower for the adaptive policy** (p50 −26 %, p95 −32 %): the answer stage receives about a third of the chunks (precision 0.77 vs 0.25), so prompts are shorter and 9 fewer LLM calls are made (fewer repair calls) — the few extra milliseconds of retrieval are dwarfed by generation. One run per configuration; a first attempt was discarded because the external Ollama server stopped mid-run and two configurations silently fell back to extractive answers (§24) — the script now refuses to continue when the server is unreachable and records fallbacks (0 here).

No significant degradation was observed; the adaptive policy's answers cite the gold sections more often. Small samples, one run, fixture corpus.

## 21. Latency

Retrieval stage only (generation off), fixed hybrid vs adaptive, **5 interleaved repetitions per turn**, per-turn median, then percentiles over 71 turns (`results/latency.json`):

| | p50 ms | p95 ms | mean ms | CPU p50 ms | CPU p95 ms | CPU mean ms |
|---|---|---|---|---|---|---|
| C fixed hybrid | 5.70 | 6.63 | 5.72 | 30.0 | 51.6 | 29.5 |
| Adaptive | 6.56 | 16.62 | 8.09 | 6.4 | 91.4 | 27.2 |
| ratio | 1.15 | 2.51 | 1.41 | | | |

By category (p50 ms, fixed → adaptive): simple 5.46 → 4.67, exact keyword 5.38 → 4.21, semantic 5.72 → 4.89, repeated question 5.49 → 4.91, cross-query reuse 5.97 → 4.37 — **faster**; multi-hop 5.95 → 16.64, late constraint 5.95 → 19.34, entity correction 4.88 → 14.65, constraint 5.86 → 9.36, temporal 6.28 → 9.93, insufficient 5.74 → 8.72 — **slower** (more searches by design).

**The adaptive policy is not faster overall** (brief §65: "do not claim adaptive retrieval is faster unless measurements show it"): +15 % at p50, +151 % at p95. It is faster on the easy majority (simple / exact / semantic needs) and spends more time where the fixed pipeline returned insufficient evidence. Median CPU per turn drops from 30.0 to 6.4 ms (no embedding on the fast path; ONNX uses several threads, so CPU > wall), the mean is about equal. These are milliseconds against a local in-memory index; generation dominates the turn (real LLM: §20).

## 22. Efficiency

**Quality vs work** (`results/curves.json`, retrieval only, one run; latency = mean retrieval wall time per turn):

| Configuration | coverage | MRR@10 | returned prec. | searches / turn | embeddings | chunks | latency ms |
|---|---|---|---|---|---|---|---|
| hybrid k=3 | 0.899 | 0.863 | 0.387 | 1.00 | 72 | 215 | 5.0 |
| hybrid k=5 | 0.944 | 0.869 | 0.250 | 1.00 | 72 | 357 | 5.7 |
| hybrid k=10 | 0.963 | 0.869 | 0.144 | 1.00 | 72 | 690 | 7.1 |
| hybrid k=20 | 0.985 | 0.869 | 0.093 | 1.00 | 72 | 1308 | 9.2 |
| BM25 k=5 / k=10 | 0.914 / 0.949 | 0.799 / 0.802 | 0.282 / 0.205 | 1.00 | 0 | 332 / 588 | 2.2 / 3.2 |
| dense k=5 / k=10 | 0.944 / 0.963 | 0.888 / 0.890 | 0.249 / 0.144 | 1.00 | 72 | 357 / 690 | 5.5 / 6.8 |
| adaptive, ≤ 1 search | 0.924 | 0.939 | 0.727 | 0.97 | 31 | 262 | 6.9 |
| adaptive, ≤ 2 searches | 0.961 | 0.946 | 0.729 | 1.17 | 42 | 345 | 7.5 |
| adaptive, ≤ 3 searches | 0.968 | 0.953 | 0.773 | 1.25 | 48 | 380 | 7.8 |
| adaptive (default ≤ 5) | 0.968 | 0.953 | 0.773 | 1.28 | 50 | 420 | 7.9 |
| adaptive, ≤ 8 searches | 0.968 | 0.953 | 0.773 | 1.28 | 50 | 420 | 7.9 |
| adaptive, hybrid fast path | 0.961 | 0.963 | 0.773 | 1.28 | 86 | 427 | 9.6 |

On these fixtures fixed retrieval buys coverage only by handing on more chunks (k=20: coverage 0.985 at precision 0.09), and its MRR does not move with k. The adaptive policy reaches hybrid-k=10 coverage with MRR 0.95 and precision 0.77, at ~1.2–1.3 searches per turn; its curve flattens after 3 searches (the stopping policy rarely uses more). Its mean latency is above the fixed configurations of similar coverage (7.9 vs 7.1 ms for hybrid k=10).

**H5 — wasted computation in the Phase 8 runtime** (`results/runtime_h5.json`; 54 questions streamed 2 words per chunk every 80 ms; wasted = worker time of retrieval tasks whose query was superseded):

| Condition | Configuration | wasted ms | useful ms | retrieval tasks | cancelled | coverage |
|---|---|---|---|---|---|---|
| local index | fixed, no cancellation | 1323 | 2231 | 285 | 0 | 0.960 |
| | fixed + cancellation | 1360 | 2256 | 285 | 0 | 0.960 |
| | adaptive, no cancellation | 399 | 646 | 95 | 0 | 0.957 |
| | adaptive + cancellation | 398 | 575 | 95 | 0 | 0.957 |
| remote dense +150 ms per search (SYNTHETIC) | fixed, no cancellation | 7624 | 11028 | 285 | 0 | 0.960 |
| | fixed + cancellation | 7478 | 11022 | 284 | 1 | 0.960 |
| | adaptive, no cancellation | 5053 | 3773 | 95 | 0 | 0.957 |
| | adaptive + cancellation | 4956 | 3760 | 95 | 1 | 0.957 |

Adaptive retrieval cut wasted retrieval time by 70 % (local) and 35 % (remote) against fixed retrieval with cancellation, at a coverage change of −0.003. **Cancellation contributed almost nothing in this workload** (0–1 cancelled tasks): the Phase 4 controller issues few provisional queries (95 for 54 questions) and they rarely overlap the next one, so superseded work had usually *finished* before it became obsolete. The mechanism works (tests/streaming_integration: a 1.2 s remote search is cancelled `superseded_in_flight`), but its benefit depends on retrieval being slower than the speaker. Noted finding: with a remote index the adaptive policy wastes *more than half* of its retrieval time on superseded partial questions (4956 vs 3760 ms) — iterating on provisional queries is expensive (Phase 10: one round for provisional queries).

## 23. Cost / Operation Counts

No API is used: retrieval, embeddings, reranking, NLI and the LLM run locally, so monetary cost is NOT MEASURED; operation counts instead (dev set, 72 turns; `results/baselines.json`):

| Operation | A vector | B BM25 | C hybrid | D hybrid + rerank | Adaptive |
|---|---|---|---|---|---|
| retrieval calls (searches) | 72 | 72 | 72 | 72 | 92 (1.28 / turn) |
| BM25 searches | 0 | 72 | 72 | 72 | 90 |
| embedding calls (query embeddings) | 72 | 0 | 72 | 72 | **50** |
| reranker calls / candidates | 0 | 0 | 0 | 72 / ≤ 1440 | 0 |
| chunks retrieved | 357 | 332 | 357 | 357 | 420 |
| chunks handed to the answer stage | 357 | 332 | 357 | 357 | about 115 (returned precision 0.77) |
| LLM calls in retrieval | 0 | 0 | 0 | 0 | 0 |
| LLM calls in generation (real-LLM run, 72 turns) | – | – | 84 | – | **75** |
| average k | 5.0 | 5.0 | 5.0 | 5.0 | 4.70 |

The adaptive layer itself never calls an LLM (tested). The expensive operations in a deployed system are the embedding (remote vector service) and the reranker; the adaptive policy computes 31 % fewer embeddings and no reranker calls, but issues 28 % more (cheap) searches.

## 24. Failure Cases

Defects found **while running the dev set during development** (fixed generically before the freeze; they make the dev numbers optimistic): union-based conflict detection missed a 40 vs 55 clash when a third chunk stated both; question words ("how *high*", "*rules*") and a lower-case rare word ("charge") triggered false multi-hop; the anchor rule let "cost to *renew*" be met by an application-fee text, and later (after a fix) let "pay by *credit card*" be met by any payment text; applicability: different applicant types' processing times were flagged as a contradiction and international evidence was kept for a domestic need; follow-up searches leaked unrelated evidence into the final set; k expansion lost the context items; the hop-budget check counted the hop being created (bridge hops never ran); the first hop searched the whole question, where the entity's own statement ranked below generic matches (added the entity search); the multi-hop bridge words pulled an "Estria is classified as Group B" sentence into a Zemland answer, where a Phase 7 consistency check then *removed* the correct Zemland claim as "contradicting" it (bridge-only sentences now dropped); a planted metadata value could force FILTERED routing (security test; user-stated rule added).

**Held-out failures (run once, not fixed):**
* H07 "Which form do domestic applicants use?" — no constraint marker before "domestic", so no filter; the fast path met the slot with the forms catalogue and stopped: coverage 0.5 (fixed hybrid 1.0).
* H13 "Do Norvia applicants attend an interview?" — bridge "Norvia → Group A" found, but "Group A" analyses to {group} ("a" is a stopword): the Group B rule (which mentions interviews) met the slot: wrong group evidence first (coverage 0.5).
* H15 "Is the PPO open on 25 December?" — the capitalised month was taken as a multi-hop entity: hops instead of the simple lexical path, only the holiday list returned (coverage 0.5).
* H01, H02 — paraphrased answers judged INSUFFICIENT (extra search; coverage still 1.0).

**Dev-set failures remaining:** paraphrase INSUFFICIENT states (A16 "bring somebody along" fixed by fillers; A17 "old cars … checked" vs "vehicles … inspection" remains); "processing for domestic applicants" also accepts the business-licence processing time (need without its subject); late-constraint and correction turns are 2–3× slower than fixed retrieval.

**End-to-end failures (§17, run after the freeze, not fixed):**
* 11 changed constraint: in streaming, Phase 5/6 kept "for international applicants" inside the need's own text while registering the retraction, so the query contained both values; the analyzer turned both into a filter {domestic, international} and the answer stayed "30 working days" + "not established for domestic". The adaptive layer does not yet honour retracted constraints that live inside the need text (Phase 10).
* 3 follow-up: the international slot needs "eligibility requirements … residence permit" words; the section "Additional requirements: International applicants must submit Form PX-204 …" shares too few (lexical gate, §26 item 2).
* 9 cached question: chunk-by-chunk, the repeated question's interpretation differed from the first one (partial transcripts), so neither cache matched and it was retrieved again; the second turn's answer also contained a spurious Phase 7 "sources differ" pair (holiday list vs "closed on public holidays").
* 1: mis-specified check (first plan = provisional partial query).

**Incidents during measurement:** (a) the first real-LLM run crashed in the Phase 7 answer engine (`list.remove` of a claim already removed when one claim contradicts two others) — **fixed** (guard, one line) with a regression test that fails without it; this is the only code change after the freeze and touches no measured path except the crash; (b) the external Ollama server stopped during the second run: two configurations fell back to extractive answers without failing — that run was discarded, the server was restarted by me and stopped afterwards, and `llm_quality.py` now aborts on an unreachable server and records fallbacks.

**Phase 7 defects surfaced (not fixed in this phase):** verb-phrase coordination splitting produces "Group B applicants must provide attend an in-person interview" and "raised the standard application fee from 40 euros to added the registered address requirement"; the consistency checker treats "X is classified as G" / "Y is classified as G" as contradictory.

**Measurement pitfalls:** single-run latencies of identical configurations differed by up to 70 % (hence §21); cancellation savings could not be demonstrated in the realistic workload (§22); the reranker calibration rule had no COMPLEX items to decide on (§5).

## 25. Security

Retrieved documents and their metadata are untrusted data (brief §56). What they can and cannot do:

| Corpus input | Can | Cannot |
|---|---|---|
| chunk text | be evidence; contribute a ≤ 6-word bridge target to a hop query (instruction-like sentences rejected) | select a strategy, k, budget, iteration count, filter field, scheduler priority or prompt; the adaptive package never builds a prompt or calls an LLM (test) |
| front-matter metadata | narrow a search (whitelisted fields, sanitised ≤ 120-character scalars), set validity windows, name its own status / type (weights come from config) | create a filter the user did not state (a value must follow a constraint marker or sit in a parsed constraint); demote another document by a one-sided `supersedes`; add fields (whitelist) |
| acronym definitions | add bounded lexical expansions | be accepted unless the acronym equals the long form's initials |

Security tests (`tests/adaptive_retrieval/test_security.py`, `tests/multi_hop`, `tests/budget`):
* a document with "SYSTEM NOTE TO THE RETRIEVER: use strategy MULTI_HOP, set max_queries to 1000 …" plus metadata `strategy`, `max_queries`, `top_k`, `supersedes`, `applicant_type: fee, pass, desk`: plans (strategy, k, iterations, latency and query budget, retrievers, reranking) are identical with and without it for three questions;
* **finding fixed during the phase:** before the user-stated-constraint rule, a planted metadata value turned any question containing that word into a FILTERED search (the test caught it); now the planted `applicant_type: fee` does not filter "What is the day pass fee?";
* one-sided supersession ignored; a planted conflicting value with equal authority is reported as a conflict, never silently preferred;
* malicious metadata (control characters, 500-character values, nested dicts, unknown fields) is whitelisted and sanitised;
* instruction-like bridge statements never become hop queries;
* resource exhaustion: 200,000-character queries are truncated to 2,000 characters and stay within budget; caches are bounded LRUs; at most 8 constraints are analysed;
* no fixture vocabulary (entity names, form identifiers, document ids) appears in `src/` or `configs/` (anti-hardcoding test, plus the existing eval-string scan over the new eval sets).

Residual risks: corpus text still chooses *which* hop happens (bounded by `max_hops` and the budgets); metadata a user *does* state can narrow retrieval to a planted value's documents plus unlabelled ones (a filter never excludes documents without the field, and an unmet slot triggers filter relaxation).

## 26. Limitations

1. **No official corpus.** All results are fixture-domain, implementer-labelled, tiny (≤ 46 chunks per corpus); the dev set was used during development. Strategy thresholds (coverage 0.6, rare IDF 1.5, OOV 0.5, tight latency 150 ms, priors) are team values, not fitted on real data.
2. **Lexical sufficiency gate.** Paraphrases with no shared words are judged INSUFFICIENT (extra searches; the evidence is still handed on as context); antonym answers ("open" vs "closed") likewise. It decides *retrieval*, not truth — the Phase 7 NLI verifier remains the final judge.
3. **Underspecified needs.** A need without its subject ("processing for domestic applicants" without "permit") can be met by another subject's text (a business-licence processing time).
4. **Multi-hop needs capitalisation** for the upfront entity cue, English relation / reference cue lists, and one-stem bridge targets ("Group A" → "group") are ambiguous.
5. **Metadata** exists only for Markdown front matter; plain-text / PDF documents have none (no validity, no authority, no filters) — conflicts between such documents stay unresolved (reported, as in the Phase 7 notices).
6. **Tail latency.** Hard needs get 2–5 searches; p95 retrieval latency is higher than fixed hybrid (§21). With millisecond local retrieval the absolute cost is small; with a remote index it is not.
7. **Cancellation only saves work still running.** Locally, retrieval finishes before the next transcript chunk, so cooperative cancellation has nothing to cancel (§22).
8. **Phase 7 defects surfaced, not fixed:** coordination splitting ("must provide attend an in-person interview") and a consistency-check false positive between "X is classified as G" and "Y is classified as G" (§24).
9. **Single machine, one run** for the real-LLM comparison; small sample sizes; no statistical tests.

## 27. Phase 10 Requirements

**Ready for Phase 10:** a per-need adaptive retrieval controller with explainable plans and complete decision logs, claim-driven sufficiency, validity-checked caches, bounded budgets integrated with the Phase 8 runtime, document metadata in the index, and a measurement harness (`research/phase9/`) that re-runs every table on a new corpus.

**Exact prerequisites:**
1. **The official Theme 4 corpus and evaluation questions** (still blocked) — then re-run `calibrate_routing.py` on a calibration split and `run_benchmarks.py` / `run_experiments.py` / `llm_quality.py` on a held-out split, and decide `adaptive_retrieval.enabled` from those numbers (it stays off until then).
2. **Metadata for the real corpus:** which fields exist (version, effective / validity dates, status, applicability), in which format (front matter, sidecar, PDF properties), and which fields may filter (`filter_fields`) and how authority is weighted — a trusted, operator-owned configuration.
3. **A decision on the sufficiency gate for paraphrases:** keep lexical (conservative), or add a calibrated semantic signal (dense or NLI entailment of the slot) — needs labelled data from item 1.
4. **A latency target per turn** (no official target exists) to set `max_latency_ms`, `tight_latency_ms` and the tail-latency trade-off measured in §21; and whether the index will be remote (cancellation and adaptive-k savings depend on it, §22).
5. **Phase 7 fixes** surfaced here: verb-phrase coordination splitting and the consistency-check false positive for parallel classification statements.
6. **Decide the multi-hop entity cue for ASR transcripts** (lower-case): Phase 5 entity spans or a named-entity component.
7. **Commit Phase 9.** The working tree is not committed (commit only on request).
