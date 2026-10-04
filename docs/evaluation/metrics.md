# Metrics (Phase 10)

Implementations: `src/streamrag/evaluation/metrics/` — one module per family, no cross-imports; each module's
docstring is the normative definition. Validated on known examples in `tests/evaluation/test_metrics_validation.py`
and `test_streaming_metrics.py` (brief §55).

| Family | Metrics | Basis |
|---|---|---|
| Retrieval | Recall@1/3/5/10, Precision@1/3/5, MRR, nDCG@10 | ranked evidence the system hands to its answer stage vs gold sections; only when gold exists |
| Evidence | evidence recall / precision, evidence coverage (expected facts present in one evidence chunk), unsupported evidence rate, contradictory (stale / forbidden) evidence, redundancy (mean pairwise cosine), source diversity | gold + key strings + forbidden patterns |
| Claims | claim support, unsupported, contradicted rates (verifier-judged); claim coverage (expected claims stated) | instrument + labels |
| Claim verification accuracy | accuracy, precision / recall of SUPPORTED, false acceptance per perturbation type | perturbations with labels by construction (`experiments/runners/verifier_validation.py`) |
| Generation | answer correctness (all expected key facts, no forbidden match), completeness, forbidden hit, abstention, insufficiency handled, conflict reported, faithfulness, groundedness | labels (model-free) + instrument |
| Citation | precision (citation supports its own claim), recall (factual claims with a supporting citation), completeness (gold sections cited), entailment, source validity, position correctness | instrument |
| Hallucination | unsupported claim rate, **hallucinated** claim rate (values found in no evidence and not in the question), citation-less factual claim rate, grounding failure | instrument + model-free value check |
| Latency | TTFT, TTFE, TTFA, TTVA, total — from the first chunk and after the utterance end; p50 / p90 / p95 / p99 (p90 ≥ 10, p95 ≥ 20, p99 ≥ 100 samples) | wall clock |
| Streaming | answer updates, gaps between updates, revisions, time to useful partial answer, time to final, stale update rate, ordering errors, interruption | runtime event log |
| Efficiency | retrieval calls, embeddings, BM25 searches, reranker calls, chunks retrieved, avg k, iterations, expansions, documents retrieved, cache hit, evidence reused, LLM calls, prompt / output tokens | counters of work done |
| Cancellation | cancelled tasks by type, wasted worker time (tasks of superseded / cancelled / stale work), LLM calls, latency | runtime events (`exec_ms`) |
| Robustness | recovery, degraded-mode success, incorrect-answer rate, failure propagation | fault-injection runs |

**Terminology (hallucination family).** *Unsupported*: the evidence the system had does not entail the claim (it may
still be true). *Hallucinated*: the claim states a value that occurs in no evidence item and not in the question
(invented content). A *wrong but supported* answer (e.g. a stale fee taken from a superseded document) is an
evidence / generation error, not a hallucination. Abstentions are not claims.

**The instrument and its bias.** Claim / citation metrics use the Phase 7 claim verifier (nli-deberta-v3-xsmall +
rules) against the evidence each system handed to its answer stage. The proposed system filters its own claims with
the same verifier, so verifier-judged support favours it by construction; model-free metrics (answer correctness,
claim coverage, hallucinated values, retrieval) do not.

**Not measured:** answer relevance (needs human or judge scores — no LLM judge is used and no human annotators were
available; protocol in `human_evaluation.md`), GPU utilisation (Ollama on Apple GPU, no counter available), money
(no paid API).
