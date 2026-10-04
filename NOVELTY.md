# Novelty

This file separates what is technically different from what is standard, and engineering contribution from research
contribution. Every component named here is implemented, tested and measured (`docs/architecture/14_final_architecture.md`,
`PHASE_10_EVALUATION_REPORT.md`, `FINAL_BENCHMARK_RESULTS/`).

## What is *not* novel

These are standard techniques, used deliberately:
* hybrid BM25 + dense retrieval with reciprocal-rank fusion;
* a small ONNX embedder (bge-small);
* an NLI model for entailment;
* a local LLM with JSON-schema output;
* cross-encoder reranking, which is implemented but off.

Rule-based query decomposition and retrieve / wait / skip decisions are simple by design. The official guide penalises
heavy orchestration, and Phase 1 measured that an LLM on the decision path would cost hundreds of milliseconds per
chunk.

## What is technically different

1. **Streaming-first retrieval with an explicit unit of change.**
   * The system does not re-run RAG per transcript update. It keeps a ledger of queries per *need* and a lifecycle per
     piece of evidence: active, retained, revalidation required, stale or superseded.
   * A late detail or a correction changes only the affected needs (delta retrieval). The answer is a versioned state
     with claim-level diffs, not a regenerated paragraph.
2. **Per-need, claim-driven adaptive retrieval without an LLM in the loop.**
   * Each need gets claim requirements: what it must establish, such as an amount, a duration or a form.
   * A cheap router chooses keyword fast path, filtered, semantic, iterative, multi-hop or reuse, with an adaptive k
     and a bounded stop rule. The plan is explained in events and visible in the demo.
   * Most adaptive-retrieval work puts an LLM or a trained classifier in that decision. Here it costs milliseconds.
3. **Cancellation-aware asynchronous execution.**
   * Retrieval and generation run as prioritised, deadline-bounded tasks.
   * Work for a superseded or corrected question is cancelled cooperatively. Stale results are rejected by state-version
     checks, and a degraded mode is always declared, never silent.
4. **Verified incremental generation.**
   * Drafts stream while the user speaks.
   * Every released claim, draft or final, is checked against its cited section, and citations are rebuilt from the
     verification.
   * Conflicting sources are shown side by side. A version conflict is labelled current vs superseded (Phase 11).
     Unsupported content is repaired or removed.
5. **Early-commitment control (Phase 11).**
   * A streaming system can lock in decisions made on a partial transcript. Phase 10 measured this as the main quality
     cost of streaming: 7 turns lost against batch mode on v1.
   * Phase 11 added two general rules:
     * evidence that only superseded partial-transcript queries retrieved is dropped when the refined query returns;
     * a need whose query budget is exhausted is re-validated against its latest result.
   * This failure mode does not exist in turn-based RAG.

## Strongest innovation

The combination in points 1 and 5:
* incremental, need-level state for RAG over a live transcript, in which late details and corrections re-retrieve
  only what changed;
* explicit protection against the errors that incrementality itself causes.

The measured trade-off is real and reported:
* streaming delivers first evidence and first answer content far earlier;
* reuse and early commitment can cost answer quality, which the Phase 11 fixes address and the held-out v2 benchmark
  measures.

## Engineering vs research novelty

| contribution | type | evidence |
|---|---|---|
| need-level ledger, delta planning, evidence lifecycle | engineering design with a research question (RQ3, RQ6) | Phase 6; Phase 10 Exp 4–5 |
| claim-requirement-driven adaptive retrieval without an LLM | research question (RQ1, RQ2) | Phase 9; Phase 10 Exp 2–3, 8–10; v2 held-out |
| streaming runtime with cancellation and degraded modes | engineering | Phase 8; Phase 10 Exp 6, robustness |
| claim verification on every streamed answer | engineering (known technique, new placement) | Phase 7; Phase 10 §23 |
| early-commitment analysis and fixes | research finding plus engineering fix | Phase 10 §14, §20; Phase 11 development iterations and v2 |
| evaluation protocol: held-out sets frozen before changes, paired tests, negative results | methodology | Phase 10, Phase 11 |

No claim of state-of-the-art performance is made. All measurements are on fictional fixture corpora with
implementer-written labels.
