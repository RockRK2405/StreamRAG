# Final Presentation: StreamRAG (18 slides)

Each slide gives its content, a visual and the speaker note.
* Numbers are from the held-out v2 benchmark (`FINAL_BENCHMARK_RESULTS/README.md`): a fictional fixture corpus,
  81 turns, one laptop, local `qwen3:4b`. Say "fixture data" once, on slide 12.
* Every number on a slide is in that README.

---

## Slide 1: StreamRAG, answers that start while you speak

* Samsung PRISM GenAI Hackathon, Theme 4: Streaming Live RAG
* One line: *retrieve while the user speaks, verify every statement, update only what changed.*

**Visual:** the demo UI with a live transcript, a cited answer and the timing line.

**Note:** Open with the demo screenshot, not with text.

## Slide 2: People don't speak in finished queries

* "What documents do I need, and how long does it take — sorry, for international students?"
* That one utterance holds two needs, a late detail and a correction.
* Answers are about rules, fees and deadlines, so a wrong number is worse than no answer.

**Visual:** the utterance as a timeline, with the needs and the correction highlighted.

## Slide 3: Why conventional RAG struggles

* It waits for silence, then runs one query.
* A multi-part question becomes one blurred query.
* A correction means starting over.
* There is no notion of which statement came from which source, or of outdated documents.

**Visual:** two timelines. Turn-based: silence, then a query, then an answer. StreamRAG: evidence and drafts during
speech.

## Slide 4: Our solution

1. **Streaming retrieval:** retrieve / wait / skip on every chunk.
2. **Needs, not queries:** multi-intent decomposition plus claim requirements per need.
3. **Incremental state:** an evidence lifecycle; late details and corrections re-retrieve only what changed.
4. **Verified answers:** every claim checked against its cited source; conflicts and superseded versions labelled.

**Note:** These four points are the whole talk. Everything else supports them.

## Slide 5: Architecture (§32 simplified)

```
Live input → Streaming → Intent → Adaptive retrieval → Evidence → Grounded generation → Verification → Streamed answer
                         ↑ session memory ↑        ↑ cache                     ↑ telemetry on every stage ↑
```

* Frozen pipeline: `docs/architecture/14_final_architecture.md`.
* Rule-based front end. Hybrid BM25 + bge-small retrieval. Local LLM. NLI verifier. An asynchronous runtime with
  cancellation.

**Visual:** the one-slide diagram from §14.3.

## Slide 6: Streaming controller

* A rule-based decision on every transcript chunk, costing milliseconds. An LLM here would cost hundreds of
  milliseconds per chunk (Phase 1).
* Early retrieval on **76 of 77** eligible held-out turns (98.7%; guide target ≥ 80%).
* First evidence **0.26 s** and first answer content **0.27 s** after the first word (p50). The batch version of the
  same pipeline takes 0.77 s and 2.48 s.

**Visual:** a stage-chip sequence with timestamps from demo scenario A.

## Slide 7: Needs, session memory and delta retrieval

* Multi-intent decomposition. On the held-out multi-intent turns, the intent count was right in 4 of 4.
* A ledger of queries per need. Evidence is active, retained, stale or superseded.
* Corrections are typed: refine, correct or extend. Only the affected need is searched again.
* Full re-retrieval vs delta: 1.49 vs 1.35 retrieval calls per turn, same answers.

**Visual:** demo scenario D: "New condition added", "Dropped an outdated search", updated answer.

## Slide 8: Adaptive retrieval without an LLM

* Claim requirements per need (an amount, a duration, a form) drive the plan.
* Plans: keyword fast path, filtered, semantic, hybrid, iterative, multi-hop or reuse. A bounded stop rule.
* On the held-out set, 55% of needs took the keyword fast path.
* The answer stage gets 2.2 times the evidence precision of top-5, with 0.83 embeddings per turn vs 1.0.

**Visual:** plan chips C1 ("Fast path: keyword lookup") and C2 ("Filtered search").

## Slide 9: Grounded, verified generation

* Claim plan → local LLM with structured output (or an extractive fallback).
* Every claim is checked by NLI against its cited section, then repaired or removed. Citations are rebuilt from the
  check.
* Conflicting notices: both reported. Superseded document: current value plus a labelled earlier version.
* Model-free check: **0 hallucinated values** vs 6–9% of claims for the RAG baselines.

**Visual:** an answer with citation chips and the source panel open.

## Slide 10: A runtime built for change

* Prioritised, deadline-bounded tasks. Bounded queues. Retries. Declared degraded modes.
* Cancellation on corrected questions: **−27% worker time**, LLM calls 15 → 9, final answers unchanged.
* Fault injection: retrieval, index, network, LLM error and LLM timeout faults all recovered; the timeout is fixed in
  Phase 11. Numbers are on slide 15.

**Visual:** a bar pair of worker time with and without cancellation.

## Slide 11: Live demo

Scenarios, in order:
* **A** normal question;
* **C1 / C2** adaptive retrieval;
* **B** multi-intent;
* **D** late correction;
* **D2** ASR revision;
* **E** citation click;
* **F1** contradiction;
* **F3** uncertainty.

* Runs offline. With no LLM it switches to verified extractive answers.
* An automated `demo-check` runs every scenario against declared expectations.

**Note:** Follow `DEMO_SCRIPT.md`. If something fails, start a new conversation and re-run; the scenarios are
deterministic.

## Slide 12: Evaluation protocol

* Phase 10 evaluated on a held-out set (v1), then inspected it in error analysis, so v1 became development data.
* **v2:** a new domain, 81 turns, all 14 query types. Written and hash-frozen **before** any Phase 11 change. Run
  once.
* Eleven baselines and variants. Paired tests (McNemar, Wilcoxon) and bootstrap CIs.
* **Fixture data, implementer labels. Not official results.**

## Slide 13: Results, latency and grounding

| | full system | naive RAG | hybrid + rerank |
|---|---|---|---|
| first evidence / first answer (p50) | 0.26 / 0.27 s | after the user stops | after the user stops |
| stale values | **0 / 13** | 3 / 13 | 2 / 13 |
| hallucinated-value claims | **0%** | 7.0% | 6.2% |
| evidence precision | **0.55** | 0.25 | 0.24 |

**Visual:** two bar charts: stale values and hallucinated values.

## Slide 14: Results, what did not improve

* Answer correctness: 0.72 vs 0.76 (naive) and 0.80 (hybrid + rerank). **Not significant.**
* Recall@5: 0.88 vs 0.99, because adaptive retrieval trades recall for precision.
* Verified answer after the user stops: 1.64 s vs 1.41 s (naive). Not significant, and not faster.
* Follow-up and correction turns in the new domain: 0.61, vs 0.72 in batch mode.
* Unanswerable questions recognised: 1 of 4.

**Note:** Say this slide plainly. It is the reason the other numbers are credible.

## Slide 15: Ablations

| removed or changed | effect (held-out v2) |
|---|---|
| streaming → batch (same pipeline) | first answer content 0.27 s → 2.48 s; answer correctness 0.72 vs 0.73 (2 vs 1 discordant turns) |
| adaptive retrieval → fixed top-5 (extractive answers) | stale values 0 → 7 / 13; evidence precision 0.55 → 0.24; Recall@5 0.94 → 0.98 |
| cancellation off | worker time +36% (16.0 → 21.8 s); LLM calls 9 → 15 |
| delta retrieval → full re-retrieval | retrieval calls 1.35 → 1.49 per turn |
| LLM answerability check on | unanswerable recognised 1 → 2 of 4, but answer correctness 0.72 → 0.66; kept off |
| fault injection (robustness) | see `FINAL_BENCHMARK_RESULTS/README.md` §6 |

## Slide 16: Samsung relevance (design, not a hardware claim)

* Voice-first products need incremental, correction-aware answers with sources.
* A natural device / server split:
  * **device:** the controller, decomposition and session state (rule-based, no model; the raw transcript stays
    local);
  * **server:** corpus search, the LLM and verification.
* Privacy: in the documented deployments nothing leaves the machine. The LLM is local, and egress is allowlisted.
* **No Samsung hardware, SDK or model was used. On-device execution was not measured.**

**Visual:** the edge / cloud table from `docs/security/README.md` §3.

## Slide 17: Limitations and next steps

* There is no official corpus yet; the results are fixture-only. There was no real ASR and no human evaluation.
* Weak spots: follow-ups in a new domain, unanswerable questions and multi-constraint recall.
* Next steps:
  * evaluate on the official corpus;
  * feed real ASR;
  * add a small learned rewriter for follow-ups;
  * re-tune the reranker and the answerability check on a fresh held-out set;
  * measure the front end on a device.

## Slide 18: Summary

* **Built:** a streaming, verified, incremental RAG system with a live demo, one-command Docker, and a reproducible,
  held-out benchmark.
* **Shown:** evidence in a quarter of a second, no stale or hallucinated values, cheaper corrections, explainable
  retrieval plans.
* **Honest:** not more accurate than strong baselines. The weaknesses are named.
* Thank you. Questions: see `JUDGE_QA.md`.
