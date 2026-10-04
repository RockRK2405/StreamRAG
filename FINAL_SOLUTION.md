# Final Solution: StreamRAG

Five lengths of the same explanation. Every version makes the same claims, and every number is from the held-out v2
benchmark (`FINAL_BENCHMARK_RESULTS/README.md`): fictional fixture corpus, 81 turns, one laptop, local LLM. The numbers
are **not** official Theme 4 results.

## One sentence

StreamRAG answers spoken, multi-part questions from trusted documents while the user is still talking, verifies every
statement against the section it cites, and updates only what changed when the user adds a detail or corrects
themselves.

## Three sentences

StreamRAG retrieves while the user speaks, splits the request into separate needs, and keeps a per-need record of
queries and evidence, so a late detail or correction re-retrieves only the affected need. Every claim in a draft or
final answer is checked against its cited section by an entailment model, and conflicting or superseded sources are
labelled instead of merged. On a held-out set it delivered first evidence about 0.26 s after the first word and stated
no stale or unsupported values, where the baselines did in 6–54% of the relevant cases. Its answer correctness was not
better than the baselines'.

## 30 seconds

* **Problem.** People don't speak finished queries: they ask several things at once, add details late, and correct
  themselves. Conventional RAG waits for silence, runs one blurred query and starts over on every correction.
* **Solution.** StreamRAG listens to the live transcript.
  * It decides when to retrieve, splits the request into needs and picks a retrieval plan per need, from a cheap
    keyword lookup up to multi-hop.
  * It streams an answer whose every claim is verified against its source.
  * When the user corrects themselves, only the changed need is searched again, and obsolete work is cancelled.
* **Held-out test result.** Evidence arrives in about a quarter of a second. Stale and hallucinated values went to
  zero. Corrections used 27% less compute.

## 2 minutes

**The problem.** In a voice assistant the question arrives as a stream: "What documents do I need, and how long does
it take — sorry, for international students?" That one utterance holds two needs and a late correction. Turn-based
RAG has three problems here:
1. it waits for the end;
2. it sends one query that blurs both needs;
3. when the correction arrives, it starts over.

It also cannot tell the user which statement came from which document, or whether two documents disagree.

**What StreamRAG does.**
1. **Live input.** A rule-based controller decides, chunk by chunk, whether to retrieve, wait or skip.
2. **Needs.** It splits the request into needs and gives each need a list of what it must establish: an amount, a
   duration, a form.
3. **Adaptive retrieval per need.** A cheap router picks the plan: keyword fast path, filtered, semantic, iterative or
   multi-hop. It stops when the requirements are covered. No LLM is in this loop.
4. **Evidence lifecycle.** Every piece of evidence is active, retained, stale or superseded. A correction re-retrieves
   only the affected need. Evidence that only a partial transcript retrieved is dropped when the full sentence
   arrives.
5. **Verified streaming answer.**
   * A local LLM (or an extractive fallback) drafts the answer while the user speaks.
   * Every claim is checked against its cited section by an entailment model. Citations come from that check.
   * Conflicting notices are both reported, and superseded versions are labelled.
6. **Asynchronous runtime.** Prioritised, deadline-bounded tasks. Superseded work is cancelled. Degraded modes are
   always declared.

**What we measured** on a held-out set written before the final changes (81 turns, new domain):
* First evidence 0.26 s and first answer content 0.27 s after the first word (p50). The batch version of the same
  pipeline takes 0.77 s and 2.48 s.
* Zero stale values (baselines 15–54%) and zero hallucinated values (baselines 6–9%).
* About twice the evidence precision.
* Early retrieval on 98.7% of eligible turns.
* Cancellation cut worker time by 27% on corrected questions, with the same final answers.

**What did not improve.**
* Answer correctness (0.72) was not significantly different from the baselines (0.76–0.80).
* Retrieval recall is lower.
* Follow-up questions in a new domain are a weakness.

All of this is in `LIMITATIONS.md`.

## 5 minutes

### 1. The problem (45 s)

Spoken interaction breaks three assumptions of RAG:
1. that the query is complete;
2. that it asks one thing;
3. that it does not change.

Theme 4 asks for a pipeline that does the following:
* retrieves while the user is still speaking;
* decomposes multi-intent requests;
* fuses the evidence into one grounded, cited answer;
* refines that answer when late details arrive, without restarting.

The answers concern rules, fees and deadlines. A wrong number is worse than no answer, so every claim must be traceable
to a section and checked.

### 2. Architecture (90 s)

The frozen pipeline is described in `docs/architecture/14_final_architecture.md`:

```
live input → stream ingestion → query / intent analysis → session state → claim requirements →
adaptive retrieval → evidence fusion → evidence lifecycle → grounded generation → claim verification →
citation validation → streamed answer → session update
```

* **Streaming controller (Phases 4–5).** Rule-based retrieve / wait / skip and multi-intent decomposition. Each costs
  milliseconds; an LLM on this path would cost hundreds of milliseconds per chunk (Phase 1).
* **Session memory and delta retrieval (Phase 6).**
  * A ledger records queries per need.
  * Each piece of evidence has a lifecycle.
  * Context changes are typed: refine, correct, extend.
  * Only affected needs are searched again.
* **Adaptive retrieval (Phase 9).** Claim requirements drive the plan per need, with a bounded stop rule. The plan is
  visible in the UI.
* **Grounding (Phase 7).**
  * The claim plan feeds structured generation by a local `qwen3:4b`.
  * NLI verification runs on every claim, followed by repair or removal.
  * Citations are rebuilt from the verification.
  * Conflict and version handling: superseded versions are labelled.
* **Runtime (Phase 8).**
  * Prioritised task pools with deadlines.
  * Cooperative cancellation and state-version checks.
  * Backpressure.
  * Retries, and degraded modes: lexical-only, extractive, rules-only.
* **Phase 11.**
  * Fixes for early commitment, budget exhaustion, LLM-timeout fallback, meta-question misclassification, claim
    decomposition and false conflicts.
  * An HTTP / SSE server with the live demo UI, `/health` and `/ready`, Docker packaging and environment
    configuration.

### 3. Evidence (90 s)

**Protocol.**
* Phase 10's test set was inspected during its error analysis, so it became development data.
* A new held-out set (v2: utility services, 81 turns, all 14 query types) was written and hash-frozen before any
  Phase 11 change.
* Fixes were developed on other data. The final system and eleven baselines and variants were run once on v2.

**Results** (full system vs baselines; p-values from paired tests):

| measure | full system | baselines | difference |
|---|---|---|---|
| time to first evidence (p50) | 0.26 s | batch version of the same pipeline: 0.77 s | |
| time to first answer content (p50) | 0.27 s | batch version: 2.48 s | |
| stale / forbidden values asserted | 0 / 13 | 2–4 / 13 for RAG baselines, 7 / 13 for fixed-retrieval streaming | |
| hallucinated values | 0 | 6–9% of claims | p ≤ 0.008 |
| claims supported by their citation (verifier) | 100% | 79–84% | |
| evidence precision | 0.55 | 0.24–0.25 | |
| cancellation on corrections: worker time | −27% | | |
| cancellation on corrections: LLM calls | 15 → 9 | | |
| cancellation on corrections: final-answer correctness | unchanged | | |
| answer correctness | 0.72 | 0.76 / 0.75 / 0.80 | not significant |
| Recall@5 | 0.88 | 0.96–0.99 | |
| time to verified answer after the user stops (p50) | 1.64 s | naive RAG: 1.41 s | |
| follow-up and correction turns | 0.61 | batch mode: 0.72 | weakness |

### 4. Why it matters (45 s)

* **Voice assistants.** Users speak incrementally and correct themselves, so the incremental, need-level state is
  the core of the design.
* **Device and server split.** The parts that read the raw transcript continuously are cheap and model-free: the
  controller, decomposition and session state. They suit a device. Retrieval over a large corpus, the LLM and
  verification suit a server. On-device execution was **not measured**, and no Samsung hardware, SDK or model was
  used.
* **Privacy.** In the documented deployments nothing leaves the machine. The LLM is local, and egress is allowlisted.

### 5. Honest summary (30 s)

**Built and measured:** a streaming, verified, incremental RAG system with a working demo, Docker packaging and a
reproducible benchmark.

**Strengths:** early evidence, no stale or hallucinated values, explained adaptive retrieval, and cheaper corrections.

**Not achieved:** better answer correctness than strong baselines, and robust follow-up handling in a new domain.

**Next:** evaluate on the official corpus, add real ASR input, add a learned query rewriter for follow-ups, and
re-tune the reranker on a fresh held-out set.
