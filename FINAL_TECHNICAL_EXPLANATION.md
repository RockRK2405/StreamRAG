# Final Technical Explanation

How the frozen pipeline works, stage by stage. Each stage points to its module, its configuration (in
`configs/default.yaml` with `configs/profiles/final.yaml` on top) and its measured cost. The architecture diagrams are
in `docs/architecture/14_final_architecture.md`; the design decisions are in the ADRs in `docs/decisions/`.

## 0. Contracts and conventions

* **Data contracts.**
  * Every inter-stage object is a pydantic model with `extra="forbid"` (`src/streamrag/models/`).
  * JSON schemas are in `docs/schemas/`.
* **Events.**
  * Every observable step is an event on a per-session bus (`runtime/events.py`) carrying `session_id`,
    `utterance_id`, `correlation_id`, `causation_id`, `parent_event_id`, `state_version`, `seq` and timestamps.
  * Traces replay deterministically (`streamrag replay`).
  * On the held-out traces, every one of the 10,562 events carried all required fields in order.
* **Citations.** Citations render as `Doc_ID §Section`. Native document IDs and section numbers are kept. A citation
  is only ever produced from a verified claim–section pair.

## 1. Live input and stream ingestion

* **Input.**
  * `server/app.py` (HTTP + SSE) and `runtime/runtime.py` take transcript chunks with an optional `replaces` index,
    which carries ASR revisions, plus an end-of-utterance signal.
  * In replay and evaluation, `streaming/simulator.py` times chunks at 2.6 words per second.
* **Queues.**
  * Input queues are bounded (`runtime.queues.input: 64`). Chunks are capped at 4,000 characters.
  * Deltas already queued are coalesced (`runtime/backpressure.py`).
* **Per-session actor.**
  * One actor per session runs on the event loop.
  * Retrieval runs on a bounded thread pool (4 workers), the LLM on a pool of 1 (Ollama serves one request at a time),
    and NLI verification on 2.

## 2. Query and intent analysis

* **Retrieve / wait / skip** (`controller/`). On every chunk the controller does three things:
  1. classifies the dialogue act (`acts.py`): information request, meta, social, correction…;
  2. scores **specificity** (content tokens plus a corpus-vocabulary anchor with IDF ≥ 1.0), **semantic stability**
     (≥ 0.65) and **retrieval-worthiness** (≥ 0.5);
  3. checks novelty against earlier queries (term Jaccard).

  It retrieves provisionally when all three pass. There is a 400 ms cooldown and at most 4 retrievals per utterance,
  with one reserved for the final query.
  * Phase 11 fix: a meta phrase ("which documents…") counts as META only when no corpus-anchored word appears outside
    it.
* **Multi-intent decomposition** (`intents/`).
  * Rules split the request into intents at coordinators and question boundaries, and validate each one: it needs an
    anchor, and it must not be a fragment.
  * Intent sets are versioned. Only added or changed intents are retrieved while the user speaks.
* **Cost.** The controller and decomposition are rule-based, with no model, and cost milliseconds per chunk.
  Measured: early retrieval on 76 of 77 eligible held-out turns.

## 3. Session state, context changes and delta planning

* **Session memory** (`session/`, `context/`, `ledger/`, `delta/`) has four layers: transcript, needs (topic frames),
  the query ledger and the evidence store.
* **Context changes.** Each new utterance is classified as a typed context change: refine, correct, extend, retract,
  new topic or follow-up.
* **The delta planner decides per need.** Options: reuse active evidence, re-validate, re-query with the new
  constraint, or drop.
* **Evidence lifecycle.** Each piece of evidence is ACTIVE, RETAINED, REVALIDATION_REQUIRED, STALE, SUPERSEDED or
  INVALID. Only usable evidence (active, retained) reaches the answer.
* **Phase 11 early-commitment rules:**
  * when a refined query of the same utterance returns, evidence that only superseded partial-transcript queries
    retrieved is marked STALE (rule `not_confirmed_by_refined_query`);
  * when a need's query budget is exhausted, the need is re-validated against the latest completed query
    (`multi_retrieval/coordinator.py::_reuse_latest`) instead of being left pending.

## 4. Claim requirements and adaptive retrieval

* **Claim requirements** (`adaptive/analyzer.py`, `adaptive/requirements.py`). Per need, a query analysis yields its complexity (simple,
  moderate, complex, multi-hop) and **claim slots**: what the answer must establish, such as an amount, a duration, a
  date, a form, a condition or a document list.
* **Routing** (`adaptive/policy.py`, `adaptive/controller.py`). One plan per need:
  * `LEXICAL`: keyword fast path; exact IDs and simple questions;
  * `FAST_VECTOR`, `HYBRID` (BM25 + bge-small with RRF), `SEMANTIC`: vocabulary mismatch;
  * `FILTERED`: user-stated metadata such as applicant type, country or region;
  * `ITERATIVE`: k 5 → 10 → 20, with query expansion;
  * `MULTI_HOP`: a bridge entity, then a second search;
  * `CACHE_REUSE` / `SESSION_REUSE`: only when the validity signature matches (index hash, filters, intent version).
* **Stop rule** (`adaptive/stopping.py`, `adaptive/sufficiency.py`). Stop when ≥ 60% of the claim slots are covered by evidence, when the expected gain falls below
  0.05, when a contradiction is found, or when the budget is spent (5 queries, 40 results, 3 iterations, 1.5 s,
  2 hops).
* **Temporal and authority handling.** Superseded or expired documents are down-weighted, and the current version is
  preferred.
* **Measured.**
  * On the held-out set, 55% of needs took `LEXICAL`, 17% `HYBRID`, 13% `SEMANTIC`, 7% `FILTERED`, 5% `ITERATIVE` and
    3% `MULTI_HOP`.
  * Evidence precision 0.55 vs 0.24 for fixed top-5, with 0.83 embeddings per turn.

## 5. Retrieval and evidence fusion

* **Retrievers** (`retrieval/`).
  * BM25 with Snowball stemming.
  * Dense retrieval: bge-small-en-v1.5 (ONNX, CPU, about 3 ms per query embedding).
  * Reciprocal-rank fusion, plus deduplication of near-identical chunks.
* **Fusion** (`fusion/`).
  * Per-intent results are fused into one `UnifiedEvidenceSet` with intent-aware ranking.
  * Evidence is shared across intents and cross-intent duplicates are removed.
  * Conflicts are detected.
  * Phase 11: documents scoped to different groups (applicant type, country, region, product, language) are no longer
    reported as conflicting.
* **The cross-encoder reranker is implemented but off.** It costs 34–56 ms more per search, with no gain on the
  development data (Phase 10 §18).

## 6. Grounded generation

* **Claim plan** (`answer_state/engine.py`, `generation/`). From the usable evidence, a plan lists the sections, one
  per need, and the candidate facts for each.
* **Generation.**
  * A local `qwen3:4b` through Ollama, at temperature 0 with seed 7, writes the answer as structured JSON with sections
    and claims (JSON schema).
  * If the JSON is invalid, the model is asked once more. If that fails too, the extractive generator is used.
* **Claim extraction** (`generation/extraction.py`). Each generated sentence becomes a candidate claim, and inline
  markers become cited labels. A code or acronym the model mangled only in punctuation ("IEL:TS") gets the evidence
  spelling ("IELTS"); letters and digits never change.
* **Prompt safety.** Evidence is quoted and delimited as untrusted data. Instruction-like sentences are excluded from
  the facts.
* **Drafts.** While the user speaks, verified *extractive* drafts are streamed. The LLM answer follows when the
  utterance ends.
* **Fallbacks.**
  * An LLM error or timeout falls back to verified extractive answers.
  * Phase 11 fix: the extractive redo now gets its own deadline, so a timeout no longer ends the turn without an
    answer (robustness `llm_timeout`: 10 / 10 recovered).
* **Answerability flag.** An optional LLM check that marks a section "not answered by the evidence". It is **off**:
  on development data and on v2 it cost more correct answers than it saved.

## 7. Claim verification and citation validation

* **Verification** (`claims/`, `validation/`). Each claim is split into atomic facts (`claims/decomposer.py`; Phase 11
  fixed garbled "A and B of C" lists). Each fact is then checked in two steps:
  1. rules: values, numbers, dates and negation must appear in the evidence;
  2. an NLI entailment model, nli-deberta-v3-xsmall (ONNX), runs against the cited section and its neighbours.

  About 22 ms per claim against 5 sections (wall, median).
* **Policy** (`validation_mode: strict`).
  * An unsupported factual claim is repaired first: relabelled, reduced to its supported atoms, or qualified as "not
    established".
  * If it cannot be repaired, it is removed or the answer is blocked.
  * A critical unsupported claim triggers one validation retrieval.
* **Citations** (`citations/`).
  * Citations are rebuilt from the verification, never copied from the model.
  * Every citation must point to a section that entails its claim. Orphan citations are dropped.
* **Conflicts** (`answer_state/render.py`).
  * Genuinely conflicting sources are rendered as "The sources differ: …".
  * A conflict between a document and the one it supersedes is rendered as "Current version: … [k]. Earlier,
    superseded version: … [k2]."

## 8. Streaming response and session update

* **Answer versions.** The answer lane (`runtime/answers.py`) emits draft and final answer versions. Each version is a
  claim-level state (`answers/`, `answer_state/`) with diffs against the previous one.
* **Superseded answers.** An answer request for a superseded utterance is cancelled (`cancel_on_correction`).
  Stale results are rejected by a state-version check.
* **Session cache.** Validated claims and evidence are cached in the session for reuse by follow-ups.
* **UI mapping.** The server maps runtime events to safe UI events (`server/ui_events.py`): stages, plans, evidence,
  changes, cancellations, answers with citations, and TTFE / TTFA / TTVA metrics from the first chunk. The UI renders
  them as text only (`textContent`).

## 9. Asynchronous runtime (cross-cutting)

* **Scheduling** (`runtime/scheduler.py`).
  * Priority with aging. Final answer: 0. Final retrieval and validation retrieval: 1. Provisional retrieval and
    drafts: 2. Analytics: 3.
  * Aging is 500 ms per level.
* **Deadlines** (`runtime/timeouts.py`).
  * Turn budget 30 s, with reserves for generation (4.2 s) and validation (0.4 s).
  * Per-task timeouts: lexical 2 s, dense 3 s, generation 60 s.
* **Retries** (`runtime/retry.py`). Exponential backoff with jitter, at most 2 retries, idempotent tasks only.
* **Cancellation** (`runtime/cancellation.py`). Cooperative: superseded queued work is dropped and running work is
  signalled. On held-out corrections it cut worker time by 27% and LLM calls from 15 to 9.
* **Degraded modes** are always announced: lexical-only (no dense index), extractive (no LLM), rules-only (no NLI).
* **Fault injection** (`runtime/faults.py`). Robustness scenarios: retrieval failure, index failure, network
  transient, LLM error, LLM timeout, verifier failure and malformed query.

## 10. Server, packaging and operations

* **Server** (`server/`). A minimal HTTP/1.1 server on asyncio streams, with no web framework:
  * request limits: 16 KB headers, 64 KB bodies, 10 s timeout;
  * security headers: CSP `default-src 'self'`, `nosniff`, `DENY`, `no-referrer`;
  * SSE for events;
  * `/health` and `/ready`, which report index, models, the LLM and degraded state;
  * at most 8 sessions, with idle eviction after 15 minutes.
* **Configuration** (`config/settings.py`).
  * Precedence: defaults < profile < `STREAMRAG_*` environment < `--set` overrides.
  * The LLM URL must be loopback unless its host is in `generation.allowed_llm_hosts`.
* **Packaging.**
  * Docker: `python:3.12-slim`, non-root user, pinned dependencies, models and the demo index baked in, `HEALTHCHECK`
    on `/ready`.
  * Compose: the app, plus an optional Ollama sidecar under the `llm` profile.
* **Logging.** Structured JSON on stderr: request metadata, never transcript or answer text.

## 11. Evaluation instrument

* **Code** (`src/streamrag/evaluation/`, `experiments/`).
  * A runner replays each dataset turn through a system variant and stores raw runs and traces.
  * The instrument scores retrieval, evidence, claims, generation, citations, hallucination, latency and efficiency.
* **Answer correctness** means all labelled key strings are stated and no forbidden (stale) value is.
* **Hallucination** is model-free: a value that appears nowhere in the evidence or the question.
* **Verifier-judged metrics** use the system's own NLI model, so they are biased toward verified systems.
* **Statistics.** Paired McNemar and Wilcoxon tests with bootstrap CIs (`evaluation/stats.py`).
* **Final tables.** All final tables are generated from the stored files (`experiments/runners/final_tables.py`).
