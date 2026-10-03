# ADR-016: Adaptive Session RAG: Change Detection, Delta Planning, Evidence and Claim Lifecycles (Phase 6)

- **Status:** Accepted (Phase 6)
- **Date:** 2026-10-03
- **Refines:** ADR-005 (session state), ADR-006 (answer refinement), ADR-012 (events), ADR-015 (multi-intent)

## Context

Phase 5 handled several needs inside one utterance. Users also add, change and retract information **across** turns:
- "Specifically overnight."
- "Sorry, I meant crates instead of ladders."
- "Ignore the overnight restriction."

Re-running the pipeline per turn repeats retrieval, re-validation and answer generation that did not change. Constraints:
- no LLM backend is configured (ADR-007);
- every decision must be traceable and replayable;
- no fabricated confidence or performance numbers.

## Decision

1. **Session state in four separate layers** (transcript / semantic / retrieval / answer) behind a `SessionMemory` façade.
   - Every state change is a `SessionStateVersion` with parent, trigger, change ids and a deterministic snapshot.
   - No version is created for a no-op.
   - Snapshot / restore / reset / archive cover all layers plus the engine's own state.
2. **Change → impact → targeted update.**
   - `ContextChangeDetector` compares need versions semantically (terms, topic, aspect, constraint ids) and emits typed `ContextChange`s from a closed taxonomy of 9 types.
   - Change confidence = the minimum of computed signals.
   - `DeltaPlanner` maps each change to: queries to create (delta of the previous query) / reuse (active or semantic cache) / supersede; evidence actions; claims to revalidate.
3. **Late details are cross-turn operations on earlier needs** (set / update / retract), not new queries.
   - "What about Z?" is a constraint if the indexed corpus has Z together with the previous need's most specific topic term, else a parallel need inheriting only the aspect.
   - The decision is corpus-driven, not keyword-driven.
4. **Evidence has a per-need lifecycle with rule-named transitions.** Records are never deleted. Only ACTIVE / RETAINED evidence may support claims.
5. **Claims are extractive in Phase 6** (verbatim evidence sentences with spans), so SUPPORTS is justified by construction. Constraint coverage gives PARTIALLY_SUPPORTS, numeric disagreement gives potential CONTRADICTS. Revalidation is targeted to the claims of affected needs.
6. **Answer state is sectioned per need and versioned per topic frame.** Versions are committed only on a non-empty diff, and `needs_regeneration` marks the sections Phase 7 must re-render.
7. **Topic frames** bound context inheritance. Dormant frames are reactivated only by name.
8. **Memory safety at ingestion.** Interpretation sees redacted text; only the window keeps verbatim (redacted) utterances; archives keep hashes and counts.
9. **Streaming integration behind `session.enabled`.** The Phase 5 path is unchanged when the flag is off (tested).
10. **Telemetry.** 13 events: SESSION_VERSION_CREATED, CONTEXT_CHANGE_DETECTED, DELTA_PLAN_CREATED, QUERY_REUSED, QUERY_SUPERSEDED, EVIDENCE_RETAINED / _INVALIDATED / _REVALIDATED, CLAIM_CREATED / _INVALIDATED / _REVALIDATED, ANSWER_VERSION_CREATED / _UPDATED. Each carries its ids (change, plan, query, evidence + need, claim, answer + frame).

## Alternatives

| Alternative | Why not chosen |
|---|---|
| Full restart per turn | Implemented as the baseline. On the 16-turn fixture workload: 25 retrievals vs 9 and 2.7× the summed turn latency (159.1 vs 57.8 ms, measured, `scaling.json`). It also forgets which claims changed, so every turn re-renders everything |
| Concatenate the transcript into the query (full-transcript memory) | Measured: 8 queries carried finished topics' words, 81 vs 59 retrieval calls, lower need-query correctness (0.938 vs 0.984) |
| Keyword list for "constraint vs new question" | Brittle and domain-specific. The corpus co-occurrence test decides using the indexed documents themselves |
| LLM change classifier / claim extractor | No backend (ADR-007). Would put a non-deterministic step before retrieval. Possible Phase 7 addition behind validation |
| Delete invalidated evidence | Loses provenance and makes "undo" (a retraction) cost a retrieval. Status-only lifecycle plus semantic cache turns undo into a cache hit |
| Raw-string query cache | Misses reorderings ("orchard ladder rules"). Analyzed-term keys with index and options hashes never reuse across corpus or config changes |

## Consequences

- **Determinism.** Virtual traces replay exactly (28/28 dev sessions, including session events); a realtime trace replays with identical behaviour.
- **Costs incremental pays that a restart does not.** Evidence re-validation adds about 1 ms to delta planning on changed turns (measured p50 1.03 ms vs 0.08 ms). Savings come from avoided retrievals and claim work, and grow with session length and frame size.
- **Known gaps** (report §22–23):
  - phrasing-dependent cross-turn operations ("make that …", "not X, Y", "can you drop …");
  - anaphoric late details ("is that also true at night?");
  - non-numeric contradictions;
  - pattern-only PII redaction;
  - document-scoped delta retrieval not implemented.
