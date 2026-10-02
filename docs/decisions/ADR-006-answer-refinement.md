# ADR-006: Answer Refinement and Versioning

- **Status:** Accepted
- **Date:** 2026-10-02
- **Spec:** §14, §17; correction **K7** (presentation versions are our design choice)

## Context

When a late constraint arrives, the system must update the existing answer, keep established facts and prior citations, add delta citations and increment the version. It must not restart the session or re-execute full-corpus search [G§4 p4], [G§5 p5 G5]. Pitfall 2 is context loss [G§6 p5].

## Decision

1. Answers are **claim-level versions**:
   - claims have stable IDs across versions;
   - each version has a parent and a diff (kept, modified, added, retracted; citations ±; uncertainty ±).
2. On a refinement turn:
   - extract Δ (set / update / retract);
   - mark **affected** claims (intent scope, old-value mentions, dependencies);
   - issue **delta queries only** (topic ⊕ constraint), with ledger dedup, first scoped to the session's documents and widened to the corpus only if not covered.
3. **Unaffected claims are copied verbatim by code**, with their citations, and emitted first. The LLM can only keep, modify or retract *affected* claims and add new ones.
4. Presentation turns produce a version with `kind=presentation`: same claim IDs, re-rendered text, citations ⊆ parent. This is our design choice (K7); the guide does not specify it.
5. Commits are serialized per session, so lineage is linear. Constraints arriving mid-synthesis are queued and applied as v(n+1).

## Alternatives

| Alternative | Why not chosen |
|---|---|
| Restart with concatenated turns | G5 violation; drift (kept as the Exp 6 baseline arm) |
| Let the LLM rewrite the whole answer from v1 + new evidence | Cannot guarantee preserved facts |
| Version only the text, not claims | Cannot compute affected claims or verify retention |

## Why selected

Deterministic carry-over turns "preserve established facts" from a hope into a guarantee. It also makes G5 provable from telemetry and gives refinement turns an almost immediate first validated sentence.

## Trade-offs

- The affected-claim heuristic can over-mark (extra LLM work) or under-mark (a stale claim survives). The stale-fact rate is measured in Categories 6 and 7.
- Verbatim reuse can make prose less fluent. Connective phrases are allowed without citations.

## Consequences

- Claims need intent and constraint provenance (`introduced_by_constraint`).
- The test suite needs gold affected-intent labels (§22).
