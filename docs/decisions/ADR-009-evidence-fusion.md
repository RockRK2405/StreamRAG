# ADR-009: Evidence Fusion

- **Status:** Accepted
- **Date:** 2026-10-02
- **Spec:** §12

## Context

Fusion must merge evidence across sub-queries, reconcile redundancy and rank for relevance and factual density, without diluting the context window or introducing contradictory facts [G§2 p2]. Several intents compete for a limited token budget, which matters most for local CPU-LLM prefill.

## Decision

| Level | Method |
|---|---|
| **Within an intent** | RRF (k=60) over lexical and dense lists, and over multiple retrievals for the same intent (provisional + final) |
| **Across intents** | Quota round-robin: q=3 per intent; section cap 2 per intent; cross-intent dedup (one item tagged with several intents); near-duplicate collapse via precomputed groups; token budget 2,000 |
| **Conflicts** | Cheap typed-value check (number+unit, currency, duration, date) across documents within an intent → uncertainty `kind=conflict`, both sources cited. No hardcoded precedence. |
| **Labels** | Assigned by intent priority (order of mention), then rank |

## Alternatives

| Alternative | Why not chosen |
|---|---|
| Global RRF across intents | A strong intent crowds out weak ones |
| Global top-k by score | Scores are not comparable across queries |
| MMR over everything | Extra tuning; quotas plus caps give enough diversity |
| Clustering-based or LLM-based selection | Not justified under C5 |
| NLI-based contradiction detection | Kept as a possible extension |

## Why selected

It guarantees per-intent coverage, which G3 and per-intent uncertainty depend on. It is deterministic and needs no calibration.

## Trade-offs

- Fixed quotas can under-serve an intent that needs more evidence. q is tunable (Exp 10 family).
- The conflict heuristic is conservative and can miss non-numeric contradictions.

## Consequences

- `EVIDENCE_FUSED.per_intent` exposes coverage per intent for telemetry and evaluation.
- Carried evidence (refinement) is inserted before quotas so kept claims stay citable.
