# 07: Affected Claims and Targeted Revalidation

**Code:** `session/engine.py` (`interpret`, `on_result`), `claims/graph.py` (`ClaimRevalidator`)

## Claim statuses

`PENDING_VALIDATION` · `SUPPORTED` · `PARTIALLY_SUPPORTED` · `UNSUPPORTED` · `SUPERSEDED` · `STALE`. Every transition is a `ClaimTransition` (from, to, reason, change id, time) and is emitted as CLAIM_CREATED / CLAIM_INVALIDATED / CLAIM_REVALIDATED.

## Affected-claim detection (at interpretation time)

A claim's validity depends on the (evidence, **its own need**) assignment. A change therefore affects exactly:
1. the live claims (not STALE / SUPERSEDED) of the needs the change affects or supersedes;
2. the live claims of a need whose evidence assignment *for that need* changed status.

Claims of other needs that cite the same chunk are **not** affected.

This was a measured defect, fixed: the first version also re-checked other needs' claims on shared chunks. On the dev suite the incremental pipeline then re-validated more claims than the full restart (124 vs 100). A test now pins the targeting (`test_new_need_sharing_evidence_does_not_revalidate_other_needs_claims`).

Affected claims of still-active needs are set to `PENDING_VALIDATION` (`affected_by_change`). Claims of superseded needs are validated immediately (→ SUPERSEDED).

## Revalidation rules (`ClaimRevalidator.validate`, only for the targeted claims)

| Condition | → status | reason |
|---|---|---|
| need superseded (correction) | SUPERSEDED | `intent_superseded` |
| need removed | STALE | `intent_removed` |
| claim not selected for the need's current version | STALE | `not_selected_for_version` |
| no linked evidence usable; some REVALIDATION_REQUIRED | PENDING_VALIDATION | `awaiting_evidence_revalidation` |
| no linked evidence usable | UNSUPPORTED | `evidence_<status>` |
| usable, but lacks the terms of an active constraint | PARTIALLY_SUPPORTED | `does_not_address_constraint:K<n>` |
| usable and covers all active constraints | SUPPORTED | `verbatim_in_valid_evidence` |

When a result arrives for a need (`on_result`), the claims re-checked are:
- the need's newly selected claims;
- its previously selected claims;
- its claims on evidence whose assignment to this need changed.

**Invariant:** a claim is never left SUPPORTED when its evidence is no longer usable (tested: REVALIDATION_REQUIRED → PENDING, STALE → UNSUPPORTED).

## Example (generated, dev S01; `research/phase6/results/demo_trace.md`)

```
CH4 CONSTRAINT_ADDITION I1 +overnight
  C1 SUPPORTED -> PENDING_VALIDATION (affected_by_change)
  C2 SUPPORTED -> PENDING_VALIDATION (affected_by_change)
Q4 retrieved ...
  C1 PENDING_VALIDATION -> PARTIALLY_SUPPORTED (does_not_address_constraint:K1)   "...only when a second worker holds the base"
  C2 PENDING_VALIDATION -> SUPPORTED (verbatim_in_valid_evidence)                  "...not permitted in the orchard overnight..."
```

## Measured (dev suite, fixture domain, NOT REPORTABLE)

| | Incremental | Full restart |
|---|---|---|
| claim validations, all 64 turns | 158 | 161 |
| claim validations, 32 follow-up turns with a change | 88 | 91 |

The difference is small on these short sessions because most frames hold one need. On the 16-turn workload (`scaling.json`) the restart re-validates every need of the frame on every change.
