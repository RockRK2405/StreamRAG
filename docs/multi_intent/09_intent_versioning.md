# 09: Intent Versioning and Delta Retrieval

**Code:** `intents/tracker.py` (`IntentTracker`); contracts `IntentSetDelta`, `IntentChange`, `Supersession`

## Reconciliation of a new draft with the previous version

| Draft vs previous | Result |
|---|---|
| Matches an active intent (content-term containment ≥ `refine_containment` 0.6, or Jaccard ≥ `match_jaccard` 0.5; one-to-one, best first) and nothing changed | Same id, same version; **no retrieval** |
| Matches, but text, constraints or inherited context changed | Same id, version + 1, `IntentChange{changed: [text \| constraints \| context]}`; **re-retrieve** |
| No match | New id: **ADDED → retrieve** |
| Superseded by a correction | `SUPERSEDED` (maps back to its existing id, so re-decomposing never mints a new one) |
| Previous active intent with no match | `DROPPED`, reason `no_longer_in_decomposition` |

- A new `IntentSet` version is created only when the delta is non-empty. Example: a final "." changes nothing, so no version is created.
- Constraint ids are stable by normalized text within the utterance.

## `IntentSetDelta`

```json
{"utterance_id": "u1", "version": 3, "previous_version": 2,
 "added": [], "modified": [{"intent_id": "I1", "from_version": 1, "to_version": 2, "changed": ["constraints"]},
                           {"intent_id": "I2", "from_version": 1, "to_version": 2, "changed": ["constraints"]}],
 "removed": [], "superseded": [], "constraints_added": ["K1"], "constraints_removed": [],
 "affected_intents": ["I1", "I2"]}
```

This is the actual V3 of the incremental example in docs/08 (tested in `test_6_incremental_intents_and_delta`).

## Delta retrieval rule

- Retrieve exactly the `affected_intents` whose new query text differs from their active query.
- Every other intent keeps its evidence.
- Unchanged queries are never re-sent: the streaming dev run counted **0** duplicate queries of the same intent.
- A modified intent's new query supersedes its previous version **in the ledger** (`relation: refines | replaces`). Other intents' lineages are untouched (per-intent supersession, `QueryLedger.create(intent_id=…)`).

**Phase 6 hand-off:**
- The tracker exposes `versions(utterance)`, `current(utterance)` and the per-intent lineage, so late-detail refinement can compute which claims are affected.
- `IntentSetDelta.affected_intents` is the delta Phase 6 needs.
