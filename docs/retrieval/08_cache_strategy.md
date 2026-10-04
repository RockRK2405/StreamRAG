# 08 · Cache strategy and invalidation (Phase 9)

Code: `cache.py` (`AdaptiveQueryCache`, `ValidatedClaimCache`), `integration.py` (Phase 6 changes, Phase 7 claims).

| Cache | Key / content | Reused when |
|---|---|---|
| query cache | sha1(analyzed terms of the contextual query, filters, validity date, claim-driven flag) → final evidence + validity signature (entity terms, constraint terms, per-document source signature, date) | the signature still holds |
| session evidence | Phase 6 `EvidenceStore` (ACTIVE / RETAINED evidence of active needs), read as a snapshot | every requirement is met → SESSION_REUSE (no search); some → delta requirements |
| validated claims | Phase 7 claims verified SUPPORTED → their evidence ids | their terms cover a requirement and the evidence is still usable |
| Phase 6 semantic cache | term-set key → earlier query's evidence (unchanged) | DeltaPlanner `cache_hit` |

The index hash is deliberately not in the query-cache key: each entry stores a source signature per document (hash of
its chunk texts + version / status / validity metadata), checked on every `get`, so an index update invalidates only
entries whose documents changed.

## Invalidation (each with a reason; RETRIEVAL_INVALIDATED)
| Reason | Trigger |
|---|---|
| `entity_changed` | Phase 6 CORRECTION / ENTITY_CHANGE / QUESTION_CHANGE: entries whose entity terms include a removed term (conservative: may over-invalidate, never under) |
| `constraint_changed` | Phase 6 CONSTRAINT_REMOVAL: entries whose constraint terms include the retracted constraint |
| `source_version_changed` | a document's source signature differs (content, version, status, validity) |
| `temporal_validity` | a document is no longer valid at the entry's date |
| `stale_evidence` | an evidence id is no longer usable in the session (Phase 6 lifecycle) |
| `corpus_changed` | explicit full flush for a new corpus snapshot |

Bounded LRU (`cache_max_entries`), per session (no cross-session sharing of user queries).
