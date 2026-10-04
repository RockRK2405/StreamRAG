# 09 · Contradiction-aware retrieval and temporal conflict resolution (Phase 9)

Code: `sufficiency.py` (`_conflict`, `_comparable`), `controller.py` (`contradiction_search`), `catalog.py`
(validity, agreed supersession, authority).

**Detection.** For a value slot, the values of the asked kind (`values.py`: amounts, durations, clock times, ages,
measures, counts — normalised) in each supporting chunk's best-matching sentence(s) are compared across chunks of
*different* documents. A clash is a pair with disjoint value sets. Not compared: documents for different populations
(different specific values of a filter field, e.g. domestic vs international), evidence that does not meet the need's
conditions, and one-word slots (subject unspecified: "what about the fee?").

**Resolution** (first rule that decides; recorded in `resolution`):
1. *temporal validity* — handled before conflicts: evidence from documents not valid at the question's date never
   supports the slot (`excluded: temporal_validity`); publication date is never used as validity start;
2. *agreed supersession* — "A supersedes B" counts only if B agrees (status superseded / expired / archived, or its
   validity ends before A starts); current questions keep A, **"past" questions keep B** (`past_version`);
3. *authority* — configured weights (status × doc_type) differing by ≥ 0.3.
Losers are only chunks whose values disagree with every kept value (a chunk stating both — "raised from 40 to 55" —
stays). Old evidence is never discarded without one of these checks.

**Targeted retrieval.** An unresolved clash triggers one `contradiction_search`: the slot's query + "current version
effective", restricted to documents *other than* the conflicting ones (a third source), next k. If the clash remains,
the loop stops with `CONTRADICTION` and both sides are handed on — the Phase 7 answer reports the conflict.
`contradiction_retrieval: false` (ablation) stops at once.
