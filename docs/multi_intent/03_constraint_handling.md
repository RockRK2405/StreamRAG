# 03: Constraint Handling

**Code:** `intents/decomposer.py` (`_constraint`, `_scope_constraints`, `_distribute_trailing_pp`)
**Contract:** `Constraint` (`models/intents.py`)

## Representation

| Field | Meaning |
|---|---|
| `constraint_id` | Session-scoped `K1, K2, …`. Stable by normalized text within an utterance. |
| `kind` | `focus` ("especially …", "specifically …") · `condition` ("if …", "unless …") · `restriction` (a prepositional limit: "for visitors", "during a storm") |
| `marker`, `text` | The marker and the constraint's words, verbatim, with `source_span` |
| `scope` | `global` (applies to every active need of the utterance) or `local` |
| `applies_to` | Intent ids |
| `scope_reason`, `scope_confidence` | The rule that decided the scope, and how certain it is |

A constraint is never an intent (REQ-MI-004). A marker with no content yet ("especially during", mid-stream) is not a constraint yet.

## Scope rules (first match wins)

| # | Situation | Scope | Reason | Confidence |
|---|---|---|---|---|
| 1 | Trailing PP after a coordination whose conjuncts have their own topics ("the dome and the telescope **for visitors from abroad**") | global over those conjuncts | `trailing_pp_after_coordination` | 1.0 |
| 2 | Explicit "for both / for all of them / in both cases" | global | `explicit_all` | 1.0 |
| 3 | Focus marker followed by a question ("the harvest limits, **specifically how high the crates can be stacked**") | local, previous need | `focus_question_refines_previous` | 0.8 |
| 4 | Focus constraint sharing a term with exactly one need ("requirements for ladders, **especially the safety requirement**") | local | `shares_term_with_one_intent` | 0.9 |
| 5 | Restriction or condition after the needs ("… and also the lens **especially during a storm**") | global | `restriction_applies_to_request` | 1.0 with one need; **0.6 with ≥ 2 (ambiguous)** |
| 5b | Fronted restriction ("**For visitors,** what are the rules and when …") | global | `fronted_restriction` | 1.0 |
| 6 | Bare focus noun phrase after the needs | local, nearest preceding need | `nearest_preceding_intent` | 0.6 |

**Notes:**
- With a single need, scope cannot be ambiguous (confidence 1.0). `local` vs `global` then follows the rule *type*: a focus refinement is local, a restriction is global.
- Ambiguous scope (confidence < 1) lowers `decomposition_confidence` and is a gating signal (G-c) for the optional LLM check.
- The rule-5 choice (global) follows the brief's streaming example: "especially for category Z" re-queries both I1 and I2 ("delta retrieval if relevant").

## Global vs shared topic

A trailing PP after facet-only conjuncts ("the rules and the schedule **for pruning**") is the needs' shared **topic**, not a constraint. It is inherited by each conjunct as context (`distributed_pp`).

## Effect on queries and delta retrieval

- Constraint text in scope is appended to each affected intent's query, unless its terms are already present.
- A new or removed constraint changes `constraint_ids` of the intents in scope. Those intents become `MODIFIED` (version + 1), and only they are re-queried.

## Known gaps (measured in the Phase 5 report)

- **Cross-clause restrictions.** A restriction attached inside a later question clause without a marker ("… and how many can a picker fill **for the night shift**") stays with that clause; it is not distributed to the earlier question.
- **Topic vs restriction.** Shared topic and restriction are told apart only by whether the earlier conjuncts have their own topic; "cleaning and logging **for the lamp**" is classified as a restriction.
