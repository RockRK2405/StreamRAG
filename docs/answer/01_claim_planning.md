# 01: Claim Planning and Answer Planning

**Code:** `claims/planner.py` (`ClaimPlanner`), `generation/answer_planner.py` (`AnswerPlanner`), `claims/models.py` (`ClaimPlan`, `IntentPlan`, `PlannedClaim`, `IntentGap`)
**Schema:** `docs/schemas/ClaimPlan.schema.json`

**Principle:** Evidence → Candidate Claims → Verification → Answer. The generator is never asked to "answer the question" freely and have citations found afterwards.

## ClaimPlanner

**Inputs.** Each input maps onto the brief's input contract as follows:
- **UnifiedEvidenceSet:** the usable (ACTIVE / RETAINED) evidence of each need, from the Phase 6 evidence store.
- **IntentSet:** the active needs of the topic frame.
- **SessionState:** the Phase 6 answer state (`AnswerVersion`). It has one section per need, with its extractive claims, constraints and uncertainty items.

**Output.** A `ClaimPlan`: per need, the planned facts `F1..Fn`, its gaps, and plan-level conflicts. Each planned fact records:

| Field | Meaning |
|---|---|
| `text` | the claim, **verbatim** evidence text or an atom of it |
| `source` | evidence id + character span of the source sentence |
| `evidence_ids` | evidence candidates (source first) |
| `phase6_claim_id`, `atom_of`, `depends_on` | provenance; atoms of one sentence depend on the first atom, whose subject they borrow |
| `importance` + `importance_basis` | see below |
| `order` | order of mention; atoms stay together |

**Facts are derived, never generated.**
- A sentence is split into atoms by the `ClaimDecomposer` (doc 03).
- The split is used only if the entailment check finds every atom entailed by the source sentence; otherwise the sentence stays whole.
- Excluded, never facts:
  - template text: sentences repeated verbatim in several documents (≥ 2 documents in a corpus of up to 4, otherwise ≥ 3);
  - instruction-like text: prompt-injection markers (doc 02).

**Importance** (deterministic):

| Importance | When |
|---|---|
| critical | the need has active constraints and the claim covers them (Phase 6 SUPPORTED), or, without constraints, the need's top-ranked claim |
| important | other SUPPORTED claims |
| supplementary | PARTIALLY_SUPPORTED claims: true, but they do not address an active constraint |

**Gaps.** Gaps are what the answer must say is *not established*:
- `no_evidence`, `constraint_not_covered` and `conflict` come from the Phase 6 section.
- `value_not_stated` is added when the need asks for a value of a kind and no planned fact:
  - states a value of that kind,
  - shares content terms with the question: ≥ 2 for a bare value ("2 years", "every week"); ≥ 1 for a value that names its own property ("18 years old" for an age; "40 crates" for "how many crates");
  - for "how many <noun>", counts that noun: a number within 3 words of it. "Permits must be renewed every 2 years" does not answer "How many renewals…".

  The kinds come from `configs/claim_lexicon.yaml` `value_questions`: "how long" → duration, "how much" / "fee" → amount, "when" → date or time, "minimum age" → age, and so on. A two-word cue starting with *what / which* also matches with one modifier in between ("which calendar date").

  Similar evidence is not an answer. For "How long does processing take?" the handbook's sentences about applications do not state a duration, so the answer reports that.

**Conflicts.** Phase 6 numeric conflicts between claims become plan conflict pairs. The answer presents both sides (doc 05).

## AnswerPlanner

The AnswerPlanner organises the plan and never adds information:
- **structure:** one section per need, in order of mention; multi-intent answers stay separated;
- **detail:** `detailed` includes every fact up to `max_claims_per_section` (6); `concise` includes critical and important facts only;
- **citations:** `per_claim`;
- **gaps:** passed to the deterministic uncertainty renderer; the generator is told not to write about them;
- **reuse:** sections the Phase 6 diff marks unchanged are not regenerated (doc 08).

Labels `E1..En` are assigned to the frame's evidence for the generator. Facts carry the labels of their own evidence.
