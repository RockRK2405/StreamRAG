# 07: Unsupported Claims: Policy, Repair, Retrieval Fallback, Validation Loop

**Code:** `validation/policy.py` (`decide`), `validation/repair.py` (`ClaimRepairer`, `gap_sentence`), `answer_state/engine.py` (`_process`, revision loop)

## Deterministic policy (brief §23)

| Verification | Claim | Action |
|---|---|---|
| SUPPORTED | any | **keep** (citations from the verification) |
| CONTRADICTED, also supported | any | **present_conflict**: both sides verbatim, each cited by its own evidence, no resolution |
| PARTIALLY_SUPPORTED | any | **keep_atoms**: only the verified atoms stay, each a claim of its own; the rest is rejected |
| UNSUPPORTED / CONTRADICTED | expresses planned facts | **restore_facts**: the planned, evidence-derived facts it claimed to express, verbatim |
| UNSUPPORTED | unplanned, *material* (number / amount / date) | **retrieve**: one fallback retrieval with the claim text, then keep if now supported, else remove |
| otherwise | | **remove** (kept in `rejected` with reasons) |

With `generation.repair: false` (ablation arm D), everything that is not supported is removed: no repair, revision, fallback or completion.

## Repairs never create facts (brief §24)

| Repair | What it does |
|---|---|
| keep_atoms | "X requires A and B and takes 30 days" → "X requires A.", "X requires B."; 30 days dropped |
| restore_facts | a distorted sentence is replaced by the facts it listed, verbatim |
| llm_rewrite (optional, `generation.llm_repair`, ablation) | one LLM call to rewrite using only the closest evidence sentences; the result is verified like any claim |
| qualify (gap sentences, deterministic templates) | `no_evidence`: "The retrieved documents do not contain an answer to “…”."; `value_not_stated`: "… do not state the value asked for in “…”."; `constraint_not_covered`: "… do not say how this applies to …" |

Qualifications are typed `uncertainty`, rendered as "Not established: …", never cited and never presented as fact.

## Retrieval fallback (brief §25)

- **Trigger:** an unsupported *material* claim, while `max_validation_retrievals` (1 per answer version) remains.
- **Query:** one bounded retrieval with the claim text through the Phase 3 service (top 3).
- **Storage:** new evidence is assigned to the need with rule `validation_retrieval`. The ledger is not touched, since this is not a new version of the need's query; `VALIDATION_RETRIEVAL` records it.
- **Outcome:** the claim is re-verified against the extended pool. Tested: the claim "Permits must be renewed every 2 years." was absent from an application-process pool, was found by fallback, and was kept with a citation to the new evidence. A second material claim in the same answer got no retrieval (budget).

## Validation loop and modes (brief §26, §34)

```
generate -> extract -> verify -> policy (keep / conflict / atoms / restore / retrieve / remove)
   strict:  removed factual content in a section whose planned facts are not all expressed -> regenerate the missing
            facts with feedback (REJECTED statements listed, kept sentences as ALREADY WRITTEN), at most
            max_answer_revision_attempts (2); a conflict side counts as expressing the planned fact it states
   both:    a missing *critical* planned fact -> regenerate; still missing -> added verbatim
   strict:  a section that produced unsupported content -> its missing planned facts added verbatim
-> consistency (doc 08) -> citations -> coverage -> status
```

- **Stop condition:** all critical claims are supported, or the budget is exhausted. The extractive completion guarantees the first.
- **Strict vs relaxed:** relaxed mode only revises for missing critical facts; non-critical unsupported content is just removed (tested).

## Failure handling (brief §59)

| Failure | Behaviour |
|---|---|
| LLM error / timeout | extractive rendering of the sections (`fallback = backend_error: …`) |
| schema violation | re-ask once, then extractive (`schema_invalid: …`) |
| verification cannot support a claim | the claim never appears as a fact |
| citation mapping fails for a claim | the claim is removed (`NO_CITATION`), never cited with an invented id |
| a need cannot be handled | `BLOCKED`, never `VALIDATED_FINAL` |
| retrieval cannot support a critical claim | the evidence is reported insufficient (uncertainty) |
