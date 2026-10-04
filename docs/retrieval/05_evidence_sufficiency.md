# 05 · Claim-driven retrieval and evidence sufficiency (Phase 9)

Code: `requirements.py` (`RequirementBuilder`), `sufficiency.py` (`EvidenceSufficiencyEvaluator`), contracts
`EvidenceRequirement`, `SufficiencyAssessment`.

## Intent → claim requirements → evidence requirements
Before searching, the need is turned into the claim slots its answer must fill:

| Slot | Built from | Met by evidence that … |
|---|---|---|
| main (`value` / `fact`) | the need's content terms; a value slot when the question asks a value kind | covers ≥ `requirement_coverage` of the terms, contains the **anchor** (most specific term; an unknown content word is most specific), contains every **key** term (identifiers, multi-hop entity / target), and — value slots — has a sentence with a value of the asked kind (`has_value_of`) |
| condition | each active constraint / user-stated metadata value | also contains the constraint words, or comes from a document whose metadata names the value; a document "for all" meets a *value* condition (the general value applies) but not a fact / list condition |
| comparison item | each compared entity | as main, per entity |
| bridge / link | added by a hop (07) | link: "<entity> … <target>"; bridge: the asked aspect for the target |

Term matching accepts corpus-defined acronym equivalents, the rewrite's synonym expansions, and stem variants (one
analyzed stem a prefix of the other, ≥ 4 characters). Validity: evidence from a document not valid at the question's
date (explicit, else the reference date for current / undated questions; "past" questions are not filtered) cannot
meet a slot and is reported in `excluded` (`temporal_validity`). Applicability: a need constrained to a metadata value
cannot use a document naming a different specific value (`not_applicable`).

## States
`CONTRADICTORY` (a value slot has unresolved differing values, see 09) › `SUFFICIENT` (every slot MET) › `PARTIAL`
(some) › `INSUFFICIENT` (none). A slot is *unattainable* when its terms cannot reach the threshold anywhere in the index
(vocabulary check) — the stopping policy then stops at once (10).

`claim_driven: false` (ablation) = one slot with all content terms.

## Integration with Phase 7
The same claim lexicon (value kinds) and value checks as the Phase 7 claim planner; Phase 7 verified claims feed the
`ValidatedClaimCache` (08); bridge targets of completed hops are added to the need's claim-selection terms in the
Phase 6 engine (`retrieval_terms`) so the answer stage can use multi-hop evidence (sentences relevant *only* through the
bridge words are dropped: they are about another member of the class).

## Limitations
Sufficiency is lexical: a paraphrase that shares no words ("old cars … checked" vs "vehicles … inspection") is judged
INSUFFICIENT even when dense retrieval found it (the evidence is still handed on as context; the cost is extra
searches). It is a gate for *retrieval*, not a verifier: the Phase 7 NLI verifier remains the final judge of support.
