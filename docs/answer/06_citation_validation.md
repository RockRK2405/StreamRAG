# 06: Citation Validation, Orphans and Coverage

**Code:** `citations/validator.py` (`CitationValidator`), `validation/coverage.py` (`AnswerCoverageValidator`, `CoverageReport`), `validation/metrics.py`

## CitationValidator (brief §31)

| Check | Failure kind |
|---|---|
| evidence exists in the session store | `missing_evidence` |
| chunk exists in the index | `unknown_chunk` (fabricated / stale id) |
| source document exists, and is the chunk's document | `unknown_document` |
| evidence text == indexed chunk text | `text_mismatch` |
| the citation's claim is a claim of the answer | `wrong_claim` |
| the claim's verification aligns STRONG / MODERATE with this evidence | `not_supporting` |
| page numbers equal the index's | `fabricated_page` |
| cited span lies inside the chunk | `bad_span` |

- Invalid citations are marked `invalid` and never rendered.
- A factual claim left without a valid citation is removed from the answer (`NO_CITATION` in `rejected`).
- All checks are tested with deliberately fabricated citations (`tests/citations/test_citation_integrity.py`).

## Orphans (brief §32), reported separately

- **orphan claims:** factual claims without a valid citation (always 0 in a final answer, by the rule above);
- **orphan citations:** citations whose claim is not in the answer;
- **unused evidence:** evidence of the answer's needs that no claim cites.

## Intent and evidence coverage (brief §18-19)

- **Covered:** a need of the frame is *covered* when its section has ≥ 1 supported factual claim.
- **Explicitly handled:** a need is *explicitly handled* when the section has an uncertainty statement instead.
- **Coverage failure:** a need that is neither, which blocks finalization (`BLOCKED`, `intent_not_handled:<id>`). The system never silently declares success.
- **Critical facts:** a planned *critical* fact not expressed is completed verbatim in both modes. If one is still missing, it blocks.
- **Maps:** the report carries claim → evidence, intent → claims and section → intent maps (complete traceability).

## Grounding metrics (brief §33): verifier-judged

`validation/metrics.py` computes, per answer version:

| Metric | Formula |
|---|---|
| raw_support_rate (CSR, pre-policy) | supported (incl. disputed evidence conflicts) / raw generated claims |
| raw_unsupported_rate | unsupported or contradicted without support / raw claims |
| raw_partial_rate | partially supported / raw claims |
| final_unsupported_rate | final factual claims without verified support / final claims (0 by construction) |
| citation_coverage (CC) | final claims with a valid citation / final claims |
| citation_precision | valid citations / citations created |
| evidence_utilization (EU) | evidence cited / usable evidence of the answer's needs |
| intent_coverage (IC) | needs covered or explicitly handled / needs |

These numbers state what the *verifier* decided. Their agreement with labelled data (perturbation set with labels by construction; hand-labelled LLM claims) is measured in `research/phase7` and reported separately (report §6, §16). They are never presented as ground truth.
