# Phase 7 report tables (generated from research/phase7/results*/; NOT REPORTABLE, fixture dev suite)

## Ablation by arm: first run

| metric | A | B | C | D | E | F | X |
|---|---|---|---|---|---|---|---|
| required-fact recall | 11.8 % | 73.5 % | 79.4 % | 58.8 % | 58.8 % | 100.0 % | 100.0 % |
| raw claims | 36 | 56 | 85 | 85 | 85 | 98 | 121 |
| raw support (verifier) | 8.3 % | 89.3 % | 65.9 % | 65.9 % | 65.9 % | 98.0 % | 100.0 % |
| raw unsupported (verifier) | 80.6 % | 8.9 % | 31.8 % | 31.8 % | 31.8 % | 2.0 % | 0.0 % |
| final claims | 33 | 56 | 83 | 56 | 56 | 101 | 131 |
| final unsupported (verifier) | 90.9 % | 10.7 % | 32.5 % | 0.0 % | 0.0 % | 0.0 % | 0.0 % |
| citation coverage | – | – | 100.0 % | 100.0 % | 100.0 % | 100.0 % | 100.0 % |
| citation precision (verifier) | – | – | 65.9 % | 100.0 % | 100.0 % | 100.0 % | 100.0 % |
| intent coverage | – | – | – | – | – | 100.0 % | 100.0 % |
| forbidden assertions | 3 | 0 | 0 | 0 | 0 | 0 | 0 |
| uncertainty when expected | 10.0 % | 0.0 % | 20.0 % | 20.0 % | 20.0 % | 40.0 % | 40.0 % |
| conflict presented | 0.0 % | 0.0 % | 66.7 % | 66.7 % | 66.7 % | 100.0 % | 100.0 % |
| LLM calls | 33 | 33 | 33 | 33 | 33 | 37 | 0 |
| generation ms p50 / p95 | 1086 / 2150 | 1400 / 3707 | 2097 / 6520 | 0 / 0 | 0 / 0 | 2130 / 4991 | 0 / 0 |
| validation ms p50 / p95 | 151 / 742 | 63 / 394 | 40 / 631 | 11 / 36 | 10 / 47 | 92 / 364 | 25 / 400 |
| total ms p50 / p95 | 1268 / 2639 | 1405 / 4033 | 2156 / 6636 | 11 / 37 | 10 / 47 | 2206 / 5257 | 25 / 400 |

## Full system (F) by category: first run

| category | turns | gold required fact recall | raw claim support rate | final unsupported rate | uncertainty expressed when expected | conflict presented rate | forbidden assertions |
|---|---|---|---|---|---|---|---|
| A_fully_supported | 4 | 100.0 % | 100.0 % | 0.0 % | – | – | 0 |
| B_partially_supported | 2 | 100.0 % | 100.0 % | 0.0 % | 100.0 % | – | 0 |
| C_unsupported_claim | 4 | – | 100.0 % | 0.0 % | 25.0 % | – | 0 |
| D_multi_intent | 3 | 100.0 % | 100.0 % | 0.0 % | – | – | 0 |
| E_contradictory | 3 | – | 86.7 % | 0.0 % | – | 100.0 % | 0 |
| F_incremental_refinement | 6 | 100.0 % | 100.0 % | 0.0 % | 100.0 % | – | 0 |
| G_entity_correction | 4 | 100.0 % | 100.0 % | 0.0 % | – | – | 0 |
| H_missing_evidence | 3 | – | 100.0 % | 0.0 % | 0.0 % | – | 0 |
| I_citation_integrity | 2 | 100.0 % | 100.0 % | 0.0 % | – | – | 0 |
| J_long_answer | 2 | 100.0 % | 100.0 % | 0.0 % | – | – | 0 |

## Hallucination tags: first run

| tag | turns | arm A forbidden / raw unsupported | arm C forbidden / final unsupported | arm F forbidden / final unsupported | F uncertainty when expected |
|---|---|---|---|---|---|
| ambiguous_evidence | 1 | 0 / 100.0 % | 0 / 0.0 % | 0 / 0.0 % | – |
| contradictory_documents | 3 | 0 / 100.0 % | 0 / 0.0 % | 0 / 0.0 % | – |
| incomplete_evidence | 3 | 1 / 100.0 % | 0 / 60.0 % | 0 / 0.0 % | 100.0 % |
| missing_date | 1 | 1 / 100.0 % | 0 / 0.0 % | 0 / 0.0 % | 0.0 % |
| missing_eligibility_rule | 1 | 0 / – | 0 / – | 0 / 0.0 % | 0.0 % |
| missing_numeric_value | 6 | 2 / 100.0 % | 0 / 18.2 % | 0 / 0.0 % | 50.0 % |
| missing_rule | 2 | 0 / 50.0 % | 0 / 0.0 % | 0 / 0.0 % | 0.0 % |
| prompt_injection | 2 | 0 / 50.0 % | 0 / 0.0 % | 0 / 0.0 % | – |
| similar_but_not_supporting | 2 | 0 / 100.0 % | 0 / 0.0 % | 0 / 0.0 % | 50.0 % |

## Revision: first run

| mode | later turns | LLM calls | claims kept | required recall | forbidden | wall ms p50 / p95 |
|---|---|---|---|---|---|---|
| full_restart | 5 | 5 | 40.0 % | 100.0 % | 0 | 2352 / 3569 |
| incremental | 5 | 3 | 46.7 % | 100.0 % | 0 | 995 / 2453 |

## F stage latency (ms): first run

| stage | n | p50 | p95 | max |
|---|---|---|---|---|
| citation_mapping | 33 | 0.0 | 0.1 | 0.1 |
| claim_extraction | 33 | 0.0 | 0.1 | 0.1 |
| claim_planning | 33 | 0.2 | 6.1 | 15.4 |
| claim_verification | 33 | 91.4 | 358.9 | 411.0 |
| generation | 33 | 2130.0 | 4990.9 | 6656.6 |
| repair | 33 | 0.0 | 12.3 | 25.2 |
| retrieval | 33 | 0.0 | 0.0 | 0.0 |
| total | 33 | 2206.4 | 5256.9 | 6759.0 |
| validation | 33 | 0.2 | 3.3 | 9.6 |
| raw first token | 31 | 102.0 | 193.8 | 312.5 |

Measured tokens (F, all new versions): prompt 13119, output 6025.

## Verifier on perturbations (labels by construction): first run

| verifier | items | accuracy | support precision | support recall | unsupported detected | ms / claim |
|---|---|---|---|---|---|---|
| nli | 125 | 89.6 % | 78.4 % | 95.2 % | 86.8 % | 87.4 |
| rules | 125 | 88.0 % | 73.7 % | 100.0 % | 81.9 % | 5.6 |

| perturbation | n | nli accepted | rules accepted |
|---|---|---|---|
| modality_or_negation_changed | 15 | 4 | 0 |
| negation_inserted | 16 | 1 | 10 |
| number_changed | 10 | 0 | 0 |
| original | 42 | 40 | 42 |
| unsupported_conjunct | 42 | 6 | 5 |

## Ablation by arm: post-fix run

| metric | A | B | C | D | E | F | X |
|---|---|---|---|---|---|---|---|
| required-fact recall | 11.8 % | 100.0 % | 97.1 % | 97.1 % | 97.1 % | 100.0 % | 100.0 % |
| raw claims | 59 | 135 | 109 | 109 | 109 | 106 | 121 |
| raw support (verifier) | 15.2 % | 97.8 % | 98.2 % | 98.2 % | 98.2 % | 98.1 % | 100.0 % |
| raw unsupported (verifier) | 78.0 % | 2.2 % | 0.0 % | 0.0 % | 0.0 % | 1.9 % | 0.0 % |
| final claims | 57 | 135 | 107 | 107 | 107 | 104 | 131 |
| final unsupported (verifier) | 84.2 % | 2.2 % | 0.0 % | 0.0 % | 0.0 % | 0.0 % | 0.0 % |
| citation coverage | – | – | 100.0 % | 100.0 % | 100.0 % | 100.0 % | 100.0 % |
| citation precision (verifier) | – | – | 98.2 % | 100.0 % | 100.0 % | 100.0 % | 100.0 % |
| intent coverage | – | – | – | – | – | 100.0 % | 100.0 % |
| forbidden assertions | 3 | 0 | 0 | 0 | 0 | 0 | 0 |
| uncertainty when expected | 10.0 % | 0.0 % | 20.0 % | 20.0 % | 20.0 % | 60.0 % | 60.0 % |
| conflict presented | 0.0 % | 0.0 % | 66.7 % | 66.7 % | 66.7 % | 100.0 % | 100.0 % |
| LLM calls | 33 | 33 | 33 | 33 | 33 | 38 | 0 |
| generation ms p50 / p95 | 1166 / 2434 | 1459 / 3939 | 2124 / 6441 | 0 / 0 | 0 / 0 | 2197 / 6077 | 0 / 0 |
| validation ms p50 / p95 | 177 / 943 | 51 / 352 | 26 / 359 | 13 / 54 | 13 / 53 | 73 / 429 | 19 / 267 |
| total ms p50 / p95 | 1391 / 3422 | 1509 / 4082 | 2202 / 6483 | 13 / 55 | 13 / 53 | 2270 / 6341 | 19 / 267 |

## Full system (F) by category: post-fix run

| category | turns | gold required fact recall | raw claim support rate | final unsupported rate | uncertainty expressed when expected | conflict presented rate | forbidden assertions |
|---|---|---|---|---|---|---|---|
| A_fully_supported | 4 | 100.0 % | 100.0 % | 0.0 % | – | – | 0 |
| B_partially_supported | 2 | 100.0 % | 100.0 % | 0.0 % | 100.0 % | – | 0 |
| C_unsupported_claim | 4 | – | 77.8 % | 0.0 % | 75.0 % | – | 0 |
| D_multi_intent | 3 | 100.0 % | 100.0 % | 0.0 % | – | – | 0 |
| E_contradictory | 3 | – | 100.0 % | 0.0 % | – | 100.0 % | 0 |
| F_incremental_refinement | 6 | 100.0 % | 100.0 % | 0.0 % | 100.0 % | – | 0 |
| G_entity_correction | 4 | 100.0 % | 100.0 % | 0.0 % | – | – | 0 |
| H_missing_evidence | 3 | – | 100.0 % | 0.0 % | 0.0 % | – | 0 |
| I_citation_integrity | 2 | 100.0 % | 100.0 % | 0.0 % | – | – | 0 |
| J_long_answer | 2 | 100.0 % | 100.0 % | 0.0 % | – | – | 0 |

## Hallucination tags: post-fix run

| tag | turns | arm A forbidden / raw unsupported | arm C forbidden / final unsupported | arm F forbidden / final unsupported | F uncertainty when expected |
|---|---|---|---|---|---|
| ambiguous_evidence | 1 | 0 / 100.0 % | 0 / 0.0 % | 0 / 0.0 % | – |
| contradictory_documents | 3 | 0 / 100.0 % | 0 / 0.0 % | 0 / 0.0 % | – |
| incomplete_evidence | 3 | 1 / 83.3 % | 0 / 0.0 % | 0 / 0.0 % | 100.0 % |
| missing_date | 1 | 1 / 100.0 % | 0 / 0.0 % | 0 / 0.0 % | 0.0 % |
| missing_eligibility_rule | 1 | 0 / – | 0 / – | 0 / 0.0 % | 100.0 % |
| missing_numeric_value | 6 | 2 / 85.7 % | 0 / 0.0 % | 0 / 0.0 % | 83.3 % |
| missing_rule | 2 | 0 / 80.0 % | 0 / 0.0 % | 0 / 0.0 % | 0.0 % |
| prompt_injection | 2 | 0 / 50.0 % | 0 / 0.0 % | 0 / 0.0 % | – |
| similar_but_not_supporting | 2 | 0 / 87.5 % | 0 / 0.0 % | 0 / 0.0 % | 50.0 % |

## Revision: post-fix run

| mode | later turns | LLM calls | claims kept | required recall | forbidden | wall ms p50 / p95 |
|---|---|---|---|---|---|---|
| full_restart | 5 | 5 | 40.0 % | 100.0 % | 0 | 2563 / 3408 |
| incremental | 5 | 3 | 46.7 % | 100.0 % | 0 | 1124 / 2570 |

## F stage latency (ms): post-fix run

| stage | n | p50 | p95 | max |
|---|---|---|---|---|
| citation_mapping | 33 | 0.0 | 0.1 | 0.1 |
| claim_extraction | 33 | 0.1 | 0.2 | 0.2 |
| claim_planning | 33 | 0.3 | 6.1 | 14.2 |
| claim_verification | 33 | 71.6 | 424.9 | 653.6 |
| generation | 33 | 2196.8 | 6077.2 | 8448.3 |
| repair | 33 | 0.0 | 24.6 | 510.9 |
| retrieval | 33 | 0.0 | 0.0 | 0.0 |
| total | 33 | 2269.9 | 6341.3 | 8541.9 |
| validation | 33 | 0.2 | 2.4 | 5.7 |
| raw first token | 31 | 103.1 | 207.5 | 344.7 |

Measured tokens (F, all new versions): prompt 13493, output 6471.

## Verifier on perturbations (labels by construction): post-fix run

| verifier | items | accuracy | support precision | support recall | unsupported detected | ms / claim |
|---|---|---|---|---|---|---|
| nli | 125 | 89.6 % | 78.4 % | 95.2 % | 86.8 % | 86.9 |
| rules | 125 | 88.0 % | 73.7 % | 100.0 % | 81.9 % | 5.2 |

| perturbation | n | nli accepted | rules accepted |
|---|---|---|---|
| modality_or_negation_changed | 15 | 4 | 0 |
| negation_inserted | 16 | 1 | 10 |
| number_changed | 10 | 0 | 0 |
| original | 42 | 40 | 42 |
| unsupported_conjunct | 42 | 6 | 5 |

