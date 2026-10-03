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

## Ablation by arm: final run

| metric | A | B | C | D | E | F | X |
|---|---|---|---|---|---|---|---|
| required-fact recall | 11.8 % | 100.0 % | 97.1 % | 97.1 % | 97.1 % | 100.0 % | 100.0 % |
| raw claims | 59 | 135 | 109 | 109 | 109 | 92 | 121 |
| raw support (verifier) | 15.2 % | 97.8 % | 98.2 % | 98.2 % | 98.2 % | 97.8 % | 100.0 % |
| raw unsupported (verifier) | 78.0 % | 2.2 % | 0.0 % | 0.0 % | 0.0 % | 2.2 % | 0.0 % |
| final claims | 57 | 135 | 107 | 107 | 107 | 104 | 131 |
| final unsupported (verifier) | 84.2 % | 2.2 % | 0.0 % | 0.0 % | 0.0 % | 0.0 % | 0.0 % |
| citation coverage | – | – | 100.0 % | 100.0 % | 100.0 % | 100.0 % | 100.0 % |
| citation precision (verifier) | – | – | 98.2 % | 100.0 % | 100.0 % | 100.0 % | 100.0 % |
| intent coverage | – | – | – | – | – | 100.0 % | 100.0 % |
| forbidden assertions | 3 | 0 | 0 | 0 | 0 | 0 | 0 |
| uncertainty when expected | 10.0 % | 0.0 % | 20.0 % | 20.0 % | 20.0 % | 60.0 % | 60.0 % |
| conflict presented | 0.0 % | 0.0 % | 66.7 % | 66.7 % | 66.7 % | 100.0 % | 100.0 % |
| LLM calls | 33 | 33 | 33 | 33 | 33 | 32 | 0 |
| generation ms p50 / p95 | 1046 / 2176 | 1109 / 3492 | 2089 / 6426 | 0 / 0 | 0 / 0 | 2057 / 4188 | 0 / 0 |
| validation ms p50 / p95 | 151 / 780 | 45 / 359 | 29 / 357 | 15 / 50 | 14 / 47 | 58 / 377 | 21 / 255 |
| total ms p50 / p95 | 1311 / 2965 | 1256 / 3676 | 2134 / 6471 | 15 / 51 | 14 / 47 | 2077 / 4442 | 21 / 255 |

## Full system (F) by category: final run

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

## Hallucination tags: final run

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

## Revision: final run

| mode | later turns | LLM calls | claims kept | required recall | forbidden | wall ms p50 / p95 |
|---|---|---|---|---|---|---|
| full_restart | 5 | 5 | 40.0 % | 100.0 % | 0 | 2417 / 3030 |
| incremental | 5 | 3 | 46.7 % | 100.0 % | 0 | 969 / 2469 |

## F stage latency (ms): final run

| stage | n | p50 | p95 | max |
|---|---|---|---|---|
| citation_mapping | 33 | 0.0 | 0.0 | 0.1 |
| claim_extraction | 33 | 0.1 | 0.1 | 0.1 |
| claim_planning | 33 | 0.3 | 6.0 | 14.4 |
| claim_verification | 33 | 56.6 | 373.3 | 634.5 |
| generation | 33 | 2056.8 | 4188.1 | 6615.7 |
| repair | 33 | 0.0 | 20.9 | 493.4 |
| retrieval | 33 | 0.0 | 0.0 | 0.0 |
| total | 33 | 2076.8 | 4441.8 | 6660.0 |
| validation | 33 | 0.2 | 1.5 | 5.5 |
| raw first token | 31 | 96.0 | 154.0 | 209.0 |

Measured tokens (F, all new versions): prompt 11455, output 5563.

## Verifier on perturbations (labels by construction): final run

| verifier | items | accuracy | support precision | support recall | unsupported detected | ms / claim |
|---|---|---|---|---|---|---|
| nli | 125 | 89.6 % | 78.4 % | 95.2 % | 86.8 % | 86.4 |
| rules | 125 | 88.0 % | 73.7 % | 100.0 % | 81.9 % | 5.1 |

| perturbation | n | nli accepted | rules accepted |
|---|---|---|---|
| modality_or_negation_changed | 15 | 4 | 0 |
| negation_inserted | 16 | 1 | 10 |
| number_changed | 10 | 0 | 0 |
| original | 42 | 40 | 42 |
| unsupported_conjunct | 42 | 6 | 5 |

## Verifier vs hand labels (final run claims; labels made blind on the post-fix run sheet)

| source | n | accuracy | support precision | support recall | unsupported detected |
|---|---|---|---|---|---|
| pooled | 188 | 96.8 % | 96.8 % | 100.0 % | 33.3 % |
| B | 129 | 97.7 % | 97.6 % | 100.0 % | 50.0 % |
| C | 105 | 98.1 % | 98.1 % | 100.0 % | 0.0 % |
| F_released | 88 | 98.9 % | 98.9 % | 100.0 % | 0.0 % |

Released-claim precision against the labels: 98.9 % (200 of 200 items labelled; 12 unclear).

