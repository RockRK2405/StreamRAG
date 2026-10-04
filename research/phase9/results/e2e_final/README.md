# Phase 9 final end-to-end test (generated from the runtime event logs)

> Real local LLM (qwen3:4b), real retrieval / NLI, Phase 8 runtime (realtime), adaptive retrieval enabled. Fixture corpus tests/fixtures/corpus_adaptive - behaviour check, NOT a benchmark result.

Checks passed: 19 / 24 (expectations written before the run).

## 01_simple_question
- u1: strategy **HYBRID** (complexity MODERATE: ambiguous); cache ['CACHE_MISS', 'CACHE_MISS']; searches 1; sufficiency ['SUFFICIENT', 'SUFFICIENT']; stop ['SUFFICIENT_EVIDENCE', 'SUFFICIENT_EVIDENCE']; hops []; invalidations []; LLM calls 1; claims verified 5, rejected 0; citations validated 3
  - answer (VALIDATED_FINAL, ollama): A residence permit must be renewed every 2 years. [RENEW §1] Renewal requests can be submitted up to 30 days before the permit expires using Form PX-118. [RENEW §1]
- checks: fast path LEXICAL: FAIL; answer cites RENEW §1: PASS
- events 157, untraceable 0

## 02_multi_intent
- u1: strategy **LEXICAL** (simple-query fast path (simple_strategy, calibrated)); cache ['CACHE_MISS', 'CACHE_MISS', 'CACHE_MISS']; searches 2; sufficiency ['SUFFICIENT', 'SUFFICIENT', 'SUFFICIENT']; stop ['SUFFICIENT_EVIDENCE', 'SUFFICIENT_EVIDENCE', 'SUFFICIENT_EVIDENCE']; hops []; invalidations []; LLM calls 1; claims verified 8, rejected 1; citations validated 3
  - answer (VALIDATED_FINAL, ollama): What is the application fee:
The application fee for a residence permit is 55 euros. [ELIG-2026 §2] Applicants under 25 pay a reduced fee of: 30 euros. [ELIG-2026 §2]

How long does processing take for domestic applicants:
Applications from domestic applicants are processed within 10 working days. [DOM-2026 §2]
- checks: one plan per need (2): PASS; cites ELIG-2026 §2 and DOM-2026 §2: PASS
- events 204, untraceable 0

## 03_contextual_follow_up
- u1: strategy **LEXICAL** (simple-query fast path (simple_strategy, calibrated)); cache ['CACHE_MISS', 'CACHE_MISS']; searches 2; sufficiency ['SUFFICIENT', 'SUFFICIENT']; stop ['SUFFICIENT_EVIDENCE', 'SUFFICIENT_EVIDENCE']; hops []; invalidations []; LLM calls 1; claims verified 6, rejected 0; citations validated 2
  - answer (VALIDATED_FINAL, ollama): Applicants must be at least 18 years old. [ELIG-2026 §1] Applicants must hold a valid passport. [ELIG-2026 §1] Applicants must have a registered address in Fixture City. [ELIG-2026 §1]
- u2: strategy **FILTERED** (metadata constraint {'applicant_type': ['international']}); cache ['CACHE_MISS']; searches 2; sufficiency ['PARTIAL']; stop ['NO_EXPECTED_GAIN']; hops []; invalidations []; LLM calls 1; claims verified 3, rejected 0; citations validated 1
  - answer (VALIDATED_FINAL, ollama): Applicants must be at least 18 years old. [ELIG-2026 §1] Applicants must hold a valid passport. [ELIG-2026 §1] Applicants must have a registered address in Fixture City. [ELIG-2026 §1] Not established: The retrieved documents do not say how this applies to international applicants.
- checks: follow-up retrieval FILTERED international: PASS; follow-up cites INTL-2026 §1: FAIL
- events 219, untraceable 0

## 04_exact_entity
- u1: strategy **LEXICAL** (exact identifiers ['PX-204'] known to the lexical index); cache ['CACHE_MISS', 'CACHE_MISS']; searches 1; sufficiency ['SUFFICIENT', 'SUFFICIENT']; stop ['SUFFICIENT_EVIDENCE', 'SUFFICIENT_EVIDENCE']; hops []; invalidations []; LLM calls 1; claims verified 6, rejected 0; citations validated 2
  - answer (VALIDATED_FINAL, ollama): Form PX-204 is the application form for international applicants. [FORMS §1; INTL-2026 §1] International applicants must submit Form PX-204 together with a certified copy of their visa. [INTL-2026 §1]
- checks: LEXICAL on the identifier: PASS; cites FORMS §1 or INTL-2026 §1: PASS
- events 139, untraceable 0

## 05_temporal
- u1: strategy **LEXICAL** (simple-query fast path (simple_strategy, calibrated)); cache ['CACHE_MISS', 'CACHE_MISS']; searches 2; sufficiency ['SUFFICIENT', 'SUFFICIENT']; stop ['SUFFICIENT_EVIDENCE', 'SUFFICIENT_EVIDENCE']; hops []; invalidations []; LLM calls 1; claims verified 3, rejected 0; citations validated 2
  - answer (VALIDATED_FINAL, ollama): The application fee for a residence permit is 40 euros. [ELIG-2024 §2]
- checks: FILTERED by the December 2025 period: PASS; cites ELIG-2024 §2, not ELIG-2026 §2: PASS
- events 119, untraceable 0

## 06_insufficient
- u1: strategy **LEXICAL** (simple-query fast path (simple_strategy, calibrated)); cache ['CACHE_MISS', 'CACHE_MISS']; searches 2; sufficiency ['SUFFICIENT', 'INSUFFICIENT']; stop ['SUFFICIENT_EVIDENCE', 'NO_EXPECTED_GAIN']; hops []; invalidations []; LLM calls 1; claims verified 6, rejected 1; citations validated 2
  - answer (VALIDATED_FINAL, ollama): Applicants who receive income support do not pay the residence permit application fee. [WAIVERS §1] The sources differ: “Applicants under 25 pay a reduced fee of 30 euros” [ELIG-2026 §2] / “Market stalls pay a reduced fee of 60 euros” [BIZ-2026 §2] / “The application fee for a residence permit is 55 euros” [ELIG-2026 §2].
- checks: evidence INSUFFICIENT: PASS; answer states the gap: PASS
- events 139, untraceable 0

## 07_contradictory
- u1: strategy **LEXICAL** (simple-query fast path (simple_strategy, calibrated)); cache ['CACHE_MISS']; searches 2; sufficiency ['CONTRADICTORY']; stop ['CONTRADICTION']; hops []; invalidations []; LLM calls 2; claims verified 2, rejected 5; citations validated 2
  - answer (VALIDATED_FINAL, ollama): The sources differ: “The Permit Processing Office opens at 9:00 on weekdays” [NOTICE-A §1] / “The Permit Processing Office opens at 8:30 on weekdays” [NOTICE-B §1]. The Permit Processing Office (PPO) handles all residence permit applications. [FORMS §2] The PPO is closed on public holidays. [FORMS §2]
- checks: evidence CONTRADICTORY, stop CONTRADICTION: PASS; answer reports 9:00 and 8:30: PASS
- events 124, untraceable 0

## 08_multi_hop
- u1: strategy **MULTI_HOP** (complexity MULTI_HOP: entity_aspect_gap:zemland); cache ['CACHE_MISS']; searches 3; sufficiency ['SUFFICIENT']; stop ['SUFFICIENT_EVIDENCE']; hops ['zemland -> Group B']; invalidations []; LLM calls 2; claims verified 4, rejected 2; citations validated 2
  - answer (VALIDATED_FINAL, ollama): Zemland is classified as Group B. [ANNEX-C §1; GROUP-RULES §2] Group B applicants must provide a certified police clearance certificate. [GROUP-RULES §2] Group B applicants must provide attend an in-person interview. [GROUP-RULES §2]
- checks: hop zemland -> Group B: PASS; cites GROUP-RULES §2: PASS
- events 119, untraceable 0

## 09_cached_question
- u1: strategy **LEXICAL** (simple-query fast path (simple_strategy, calibrated)); cache ['CACHE_MISS', 'CACHE_MISS']; searches 1; sufficiency ['SUFFICIENT', 'SUFFICIENT']; stop ['SUFFICIENT_EVIDENCE', 'SUFFICIENT_EVIDENCE']; hops []; invalidations []; LLM calls 1; claims verified 5, rejected 0; citations validated 2
  - answer (VALIDATED_FINAL, ollama): The application fee for a residence permit is 55 euros. [ELIG-2026 §2] Applicants under 25 pay a reduced fee of 30 euros. [ELIG-2026 §2]
- u2: strategy **LEXICAL** (simple-query fast path (simple_strategy, calibrated)); cache ['CACHE_MISS', 'CACHE_MISS']; searches 1; sufficiency ['SUFFICIENT', 'SUFFICIENT']; stop ['SUFFICIENT_EVIDENCE', 'SUFFICIENT_EVIDENCE']; hops []; invalidations []; LLM calls 1; claims verified 4, rejected 2; citations validated 2
  - answer (VALIDATED_FINAL, ollama): The public holidays in 2026 are 1 January, 1 May, and 25 December. [HOLIDAYS §1] The sources differ: “The public holidays in 2026 are 1 January, 1 May and 25 December” [HOLIDAYS §1] / “The PPO is closed on public holidays” [FORMS §2]. Municipal offices are closed on these public holidays. [FORMS §2; HOLIDAYS §1]
- u3: strategy **HYBRID** (complexity MODERATE: context_dependent); cache ['CACHE_MISS']; searches 2; sufficiency ['INSUFFICIENT']; stop ['NO_EXPECTED_GAIN']; hops []; invalidations []; LLM calls 1; claims verified 5, rejected 1; citations validated 2
  - answer (VALIDATED_FINAL, ollama): Which public holidays are in 2026:
The public holidays in 2026 are 1 January, 1 May, and 25 December. [HOLIDAYS §1] Municipal offices are closed on these public holidays. [FORMS §2; HOLIDAYS §1]

What is the application fee for a residence permit:
The application fee for a residence permit is 55 euros. [ELIG-2026 §2] Applicants under 25 pay a reduced fee of 30 euros. [ELIG-2026 §2]
- checks: no adaptive search for the repeat: FAIL; repeat cites ELIG-2026 §2: PASS
- events 427, untraceable 0

## 10_changed_entity
- u1: strategy **MULTI_HOP** (complexity MULTI_HOP: entity_aspect_gap:zemland); cache ['CACHE_MISS', 'CACHE_MISS']; searches 6; sufficiency ['SUFFICIENT', 'SUFFICIENT']; stop ['SUFFICIENT_EVIDENCE', 'SUFFICIENT_EVIDENCE']; hops ['zemland -> Group B', 'zemland -> Group B']; invalidations []; LLM calls 1; claims verified 6, rejected 0; citations validated 2
  - answer (VALIDATED_FINAL, ollama): Group B applicants must provide a certified police clearance certificate. [GROUP-RULES §2] Group B applicants must attend an in-person interview. [GROUP-RULES §2] Zemland is classified as Group B. [ANNEX-C §1; GROUP-RULES §2]
- u2: strategy **MULTI_HOP** (complexity MULTI_HOP: entity_aspect_gap:norvia); cache ['CACHE_MISS']; searches 3; sufficiency ['SUFFICIENT']; stop ['SUFFICIENT_EVIDENCE']; hops ['norvia -> Group A']; invalidations ['entity_changed']; LLM calls 2; claims verified 5, rejected 2; citations validated 2
  - answer (VALIDATED_FINAL, ollama): The sources differ: “Group A applicants need no additional documents” [GROUP-RULES §1] / “Further requirements depend on the country group of the applicant; see the Country Classification Annex” [INTL-2026 §1]. Norvia is classified as Group A. [ANNEX-C §1; GROUP-RULES §1] Further requirements depend on the country group of the applicant. [INTL-2026 §1] See the Country Classification Annex. [INTL-2026 §1]
- checks: cache invalidated: entity_changed: PASS; hop norvia -> Group A: PASS
- events 284, untraceable 0

## 11_changed_constraint
- u1: strategy **HYBRID** (complexity MODERATE: ambiguous); cache ['CACHE_MISS', 'CACHE_MISS']; searches 1; sufficiency ['SUFFICIENT', 'SUFFICIENT']; stop ['SUFFICIENT_EVIDENCE', 'SUFFICIENT_EVIDENCE']; hops []; invalidations []; LLM calls 1; claims verified 4, rejected 1; citations validated 2
  - answer (VALIDATED_FINAL, ollama): International applicants have applications processed within 30 working days. [INTL-2026 §2]
- u2: strategy **FILTERED** (metadata constraint {'applicant_type': ['domestic', 'international']}); cache ['CACHE_MISS', 'CACHE_MISS']; searches 2; sufficiency ['SUFFICIENT', 'PARTIAL']; stop ['SUFFICIENT_EVIDENCE', 'NO_EXPECTED_GAIN']; hops []; invalidations ['constraint_changed']; LLM calls 1; claims verified 3, rejected 1; citations validated 2
  - answer (VALIDATED_FINAL, ollama): International applicants have applications processed within 30 working days. [INTL-2026 §2] Not established: The retrieved documents do not say how this applies for domestic applicants instead.
- checks: re-plan FILTERED domestic: FAIL; cites DOM-2026 §2: FAIL
- events 280, untraceable 0

## 12_streaming_correction
- u1: strategy **HYBRID** (complexity MODERATE: ambiguous); cache ['CACHE_MISS', 'CACHE_MISS', 'CACHE_MISS']; searches 1; sufficiency ['SUFFICIENT', 'SUFFICIENT', 'SUFFICIENT']; stop ['SUFFICIENT_EVIDENCE', 'SUFFICIENT_EVIDENCE', 'SUFFICIENT_EVIDENCE']; hops []; invalidations []; LLM calls 1; claims verified 5, rejected 2; citations validated 3
  - answer (VALIDATED_FINAL, ollama): International applicants have applications processed within 30 working days. [INTL-2026 §2]
- checks: last plan filters international: PASS; cites INTL-2026 §2: PASS
- events 213, untraceable 0

