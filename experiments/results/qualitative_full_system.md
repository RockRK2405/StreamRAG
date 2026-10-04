# Qualitative sample - full_system (seeded random: 12 of 87 turns; 31 with an error category, 56 without)

## S06.1 - MULTI_HOP / HARD - QUERY_ANALYSIS_FAILURE

* query: What is the supplement for trips to Harbor Point?
* expected: Harbor Point is in Zone C: a supplement of 1.20 euros.
* answer: Harbor Point is classified as Zone C. [ZONES §1] Trips that end in Zone C require a supplement of 1.20 euros. [ZONE-RULES §3] Trips that end in Zone B need a supplement of 0.50 euros. [ZONE-RULES §2] Trips that end in Zone C need a supplement of 1.20 euros. [ZONE-RULES §3] The ticket must be validated before boarding. [ZONE-RULES §3]
* retrieved sections: ZONE-RULES §1, ZONE-RULES §2, ZONES §1, ZONE-RULES §3
* gold sections: ZONES §1, ZONE-RULES §3
* trace: experiments/results/runs/traces/full_system__S06.jsonl

## S02.1 - MULTI_CONSTRAINT / MEDIUM - QUERY_ANALYSIS_FAILURE

* query: How do students apply for the concession?
* expected: Students apply with Form TR-210 and a current enrolment certificate.
* answer: Students apply with Form TR-210. [CONC-STUDENT §2] Students must submit a current enrolment certificate with their application. [CONC-STUDENT §2] Student applications are processed within 5 working days. [CONC-STUDENT §2] Students receive a 50 percent discount on the monthly pass. [CONC-STUDENT §1]
* retrieved sections: CONC-STUDENT §2, CONC-STUDENT §1
* gold sections: CONC-STUDENT §2
* trace: experiments/results/runs/traces/full_system__S02.jsonl

## T35 - MULTI_HOP / HARD - QUERY_ANALYSIS_FAILURE, EVIDENCE_FAILURE

* query: Is the office that issues concession cards open on Sundays?
* expected: No - the Travel Card Office, which issues concession cards, is closed on Sundays.
* answer: The Travel Card Office (TCO) issues all concession cards and replaces lost travel cards. [TCO §1] A replacement card costs 5 euros. [TCO §1]
* retrieved sections: TCO §1
* gold sections: TCO §1, TCO §2
* trace: experiments/results/runs/traces/full_system__T35.jsonl

## S14.1 - MULTI_HOP / HARD - QUERY_ANALYSIS_FAILURE, EVIDENCE_FAILURE

* query: Do I need a supplement for Quarry Lane?
* expected: Yes - Quarry Lane is in Zone B: 0.50 euros.
* answer: Quarry Lane is classified as Zone B. [ZONES §1]
* retrieved sections: ZONES §1
* gold sections: ZONES §1, ZONE-RULES §2
* trace: experiments/results/runs/traces/full_system__S14.jsonl

## SC2 - STREAMING_CORRECTION / MEDIUM - EVIDENCE_FAILURE

* query: What does a monthly pass cost?
* expected: A monthly pass costs 72 euros.
* answer: A monthly pass costs 72 euros. [FARES-2026 §1]
* retrieved sections: FARES-2026 §1
* gold sections: FARES-2026 §1
* trace: experiments/results/runs/traces/full_system__SC2.jsonl

## S10.2 - ENTITY_CORRECTION / HARD - ENTITY_FAILURE, GENERATION_FAILURE

* query: I meant folding bicycles, not ordinary ones.
* expected: Folded bicycles are allowed at all times.
* answer: (empty)
* retrieved sections: BIKES §2
* gold sections: BIKES §2
* trace: experiments/results/runs/traces/full_system__S10.jsonl

## T22 - MULTI_CONSTRAINT / HARD - QUERY_ANALYSIS_FAILURE

* query: What discount do students get on the monthly pass?
* expected: Students receive a 50 percent discount on the monthly pass.
* answer: Students receive a 50 percent discount on the monthly pass. [CONC-STUDENT §1]
* retrieved sections: CONC-STUDENT §1
* gold sections: CONC-STUDENT §1
* trace: experiments/results/runs/traces/full_system__T22.jsonl

## S06.2 - ENTITY_CORRECTION / MEDIUM - ENTITY_FAILURE, EVIDENCE_FAILURE, GENERATION_FAILURE

* query: Sorry, I meant Quarry Lane, not Harbor Point.
* expected: Quarry Lane is in Zone B: a supplement of 0.50 euros.
* answer: Quarry Lane is classified as Zone B. [ZONES §1]
* retrieved sections: ZONE-RULES §1, ZONE-RULES §2, ZONE-RULES §3, ZONES §1
* gold sections: ZONES §1, ZONE-RULES §2
* trace: experiments/results/runs/traces/full_system__S06.jsonl

## T41 - CONTRADICTORY / MEDIUM - no error category

* query: How often does the N4 night bus run?
* expected: The sources disagree: one notice says every 20 minutes, the other every 30 minutes.
* answer: The sources differ: “The night bus N4 departs every 20 minutes between midnight and 5:00” [NOTICE-X §1] / “The night bus N4 departs every 30 minutes between midnight and 5:00” [NOTICE-Y §1].
* retrieved sections: NOTICE-X §1, NOTICE-Y §1, NOTICE-X §0, ACCESS §1, HOLIDAY-SERVICE §1
* gold sections: NOTICE-X §1, NOTICE-Y §1
* trace: experiments/results/runs/traces/full_system__T41.jsonl

## T36 - AMBIGUOUS / EASY - no error category

* query: How much is it?
* expected: 
* answer: Not established: The retrieved documents do not contain an answer to “How much is it”.
* retrieved sections: LOST §2, FARES-2025 §1, FARES-2026 §1, ZONE-RULES §3, ZONE-RULES §2
* gold sections: FARES-2026 §1, TCO §1, PENALTIES §1, LOST §2
* trace: experiments/results/runs/traces/full_system__T36.jsonl

## T30 - TEMPORAL / MEDIUM - no error category

* query: Which days have a holiday timetable in 2026?
* expected: 1 January, 1 May and 25 December 2026 (Sunday timetable).
* answer: On 1 January, 1 May and 25 December 2026 all lines run the Sunday timetable. [HOLIDAY-SERVICE §1]
* retrieved sections: HOLIDAY-SERVICE §1
* gold sections: HOLIDAY-SERVICE §1
* trace: experiments/results/runs/traces/full_system__T30.jsonl

## T23 - MULTI_CONSTRAINT / MEDIUM - no error category

* query: How long does processing take for seniors applying for the concession?
* expected: Senior applications are processed within 10 working days.
* answer: Senior applications are processed within 10 working days. [CONC-SENIOR §2]
* retrieved sections: CONC-SENIOR §2
* gold sections: CONC-SENIOR §2
* trace: experiments/results/runs/traces/full_system__T23.jsonl

