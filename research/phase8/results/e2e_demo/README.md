# Phase 8 end-to-end demo (generated from the runtime's event log)

> Fixture corpus, real retrieval / NLI / local LLM (qwen3:4b). Dense index latency +1200 ms injected (SYNTHETIC remote vector store). Behaviour demonstration, NOT a benchmark result.

User (streamed): `What are the eligibility | requirements | for the permit?` ... (answer being generated) ... `For international | applicants.`

```
[00:00.025] u1   USER       #0: "What are the eligibility"
[00:00.026] u1   CONTROLLER RETRIEVE stable_retrieval_worthy_request  conf=0.88 stab=0.767 worth=0.933 nov=1.0 act=INFO_REQUEST
[00:00.026] u1   INTENTS    v1 +I1 | I1[REQUIREMENT] 'What are the eligibility'
[00:00.026] u1   QUERY      Q1 for I1 v1 "What are the eligibility"
[00:00.026] u1   MULTI      B1 start Q1 (parallel)
[00.026] u1   TASK       T1 lexical for Q1 scheduled prio=1
[00.026] u1   TASK       T1 lexical for Q1 started prio=1
[00:00.026] u1   RETRIEVAL  START Q1/I1 [provisional] queue_wait=0.098ms
[00.027] u1   TASK       T2 dense for Q1 scheduled prio=1
[00.027] u1   TASK       T2 dense for Q1 started prio=1
[00.027] u1   TASK       T1 lexical for Q1 completed prio=1
[00:00.027] u1   PARTIAL    Q1 lexical hits=1 top=['fixture_permit_handbook §1']
[00:00.376] u1   USER       #1: "requirements"
[00:00.376] u1   CONTROLLER WAIT     cooldown  conf=1.00 stab=0.883 worth=0.933 nov=0.5 act=INFO_REQUEST
[00:00.377] u1   INTENTS    v2 ~I1 | I1[REQUIREMENT] 'What are the eligibility requirements'
[00.378] u1   SUPERSEDED Q1 (delta plan P2)
[00:00.378] u1   RETRIEVAL  SKIP I1 (intent_cooldown)
[00:00.726] u1   USER       #2: "for the permit?"
[00:00.727] u1   CONTROLLER RETRIEVE stable_retrieval_worthy_request  conf=0.91 stab=1.0 worth=0.933 nov=0.667 act=INFO_REQUEST
[00:00.728] u1   INTENTS    v3 ~I1 | I1[REQUIREMENT] 'What are the eligibility requirements for the permit'
[00.729] u1   SUPERSEDED Q1 (delta plan P3)
[00:00.729] u1   QUERY      Q2 for I1 v3 "What are the eligibility requirements for the permit" supersedes Q1 (refines)
[00:00.730] u1   MULTI      B2 start Q2 (parallel)
[00.730] u1   TASK       T3 lexical for Q2 scheduled prio=1
[00.730] u1   TASK       T3 lexical for Q2 started prio=1
[00:00.730] u1   RETRIEVAL  START Q2/I1 [provisional] queue_wait=0.213ms
[00.731] u1   TASK       T4 dense for Q2 scheduled prio=1
[00.731] u1   TASK       T4 dense for Q2 started prio=1
[00:00.733] u1   TASK       T2 dense CANCELLED: superseded_in_flight
[00:00.733] u1   RETRIEVAL  CANCEL Q1 (superseded_in_flight)
[00:00.733] u1   MULTI      B1 done makespan=706.184ms
[00:00.733] u1   FUSED      provisional intent_aware/none coverage=0.0 items=0: 
[00.733] u1   TASK       T3 lexical for Q2 completed prio=1
[00:00.733] u1   PARTIAL    Q2 lexical hits=6 top=['fixture_permit_handbook §1', 'fixture_permit_handbook §3', 'fixture_permit_handbook §2']
[00:01.077] u1   UTTERANCE  FINALIZED reason=endpoint transcript="What are the eligibility requirements for the permit?"
[00:01.078] u1   CONTROLLER SKIP     redundant_query  conf=1.00 stab=1.0 worth=0.933 nov=0.0 act=INFO_REQUEST
[00:01.678] u2   USER       #0: "For international"
[00:01.678] u2   CONTROLLER WAIT     low_specificity  conf=0.51 stab=0.492 worth=0.2 nov=1.0 act=UNKNOWN
[00:01.679] u2   INTENTS    v1 ~I1 |  | constraints K1:'For international'->I1
[01.680] u2   SUPERSEDED Q2 (delta plan P4)
[00:01.681] u2   QUERY      Q3 for I1 v4 "What are the eligibility requirements for the permit For international" supersedes Q2 (refines)
[00:01.681] u2   MULTI      B3 start Q3 (parallel)
[01.681] u2   TASK       T5 lexical for Q3 scheduled prio=1
[01.681] u2   TASK       T5 lexical for Q3 started prio=1
[00:01.681] u2   RETRIEVAL  START Q3/I1 [provisional] queue_wait=0.249ms
[01.681] u2   TASK       T6 dense for Q3 scheduled prio=1
[01.681] u2   TASK       T6 dense for Q3 started prio=1
[00:01.683] u1   TASK       T4 dense CANCELLED: superseded_in_flight
[00:01.683] u1   RETRIEVAL  CANCEL Q2 (superseded_in_flight)
[00:01.683] u1   MULTI      B2 done makespan=952.913ms
[00:01.683] u1   FUSED      final intent_aware/none coverage=0.0 items=0: 
[00:01.683] u1   ANSWER     A1 v1 initial frame=T1 claims=[] re-render=['S-I1'] | NEW_INTENT(I1); REFINEMENT(I1); REFINEMENT(I1); CONSTRAINT_ADDITION(I1) +['For international'] | claims: 0 kept, 0 modified, 0 added, 0 retracted | sections to regenerate: S-I1
[00:01.684] u1   TURN       net=['NEW_INTENT'] turn_needs=['I1'] sub_queries=['What are the eligibility requirements for the permit For international'] answer=A1 (updated) session_v=5
[01.684] u2   TASK       T5 lexical for Q3 completed prio=1
[00:01.684] u2   PARTIAL    Q3 lexical hits=6 top=['fixture_permit_handbook §1', 'fixture_permit_handbook §3', 'fixture_permit_handbook §2']
[00:02.028] u2   USER       #1: "applicants."
[00:02.029] u2   CONTROLLER WAIT     low_specificity  conf=0.34 stab=0.658 worth=0.2 nov=0.8 act=UNKNOWN
[00:02.030] u2   INTENTS    v2 ~I1 |  | constraints K2:'For international applicants'->I1
[02.031] u2   SUPERSEDED Q3 (delta plan P5)
[00:02.031] u2   RETRIEVAL  SKIP I1 (intent_cooldown)
[00:02.378] u2   UTTERANCE  FINALIZED reason=endpoint transcript="For international applicants."
[00:02.379] u2   CONTROLLER SKIP     not_retrieval_worthy  conf=0.80 stab=0.658 worth=0.2 nov=0.8 act=UNKNOWN
[00:02.380] u2   QUERY      Q4 for I1 v5 "What are the eligibility requirements for the permit For international applicants" supersedes Q3 (refines)
[00:02.380] u2   MULTI      B4 start Q4 (parallel)
[02.380] u2   TASK       T7 lexical for Q4 scheduled prio=1
[02.380] u2   TASK       T7 lexical for Q4 started prio=1
[00:02.380] u2   RETRIEVAL  START Q4/I1 [final] queue_wait=0.252ms
[02.381] u2   TASK       T8 dense for Q4 scheduled prio=1
[02.381] u2   TASK       T8 dense for Q4 started prio=1
[00:02.382] u2   TASK       T6 dense CANCELLED: superseded_in_flight
[00:02.382] u2   RETRIEVAL  CANCEL Q3 (superseded_in_flight)
[00:02.382] u2   MULTI      B3 done makespan=701.23ms
[02.382] u2   TASK       T7 lexical for Q4 completed prio=1
[00:02.382] u2   PARTIAL    Q4 lexical hits=6 top=['fixture_permit_handbook §1', 'fixture_permit_handbook §2', 'fixture_permit_notice_2023 §1']
[03.593] u2   TASK       T8 dense for Q4 completed prio=1
[00:03.593] u2   PARTIAL    Q4 dense hits=6 top=['fixture_permit_handbook §1', 'fixture_permit_notice_2023 §1', 'fixture_permit_notice_2025 §1']
[00:03.593] u2   RETRIEVAL  DONE  Q4/I1 status=ok 1212.991ms top: fixture_permit_handbook §1, fixture_permit_notice_2023 §1, fixture_permit_handbook §2
[00:03.596] u2   CLAIM      C1 new for I1: "Applicants must be at least 18 years old."
[00:03.597] u2   CLAIM      C2 new for I1: "Applicants must provide proof of residence."
[00:03.597] u2   CLAIM      C3 new for I1: "Applicants who hold a suspended permit cannot apply."
[00:03.597] u2   CLAIM      C4 new for I1: "The application fee for a new permit is 40 euros."
[03.598] u2   CLAIM      C1 PENDING_VALIDATION->PARTIALLY_SUPPORTED (does_not_address_constraint:K2)
[03.598] u2   CLAIM      C2 PENDING_VALIDATION->PARTIALLY_SUPPORTED (does_not_address_constraint:K2)
[03.598] u2   CLAIM      C3 PENDING_VALIDATION->PARTIALLY_SUPPORTED (does_not_address_constraint:K2)
[03.598] u2   CLAIM      C4 PENDING_VALIDATION->PARTIALLY_SUPPORTED (does_not_address_constraint:K2)
[00:03.598] u2   MULTI      B4 done makespan=1212.991ms
[00:03.598] u2   FUSED      final intent_aware/none coverage=1.0 items=5: E1=fixture_permit_handbook §1<I1>, E2=fixture_permit_notice_2023 §1<I1>, E3=fixture_permit_handbook §2<I1>, E4=fixture_permit_notice_2025 §1<I1>, E5=fixture_permit_handbook §3<I1>
[00:03.598] u2   ANSWER     A2 v2 refinement frame=T1 claims=['C1', 'C2', 'C3', 'C4'] re-render=['S-I1'] | CONSTRAINT_ADDITION(I1) +['For international applicants']; CONSTRAINT_REMOVAL(I1) -['For international'] | claims: 0 kept, 0 modified, 4 added, 0 retracted | sections to regenerate: S-I1
[03.599] u2   TASK       T10 generation (final) scheduled prio=0
[03.599] u2   TASK       T10 generation (final) started prio=0
[00:03.600] u2   TURN       net=['CONSTRAINT_ADDITION'] turn_needs=['I1'] sub_queries=['What are the eligibility requirements for the permit For international applicants'] answer=A2 (updated) session_v=8
[05.919] u2   TASK       T10 generation (final) completed prio=0
[00:05.920] u2   LLM        qwen3:4b generate ok=True tokens 358->164 total=1956ms
[00:05.920] u2   VERIFY     CONTRADICTED -> present_conflict: "Applicants must provide proof of residence."
[00:05.920] u2   GROUNDED   GA1 v1 VALIDATED_FINAL partial
               Applicants must be at least 18 years old. [fixture_permit_handbook §1] The sources differ: “Applicants must provide proof of residence” [fixture_permit_handbook §1] / “Proof of residence is optional for all applicants” [fixture_permit_notice_2025 §1]. Applicants who hold a suspended permit cannot apply. [fixture_permit_handbook §1] Not established: The retrieved documents do not say how this applies for international applicants.
[05.920] u2   DELTA      GA1 v1 VALIDATED_FINAL: +4 claims, -0, 0 unchanged
[00:05.920] u2   COMMITTED  GA1 v1 VALIDATED_FINAL (full, ollama)
```

Trace path of the final answer (parent links): SESSION_STARTED (demo:000000) -> CHUNK_RECEIVED (demo:000001) -> INTENT_DETECTED (demo:000005) -> QUERY_GENERATED (demo:000094) -> TASK_SCHEDULED (demo:000109) -> TASK_SCHEDULED (demo:000127) -> ANSWER_COMMITTED (demo:000162)

Events: 165; untraceable: 0; virtual-clock replay with the recorded LLM output: behaviour identical = True (exact = False: realtime timings differ by design).

