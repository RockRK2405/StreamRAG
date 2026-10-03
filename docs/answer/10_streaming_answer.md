# 10: Streaming Answers, Drafts and the Output Contract

**Code:** `multi_retrieval/coordinator.py` (`_draft`, `_session_turn_payload`), `answers/manager.py` (`preview`), `answer_state/engine.py` (`_emit_answer`), `streaming/printer.py`, `replay/replay.py`

## Drafts vs validated finals (brief §35, §39-40)

- **Drafts while the user speaks.** A verified *extractive* draft is produced after each provisional evidence batch (Phase 4/5 early retrieval). It comes from the uncommitted Phase 6 preview, and is marked **DRAFT**:
  - no LLM call;
  - claims are still verified and cited;
  - `generation.draft_mode: off` disables drafts.
- **Final at turn end.** The LLM writes the final from the committed Phase 6 answer state, and it is validated: **VALIDATED_FINAL**, or BLOCKED.
- **Reuse base.** Drafts never become the reuse base of a final. A final reuses only the previous final, so the model is not bypassed by draft sentences.
- **Partial answers.** A partial answer keeps supported information and "Not established: …" statements separate (`partial = true`).

## Event contract (brief §38)

Each event carries `session_id` (envelope), `answer_id`, `version`, `status`, and `intent_id` / `claim_id` where relevant.

```
ANSWER_STARTED -> per section: ANSWER_SECTION_STARTED -> per claim: ANSWER_CLAIM_READY (+ ANSWER_CITATION_READY per citation)
               -> ANSWER_SECTION_COMPLETED -> ANSWER_COMPLETED (text, diff)   [+ ANSWER_FINALIZED for VALIDATED_FINAL]
```

Observability (brief §56), in order per version:

```
CLAIM_PLAN_CREATED, ANSWER_GENERATION_STARTED, LLM_CALL (request hash, output, measured tokens; timings under wall),
ANSWER_GENERATION_COMPLETED, CLAIMS_EXTRACTED, CLAIM_VERIFICATION_STARTED, CLAIM_VERIFIED / CLAIM_REJECTED,
CLAIM_REPAIRED, VALIDATION_RETRIEVAL, CITATION_CREATED, CITATION_VALIDATED, ANSWER_VALIDATED, ANSWER_REVISED
```

`TURN_COMPLETED.answer` (previously `null`) now carries the final answer's id, version, status, partial flag, text, claim ids, citation keys, backend and metrics.

## Sentence gating

- **Validated before release.** Claims are released (`ANSWER_CLAIM_READY`) only after verification, policy and citation validation. No unvalidated factual text is emitted as final (REQ-GRD-009).
- **No mid-answer streaming.** Structured JSON is generated in one call, so the gate is per answer version, not per streamed sentence.
- **Two latencies are recorded:** `ttft_raw_ms` (the model's first token) and the time to the first released claim.

## Replay

- **LLM outputs are part of the trace.** Each `LLM_CALL` stores the request hash and the raw output. `ReplayEngine` feeds them back through a `RecordedBackend`, so a session that used the local LLM replays **exactly** (tested), without the model.
- **Realtime limitation.** Generation runs synchronously at turn completion. In realtime mode the event loop waits for the model (seconds); asynchronous generation is a Phase 8 item.
