# 10: Memory Boundaries, Safety, Window, Reset

**Code:** `session/safety.py` (`redact`), `session/memory.py`, `session/pipeline.py` / `multi_retrieval/coordinator.py` (redaction at ingestion), `context/detector.py` (`FrameManager`)

## Context boundaries (no contamination)

- **Topic frames isolate needs** (doc 03). Constraints apply only to the needs they were scoped to (`Constraint.applies_to`). A parallel follow-up inherits the previous need's *aspect* ("rules") but not its constraints.
- **New topics start clean.** A new independent question opens a new frame, so its query is built from its own words only. In the dev suite, 0 queries contained terms the gold forbids (`forbidden_new`, context isolation).
- **Earlier topics are reachable, not ambient.** A dormant frame is reactivated only when the user names its topic again ("Back to the ladders, …").
- **Relevance-based inheritance.** Anaphora and ellipsis inherit from the most recent utterance that has active needs, or that changed earlier needs, never from the whole history.

## Memory window

- `session.transcript_window` (default 6) utterances are kept verbatim (redacted).
- Older utterances keep only a SHA-1 and a character count. Their needs, constraints and spans stay in semantic memory: the useful state survives, the words do not.
- The window applies to the interpretation layer too (`IntentTracker.compress_transcript`).

## Memory safety

Do not store unnecessary personal information, secrets, credentials or tokens.

- **Redaction at ingestion.** With `session.redact_pii: true` (default), the text that interpretation sees is redacted before the tracker receives it. This holds both in the synchronous pipelines and in the streaming coordinator. So PII never enters semantic memory, queries, the ledger, snapshots or archives (tested with an e-mail address in a question).
- **Patterns:**
  - e-mail addresses;
  - "password / token / api key / secret: value";
  - card-like digit runs (13–19);
  - phone-like digit runs;
  - long mixed letter-and-digit tokens (≥ 24 characters).

  Short numbers that are task content ("40 crates", "5 high", "after 9 pm", "4 millimetres") are kept (tested).
- **Archives** (`archive_session`) keep hashes and counts only: no transcript or evidence text (tested).
- **Limitations:**
  - redaction is pattern-based, so names and street addresses are not detected;
  - a partially streamed number may not match yet;
  - the Phase 4 telemetry trace (`CHUNK_RECEIVED`) still carries the raw input, because replay needs it. Traces are debugging artifacts, not session memory, and should be treated as sensitive.

## Reset and archive

- **`reset_session()`** clears every layer: transcript, frames, needs, constraints, ledger, cache, evidence, claims, answers, versions, and the engine extras. The session keeps working for a new conversation (tested).
- **`archive_session()`** returns a `SessionArchive` with:
  - config and index hashes;
  - utterance count and transcript hashes;
  - change types, query statuses, evidence and claim statuses;
  - number of answer versions, counters.

**Not implemented (brief §50):** long-term user memory across sessions.
