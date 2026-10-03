# 03: Context Change Detection

**Code:**
- `context/detector.py`: `ContextChangeDetector`, `FrameManager`, `semantic_diff`, `net_change_types`
- `context/cues.py`: the late-detail gate
- `intents/decomposer.py`: cross-turn operations and elliptical follow-ups
- `intents/tracker.py`: applying operations to earlier needs

**Schema:** `docs/schemas/ContextChange.schema.json`

## Where changes come from

1. **The gate.** The Phase 4 controller still decides whether an utterance is worth interpreting. Phase 6 adds the **late-detail gate**. When a need is active, a decision of `not_retrieval_worthy` / `not_yet_retrieval_worthy` / `low_specificity` / `awaiting_stability` still opens the gate if the utterance carries one of these cues, plus content beyond the cue words:
   - a correction or retraction phrase;
   - a focus or condition marker at the start;
   - a restriction preposition at the start.

   A bare "Specifically" mid-stream waits for its content. Suppressed acts (backchannels, closings) never open the gate.
2. **The interpretation update.** `IntentTracker.update` re-decomposes the utterance against the session context. Phase 6 adds **cross-turn operations** applied to earlier needs:
   - `set`: a constraint-only utterance ("Specifically overnight.") restricts the previous need(s). Its scope is the need sharing terms with it, otherwise all recent needs (confidence 0.6 when more than one).
   - `update`: a new value for the same constraint dimension (shared head term) replaces the old constraint ("night shift" → "day shift").
   - `retract`: "ignore / forget / never mind / regardless of … X" retracts the active constraint with the best term overlap.
   - **elliptical follow-up** "what about Z?": the corpus decides. If some chunk contains Z's terms together with the previous need's most specific topic term (max IDF), Z is a **constraint** of that need. Otherwise Z is a **parallel need** that inherits the previous need's aspect only ("rules" + "crates"), not its constraints.
   - **topic return**: if the rest of the utterance names an earlier need's topic ("Back to the ladders, what about Z?"), the follow-up refers to that need, and its dormant frame is reactivated.
3. **The detector.** It compares the engine's last-known need versions with the tracker's new ones, semantically (analyzed term sets, topic, aspect, constraint ids). It never compares raw strings: a casing or punctuation difference yields no term diff (tested).

## `ContextChange`

Fields:
- `change_id` (`CH<n>`), `change_type`, `utterance_id`
- `affected_intents`, `new_intents`, `superseded_intents`
- `added_constraints`, `removed_constraints`
- `affected_queries`
- `relation`: `follow_up` / `independent` / `correction` / `same_need`
- `frame_action`: `same` / `new_frame` / `reactivated` / `none`
- `diffs`: `SemanticDiff` per need, with terms added / removed, topic and aspect before / after, constraints added / removed
- `cue`
- `confidence`, `confidence_signals`
- `session_version_from`, `at_ms`

**Confidence is computed, never authored.** It is the minimum of the signals the change rests on: the decomposer's confidence of each new or modified need, and the `scope_confidence` of each added constraint. Explicit removal and supersession markers count 1.0 (tested: `confidence == min(signals)`).

## Topic frames (context boundaries)

A new need joins the active frame if any of these hold:
- it is related to the frame (FOLLOW_UP / DEPENDENT relation, an elliptical follow-up decision, a correction of a frame need, a cross-turn change);
- it shares ≥ `frame_overlap_min` topic terms with the frame.

Otherwise the active frame goes dormant and a new one opens. A dormant frame whose topic terms the new needs share is reactivated. A frame opened by a mid-stream fragment whose needs all disappeared again is dropped. Frame ids are never reused.

## Streaming: provisional vs net changes

While an utterance streams, every gate-open tick may produce changes. For example, "What are the rules" → "… for ladders" → "… in the orchard" gives NEW_INTENT then two REFINEMENTs. All of them stay in the trace. `net_change_types` summarises the turn:
- changes to needs created in the same turn fold into their NEW_INTENT;
- fragment needs that were created and removed again vanish;
- a constraint added and then revised within the turn cancels.

On the dev suite streamed chunk by chunk:
- per-turn changes averaged 1.58 (p95 3);
- net change types matched the gold for 64 of 64 turns (`research/phase6/results/streaming_virtual.json`).

## Measured (dev suite: fixture domain, NOT REPORTABLE, not held-out)

| Set | Turns | Change-type accuracy | Note |
|---|---|---|---|
| `eval/dev_adaptive` (synchronous) | 64 | 1.000 (unambiguous 59: 1.000) | written after most rules; optimistic |
| `eval/dev_adaptive_stress` (written after the code freeze, run once) | 31 | 0.903 (unambiguous 27: 0.889) | failures: "make that the day shift instead", "not the lens, the wick", "can you drop the overnight condition?", "is that also true at night?" |
