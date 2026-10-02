# 08: Incremental Intents in a Stream

**Code:** `multi_retrieval/coordinator.py` (`MultiIntentCoordinator`), `streaming/session.py` (Phase 4 session; `multi_intent.enabled: true`)

## Controller = gate, coordinator = per-intent work

The Phase 4 `RetrievalController` still runs on every chunk, quiet tick and utterance end. In multi-intent mode its decision is a **gate**.

| Decision | Gate | Why |
|---|---|---|
| `RETRIEVE` | open | — |
| `WAIT cooldown` / `provisional_budget_exhausted` / `retrieval_in_flight` | open | These storm guards assume one query per utterance; per-intent guards replace them |
| `SKIP redundant` / `SKIP budget` | open | Per-intent dedup and ledger hits decide |
| `SKIP suppressed` / `SKIP not_worthy` | closed | Suppression is unchanged: no intents, no retrieval for presentation, social, backchannel or meta turns |
| `WAIT no_content` / `trailing_function_word` / `low_specificity` / `awaiting_stability` / `not_yet_retrieval_worthy` | closed | The transcript is not stable or specific enough yet |

When the gate is open:
1. The **current** transcript is re-decomposed (deterministic; p50 under 1 ms, measured).
2. It is reconciled with the previous `IntentSet` version (docs/multi_intent/09).
3. Only added or modified intents are retrieved (docs/multi_intent/05).

## Example (virtual clock, fixture corpus; from `research/phase5/traces/C1.txt` and the CLI)

```
"I need information about | the fog signal | and also | the lens | especially during | a storm"
0.4 s  INTENTS v1 +I1 'the fog signal'              -> Q1 (I1)                      -> fused (I1)
0.8 s  "and also": controller WAIT trailing_function_word (gate closed)
1.2 s  INTENTS v2 +I2 'the lens'                    -> Q2 (I2) only, reuse I1       -> fused (I1, I2)
1.6 s  "especially during": marker without content -> no constraint yet, no new version
2.0 s  INTENTS v3 ~I1,I2  K1 'during a storm' -> I1,I2
                                                    -> Q3 (I1 v2), Q4 (I2 v2) in parallel
2.5 s  end: no query changed -> no retrieval; final fusion; TURN (2 intents, 4 retrievals, 3 versions)
```

## Stale intents (corrections)

"Tell me the requirements for ladders. Actually, I meant crates instead of ladders.":
1. `INTENT_SUPERSEDED I1 -> I2`.
2. I1's queued queries are cancelled; its completed queries become `stale` with `stale_reason=intent_superseded` (provenance kept).
3. I1 is excluded from fusion.
4. I2 is retrieved.

An intent that disappears from the decomposition (e.g. an ASR revision) is `DROPPED`; its queries are stale with `stale_reason=intent_removed`.

## Across utterances

- **Follow-ups:** the next utterance's decomposition receives the previous utterance's active intents as context (topics with source spans), for pronouns and ellipsis.
- **Corrections:** a correction may supersede an intent of an earlier utterance.
- **Repeats:** a repeated question reuses the earlier evidence (ledger hit).
- **Out of scope:** constraint-only follow-ups ("especially for visitors" as its own utterance) are late-detail refinement, deferred to Phase 6. They fall back to a single Phase 4-style query and are recorded in `dropped` with that reason.

## Determinism and replay

- Decomposition and dispatch run on stream time.
- Virtual multi-intent traces replay **exactly**; realtime ones replay with identical **behaviour**, including intent versions, generated queries, supersessions and final fused evidence ids (tested; `research/phase5/results/replay_check.json`).
- Wall-clock measurements (decomposition, fusion) live under `payload.wall`.
