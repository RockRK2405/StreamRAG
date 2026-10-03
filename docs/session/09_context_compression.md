# 09: Relevant-Context Selection and Context Compression

**Code:** `context/selector.py` (`RelevantContextSelector`, `ContextCompressor`, `token_counter`)

## Relevant-context selection

`RelevantContextSelector.select(mode, intent_id=None)` returns a `ContextPackage`. This is what a later stage (Phase 7 generation) may see: never "the whole history".

| Mode | Content |
|---|---|
| `none` | the current utterance only |
| `full_transcript` | every utterance verbatim (only those still inside the transcript window survive) |
| `structured` | for the active frame (or one need's frame): active needs (resolved text), their **active** constraints, the frame's entities, the selected SUPPORTED / PARTIALLY_SUPPORTED claims, and the usable evidence ids with citations |

**Exclusions are counted, never silent.** `excluded` maps each reason to a count:
- `frame_dormant`
- `need_not_active`
- `constraint_retracted_or_replaced`
- `claim_<status>`
- `evidence_<status>`

**Provenance.** Every item carries it:
- needs, constraints and entities: utterance id plus character span;
- claims: evidence id plus character span (a test checks the claim is verbatim at its span).

**Size.** Characters always, tokens only when a tokenizer is supplied. Reported separately as `size_by_kind["conversation"]` (needs / constraints / entities / utterances) and `["retrieval"]` (claims / evidence).

## Context compression

`ContextCompressor.compress()` represents the conversation as structured state:
- per utterance: its SHA-1, its length, and the needs and constraints derived from it, with spans;
- the verbatim text only for utterances inside `transcript_window` (default 6).

Outside the window the tracker drops the utterance's text too (`IntentTracker.compress_transcript`). Its needs and spans stay; only the raw words go. Reported sizes:
- `raw`: the verbatim conversation;
- `compressed`: window verbatim plus derived state of older turns;
- `structured`: derived state only;
- `ratio_chars` for both.

## Measured: memory ablation (`research/phase6/results/ablation.json`; 64 dev turns, fixture, NOT REPORTABLE)

Tokens are counted with the bge-small WordPiece tokenizer as a **proxy**; the Phase 7 generator's tokenizer is not known.

| Arm | Need-query correctness | Queries with context leaks | Retrieval calls | Wall ms / turn p50 (mean) | Conversation context tokens / turn p50 (mean) |
|---|---|---|---|---|---|
| A no session memory (each turn alone) | 0.594 | 0 | 49 | 5.20 (4.43) | 7 (7.6) |
| B full transcript memory (fresh session over the concatenated transcript) | 0.938 | 8 | 81 | 5.52 (6.60) | 13 (14.0) |
| C structured session memory | **0.984** | 0 | 59 | 5.63 (5.51) | 13 (11.8) |

**Reading the ablation.**
- **A** loses every late detail: "Specifically overnight." alone has no need to refine.
- **B** recovers most needs, but carries finished topics into new queries (8 leaks: earlier topics' words in later queries). It also re-retrieves everything each turn.
- **C** keeps only the active frame. Its conversation context is similar in size to B on these 2–5-turn sessions. By construction B grows with session length and C with the active frame's needs; this was not measured beyond these short sessions.
- The evidence and claim part of C's package (the retrieval context) was 69.5 tokens per turn (p50).

No generation tokens exist yet, so no token *savings* for generation are claimed.
