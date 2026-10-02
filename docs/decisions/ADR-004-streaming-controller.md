# ADR-004: Streaming Retrieval Controller

- **Status:** Accepted
- **Phase 4 update:** implemented as `controller/`. Decisions are WAIT / RETRIEVE / SKIP, where SKIP = Phase 2 NO_RETRIEVE (suppressed / not worthy) plus ledger-hit (redundant) / budget. Single active query per utterance. The prototype-classifier variant is implemented for Exp 8. See `docs/streaming/03` and ADR-014.
- **Date:** 2026-10-02
- **Spec:** §8, §15; corrections **K4**, **K6**

## Context

The controller must decide WAIT / RETRIEVE / NO_RETRIEVE as chunks arrive [G§2 p2]. G2 needs retrieval to start before the end of the utterance in ≥80% of eligible queries, with low false triggers on no-retrieval cases [G§5 p5]. Guide Example 1 logs retrieval at the *same timestamp* as the enabling chunk (0.8 s) [G§4 p3–4]. Retrieving on every chunk is an explicit pitfall [G§6 p5 #1].

## Decision

- **A rule-first controller, run per segment, with no LLM.**
- Retrieval fires on **segment events**: clause closure, stability (no new tokens for 0.6 s), or forced close at 25 content tokens. It never fires on chunk arrival.
- **Guards before dispatch:** not dangling; ≥2 content tokens; ≥1 corpus-vocabulary anchor (IDF dictionary lookup); novelty against the query ledger; per-utterance provisional budget (4); minimum interval (0.4 s).
- **Suppression** (NO_RETRIEVE) only for high-confidence PRESENTATION, SOCIAL, BACKCHANNEL and META acts (confidence ≥ 0.8). PRESENTATION also requires a prior answer.
- **K4/K6:** every information request is retrieved at the latest at `UTTERANCE_END`. Corpus anchors only make retrieval *earlier*; they never suppress it.
- **Model-based ablation** (Exp 8): a prototype-embedding or small-LLM act classifier behind the same interface, with timing, budget and ledger logic unchanged.

## Alternatives

| Alternative | Why not chosen |
|---|---|
| Retrieve on every chunk | Storms, noise, false triggers |
| Wait for the end of the utterance | G2 = 0 |
| LLM controller per chunk | +0.3–2 s per decision [E], cost; erodes G2 timing |
| Learned classifier trained on our labels | Overfitting risk; C2 optics |

## Why selected

It meets the timing implied by Ex1 (milliseconds per decision), is explainable (closed reason-code vocabulary), is cheap, and the ablation the guide asks for falls out naturally.

## Trade-offs

- Generic lexical rules may miss unusual phrasings. Mitigations: fail toward retrieval at the end of the utterance; the gated LLM check in decomposition; tune/test split calibration.

## Consequences

- Provisional evidence must be re-scored at finalize (REQ-STREAM-004).
- Lexicons live in config and are disclosed.
- G2 eligibility is defined by us (Q4) until the organizers clarify.
