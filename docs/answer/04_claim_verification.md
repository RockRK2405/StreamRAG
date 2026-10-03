# 04: Claim–Evidence Alignment and Verification

**Code:** `claims/aligner.py` (`ClaimEvidenceAligner`), `claims/verifier.py` (`ClaimVerifier`), `claims/nli.py` (`NliModel`), `claims/textcheck.py`
**Schema:** `docs/schemas/ClaimVerification.schema.json`

## Support is entailment, not similarity (brief §10-11)

A chunk can be about the question without stating the asked fact. Support is therefore decided by **entailment**:
- by an NLI cross-encoder (`cross-encoder/nli-deberta-v3-xsmall`, Apache-2.0, pinned revision `a150876…`, ONNX, CPU, about 5 ms per pair);
- plus deterministic rules.

Retrieval scores and term overlap never count as support.

For each evidence item, the premises tried are:
- each sentence;
- each two-sentence window;
- the whole chunk.

Only sentences sharing a content term or number with the claim are scored, plus the chunk. Instruction-like sentences are never premises (doc 02).

| Strength | Rule |
|---|---|
| STRONG | a single sentence entails the claim (NLI argmax = entailment) **and** every number of the claim occurs in that sentence |
| MODERATE | only a window or the whole chunk entails it (numbers likewise) |
| CONTRADICTORY | nothing entails it, **and** a single sentence sharing ≥ half of the claim's content terms is labelled contradiction |
| WEAK | ≥ half of the claim's content terms occur, no entailment: *similar, not support* |
| NONE | otherwise |

- **Contradiction gating.** NLI models label "same topic, different statement" as contradiction. For example, a ladder-safety rule was "contradicted" by a sapling-planting sentence in the first integration run. Requiring a sentence premise about the same proposition removed that artefact.
- **Probabilities are kept but not used.** The NLI probabilities are stored as raw `signals` only. They are not calibrated, and no score is reported as confidence.
- **Numbers (ADR-008 L1).** Numbers are normalised: number words, ordinals, decimals (".4" → 0.4) and percentages. A claim number missing from the premise blocks support, whatever the model says. Without this rule, "four" vs "4" style errors and decimal hallucinations pass.

## The verifier's five questions

| Question | Field |
|---|---|
| 1 supported? | `supported`: ≥ 1 STRONG / MODERATE alignment |
| 2 which evidence? | `supporting_evidence` (citations are built from it) |
| 3 entailed? | `entailed` |
| 4 sufficient? | `sufficient`: entailed and numbers present |
| 5 contradicted? | `contradicting_evidence`: checked against the whole section pool, not only the cited items |

**Status:**
- SUPPORTED.
- CONTRADICTED. With support from other evidence, this is an *evidence conflict*: never resolved, presented with both sources.
- PARTIALLY_SUPPORTED: the statement was decomposed and some atoms are supported.
- UNSUPPORTED.
- DRAFT is the planned / unverified state of claims in drafts.

**Labels and decomposition.**
- **Invalid labels** (labels not in the evidence set) are recorded and ignored.
- **Citation repair.** Support from evidence the model did *not* cite is accepted, and reported as `support_outside_citations` / `citation_repaired`.
- **Decomposition** happens only when the whole statement is not supported. All atoms supported → SUPPORTED (composed support); some → PARTIALLY_SUPPORTED.

## Modes and limitations

- **`nli` (default)** is the entailment model plus the rules above.
- **`rules`** (no model) is entailment approximated by all content terms + numbers + negation polarity in the premise. It is strictly weaker: it accepts no paraphrase.
- **Measured accuracy** on perturbations with labels by construction is in `research/phase7/results/verifier_eval.json` and report §6.
- **Known limitations:**
  - **Long-range context:** coreference across sentences can fail ("It must be …").
  - **Generic entailments can pass:** e.g. a claim dropping a qualifier ("Ladders are permitted in the orchard" from "… only when a second worker holds the base") is often *entailed*. Removing a restriction is a real error the NLI model may accept.
  - **Modality:** shifts (must ↔ may) are not reliably caught.
  - **Language:** English only.
