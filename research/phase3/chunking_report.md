# Phase 3 — Chunking Experiment

- **Script:** `research/phase3/chunking_experiment.py`
- **Raw results:** `results/chunking_experiment.json`
- **Date:** 2026-10-02

> **Status: provisional.** The official corpus is unavailable, so the chunk size cannot be tuned against retrieval quality (Exp 10 is blocked). This experiment compares configurations on **structural** properties that matter regardless of quality: statement integrity, section preservation, duplication and embedder truncation.

## Datasets (none is the official corpus)

| Dataset | What it is | Why it is used |
|---|---|---|
| `fixture` | 3 fictional test documents (`tests/fixtures/corpus`, TEST_FIXTURE_ONLY) | Software behavior |
| `synthetic` | 40 generated documents, 8 numbered sections × 3 paragraphs (`synth.py`) | Larger, uniform structure |
| `pdf_smoke` | The other hackathon theme guide PDFs from the user's Downloads, copied to a scratch dir with a TEST_FIXTURE_ONLY marker. 4 loaded; 2 skipped (see below). | Real-world PDF layouts (headings, lists, page furniture). Only statistics are reported; no content. |

**Skipped in `pdf_smoke`, by design:** `Theme 4 Guide_RAG.pdf` and `Theme 2_…Engine.pdf`. Both have **no extractable text** (image-only). The pipeline recorded them as `skipped` with that reason rather than indexing empty documents. If the official corpus contains scanned PDFs, an OCR adapter will be required (see the Phase 3 report, §17).

## Configurations

Tokens are regex words. ★ marks the current default.

| Name | Strategy | Target / max tokens | Overlap |
|---|---|---|---|
| section(max300) | one chunk per section; paragraph fallback above 300 | — / 300 | 0 |
| para(120/200,ov0) | paragraph packing | 120 / 200 | 0 |
| para(120/200,ov40) | paragraph packing | 120 / 200 | 40 (sentences, inside split paragraphs only) |
| para(180/300,ov0) | paragraph packing | 180 / 300 | 0 |
| **para(180/300,ov40) ★** | paragraph packing | 180 / 300 | 40 |
| para(250/350,ov40) | paragraph packing | 250 / 350 | 40 |
| token(128,ov32) | fixed word windows | 128 | 32 |
| token(256,ov64) | fixed word windows | 256 | 64 |

## Results: `pdf_smoke` (most realistic layout)

| Config | Chunks | Mean tok | Min | Max | p95 | Char duplication ratio | Exact dup rate | Sections kept whole | Crossing sections | Start mid-sentence | **End mid-sentence** | Truncated @512 wp |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| section(max300) | 48 | 96.7 | 5 | 297 | 270 | 0.966 | 0 | 0.853 | 0 | 0 | **0** | 0 |
| para(120/200,ov0) | 59 | 78.7 | 5 | 199 | 183 | 0.966 | 0 | 0.765 | 0 | 0 | **0** | 0 |
| para(120/200,ov40) | 61 | 82.1 | 5 | 199 | 182 | 1.039 | 0 | 0.765 | 0 | 0 | **0** | 0 |
| para(180/300,ov0) | 49 | 94.7 | 5 | 296 | 262 | 0.966 | 0 | 0.824 | 0 | 0 | **0** | 0 |
| **para(180/300,ov40) ★** | 49 | 94.7 | 5 | 296 | 262 | 0.966 | 0 | 0.824 | 0 | 0 | **0** | 0 |
| para(250/350,ov40) | 46 | 100.9 | 5 | 296 | 265 | 0.966 | 0 | 0.824 | 0 | 0 | **0** | 0 |
| token(128,ov32) | 67 | 85.0 | 5 | 128 | 128 | 1.178 | 0 | 0.765 | 0 | 31 | **57** | 0 |
| token(256,ov64) | 46 | 117.6 | 5 | 256 | 256 | 1.115 | 0 | 0.794 | 0 | 12 | **37** | 0 |

## Results: `synthetic` (40 documents)

| Config | Chunks | Mean tok | Max | Sections kept whole | End mid-sentence | Char duplication ratio |
|---|---|---|---|---|---|---|
| section(max300) | 360 | 169.5 | 216 | 1.000 | 0 | 0.979 |
| para(120/200,*) | 871 | 70.1 | 120 | 0.111 | 0 | 0.976 |
| **para(180/300,*) ★** | 595 | 102.6 | 180 | 0.347 | 0 | 0.978 |
| para(250/350,ov40) | 360 | 169.5 | 216 | 1.000 | 0 | 0.979 |
| token(128,ov32) | 680 | 104.8 | 128 | 0.111 | **680** | 1.142 |
| token(256,ov64) | 360 | 169.5 | 216 | 1.000 | **360** | 0.978 |

On `fixture`, all strategies produce the same 14 chunks, because every fixture section is small. The token strategies still end 13 of 14 chunks mid-sentence, since a window ends at the section end rather than at a sentence boundary.

## Findings (measured)

1. **Fixed token windows split statements.** On real PDF layouts, `token(128)` ends 57 of 67 chunks mid-sentence and `token(256)` ends 37 of 46. Every paragraph- or section-aware configuration ends **0** chunks mid-sentence. This is the strongest structural result: the brief's "avoid splitting important policy statements across chunks" rules out token windows.
2. **No configuration crosses a section boundary.** This holds by construction, and it is verified per chunk. Citation keys therefore always point at exactly one section.
3. **No chunk exceeds bge-small's 512-wordpiece window** in any configuration, measured on `index_text`, which includes the "title > section path:" header. The max-300-word cap is safe for bge-small. It would *not* be safe for MiniLM's 256-wordpiece window, a further argument for bge-small.
4. **Overlap has almost no effect with paragraph packing.** Overlap applies only when an oversized paragraph is split at sentences. On `pdf_smoke`, overlap 40 adds 2 chunks and 7% duplicated characters at 120/200, and nothing at 180/300. Token windows duplicate 11–18% of characters.
5. **Larger targets keep more sections whole** (0.824–0.853 at 180–300 vs 0.765 at 120). Smaller chunks give more precise citations and less noise per evidence item. **Which side of that trade-off retrieves better is a quality question (Exp 10).**

## Decision

**Keep `paragraph`, target 180 / max 300, overlap 40, min 25 (the current default), provisionally.**

- It never splits a sentence.
- It never crosses a section.
- It never truncates.
- Its moderate chunk count (an index-size and latency factor) keeps about 82% of sections whole on real layouts.
- `section(max300)` is the main alternative. It is nearly identical on PDFs (48 vs 49 chunks).

Exp 10 on the official corpus decides between `paragraph(180/300)`, `paragraph(120/200)` and `section(max300)` by Recall@5 and citation precision. The script reruns with one command:

```bash
.venv/bin/python research/phase3/chunking_experiment.py --scratch /tmp/x   # add --pdf-dir for PDFs
```

## Limitations

- Sentence boundaries come from a regex splitter with an abbreviation guard. "End mid-sentence" is a proxy for statement integrity, not a semantic judgment.
- Tiny chunks (minimum 5 tokens on PDFs) come from sections whose body is a single short line. They are kept so that no text is dropped, and `min_tokens` merging only applies within a section.
