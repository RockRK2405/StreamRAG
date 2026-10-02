# 02: Sections, IDs and Chunking

**Code:** `corpus/sections.py`, `corpus/ids.py`, `corpus/chunker.py`
**Experiment:** `research/phase3/chunking_report.md`

## Purpose

Produce citation-unit sections and retrieval-unit chunks with **stable, deterministic IDs**.

## Sections

**Heading kinds**, in priority order:

1. Markdown `#…` (may carry a number, e.g. `## 2.1 Title`).
2. `§ N Title`, and `Section/Article/Clause/Part/Chapter N`.
3. Numbered `N.` / `N.M` with title-like text: no terminal punctuation, at most 12 words, capitalized.
4. ALL-CAPS lines, used only when a document has none of the above.

**List protection.** A run of at least 2 adjacent numbered candidates of the same level, *numbered consecutively*, with no body text between them, is a list, not headings. The sequence check stops a list from swallowing a real heading that follows it. That was a bug found and fixed with a regression test.

**Section IDs:**

| Case | ID |
|---|---|
| Native number in the source | That number (`2.1`, `4`) |
| No native number | Ordinal path (`1`, `1.2`); prefixed `u` in documents that also have native numbers, so the two can never collide |
| Text before the first heading | `0` (preamble) |
| Heading-less PDF | `p<N>` (one section per page) |
| Duplicates | Suffixed `-2`, `-3` |

Sections keep: `level`, `path` (ancestor titles), `parent_id`, char span, page span, and paragraphs with char spans and pages.

**Document title:** front-matter `title`, else the single leading Markdown H1, else a leading ALL-CAPS line (txt/pdf), else the humanized file name.

## IDs

| ID | Rule |
|---|---|
| `document_id` | A native ID wins: front-matter `id`, or a file name matching `native_doc_id_pattern` (e.g. `Doc_07_x.txt` → `Doc_07`). Otherwise `native_or_stem` (default) gives the sanitized relative path, or `native_or_ordinal` gives `Doc_<n>`. |
| `chunk_id` | `chunk_id_template` = `{document_id}§{section_id}#{part}`, e.g. `Doc_07§2.1#1` |
| Citation key | `citation_template` = `{document_id} §{section_id}`, e.g. `Doc_07 §2.1`, the guide's `[Doc_ID §Section]` format |

The default document-ID strategy changed from Phase 2's "ordinal" to `native_or_stem`. Stem-based IDs do not shift when files are added, and they are readable. Native corpus IDs always take precedence. See ADR-013.

## Chunking strategies (`chunking.strategy`)

| Strategy | Behavior |
|---|---|
| `paragraph` **(default)** | Pack whole paragraphs to `target_tokens` (180), hard cap `max_tokens` (300). Oversized paragraphs are split at sentence boundaries, with an abbreviation guard. A paragraph ending with `:` is glued to the following list. Overlap means trailing whole sentences (≤ `overlap_tokens`, 40) carried into the next chunk, only when a split falls *inside* a paragraph. A trailing piece under `min_tokens` merges into its predecessor. |
| `section` | One chunk per section. Sections over `max_tokens` fall back to `paragraph`. |
| `token` | Fixed word windows with overlap. Baseline only: it splits sentences. |

**Guarantees (tested):**
- No chunk crosses a section boundary.
- Chunk text is an exact slice of the normalized document.
- Output is deterministic.
- `token_count ≤ max_tokens`.

`index_text = "{header}: {text}"`, where `header` is the document title followed by the section path, deduplicated. Both BM25 and the embedder index it.

## Inputs, outputs and configuration

- **Input:** normalized document text, page offsets, format.
- **Output:** `CorpusSection[]`, then `CorpusChunk[]`.
- **Configuration:** `sections.*`, `ids.*`, `chunking.*`, `index_text.template`.

## Failure modes

- Ambiguous layouts (e.g. a numbered list item that looks like a title) can be misread as a heading. This is documented in code and covered by tests.
- A run-on "sentence" longer than `max_tokens` is cut into word windows and flagged `starts_mid_sentence`.

## Trade-offs

The 180/300 default never splits a sentence, never crosses sections and never exceeds bge-small's 512-wordpiece window (measured). Smaller chunks would give more precise citations, larger ones more context. That choice is deferred to Exp 10 on the real corpus.
