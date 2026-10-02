# 01: Corpus Pipeline

**Code:** `src/streamrag/corpus/`
**Spec:** Phase 2 §10.1

## Purpose

Turn a directory of documents into normalized, sectioned, traceable chunks. This is the **only** way content enters the system.

## Architecture

```
CorpusSource (scan, hash, fixture detection)
  → loaders (txt | md | pdf)
  → normalize
  → sections
  → ids
  → chunker
  → BuiltCorpus
  → manifest
```

| Module | Responsibility |
|---|---|
| `source.py` | Deterministic file discovery (sorted POSIX paths, hidden files ignored, extension allow-list); `corpus_hash` over every file's bytes; `TEST_FIXTURE_ONLY` marker detection; `status()` returns AVAILABLE / NOT_AVAILABLE / TEST_FIXTURE |
| `loaders.py` | Format adapters. UTF-8 with cp1252 fallback (logged); Markdown front matter (`id`, `title`); PDF via pypdf, one string per page. Image-only, encrypted or unreadable PDFs and empty files raise `DocumentLoadError`. |
| `normalize.py` | Deterministic cleanup with every removal logged (see below) |
| `headings.py` / `sections.py` | Heading recognition and the section tree (doc 02) |
| `ids.py` | Document, chunk and citation IDs (doc 02) |
| `pipeline.py` | Orchestration, per-stage timings, the skip/error policy |
| `manifest.py` | `CorpusManifest` and its content hash (doc 08) |

### Normalization steps (all configurable under `normalization:`)

1. Unicode NFKC; remove invisible characters (ZWSP, BOM, soft hyphen); unify line endings and whitespace.
2. Drop consecutive duplicate lines (a PDF extraction artifact). Logged.
3. **Paged sources only:** remove repeated header/footer lines and page-number lines.
   - Top and bottom positions are tracked separately.
   - Only one edge line per side is considered on short pages, so short-page body text is never removed.
4. De-hyphenate words split across a line break (`calibra-\ntion` → `calibration`).
5. Join hard-wrapped lines. Headings, list items, table rows and ALL-CAPS title lines are protected, and headings become their own blocks.
6. Join pages, tracking `(page, char_start, char_end)`. A paragraph that continues across a page break is joined with a space.

## Inputs and outputs

| | |
|---|---|
| **Input** | `paths.corpus`, a directory |
| **Output** | `BuiltCorpus{documents, chunks, source_files, corpus_hash, is_test_fixture, timings}` |

**Traceability invariant (tested for every chunk):** `document.text[chunk.char_start:chunk.char_end] == chunk.text`. The normalized document texts are persisted in `documents.jsonl`, and each document keeps its `source_path`, `source_sha256` and page offsets.

## Configuration

`corpus.*` (extensions, fixture marker, `on_invalid_document: skip|error`, `doc_id_strategy`), `normalization.*`, `sections.*`.

## Failure modes

| Condition | Behavior |
|---|---|
| Missing directory | `CorpusNotFoundError` |
| No supported files | `EmptyCorpusError` |
| Invalid, empty or image-only document | Skipped and **recorded in the manifest with its reason**, or raised under `on_invalid_document: error` |
| Two documents with the same native ID | `CorpusIntegrityError` (fail fast) |
| All documents empty | `EmptyCorpusError` |

## Trade-offs

- Heading detection is lexical and generic, with no corpus-specific vocabulary (anti-hardcoding). Unusual layouts may need a parser tweak, verified on the real corpus.
- Only txt, md and pdf adapters exist, which is enough for plausible formats. DOCX, HTML or OCR are added only if the official corpus needs them.
- Normalization never rewrites words. That keeps citations faithful, at the cost of leaving some PDF artifacts in place.
