"""Corpus pipeline: CorpusSource -> loader -> normalizer -> section parser -> chunker -> chunks + manifest data.

Invalid documents are skipped (or raise, per ``corpus.on_invalid_document``) and always recorded in the
manifest's ``source_files`` with the reason — never silently dropped.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

from streamrag.corpus.metadata import sanitize_metadata
from streamrag.config.settings import StreamRagConfig
from streamrag.corpus.chunker import chunk_section, count_tokens
from streamrag.corpus.ids import assign_document_ids, native_document_id, render_chunk_id, render_citation
from streamrag.corpus.loaders import RawDocument, humanize_stem, load_document
from streamrag.corpus.normalize import normalize_pages, page_at
from streamrag.corpus.sections import parse_sections
from streamrag.corpus.source import CorpusSource
from streamrag.errors import DocumentLoadError, EmptyCorpusError
from streamrag.models.corpus import CorpusChunk, CorpusDocument, SourceFile
from streamrag.telemetry.logging import get_logger
from streamrag.telemetry.timing import Stopwatch

log = get_logger("corpus")


@dataclass
class BuiltCorpus:
    corpus_root: str
    corpus_hash: str
    is_test_fixture: bool
    documents: list[CorpusDocument]
    chunks: list[CorpusChunk]
    source_files: list[SourceFile]
    timings_ms: dict[str, float] = field(default_factory=dict)

    @property
    def section_count(self) -> int:
        return sum(len(d.sections) for d in self.documents)


def render_index_text(template: str, title: str, section_path: list[str], section_title: str, text: str) -> str:
    header_parts: list[str] = []
    for part in [title, *section_path]:
        if part and (not header_parts or header_parts[-1] != part):
            header_parts.append(part)
    return template.format(header=" > ".join(header_parts), title=title, section_path=" > ".join(section_path),
                           section_title=section_title, text=text)


def build_corpus(cfg: StreamRagConfig) -> BuiltCorpus:
    source = CorpusSource(cfg.paths.corpus, cfg.corpus.include_extensions, cfg.corpus.fixture_marker)
    sw = Stopwatch().__enter__()
    entries = source.entries()                       # raises CorpusNotFoundError / EmptyCorpusError
    corpus_hash = source.corpus_hash(entries)
    sw.lap("scan_hash")

    raws: list[RawDocument] = []
    files: dict[str, SourceFile] = {}
    for e in entries:
        try:
            raws.append(load_document(e))
            files[e.relpath] = SourceFile(path=e.relpath, sha256=e.sha256, bytes=e.size, format=e.extension.lstrip("."))
        except DocumentLoadError as exc:
            if cfg.corpus.on_invalid_document == "error":
                raise
            files[e.relpath] = SourceFile(path=e.relpath, sha256=e.sha256, bytes=e.size,
                                          format=e.extension.lstrip("."), status="skipped", reason=str(exc))
            log.warning("document_skipped", extra={"fields": {"path": e.relpath, "reason": str(exc)}})
    sw.lap("load")
    if not raws:
        raise EmptyCorpusError(f"no loadable documents under {cfg.paths.corpus} "
                               f"({sum(1 for f in files.values() if f.status != 'loaded')} skipped)")

    doc_ids = assign_document_ids([r.entry.relpath for r in raws], [r.front_matter for r in raws],
                                  cfg.corpus.doc_id_strategy, cfg.corpus.native_doc_id_pattern)
    documents: list[CorpusDocument] = []
    chunks: list[CorpusChunk] = []
    t_norm = t_sec = t_chunk = 0.0
    for raw, doc_id in zip(raws, doc_ids):
        with Stopwatch() as s1:
            norm = normalize_pages(raw.pages, raw.paged, cfg.normalization, cfg.sections, raw.format)
        t_norm += s1.ms
        if not norm.text.strip():
            files[raw.entry.relpath] = files[raw.entry.relpath].model_copy(
                update={"status": "skipped", "reason": "empty after normalization"})
            continue
        with Stopwatch() as s2:
            title, sections = parse_sections(doc_id, norm.text, norm.page_offsets, raw.format,
                                             humanize_stem(raw.entry.relpath), cfg.sections, raw.front_matter)
        t_sec += s2.ms
        native = native_document_id(raw.entry.relpath, raw.front_matter, cfg.corpus.native_doc_id_pattern) is not None
        doc = CorpusDocument(document_id=doc_id, native_id=native, source_path=raw.entry.relpath,
                             source_sha256=raw.entry.sha256, format=raw.format, title=title, text=norm.text,
                             page_offsets=norm.page_offsets, sections=sections, normalization_log=raw.log + norm.log,
                             metadata=sanitize_metadata(raw.front_matter, cfg.corpus.metadata_fields))
        documents.append(doc)
        files[raw.entry.relpath] = files[raw.entry.relpath].model_copy(update={"document_id": doc_id})
        with Stopwatch() as s3:
            for sec in sections:
                for part, span in enumerate(chunk_section(sec, norm.text, cfg.chunking), start=1):
                    text = norm.text[span.start:span.end]
                    chunks.append(CorpusChunk(
                        chunk_id=render_chunk_id(cfg.ids.chunk_id_template, doc_id, sec.section_id, part),
                        document_id=doc_id, section_id=sec.section_id, part=part,
                        citation=render_citation(cfg.ids.citation_template, doc_id, sec.section_id),
                        title=title, section_title=sec.title, section_path=sec.path, text=text,
                        index_text=render_index_text(cfg.index_text.template, title, sec.path, sec.title, text),
                        source_path=raw.entry.relpath, char_start=span.start, char_end=span.end,
                        page_start=page_at(norm.page_offsets, span.start), page_end=page_at(norm.page_offsets, span.end),
                        position=len(chunks), position_in_section=part - 1, token_count=count_tokens(text),
                        text_sha1=hashlib.sha1(text.encode()).hexdigest(), starts_mid_sentence=span.starts_mid_sentence))
        t_chunk += s3.ms
    if not chunks:
        raise EmptyCorpusError("corpus produced zero chunks (all documents empty after normalization?)")
    ids = [c.chunk_id for c in chunks]
    if len(set(ids)) != len(ids):  # pragma: no cover - guarded by section-id uniqueness
        raise RuntimeError("chunk id collision")
    sw.__exit__()
    timings = dict(sw.laps, normalize=t_norm, sections=t_sec, chunking=t_chunk, total=sw.ms)
    return BuiltCorpus(corpus_root=str(cfg.paths.corpus), corpus_hash=corpus_hash,
                       is_test_fixture=source.is_test_fixture(), documents=documents, chunks=chunks,
                       source_files=[files[e.relpath] for e in entries], timings_ms=timings)
