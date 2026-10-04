"""Corpus contracts: source files, documents, sections, chunks, manifest.

Invariant (tested): for every chunk, ``document.text[chunk.char_start:chunk.char_end] == chunk.text``.
The normalized document text is persisted with the index, so every chunk is traceable to its source.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field

from streamrag.models.base import Contract


class SourceFile(Contract):
    path: str                         # POSIX path relative to the corpus root
    sha256: str
    bytes: int = Field(ge=0)
    format: str
    status: Literal["loaded", "skipped", "error"] = "loaded"
    reason: str | None = None
    document_id: str | None = None


class Paragraph(Contract):
    text: str
    char_start: int = Field(ge=0)
    char_end: int = Field(ge=0)
    page: int | None = None


class CorpusSection(Contract):
    section_id: str
    document_id: str
    title: str
    level: int = Field(ge=0)          # 0 = preamble / whole document
    path: list[str] = []              # ancestor titles (outermost first) + own title
    parent_id: str | None = None
    ordinal: int = Field(ge=0)        # position within the document
    native_id: bool = False           # section number taken from the source (e.g. "2.1", "§4")
    char_start: int = Field(ge=0)
    char_end: int = Field(ge=0)
    page_start: int | None = None
    page_end: int | None = None
    paragraphs: list[Paragraph] = []


class CorpusDocument(Contract):
    document_id: str
    native_id: bool
    source_path: str
    source_sha256: str
    format: str
    title: str
    text: str                         # normalized full text (citations point into this)
    page_offsets: list[tuple[int, int, int]] = []   # (page_number, char_start, char_end)
    sections: list[CorpusSection] = []
    normalization_log: list[str] = []  # e.g. removed headers/footers, encoding fallbacks
    metadata: dict[str, str | int | float | bool] = {}   # Phase 9: sanitised whitelisted front matter


class CorpusChunk(Contract):
    chunk_id: str
    document_id: str
    section_id: str
    part: int = Field(ge=1)
    citation: str
    title: str                        # document title
    section_title: str
    section_path: list[str] = []
    text: str
    index_text: str                   # text actually indexed (template applied)
    source_path: str
    char_start: int = Field(ge=0)
    char_end: int = Field(ge=0)
    page_start: int | None = None
    page_end: int | None = None
    position: int = Field(ge=0)       # global order in the corpus
    position_in_section: int = Field(ge=0)
    token_count: int = Field(ge=0)
    text_sha1: str
    starts_mid_sentence: bool = False


class EmbeddingInfo(Contract):
    name: str
    repo: str | None = None
    revision: str | None = None
    runtime: str
    dimension: int = Field(ge=1)
    pooling: str | None = None
    normalize: bool = True
    max_length: int | None = None
    model_files_sha256: dict[str, str] = {}
    truncated_chunks: int = Field(0, ge=0)
    build_batch_size: int | None = None


class CorpusManifest(Contract):
    """Everything needed to reproduce an index. Never contains benchmark answers."""

    index_version: str
    corpus_version: str               # == corpus_hash
    corpus_root: str
    is_test_fixture: bool
    generated_at: str                 # informational; excluded from content_hash
    source_files: list[SourceFile]
    document_count: int = Field(ge=0)
    section_count: int = Field(ge=0)
    chunk_count: int = Field(ge=0)
    chunking_configuration: dict[str, Any]
    normalization_configuration: dict[str, Any]
    sections_configuration: dict[str, Any]
    ids_configuration: dict[str, Any]
    index_text_configuration: dict[str, Any]
    bm25_configuration: dict[str, Any]
    embedding_model: EmbeddingInfo | None = None
    index_config_hash: str
    artifacts: dict[str, str] = {}    # artifact file name -> sha256
    environment: dict[str, Any] = {}
    content_hash: str = ""            # sha256 over everything above except generated_at/environment
