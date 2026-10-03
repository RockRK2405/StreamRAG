"""Document loaders (format adapters). Only formats that a corpus plausibly uses are implemented:
plain text, Markdown, PDF. Add an adapter here when the official corpus needs another format.

Loaders return raw text (per page for paged formats). They never modify content beyond decoding;
all cleanup happens in ``normalize`` so it is configurable and logged.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from streamrag.corpus.source import SourceEntry
from streamrag.errors import DocumentLoadError


@dataclass
class RawDocument:
    entry: SourceEntry
    format: str
    pages: list[str]                 # one element for non-paged formats
    paged: bool
    front_matter: dict = field(default_factory=dict)
    log: list[str] = field(default_factory=list)


def _decode(data: bytes, log: list[str]) -> str:
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError:
        log.append("encoding: utf-8 decode failed; fell back to cp1252 (lossless for single-byte text)")
        return data.decode("cp1252", errors="replace")


_FRONT_MATTER = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.S)


def load_text(entry: SourceEntry) -> RawDocument:
    log: list[str] = []
    text = _decode(entry.abspath.read_bytes(), log)
    if not text.strip():
        raise DocumentLoadError(f"{entry.relpath}: empty document")
    return RawDocument(entry, "txt", [text], paged=False, log=log)


def load_markdown(entry: SourceEntry) -> RawDocument:
    log: list[str] = []
    text = _decode(entry.abspath.read_bytes(), log)
    front: dict = {}
    m = _FRONT_MATTER.match(text)
    if m:
        try:
            parsed = yaml.safe_load(m.group(1)) or {}
            front = parsed if isinstance(parsed, dict) else {}
        except yaml.YAMLError as exc:
            log.append(f"front matter ignored (invalid YAML: {exc.__class__.__name__})")
        # Replace front matter with blank lines of equal count so later line structure is unaffected.
        text = "\n" * m.group(0).count("\n") + text[m.end():]
    if not text.strip():
        raise DocumentLoadError(f"{entry.relpath}: empty document")
    return RawDocument(entry, "md", [text], paged=False, front_matter=front, log=log)


def load_pdf(entry: SourceEntry) -> RawDocument:
    try:
        from pypdf import PdfReader  # optional dependency: streamrag[pdf]
        from pypdf.errors import PdfReadError
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise DocumentLoadError(f"{entry.relpath}: PDF support requires 'pypdf' (pip install streamrag[pdf])") from exc
    try:
        reader = PdfReader(str(entry.abspath))
        if reader.is_encrypted:
            raise DocumentLoadError(f"{entry.relpath}: encrypted PDF")
        pages = [(page.extract_text() or "") for page in reader.pages]
    except DocumentLoadError:
        raise
    except (PdfReadError, ValueError, KeyError, OSError) as exc:
        raise DocumentLoadError(f"{entry.relpath}: unreadable PDF ({exc.__class__.__name__}: {exc})") from exc
    if not pages or not any(p.strip() for p in pages):
        raise DocumentLoadError(f"{entry.relpath}: no extractable text (image-only/scanned PDF needs OCR)")
    return RawDocument(entry, "pdf", pages, paged=True)


LOADERS = {".txt": load_text, ".md": load_markdown, ".markdown": load_markdown, ".pdf": load_pdf}


def load_document(entry: SourceEntry) -> RawDocument:
    loader = LOADERS.get(entry.extension)
    if loader is None:
        raise DocumentLoadError(f"{entry.relpath}: no loader for extension '{entry.extension}'")
    return loader(entry)


def humanize_stem(path: str) -> str:
    stem = Path(path).stem
    return re.sub(r"[_\-]+", " ", stem).strip() or stem
