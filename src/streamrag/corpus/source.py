"""CorpusSource: the only way documents enter the system (corpus isolation).

A corpus is a directory. Files are discovered deterministically (sorted POSIX relative paths), hidden
files are ignored, and the corpus hash covers every file's bytes, so any change is detected.
A directory containing the fixture marker file is a TEST FIXTURE and is labeled as such everywhere.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from streamrag.errors import CorpusNotFoundError, EmptyCorpusError


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


@dataclass(frozen=True)
class SourceEntry:
    relpath: str
    abspath: Path
    sha256: str
    size: int
    extension: str


class CorpusSource:
    def __init__(self, root: Path, include_extensions: list[str], fixture_marker: str) -> None:
        self.root = Path(root)
        self.include = {e.lower() for e in include_extensions}
        self.fixture_marker = fixture_marker

    def exists(self) -> bool:
        return self.root.is_dir()

    def is_test_fixture(self) -> bool:
        return (self.root / self.fixture_marker).exists()

    def status(self) -> str:
        """AVAILABLE / NOT_AVAILABLE / TEST_FIXTURE (never claims a fixture is the official corpus)."""
        if not self.exists():
            return "NOT_AVAILABLE"
        if self.is_test_fixture():
            return "TEST_FIXTURE"
        try:
            return "AVAILABLE" if self.entries() else "NOT_AVAILABLE"
        except EmptyCorpusError:
            return "NOT_AVAILABLE"

    def entries(self) -> list[SourceEntry]:
        if not self.exists():
            raise CorpusNotFoundError(f"corpus directory not found: {self.root}")
        out = []
        for p in sorted(self.root.rglob("*"), key=lambda q: q.relative_to(self.root).as_posix()):
            rel = p.relative_to(self.root)
            if not p.is_file() or any(part.startswith(".") for part in rel.parts):
                continue
            ext = p.suffix.lower()
            if ext not in self.include:
                continue
            out.append(SourceEntry(rel.as_posix(), p, sha256_file(p), p.stat().st_size, ext))
        if not out:
            raise EmptyCorpusError(f"no documents with extensions {sorted(self.include)} under {self.root}")
        return out

    def corpus_hash(self, entries: list[SourceEntry] | None = None) -> str:
        entries = entries if entries is not None else self.entries()
        h = hashlib.sha256()
        h.update(b"fixture=1\n" if self.is_test_fixture() else b"fixture=0\n")
        for e in entries:
            h.update(f"{e.relpath}\t{e.sha256}\n".encode())
        return h.hexdigest()
