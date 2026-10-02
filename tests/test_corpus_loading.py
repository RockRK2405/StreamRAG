import shutil

import pytest

from streamrag.corpus import CorpusSource, build_corpus
from streamrag.corpus.loaders import load_document
from streamrag.errors import CorpusNotFoundError, DocumentLoadError, EmptyCorpusError

from conftest import FIX


def _src(path):
    return CorpusSource(path, [".txt", ".md", ".pdf"], "TEST_FIXTURE_ONLY")


def test_scan_is_sorted_filtered_and_marks_fixture():
    src = _src(FIX / "corpus")
    rels = [e.relpath for e in src.entries()]
    assert rels == sorted(rels) and all(r.endswith((".txt", ".md")) for r in rels)
    assert src.is_test_fixture() and src.status() == "TEST_FIXTURE"


def test_missing_and_empty_corpus(tmp_path):
    with pytest.raises(CorpusNotFoundError):
        _src(tmp_path / "missing").entries()
    assert _src(tmp_path / "missing").status() == "NOT_AVAILABLE"
    (tmp_path / "c").mkdir()
    (tmp_path / "c" / "notes.xyz").write_text("x")
    (tmp_path / "c" / ".hidden.txt").write_text("hidden")
    with pytest.raises(EmptyCorpusError):
        _src(tmp_path / "c").entries()


def test_corpus_hash_tracks_content(tmp_path):
    d = tmp_path / "c"
    shutil.copytree(FIX / "corpus", d)
    h1 = _src(d).corpus_hash()
    (d / "fixture_observatory_notes.txt").write_text("changed text")
    assert _src(d).corpus_hash() != h1


def test_unmarked_directory_is_not_a_fixture(tmp_path):
    d = tmp_path / "c"
    d.mkdir()
    (d / "a.txt").write_text("A document about nothing in particular.")
    assert _src(d).status() == "AVAILABLE" and not _src(d).is_test_fixture()


def test_markdown_front_matter_and_title(cfg_factory):
    built = build_corpus(cfg_factory())
    doc = next(d for d in built.documents if d.source_path.endswith(".md"))
    assert doc.title == "Fixture Lighthouse Keeping Manual"
    assert "---" not in doc.text and "title:" not in doc.text


def test_invalid_documents_skipped_and_recorded(cfg_factory):
    built = build_corpus(cfg_factory(corpus=FIX / "corpus_errors"))
    status = {f.path: (f.status, f.reason) for f in built.source_files}
    assert status["empty.txt"][0] == "skipped" and "empty" in status["empty.txt"][1]
    assert status["whitespace_only.md"][0] == "skipped"
    assert "ignored.xyz" not in status                          # unsupported extension never enters
    cafe = next(d for d in built.documents if d.source_path == "cp1252_cafe.txt")
    assert "café" in cafe.text and any("cp1252" in m for m in cafe.normalization_log)


def test_invalid_document_policy_error(cfg_factory):
    with pytest.raises(DocumentLoadError):
        build_corpus(cfg_factory(corpus=FIX / "corpus_errors", **{"corpus.on_invalid_document": "error"}))


def test_pdf_loader_extracts_pages():
    entry = next(e for e in _src(FIX / "corpus_pdf").entries() if e.relpath.endswith(".pdf"))
    raw = load_document(entry)
    assert raw.paged and len(raw.pages) == 3 and all(p.strip() for p in raw.pages)
