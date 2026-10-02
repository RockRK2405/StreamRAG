import pytest

from streamrag.corpus import build_corpus
from streamrag.corpus.ids import assign_document_ids, render_chunk_id, render_citation
from streamrag.errors import CorpusIntegrityError

PAT = r"^(?i:doc)[ _-]?(\d+)"


def test_native_ids_from_filename_and_front_matter():
    ids = assign_document_ids(["Doc_07_orchard.txt", "x/notes.md", "y.md"], [{}, {"id": "Doc_31"}, {}],
                              "native_or_stem", PAT)
    assert ids == ["Doc_07", "Doc_31", "y"]


def test_stem_strategy_is_path_based_and_collision_safe():
    ids = assign_document_ids(["a-b.txt", "a_b.md", "sub/a-b.txt"], [{}, {}, {}], "native_or_stem", PAT)
    assert ids == ["a_b", "a_b_2", "sub_a_b"]


def test_ordinal_strategy_skips_native_numbers():
    ids = assign_document_ids(["a.txt", "doc_2.txt", "b.txt"], [{}, {}, {}], "native_or_ordinal", PAT)
    assert ids == ["Doc_1", "Doc_2", "Doc_3"]


def test_duplicate_native_ids_fail_fast():
    with pytest.raises(CorpusIntegrityError):
        assign_document_ids(["Doc_1_a.txt", "doc-1-b.txt"], [{}, {}], "native_or_stem", PAT)


def test_templates():
    assert render_chunk_id("{document_id}§{section_id}#{part}", "Doc_07", "2.1", 3) == "Doc_07§2.1#3"
    assert render_citation("{document_id} §{section_id}", "Doc_07", "2.1") == "Doc_07 §2.1"


def test_ids_stable_across_builds_and_unique(cfg_factory):
    cfg = cfg_factory()
    a, b = build_corpus(cfg), build_corpus(cfg)
    ids_a = [c.chunk_id for c in a.chunks]
    assert ids_a == [c.chunk_id for c in b.chunks] and len(set(ids_a)) == len(ids_a)
    assert "Doc_07§2.2#1" in ids_a and "fixture_lighthouse_manual§1.1#1" in ids_a
    assert all(c.citation == f"{c.document_id} §{c.section_id}" for c in a.chunks)
