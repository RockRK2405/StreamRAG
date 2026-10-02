"""Manifest completeness + index integrity / error handling."""

import json
import shutil

import numpy as np
import pytest

from streamrag.corpus.manifest import manifest_content_hash
from streamrag.errors import (
    EmbeddingDimensionMismatchError,
    IndexCorruptError,
    IndexNotFoundError,
    IndexVersionMismatchError,
)
from streamrag.retrieval import RetrievalService, build_index, load_index, resolve_index
from streamrag.retrieval.embedders import HashingEmbedder


def test_manifest_is_complete_and_fixture_labeled(fixture_bundle):
    _, b = fixture_bundle
    m = b.manifest
    assert m.is_test_fixture is True
    assert (m.document_count, m.chunk_count) == (3, len(b.chunks)) and m.section_count >= 10
    for key in ("strategy", "target_tokens", "max_tokens", "overlap_tokens"):
        assert key in m.chunking_configuration
    assert m.bm25_configuration["k1"] == 1.5 and m.normalization_configuration["unicode_form"] == "NFKC"
    assert m.embedding_model.name == "hashing" and m.embedding_model.dimension == 256
    assert {"chunks.jsonl", "documents.jsonl", "bm25/weights.npz", "dense/embeddings.npy"} <= set(m.artifacts)
    assert m.environment["python"] and m.content_hash == manifest_content_hash(m)
    text = (b.path / "manifest.json").read_text().lower()
    assert "gold" not in text and "expected" not in text          # never benchmark answers


def test_existing_index_is_reused_unless_forced(fixture_bundle):
    cfg, b = fixture_bundle
    again = build_index(cfg)
    assert again.path == b.path and again.timings_ms == {"reused": 1.0}
    assert resolve_index(cfg) == b.path


def _copy(b, tmp_path):
    dst = tmp_path / "copy"
    shutil.copytree(b.path, dst)
    return dst


def test_missing_index(tmp_path, cfg_factory):
    with pytest.raises(IndexNotFoundError):
        load_index(tmp_path / "nothing")
    with pytest.raises(IndexNotFoundError):
        resolve_index(cfg_factory())                                # nothing built in this tmp index_root


def test_corrupt_artifact_detected(fixture_bundle, tmp_path):
    p = _copy(fixture_bundle[1], tmp_path)
    with (p / "chunks.jsonl").open("a") as f:
        f.write("\n")
    with pytest.raises(IndexCorruptError):
        load_index(p)


def test_missing_artifact_detected(fixture_bundle, tmp_path):
    p = _copy(fixture_bundle[1], tmp_path)
    (p / "bm25" / "vocab.json").unlink()
    with pytest.raises(IndexCorruptError):
        load_index(p)


def test_index_version_mismatch(fixture_bundle, tmp_path):
    p = _copy(fixture_bundle[1], tmp_path)
    m = json.loads((p / "manifest.json").read_text())
    m["index_version"] = "0.1"
    (p / "manifest.json").write_text(json.dumps(m))
    with pytest.raises(IndexVersionMismatchError):
        load_index(p)


def test_embedding_dimension_mismatch(fixture_bundle):
    cfg, b = fixture_bundle
    with pytest.raises(EmbeddingDimensionMismatchError):
        RetrievalService(b, cfg, embedder=HashingEmbedder(b.analyzer, dim=64))
    with pytest.raises(EmbeddingDimensionMismatchError):
        b.dense.search(np.ones(64, dtype=np.float32), 3)


def test_documents_persisted_for_traceability(fixture_bundle):
    _, b = fixture_bundle
    docs = {d.document_id: d for d in b.documents()}
    for c in b.chunks:
        assert docs[c.document_id].text[c.char_start:c.char_end] == c.text
