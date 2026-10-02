"""Same corpus + config + model + code => identical index artifacts."""

import json

import pytest

from streamrag.retrieval import build_index

from conftest import make_cfg, requires_bge

HASHED_ARTIFACTS = ["chunks.jsonl", "documents.jsonl", "bm25/weights.npz", "bm25/vocab.json", "dense/embeddings.npy"]


def _build(tmp, name, embedder):
    return build_index(make_cfg(tmp / name, embedder=embedder))


@pytest.mark.parametrize("embedder", ["hashing", pytest.param("bge-small-en-v1.5", marks=requires_bge)])
def test_two_builds_identical(tmp_path, embedder):
    a, b = _build(tmp_path, "a", embedder), _build(tmp_path, "b", embedder)
    assert a.manifest.content_hash == b.manifest.content_hash
    assert a.manifest.generated_at  # informational field exists but is excluded from the content hash
    for art in HASHED_ARTIFACTS:
        assert a.manifest.artifacts[art] == b.manifest.artifacts[art], art
    ma = json.loads((a.path / "manifest.json").read_text())
    assert ma["environment"]["packages"]["numpy"] and ma["index_config_hash"] == b.manifest.index_config_hash
