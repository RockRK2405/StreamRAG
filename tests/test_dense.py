import numpy as np
import pytest

from streamrag.errors import ConfigError, ModelNotAvailableError
from streamrag.models.corpus import EmbeddingInfo
from streamrag.retrieval.dense import DenseIndex
from streamrag.retrieval.embedders import HashingEmbedder, OnnxEmbedder, load_embedder, load_registry
from streamrag.retrieval.text import Analyzer

from conftest import MODELS, REPO, requires_bge

A = Analyzer()
REGISTRY = REPO / "configs" / "models.yaml"


def test_hashing_embedder_is_deterministic_and_normalized():
    e = HashingEmbedder(A)
    v1, v2 = e.embed(["fog signal sounds"], "query"), e.embed(["fog signal sounds"], "query")
    assert np.array_equal(v1, v2) and abs(np.linalg.norm(v1[0]) - 1.0) < 1e-6


def test_dense_index_search_save_load_readonly(tmp_path):
    e = HashingEmbedder(A)
    docs = ["lamp cleaned at sunset", "fog signal in a storm", "wick trimming height"]
    idx = DenseIndex(e.embed(docs, "document"), e.info())
    assert idx.search(e.embed(["storm fog signal"], "query")[0], 1)[0][0] == 1
    assert not idx.matrix.flags.writeable
    idx.save(tmp_path / "d")
    again = DenseIndex.load(tmp_path / "d")
    assert np.array_equal(again.matrix, idx.matrix) and again.info == idx.info


def test_min_similarity_cutoff():
    e = HashingEmbedder(A)
    idx = DenseIndex(e.embed(["alpha beta", "gamma delta"], "document"), e.info())
    assert idx.search(e.embed(["alpha beta"], "query")[0], 2, min_similarity=0.99) == [(0, pytest.approx(1.0, abs=1e-5))]


def test_matrix_info_mismatch_rejected():
    from streamrag.errors import EmbeddingDimensionMismatchError
    with pytest.raises(EmbeddingDimensionMismatchError):
        DenseIndex(np.zeros((2, 8), dtype=np.float32), EmbeddingInfo(name="x", runtime="r", dimension=16))


def test_missing_model_and_unknown_model(tmp_path):
    with pytest.raises(ModelNotAvailableError):
        load_embedder("bge-small-en-v1.5", REGISTRY, tmp_path / "no-models", A)
    with pytest.raises(ConfigError):
        load_embedder("not-a-model", REGISTRY, MODELS, A)


@requires_bge
def test_onnx_bge_properties():
    e = load_embedder("bge-small-en-v1.5", REGISTRY, MODELS, A)
    assert e.dimension == 384 and e.pooling == "cls"
    texts = ["The meeting room seats twelve people.", "A small conference room for a dozen attendees.",
             "Bananas are rich in potassium."]
    v = e.embed(texts, "document")
    assert np.allclose(np.linalg.norm(v, axis=1), 1.0, atol=1e-5)
    assert np.array_equal(v, e.embed(texts, "document"))                      # deterministic
    single = np.vstack([e.embed([t], "document") for t in texts])
    assert np.allclose(single, v, atol=1e-5)                                   # batching does not change vectors
    assert float(v[0] @ v[1]) > float(v[0] @ v[2])                             # generic semantic sanity only
    assert e.count_truncated(["word " * 600]) == 1


@requires_bge
def test_pooling_cross_check(tmp_path):
    spec = {**load_registry(REGISTRY)["bge-small-en-v1.5"], "pooling": "mean"}
    with pytest.raises(ConfigError):
        OnnxEmbedder("bge-small-en-v1.5", MODELS / "bge-small-en-v1.5", spec)
