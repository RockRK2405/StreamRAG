"""Shared fixtures. All corpora under tests/fixtures are TEST FIXTURES (synthetic, fictional)."""

from __future__ import annotations

from pathlib import Path

import pytest

from streamrag.config import load_config
from streamrag.retrieval import build_index

REPO = Path(__file__).resolve().parents[1]
FIX = REPO / "tests" / "fixtures"
MODELS = REPO / "models"


def model_available(name: str) -> bool:
    return (MODELS / name / "MODEL_INFO.json").exists()


requires_bge = pytest.mark.skipif(not model_available("bge-small-en-v1.5"), reason="bge-small model not downloaded")
requires_ce = pytest.mark.skipif(not model_available("ms-marco-minilm-l6-v2"), reason="cross-encoder not downloaded")


def make_cfg(tmp: Path, corpus: str | Path = FIX / "corpus", embedder: str = "hashing", **overrides):
    ov = {"paths.corpus": str(corpus), "paths.index_root": str(tmp / "indexes"), "paths.runs_dir": str(tmp / "runs"),
          "dense.embedder": embedder, "telemetry.log_level": "WARNING"}
    ov.update(overrides)
    return load_config(REPO / "configs" / "default.yaml", ov, base_dir=REPO)


@pytest.fixture(scope="session")
def fixture_bundle(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("bundle")
    cfg = make_cfg(tmp)
    return cfg, build_index(cfg)


@pytest.fixture()
def cfg_factory(tmp_path):
    def factory(**kw):
        return make_cfg(tmp_path, **kw)
    return factory


# ---------------------------------------------------------------- Phase 9 adaptive retrieval
ADAPTIVE_CORPUS = FIX / "corpus_adaptive"
REFERENCE_DATE = "2026-10-03"


@pytest.fixture(scope="session")
def adaptive_env(tmp_path_factory):
    """(cfg, service) over the Phase 9 fixture corpus with the deterministic hashing embedder."""
    from streamrag.retrieval import RetrievalService
    tmp = tmp_path_factory.mktemp("adaptive")
    cfg = make_cfg(tmp, corpus=ADAPTIVE_CORPUS, **{"adaptive_retrieval.enabled": True,
                                                    "adaptive_retrieval.reference_date": REFERENCE_DATE})
    b = build_index(cfg)
    svc = RetrievalService.from_config(cfg, index_path=b.path)
    yield cfg, svc
    svc.close()
