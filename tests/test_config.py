import pytest

from streamrag.config import config_hash, index_config_hash, load_config
from streamrag.errors import ConfigError

from conftest import REPO

DEFAULT = REPO / "configs" / "default.yaml"


def test_default_config_loads_and_resolves_paths():
    cfg = load_config(DEFAULT)
    assert cfg.paths.corpus.is_absolute() and cfg.paths.corpus == (REPO / "corpus").resolve()
    assert cfg.retrieval.mode == "hybrid" and cfg.rerank.enabled is False


def test_unknown_key_rejected():
    with pytest.raises(ConfigError):
        load_config(DEFAULT, {"retrieval.topk": 5})


def test_invalid_value_rejected():
    with pytest.raises(ConfigError):
        load_config(DEFAULT, {"chunking.target_tokens": 400, "chunking.max_tokens": 300})


def test_missing_file():
    with pytest.raises(ConfigError):
        load_config(REPO / "configs" / "nope.yaml")


def test_env_corpus_override(monkeypatch, tmp_path):
    monkeypatch.setenv("STREAMRAG_CORPUS", str(tmp_path))
    assert load_config(DEFAULT).paths.corpus == tmp_path.resolve()


def test_hashes_are_stable_and_scoped():
    a, b = load_config(DEFAULT), load_config(DEFAULT)
    assert config_hash(a) == config_hash(b) and index_config_hash(a) == index_config_hash(b)
    c = load_config(DEFAULT, {"retrieval.top_k": 3})
    assert config_hash(c) != config_hash(a)
    assert index_config_hash(c) == index_config_hash(a)          # query-time setting: no rebuild
    d = load_config(DEFAULT, {"chunking.target_tokens": 120})
    assert index_config_hash(d) != index_config_hash(a)          # index-time setting: rebuild


def test_environment_settings_override_yaml(monkeypatch, tmp_path):
    from streamrag.config.settings import ENV_OVERRIDES
    monkeypatch.setenv("STREAMRAG_LLM_MODEL", "some-model")
    monkeypatch.setenv("STREAMRAG_LLM_BACKEND", "extractive")
    monkeypatch.setenv("STREAMRAG_ALLOWED_LLM_HOSTS", "ollama, other")
    cfg = load_config(REPO / "configs" / "default.yaml", base_dir=REPO)
    assert cfg.generation.model == "some-model" and cfg.generation.backend == "extractive"
    assert cfg.generation.allowed_llm_hosts == ["ollama", "other"]
    assert set(ENV_OVERRIDES) >= {"STREAMRAG_CORPUS", "STREAMRAG_LLM_URL"}


def test_llm_url_must_be_loopback_unless_explicitly_allowed():
    import pytest
    from streamrag.generation.llm import check_loopback
    check_loopback("http://127.0.0.1:11434")
    with pytest.raises(ValueError):
        check_loopback("http://ollama:11434")
    check_loopback("http://ollama:11434", ["ollama"])
    with pytest.raises(ValueError):
        check_loopback("http://example.com:11434", ["ollama"])


def test_final_profile_enables_the_frozen_pipeline_and_env_wins(monkeypatch):
    from streamrag.config import load_final_config
    cfg = load_final_config(REPO)
    assert cfg.multi_intent.enabled and cfg.session.enabled and cfg.generation.enabled
    assert cfg.adaptive_retrieval.enabled and not cfg.generation.answerability and not cfg.streaming.rerank
    monkeypatch.setenv("STREAMRAG_LLM_BACKEND", "extractive")
    assert load_final_config(REPO).generation.backend == "extractive"
    assert load_final_config(REPO, {"generation.backend": "auto"}).generation.backend == "auto"
