"""Corpus isolation: retrieval can only search the configured, indexed corpus; no network at query time."""

import ast
import socket

import pytest

from streamrag.retrieval import RetrievalService, build_index
from streamrag.retrieval.embedders import HashingEmbedder

from conftest import REPO, make_cfg

PKG = REPO / "src" / "streamrag"
NETWORK_MODULES = {"socket", "urllib", "http", "requests", "httpx", "aiohttp", "huggingface_hub", "ftplib",
                   "smtplib", "websocket", "websockets", "streamrag.tools"}
ISOLATED_PACKAGES = ["corpus", "retrieval", "models", "config", "telemetry", "bench", "streaming", "controller",
                     "ledger", "replay", "intents", "multi_retrieval", "fusion"]


def _imports(path):
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            yield from (a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            yield node.module


@pytest.mark.parametrize("pkg", ISOLATED_PACKAGES)
def test_no_network_imports(pkg):
    for f in (PKG / pkg).rglob("*.py"):
        for mod in _imports(f):
            assert not any(mod == n or mod.startswith(n + ".") for n in NETWORK_MODULES), f"{f}: imports {mod}"


def test_build_and_retrieve_with_network_blocked(tmp_path, monkeypatch):
    def refuse(*a, **k):
        raise AssertionError("network access attempted")
    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    cfg = make_cfg(tmp_path)
    b = build_index(cfg)
    s = RetrievalService(b, cfg, embedder=HashingEmbedder(b.analyzer))
    es = s.retrieve("fog signal")
    assert es.items and set(es.chunk_ids()) <= {c.chunk_id for c in b.chunks}
    s.close()


def test_no_document_injection_api():
    public = {n for n in dir(RetrievalService) if not n.startswith("_")}
    assert public == {"retrieve", "retrieve_batch", "from_config", "close", "to_retrieval_result"}
