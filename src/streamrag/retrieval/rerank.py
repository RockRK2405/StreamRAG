"""Pluggable rerankers (ADR-003). The pipeline runs identically with ``NoopReranker`` (RRF + dedup order,
the official minimum) or a cross-encoder; switching is a config flag, which makes the ablation trivial."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np

from streamrag.errors import ConfigError, ModelNotAvailableError
from streamrag.retrieval.embedders import load_registry


class Reranker(Protocol):
    name: str

    def score(self, query: str, passages: list[str]) -> np.ndarray: ...


class NoopReranker:
    """Keeps the fused order (method ``rrf_dedup``)."""

    name = "none"

    def score(self, query: str, passages: list[str]) -> np.ndarray:  # pragma: no cover - never called
        raise NotImplementedError


class OnnxCrossEncoder:
    def __init__(self, name: str, model_dir: Path, spec: dict, intra_op_threads: int = 0, batch_size: int = 32) -> None:
        import onnxruntime as ort
        from tokenizers import Tokenizer

        self.name = name
        self.spec = spec
        onnx_path, tok_path = model_dir / spec["onnx_file"], model_dir / "tokenizer.json"
        missing = [str(p) for p in (onnx_path, tok_path) if not p.exists()]
        if missing:
            raise ModelNotAvailableError(f"reranker '{name}' missing files {missing}; run `streamrag fetch-models {name}`")
        self.tokenizer = Tokenizer.from_file(str(tok_path))
        self.tokenizer.enable_truncation(max_length=int(spec.get("max_length", 512)), strategy="only_second")
        pad_id = self.tokenizer.token_to_id("[PAD]")
        self.tokenizer.enable_padding(pad_id=0 if pad_id is None else pad_id, pad_token="[PAD]")
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if intra_op_threads > 0:
            so.intra_op_num_threads = intra_op_threads
        self.session = ort.InferenceSession(str(onnx_path), sess_options=so, providers=["CPUExecutionProvider"])
        self._inputs = {i.name for i in self.session.get_inputs()}
        self.batch_size = batch_size

    def score(self, query: str, passages: list[str]) -> np.ndarray:
        out = np.zeros(len(passages), dtype=np.float32)
        for s in range(0, len(passages), self.batch_size):
            batch = passages[s:s + self.batch_size]
            enc = self.tokenizer.encode_batch([(query, p) for p in batch])
            feeds = {"input_ids": np.array([e.ids for e in enc], dtype=np.int64),
                     "attention_mask": np.array([e.attention_mask for e in enc], dtype=np.int64)}
            if "token_type_ids" in self._inputs:
                feeds["token_type_ids"] = np.array([e.type_ids for e in enc], dtype=np.int64)
            out[s:s + len(batch)] = self.session.run(None, feeds)[0][:, 0]
        return out


def load_reranker(name: str, registry_path: Path, models_dir: Path, intra_op_threads: int = 0,
                  batch_size: int = 32) -> OnnxCrossEncoder:
    reg = load_registry(registry_path)
    spec = reg.get(name)
    if spec is None or spec["kind"] != "rerankers":
        raise ConfigError(f"unknown reranker '{name}'")
    return OnnxCrossEncoder(name, models_dir / name, spec, intra_op_threads=intra_op_threads, batch_size=batch_size)
