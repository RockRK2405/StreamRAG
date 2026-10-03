"""Embedding runtimes.

* ``OnnxEmbedder`` — production path: HF ``tokenizers`` + ONNX Runtime (CPU EP only), pooling read from the
  model's own ``1_Pooling/config.json`` and cross-checked against the pinned registry (ADR-002).
* ``HashingEmbedder`` — **TEST/OFFLINE ONLY**. Deterministic feature hashing of analyzer tokens. It is NOT a
  semantic model; it exists so unit tests and keyless/offline smoke runs can exercise the dense code path.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal, Protocol

import numpy as np
import yaml

from streamrag.errors import ConfigError, ModelNotAvailableError
from streamrag.models.corpus import EmbeddingInfo
from streamrag.retrieval.text import Analyzer

Kind = Literal["query", "document"]


class Embedder(Protocol):
    name: str

    @property
    def dimension(self) -> int: ...

    def embed(self, texts: list[str], kind: Kind) -> np.ndarray: ...

    def info(self) -> EmbeddingInfo: ...


def load_registry(path: Path) -> dict[str, dict]:
    if not path.exists():
        raise ConfigError(f"model registry not found: {path}")
    raw = yaml.safe_load(path.read_text()) or {}
    reg = {}
    for kind in ("embedders", "rerankers", "nli"):
        for name, spec in (raw.get(kind) or {}).items():
            reg[name] = {**spec, "kind": kind}
    return reg


def _model_files_sha(model_dir: Path) -> dict[str, str]:
    info = model_dir / "MODEL_INFO.json"
    if not info.exists():
        return {}
    return {k: v["sha256"] for k, v in json.loads(info.read_text()).get("files", {}).items()}


class OnnxEmbedder:
    def __init__(self, name: str, model_dir: Path, spec: dict, intra_op_threads: int = 0, batch_size: int = 32) -> None:
        import onnxruntime as ort
        from tokenizers import Tokenizer

        self.name = name
        self.spec = spec
        self.model_dir = model_dir
        onnx_path = model_dir / spec["onnx_file"]
        tok_path = model_dir / "tokenizer.json"
        missing = [str(p) for p in (onnx_path, tok_path) if not p.exists()]
        if missing:
            raise ModelNotAvailableError(f"embedding model '{name}' missing files {missing}; "
                                         f"run `streamrag fetch-models {name}`")
        pooling_cfg = model_dir / "1_Pooling" / "config.json"
        if pooling_cfg.exists():
            pc = json.loads(pooling_cfg.read_text())
            pooling = "cls" if pc.get("pooling_mode_cls_token") else "mean" if pc.get("pooling_mode_mean_tokens") else None
            if pooling and pooling != spec["pooling"]:
                raise ConfigError(f"{name}: registry pooling '{spec['pooling']}' != model pooling '{pooling}'")
        self.pooling = spec["pooling"]
        self.max_length = int(spec["max_length"])
        self.normalize = bool(spec.get("normalize", True))
        self.query_prefix = spec.get("query_prefix", "") or ""
        self.document_prefix = spec.get("document_prefix", "") or ""
        self.batch_size = batch_size

        self.tokenizer = Tokenizer.from_file(str(tok_path))
        self.tokenizer.enable_truncation(max_length=self.max_length)
        pad_id = self.tokenizer.token_to_id("[PAD]")
        self.tokenizer.enable_padding(pad_id=0 if pad_id is None else pad_id, pad_token="[PAD]")
        self._counter = Tokenizer.from_file(str(tok_path))   # no truncation: used to count truncation
        self._counter.no_truncation()
        self._counter.no_padding()

        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if intra_op_threads > 0:
            so.intra_op_num_threads = intra_op_threads
        self.session = ort.InferenceSession(str(onnx_path), sess_options=so, providers=["CPUExecutionProvider"])
        self._input_names = {i.name for i in self.session.get_inputs()}
        self._dim = int(self.session.get_outputs()[0].shape[-1])

    @property
    def dimension(self) -> int:
        return self._dim

    def _run(self, texts: list[str]) -> np.ndarray:
        enc = self.tokenizer.encode_batch(texts)
        ids = np.array([e.ids for e in enc], dtype=np.int64)
        mask = np.array([e.attention_mask for e in enc], dtype=np.int64)
        feeds = {"input_ids": ids, "attention_mask": mask}
        if "token_type_ids" in self._input_names:
            feeds["token_type_ids"] = np.array([e.type_ids for e in enc], dtype=np.int64)
        hidden = self.session.run(None, feeds)[0]
        if self.pooling == "cls":
            vec = hidden[:, 0, :]
        else:
            m = mask[..., None].astype(np.float32)
            vec = (hidden * m).sum(axis=1) / np.clip(m.sum(axis=1), 1e-9, None)
        vec = vec.astype(np.float32)
        if self.normalize:
            vec /= np.clip(np.linalg.norm(vec, axis=1, keepdims=True), 1e-12, None)
        return vec

    def embed(self, texts: list[str], kind: Kind) -> np.ndarray:
        if not texts:
            return np.zeros((0, self._dim), dtype=np.float32)
        prefix = self.query_prefix if kind == "query" else self.document_prefix
        inputs = [prefix + t for t in texts]
        order = sorted(range(len(inputs)), key=lambda i: (-len(inputs[i]), i))   # length-bucketed batches
        out = np.zeros((len(inputs), self._dim), dtype=np.float32)
        for s in range(0, len(order), self.batch_size):
            idx = order[s:s + self.batch_size]
            out[idx] = self._run([inputs[i] for i in idx])
        return out

    def count_truncated(self, texts: list[str], kind: Kind = "document") -> int:
        prefix = self.query_prefix if kind == "query" else self.document_prefix
        return sum(1 for e in self._counter.encode_batch([prefix + t for t in texts]) if len(e.ids) > self.max_length)

    def info(self, truncated: int = 0) -> EmbeddingInfo:
        return EmbeddingInfo(name=self.name, repo=self.spec.get("repo"), revision=self.spec.get("revision"),
                             runtime="onnxruntime-cpu", dimension=self._dim, pooling=self.pooling,
                             normalize=self.normalize, max_length=self.max_length,
                             model_files_sha256=_model_files_sha(self.model_dir), truncated_chunks=truncated)


class HashingEmbedder:
    """TEST/OFFLINE ONLY — not semantic. Deterministic signed feature hashing of stemmed tokens + bigrams."""

    name = "hashing"

    def __init__(self, analyzer: Analyzer, dim: int = 256) -> None:
        self.analyzer = analyzer
        self._dim = dim

    @property
    def dimension(self) -> int:
        return self._dim

    def _bucket(self, feat: str) -> tuple[int, float]:
        h = int.from_bytes(hashlib.blake2b(feat.encode(), digest_size=8).digest(), "little")
        return h % self._dim, (1.0 if (h >> 63) & 1 else -1.0)

    def embed(self, texts: list[str], kind: Kind) -> np.ndarray:
        out = np.zeros((len(texts), self._dim), dtype=np.float32)
        for r, t in enumerate(texts):
            toks = self.analyzer.tokens(t)
            for feat in toks + [f"{a}_{b}" for a, b in zip(toks, toks[1:])]:
                j, sgn = self._bucket(feat)
                out[r, j] += sgn
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.clip(norms, 1e-12, None)

    def count_truncated(self, texts: list[str], kind: Kind = "document") -> int:
        return 0

    def info(self, truncated: int = 0) -> EmbeddingInfo:
        return EmbeddingInfo(name="hashing", runtime="hashing-test-only", dimension=self._dim, pooling=None,
                             normalize=True, max_length=None, truncated_chunks=0)


def load_embedder(name: str, registry_path: Path, models_dir: Path, analyzer: Analyzer,
                  intra_op_threads: int = 0, batch_size: int = 32) -> Embedder:
    if name == "hashing":
        return HashingEmbedder(analyzer)
    reg = load_registry(registry_path)
    spec = reg.get(name)
    if spec is None or spec["kind"] != "embedders":
        raise ConfigError(f"unknown embedder '{name}'; registry has {sorted(k for k, v in reg.items() if v['kind'] == 'embedders')}")
    return OnnxEmbedder(name, models_dir / name, spec, intra_op_threads=intra_op_threads, batch_size=batch_size)
