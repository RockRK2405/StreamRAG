"""Entailment model wrapper (ADR-008 level L3; docs/answer/04).

An ONNX cross-encoder NLI model (default ``nli-deberta-v3-xsmall``, pinned in configs/models.yaml) scores
(premise, hypothesis) pairs as contradiction / entailment / neutral. Labels come from the model's config.json.
Results are cached per pair (deterministic CPU inference), so re-verifying an unchanged claim costs nothing.

The probabilities are *not calibrated*; callers use the argmax label only and keep the probabilities as raw
signals. Limitations (documented, tested): single-sentence premises work best; multi-sentence reasoning,
world knowledge and long-range coreference are weak; numbers are additionally checked by rules.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class NliModel:
    def __init__(self, model_dir: Path, max_length: int = 512, batch_size: int = 16) -> None:
        import onnxruntime as ort
        from tokenizers import Tokenizer
        info = json.loads((model_dir / "config.json").read_text())
        self.labels = {int(k): v.lower() for k, v in info["id2label"].items()}
        self.name = model_dir.name
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 4
        self.session = ort.InferenceSession(str(model_dir / "onnx" / "model.onnx"), opts,
                                            providers=["CPUExecutionProvider"])
        self.inputs = {i.name for i in self.session.get_inputs()}
        self.tok = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        self.tok.enable_truncation(max_length=max_length)
        self.batch_size = batch_size
        self.cache: dict[tuple[str, str], dict[str, float]] = {}
        self.calls = 0                                    # model forward passes (pairs actually scored)

    @classmethod
    def load(cls, models_dir: Path, name: str) -> "NliModel":
        d = Path(models_dir) / name
        if not (d / "onnx" / "model.onnx").exists():
            raise FileNotFoundError(f"NLI model '{name}' not found in {models_dir}; run `streamrag fetch-models {name}`")
        return cls(d)

    def predict(self, pairs: list[tuple[str, str]]) -> list[dict[str, float]]:
        """(premise, hypothesis) -> {"entailment": p, "neutral": p, "contradiction": p, "label": ...}."""
        todo = [p for p in dict.fromkeys(pairs) if p not in self.cache]
        for i in range(0, len(todo), self.batch_size):
            batch = todo[i:i + self.batch_size]
            enc = self.tok.encode_batch(batch)
            n = max(len(e.ids) for e in enc)
            feed = {"input_ids": np.array([e.ids + [0] * (n - len(e.ids)) for e in enc], dtype=np.int64),
                    "attention_mask": np.array([e.attention_mask + [0] * (n - len(e.ids)) for e in enc],
                                               dtype=np.int64)}
            if "token_type_ids" in self.inputs:
                feed["token_type_ids"] = np.array([e.type_ids + [0] * (n - len(e.ids)) for e in enc], dtype=np.int64)
            logits = self.session.run(None, feed)[0]
            z = np.exp(logits - logits.max(1, keepdims=True))
            probs = z / z.sum(1, keepdims=True)
            self.calls += len(batch)
            for pair, row in zip(batch, probs):
                d = {self.labels[j]: round(float(row[j]), 4) for j in range(len(row))}
                self.cache[pair] = {**d, "label": self.labels[int(row.argmax())]}
        return [self.cache[p] for p in pairs]
