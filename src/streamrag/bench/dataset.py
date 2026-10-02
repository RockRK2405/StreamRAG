"""Evaluation data loading: a JSONL of RetrievalEvalItem, or a directory of BenchmarkCase JSON files
(flattened to one retrieval item per answerable gold intent -> per-intent Recall@k)."""

from __future__ import annotations

import hashlib
from pathlib import Path

from streamrag.models.benchmark import BenchmarkCase, RetrievalEvalItem


def dataset_sha256(path: Path) -> str:
    h = hashlib.sha256()
    files = sorted(p for p in path.rglob("*") if p.is_file()) if path.is_dir() else [path]
    for f in files:
        h.update(f.name.encode())
        h.update(f.read_bytes())
    return h.hexdigest()


def load_eval_items(path: Path) -> list[RetrievalEvalItem]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"evaluation data not found: {path}")
    items: list[RetrievalEvalItem] = []
    if path.is_dir():
        for f in sorted(path.rglob("*.json")):
            items.extend(BenchmarkCase.model_validate_json(f.read_text()).retrieval_items())
    else:
        for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if line.strip():
                try:
                    items.append(RetrievalEvalItem.model_validate_json(line))
                except Exception as exc:
                    raise ValueError(f"{path}:{n}: invalid evaluation item: {exc}") from exc
    ids = [i.item_id for i in items]
    if len(set(ids)) != len(ids):
        raise ValueError(f"duplicate item_id in {path}")
    if not items:
        raise ValueError(f"no evaluation items in {path}")
    return items
