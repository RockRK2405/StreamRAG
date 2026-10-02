"""Download pinned model files listed in configs/models.yaml into ./models/<name>/.

Build-time only. Each model directory gets a ``MODEL_INFO.json`` recording repo, revision and
file hashes so that run manifests can prove which weights were used.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_registry(path: Path) -> dict[str, dict]:
    raw = yaml.safe_load(path.read_text())
    out: dict[str, dict] = {}
    for kind in ("embedders", "rerankers"):
        for name, spec in (raw.get(kind) or {}).items():
            out[name] = {**spec, "kind": kind[:-1]}
    return out


def fetch(names: list[str], registry_path: Path, models_dir: Path) -> list[Path]:
    from huggingface_hub import hf_hub_download  # optional dependency: streamrag[models]

    registry = load_registry(registry_path)
    fetched = []
    for name in names:
        if name not in registry:
            raise KeyError(f"unknown model '{name}'; known: {sorted(registry)}")
        spec = registry[name]
        target = models_dir / name
        target.mkdir(parents=True, exist_ok=True)
        files = {}
        for rel in spec["files"]:
            local = hf_hub_download(repo_id=spec["repo"], filename=rel, revision=spec["revision"], local_dir=target)
            files[rel] = {"sha256": _sha256(Path(local)), "bytes": Path(local).stat().st_size}
        info = {"name": name, "repo": spec["repo"], "revision": spec["revision"], "license": spec.get("license"),
                "kind": spec["kind"], "files": files}
        (target / "MODEL_INFO.json").write_text(json.dumps(info, indent=2, sort_keys=True))
        fetched.append(target)
    return fetched
