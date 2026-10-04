"""Writes experiments/results/environment.json (brief §59-60): Python and package versions, model versions,
LLM server and model digest, hardware, OS, configuration hash, git commit.

Usage: .venv/bin/python experiments/runners/environment.py   (run while `ollama serve` is up to record the LLM)
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from importlib import metadata
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

PKGS = ["numpy", "scipy", "onnxruntime", "tokenizers", "pydantic", "pyyaml", "snowballstemmer", "pypdf", "pytest",
        "huggingface-hub"]


def sh(cmd: list[str]) -> str | None:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=10).stdout.strip() or None
    except (OSError, subprocess.TimeoutExpired):
        return None


def main() -> None:
    from run_experiments import llm_info
    from streamrag.config import load_config
    from streamrag.config.settings import config_hash
    from streamrag.evaluation.runner import git_commit
    pk = {}
    for p in PKGS:
        try:
            pk[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            pk[p] = None
    models = {}
    for d in sorted((REPO / "models").iterdir()):
        info = d / "MODEL_INFO.json"
        if info.exists():
            m = json.loads(info.read_text())
            files = m.get("files") or {}
            models[d.name] = {**{k: v for k, v in m.items() if k != "files"},
                              "weights_sha256": {f: x.get("sha256") for f, x in files.items()
                                                 if f.endswith((".onnx", ".bin", ".safetensors"))}}
    cfg = load_config(REPO / "configs" / "default.yaml", {}, base_dir=REPO)
    env = {
        "python": {"version": platform.python_version(), "implementation": platform.python_implementation(),
                   "executable": sys.executable},
        "packages": pk,
        "models": models,
        "llm": llm_info("http://127.0.0.1:11434", "qwen3:4b"),
        "database": "none - in-process BM25 (scipy sparse) and exact dense search (numpy) over the index files",
        "hardware": {"machine": platform.machine(), "cpu": sh(["sysctl", "-n", "machdep.cpu.brand_string"]),
                     "cpu_count": os.cpu_count(), "memory_bytes": sh(["sysctl", "-n", "hw.memsize"]),
                     "model": sh(["sysctl", "-n", "hw.model"])},
        "os": {"platform": platform.platform(), "release": platform.release()},
        "configuration": {"file": "configs/default.yaml", "config_hash": config_hash(cfg),
                          "experiments": "experiments/configs/{systems,experiments}.yaml"},
        "git": git_commit(REPO),
        "dataset": json.loads((REPO / "experiments" / "datasets" / "streamrag_eval_v1" / "manifest.json").read_text()),
    }
    out = REPO / "experiments" / "results" / "environment.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(env, indent=2, default=str))
    print(json.dumps({k: env[k] for k in ("python", "hardware", "os", "git")}, indent=2))


if __name__ == "__main__":
    main()
