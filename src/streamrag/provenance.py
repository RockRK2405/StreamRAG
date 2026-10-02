"""Reproducibility metadata: interpreter, platform, dependency versions, code version."""

from __future__ import annotations

import platform
import subprocess
import sys
from importlib import metadata
from pathlib import Path

import streamrag

KEY_PACKAGES = ["streamrag", "numpy", "scipy", "pydantic", "pyyaml", "snowballstemmer", "tokenizers",
                "onnxruntime", "pypdf", "huggingface_hub", "torch", "sentence-transformers"]


def package_versions() -> dict[str, str | None]:
    out: dict[str, str | None] = {}
    for name in KEY_PACKAGES:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = None
    return out


def code_version(repo_root: Path | None = None) -> dict[str, str | bool | None]:
    root = repo_root or Path(streamrag.__file__).resolve().parents[2]
    def git(*args: str) -> str | None:
        try:
            r = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True, timeout=5)
            return r.stdout.strip() if r.returncode == 0 else None
        except (OSError, subprocess.SubprocessError):
            return None
    sha = git("rev-parse", "HEAD")
    status = git("status", "--porcelain")
    return {"package_version": streamrag.__version__, "git_sha": sha or "uncommitted",
            "git_dirty": bool(status) if status is not None else None}


def environment() -> dict:
    return {"python": sys.version.split()[0], "implementation": platform.python_implementation(),
            "platform": platform.platform(), "machine": platform.machine(), "packages": package_versions(),
            "code": code_version()}
