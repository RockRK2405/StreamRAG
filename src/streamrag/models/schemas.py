"""Export JSON Schemas for all contract models (docs/schemas/<Model>.schema.json)."""

from __future__ import annotations

import json
from pathlib import Path

from streamrag.models import SCHEMA_MODELS, _phase4_models


def all_schema_models() -> dict:
    return {**SCHEMA_MODELS, **_phase4_models()}


def render_schemas() -> dict[str, str]:
    out = {}
    for name, model in all_schema_models().items():
        schema = model.model_json_schema()
        out[f"{name}.schema.json"] = json.dumps(schema, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    return out


def export_schemas(target: Path) -> list[Path]:
    target.mkdir(parents=True, exist_ok=True)
    written = []
    for fname, text in render_schemas().items():
        p = target / fname
        p.write_text(text)
        written.append(p)
    return written
