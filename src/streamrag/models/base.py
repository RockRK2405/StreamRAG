"""Shared base for all contract models: strict validation + deterministic serialization."""

from __future__ import annotations

import json
from typing import Any, Self

from pydantic import BaseModel, ConfigDict

SCHEMA_VERSION = "1.0"


def canonical_dumps(obj: Any) -> str:
    """Deterministic JSON: sorted keys, no insignificant whitespace, UTF-8 preserved."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


class Contract(BaseModel):
    """Base contract: unknown fields are rejected; output is deterministic."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True, validate_assignment=True)

    def canonical_json(self) -> str:
        return canonical_dumps(self.model_dump(mode="json"))

    @classmethod
    def from_json(cls, data: str | bytes) -> Self:
        return cls.model_validate_json(data)
