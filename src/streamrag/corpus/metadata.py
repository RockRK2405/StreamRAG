"""Document metadata from front matter (Phase 9, docs/retrieval/09).

Only whitelisted fields (``corpus.metadata_fields``) are kept; values are reduced to short scalars: dates become ISO
strings, lists comma-joined, control characters removed, length capped. Metadata is corpus content and therefore
untrusted: it can narrow retrieval (metadata filters, validity dates) but is never used for routing, budgets, the
scheduler or prompts.
"""

from __future__ import annotations

import datetime as _dt
import re

MAX_LEN = 120
_CTRL = re.compile(r"[\x00-\x1f\x7f]")
_ISO = re.compile(r"^\d{4}-\d{2}-\d{2}")


def _scalar(v) -> str | int | float | bool | None:
    if isinstance(v, bool | int | float):
        return v
    if isinstance(v, _dt.date | _dt.datetime):
        return v.isoformat()[:10]
    if isinstance(v, list | tuple):
        return ", ".join(str(_scalar(x)) for x in v if _scalar(x) is not None)[:MAX_LEN] or None
    if v is None or isinstance(v, dict):
        return None
    s = _CTRL.sub(" ", str(v)).strip()
    return s[:MAX_LEN] or None


def sanitize_metadata(front_matter: dict, fields: list[str]) -> dict[str, str | int | float | bool]:
    out = {}
    allowed = set(fields)
    for k, v in (front_matter or {}).items():
        key = str(k).strip().lower()
        if key in allowed:
            val = _scalar(v)
            if isinstance(val, str) and (key.endswith("_date") or key == "valid_until") and _ISO.match(val):
                val = val[:10]                    # a datetime string is reduced to its date
            if val is not None:
                out[key] = val
    return out
