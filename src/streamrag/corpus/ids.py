"""Deterministic document / chunk identifiers and citation keys.

Document IDs: a native ID wins (front-matter ``id`` or a file name matching ``native_doc_id_pattern``, e.g.
``Doc_42_handbook.pdf`` -> ``Doc_42``). Otherwise:
  * ``native_or_stem``    -> sanitized relative path without extension (stable when other files are added);
  * ``native_or_ordinal`` -> ``Doc_<n>`` by sorted path order, skipping numbers used by native IDs.
Same corpus + same configuration => identical IDs (tested).
"""

from __future__ import annotations

import re
from pathlib import PurePosixPath

from streamrag.errors import CorpusIntegrityError


def _sanitize(relpath: str) -> str:
    stem = str(PurePosixPath(relpath).with_suffix(""))
    return re.sub(r"[^A-Za-z0-9]+", "_", stem).strip("_") or "doc"


def native_document_id(relpath: str, front_matter: dict, pattern: str) -> str | None:
    fm = front_matter.get("id") or front_matter.get("doc_id") or front_matter.get("document_id")
    if fm:
        return str(fm).strip()
    m = re.match(pattern, PurePosixPath(relpath).name)
    return f"Doc_{m.group(1)}" if m else None


def assign_document_ids(relpaths: list[str], front_matters: list[dict], strategy: str, pattern: str) -> list[str]:
    natives = [native_document_id(p, fm, pattern) for p, fm in zip(relpaths, front_matters)]
    seen: dict[str, str] = {}
    for p, n in zip(relpaths, natives):
        if n is None:
            continue
        if n in seen:
            raise CorpusIntegrityError(f"duplicate native document id '{n}' in '{seen[n]}' and '{p}'")
        seen[n] = p
    out: list[str] = []
    used = set(seen)
    used_numbers = {int(m.group(1)) for n in seen if (m := re.match(r"^Doc_(\d+)$", n))}
    next_ordinal = 1
    for p, n in zip(relpaths, natives):
        if n is not None:
            out.append(n)
            continue
        if strategy == "native_or_ordinal":
            while next_ordinal in used_numbers:
                next_ordinal += 1
            cand = f"Doc_{next_ordinal}"
            used_numbers.add(next_ordinal)
        else:
            cand = _sanitize(p)
        base, k = cand, 2
        while cand in used:
            cand = f"{base}_{k}"
            k += 1
        used.add(cand)
        out.append(cand)
    return out


def render_chunk_id(template: str, document_id: str, section_id: str, part: int) -> str:
    return template.format(document_id=document_id, section_id=section_id, part=part)


def render_citation(template: str, document_id: str, section_id: str) -> str:
    return template.format(document_id=document_id, section_id=section_id)
