"""Answer rendering (Phase 7; docs/answer/05, 10). Deterministic; adds no content.

* factual claim  -> the claim text followed by its own citations: "… shed. [Doc_07 §2.2]" (claim-level placement)
* conflict group -> "The sources differ: “A” [k1] / “B” [k2]." (both sides verbatim, each with its own citation);
                    when the sides come from a document and an older version it supersedes (agreed supersession:
                    the old document is marked superseded / expired), "Current version: “A” [k1]. Earlier, superseded
                    version: “B” [k2]." - both sides stay visible (Phase 11)
* uncertainty    -> "Not established: …" (never cited, never phrased as a fact)
Sections are headed by their need when the answer has more than one section (multi-intent answers stay separated).
"""

from __future__ import annotations


def _cites(c, cits) -> str:
    keys = list(dict.fromkeys(cits[x].display_metadata.get("key", "") for x in c.citation_ids if x in cits))
    return f" [{'; '.join(keys)}]" if keys else ""


def _title(t: str) -> str:
    t = " ".join(t.split()).rstrip("?.")
    return t[:1].upper() + t[1:] if t else t


_DEMOTED = {"superseded", "expired", "archived", "withdrawn", "replaced"}


def _versions(side, doc_info) -> tuple[list, list] | None:
    """(current side, superseded side) when the conflict is between a document and a version it supersedes."""
    if doc_info is None:
        return None
    info = {id(x): doc_info(x) for x in side}
    cur = [x for x in side if info[id(x)] and info[id(x)][1] not in _DEMOTED]
    old = [x for x in side if info[id(x)] and info[id(x)][1] in _DEMOTED]
    if not cur or not old or len(cur) + len(old) != len(side):
        return None
    sup = {d for x in cur for d in (info[id(x)][2] or [])}
    if not all(info[id(x)][0] in sup for x in old):
        return None
    return cur, old


def render_answer(sections, claims, cmap, doc_info=None):
    """``doc_info(claim) -> (document_id, status, [ids it supersedes])`` or None (enables version-aware conflicts)."""
    cits = {c.citation_id: c for c in cmap.citations if c.status == "valid"}
    out_sections, blocks = [], []
    multi = len(sections) > 1
    for s in sections:
        lines, done_groups = [], set()
        for cid in s.claim_ids:
            c = claims[cid]
            if c.kind == "conflict" and c.conflict_group:
                if c.conflict_group in done_groups:
                    continue
                done_groups.add(c.conflict_group)
                side = [claims[x] for x in s.claim_ids if claims[x].conflict_group == c.conflict_group]
                v = _versions(side, doc_info)
                if v is not None:
                    cur, old = v
                    lines.append("Current version: " + " / ".join(f"“{x.text.rstrip('.')}”{_cites(x, cits)}"
                                                                  for x in cur) + ".")
                    lines.append("Earlier, superseded version: " + " / ".join(
                        f"“{x.text.rstrip('.')}”{_cites(x, cits)}" for x in old) + ".")
                    continue
                parts = " / ".join(f"“{x.text.rstrip('.')}”{_cites(x, cits)}" for x in side)
                lines.append(f"The sources differ: {parts}.")
            elif c.kind == "uncertainty":
                lines.append(f"Not established: {c.text}")
            elif c.kind == "fact":
                lines.append(f"{c.text}{_cites(c, cits)}")
        text = " ".join(lines)
        out_sections.append(s.model_copy(update={"text": text}))
        blocks.append(f"{_title(s.title)}:\n{text}" if multi else text)
    return out_sections, "\n\n".join(b for b in blocks if b.strip())
