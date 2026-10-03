"""Answer rendering (Phase 7; docs/answer/05, 10). Deterministic; adds no content.

* factual claim  -> the claim text followed by its own citations: "… shed. [Doc_07 §2.2]" (claim-level placement)
* conflict group -> "The sources differ: “A” [k1] / “B” [k2]." (both sides verbatim, each with its own citation)
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


def render_answer(sections, claims, cmap):
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
