"""Section parsing: normalized document text -> hierarchical sections with paragraphs and char spans.

Section IDs (citation unit ``Doc_ID §Section``):
  * native numbers from the source when present ("2.1", "§4", "Section 3") — the official convention wins;
  * otherwise deterministic ordinal paths ("1", "1.2"); prefixed with "u" in documents that also have native
    numbers so derived IDs can never collide with native ones;
  * "0" = preamble (text before the first heading); "p<N>" = page sections for heading-less PDFs.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from streamrag.config.settings import SectionsConfig
from streamrag.corpus.headings import HeadingMatch, match_caps_heading, match_heading
from streamrag.corpus.normalize import page_at
from streamrag.models.corpus import CorpusSection, Paragraph

_BLOCK = re.compile(r"[^\n]+(?:\n[^\n]+)*")


@dataclass
class Block:
    text: str
    start: int
    end: int


def split_blocks(text: str) -> list[Block]:
    return [Block(m.group(0), m.start(), m.end()) for m in _BLOCK.finditer(text)]


def _detect(blocks: list[Block], cfg: SectionsConfig, fmt: str) -> list[HeadingMatch | None]:
    cands: list[HeadingMatch | None] = []
    for b in blocks:
        single = "\n" not in b.text
        cands.append(match_heading(b.text, cfg.max_heading_words, allow_markdown=cfg.markdown_headings,
                                   allow_numbered=cfg.numbered_headings) if single else None)
    # A run of >=2 adjacent, *sequentially numbered* candidates of the same level with no body between is a
    # list ("1. A / 2. B / 3. C"), not headings. The sequence check stops the run at a real heading that
    # follows the list ("... 3. C" then "2. Pruning").
    def _next_in_sequence(a: HeadingMatch, b: HeadingMatch) -> bool:
        pa, pb = a.number.split("."), b.number.split(".")   # type: ignore[union-attr]
        return len(pa) == len(pb) and pa[:-1] == pb[:-1] and int(pb[-1]) == int(pa[-1]) + 1

    i = 0
    while i < len(cands):
        c = cands[i]
        if c is not None and c.kind == "numbered":
            j = i
            while j + 1 < len(cands) and cands[j + 1] is not None and cands[j + 1].kind == "numbered" \
                    and cands[j + 1].level == c.level and _next_in_sequence(cands[j], cands[j + 1]):
                j += 1
            if j > i:
                for k in range(i, j + 1):
                    cands[k] = None
            i = j + 1
        else:
            i += 1
    if not any(cands) and cfg.caps_headings:
        cands = [match_caps_heading(b.text, cfg.max_heading_words) if "\n" not in b.text else None for b in blocks]
    return cands


def parse_sections(document_id: str, text: str, page_offsets: list[tuple[int, int, int]], fmt: str,
                   default_title: str, cfg: SectionsConfig, front_matter: dict | None = None
                   ) -> tuple[str, list[CorpusSection]]:
    """Return (document title, sections)."""
    blocks = split_blocks(text)
    cands = _detect(blocks, cfg, fmt)
    title = str((front_matter or {}).get("title") or "").strip()

    # Markdown convention: a single leading unnumbered H1 is the document title, not a section.
    h1 = [i for i, c in enumerate(cands) if c is not None and c.kind == "markdown" and c.level == 1]
    first = next((i for i, c in enumerate(cands) if c is not None), None)
    title_block: int | None = None
    if fmt == "md" and len(h1) == 1 and h1[0] == first and cands[first].number is None:
        title_block = first
        title = title or cands[first].title
        cands[first] = None
    # txt/pdf: an ALL-CAPS first line of the preamble is the document title (kept in the preamble text).
    if not title and fmt != "md" and blocks and (first is None or first > 0):
        caps = match_caps_heading(blocks[0].text.split("\n", 1)[0], cfg.max_heading_words)
        if caps is not None:
            title = caps.title
    title = title or default_title

    def para(b: Block) -> Paragraph:
        return Paragraph(text=b.text, char_start=b.start, char_end=b.end, page=page_at(page_offsets, b.start))

    sections: list[CorpusSection] = []
    has_native = any(c is not None and c.number for c in cands)
    body_blocks = [i for i in range(len(blocks)) if i != title_block]

    if not any(cands):
        if page_offsets and len(page_offsets) > 1 and cfg.pdf_page_sections_when_no_headings:
            for ordinal, (page, start, end) in enumerate(page_offsets):
                ps = [para(blocks[i]) for i in body_blocks if start <= blocks[i].start <= end]
                if ps:
                    sections.append(CorpusSection(section_id=f"p{page}", document_id=document_id, title=f"Page {page}",
                                                  level=1, path=[f"Page {page}"], ordinal=ordinal,
                                                  char_start=ps[0].char_start, char_end=ps[-1].char_end,
                                                  page_start=page, page_end=page, paragraphs=ps))
            return title, sections
        ps = [para(blocks[i]) for i in body_blocks]
        if ps:
            sections.append(CorpusSection(section_id="1", document_id=document_id, title=title, level=1, path=[title],
                                          ordinal=0, char_start=ps[0].char_start, char_end=ps[-1].char_end,
                                          page_start=ps[0].page, page_end=ps[-1].page, paragraphs=ps))
        return title, sections

    # Preamble
    first_heading = next(i for i, c in enumerate(cands) if c is not None)
    pre = [para(blocks[i]) for i in body_blocks if i < first_heading]
    if pre:
        sections.append(CorpusSection(section_id="0", document_id=document_id, title=title, level=0, path=[],
                                      ordinal=0, char_start=pre[0].char_start, char_end=pre[-1].char_end,
                                      page_start=pre[0].page, page_end=pre[-1].page, paragraphs=pre))

    used: set[str] = {s.section_id for s in sections}
    stack: list[tuple[int, CorpusSection, list[int]]] = []   # (level, section, derived ordinal path)
    child_counts: dict[str | None, int] = {}
    heading_idx = [i for i, c in enumerate(cands) if c is not None]
    for n, hi in enumerate(heading_idx):
        h = cands[hi]
        assert h is not None
        while stack and stack[-1][0] >= h.level:
            stack.pop()
        parent = stack[-1][1] if stack else None
        parent_key = parent.section_id if parent else None
        child_counts[parent_key] = child_counts.get(parent_key, 0) + 1
        derived = (stack[-1][2] if stack else []) + [child_counts[parent_key]]
        if h.number:
            sid, native = h.number, True
        else:
            sid, native = ("u" if has_native else "") + ".".join(map(str, derived)), False
        base, k = sid, 2
        while sid in used:
            sid = f"{base}-{k}"
            k += 1
        used.add(sid)
        nxt = heading_idx[n + 1] if n + 1 < len(heading_idx) else len(blocks)
        ps = [para(blocks[i]) for i in body_blocks if hi < i < nxt]
        hb = blocks[hi]
        end = ps[-1].char_end if ps else hb.end
        sec = CorpusSection(section_id=sid, document_id=document_id, title=h.title, level=h.level,
                            path=(parent.path if parent else []) + [h.title], parent_id=parent_key,
                            ordinal=len(sections), native_id=native, char_start=hb.start, char_end=end,
                            page_start=page_at(page_offsets, hb.start), page_end=page_at(page_offsets, end),
                            paragraphs=ps)
        sections.append(sec)
        stack.append((h.level, sec, derived))
    return title, sections
