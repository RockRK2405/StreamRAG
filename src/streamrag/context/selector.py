"""Relevant-context selection and context compression (Phase 6; docs/session/08-09).

``RelevantContextSelector.select`` builds the context package a later stage (Phase 7 generation) may see for the
active topic frame (or the frame of one need). It is *relevance-based*, never "the whole history":

  included   active needs of the frame (resolved text), their active constraints, the frame's entities, the selected
             SUPPORTED / PARTIALLY_SUPPORTED claims of those needs, and the usable (ACTIVE / RETAINED) evidence ids
  excluded   (counted by reason, never silently dropped) other / dormant frames, superseded or removed needs,
             retracted or replaced constraints, evidence that is not usable, claims that are not supported

Every item carries its provenance (utterance ids + character spans for needs / constraints / entities; evidence id +
character span for claims), so compression never loses the link back to the source.

``ContextCompressor.compress`` represents the whole conversation as structured state: per utterance its hash plus
the needs / constraints / entities derived from it (with spans), and the raw text only for utterances inside the
transcript window. Sizes are measured (characters always; tokens only when a tokenizer is supplied - the benchmark
passes the bge-small WordPiece tokenizer and labels the count as a proxy, since the Phase 7 generator's tokenizer is
not known yet).

Three context modes exist for the memory ablation (research/phase6):
  ``none``             the current utterance only
  ``full_transcript``  every utterance verbatim (the naive approach)
  ``structured``       the selector's package
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

from pydantic import Field

from streamrag.delta.models import USABLE_EVIDENCE
from streamrag.models.base import Contract

ContextMode = Literal["none", "full_transcript", "structured"]
_SUPPORTED = ("SUPPORTED", "PARTIALLY_SUPPORTED")


class Provenance(Contract):
    utterance_id: str | None = None
    start: int | None = None
    end: int | None = None
    evidence_id: str | None = None


class ContextItem(Contract):
    kind: Literal["utterance", "need", "constraint", "entity", "claim", "evidence"]
    item_id: str
    text: str
    intent_id: str | None = None
    provenance: list[Provenance] = []


class ContextSize(Contract):
    items: int = Field(0, ge=0)
    chars: int = Field(0, ge=0)
    tokens: int | None = None                  # measured with the supplied tokenizer; None when none supplied
    tokenizer: str | None = None


class ContextPackage(Contract):
    mode: str
    frame_id: str | None = None
    intent_ids: list[str] = []
    items: list[ContextItem] = []
    excluded: dict[str, int] = {}              # reason -> count
    size: ContextSize = ContextSize()
    size_by_kind: dict[str, ContextSize] = {}  # "conversation" state (needs/constraints/entities/utterances) vs
                                               # retrieval output (claims/evidence) are reported separately


class CompressedUtterance(Contract):
    utterance_id: str
    text_sha1: str
    chars: int = Field(0, ge=0)
    verbatim: str | None = None                # only inside the transcript window
    derived: list[ContextItem] = []


class CompressedContext(Contract):
    utterances: list[CompressedUtterance] = []
    raw: ContextSize = ContextSize()           # the verbatim conversation
    compressed: ContextSize = ContextSize()    # what memory holds: window verbatim + derived state of older turns
    structured: ContextSize = ContextSize()    # derived state (needs / constraints) of every turn
    ratio_chars: float | None = None           # compressed / raw
    ratio_structured_chars: float | None = None


def _size(texts: list[str], count: Callable[[str], int] | None, name: str | None) -> ContextSize:
    return ContextSize(items=len(texts), chars=sum(len(t) for t in texts),
                       tokens=sum(count(t) for t in texts) if count else None, tokenizer=name if count else None)


def token_counter(embedder) -> tuple[Callable[[str], int] | None, str | None]:
    """A token counter from the embedder's HF tokenizer (no truncation / padding), or (None, None)."""
    tok = getattr(embedder, "tokenizer", None)
    if tok is None:
        return None, None
    from tokenizers import Tokenizer
    t = Tokenizer.from_str(tok.to_str())
    t.no_truncation()
    t.no_padding()
    name = getattr(embedder, "name", "embedder")
    return (lambda s: len(t.encode(s, add_special_tokens=False).ids)), f"{name} (proxy)"


class RelevantContextSelector:
    def __init__(self, memory, count: Callable[[str], int] | None = None, tokenizer_name: str | None = None) -> None:
        self.m, self.count, self.tok_name = memory, count, tokenizer_name

    def _package(self, mode: str, frame_id, intent_ids, items, excluded) -> ContextPackage:
        groups = {"conversation": [i.text for i in items if i.kind not in ("claim", "evidence")],
                  "retrieval": [i.text for i in items if i.kind in ("claim", "evidence")]}
        return ContextPackage(mode=mode, frame_id=frame_id, intent_ids=intent_ids, items=items,
                              excluded=dict(sorted(excluded.items())),
                              size=_size([i.text for i in items], self.count, self.tok_name),
                              size_by_kind={k: _size(v, self.count, self.tok_name) for k, v in groups.items()})

    def select(self, mode: ContextMode = "structured", intent_id: str | None = None,
               current_utterance: str | None = None) -> ContextPackage:
        m = self.m
        if mode == "none":
            t = next((e for e in reversed(m.transcript) if e.utterance_id == (current_utterance or m.current_utterance_id)),
                     None)
            items = [ContextItem(kind="utterance", item_id=t.utterance_id, text=t.text,
                                 provenance=[Provenance(utterance_id=t.utterance_id)])] if t and t.text else []
            return self._package(mode, None, [], items, {})
        if mode == "full_transcript":
            items = [ContextItem(kind="utterance", item_id=t.utterance_id, text=t.text,
                                 provenance=[Provenance(utterance_id=t.utterance_id)]) for t in m.transcript if t.text]
            lost = sum(1 for t in m.transcript if not t.text)
            return self._package(mode, None, [], items, {"utterance_outside_window": lost} if lost else {})
        tr, frames = m.tracker, m.frames
        frame = frames.frame_of(intent_id) if intent_id else frames.active
        excluded: dict[str, int] = {}

        def skip(reason: str, n: int = 1) -> None:
            if n:
                excluded[reason] = excluded.get(reason, 0) + n
        for f in frames.frames:
            if frame is None or f.frame_id != frame.frame_id:
                skip(f"frame_{f.status}", sum(1 for i in f.intent_ids if i in tr.intents))
        if frame is None:
            return self._package(mode, None, [], [], excluded)
        ids = [i for i in frame.intent_ids if i in tr.intents and tr.intents[i].status == "ACTIVE"]
        if intent_id:
            ids = [intent_id] if intent_id in ids else []
        skip("need_not_active", sum(1 for i in frame.intent_ids if i in tr.intents and tr.intents[i].status != "ACTIVE"))
        items: list[ContextItem] = []
        for iid in ids:
            it = tr.intents[iid]
            items.append(ContextItem(kind="need", item_id=iid, text=it.resolved_text, intent_id=iid,
                                     provenance=[Provenance(utterance_id=c.span.utterance_id, start=c.span.start,
                                                            end=c.span.end) for c in it.components]))
            active = tr.constraints_for(it)
            for k in active:
                items.append(ContextItem(kind="constraint", item_id=k.constraint_id, text=k.text, intent_id=iid,
                                         provenance=[Provenance(utterance_id=k.source_span.utterance_id,
                                                                start=k.source_span.start, end=k.source_span.end)]))
            act_ids = {k.constraint_id for k in active}
            skip("constraint_retracted_or_replaced",
                 sum(1 for k in tr.constraints.values() if iid in k.applies_to and k.constraint_id not in act_ids))
            for cid in m.graph.selected.get(iid, []):
                c = m.graph.claims[cid]
                if c.status not in _SUPPORTED:
                    skip(f"claim_{c.status.lower()}")
                    continue
                items.append(ContextItem(kind="claim", item_id=cid, text=c.text, intent_id=iid, provenance=[
                    Provenance(evidence_id=c.source.evidence_id, start=c.source.char_start, end=c.source.char_end)]
                    if c.source else []))
            for a in m.store.for_intent(iid):
                if a.status not in USABLE_EVIDENCE:
                    skip(f"evidence_{a.status.lower()}")
                    continue
                items.append(ContextItem(kind="evidence", item_id=a.evidence_id, intent_id=iid,
                                         text=m.store.records[a.evidence_id].citation,
                                         provenance=[Provenance(evidence_id=a.evidence_id)]))
        for e in m.get_entity_context(frame_only=True):
            if not set(e.intent_ids) & set(ids):
                continue
            items.append(ContextItem(kind="entity", item_id=" ".join(sorted(e.terms)), text=e.text,
                                     provenance=[Provenance(utterance_id=e.first_seen)]))
        return self._package(mode, frame.frame_id, ids, items, excluded)


class ContextCompressor:
    def __init__(self, memory, count: Callable[[str], int] | None = None, tokenizer_name: str | None = None) -> None:
        self.m, self.count, self.tok_name = memory, count, tokenizer_name

    def compress(self, raw_texts: dict[str, str] | None = None) -> CompressedContext:
        """Structured view of the conversation. ``raw_texts`` (utterance id -> original text) is only used to measure
        the raw size; memory itself never keeps text outside the window."""
        m, tr = self.m, self.m.tracker
        out = []
        for t in m.transcript:
            derived = []
            for it in tr.intents.values():
                if it.utterance_id == t.utterance_id and it.status == "ACTIVE":
                    derived.append(ContextItem(kind="need", item_id=it.intent_id, text=it.resolved_text,
                                               intent_id=it.intent_id, provenance=[Provenance(
                                                   utterance_id=c.span.utterance_id, start=c.span.start,
                                                   end=c.span.end) for c in it.components]))
            for k in tr.constraints.values():
                if k.source_span.utterance_id == t.utterance_id and k.status == "active":
                    derived.append(ContextItem(kind="constraint", item_id=k.constraint_id, text=k.text,
                                               provenance=[Provenance(utterance_id=t.utterance_id,
                                                                      start=k.source_span.start,
                                                                      end=k.source_span.end)]))
            out.append(CompressedUtterance(utterance_id=t.utterance_id, text_sha1=t.text_sha1, chars=t.chars,
                                           verbatim=t.text or None, derived=derived))
        raw = [raw_texts.get(u.utterance_id, "") for u in out] if raw_texts else []
        comp = [u.verbatim if u.verbatim else d.text for u in out for d in ([None] if u.verbatim else u.derived)]
        rs = _size(raw, self.count, self.tok_name) if raw_texts else ContextSize(items=len(out),
                                                                               chars=sum(u.chars for u in out))
        cs = _size([c for c in comp if c], self.count, self.tok_name)
        ss = _size([d.text for u in out for d in u.derived], self.count, self.tok_name)
        return CompressedContext(utterances=out, raw=rs, compressed=cs, structured=ss,
                                 ratio_chars=round(cs.chars / rs.chars, 4) if rs.chars else None,
                                 ratio_structured_chars=round(ss.chars / rs.chars, 4) if rs.chars else None)
