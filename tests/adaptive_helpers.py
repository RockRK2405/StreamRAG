"""Helpers for the Phase 9 adaptive retrieval tests (fixture corpus tests/fixtures/corpus_adaptive)."""

from __future__ import annotations

from streamrag.adaptive.controller import AdaptiveRequest
from streamrag.adaptive.integration import make_controller
from streamrag.models.evidence import Evidence


def with_cfg(cfg, **dotted):
    for key, val in dotted.items():
        parts = key.split(".")
        objs = [cfg]
        for p in parts[:-1]:
            objs.append(getattr(objs[-1], p))
        new = objs[-1].model_copy(update={parts[-1]: val})
        for obj, p in zip(reversed(objs[:-1]), reversed(parts[:-1])):
            new = obj.model_copy(update={p: new})
        cfg = new
    return cfg


def controller(env, **overrides):
    cfg, svc = env
    return make_controller(svc, with_cfg(cfg, **overrides))


def run(env, text, qid="q1", **overrides):
    ctl = controller(env, **overrides)
    try:
        return ctl.run(AdaptiveRequest(qid, text, original_text=text))
    finally:
        ctl.close()


def citations(res) -> list[str]:
    return [e.citation for e in res.evidence.items]


def evidence(svc, citation: str, rank: int = 1) -> Evidence:
    """An Evidence object for an indexed chunk (by citation) - for evaluator unit tests."""
    ch = next(c for c in svc.bundle.chunks if c.citation == citation)
    return Evidence(evidence_id=ch.chunk_id, document_id=ch.document_id, section_id=ch.section_id,
                    chunk_id=ch.chunk_id, citation=ch.citation, section_title=ch.section_title,
                    document_title=ch.title, text=ch.text, source_path=ch.source_path, char_start=ch.char_start,
                    char_end=ch.char_end, rank=rank, score=1.0, retrieval_method="hybrid_rrf",
                    metadata=dict(svc.doc_meta.get(ch.document_id, {})))
