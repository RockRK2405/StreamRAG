"""Corpus/index manifest: everything required to reproduce the index (never benchmark answers)."""

from __future__ import annotations

import hashlib
import time

from streamrag.config.settings import StreamRagConfig, index_config_hash
from streamrag.corpus.pipeline import BuiltCorpus
from streamrag.models.base import canonical_dumps
from streamrag.models.corpus import CorpusManifest, EmbeddingInfo
from streamrag.provenance import environment

INDEX_VERSION = "3.0"  # bump on any change to artifact layout or ID/chunk semantics


def manifest_content_hash(m: CorpusManifest) -> str:
    data = m.model_dump(mode="json", exclude={"generated_at", "environment", "content_hash", "corpus_root"})
    return hashlib.sha256(canonical_dumps(data).encode()).hexdigest()


def build_manifest(built: BuiltCorpus, cfg: StreamRagConfig, embedding: EmbeddingInfo | None,
                   artifacts: dict[str, str]) -> CorpusManifest:
    m = CorpusManifest(
        index_version=INDEX_VERSION,
        corpus_version=built.corpus_hash,
        corpus_root=built.corpus_root,
        is_test_fixture=built.is_test_fixture,
        generated_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        source_files=built.source_files,
        document_count=len(built.documents),
        section_count=built.section_count,
        chunk_count=len(built.chunks),
        chunking_configuration=cfg.chunking.model_dump(mode="json"),
        normalization_configuration=cfg.normalization.model_dump(mode="json"),
        sections_configuration=cfg.sections.model_dump(mode="json"),
        ids_configuration=cfg.ids.model_dump(mode="json"),
        index_text_configuration=cfg.index_text.model_dump(mode="json"),
        bm25_configuration=cfg.lexical.model_dump(mode="json"),
        embedding_model=embedding,
        index_config_hash=index_config_hash(cfg),
        artifacts=dict(sorted(artifacts.items())),
        environment=environment(),
    )
    return m.model_copy(update={"content_hash": manifest_content_hash(m)})
