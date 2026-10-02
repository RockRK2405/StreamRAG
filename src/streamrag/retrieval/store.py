"""Index build / load with integrity checks.

Layout of ``indexes/<corpus_hash[:12]>-<index_config_hash[:10]>/``::

    manifest.json        CorpusManifest (reproducibility record; artifact sha256s)
    chunks.jsonl         one CorpusChunk per line (canonical JSON)
    documents.jsonl      normalized documents + sections (traceability: chunk spans point into these texts)
    bm25/                weights.npz, doc_len.npy, vocab.json, meta.json
    dense/               embeddings.npy, meta.json   (omitted when built lexical-only)
    build_timings.json   informational (not hashed)

Loading verifies the index version, every artifact hash, chunk counts and embedding dimensions; any mismatch
raises an explicit error instead of serving possibly-wrong evidence.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from pydantic import ValidationError

from streamrag.config.settings import StreamRagConfig, index_config_hash
from streamrag.corpus.manifest import INDEX_VERSION, build_manifest
from streamrag.corpus.pipeline import build_corpus
from streamrag.corpus.source import CorpusSource, sha256_file
from streamrag.errors import (
    EmbeddingDimensionMismatchError,
    IndexCorruptError,
    IndexNotFoundError,
    IndexVersionMismatchError,
)
from streamrag.models.corpus import CorpusChunk, CorpusDocument, CorpusManifest
from streamrag.retrieval.bm25 import BM25Index
from streamrag.retrieval.dense import DenseIndex
from streamrag.retrieval.embedders import Embedder, load_embedder
from streamrag.retrieval.text import Analyzer
from streamrag.telemetry.logging import get_logger
from streamrag.telemetry.timing import Stopwatch

log = get_logger("index")


@dataclass
class IndexBundle:
    path: Path
    manifest: CorpusManifest
    chunks: list[CorpusChunk]
    bm25: BM25Index
    dense: DenseIndex | None
    analyzer: Analyzer
    timings_ms: dict[str, float] = field(default_factory=dict)

    def documents(self) -> list[CorpusDocument]:
        with (self.path / "documents.jsonl").open(encoding="utf-8") as f:
            return [CorpusDocument.model_validate_json(line) for line in f if line.strip()]


def make_analyzer(lexical: dict) -> Analyzer:
    return Analyzer(stemming=lexical["stemming"], stopwords=lexical["stopwords"],
                    normalize_number_words=lexical["normalize_number_words"])


def index_dir_for(cfg: StreamRagConfig, corpus_hash: str) -> Path:
    return cfg.paths.index_root / f"{corpus_hash[:12]}-{index_config_hash(cfg)[:10]}"


def resolve_index(cfg: StreamRagConfig) -> Path:
    src = CorpusSource(cfg.paths.corpus, cfg.corpus.include_extensions, cfg.corpus.fixture_marker)
    path = index_dir_for(cfg, src.corpus_hash())
    if not (path / "manifest.json").exists():
        raise IndexNotFoundError(f"no index for the current corpus+config at {path}; run `streamrag build-index`")
    return path


def _write_jsonl(path: Path, items: list) -> None:
    with path.open("w", encoding="utf-8") as f:
        for it in items:
            f.write(it.canonical_json() + "\n")


def build_index(cfg: StreamRagConfig, embedder: Embedder | None = None, force: bool = False,
                out_dir: Path | None = None) -> IndexBundle:
    timings: dict[str, float] = {}
    with Stopwatch() as sw:
        built = build_corpus(cfg)
    timings.update({f"corpus_{k}": v for k, v in built.timings_ms.items()})
    timings["corpus_pipeline_total"] = sw.ms
    target = out_dir or index_dir_for(cfg, built.corpus_hash)
    if (target / "manifest.json").exists() and not force:
        log.info("index_reused", extra={"fields": {"path": str(target)}})
        bundle = load_index(target)
        bundle.timings_ms = {"reused": 1.0}
        return bundle

    tmp = target.parent / (target.name + ".tmp-build")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    texts = [c.index_text for c in built.chunks]
    analyzer = make_analyzer(cfg.lexical.model_dump())

    with Stopwatch() as sw:
        bm25 = BM25Index.build(texts, analyzer, cfg.lexical.k1, cfg.lexical.b)
    timings["bm25_build"] = sw.ms
    bm25.save(tmp / "bm25")

    if embedder is None:
        embedder = load_embedder(cfg.dense.embedder, cfg.paths.model_registry, cfg.paths.models_dir, analyzer,
                                 cfg.dense.intra_op_threads, cfg.dense.batch_size)
    with Stopwatch() as sw:
        vectors = embedder.embed(texts, "document")
    timings["embedding_build"] = sw.ms
    truncated = embedder.count_truncated(texts, "document")
    info = embedder.info(truncated).model_copy(update={"build_batch_size": getattr(embedder, "batch_size", None)})
    DenseIndex(vectors, info).save(tmp / "dense")

    _write_jsonl(tmp / "chunks.jsonl", built.chunks)
    _write_jsonl(tmp / "documents.jsonl", built.documents)
    artifacts = {p.relative_to(tmp).as_posix(): sha256_file(p)
                 for p in sorted(tmp.rglob("*")) if p.is_file()}
    manifest = build_manifest(built, cfg, info, artifacts)
    (tmp / "manifest.json").write_text(json.dumps(manifest.model_dump(mode="json"), indent=2, sort_keys=True,
                                                  ensure_ascii=False))
    (tmp / "build_timings.json").write_text(json.dumps(timings, indent=2, sort_keys=True))
    if target.exists():
        shutil.rmtree(target)
    tmp.rename(target)
    log.info("index_built", extra={"fields": {"path": str(target), "chunks": len(built.chunks),
                                              "fixture": built.is_test_fixture, "truncated": truncated}})
    with Stopwatch() as sw:
        bundle = load_index(target)
    timings["index_load"] = sw.ms
    bundle.timings_ms = timings
    return bundle


def load_index(path: Path, verify: bool = True) -> IndexBundle:
    path = Path(path)
    mpath = path / "manifest.json"
    if not mpath.exists():
        raise IndexNotFoundError(f"index manifest not found: {mpath}")
    try:
        manifest = CorpusManifest.model_validate_json(mpath.read_text())
    except ValidationError as exc:
        raise IndexCorruptError(f"invalid manifest {mpath}: {exc}") from exc
    if manifest.index_version != INDEX_VERSION:
        raise IndexVersionMismatchError(f"index version {manifest.index_version} != supported {INDEX_VERSION}; rebuild")
    if verify:
        for rel, sha in manifest.artifacts.items():
            f = path / rel
            if not f.exists():
                raise IndexCorruptError(f"missing index artifact: {rel}")
            if sha256_file(f) != sha:
                raise IndexCorruptError(f"checksum mismatch for index artifact: {rel}")
    try:
        with (path / "chunks.jsonl").open(encoding="utf-8") as fh:
            chunks = [CorpusChunk.model_validate_json(line) for line in fh if line.strip()]
    except (OSError, ValidationError) as exc:
        raise IndexCorruptError(f"chunks unreadable: {exc}") from exc
    if len(chunks) != manifest.chunk_count:
        raise IndexCorruptError(f"chunk count {len(chunks)} != manifest {manifest.chunk_count}")
    bm25 = BM25Index.load(path / "bm25")
    if bm25.n_docs != len(chunks):
        raise IndexCorruptError("BM25 row count does not match chunk count")
    dense = None
    if (path / "dense" / "meta.json").exists():
        dense = DenseIndex.load(path / "dense")
        if dense.n_docs != len(chunks):
            raise IndexCorruptError("dense row count does not match chunk count")
        em = manifest.embedding_model
        if em is None or em.name != dense.info.name or em.dimension != dense.info.dimension:
            raise EmbeddingDimensionMismatchError("dense index metadata disagrees with the manifest")
    return IndexBundle(path=path, manifest=manifest, chunks=chunks, bm25=bm25, dense=dense,
                       analyzer=make_analyzer(manifest.bm25_configuration))
