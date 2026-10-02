"""Benchmark harness: evaluation data -> per-mode metrics JSON + per-query JSONL/CSV + run manifest.

Modes: ``bm25`` (Exp A), ``dense`` (Exp B), ``hybrid`` (Exp C, RRF), ``hybrid_rerank`` (Exp D). The same index,
same queries and same top_k are used for every mode, so only the retrieval strategy varies.
"""

from __future__ import annotations

import csv
import json
import statistics
import time
from collections import defaultdict
from pathlib import Path

from streamrag.bench.dataset import dataset_sha256, load_eval_items
from streamrag.bench.metrics import item_key, mrr_at_k, ndcg_at_k, recall_at_k, success_at_k
from streamrag.config.settings import StreamRagConfig, config_hash
from streamrag.models.retrieval import RetrievalOptions
from streamrag.provenance import environment
from streamrag.retrieval.embedders import load_embedder
from streamrag.retrieval.rerank import load_reranker
from streamrag.retrieval.service import RetrievalService
from streamrag.retrieval.store import load_index, resolve_index
from streamrag.telemetry.timing import Stopwatch, percentile_summary

KS = (1, 5, 10)
MODES = {"bm25": ("bm25", False), "dense": ("dense", False), "hybrid": ("hybrid", False),
         "hybrid_rerank": ("hybrid", True)}
NOT_REPORTABLE_BANNER = "TEST FIXTURE / NON-OFFICIAL DATA - NOT A BENCHMARK RESULT - DO NOT REPORT"


def _mean(xs: list[float]) -> float:
    return round(statistics.fmean(xs), 4) if xs else 0.0


def run_benchmark(cfg: StreamRagConfig, eval_path: Path, modes: list[str], out_dir: Path,
                  index_path: Path | None = None, command: str = "") -> dict:
    unknown = [m for m in modes if m not in MODES]
    if unknown:
        raise ValueError(f"unknown modes {unknown}; choose from {sorted(MODES)}")
    items = load_eval_items(eval_path)
    with Stopwatch() as sw:
        bundle = load_index(index_path or resolve_index(cfg))
    index_load_ms = sw.ms
    embedder = None
    if bundle.dense is not None and any(MODES[m][0] in ("dense", "hybrid") for m in modes):
        embedder = load_embedder(bundle.dense.info.name, cfg.paths.model_registry, cfg.paths.models_dir,
                                 bundle.analyzer, cfg.dense.intra_op_threads, cfg.dense.batch_size)
    reranker = None
    if any(MODES[m][1] for m in modes):
        reranker = load_reranker(cfg.rerank.model, cfg.paths.model_registry, cfg.paths.models_dir,
                                 cfg.dense.intra_op_threads, cfg.rerank.batch_size)
    svc = RetrievalService(bundle, cfg, embedder, reranker)

    fixture_data = any(i.is_fixture for i in items)
    reportable = not (bundle.manifest.is_test_fixture or fixture_data)
    reason = None if reportable else ("index built from a TEST FIXTURE corpus" if bundle.manifest.is_test_fixture
                                      else "evaluation items flagged is_fixture")
    max_k = max(KS)
    out_dir.mkdir(parents=True, exist_ok=True)
    per_query_rows: list[dict] = []
    summary: dict[str, dict] = {}
    for mode in modes:
        m_mode, m_rerank = MODES[mode]
        opts = RetrievalOptions(mode=m_mode, top_k=max_k, rerank=m_rerank)
        svc.retrieve(items[0].query, opts)                     # warm-up (excluded from latency stats)
        per_metric: dict[str, list[float]] = defaultdict(list)
        stage_lat: dict[str, list[float]] = defaultdict(list)
        by_category: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
        case_cover: dict[str, list[bool]] = defaultdict(list)
        statuses: dict[str, int] = defaultdict(int)
        for it in items:
            es = svc.retrieve(it.query, opts)
            keys = [item_key(e, it.gold_level) for e in es.items]
            gold = set(it.gold)
            row = {"mode": mode, "item_id": it.item_id, "query": it.query, "category": it.category,
                   "case_id": it.case_id, "intent_id": it.intent_id, "status": es.trace.status,
                   "returned": keys, "gold": sorted(gold)}
            for k in KS:
                row[f"recall@{k}"] = recall_at_k(keys, gold, k)
                row[f"success@{k}"] = success_at_k(keys, gold, k)
            row["mrr@10"] = mrr_at_k(keys, gold, 10)
            row["ndcg@10"] = ndcg_at_k(keys, gold, 10)
            for name, v in row.items():
                if "@" in name:
                    per_metric[name].append(v)
                    if it.category:
                        by_category[it.category][name].append(v)
            for stage, ms in es.trace.timings_ms.items():
                stage_lat[stage].append(ms)
                row[f"ms_{stage}"] = ms
            if it.case_id:
                case_cover[it.case_id].append(row["success@5"] == 1.0)
            statuses[es.trace.status] += 1
            per_query_rows.append(row)
        summary[mode] = {
            "n_items": len(items),
            "metrics": {k: _mean(v) for k, v in sorted(per_metric.items())},
            "per_category": {c: {k: _mean(v) for k, v in sorted(d.items())} for c, d in sorted(by_category.items())},
            "all_intents_covered@5": _mean([1.0 if all(v) else 0.0 for v in case_cover.values()]) if case_cover else None,
            "latency_ms": {s: percentile_summary(v) for s, v in sorted(stage_lat.items())},
            "statuses": dict(statuses),
        }

    metrics = {"REPORTABLE": reportable, "not_reportable_reason": reason,
               "banner": None if reportable else NOT_REPORTABLE_BANNER,
               "index_load_ms": round(index_load_ms, 3), "modes": summary}
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True))
    with (out_dir / "per_query.jsonl").open("w", encoding="utf-8") as f:
        for r in per_query_rows:
            f.write(json.dumps(r, sort_keys=True, ensure_ascii=False) + "\n")
    fields = sorted({k for r in per_query_rows for k in r if not isinstance(r[k], list)})
    with (out_dir / "per_query.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(per_query_rows)
    manifest = {
        "run_id": out_dir.name, "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "command": command,
        "REPORTABLE": reportable, "not_reportable_reason": reason, "modes": modes, "ks": list(KS),
        "config_hash": config_hash(cfg), "index_path": str(bundle.path),
        "index_version": bundle.manifest.index_version, "index_config_hash": bundle.manifest.index_config_hash,
        "index_content_hash": bundle.manifest.content_hash, "corpus_hash": bundle.manifest.corpus_version,
        "corpus_is_test_fixture": bundle.manifest.is_test_fixture,
        "embedding_model": bundle.manifest.embedding_model.model_dump(mode="json") if bundle.manifest.embedding_model else None,
        "reranker": ({"name": cfg.rerank.model} if reranker else None),
        "eval_data": {"path": str(eval_path), "sha256": dataset_sha256(Path(eval_path)), "n_items": len(items),
                      "fixture_items": sum(1 for i in items if i.is_fixture)},
        "environment": environment(),
    }
    (out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str))
    svc.close()
    return metrics
