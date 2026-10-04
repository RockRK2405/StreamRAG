"""Shared harness for the Phase 9 measurements (fixture domain; NOT REPORTABLE).

Every system runs through the same Phase 5-7 session pipeline (``AdaptivePipeline``: intent decomposition, delta
planning, evidence lifecycle, extractive grounded answer + NLI verification); only the retrieval policy differs.
Retrieval is real (bge-small ONNX + BM25 [+ ms-marco cross-encoder]); generation is the extractive backend (no LLM,
so generation costs are equal across systems and LLM calls are 0 everywhere - the real LLM is used only in the final
end-to-end test). Timings are wall-clock / process-CPU measurements on this machine, one run per case.
"""

from __future__ import annotations

import json
import math
import os
import platform
import statistics
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "results"

from streamrag.answer_state.resources import GroundingResources  # noqa: E402
from streamrag.citations.mapper import ChunkCatalog  # noqa: E402
from streamrag.claims.nli import NliModel  # noqa: E402
from streamrag.config import load_config  # noqa: E402
from streamrag.retrieval import build_index  # noqa: E402
from streamrag.session.pipeline import AdaptivePipeline  # noqa: E402
from streamrag.streaming.factory import build_stack  # noqa: E402

CORPORA = {"adaptive": "tests/fixtures/corpus_adaptive", "fixture": "tests/fixtures/corpus",
           "grounding": "tests/fixtures/corpus_grounding", "conflict": "tests/fixtures/corpus_conflict",
           "injection": "tests/fixtures/corpus_injection"}
BANNER = "DEV FIXTURE DOMAIN - IMPLEMENTER-LABELLED - NOT AN OFFICIAL BENCHMARK RESULT - NOT REPORTABLE"
REFERENCE_DATE = "2026-10-03"          # fixed "today" for validity checks (reproducible runs)
K = 5                                   # fixed k of the baselines = the Phase 5-8 pipeline default per need

ADAPTIVE_OFF = {"adaptive_retrieval.enabled": False}
SYSTEMS = {
    # brief §47 baselines (fixed k = 5 per need, the pipeline default)
    "A_vector_only": {**ADAPTIVE_OFF, "streaming.retrieval_mode": "dense", "streaming.rerank": False},
    "B_bm25_only": {**ADAPTIVE_OFF, "streaming.retrieval_mode": "bm25", "streaming.rerank": False},
    "C_fixed_hybrid": {**ADAPTIVE_OFF, "streaming.retrieval_mode": "hybrid", "streaming.rerank": False},
    "D_hybrid_rerank": {**ADAPTIVE_OFF, "streaming.retrieval_mode": "hybrid", "streaming.rerank": True},
    "ADAPTIVE": {"adaptive_retrieval.enabled": True},
    # reranking decision measured separately (brief §28-29): the adaptive policy with rerank="policy"
    "ADAPTIVE_rerank_policy": {"adaptive_retrieval.enabled": True, "adaptive_retrieval.rerank": "policy"},
}
_OFF = {"adaptive_retrieval.adaptive_k": False, "adaptive_retrieval.iterative": False,
        "adaptive_retrieval.claim_driven": False, "adaptive_retrieval.cache": False,
        "adaptive_retrieval.session_reuse": False, "adaptive_retrieval.contradiction_retrieval": False,
        "adaptive_retrieval.multi_hop": False, "adaptive_retrieval.temporal": False,
        "adaptive_retrieval.expansion": False}
ABLATIONS = {   # brief §51, cumulative: each row adds one component to the previous one
    "AB_A_fixed": SYSTEMS["C_fixed_hybrid"],
    "AB_B_strategy": {"adaptive_retrieval.enabled": True, **_OFF},
    "AB_C_adaptive_k": {"adaptive_retrieval.enabled": True, **_OFF, "adaptive_retrieval.adaptive_k": True,
                        "adaptive_retrieval.iterative": True},
    "AB_D_claim_driven": {"adaptive_retrieval.enabled": True, **_OFF, "adaptive_retrieval.adaptive_k": True,
                          "adaptive_retrieval.iterative": True, "adaptive_retrieval.claim_driven": True},
    "AB_E_cache": {"adaptive_retrieval.enabled": True, **_OFF, "adaptive_retrieval.adaptive_k": True,
                   "adaptive_retrieval.iterative": True, "adaptive_retrieval.claim_driven": True,
                   "adaptive_retrieval.cache": True, "adaptive_retrieval.session_reuse": True},
    "AB_F_contradiction": {"adaptive_retrieval.enabled": True, **_OFF, "adaptive_retrieval.adaptive_k": True,
                           "adaptive_retrieval.iterative": True, "adaptive_retrieval.claim_driven": True,
                           "adaptive_retrieval.cache": True, "adaptive_retrieval.session_reuse": True,
                           "adaptive_retrieval.contradiction_retrieval": True},
    "AB_G_full": {"adaptive_retrieval.enabled": True},
}

_STACKS: dict = {}
_NLI = None


def nli():
    global _NLI
    if _NLI is None:
        _NLI = NliModel.load(REPO / "models", "nli-deberta-v3-xsmall")
    return _NLI


def stack(corpus: str, index_root: Path, rerank: bool = True, llm=None):
    """Phase 5-7 stack over a fixture corpus (cached). The reranker is loaded so D / rerank policies can use it.
    ``llm``: a local LLM backend for generation (else the extractive generator)."""
    key = (corpus, llm is not None)
    if key in _STACKS:
        return _STACKS[key]
    ov = {"paths.corpus": str(REPO / CORPORA[corpus]), "paths.index_root": str(index_root / corpus),
          "telemetry.log_level": "ERROR", "multi_intent.enabled": True, "session.enabled": True,
          "generation.enabled": True, "generation.backend": "ollama" if llm is not None else "extractive",
          "streaming.rerank": rerank, "adaptive_retrieval.reference_date": REFERENCE_DATE}
    cfg = load_config(REPO / "configs" / "default.yaml", ov, base_dir=REPO)
    st = build_stack(cfg, build_index(cfg).path)
    an = st.bundle.analyzer
    st._grounding = GroundingResources(ChunkCatalog.from_bundle(st.bundle), nli(), llm,
                                       lambda t: list(dict.fromkeys(an.tokens(t))))
    _STACKS[key] = st
    return st


def with_cfg(st, **dotted):
    cfg = st.cfg
    for key, val in dotted.items():
        parts = key.split(".")
        objs = [cfg]
        for p in parts[:-1]:
            objs.append(getattr(objs[-1], p))
        new = objs[-1].model_copy(update={parts[-1]: val})
        for obj, p in zip(reversed(objs[:-1]), reversed(parts[:-1])):
            new = obj.model_copy(update={p: new})
        cfg = new
    s2 = st.with_config(cfg)
    s2._intent_stack = st.intent_stack
    s2._grounding = st.grounding
    return s2


EVAL_DIR = "dev_adaptive_retrieval"


def cases(prefix: str | None = None, eval_dir: str | None = None) -> list[dict]:
    d = REPO / "eval" / (eval_dir or EVAL_DIR)
    out = [json.loads(p.read_text()) for p in sorted(d.glob("*.json"))]
    return [c for c in out if prefix is None or c["case_id"].startswith(prefix)]


# ---------------------------------------------------------------------------------------------------- metrics
def recall(keys, gold, k=None):
    ks = keys[:k] if k else keys
    return len(set(gold) & set(ks)) / len(set(gold)) if gold else None


def precision(keys, gold, k):
    return len(set(gold) & set(keys[:k])) / k if gold else None


def returned_precision(keys, gold):
    return (len([x for x in keys if x in set(gold)]) / len(keys)) if gold and keys else (0.0 if gold else None)


def mrr(keys, gold, k=10):
    for r, x in enumerate(keys[:k], start=1):
        if x in set(gold):
            return 1.0 / r
    return 0.0 if gold else None


def ndcg(keys, gold, k=10):
    if not gold:
        return None
    seen, dcg = set(), 0.0
    for r, x in enumerate(keys[:k], start=1):
        if x in set(gold) and x not in seen:
            dcg += 1.0 / math.log2(r + 1)
            seen.add(x)
    ideal = sum(1.0 / math.log2(r + 1) for r in range(1, min(len(set(gold)), k) + 1))
    return dcg / ideal


def pct(values) -> dict:
    v = sorted(x for x in values if x is not None)
    if not v:
        return {"n": 0, "p50": None, "p95": None, "mean": None, "max": None}

    def q(p):
        k = (len(v) - 1) * p
        lo, hi = int(k), min(int(k) + 1, len(v) - 1)
        return v[lo] + (v[hi] - v[lo]) * (k - lo)
    return {"n": len(v), "p50": round(q(0.5), 3), "p95": round(q(0.95), 3), "mean": round(statistics.fmean(v), 3),
            "max": round(v[-1], 3)}


def mean(values):
    v = [x for x in values if x is not None]
    return round(statistics.fmean(v), 4) if v else None


def redundancy(st, chunk_ids: list[str]) -> float | None:
    """Mean pairwise cosine similarity of the returned chunks (bge-small vectors of the index)."""
    d = st.bundle.dense
    if d is None or len(chunk_ids) < 2:
        return None
    rows = {c.chunk_id: i for i, c in enumerate(st.bundle.chunks)}
    vecs = [d.matrix[rows[c]] for c in chunk_ids if c in rows]
    sims = [float(vecs[i] @ vecs[j]) for i in range(len(vecs)) for j in range(i + 1, len(vecs))]
    return round(statistics.fmean(sims), 4) if sims else None


# ---------------------------------------------------------------------------------------------------- runner
def turn_evidence(p: AdaptivePipeline, res, st) -> tuple[list[str], list[str]]:
    """Citations / chunk ids of the evidence the turn's needs end with (round-robin over the needs)."""
    cite = {c.chunk_id: c.citation for c in st.bundle.chunks}
    needs = []
    for ch in res.changes:
        for i in list(ch.new_intents) + list(ch.affected_intents):
            it = p.tracker.intents.get(i)
            if it is not None and it.status == "ACTIVE" and i not in needs:
                needs.append(i)
    lists = []
    for i in needs:
        rec = p.ledger.active_for_intent(i)
        if rec is not None and rec.evidence_ids:
            lists.append(list(rec.evidence_ids))
    ids: list[str] = []
    for n in range(max((len(x) for x in lists), default=0)):
        for x in lists:
            if n < len(x) and x[n] not in ids:
                ids.append(x[n])
    keys: list[str] = []
    for e in ids:
        c = cite.get(e, e)
        if c not in keys:
            keys.append(c)
    return keys, ids


def run_case(st, case: dict, overrides: dict, generation: bool = True) -> list[dict]:
    s2 = with_cfg(st, **overrides, **({} if generation else {"generation.enabled": False}))
    p = AdaptivePipeline(s2, session_id=case["case_id"].lower())
    cpu = {"ms": 0.0}
    orig = p._execute

    def timed(*a, **k):
        c0 = time.process_time()
        try:
            return orig(*a, **k)
        finally:
            cpu["ms"] += (time.process_time() - c0) * 1000.0
    p._execute = timed
    rows = []
    try:
        for n, turn in enumerate(case["turns"], start=1):
            cpu["ms"] = 0.0
            t0 = time.perf_counter()
            res = p.process(turn["utterance_id"], turn["utterance_text"], n * 5000.0)
            wall = (time.perf_counter() - t0) * 1000.0
            keys, ids = turn_evidence(p, res, st)
            gold = turn["gold"]["citations"]
            mode = s2.cfg.streaming.retrieval_mode
            ad = res.adaptive
            if ad:
                ops = {"searches": sum(r.ops.searches for r in ad),
                       "lexical_searches": sum(r.ops.lexical_searches for r in ad),
                       "embeddings": sum(r.ops.dense_searches for r in ad),
                       "reranker_calls": sum(r.ops.reranker_calls for r in ad),
                       "chunks_retrieved": sum(r.ops.chunks_returned for r in ad),
                       "cache_hits": sum(r.ops.cache_hits for r in ad), "session_reuse": sum(r.ops.session_reuse
                                                                                             for r in ad),
                       "k_values": [s["k"] for r in ad for s in r.searches]}
            else:
                n_q = res.retrievals
                created = [r for r in p.ledger.for_utterance(turn["utterance_id"]) if r.status == "completed"
                           and r.retrieval_status != "cache_hit"]
                ops = {"searches": n_q, "lexical_searches": n_q if mode in ("bm25", "hybrid") else 0,
                       "embeddings": n_q if mode in ("dense", "hybrid") else 0,
                       "reranker_calls": n_q if s2.cfg.streaming.rerank else 0,
                       "chunks_retrieved": sum(len(r.evidence_ids) for r in created), "cache_hits": 0,
                       "session_reuse": 0, "k_values": [s2.cfg.multi_intent.max_candidates_per_intent] * n_q}
            ops["phase6_cache_hits"] = res.cache_hits - (ops["cache_hits"] + ops["session_reuse"] if ad else 0)
            ga = res.grounded
            g = {}
            if ga is not None:
                m = ga.metrics or {}
                cited_keys = set()
                cite_of = {c.chunk_id: c.citation for c in st.bundle.chunks}
                for c in ga.citations.citations:
                    if c.status == "valid":
                        cited_keys.add(cite_of.get(c.evidence_id, c.evidence_id))
                g = {"claim_support": m.get("raw_support_rate"), "unsupported": m.get("raw_unsupported_rate"),
                     "citation_precision": m.get("citation_precision"), "final_claims": m.get("final_claims"),
                     "gold_cited": (len(set(gold) & cited_keys) / len(set(gold))) if gold else None,
                     "reports_gap": any(s.uncertainty for s in ga.sections),
                     "reports_conflict": any(c.kind == "conflict" for c in ga.claims),
                     "llm_calls": ga.llm_calls, "backend": ga.backend, "fallback": ga.fallback,
                     "answer": ga.text[:300]}
            rows.append({
                "case_id": case["case_id"], "category": case["category"], "turn": n, "corpus": case["corpus"],
                "expect": turn["gold"]["expect"], "gold": gold, "returned": keys, "n_returned": len(keys),
                "recall@5": recall(keys, gold, 5), "recall@10": recall(keys, gold, 10),
                "coverage": recall(keys, gold), "precision@5": precision(keys, gold, 5),
                "returned_precision": returned_precision(keys, gold), "mrr@10": mrr(keys, gold),
                "ndcg@10": ndcg(keys, gold), "source_diversity": (len({c.split(" §")[0] for c in keys}) / len(keys))
                if keys else None, "redundancy": redundancy(st, ids),
                "retrieval_ms": res.timings_ms.get("retrieval"), "retrieval_cpu_ms": round(cpu["ms"], 3),
                "turn_ms": round(wall, 3), "ops": ops,
                "assessments": [r.assessment.status for r in ad], "stops": [r.state.stop_reason.value for r in ad],
                "strategies": [r.plan.strategy.value for r in ad],
                "complexity": [r.analysis.complexity.value for r in ad], **g})
    finally:
        p.close()
    return rows


def meta(**extra) -> dict:
    return {"REPORTABLE": False, "banner": BANNER, "machine": platform.platform(), "python": platform.python_version(),
            "cpu_count": os.cpu_count(), "embedder": "bge-small-en-v1.5 (ONNX)", "reranker": "ms-marco-minilm-l6-v2",
            "verifier": "nli-deberta-v3-xsmall", "generation": "extractive (no LLM)", "reference_date": REFERENCE_DATE,
            "eval_set": "eval/dev_adaptive_retrieval (implementer-labelled, fixture corpora)", **extra}


def write(name: str, obj) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / name
    p.write_text(json.dumps(obj, indent=2, default=str))
    return p


def summarize(rows: list[dict]) -> dict:
    g = [r for r in rows if r["gold"]]
    ks = [k for r in rows for k in r["ops"]["k_values"]]
    lat = [r["retrieval_ms"] for r in rows if r["retrieval_ms"] is not None]
    return {
        "turns": len(rows), "turns_with_gold": len(g),
        "recall@5": mean(r["recall@5"] for r in g), "recall@10": mean(r["recall@10"] for r in g),
        "coverage": mean(r["coverage"] for r in g), "precision@5": mean(r["precision@5"] for r in g),
        "returned_precision": mean(r["returned_precision"] for r in g), "mrr@10": mean(r["mrr@10"] for r in g),
        "ndcg@10": mean(r["ndcg@10"] for r in g),
        "claim_support": mean(r.get("claim_support") for r in rows),
        "unsupported_rate": mean(r.get("unsupported") for r in rows),
        "citation_precision": mean(r.get("citation_precision") for r in rows),
        "gold_cited": mean(r.get("gold_cited") for r in g),
        "insufficient_reported": mean(1.0 if r.get("reports_gap") else 0.0 for r in rows
                                      if r["expect"] == "INSUFFICIENT" and "reports_gap" in r),
        "conflict_reported": mean(1.0 if r.get("reports_conflict") else 0.0 for r in rows
                                  if r["expect"] == "CONTRADICTORY" and "reports_conflict" in r),
        "source_diversity": mean(r["source_diversity"] for r in rows),
        "redundancy": mean(r["redundancy"] for r in rows),
        "searches_total": sum(r["ops"]["searches"] for r in rows),
        "searches_per_turn": mean(r["ops"]["searches"] for r in rows),
        "embeddings_total": sum(r["ops"]["embeddings"] for r in rows),
        "lexical_searches_total": sum(r["ops"]["lexical_searches"] for r in rows),
        "reranker_calls_total": sum(r["ops"]["reranker_calls"] for r in rows),
        "chunks_retrieved_total": sum(r["ops"]["chunks_retrieved"] for r in rows),
        "avg_k": round(statistics.fmean(ks), 3) if ks else None,
        "cache_hits": sum(r["ops"]["cache_hits"] + r["ops"]["session_reuse"] for r in rows),
        "phase6_cache_hits": sum(r["ops"]["phase6_cache_hits"] for r in rows),
        "llm_calls": sum(r.get("llm_calls") or 0 for r in rows),
        "retrieval_latency_ms": pct(lat), "retrieval_cpu_ms": pct(r["retrieval_cpu_ms"] for r in rows),
        "turn_latency_ms": pct(r["turn_ms"] for r in rows),
    }
