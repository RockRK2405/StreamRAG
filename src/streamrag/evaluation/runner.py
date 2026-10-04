"""ExperimentRunner (Phase 10; experiments/README.md).

load dataset -> load experiment config -> run each system variant (or reuse its stored run) -> collect events /
per-turn records -> score every turn with the metric modules and the common instrument -> aggregate (overall, by
query type, by difficulty) -> paired comparisons -> error categories + failure store -> results (JSON / CSV), config
record, log. A crash marks the experiment (or the system run) FAILED in its log and results - nothing is silently
omitted.

Runs are stored once per (system variant, split) under experiments/results/runs/ and shared by every experiment that
uses the same variant (e.g. the full system appears in experiments 1, 6, 7 and the ablation), so a variant is measured
once with one configuration and every table built from it agrees.
"""

from __future__ import annotations

import csv
import datetime as _dt
import json
import platform
import subprocess
import time
import traceback
from pathlib import Path

from streamrag.evaluation import errors as ERR
from streamrag.evaluation import stats as STATS
from streamrag.evaluation.dataset import EvalSample, file_hash, load, sessions
from streamrag.evaluation.metrics import citation as CI
from streamrag.evaluation.metrics import claims as CL
from streamrag.evaluation.metrics import efficiency as EF
from streamrag.evaluation.metrics import evidence as EV
from streamrag.evaluation.metrics import generation as GEN
from streamrag.evaluation.metrics import hallucination as HAL
from streamrag.evaluation.metrics import latency as LAT
from streamrag.evaluation.metrics import retrieval as RET

GROUPS = ("retrieval", "evidence", "claims", "generation", "citation", "hallucination", "latency", "efficiency",
          "streaming")


def git_commit(repo: Path) -> dict:
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=repo, capture_output=True,
                                    text=True).stdout.strip())
        return {"commit": head or None, "dirty": dirty}
    except OSError:
        return {"commit": None, "dirty": None}


class Logger:
    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)

    def __call__(self, event: str, **kw) -> None:
        rec = {"ts": _dt.datetime.now().isoformat(timespec="milliseconds"), "event": event, **kw}
        with self.path.open("a") as f:
            f.write(json.dumps(rec, default=str) + "\n")


class ExperimentRunner:
    def __init__(self, repo: Path, systems: dict[str, dict], stacks, instrument_factory, llm=None,
                 results_dir: Path | None = None, llm_info: dict | None = None) -> None:
        self.repo, self.systems, self.stacks = repo, systems, stacks
        self.instrument_factory = instrument_factory      # corpus -> AnswerInstrument
        self.llm, self.llm_info = llm, llm_info or {}
        self.results = results_dir or repo / "experiments" / "results"
        self.dataset_dir = repo / "experiments" / "datasets" / "streamrag_eval_v1"
        self._sessions: dict[str, dict[str, str]] = {}

    # ------------------------------------------------------------------ data
    def samples(self, split: str, types: list[str] | None = None, ids: list[str] | None = None) -> list[EvalSample]:
        rows = load(self.dataset_dir / f"{split}.jsonl")
        if types:
            keep_sessions = {r.session_id for r in rows if r.query_type in types}
            rows = [r for r in rows if r.session_id in keep_sessions]      # whole sessions (context turns run too)
        if ids:
            rows = [r for r in rows if r.sample_id in ids or r.session_id in ids]
        return rows

    # ------------------------------------------------------------------ runs (shared)
    def run_path(self, system: str, split: str) -> Path:
        return self.results / "runs" / f"{system}__{split}.jsonl"

    def ensure_run(self, system: str, split: str, log: Logger, rerun: bool = False) -> dict[str, dict]:
        """Raw per-turn records of a variant (run once, stored); scored rows are derived (and cached) separately."""
        path = self.run_path(system, split)
        if not (path.exists() and not rerun):
            self._run(system, split, log)
        else:
            log("run_reused", system=system, split=split, path=str(path.relative_to(self.repo)))
        return self.scored(system, split, log, rescore=rerun)

    def _run(self, system: str, split: str, log: Logger) -> None:
        from dataclasses import asdict
        spec = dict(self.systems[system], id=system)
        data = self.samples(split)
        if spec.get("query_types"):              # a variant measured on a subset only (whole sessions kept)
            keep = {x.session_id for x in data if x.query_type in spec["query_types"]}
            data = [x for x in data if x.session_id in keep]
        log("run_start", system=system, split=split, samples=len(data), spec=_public(spec))
        path = self.run_path(system, split)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".partial")
        n = failed = 0
        t0 = time.perf_counter()
        trace_dir = self.results / "runs" / "traces" if spec["family"] == "runtime" else None
        with tmp.open("w") as f:
            for sess in sessions(data):
                st = self.stacks.get(sess[0].corpus, spec.get("generation", "llm"))
                try:
                    recs = self._run_session(spec, st, sess, trace_dir)
                except Exception as exc:     # noqa: BLE001 - a crashed session is recorded, never dropped
                    log("session_failed", system=system, session=sess[0].session_id, error=repr(exc),
                        trace=traceback.format_exc()[-1500:])
                    from streamrag.evaluation.systems import TurnRecord
                    recs = [TurnRecord(s.sample_id, system, status="failed", error=f"{exc.__class__.__name__}: {exc}")
                            for s in sess]
                for r in recs:
                    n += 1
                    failed += r.status != "ok"
                    if r.status != "ok":
                        log("sample_failed", system=system, sample=r.sample_id, error=r.error)
                    f.write(json.dumps(asdict(r), default=str) + "\n")
        tmp.replace(path)
        meta = {"system": system, "split": split, "spec": _public(spec), "samples": n, "failed": failed,
                "wall_s": round(time.perf_counter() - t0, 1), "finished": _dt.datetime.now().isoformat()}
        path.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2, default=str))
        if path.with_suffix(".scored.jsonl").exists():
            path.with_suffix(".scored.jsonl").unlink()
        log("run_end", **meta)

    def scored(self, system: str, split: str, log: Logger, rescore: bool = False) -> dict[str, dict]:
        from streamrag.evaluation.systems import TurnRecord
        path = self.run_path(system, split)
        sp = path.with_suffix(".scored.jsonl")
        if sp.exists() and not rescore:
            return {json.loads(x)["sample_id"]: json.loads(x) for x in sp.read_text().splitlines() if x.strip()}
        samples = {s.sample_id: s for s in self.samples(split)}
        out = {}
        with sp.open("w") as f:
            for line in path.read_text().splitlines():
                if not line.strip():
                    continue
                r = TurnRecord(**json.loads(line))
                s = samples[r.sample_id]
                st = self.stacks.get(s.corpus, "extractive")
                row = score(s, r, self.instrument_factory(s.corpus), st)
                out[s.sample_id] = row
                f.write(json.dumps(row, default=str) + "\n")
        log("scored", system=system, split=split, rows=len(out))
        return out

    def _run_session(self, spec, st, sess, trace_dir):
        from streamrag.evaluation import systems as SYS
        fam = spec["family"]
        if fam == "free":
            return SYS.run_free(spec, st, sess, self.llm)            # always the LLM (naive / hybrid baselines)
        if fam == "pipeline":
            if spec.get("answer_mode") == "no_citation_validation":
                spec = {**spec, "_verifier": self.instrument_factory(sess[0].corpus).verifier}
            return SYS.run_pipeline(spec, st, sess, self.llm if spec.get("generation", "llm") == "llm" else None)
        llm = self.llm if spec.get("generation", "llm") == "llm" else None
        if fam == "runtime":
            faults_fn = None
            if spec.get("dense_delay_ms"):
                from streamrag.runtime import Fault, FaultInjector
                delay = spec["dense_delay_ms"]
                faults_fn = lambda: FaultInjector([Fault("dense", "delay", times=-1, delay_ms=delay)])  # noqa: E731
            return SYS.run_runtime(spec, st, sess, llm, trace_dir, faults_fn)
        raise ValueError(f"unknown family {fam}")

    # ------------------------------------------------------------------ experiments
    def run_experiment(self, cfg: dict, rerun: bool = False) -> dict:
        exp_id = cfg["experiment_id"]
        d = self.results / exp_id
        d.mkdir(parents=True, exist_ok=True)
        log = Logger(d / "log.jsonl")
        log("start", experiment_id=exp_id, title=cfg.get("title"))
        record = self.config_record(cfg)
        (d / "config_record.json").write_text(json.dumps(record, indent=2, default=str))
        log("configuration", **{k: record[k] for k in ("dataset_version", "systems", "git")})
        try:
            res = self._experiment(cfg, log, rerun)
            res["status"] = "COMPLETED"
        except Exception as exc:     # noqa: BLE001
            log("error", error=repr(exc), trace=traceback.format_exc()[-3000:])
            res = {"experiment_id": exp_id, "status": "FAILED", "error": repr(exc)}
        (d / "results.json").write_text(json.dumps(res, indent=2, default=str))
        log("end", status=res["status"])
        return res

    def _experiment(self, cfg: dict, log: Logger, rerun: bool) -> dict:
        exp_id = cfg["experiment_id"]
        d = self.results / exp_id
        out = {"experiment_id": exp_id, "title": cfg.get("title"), "question": cfg.get("question"),
               "splits": {}, "comparisons": {}}
        gate = cfg.get("latency_gate_ms")
        rows_all: list[dict] = []
        for split in cfg.get("splits", ["test"]):
            data = {s.sample_id: s for s in self.samples(split, cfg.get("query_types"), cfg.get("sample_ids"))}
            if not data:
                continue
            per_system = {}
            for system in cfg["systems"]:
                run = self.ensure_run(system, split, log, rerun)
                rows = []
                for sid, s in data.items():
                    if sid not in run:
                        continue
                    if cfg.get("scored_types") and s.query_type not in cfg["scored_types"]:
                        continue
                    row = dict(run[sid], experiment_id=exp_id)
                    row["error_categories"] = ERR.categorize(s, row, gate)
                    rows.append(row)
                per_system[system] = rows
                rows_all += rows
            out["splits"][split] = {"samples": len(data), "systems": {k: aggregate(v) for k, v in per_system.items()}}
            comps = {}
            for base, other in cfg.get("comparisons", []):
                if base not in per_system or other not in per_system:
                    continue
                a = {r["sample_id"]: r for r in per_system[base]}
                b = {r["sample_id"]: r for r in per_system[other]}
                comps[f"{other} vs {base}"] = {m: STATS.paired({k: _get(v, m) for k, v in a.items()},
                                                               {k: _get(v, m) for k, v in b.items()})
                                               for m in cfg.get("metrics", DEFAULT_METRICS)}
            out["comparisons"][split] = comps
        self._write_samples(d, rows_all)
        self._failures(exp_id, rows_all, log)
        return out

    def _write_samples(self, d: Path, rows: list[dict]) -> None:
        with (d / "samples.jsonl").open("w") as f:
            for r in rows:
                f.write(json.dumps(r, default=str) + "\n")
        flat = []
        for r in rows:
            x = {"experiment_id": r["experiment_id"], "sample_id": r["sample_id"], "system_variant": r["system_variant"],
                 "split": r["split"], "query_category": r["query_category"], "difficulty": r["difficulty"],
                 "status": r["status"], "error_categories": "|".join(r.get("error_categories", []))}
            for g in GROUPS:
                for k, v in (r["metrics"].get(g) or {}).items():
                    if isinstance(v, (int, float)) or v is None:
                        x[f"{g}.{k}"] = v
            flat.append(x)
        if flat:
            keys = list(dict.fromkeys(k for x in flat for k in x))
            with (d / "samples.csv").open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=keys)
                w.writeheader()
                w.writerows(flat)

    def _failures(self, exp_id: str, rows: list[dict], log: Logger) -> None:
        store = self.results.parent / "failure_cases" / f"{exp_id}.jsonl"      # experiments/failure_cases by default
        store.parent.mkdir(parents=True, exist_ok=True)
        n = 0
        with store.open("w") as f:
            for r in rows:
                for cat in r.get("error_categories", []):
                    f.write(json.dumps({"experiment_id": exp_id, "sample_id": r["sample_id"],
                                        "system_variant": r["system_variant"], "failure_category": cat,
                                        "query": r["query"], "expected": r["expected"], "actual": r["answer"],
                                        "retrieved_evidence": r["evidence_citations"], "claims": r["judged_claims"],
                                        "answer": r["answer"], "citations": r["citations"],
                                        "trace": r.get("trace") or self._trace_path(r)}, default=str) + "\n")
                    n += 1
        log("failure_cases", path=str(store), records=n)

    def _trace_path(self, r: dict) -> str | None:
        """Runtime event trace of the turn's session (runs/traces/<variant>__<session>.jsonl), if one was written."""
        if r["split"] not in self._sessions:
            self._sessions[r["split"]] = {s.sample_id: s.session_id for s in self.samples(r["split"])}
        sid = self._sessions[r["split"]].get(r["sample_id"])
        p = self.results / "runs" / "traces" / f"{r['system_variant']}__{(sid or '').replace('.', '-')}.jsonl"
        return str(p.relative_to(self.repo)) if sid and p.exists() else None

    def config_record(self, cfg: dict) -> dict:
        man = json.loads((self.dataset_dir / "manifest.json").read_text())
        st = self.stacks.get("transit", "llm" if self.llm is not None else "extractive")
        g = st.cfg.generation
        systems = {s: _public(dict(self.systems[s], id=s)) for s in cfg["systems"]}
        return {
            "experiment_id": cfg["experiment_id"], "timestamp": _dt.datetime.now().isoformat(timespec="seconds"),
            "title": cfg.get("title"), "question": cfg.get("question"), "config": cfg,
            "dataset_version": {"name": man["version"], **{k: v["sha256_16"] for k, v in man["files"].items()}},
            "model": {"llm": self.llm_info or "extractive (no LLM)", "temperature": g.temperature, "seed": g.seed,
                      "max_tokens": g.max_output_tokens, "num_ctx": g.num_ctx},
            "embedding_model": st.bundle.dense.info.name if st.bundle.dense else None,
            "reranker": st.cfg.rerank.model if st.service.reranker is not None else None,
            "verifier_instrument": "nli-deberta-v3-xsmall + rules (Phase 7 ClaimVerifier)",
            "top_k": st.cfg.multi_intent.max_candidates_per_intent, "systems": systems,
            "runtime_config": st.cfg.runtime.model_dump(mode="json"),
            "adaptive_retrieval_config": st.cfg.adaptive_retrieval.model_dump(mode="json"),
            "git": git_commit(self.repo), "python": platform.python_version(), "machine": platform.platform(),
            "stats_seed": STATS.SEED}


DEFAULT_METRICS = ("retrieval.recall@5", "retrieval.mrr", "generation.answer_correct", "generation.completeness",
                   "claims.claim_support_rate", "hallucination.unsupported_claim_rate", "citation.citation_precision",
                   "latency.ttva_after_end", "efficiency.retrieval_calls")


def _get(row: dict, dotted: str):
    g, k = dotted.split(".", 1)
    return (row["metrics"].get(g) or {}).get(k)


def _public(spec: dict) -> dict:
    return {k: v for k, v in spec.items() if not k.startswith("_")}


# ------------------------------------------------------------------------------------------------ scoring
def score(s: EvalSample, r, instrument, st) -> dict:
    judged = instrument.judge(r.claims, [e["chunk_id"] for e in r.evidence]) if r.claims else []
    fact = [c for c in judged if c["factual"]]
    vectors = None
    if st.bundle.dense is not None and len(r.evidence) >= 2:
        rows = {c.chunk_id: i for i, c in enumerate(st.bundle.chunks)}
        vectors = [st.bundle.dense.matrix[rows[e["chunk_id"]]] for e in r.evidence if e["chunk_id"] in rows]
    ev_text = " ".join(e["text"] for e in r.evidence) + " " + " ".join(instrument.text_of(c) for _, cs in r.claims
                                                                       for c in cs)
    metrics = {
        "retrieval": RET.compute(r.keys, s.ground_truth_evidence, s.gold_semantics),
        "evidence": EV.compute(r.evidence, s.ground_truth_evidence, s.expected_claims, s.forbidden, s.query_type,
                               vectors),
        "claims": CL.compute([c["verdict"] for c in fact], r.answer, s.expected_claims),
        "generation": GEN.compute(r.answer, s.expected_claims, s.forbidden, s.expected_state, s.conflict_values,
                                  [c["verdict"] for c in fact], [c["cited"] for c in fact]),
        "citation": CI.compute(judged, s.expected_claims),
        "hallucination": HAL.compute(judged, ev_text, s.query),
        "latency": dict(r.latency),
        "efficiency": EF.normalize(r.ops),
        "streaming": {k: v for k, v in r.stream.items() if k != "update_gaps_ms"} | (
            {"mean_update_gap_ms": (sum(r.stream["update_gaps_ms"]) / len(r.stream["update_gaps_ms"]))
             if r.stream.get("update_gaps_ms") else None} if r.stream else {}) | dict(r.session_stream),
    }
    return {"sample_id": s.sample_id, "system_variant": r.system, "split": s.split, "query_category": s.query_type,
            "difficulty": s.difficulty, "corpus": s.corpus, "status": r.status, "error": r.error, "query": s.query,
            "expected": {"answer": s.expected_answer, "claims": [c.model_dump() for c in s.expected_claims],
                         "evidence": s.ground_truth_evidence, "state": s.expected_state},
            "answer": r.answer, "evidence_citations": r.keys, "judged_claims": judged,
            "citations": [c["key"] for cl in judged for c in cl["citations"]], "metrics": metrics,
            "ops_raw": r.ops, "adaptive": r.adaptive, "resources": r.resources, "waste": r.events_digest,
            "trace": r.trace}


# ------------------------------------------------------------------------------------------------ aggregation
def aggregate(rows: list[dict]) -> dict:
    out: dict = {"samples": len(rows), "failed": sum(1 for r in rows if r["status"] != "ok"),
                 "failure_rate": (sum(1 for r in rows if r["status"] != "ok") / len(rows)) if rows else None}
    for g in GROUPS:
        keys = list(dict.fromkeys(k for r in rows for k, v in (r["metrics"].get(g) or {}).items()
                                  if isinstance(v, (int, float)) and not isinstance(v, bool) or v is None))
        agg = {}
        for k in keys:
            vals = [r["metrics"][g].get(k) for r in rows if isinstance(r["metrics"][g].get(k), (int, float))]
            if g in ("latency",) or k.endswith("_ms"):
                agg[k] = LAT.percentiles(vals)
            else:
                agg[k] = STATS.describe(vals)
        out[g] = agg
    out["by_category"] = _by(rows, "query_category")
    out["by_difficulty"] = _by(rows, "difficulty")
    strat: dict[str, int] = {}
    for r in rows:
        for x in (r.get("adaptive") or {}).get("strategies", []) or []:
            if x:
                strat[x] = strat.get(x, 0) + 1
    tot = sum(strat.values())
    out["adaptivity"] = {"strategy_share": {k: round(v / tot, 4) for k, v in sorted(strat.items())} if tot else {},
                         "needs_planned": tot}
    cats: dict[str, int] = {}
    for r in rows:
        for c in r.get("error_categories", []):
            cats[c] = cats.get(c, 0) + 1
    out["error_categories"] = cats
    hits = [r["metrics"]["efficiency"].get("cache_hit") for r in rows
            if r["metrics"]["efficiency"].get("cache_hit") is not None]
    out["cache_hit_rate"] = (sum(hits) / len(hits)) if hits else None
    return out


KEY_METRICS = ("retrieval.recall@5", "retrieval.mrr", "evidence.evidence_precision", "generation.answer_correct",
               "generation.completeness", "claims.claim_support_rate", "citation.citation_precision",
               "latency.ttva_after_end", "efficiency.retrieval_calls")


def _by(rows: list[dict], field: str) -> dict:
    groups: dict[str, list[dict]] = {}
    for r in rows:
        groups.setdefault(r[field], []).append(r)
    out = {}
    for k, rs in sorted(groups.items()):
        out[k] = {"samples": len(rs)}
        for m in KEY_METRICS:
            vals = [_get(r, m) for r in rs if isinstance(_get(r, m), (int, float))]
            out[k][m] = round(sum(vals) / len(vals), 4) if vals else None
    return out
