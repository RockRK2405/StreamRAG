"""Runs the Phase 10 experiments from experiments/configs/{systems,experiments}.yaml.

Usage:
  .venv/bin/python experiments/runners/run_experiments.py --index-root /tmp/idx10 [--only EXP01_baselines,...]
        [--rerun] [--no-llm] [--model qwen3:4b]

Variants with ``generation: llm`` need `ollama serve` with the model; without it (or with --no-llm) the experiments
that need them are recorded as NOT RUN (never with substitute numbers). Results: experiments/results/<experiment>/,
shared runs: experiments/results/runs/. Every config is written to <experiment>/config_record.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from corpora import CORPORA  # noqa: E402

from streamrag.evaluation.instrument import AnswerInstrument  # noqa: E402
from streamrag.evaluation.runner import ExperimentRunner  # noqa: E402
from streamrag.evaluation.systems import Stacks  # noqa: E402

CONF = REPO / "experiments" / "configs"


def instrument_factory(stacks: Stacks):
    from streamrag.claims.aligner import ClaimEvidenceAligner
    from streamrag.claims.decomposer import ClaimDecomposer, ClaimLexicon
    from streamrag.claims.verifier import ClaimVerifier
    cache: dict = {}

    def make(corpus: str) -> AnswerInstrument:
        if corpus not in cache:
            st = stacks.get(corpus, "extractive")
            an = st.bundle.analyzer
            terms = lambda t: list(dict.fromkeys(an.tokens(t)))  # noqa: E731
            v = ClaimVerifier(ClaimEvidenceAligner(terms, stacks.nli(), "nli"),
                              ClaimDecomposer(ClaimLexicon.load(REPO / "configs" / "claim_lexicon.yaml")))
            cache[corpus] = AnswerInstrument(v, {c.chunk_id: c for c in st.bundle.chunks}, terms)
        return cache[corpus]
    return make


def llm_info(url: str, model: str) -> dict:
    import urllib.request
    info = {"name": model, "backend": "ollama (local)", "url": url}
    try:
        req = urllib.request.Request(url + "/api/show", data=json.dumps({"model": model}).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=5) as r:
            d = json.load(r)
        info.update({"details": d.get("details"), "modified_at": d.get("modified_at")})
        with urllib.request.urlopen(url + "/api/version", timeout=5) as r:
            info["ollama_version"] = json.load(r).get("version")
        with urllib.request.urlopen(url + "/api/tags", timeout=5) as r:
            for m in json.load(r).get("models", []):
                if m.get("name") == model:
                    info["digest"] = m.get("digest")
    except OSError as exc:
        info["error"] = repr(exc)
    return info


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index-root", type=Path, required=True)
    ap.add_argument("--only", default=None)
    ap.add_argument("--rerun", action="store_true")
    ap.add_argument("--no-llm", action="store_true")
    ap.add_argument("--model", default="qwen3:4b")
    ap.add_argument("--dataset", default="streamrag_eval_v1", help="experiments/datasets/<name>")
    ap.add_argument("--results-dir", type=Path, default=REPO / "experiments" / "results")
    ap.add_argument("--experiments-file", type=Path, default=CONF / "experiments.yaml")
    ap.add_argument("--systems-file", type=Path, default=CONF / "systems.yaml")
    a = ap.parse_args()
    systems = yaml.safe_load(a.systems_file.read_text())
    exps = yaml.safe_load(a.experiments_file.read_text())
    url = "http://127.0.0.1:11434"
    llm, info = None, None
    if not a.no_llm:
        from streamrag.generation.llm import OllamaBackend
        if OllamaBackend.reachable(url, a.model):
            llm = OllamaBackend(url, a.model, temperature=0.0, seed=7)
            llm.complete([{"role": "user", "content": "Reply with {}"}], {"type": "object"})      # load the model
            info = llm_info(url, a.model)
    stacks = Stacks(REPO, CORPORA, a.index_root, llm=llm)
    runner = ExperimentRunner(REPO, systems, stacks, instrument_factory(stacks), llm=llm, llm_info=info,
                              results_dir=a.results_dir, dataset=a.dataset)
    only = a.only.split(",") if a.only else list(exps)
    summary = {}
    for exp_id in only:
        cfg = dict(exps[exp_id], experiment_id=exp_id)
        needs_llm = any(systems[s].get("generation") == "llm" for s in cfg["systems"])
        if needs_llm and llm is None:
            d = a.results_dir / exp_id
            d.mkdir(parents=True, exist_ok=True)
            res = {"experiment_id": exp_id, "status": "NOT RUN", "reason": "LLM variants need `ollama serve`"}
            (d / "results.json").write_text(json.dumps(res, indent=2))
            summary[exp_id] = "NOT RUN"
            print(exp_id, "NOT RUN (no LLM)", flush=True)
            continue
        res = runner.run_experiment(cfg, rerun=a.rerun)
        summary[exp_id] = res["status"]
        print(exp_id, res["status"], flush=True)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
