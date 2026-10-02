"""Chunking experiment: compare candidate chunking configurations on structural statistics.

Datasets (NONE is the official Theme 4 corpus — it is unavailable):
  * fixture   — tests/fixtures/corpus (fictional, tiny)
  * synthetic — synth.make_corpus (structure-only, generated words)
  * pdf_smoke — other hackathon theme guide PDFs from the user's Downloads, copied into a scratch dir with a
                TEST_FIXTURE_ONLY marker; used only for real-world PDF layout statistics (no content is reported)

Usage: .venv/bin/python research/phase3/chunking_experiment.py --pdf-dir <dir with PDFs> --scratch <dir>
"""

from __future__ import annotations

import argparse
import json
import shutil
import statistics
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))

from streamrag.config import load_config  # noqa: E402
from streamrag.corpus import build_corpus  # noqa: E402
from streamrag.corpus.chunker import sentence_spans  # noqa: E402

CONFIGS = {
    "section(max300)":        {"strategy": "section", "target_tokens": 180, "max_tokens": 300, "overlap_tokens": 0},
    "para(120/200,ov0)":      {"strategy": "paragraph", "target_tokens": 120, "max_tokens": 200, "overlap_tokens": 0},
    "para(120/200,ov40)":     {"strategy": "paragraph", "target_tokens": 120, "max_tokens": 200, "overlap_tokens": 40},
    "para(180/300,ov0)":      {"strategy": "paragraph", "target_tokens": 180, "max_tokens": 300, "overlap_tokens": 0},
    "para(180/300,ov40)*":    {"strategy": "paragraph", "target_tokens": 180, "max_tokens": 300, "overlap_tokens": 40},
    "para(250/350,ov40)":     {"strategy": "paragraph", "target_tokens": 250, "max_tokens": 350, "overlap_tokens": 40},
    "token(128,ov32)":        {"strategy": "token", "target_tokens": 128, "max_tokens": 128, "overlap_tokens": 32},
    "token(256,ov64)":        {"strategy": "token", "target_tokens": 256, "max_tokens": 256, "overlap_tokens": 64},
}


def wordpiece_counter():
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(REPO / "models" / "bge-small-en-v1.5" / "tokenizer.json"))
    tok.no_truncation()
    tok.no_padding()
    return lambda texts: [len(e.ids) for e in tok.encode_batch(texts)]


def stats(corpus_dir: Path, conf: dict, wp) -> dict:
    ov = {"paths.corpus": str(corpus_dir), "telemetry.log_level": "ERROR"}
    ov.update({f"chunking.{k}": v for k, v in conf.items()})
    cfg = load_config(REPO / "configs" / "default.yaml", ov, base_dir=REPO)
    b = build_corpus(cfg)
    docs = {d.document_id: d for d in b.documents}
    toks = [c.token_count for c in b.chunks]
    texts = [c.text for c in b.chunks]
    dup_exact = 1 - len(set(texts)) / len(texts)
    unique_chars = sum(len(d.text) for d in b.documents)
    chunk_chars = sum(len(t) for t in texts)
    sections = [(d.document_id, s.section_id) for d in b.documents for s in d.sections if s.paragraphs]
    per_section: dict = {}
    for c in b.chunks:
        per_section[(c.document_id, c.section_id)] = per_section.get((c.document_id, c.section_id), 0) + 1
    crossing = sum(1 for c in b.chunks
                   if not any(s.char_start <= c.char_start and c.char_end <= s.char_end
                              for s in docs[c.document_id].sections if s.section_id == c.section_id))
    # does each chunk end at a sentence boundary? (proxy for "policy statement split across chunks")
    ends_mid = 0
    for c in b.chunks:
        d = docs[c.document_id]
        sec = next(s for s in d.sections if s.section_id == c.section_id)
        ends = {e for p in sec.paragraphs for _, e in sentence_spans(d.text[p.char_start:p.char_end], p.char_start)}
        if c.char_end not in ends:
            ends_mid += 1
    wps = wp([c.index_text for c in b.chunks])
    return {
        "documents": len(b.documents), "skipped_files": [f.path + ": " + (f.reason or "") for f in b.source_files if f.status != "loaded"],
        "sections_with_text": len(sections), "chunks": len(b.chunks),
        "tokens_mean": round(statistics.fmean(toks), 1), "tokens_min": min(toks), "tokens_max": max(toks),
        "tokens_p50": float(np.percentile(toks, 50)), "tokens_p95": float(np.percentile(toks, 95)),
        "duplicate_rate_exact": round(dup_exact, 4), "duplication_ratio_chars": round(chunk_chars / unique_chars, 3),
        "sections_kept_whole": round(sum(1 for k in sections if per_section.get(k, 0) == 1) / len(sections), 3),
        "chunks_crossing_sections": crossing,
        "chunks_starting_mid_sentence": sum(1 for c in b.chunks if c.starts_mid_sentence),
        "chunks_ending_mid_sentence": ends_mid,
        "index_text_wordpieces_p95": float(np.percentile(wps, 95)), "index_text_wordpieces_max": max(wps),
        "truncated_at_512": sum(1 for w in wps if w > 512),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf-dir", type=Path)
    ap.add_argument("--scratch", type=Path, required=True)
    a = ap.parse_args()
    import synth
    datasets = {"fixture": REPO / "tests" / "fixtures" / "corpus",
                "synthetic": synth.make_corpus(a.scratch / "synthetic_chunking", n_docs=40, seed=3)}
    if a.pdf_dir:
        smoke = a.scratch / "pdf_smoke"
        if smoke.exists():
            shutil.rmtree(smoke)
        smoke.mkdir(parents=True)
        (smoke / "TEST_FIXTURE_ONLY").write_text("Non-official PDF smoke set (other hackathon theme guides). Layout statistics only.\n")
        for p in sorted(a.pdf_dir.glob("*.pdf")):
            shutil.copy(p, smoke / p.name)
        datasets["pdf_smoke"] = smoke
    wp = wordpiece_counter()
    results = {ds: {name: stats(path, conf, wp) for name, conf in CONFIGS.items()} for ds, path in datasets.items()}
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / "chunking_experiment.json").write_text(json.dumps(results, indent=2, sort_keys=True))
    cols = ["chunks", "tokens_mean", "tokens_min", "tokens_max", "tokens_p95", "duplication_ratio_chars",
            "duplicate_rate_exact", "sections_kept_whole", "chunks_crossing_sections", "chunks_starting_mid_sentence",
            "chunks_ending_mid_sentence", "truncated_at_512"]
    for ds, rows in results.items():
        print(f"\n== {ds}  docs={next(iter(rows.values()))['documents']}  skipped={next(iter(rows.values()))['skipped_files']}")
        print("config".ljust(22) + " ".join(c[:10].rjust(10) for c in cols))
        for name, r in rows.items():
            print(name.ljust(22) + " ".join(str(r[c]).rjust(10) for c in cols))


if __name__ == "__main__":
    main()
