"""Build the Phase 4 DEV streaming suite (eval/dev_streaming/*.json).

TEST-FIXTURE DATA: utterances are authored by the team about the *fictional* fixture corpus
(tests/fixtures/corpus). Every case has is_fixture=true, so any benchmark built on it is NOT REPORTABLE.
Behavior labels (retrieval required or not, turn type) need no corpus facts; gold evidence points at fixture
sections only. Each utterance gets two chunkings: the authored split ("|") and a seeded 2-3-word split.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from streamrag.models.benchmark import BenchmarkCase  # noqa: E402
from streamrag.streaming import simulator as sim  # noqa: E402

LH, OR, OB = "fixture_lighthouse_manual", "Doc_07", "fixture_observatory_notes"

# (case_key, category, turn_type, retrieval_required, utterance with "|" chunk breaks, gold citations)
UTTERANCES = [
    ("wick_height", "incremental_intent", "query", True, "so I wanted to ask | how high the wicks | should be trimmed | in the lighthouse", [f"{LH} §1.1"]),
    ("fog_storm", "single_intent", "query", True, "what happens | with the fog signal | during a storm", [f"{LH} §3"]),
    ("lamp_cleaning", "incremental_intent", "query", True, "um can you tell me | how often the lamp | has to be cleaned", [f"{LH} §1"]),
    ("logbook", "incremental_intent", "query", True, "I need to know | what has to be written | in the logbook | for each watch", [f"{LH} §2"]),
    ("crates", "single_intent", "query", True, "how many crates | can a picker fill | in one shift", [f"{OR} §3"]),
    ("ladder_overnight", "single_intent", "query", True, "are ladders allowed | to stay in the orchard | overnight or not", [f"{OR} §2.2"]),
    ("planting", "single_intent", "query", True, "when do the saplings | get planted | and how much water | do they need", [f"{OR} §1"]),
    ("ladder_rule", "single_intent", "query", True, "what is the rule | for using a ladder | in the orchard", [f"{OR} §2.1"]),
    ("dome", "single_intent", "query", True, "when does the observatory | open its dome", [f"{OB} §1"]),
    ("calibration_time", "single_intent", "query", True, "how long does | the telescope calibration | take", [f"{OB} §1"]),
    ("tool_shed", "irrelevant_continuation", "query", True, "uh hi | quick question | about the tool shed | what do workers do | before they leave", [f"{OR} §4"]),
    ("visitors_controls", "single_intent", "query", True, "I'm curious | whether visitors can | touch the telescope controls", [f"{OB} §1"]),
    ("lens", "single_intent", "query", True, "what should the keeper | use to polish | the lens", [f"{LH} §1.2"]),
    ("pruning", "single_intent", "query", True, "tell me about | pruning in the orchard", [f"{OR} §2"]),
    ("damaged_fruit", "single_intent", "query", True, "what happens to | damaged fruit | during the harvest", [f"{OR} §3"]),
    ("stack_height", "long_streaming", "query", True, "okay so we are setting up | the new schedule for next month | and I was wondering | about the harvest limits | specifically how high | the crates can be stacked | before the cart leaves", [f"{OR} §3"]),
    ("fog_asr", "early_retrieval", "query", True, "fog signal storm | how often | does it sound", [f"{LH} §3"]),
    ("dome_log", "early_retrieval", "query", True, "dome status logging | how often | do observers | record it", [f"{OB} §1"]),
    ("crates_single", "single_intent", "query", True, "how many crates per shift", [f"{OR} §3"]),
    ("recalibration_single", "single_intent", "query", True, "when is the telescope recalibrated", [f"{OB} §1"]),
    ("ink_single", "single_intent", "query", True, "what ink is used in the logbook", [f"{LH} §2"]),
    ("out_of_corpus", "single_intent", "query", True, "who won the cricket | championship yesterday", []),
    ("ack_okay", "suppression", "backchannel", False, "okay", []),
    ("ack_thanks", "suppression", "social", False, "thanks | that helps a lot", []),
    ("ack_mmhm", "suppression", "backchannel", False, "mm-hm | okay | right", []),
    ("repeat_prev", "suppression", "presentation", False, "could you | repeat the previous answer | please", []),
    ("shorter", "suppression", "presentation", False, "make that | shorter", []),
    ("summarize_evidence", "suppression", "presentation", False, "can you summarize | the retrieved evidence | for me", []),
    ("translate", "suppression", "presentation", False, "translate that | into hindi", []),
    ("got_it", "suppression", "social", False, "got it | thank you so much", []),
    ("meta", "suppression", "meta", False, "what did I | ask you before", []),
    ("thinking", "suppression", "social", False, "hmm | let me think | for a second", []),
    ("casual", "suppression", "social", False, "I'm going to | grab a coffee | first", []),
]


def auto_split(text: str, seed: int) -> list[str]:
    words = text.replace("|", " ").split()
    rng, out, i = random.Random(seed), [], 0
    while i < len(words):
        n = rng.choice([2, 3])
        out.append(" ".join(words[i:i + n]))
        i += n
    return out


def make_case(key, category, turn_type, required, utterance, gold, variant, seed):
    chunks = sim.parse_inline(utterance) if variant == "authored" else auto_split(utterance, seed)
    evs = sim.timed(chunks, words_per_second=2.6, end_gap_ms=500, jitter=0.15, seed=seed, session_id=f"dev-{key}-{variant}",
                    wrap_session=False)
    case_chunks = [{"chunk_index": e.payload.chunk_index, "timestamp_s": e.payload.timestamp_s, "text": e.payload.text}
                   for e in evs[:-1]]
    intents = []
    if required:
        intents = [{"gold_intent_id": "g1", "description": " ".join(chunks), "answerable": bool(gold),
                    "gold_evidence": gold}]
    case = {
        "case_id": f"{key}-{variant}", "category": category, "split": "test", "is_fixture": True,
        "description": f"DEV SUITE (fixture domain); chunking={variant}",
        "corpus_ref": {"corpus_id": "tests/fixtures/corpus (TEST FIXTURE)"},
        "session": {"session_id": f"dev-{key}-{variant}", "turns": [{
            "utterance_id": "u1", "utterance_text": " ".join(chunks), "chunks": case_chunks,
            "utterance_end_s": evs[-1].payload.timestamp_s,
            "chunking": {"generator": "authored" if variant == "authored" else "auto_2_3_words", "seed": seed,
                         "words_per_second": 2.6, "jitter": 0.15},
            "expected": {"turn_type": turn_type, "retrieval_required": required, "intents": intents,
                         "suppression_reason": None}}]},
        "provenance": {"authored_by": "team", "note": "fictional fixture domain; not official benchmark data"},
    }
    return BenchmarkCase.model_validate(case)


def main() -> None:
    out = REPO / "eval" / "dev_streaming"
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("*.json"):
        old.unlink()
    n = 0
    for i, (key, cat, tt, req, utt, gold) in enumerate(UTTERANCES):
        variants = ["authored"] + (["auto"] if len(utt.replace("|", " ").split()) > 3 else [])
        for variant in variants:
            case = make_case(key, cat, tt, req, utt, gold, variant, seed=100 + i)
            (out / f"{case.case_id}.json").write_text(json.dumps(case.model_dump(mode="json"), indent=2, sort_keys=True))
            n += 1
    (out / "README.md").write_text(
        "# DEV streaming suite (TEST FIXTURE DATA)\n\nGenerated by `research/phase4/build_dev_suite.py`. Utterances are team-authored "
        "about the fictional fixture corpus (`tests/fixtures/corpus`); every case has `is_fixture: true`. Use for development "
        "and controller ablations only — results are NOT official benchmark results.\n")
    print(f"wrote {n} cases to {out}")


if __name__ == "__main__":
    main()
