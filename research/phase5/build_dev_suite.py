"""Build the Phase 5 DEV multi-intent suite (eval/dev_multi_intent/*.json).

TEST-FIXTURE DATA. Utterances are authored by the team about the *fictional* fixture corpus (tests/fixtures/corpus);
every case has is_fixture=true, so results on it are NOT REPORTABLE and NOT held-out (the same team wrote the
decomposition rules). Gold labels: the needs (in annotator words, with paraphrases), constraints with their scope,
relationships, superseded needs, context that must reach a need's query, and fixture citations as gold evidence.
No answers and no retrieval results are stored.

The suite was written after the decomposer code was frozen for Phase 5 measurement; failures found on it are
reported in PHASE_5_MULTI_INTENT_REPORT.md, not tuned away.

Chunk timing: each "|"-delimited chunk arrives when its last word has been spoken (2.6 words/s, +-15% seeded
jitter), as in the Phase 4 suite; utterances of one session are 1.5 s apart.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from streamrag.models.benchmark import MultiIntentBenchmarkCase  # noqa: E402
from streamrag.streaming import simulator as sim  # noqa: E402

LH, OR, OB = "fixture_lighthouse_manual", "Doc_07", "fixture_observatory_notes"


def g(gid, desc, cites, para=(), req=(), superseded=False, answerable=True):
    return {"gold_intent_id": gid, "description": desc, "paraphrases": list(para), "gold_evidence": list(cites),
            "required_query_terms": list(req), "superseded": superseded, "answerable": answerable}


def k(text, scope, applies):
    return {"text": text, "scope": scope, "applies_to": list(applies)}


def r(type_, source, target):
    return {"type": type_, "source": source, "target": target}


def U(text, intents, n_queries, constraints=(), relations=()):
    return {"text": text, "intents": intents, "n": n_queries, "constraints": list(constraints),
            "relations": list(relations)}


CASES = [
    # ---------------------------------------------------------------- A: two independent needs
    ("A1", "A_two_independent", "two lighthouse needs, ASR style", [U(
        "how high should | the wicks be trimmed | and how do keepers | polish the lens",
        [g("u1.g1", "wick trimming height", [f"{LH} §1.1"], ["how high wicks are trimmed"]),
         g("u1.g2", "how the lens is polished", [f"{LH} §1.2"], ["lens polishing method"])], 2)]),
    ("A2", "A_two_independent", "two orchard needs", [U(
        "when are saplings planted | and what happens | to damaged fruit",
        [g("u1.g1", "when saplings are planted", [f"{OR} §1"], ["planting time of saplings"]),
         g("u1.g2", "what is done with damaged fruit", [f"{OR} §3"], ["handling of damaged fruit"])], 2)]),
    ("A3", "A_two_independent", "imperative with two topics", [U(
        "tell me about | the storm procedure | and the logbook rules",
        [g("u1.g1", "storm procedure for the keeper", [f"{LH} §3"], ["what to do in a storm"]),
         g("u1.g2", "rules for the logbook", [f"{LH} §2"], ["logbook rules"])], 2)]),
    ("A4", "A_two_independent", "two observatory needs, same section", [U(
        "can visitors touch | the telescope controls | and when does | the dome open",
        [g("u1.g1", "whether visitors may touch the telescope controls", [f"{OB} §1"]),
         g("u1.g2", "when the dome is opened", [f"{OB} §1"], ["dome opening conditions"])], 2)]),
    ("A5", "A_two_independent", "two orchard needs, question restart", [U(
        "what do workers do | before leaving the tool shed | and where do ladders | go overnight",
        [g("u1.g1", "what workers must do before leaving the tool shed", [f"{OR} §4"]),
         g("u1.g2", "where ladders are kept overnight", [f"{OR} §2.2"], ["ladder storage overnight"])], 2)]),
    # ---------------------------------------------------------------- B: three independent needs
    ("B1", "B_three_independent", "three needs across documents", [U(
        "What are the rules | for ladders in the orchard, | how are the wicks trimmed, | and when is the "
        "telescope recalibrated?",
        [g("u1.g1", "rules for using ladders in the orchard", [f"{OR} §2.1"], ["ladder rules"]),
         g("u1.g2", "how wicks are trimmed", [f"{LH} §1.1"], ["wick trimming"]),
         g("u1.g3", "when the telescope is recalibrated", [f"{OB} §1"], ["telescope recalibration schedule"])], 3)]),
    ("B2", "B_three_independent", "three orchard needs as a list", [U(
        "I'd like to know | the planting weeks, | the crate limit per shift, | and the pruning schedule",
        [g("u1.g1", "weeks when saplings are planted", [f"{OR} §1"]),
         g("u1.g2", "maximum crates per shift", [f"{OR} §3"], ["crate limit per picker"]),
         g("u1.g3", "when pruning happens", [f"{OR} §2"], ["pruning schedule"])], 3)]),
    ("B3", "B_three_independent", "three lighthouse needs, no punctuation", [U(
        "how often is the lamp cleaned | what goes into the logbook | and what does the fog signal | do in a storm",
        [g("u1.g1", "how often the lamp is cleaned", [f"{LH} §1"]),
         g("u1.g2", "what is recorded in the logbook", [f"{LH} §2"]),
         g("u1.g3", "fog signal behaviour during a storm", [f"{LH} §3"])], 3)]),
    ("B4", "B_three_independent", "three lighthouse topics as a comma list", [U(
        "tell me about | wick trimming, lens polishing, | and the lamp register",
        [g("u1.g1", "wick trimming", [f"{LH} §1.1"]), g("u1.g2", "lens polishing", [f"{LH} §1.2"]),
         g("u1.g3", "the lamp register", [f"{LH} §1"], ["recording lamp cleaning"])], 3)]),
    # ---------------------------------------------------------------- C: second need revealed late
    ("C1", "C_incremental_second", "second need after 'and also'", [U(
        "I need information about | the fog signal | and also | the lens",
        [g("u1.g1", "the fog signal", [f"{LH} §3"]), g("u1.g2", "the lens", [f"{LH} §1.2"], ["lens care"])], 2)]),
    ("C2", "C_incremental_second", "second need much later", [U(
        "so first | how many crates | can a picker fill | and then | what happens to damaged fruit",
        [g("u1.g1", "crates a picker can fill", [f"{OR} §3"]),
         g("u1.g2", "what is done with damaged fruit", [f"{OR} §3"])], 2)]),
    ("C3", "C_incremental_second", "second need after a self-interruption", [U(
        "tell me how the telescope | is recalibrated | oh and also | who may visit the observatory",
        [g("u1.g1", "how the telescope is recalibrated", [f"{OB} §1"]),
         g("u1.g2", "who may visit the observatory", [f"{OB} §1"], ["visitor rules"])], 2)]),
    ("C4", "C_incremental_second", "second need after 'plus'", [U(
        "what are the ladder safety rules | in the orchard | plus | how much water | each sapling gets",
        [g("u1.g1", "ladder safety rules in the orchard", [f"{OR} §2.1"]),
         g("u1.g2", "how much water a sapling receives", [f"{OR} §1"])], 2)]),
    # ---------------------------------------------------------------- D: need + constraint
    ("D1", "D_intent_constraint", "focus + restriction", [U(
        "tell me about | the fog signal | especially | during a storm",
        [g("u1.g1", "the fog signal in stormy weather", [f"{LH} §3"], req=["storm"])], 1,
        [k("during a storm", "global", ["u1.g1"])])]),
    ("D2", "D_intent_constraint", "bare focus word", [U(
        "what are the rules for ladders | especially overnight",
        [g("u1.g1", "ladder rules, overnight", [f"{OR} §2.2", f"{OR} §2.1"], req=["overnight"])], 1,
        [k("overnight", "local", ["u1.g1"])])]),
    ("D3", "D_intent_constraint", "restriction after two needs", [U(
        "how high are the crates stacked | and how many can a picker fill | for the night shift",
        [g("u1.g1", "maximum crate stack height", [f"{OR} §3"], req=["night"]),
         g("u1.g2", "crates a picker can fill", [f"{OR} §3"], req=["night"])], 2,
        [k("for the night shift", "global", ["u1.g1", "u1.g2"])])]),
    ("D4", "D_intent_constraint", "fronted restriction", [U(
        "for visitors, | what are the observatory rules | and when is the dome open",
        [g("u1.g1", "observatory rules for visitors", [f"{OB} §1"], req=["visitors"]),
         g("u1.g2", "when the dome is open", [f"{OB} §1"], req=["visitors"])], 2,
        [k("for visitors", "global", ["u1.g1", "u1.g2"])])]),
    ("D5", "D_intent_constraint", "focus on a sub-aspect", [U(
        "tell me the requirements | for ladders, | especially | the safety requirement",
        [g("u1.g1", "ladder requirements with focus on safety", [f"{OR} §2.1"], req=["safety"])], 1,
        [k("the safety requirement", "local", ["u1.g1"])])]),
    # ---------------------------------------------------------------- E: refinement between needs
    ("E1", "E_refinement", "general + narrower need", [U(
        "tell me about the lamp | and also explain | the lamp register",
        [g("u1.g1", "the lamp", [f"{LH} §1"]), g("u1.g2", "the lamp register", [f"{LH} §1"])], 2,
        relations=[r("REFINEMENT", "u1.g2", "u1.g1")])]),
    ("E2", "E_refinement", "narrower need with different words", [U(
        "tell me about pruning | and also explain | how thick branches are cut",
        [g("u1.g1", "pruning", [f"{OR} §2"]), g("u1.g2", "how thick branches are cut", [f"{OR} §2"])], 2,
        relations=[r("REFINEMENT", "u1.g2", "u1.g1")])]),
    ("E3", "E_refinement", "narrower logbook need", [U(
        "what are the logbook rules | and also | what ink the logbook entries use",
        [g("u1.g1", "logbook rules", [f"{LH} §2"]), g("u1.g2", "ink used for logbook entries", [f"{LH} §2"])], 2,
        relations=[r("REFINEMENT", "u1.g2", "u1.g1")])]),
    ("E4", "E_refinement", "topic + its register", [U(
        "tell me about the tool shed | and also | the tool shed register",
        [g("u1.g1", "the tool shed", [f"{OR} §4"]), g("u1.g2", "the tool shed register", [f"{OR} §4"])], 2,
        relations=[r("REFINEMENT", "u1.g2", "u1.g1")])]),
    # ---------------------------------------------------------------- F: follow-up (next utterance)
    ("F1", "F_follow_up", "follow-up with a new noun", [
        U("what are the safety rules | for ladders | in the orchard",
          [g("u1.g1", "ladder safety rules", [f"{OR} §2.1"])], 1),
        U("and what about | the storage rules",
          [g("u2.g1", "ladder storage rules", [f"{OR} §2.2"], req=["ladders"])], 1,
          relations=[r("FOLLOW_UP", "u2.g1", "u1.g1")])]),
    ("F2", "F_follow_up", "follow-up with an unseen facet", [
        U("what are the rules | for observatory visits",
          [g("u1.g1", "rules for observatory visits", [f"{OB} §1"])], 1),
        U("and the opening hours?",
          [g("u2.g1", "when the observatory is open for visits", [f"{OB} §1"], req=["observatory"])], 1,
          relations=[r("FOLLOW_UP", "u2.g1", "u1.g1")])]),
    ("F3", "F_follow_up", "pronoun follow-up", [
        U("how is the telescope | recalibrated", [g("u1.g1", "how the telescope is recalibrated", [f"{OB} §1"])], 1),
        U("and how long | does it take",
          [g("u2.g1", "duration of telescope recalibration", [f"{OB} §1"], req=["telescope"])], 1,
          relations=[r("FOLLOW_UP", "u2.g1", "u1.g1")])]),
    ("F4", "F_follow_up", "aspect-only follow-up", [
        U("what are the requirements | for using a ladder", [g("u1.g1", "ladder requirements", [f"{OR} §2.1"])], 1),
        U("what about | the time limits",
          [g("u2.g1", "time limits for ladder use", [f"{OR} §2.2"], req=["ladder"])], 1,
          relations=[r("FOLLOW_UP", "u2.g1", "u1.g1")])]),
    # ---------------------------------------------------------------- G: pronoun-based
    ("G1", "G_pronoun", "pronoun in the next sentence", [U(
        "Tell me about | the fog signal. | And how often | does it sound?",
        [g("u1.g1", "the fog signal", [f"{LH} §3"]),
         g("u1.g2", "how often the fog signal sounds", [f"{LH} §3"], req=["fog"])], 2,
        relations=[r("DEPENDENT", "u1.g2", "u1.g1")])]),
    ("G2", "G_pronoun", "pronoun after 'and when'", [U(
        "what is the lamp register | and when is it | filled in",
        [g("u1.g1", "what the lamp register is", [f"{LH} §1"]),
         g("u1.g2", "when the lamp register is filled in", [f"{LH} §1"], req=["register"])], 2,
        relations=[r("DEPENDENT", "u1.g2", "u1.g1")])]),
    ("G3", "G_pronoun", "pronoun across utterances", [
        U("how do keepers | polish the lens", [g("u1.g1", "how the lens is polished", [f"{LH} §1.2"])], 1),
        U("and what must | never be used on it",
          [g("u2.g1", "what must not be used on the lens", [f"{LH} §1.2"], req=["lens"])], 1,
          relations=[r("FOLLOW_UP", "u2.g1", "u1.g1")])]),
    ("G4", "G_pronoun", "pronoun with a competing antecedent", [U(
        "tell me about the tool shed | and what workers must do | before they leave it",
        [g("u1.g1", "the tool shed", [f"{OR} §4"]),
         g("u1.g2", "what workers must do before leaving the tool shed", [f"{OR} §4"], req=["shed"])], 2,
        relations=[r("DEPENDENT", "u1.g2", "u1.g1")])]),
    # ---------------------------------------------------------------- H: shared context
    ("H1", "H_shared_context", "distributed topic", [U(
        "what are the rules | and the schedule | for pruning",
        [g("u1.g1", "pruning rules", [f"{OR} §2"], req=["pruning"]),
         g("u1.g2", "pruning schedule", [f"{OR} §2"], req=["pruning"])], 2)]),
    ("H2", "H_shared_context", "fronted restriction, two needs", [U(
        "For the night shift, | how many crates | can a picker fill | and what happens | to fruit picked after sunset",
        [g("u1.g1", "crates per picker on the night shift", [f"{OR} §3"], req=["night"]),
         g("u1.g2", "handling of fruit picked after sunset", [f"{OR} §3"], req=["night"])], 2,
        [k("For the night shift", "global", ["u1.g1", "u1.g2"])])]),
    ("H3", "H_shared_context", "two activities sharing an object", [U(
        "tell me about | cleaning and logging | for the lamp",
        [g("u1.g1", "lamp cleaning", [f"{LH} §1"], req=["lamp"]),
         g("u1.g2", "lamp logging", [f"{LH} §1"], req=["lamp"])], 2)]),
    ("H4", "H_shared_context", "two facets of one topic", [U(
        "what are the requirements | and the timeline | for telescope calibration",
        [g("u1.g1", "telescope calibration requirements", [f"{OB} §1"], req=["telescope"]),
         g("u1.g2", "telescope calibration timeline", [f"{OB} §1"], req=["telescope"])], 2)]),
    # ---------------------------------------------------------------- I: over-decomposition traps (one need)
    ("I1", "I_over_decomposition_trap", "comparison", [U(
        "what is the difference | between the wick | and the lens",
        [g("u1.g1", "difference between the wick and the lens", [f"{LH} §1.1", f"{LH} §1.2"])], 1)]),
    ("I2", "I_over_decomposition_trap", "alternative", [U(
        "can ladders or crates | be left in the orchard | overnight",
        [g("u1.g1", "whether ladders or crates may stay overnight", [f"{OR} §2.2"])], 1)]),
    ("I3", "I_over_decomposition_trap", "fixed corpus phrase", [U(
        "tell me about | the third and fifth week | of planting",
        [g("u1.g1", "planting weeks", [f"{OR} §1"])], 1)]),
    ("I4", "I_over_decomposition_trap", "narrative with many nouns", [U(
        "so yesterday I walked past | the orchard and the tool shed | and the cool room | with my sister and her dog | "
        "anyway how many crates | can a picker fill per shift",
        [g("u1.g1", "crates per picker per shift", [f"{OR} §3"])], 1)]),
    ("I5", "I_over_decomposition_trap", "generic nouns", [U(
        "I need information | and details | about the lens",
        [g("u1.g1", "the lens", [f"{LH} §1.2"])], 1)]),
    # ---------------------------------------------------------------- J: under-decomposition traps
    ("J1", "J_under_decomposition_trap", "two facet nouns under one head", [U(
        "tell me the crate limit | and the stacking height",
        [g("u1.g1", "crate limit per shift", [f"{OR} §3"]), g("u1.g2", "crate stacking height", [f"{OR} §3"])], 2)]),
    ("J2", "J_under_decomposition_trap", "nominal need + question, no punctuation", [U(
        "the requirements for ladders | how long does pruning take",
        [g("u1.g1", "ladder requirements", [f"{OR} §2.1"]), g("u1.g2", "duration of pruning", [f"{OR} §2"])], 2)]),
    ("J3", "J_under_decomposition_trap", "two questions without a conjunction", [U(
        "how high are wicks trimmed | how often is the lamp cleaned",
        [g("u1.g1", "wick trimming height", [f"{LH} §1.1"]), g("u1.g2", "lamp cleaning frequency", [f"{LH} §1"])], 2)]),
    ("J4", "J_under_decomposition_trap", "ASR list without commas", [U(
        "tell me about | the planting schedule | the water each sapling gets | and the approved varieties",
        [g("u1.g1", "planting schedule", [f"{OR} §1"]), g("u1.g2", "water per sapling", [f"{OR} §1"]),
         g("u1.g3", "approved sapling varieties", [f"{OR} §1"])], 3)]),
    # ---------------------------------------------------------------- K: duplicates
    ("K1", "K_duplicate", "same need twice", [U(
        "tell me about the wicks | and the wicks", [g("u1.g1", "the wicks", [f"{LH} §1.1"])], 1)]),
    ("K2", "K_duplicate", "same need, two phrasings", [U(
        "how is the lens polished | and how do you | polish the lens",
        [g("u1.g1", "how the lens is polished", [f"{LH} §1.2"])], 1)]),
    ("K3", "K_duplicate", "question repeated in the next utterance", [
        U("how many crates | can a picker fill", [g("u1.g1", "crates a picker can fill", [f"{OR} §3"])], 1),
        U("how many crates | can a picker fill", [g("u2.g1", "crates a picker can fill", [f"{OR} §3"])], 0)]),
    # ---------------------------------------------------------------- L: changed / corrected need
    ("L1", "L_changed_intent", "explicit replacement", [U(
        "Tell me the requirements | for ladders. | Actually, I meant crates | instead of ladders.",
        [g("u1.g1", "ladder requirements", [f"{OR} §2.1"], superseded=True),
         g("u1.g2", "crate requirements", [f"{OR} §3"], req=["crates"])], 1)]),
    ("L2", "L_changed_intent", "correction in the next utterance", [
        U("tell me about | the lens", [g("u1.g1", "the lens", [f"{LH} §1.2"], superseded=True)], 1),
        U("actually I meant | the lamp | not the lens", [g("u2.g1", "the lamp", [f"{LH} §1"])], 1)]),
    ("L3", "L_changed_intent", "'no wait' correction", [U(
        "how often is the lamp cleaned | no wait | how often is the lens polished",
        [g("u1.g1", "lamp cleaning frequency", [f"{LH} §1"], superseded=True),
         g("u1.g2", "lens polishing", [f"{LH} §1.2"])], 1)]),
    ("L4", "L_changed_intent", "'scratch that' correction", [U(
        "what do workers do | before leaving the tool shed | scratch that | where are ladders stored",
        [g("u1.g1", "tool shed leaving procedure", [f"{OR} §4"], superseded=True),
         g("u1.g2", "where ladders are stored", [f"{OR} §2.2"])], 1)]),
    # ---------------------------------------------------------------- S: single-need controls
    ("S1", "S_single", "single need", [U(
        "how many crates | can a picker fill | in one shift", [g("u1.g1", "crates per shift", [f"{OR} §3"])], 1)]),
    ("S2", "S_single", "single need with a long tail", [U(
        "what should the keeper | use to polish the lens | so that it stays clear",
        [g("u1.g1", "what to polish the lens with", [f"{LH} §1.2"])], 1)]),
    ("S3", "S_single", "single need, imperative", [U(
        "tell me about | pruning in the orchard", [g("u1.g1", "pruning in the orchard", [f"{OR} §2"])], 1)]),
]


def build(case_id, category, desc, utts, seed):
    out_utts = []
    for n, u in enumerate(utts, start=1):
        chunks = sim.parse_inline(u["text"])
        evs = sim.timed(chunks, words_per_second=2.6, end_gap_ms=500, jitter=0.15, seed=seed + n,
                        session_id=f"mi-{case_id}", utterance_id=f"u{n}", wrap_session=False)
        out_utts.append({
            "utterance_id": f"u{n}", "utterance_text": " ".join(chunks),
            "chunks": [{"chunk_index": e.payload.chunk_index, "timestamp_s": e.payload.timestamp_s,
                        "text": e.payload.text} for e in evs[:-1]],
            "utterance_end_s": evs[-1].payload.timestamp_s, "expected_intents": u["intents"],
            "expected_constraints": u["constraints"], "expected_relationships": u["relations"],
            "expected_query_count": u["n"]})
    return MultiIntentBenchmarkCase.model_validate({
        "case_id": case_id, "category": category, "split": "tune", "is_fixture": True,
        "description": f"DEV SUITE (fixture domain): {desc}", "utterances": out_utts,
        "provenance": {"authored_by": "team", "note": "fictional fixture domain; not official benchmark data; "
                                                      "not held-out (same team wrote the rules)"}})


def main() -> None:
    out = REPO / "eval" / "dev_multi_intent"
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("*.json"):
        old.unlink()
    for i, (cid, cat, desc, utts) in enumerate(CASES):
        case = build(cid, cat, desc, utts, seed=500 + 10 * i)
        (out / f"{cid}.json").write_text(json.dumps(case.model_dump(mode="json"), indent=2, sort_keys=True))
    (out / "README.md").write_text(
        "# DEV multi-intent suite (TEST FIXTURE DATA)\n\nGenerated by `research/phase5/build_dev_suite.py`: "
        f"{len(CASES)} team-authored cases (categories A-L of the Phase 5 brief + S single-need controls) about the "
        "fictional fixture corpus (`tests/fixtures/corpus`); `is_fixture: true`. Gold = needs, constraints, "
        "relationships, superseded needs, required query context and fixture citations. Development only: results "
        "are NOT official benchmark results and NOT held-out.\n")
    print(f"wrote {len(CASES)} cases to {out}")


if __name__ == "__main__":
    main()
