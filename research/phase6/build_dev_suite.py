"""Build the Phase 6 DEV adaptive-session suite (eval/dev_adaptive/*.json) and the STRESS set
(eval/dev_adaptive_stress/*.json).

TEST-FIXTURE DATA. Sessions are authored by the team about the *fictional* fixture corpora (tests/fixtures/corpus,
and tests/fixtures/corpus_conflict for the contradiction cases); every case has is_fixture=true, so results on it are
NOT REPORTABLE and NOT held-out (the same team wrote the session rules). Gold labels describe what a correct adaptive
system should do after each turn, in the annotator's judgement - not what the implementation does:

  change_types        multiset of expected context-change types (alternatives allowed: change_types_any)
  retrieval           "required" (new retrieval needed) | "reuse" (same semantics as earlier: evidence reusable,
                      no new retrieval) | "none" (nothing retrieval-relevant changed)
  need_queries        for needs that must exist after the turn: analyzed terms the need's active query must
                      contain (required) / must not contain (forbidden)
  forbidden_new       terms no query created in this turn may contain (context isolation)
  gold_evidence       fixture citations that should be usable for the changed / new need after the turn
  expect_revalidation fixture citations whose applicability the change puts in question (should enter revalidation)
  expect_conflict     the corpus states conflicting values for the asked fact (answer must carry uncertainty)
  ambiguous           the annotator considers more than one reading reasonable (reported separately)

No answers and no retrieval results are stored. Chunks: "|" separates streaming chunks (2.6 words/s, as in the
Phase 4/5 suites); turns of one session are 2.5 s apart in the streaming run.
"""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "eval" / "dev_adaptive"
LH, OR, OB = "fixture_lighthouse_manual", "Doc_07", "fixture_observatory_notes"


def T(text, change, retrieval, needs=(), forbidden_new=(), evidence=(), reval=(), conflict=False, alt=None,
      ambiguous=False):
    g = {"change_types": list(change), "retrieval": retrieval,
         "need_queries": [{"required": list(r), "forbidden": list(f)} for r, f in needs],
         "forbidden_new": list(forbidden_new), "gold_evidence": list(evidence), "expect_revalidation": list(reval),
         "expect_conflict": conflict, "ambiguous": ambiguous}
    if alt:
        g["change_types_any"] = [list(change)] + [list(a) for a in alt]
    return {"utterance_text": text.replace(" | ", " "), "chunks": [c.strip() for c in text.split("|")], "gold": g}


LADDERS = "What are the rules | for ladders | in the orchard?"
LADDERS_NEED = (("ladders",), ())

CASES = [
    # ---------------------------------------------------------------- constraint addition
    ("S01", "constraint_addition", "late focus detail on the previous need", "fixture", [
        T(LADDERS, ["NEW_INTENT"], "required", [LADDERS_NEED], evidence=[f"{OR} §2.1", f"{OR} §2.2"]),
        T("Specifically | overnight.", ["CONSTRAINT_ADDITION"], "required", [(("ladders", "overnight"), ())],
          evidence=[f"{OR} §2.2"])]),
    ("S02", "constraint_addition", "late condition on a lighthouse need", "fixture", [
        T("What does the keeper | do during a storm?", ["NEW_INTENT"], "required", [(("keeper", "storm"), ())],
          evidence=[f"{LH} §3"]),
        T("Only when | visibility is low.", ["CONSTRAINT_ADDITION"], "required", [(("storm", "visibility"), ())],
          evidence=[f"{LH} §3"])]),
    ("S03", "constraint_addition", "late focus phrase on a harvest need", "fixture", [
        T("How are crates | handled at harvest?", ["NEW_INTENT"], "required", [(("crates", "harvest"), ())],
          evidence=[f"{OR} §3"]),
        T("Especially for fruit | picked after sunset.", ["CONSTRAINT_ADDITION"], "required",
          [(("crates", "sunset"), ())], evidence=[f"{OR} §3"])]),
    ("S04", "constraint_addition", "restriction after a backchannel", "fixture", [
        T(LADDERS, ["NEW_INTENT"], "required", [LADDERS_NEED]),
        T("Okay.", ["NO_CHANGE"], "none"),
        T("Only for the | night shift though.", ["CONSTRAINT_ADDITION"], "required",
          [(("ladders", "night", "shift"), ())])]),
    ("S05", "constraint_addition", "late detail applying to two needs of one utterance", "fixture", [
        T("How are the wicks trimmed | and how is the lens polished?", ["NEW_INTENT", "NEW_INTENT"], "required",
          [(("wicks",), ()), (("lens",), ())], evidence=[f"{LH} §1.1", f"{LH} §1.2"]),
        T("Especially | before sunset.", ["CONSTRAINT_ADDITION", "CONSTRAINT_ADDITION"], "required",
          [(("wicks", "sunset"), ()), (("lens", "sunset"), ())], alt=[["CONSTRAINT_ADDITION"]], ambiguous=True)]),
    # ---------------------------------------------------------------- constraint removal / update
    ("S06", "constraint_removal", "retract a constraint stated in the first turn", "fixture", [
        T("What are the rules for ladders | in the orchard, | especially overnight?", ["NEW_INTENT"], "required",
          [(("ladders", "overnight"), ())]),
        T("Actually, ignore | the overnight part.", ["CONSTRAINT_REMOVAL"], "required", [(("ladders",), ("overnight",))],
          reval=[f"{OR} §2.2"])]),
    ("S07", "constraint_removal", "retract a late detail: back to an earlier state", "fixture", [
        T(LADDERS, ["NEW_INTENT"], "required", [LADDERS_NEED]),
        T("Specifically | overnight.", ["CONSTRAINT_ADDITION"], "required", [(("ladders", "overnight"), ())]),
        T("Never mind | the overnight restriction.", ["CONSTRAINT_REMOVAL"], "reuse",
          [(("ladders",), ("overnight",))], reval=[f"{OR} §2.2"])]),
    ("S08", "constraint_removal", "retraction with 'forget'", "fixture", [
        T("Who may visit | the observatory, | only on Fridays?", ["NEW_INTENT"], "required",
          [(("visit", "observatory"), ())], evidence=[f"{OB} §1"]),
        T("Forget | the Friday part.", ["CONSTRAINT_REMOVAL"], "required", [(("observatory",), ("friday",))],
          alt=[["NO_CHANGE"]], ambiguous=True)]),
    ("S09", "constraint_update", "a constraint value is replaced", "fixture", [
        T("How are crates | handled at harvest?", ["NEW_INTENT"], "required", [(("crates",), ())]),
        T("Only for the | night shift.", ["CONSTRAINT_ADDITION"], "required", [(("crates", "night"), ())],
          reval=[f"{OR} §3"]),
        T("Sorry, | for the day shift.", ["CONSTRAINT_ADDITION", "CONSTRAINT_REMOVAL"], "required",
          [(("crates", "day"), ("night",))])]),
    # ---------------------------------------------------------------- corrections / entity replacement
    ("S10", "correction", "entity replacement with explicit old entity", "fixture", [
        T(LADDERS, ["NEW_INTENT"], "required", [LADDERS_NEED]),
        T("Sorry, I meant | crates instead of ladders.", ["CORRECTION"], "required", [(("crates",), ("ladders",))],
          evidence=[f"{OR} §3"])]),
    ("S11", "correction", "short entity correction", "fixture", [
        T("How is the lens | polished?", ["NEW_INTENT"], "required", [(("lens",), ())], evidence=[f"{LH} §1.2"]),
        T("Actually, | I meant the wick.", ["CORRECTION"], "required", [(("wick",), ("lens",))],
          evidence=[f"{LH} §1.1"])]),
    ("S12", "correction", "correction after a late constraint", "fixture", [
        T(LADDERS, ["NEW_INTENT"], "required", [LADDERS_NEED]),
        T("Specifically | overnight.", ["CONSTRAINT_ADDITION"], "required", [(("ladders", "overnight"), ())]),
        T("Sorry, I meant | crates instead of ladders.", ["CORRECTION"], "required", [(("crates",), ("ladders",))],
          ambiguous=True)]),
    # ---------------------------------------------------------------- new intents / follow-ups
    ("S13", "new_intent_follow_up", "parallel follow-up inherits the facet", "fixture", [
        T(LADDERS, ["NEW_INTENT"], "required", [LADDERS_NEED]),
        T("What about | crates?", ["NEW_INTENT"], "required", [(("crates", "rules"), ("ladders",)), LADDERS_NEED],
          evidence=[f"{OR} §3"])]),
    ("S14", "new_intent_follow_up", "follow-up that restricts the previous need", "fixture", [
        T(LADDERS, ["NEW_INTENT"], "required", [LADDERS_NEED]),
        T("What about | overnight?", ["CONSTRAINT_ADDITION"], "required", [(("ladders", "overnight"), ())],
          alt=[["NEW_INTENT"]], evidence=[f"{OR} §2.2"])]),
    ("S15", "new_intent_independent", "unrelated new question", "fixture", [
        T(LADDERS, ["NEW_INTENT"], "required", [LADDERS_NEED]),
        T("Now explain | how the telescope | is recalibrated.", ["NEW_INTENT"], "required",
          [(("telescope", "recalibrated"), ("ladders",))], forbidden_new=["ladders", "orchard"],
          evidence=[f"{OB} §1"])]),
    ("S16", "new_intent_independent", "additional need in the same conversation", "fixture", [
        T("What does the keeper | do during a storm?", ["NEW_INTENT"], "required", [(("storm",), ())]),
        T("And how are | the wicks trimmed?", ["NEW_INTENT"], "required", [(("wicks",), ("storm",))],
          forbidden_new=["storm"], evidence=[f"{LH} §1.1"])]),
    # ---------------------------------------------------------------- unchanged continuation
    ("S17", "unchanged_continuation", "backchannels", "fixture", [
        T("How are | the wicks trimmed?", ["NEW_INTENT"], "required", [(("wicks",), ())]),
        T("Okay.", ["NO_CHANGE"], "none"),
        T("Thanks.", ["NO_CHANGE"], "none")]),
    ("S18", "unchanged_continuation", "acknowledgement with filler", "fixture", [
        T("When are | saplings planted?", ["NEW_INTENT"], "required", [(("saplings",), ())], evidence=[f"{OR} §1"]),
        T("Right, | got it.", ["NO_CHANGE"], "none")]),
    # ---------------------------------------------------------------- reuse
    ("S19", "reuse", "same question rephrased", "fixture", [
        T(LADDERS, ["NEW_INTENT"], "required", [LADDERS_NEED]),
        T("Tell me | the orchard ladder rules.", ["NEW_INTENT"], "reuse", [LADDERS_NEED], alt=[["NO_CHANGE"]])]),
    ("S20", "reuse", "same question, different word order", "fixture", [
        T("How is the lens | polished?", ["NEW_INTENT"], "required", [(("lens",), ())]),
        T("The lens, | how is it polished?", ["NEW_INTENT"], "reuse", [(("lens",), ())], alt=[["NO_CHANGE"]])]),
    # ---------------------------------------------------------------- context isolation / topic return
    ("S21", "context_isolation", "topic return with a late detail", "fixture", [
        T(LADDERS, ["NEW_INTENT"], "required", [LADDERS_NEED]),
        T("Now explain | how the telescope | is recalibrated.", ["NEW_INTENT"], "required", [(("telescope",), ())]),
        T("Back to the ladders, | what about overnight?", ["CONSTRAINT_ADDITION"], "required",
          [(("ladders", "overnight"), ("telescope",))], forbidden_new=["telescope"])]),
    ("S22", "context_isolation", "long session, unrelated questions", "fixture", [
        T("What does the keeper | do during a storm?", ["NEW_INTENT"], "required", [(("storm",), ())]),
        T("Specifically | about the fog signal.", ["CONSTRAINT_ADDITION"], "required", [(("storm", "fog"), ())],
          alt=[["REFINEMENT"]]),
        T("How often | is the telescope recalibrated?", ["NEW_INTENT"], "required", [(("telescope",), ("fog",))],
          forbidden_new=["fog", "storm", "keeper"]),
        T("When are | saplings planted?", ["NEW_INTENT"], "required", [(("saplings",), ("telescope",))],
          forbidden_new=["fog", "telescope"]),
        T("How much water | do they get?", ["NEW_INTENT"], "required", [(("water",), ("telescope", "fog"))],
          forbidden_new=["telescope", "fog"], alt=[["REFINEMENT"], ["CONSTRAINT_ADDITION"]], ambiguous=True)]),
    # ---------------------------------------------------------------- evidence invalidation
    ("S23", "evidence_invalidation", "constraint puts some evidence in question", "fixture", [
        T("How are crates | handled at harvest?", ["NEW_INTENT"], "required", [(("crates",), ())]),
        T("Only for the | night shift.", ["CONSTRAINT_ADDITION"], "required", [(("crates", "night"), ())],
          reval=[f"{OR} §3"])]),
    ("S24", "evidence_invalidation", "correction invalidates the old entity's evidence", "fixture", [
        T("How is the lens | polished?", ["NEW_INTENT"], "required", [(("lens",), ())]),
        T("Sorry, I meant | the lamp, not the lens.", ["CORRECTION"], "required", [(("lamp",), ("lens",))],
          evidence=[f"{LH} §1"])]),
    # ---------------------------------------------------------------- refinement
    ("S25", "refinement", "place added to a need", "fixture", [
        T("What are | the ladder rules?", ["NEW_INTENT"], "required", [(("ladder",), ())]),
        T("In the orchard.", ["CONSTRAINT_ADDITION"], "required", [(("ladder", "orchard"), ())],
          alt=[["REFINEMENT"]])]),
    ("S26", "mixed", "late detail and a new question in one utterance", "fixture", [
        T(LADDERS, ["NEW_INTENT"], "required", [LADDERS_NEED]),
        T("Specifically overnight, | and how are | crates stacked?", ["CONSTRAINT_ADDITION", "NEW_INTENT"], "required",
          [(("ladders", "overnight"), ()), (("crates",), ())], alt=[["NEW_INTENT"]], ambiguous=True)]),
    # ---------------------------------------------------------------- contradiction (conflict fixture corpus)
    ("S27", "contradiction", "two sources state different limits", "conflict", [
        T("When does the observatory | open its dome?", ["NEW_INTENT"], "required", [(("dome",), ())],
          conflict=True)]),
    ("S28", "contradiction", "a conflicting fact arrives with a follow-up need", "conflict", [
        T("Who may visit | the observatory?", ["NEW_INTENT"], "required", [(("visit",), ())]),
        T("And when | is the dome opened?", ["NEW_INTENT"], "required", [(("dome",), ())], conflict=True)]),
]

# STRESS set: written after the Phase 6 code freeze, with phrasing deliberately unlike the cases the rules were
# developed on (ASR casing without punctuation, fillers, other correction / retraction wordings, anaphora). Run once
# and reported as measured; the code was not changed in response to it.
STRESS_CASES = [
    ("X01", "constraint_addition", "ASR style, filler before the detail", "fixture", [
        T("what are the rules | for ladders | in the orchard", ["NEW_INTENT"], "required", [LADDERS_NEED]),
        T("uh specifically | overnight", ["CONSTRAINT_ADDITION"], "required", [(("ladders", "overnight"), ())])]),
    ("X02", "constraint_update", "update without an apology", "fixture", [
        T("How are crates | handled at harvest?", ["NEW_INTENT"], "required", [(("crates",), ())]),
        T("Only for the | night shift.", ["CONSTRAINT_ADDITION"], "required", [(("crates", "night"), ())]),
        T("Make that | the day shift instead.", ["CONSTRAINT_ADDITION", "CONSTRAINT_REMOVAL"], "required",
          [(("crates", "day"), ("night",))])]),
    ("X03", "correction", "'scratch that' correction", "fixture", [
        T(LADDERS, ["NEW_INTENT"], "required", [LADDERS_NEED]),
        T("Scratch that, | I'm asking about crates.", ["CORRECTION"], "required", [(("crates",), ("ladders",))],
          alt=[["NEW_INTENT"]], ambiguous=True)]),
    ("X04", "correction", "'no wait' correction", "fixture", [
        T("How is the lens | polished?", ["NEW_INTENT"], "required", [(("lens",), ())]),
        T("No wait, | the lamp.", ["CORRECTION"], "required", [(("lamp",), ("lens",))])]),
    ("X05", "correction", "'not X, Y' correction", "fixture", [
        T("How is the lens | polished?", ["NEW_INTENT"], "required", [(("lens",), ())]),
        T("Not the lens, | the wick.", ["CORRECTION"], "required", [(("wick",), ("lens",))])]),
    ("X06", "constraint_removal", "'drop' retraction as a question", "fixture", [
        T(LADDERS, ["NEW_INTENT"], "required", [LADDERS_NEED]),
        T("Specifically | overnight.", ["CONSTRAINT_ADDITION"], "required", [(("ladders", "overnight"), ())]),
        T("Can you drop | the overnight condition?", ["CONSTRAINT_REMOVAL"], "reuse",
          [(("ladders",), ("overnight",))])]),
    ("X07", "new_intent_independent", "'also' new question", "fixture", [
        T("What does the keeper | do during a storm?", ["NEW_INTENT"], "required", [(("storm",), ())]),
        T("Also, | how is the logbook kept?", ["NEW_INTENT"], "required", [(("logbook",), ("storm",))],
          forbidden_new=["storm"])]),
    ("X08", "new_intent_independent", "elliptical question that changes topic", "fixture", [
        T(LADDERS, ["NEW_INTENT"], "required", [LADDERS_NEED]),
        T("How about | the observatory visitors?", ["NEW_INTENT"], "required", [(("visitors",), ("ladders",))],
          forbidden_new=["ladders", "orchard"])]),
    ("X09", "new_intent_follow_up", "backchannel plus follow-up in one utterance", "fixture", [
        T("What does the keeper | do during a storm?", ["NEW_INTENT"], "required", [(("storm",), ())]),
        T("Got it, | and what about | the fog signal?", ["CONSTRAINT_ADDITION"], "required",
          [(("storm", "fog"), ())], alt=[["NEW_INTENT"]], ambiguous=True)]),
    ("X10", "constraint_addition", "bare 'just for' detail", "fixture", [
        T("How is the lamp | cleaned?", ["NEW_INTENT"], "required", [(("lamp",), ())]),
        T("Just for keepers.", ["CONSTRAINT_ADDITION"], "required", [(("lamp", "keepers"), ())])]),
    ("X11", "context_isolation", "topic return without a new detail", "fixture", [
        T(LADDERS, ["NEW_INTENT"], "required", [LADDERS_NEED]),
        T("How often | is the telescope recalibrated?", ["NEW_INTENT"], "required", [(("telescope",), ())]),
        T("Let's go back | to the ladders.", ["NO_CHANGE"], "none", [LADDERS_NEED], alt=[["NEW_INTENT"]],
          ambiguous=True)]),
    ("X12", "constraint_addition", "anaphoric late detail", "fixture", [
        T(LADDERS, ["NEW_INTENT"], "required", [LADDERS_NEED]),
        T("Is that also | true at night?", ["CONSTRAINT_ADDITION"], "required", [(("ladders", "night"), ())],
          alt=[["NEW_INTENT"]], ambiguous=True)]),
    ("X13", "unchanged_continuation", "longer acknowledgement", "fixture", [
        T("When are | saplings planted?", ["NEW_INTENT"], "required", [(("saplings",), ())]),
        T("Okay that makes sense, | thank you.", ["NO_CHANGE"], "none")]),
    ("X14", "constraint_removal", "'regardless of' retraction", "fixture", [
        T("What does the keeper do | during a storm, | only when visibility is low?", ["NEW_INTENT"], "required",
          [(("storm", "visibility"), ())]),
        T("Regardless of | visibility.", ["CONSTRAINT_REMOVAL"], "required", [(("storm",), ("visibility",))])]),
]


def write(cases, out: Path, note: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for f in out.glob("*.json"):
        f.unlink()
    for cid, cat, desc, corpus, turns in cases:
        case = {"case_id": cid, "category": cat, "description": desc, "is_fixture": True, "held_out": False,
                "corpus": corpus, "turns": [{"utterance_id": f"u{n}", **t} for n, t in enumerate(turns, start=1)]}
        (out / f"{cid}.json").write_text(json.dumps(case, indent=2) + "\n")
    (out / "README.md").write_text(note)
    print(f"wrote {len(cases)} cases to {out}")


def main() -> None:
    write(CASES, OUT,
          "# eval/dev_adaptive (TEST FIXTURE ONLY)\n\nPhase 6 dev sessions over the fictional fixture corpora, built by "
          "`research/phase6/build_dev_suite.py`. Team-authored, not held-out, NOT REPORTABLE. Gold = expected change "
          "types, retrieval necessity, query terms, evidence; no answers.\n")
    write(STRESS_CASES, OUT.parent / "dev_adaptive_stress",
          "# eval/dev_adaptive_stress (TEST FIXTURE ONLY)\n\nPhase 6 stress sessions written after the code freeze with "
          "phrasing unlike the development cases; run once, reported as measured, code not changed in response. "
          "Team-authored, not held-out, NOT REPORTABLE.\n")


if __name__ == "__main__":
    main()
