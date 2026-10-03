"""RelevantContextSelector, ContextCompressor and context inheritance (Phase 6, brief §33-35). TEST FIXTURE corpus."""

from streamrag.context import ContextCompressor, RelevantContextSelector

from session_helpers import engine_of, p6_stack, run_turns  # noqa: F401  (fixture)

TURNS = ["What are the rules for ladders in the orchard?", "Specifically overnight.",
         "Now explain how the telescope is recalibrated.", "And what about the lens?"]


def test_structured_context_is_frame_scoped_with_provenance(p6_stack):
    p, _ = run_turns(p6_stack, TURNS)
    pkg = RelevantContextSelector(engine_of(p).memory).select("structured")
    assert pkg.frame_id == "T2" and pkg.intent_ids == ["I2", "I3"]
    assert pkg.excluded.get("frame_dormant") == 1                     # the ladders need is excluded, and counted
    for item in pkg.items:
        assert item.provenance, item
        if item.kind == "need":
            assert all(pv.utterance_id in ("u3", "u4") and pv.start is not None for pv in item.provenance)
        if item.kind == "claim":
            [pv] = item.provenance
            ev = engine_of(p).store.text(pv.evidence_id)
            assert ev[pv.start:pv.end] == item.text                     # claim is verbatim at its span
    assert all(i.text != "overnight" for i in pkg.items)
    assert pkg.size.chars == sum(len(i.text) for i in pkg.items) and pkg.size.tokens is None


def test_ablation_modes(p6_stack):
    p, _ = run_turns(p6_stack, TURNS)
    sel = RelevantContextSelector(engine_of(p).memory, count=lambda s: len(s.split()), tokenizer_name="words")
    none = sel.select("none")
    full = sel.select("full_transcript")
    assert [i.text for i in none.items] == [TURNS[-1]]
    assert [i.text for i in full.items] == TURNS
    assert full.size.tokens == sum(len(t.split()) for t in TURNS) and full.size.tokenizer == "words"


def test_constraint_scope_is_inherited_only_by_its_need(p6_stack):
    p, _ = run_turns(p6_stack, TURNS[:2] + ["What about crates?"])
    eng = engine_of(p)
    pkg = RelevantContextSelector(eng.memory).select("structured")
    ks = {(i.intent_id, i.text) for i in pkg.items if i.kind == "constraint"}
    assert ks == {("I1", "overnight")}


def test_retracted_constraint_is_excluded_and_counted(p6_stack):
    p, _ = run_turns(p6_stack, TURNS[:2] + ["Actually, ignore the overnight restriction."])
    pkg = RelevantContextSelector(engine_of(p).memory).select("structured")
    assert not [i for i in pkg.items if i.kind == "constraint"]
    assert pkg.excluded.get("constraint_retracted_or_replaced") == 1


def test_compression_keeps_provenance_beyond_window(p6_stack):
    p, _ = run_turns(p6_stack, TURNS)
    mem = engine_of(p).memory
    mem.window = 1
    mem.observe_utterance("u5", "Okay.", True)
    cc = ContextCompressor(mem).compress({f"u{n}": t for n, t in enumerate(TURNS + ["Okay."], 1)})
    old = [u for u in cc.utterances if u.verbatim is None]
    assert [u.utterance_id for u in old] == ["u1", "u2", "u3", "u4"]
    needs = {d.item_id: d for u in old for d in u.derived if d.kind == "need"}
    assert set(needs) == {"I1", "I2", "I3"}
    assert all(pv.utterance_id and pv.start is not None for d in needs.values() for pv in d.provenance)
    ks = [d for u in old for d in u.derived if d.kind == "constraint"]
    assert [(k.text, k.provenance[0].utterance_id) for k in ks] == [("overnight", "u2")]
    assert cc.raw.chars == sum(len(t) for t in TURNS) + len("Okay.")
    assert cc.compressed.chars < cc.raw.chars and cc.ratio_chars == round(cc.compressed.chars / cc.raw.chars, 4)
