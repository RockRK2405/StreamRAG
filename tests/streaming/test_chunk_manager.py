from streamrag.models import TranscriptChunk
from streamrag.streaming.chunk_manager import TranscriptChunkManager


def chunk(i, text, uid="u1", t=None, **kw):
    return TranscriptChunk(session_id="s", utterance_id=uid, payload={"chunk_index": i, "timestamp_s": t if t is not None else i * 0.3,
                                                                      "text": text, **kw})


def test_append_and_transcript_with_punctuation_joining():
    m = TranscriptChunkManager("s")
    for i, t in enumerate(["I need", "the fog signal", ", during a storm", "?"]):
        assert m.append_chunk(chunk(i, t), i * 300.0).status == "accepted"
    assert m.get_current_transcript("u1") == "I need the fog signal, during a storm?"


def test_duplicate_is_ignored():
    m = TranscriptChunkManager("s")
    m.append_chunk(chunk(0, "hello there"), 0)
    r = m.append_chunk(chunk(0, "hello there"), 10)
    assert r.status == "duplicate" and not r.changed and m.get_current_transcript() == "hello there"


def test_conflicting_duplicate_and_explicit_revision():
    m = TranscriptChunkManager("s")
    m.append_chunk(chunk(0, "fog signal"), 0)
    m.append_chunk(chunk(1, "during the storm"), 300)
    r = m.append_chunk(chunk(1, "during a storm"), 400)
    assert r.status == "revision" and r.changed and m.get_current_transcript() == "fog signal during a storm"
    r2 = m.append_chunk(chunk(2, "fog horn", replaces_chunk_index=0), 500)
    assert r2.status == "revision" and m.get_current_transcript() == "fog horn during a storm"
    assert m.state("u1").revisions == 2


def test_gap_then_out_of_order_fill():
    m = TranscriptChunkManager("s")
    m.append_chunk(chunk(0, "one"), 0)
    r = m.append_chunk(chunk(2, "three"), 600)
    assert r.status == "gap" and r.missing == [1] and m.get_current_transcript() == "one three"
    r = m.append_chunk(chunk(1, "two"), 700)
    assert r.status == "out_of_order" and r.missing == [] and m.get_current_transcript() == "one two three"


def test_empty_chunk_is_keepalive():
    m = TranscriptChunkManager("s")
    m.append_chunk(chunk(0, "one"), 0)
    r = m.append_chunk(chunk(1, "   "), 300)
    assert r.status == "empty" and not r.changed and m.missing_indices("u1") == []


def test_finalize_rejects_late_chunks_and_close():
    m = TranscriptChunkManager("s")
    m.append_chunk(chunk(0, "one"), 0)
    assert m.finalize_utterance("u1", 500, "endpoint") and not m.finalize_utterance("u1", 600, "endpoint")
    assert m.append_chunk(chunk(1, "late"), 700).status == "rejected_finalized"
    assert m.get_current_transcript("u1") == "one"
    m.close()
    assert m.append_chunk(chunk(0, "new", uid="u2"), 800).status == "rejected_closed"


def test_new_utterance_recent_chunks_reset_and_offsets():
    m = TranscriptChunkManager("s")
    m.append_chunk(chunk(0, "a", uid="u1", t=0.5, utterance_offset_s=2.0), 2500)
    assert m.state("u1").start_ms == 2000.0
    m.append_chunk(chunk(0, "b", uid="u2"), 5000)
    assert m.open_utterance() == "u2" and [c.text for c in m.get_recent_chunks(5, "u1")] == ["a"]
    m.reset()
    assert m.order == [] and m.get_current_transcript() == ""
