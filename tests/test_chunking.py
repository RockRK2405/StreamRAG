import pytest

from streamrag.config.settings import ChunkingConfig
from streamrag.corpus import build_corpus
from streamrag.corpus.chunker import chunk_section, count_tokens, sentence_spans
from streamrag.models.corpus import CorpusSection, Paragraph


def section(paras: list[str]) -> tuple[CorpusSection, str]:
    text, ps, pos = "", [], 0
    for i, p in enumerate(paras):
        if i:
            text += "\n\n"
        start = len(text)
        text += p
        ps.append(Paragraph(text=p, char_start=start, char_end=len(text)))
    return CorpusSection(section_id="1", document_id="D", title="T", level=1, ordinal=0, char_start=0,
                         char_end=len(text), paragraphs=ps), text


SENT = "The fixture rule number {i} applies to every worker on the site today."   # 13 tokens


def test_sentence_split_respects_abbreviations():
    t = "Bring tools, e.g. a saw. Then start. Dr. Smith checks it."
    parts = [t[s:e] for s, e in sentence_spans(t)]
    assert parts == ["Bring tools, e.g. a saw.", "Then start.", "Dr. Smith checks it."]


def test_paragraph_packing_respects_target_and_max():
    paras = [" ".join(SENT.format(i=i) for i in range(j * 3, j * 3 + 3)) for j in range(6)]   # 39 tok each
    sec, text = section(paras)
    spans = chunk_section(sec, text, ChunkingConfig(strategy="paragraph", target_tokens=80, max_tokens=100,
                                                    overlap_tokens=0, min_tokens=10))
    toks = [count_tokens(text[s.start:s.end]) for s in spans]
    assert all(t <= 100 for t in toks) and len(spans) == 3


def test_long_paragraph_split_on_sentences_with_overlap():
    para = " ".join(SENT.format(i=i) for i in range(12))          # 156 tokens, one paragraph
    sec, text = section([para])
    cfg = ChunkingConfig(strategy="paragraph", target_tokens=50, max_tokens=60, overlap_tokens=14, min_tokens=5)
    spans = chunk_section(sec, text, cfg)
    assert len(spans) > 2
    for a, b in zip(spans, spans[1:]):
        assert b.start < a.end                                     # one trailing sentence overlaps
        assert text[b.start:b.start + 3] == "The"                  # never starts mid-sentence
    assert all(not s.starts_mid_sentence for s in spans)


def test_lead_in_stays_with_list():
    sec, text = section(["x " * 40 + "end.", "Workers must check:", "- item one\n- item two\n- item three"])
    spans = chunk_section(sec, text, ChunkingConfig(strategy="paragraph", target_tokens=42, max_tokens=60,
                                                    overlap_tokens=0, min_tokens=5))
    lead = next(s for s in spans if "Workers must check:" in text[s.start:s.end])
    assert "item three" in text[lead.start:lead.end]


def test_section_strategy_and_fallback():
    small, text = section(["Short body one.", "Short body two."])
    assert len(chunk_section(small, text, ChunkingConfig(strategy="section"))) == 1
    big, text2 = section([" ".join(SENT.format(i=i) for i in range(30))])
    assert len(chunk_section(big, text2, ChunkingConfig(strategy="section", target_tokens=100, max_tokens=150,
                                                        overlap_tokens=0))) > 1


def test_token_strategy_windows():
    sec, text = section([" ".join(f"w{i}" for i in range(100))])
    spans = chunk_section(sec, text, ChunkingConfig(strategy="token", target_tokens=40, max_tokens=40,
                                                    overlap_tokens=10, min_tokens=0))
    assert [count_tokens(text[s.start:s.end]) for s in spans] == [40, 40, 40]
    assert text[spans[1].start:].startswith("w30")


@pytest.mark.parametrize("strategy", ["section", "paragraph", "token"])
def test_corpus_chunks_are_exact_slices_never_cross_sections_and_deterministic(cfg_factory, strategy):
    cfg = cfg_factory(**{"chunking.strategy": strategy})
    a, b = build_corpus(cfg), build_corpus(cfg)
    docs = {d.document_id: d for d in a.documents}
    for c in a.chunks:
        d = docs[c.document_id]
        assert d.text[c.char_start:c.char_end] == c.text
        sec = next(s for s in d.sections if s.section_id == c.section_id)
        assert sec.char_start <= c.char_start and c.char_end <= sec.char_end
        assert c.token_count == count_tokens(c.text) <= cfg.chunking.max_tokens
    assert [c.model_dump() for c in a.chunks] == [c.model_dump() for c in b.chunks]
