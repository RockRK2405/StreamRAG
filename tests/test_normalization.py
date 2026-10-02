from streamrag.config.settings import NormalizationConfig, SectionsConfig
from streamrag.corpus import build_corpus
from streamrag.corpus.normalize import normalize_pages, page_at

from conftest import FIX

N, S = NormalizationConfig(), SectionsConfig()


def norm(text, paged=False, pages=None, **kw):
    return normalize_pages(pages or [text], paged, N.model_copy(update=kw), S, "txt")


def test_unicode_invisible_and_whitespace():
    out = norm("The ﬁle​ is  here.\r\nNext\tline.").text
    assert out == "The file is here. Next line."


def test_dehyphenation_and_wrapped_lines():
    out = norm("Every instrument needs calibra-\ntion before use and\nthe result is logged.").text
    assert out == "Every instrument needs calibration before use and the result is logged."


def test_headings_and_list_items_are_not_joined():
    out = norm("1. Scope\nThis applies to all.\n- item one\n- item two").text
    assert out.split("\n\n")[0] == "1. Scope"
    assert "- item one\n- item two" in out


def test_consecutive_duplicate_lines_removed_and_logged():
    r = norm("Line A\nLine A\nLine B")
    assert r.text == "Line A Line B" and any("duplicate" in m for m in r.log)


def test_paged_header_footer_and_page_numbers_removed_with_offsets():
    pages = [f"ACME HANDBOOK\nBody text of page {i} ends here.\nPage {i} of 3" for i in (1, 2, 3)]
    r = norm(None, paged=True, pages=pages)
    assert "ACME HANDBOOK" not in r.text and "Page 1 of 3" not in r.text
    assert len(r.page_offsets) == 3 and any("header/footer" in m for m in r.log)
    for page, start, end in r.page_offsets:
        assert f"page {page}" in r.text[start:end]
        assert page_at(r.page_offsets, start) == page


def test_page_numbers_kept_for_unpaged_text():
    assert "Page 1 of 3" in norm("Page 1 of 3").text            # only paged sources lose page markers


def test_paragraph_continuing_across_page_break_is_joined():
    r = norm(None, paged=True, pages=["Instruments that fail go to the repair", "shelf until fixed."])
    assert "repair shelf until fixed." in r.text


def test_pdf_fixture_end_to_end(cfg_factory):
    built = build_corpus(cfg_factory(corpus=FIX / "corpus_pdf"))
    doc = built.documents[0]
    assert "TEST FIXTURE ONLY" not in doc.text          # repeated header removed
    assert "Page 2 of 3" not in doc.text                # footer removed
    assert "calibration before first use" in doc.text   # de-hyphenated
    assert "repair shelf until" in doc.text             # cross-page paragraph joined
    sec2 = next(s for s in doc.sections if s.section_id == "2")
    assert (sec2.page_start, sec2.page_end) == (2, 3)


def test_normalization_is_deterministic():
    text = (FIX / "corpus" / "Doc_07_fixture_orchard.txt").read_text()
    assert norm(text).text == norm(text).text
