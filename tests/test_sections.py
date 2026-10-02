from streamrag.config.settings import NormalizationConfig, SectionsConfig
from streamrag.corpus.normalize import normalize_pages
from streamrag.corpus.sections import parse_sections

S = SectionsConfig()


def parse(text, fmt="txt", pages=None):
    n = normalize_pages(pages or [text], bool(pages), NormalizationConfig(), S, fmt)
    return parse_sections("D", n.text, n.page_offsets, fmt, "Default Title", S)


def ids(sections):
    return [s.section_id for s in sections]


def test_numbered_hierarchy_and_section_sign():
    title, secs = parse("Intro text.\n\n1. Alpha\nA body.\n\n2. Beta\nB body.\n\n2.1 Beta One\nB1 body.\n\n§ 3 Gamma\nC body.")
    assert ids(secs) == ["0", "1", "2", "2.1", "3"]
    b1 = secs[3]
    assert b1.parent_id == "2" and b1.path == ["Beta", "Beta One"] and b1.native_id and b1.level == 2


def test_sequential_numbered_list_is_not_headings_but_following_heading_is():
    # regression: a list "1/2/3" followed by heading "2. Pruning" must keep the heading
    _, secs = parse("1. Planting\nBody.\n\nVarieties:\n\n1. Red Apple\n2. Gold Pear\n3. Blue Plum\n\n2. Pruning\nBody two.")
    assert ids(secs) == ["1", "2"]
    assert any("Red Apple" in p.text for p in secs[0].paragraphs)


def test_list_items_with_periods_never_headings():
    _, secs = parse("1. Scope\nWorkers must:\n1. Clean the blades.\n2. Hang the tools.")
    assert ids(secs) == ["1"]


def test_markdown_title_and_numbered_markdown_headings():
    title, secs = parse("# Manual\n\nPreamble.\n\n## 1 First\nx\n\n### 1.1 Sub\ny\n\n## Unnumbered\nz", fmt="md")
    assert title == "Manual"
    assert ids(secs) == ["0", "1", "1.1", "u2"]              # derived id prefixed in a natively-numbered doc


def test_unnumbered_markdown_gets_ordinal_paths():
    _, secs = parse("## Alpha\na\n\n### Alpha Child\nb\n\n## Beta\nc", fmt="md")
    assert ids(secs) == ["1", "1.1", "2"] and not any(s.native_id for s in secs)


def test_duplicate_native_numbers_get_suffix():
    _, secs = parse("1. One\na\n\n1. Again\nb")
    assert ids(secs) == ["1", "1-2"]


def test_no_headings_single_section_and_caps_title():
    title, secs = parse("SAFETY NOTES\nThe notes start here and continue.\n\nSecond paragraph.")
    assert title == "Safety Notes" and ids(secs) == ["1"] and len(secs[0].paragraphs) == 2


def test_headingless_pdf_gets_page_sections():
    _, secs = parse(None, pages=["First page text here.", "Second page text here."])
    assert ids(secs) == ["p1", "p2"] and secs[1].page_start == 2
