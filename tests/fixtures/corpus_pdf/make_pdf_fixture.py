"""Regenerate fixture_paged_handbook.pdf (TEST FIXTURE ONLY). Requires PyMuPDF (fitz); run manually.

The PDF exercises: repeated header/footer removal, page-number lines, hyphenation across a line break,
a paragraph continuing across a page break, numbered headings, page tracking."""
import fitz  # PyMuPDF

HEADER = "FIXTURE PAGED HANDBOOK - TEST FIXTURE ONLY"
PAGES = [
    ["1 Arrival",
     "Visitors to the fixture workshop sign the arrival sheet at the front desk.",
     "Badges are collected from the desk clerk before entering the workshop floor."],
    ["2 Equipment",
     "Every bench instrument requires calibra-",
     "tion before first use each day.",
     "The calibration result is written on the bench card, and instruments that fail",
     "the check are tagged and moved to the repair"],
    ["shelf until the technician returns them to service.",
     "3 Departure",
     "Visitors return their badges to the desk clerk when they leave the workshop."],
]
doc = fitz.open()
for i, lines in enumerate(PAGES, start=1):
    page = doc.new_page()
    page.insert_text((72, 50), HEADER, fontsize=9)
    y = 100
    for ln in lines:
        if ln[:1].isdigit() and len(ln) < 20:
            y += 10
        page.insert_text((72, y), ln, fontsize=11)
        y += 18
    page.insert_text((280, 800), f"Page {i} of {len(PAGES)}", fontsize=9)
doc.save("fixture_paged_handbook.pdf")
