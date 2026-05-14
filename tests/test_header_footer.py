from schemas import PageContent
from ocr.header_footer import suppress_repeating


def _page(n: int, text: str) -> PageContent:
    return PageContent(page=n, text=text, ocr_confidence=1.0, source="digital")


def test_repeating_header_and_footer_are_dropped_across_pages():
    pages = [
        _page(1, "Confidential - Pearson Specter Litt\nMatter PSL-2026-0418\n\nBody one.\n\nPage 1 of 4"),
        _page(2, "Confidential - Pearson Specter Litt\nMatter PSL-2026-0418\n\nBody two.\n\nPage 2 of 4"),
        _page(3, "Confidential - Pearson Specter Litt\nMatter PSL-2026-0418\n\nBody three.\n\nPage 3 of 4"),
        _page(4, "Confidential - Pearson Specter Litt\nMatter PSL-2026-0418\n\nBody four.\n\nPage 4 of 4"),
    ]
    cleaned = suppress_repeating(pages)
    for p in cleaned:
        assert "Confidential" not in p.text
        assert "Page" not in p.text or "Page" in "of 4"  # the footer is gone
    # body content survives
    assert any("Body one" in p.text for p in cleaned)
    assert any("Body four" in p.text for p in cleaned)


def test_short_documents_are_left_alone():
    pages = [
        _page(1, "Header\n\nContent A\n\nFooter"),
        _page(2, "Header\n\nContent B\n\nFooter"),
    ]
    # only 2 pages, below min_pages, suppression should not fire
    cleaned = suppress_repeating(pages, min_pages=3)
    assert cleaned[0].text == pages[0].text
    assert cleaned[1].text == pages[1].text


def test_unique_lines_are_kept():
    pages = [
        _page(1, "Header A\n\nFact one"),
        _page(2, "Header B\n\nFact two"),
        _page(3, "Header C\n\nFact three"),
    ]
    cleaned = suppress_repeating(pages)
    # no header repeats, all kept
    assert all("Header" in p.text for p in cleaned)
