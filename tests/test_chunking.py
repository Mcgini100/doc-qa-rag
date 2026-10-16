from app.ingest import Page, chunk_pages, clean_pages, count_tokens, drop_front_matter, ingest_file
from tests.conftest import ACT_PDF


def make_pages(n_pages=6, sentences=60):
    pages = []
    for p in range(1, n_pages + 1):
        lines = ["Gazette running header No. 5/2021", str(40 + p)]
        if p == 2:
            lines.append(f"{p} Section heading number {p}")
        lines += [f"Sentence {p}-{i} about data protection rules and duties of controllers." for i in range(sentences)]
        pages.append(Page("doc.txt", p, "\n".join(lines)))
    return pages


def test_clean_removes_running_headers_and_page_numbers():
    cleaned = clean_pages(make_pages())
    for p in cleaned:
        assert "Gazette running header" not in p.text
        assert str(40 + p.page) not in p.text.splitlines()


def test_clean_joins_hyphenated_line_breaks():
    (p,) = clean_pages([Page("d", 1, "protec-\ntion of privacy")])
    assert "protection of privacy" in p.text


def test_chunk_sizes_and_overlap():
    chunks = chunk_pages(clean_pages(make_pages()), chunk_tokens=200, overlap_tokens=25)
    assert len(chunks) > 5
    assert all(c.n_tokens <= 200 for c in chunks)
    assert all(c.n_tokens >= 100 for c in chunks[:-1])  # chunks are reasonably full
    for a, b in zip(chunks, chunks[1:]):
        tail = a.text.splitlines()[-1]
        assert b.text.startswith(tail), "next chunk must start with the previous chunk's last unit"


def test_chunk_page_metadata_is_exact():
    pages = clean_pages(make_pages())
    for c in chunk_pages(pages, chunk_tokens=200, overlap_tokens=25):
        for line in c.text.splitlines():
            page_of_line = int(line.split()[1].split("-")[0]) if line.startswith("Sentence") else None
            if page_of_line:
                assert c.page_start <= page_of_line <= c.page_end


def test_section_headings_are_tracked():
    chunks = chunk_pages(clean_pages(make_pages()), chunk_tokens=200, overlap_tokens=25)
    assert any("Section 2: Section heading number 2" in c.sections for c in chunks)


def test_invalid_overlap_rejected():
    import pytest
    with pytest.raises(ValueError):
        chunk_pages(make_pages(), chunk_tokens=100, overlap_tokens=100)


def test_act_ingestion():
    chunks = ingest_file(ACT_PDF)
    assert 25 <= len(chunks) <= 60
    # target 500-800 word-level tokens per chunk with 75-token overlap
    assert all(c.n_tokens <= 600 for c in chunks)
    assert sum(c.n_tokens >= 500 for c in chunks) >= len(chunks) - 1
    # the 'Arrangement of Sections' (pages 1-2) is dropped; the body starts on page 3
    assert min(c.page_start for c in chunks) == 3
    first = chunks[0]
    assert "Section 1: Short title" in first.sections
    ids = [c.id for c in chunks]
    assert len(ids) == len(set(ids))


def test_drop_front_matter_keeps_documents_without_toc():
    pages = [Page("d", 1, "Just text"), Page("d", 2, "More text")]
    assert drop_front_matter(pages) == pages


def test_count_tokens():
    assert count_tokens("Data, controller's duty.") == 7  # Data , controller ' s duty .
