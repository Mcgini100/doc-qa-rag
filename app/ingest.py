"""Load documents, clean the text and split it into overlapping chunks with metadata.

Token counts use a simple word/punctuation tokenizer (``\\w+|[^\\w\\s]``) so chunking
is deterministic and works offline. One such token is roughly 1.1-1.3 OpenAI
(cl100k) tokens for English legal text.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

TOKEN_RE = re.compile(r"\w+|[^\w\s]")
SECTION_RE = re.compile(r"^(\d{1,3}[A-Z]?)\s+([A-Z][A-Za-z0-9 ,’'\-()/.:\[\]]{2,90})$")
PART_RE = re.compile(r"^PART\s+([IVXLC]+)\b", re.IGNORECASE)
SUPPORTED = {".pdf", ".txt", ".md", ".html", ".htm"}


@dataclass
class Page:
    doc: str
    page: int  # 1-based page number in the source file
    text: str


@dataclass
class Chunk:
    id: str
    doc: str
    page_start: int
    page_end: int
    sections: list[str]
    text: str
    n_tokens: int

    def to_dict(self) -> dict:
        return asdict(self)


def count_tokens(text: str) -> int:
    return len(TOKEN_RE.findall(text))


# --------------------------------------------------------------------------- loading
def load_pages(path: Path) -> list[Page]:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        return [Page(path.name, i + 1, p.extract_text() or "") for i, p in enumerate(reader.pages)]
    raw = path.read_text(encoding="utf-8", errors="replace")
    if suffix in {".html", ".htm"}:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(raw, "html.parser")
        for tag in soup(["script", "style", "nav", "header", "footer"]):
            tag.decompose()
        raw = soup.get_text("\n")
    # Plain text has no pages; form-feed characters, if present, separate pages.
    return [Page(path.name, i + 1, t) for i, t in enumerate(raw.split("\f"))]


# --------------------------------------------------------------------------- cleaning
def _norm_line(line: str) -> str:
    return re.sub(r"[\d\s]+", "", line).lower()


def clean_pages(pages: list[Page]) -> list[Page]:
    """Normalise whitespace, join hyphenated line breaks, and drop running headers/footers
    (short lines that repeat on many pages) and bare page numbers."""
    split = [[re.sub(r"[ \t ]+", " ", ln).strip() for ln in p.text.splitlines()] for p in pages]
    # Only the first/last few lines of a page can be headers or footers.
    def edges(lines):
        nonempty = [ln for ln in lines if ln]
        return nonempty[:4] + nonempty[-3:]

    counts = Counter()
    for lines in split:
        counts.update({_norm_line(ln) for ln in edges(lines) if len(ln) < 90})
    min_repeat = max(3, int(0.4 * len(pages)))
    repeated = {k for k, c in counts.items() if c >= min_repeat and k}
    out = []
    for p, lines in zip(pages, split):
        edge = set(edges(lines))
        kept = [ln for ln in lines if ln and not ln.isdigit()
                and not (ln in edge and _norm_line(ln) in repeated)]
        text = "\n".join(kept)
        text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)  # de-hyphenate across line breaks
        out.append(Page(p.doc, p.page, text))
    return out


def drop_front_matter(pages: list[Page]) -> list[Page]:
    """Drop a statute's 'Arrangement of Sections' (table of contents) pages: they repeat every
    heading and would otherwise match almost any question without answering it."""
    toc = [i for i, p in enumerate(pages) if "ARRANGEMENT OF SECTIONS" in p.text.upper()]
    if not toc:
        return pages
    body_start = next((i for i, p in enumerate(pages) if i > toc[0] and "ENACTED by" in p.text), None)
    if body_start is None:
        return [p for i, p in enumerate(pages) if i not in toc]
    return pages[: toc[0]] + pages[body_start:]


# --------------------------------------------------------------------------- chunking
@dataclass
class _Unit:
    text: str
    page: int
    section: str | None
    n_tokens: int


def _units(pages: list[Page], min_tokens: int = 40) -> list[_Unit]:
    """Group lines into small paragraph-like units, tagged with page and current section.
    A heading always starts a new unit, so chunks tend to break at section boundaries."""
    units: list[_Unit] = []
    section = None
    buf: list[str] = []
    buf_page = None

    def flush():
        nonlocal buf
        if buf:
            text = " ".join(buf)
            units.append(_Unit(text, buf_page, section, count_tokens(text)))
            buf = []

    for p in pages:
        for line in p.text.splitlines():
            m_part = PART_RE.match(line)
            m_sec = SECTION_RE.match(line)
            is_heading = bool(m_part) or (bool(m_sec) and not line.endswith((";", ",", "—")))
            if is_heading:
                flush()
                if not m_part:
                    # Part numbers are not added to the label: the consequential amendments
                    # quote Parts of other Acts, which would make "Part X" labels misleading.
                    section = f"Section {m_sec.group(1)}: {m_sec.group(2).strip()}"
            if not buf:
                buf_page = p.page
            buf.append(line)
            if count_tokens(" ".join(buf)) >= min_tokens and not is_heading:
                flush()
        flush()  # never let a unit span a page break, so page metadata stays exact
    return units


def _split_long(unit: _Unit, max_tokens: int) -> list[_Unit]:
    if unit.n_tokens <= max_tokens:
        return [unit]
    words = unit.text.split()
    parts, cur = [], []
    for w in words:
        cur.append(w)
        if count_tokens(" ".join(cur)) >= max_tokens:
            parts.append(" ".join(cur))
            cur = []
    if cur:
        parts.append(" ".join(cur))
    return [_Unit(t, unit.page, unit.section, count_tokens(t)) for t in parts]


def chunk_pages(pages: list[Page], chunk_tokens: int = 600, overlap_tokens: int = 75) -> list[Chunk]:
    """Greedy packing of units into chunks of about ``chunk_tokens`` tokens; each chunk starts
    with the last ~``overlap_tokens`` tokens of the previous chunk."""
    if not 0 <= overlap_tokens < chunk_tokens:
        raise ValueError("overlap must be >= 0 and smaller than chunk size")
    units = [u for unit in _units(pages) for u in _split_long(unit, chunk_tokens)]
    chunks: list[Chunk] = []
    cur: list[_Unit] = []
    i = 0
    while i < len(units):
        u = units[i]
        if cur and sum(x.n_tokens for x in cur) + u.n_tokens > chunk_tokens:
            chunks.append(_make_chunk(cur, len(chunks)))
            # carry the tail of the finished chunk forward as overlap
            tail, tot = [], 0
            for x in reversed(cur):
                if tot >= overlap_tokens:
                    break
                tail.insert(0, x)
                tot += x.n_tokens
            cur = tail if tot < chunk_tokens else []
            if cur and sum(x.n_tokens for x in cur) + u.n_tokens > chunk_tokens:
                cur = []
            continue
        cur.append(u)
        i += 1
    if cur:
        chunks.append(_make_chunk(cur, len(chunks)))
    return chunks


def _make_chunk(units: list[_Unit], n: int) -> Chunk:
    text = "\n".join(u.text for u in units)
    # Every section the chunk touches, in order (including the one it starts inside).
    sections = list(dict.fromkeys(u.section for u in units if u.section))
    doc_pages = [u.page for u in units]
    return Chunk(
        id=f"c{n:04d}",
        doc="",
        page_start=min(doc_pages),
        page_end=max(doc_pages),
        sections=sections,
        text=text,
        n_tokens=count_tokens(text),
    )


def ingest_file(path: Path, chunk_tokens: int = 600, overlap_tokens: int = 75) -> list[Chunk]:
    pages = drop_front_matter(clean_pages(load_pages(path)))
    chunks = chunk_pages(pages, chunk_tokens, overlap_tokens)
    stem = re.sub(r"\W+", "_", path.stem)[:40]
    for c in chunks:
        c.doc = path.name
        c.id = f"{stem}:{c.id}"
    return chunks


def ingest_dir(docs_dir: Path, chunk_tokens: int = 600, overlap_tokens: int = 75) -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in sorted(docs_dir.iterdir()):
        if path.suffix.lower() in SUPPORTED:
            chunks.extend(ingest_file(path, chunk_tokens, overlap_tokens))
    return chunks
