"""FastAPI service: POST /ask, POST /ingest, GET /health and a small demo page at /."""
from __future__ import annotations

import logging
import re
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Header, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from app.config import get_settings
from app.embeddings import make_embedder
from app.index import DocIndex
from app.ingest import SUPPORTED, ingest_dir, ingest_file
from app.qa import QAEngine

log = logging.getLogger("doc-qa-rag")
settings = get_settings()
limiter = Limiter(key_func=get_remote_address)
STATIC = Path(__file__).parent / "static"


def build_engine() -> QAEngine:
    """Load the index for the configured embedding backend, building it from DOCS_DIR if missing."""
    s = get_settings()
    embedder = make_embedder(s.resolved_backend, s.openai_embedding_model, s.openai_key)
    index_path = s.index_dir / embedder.name
    if (index_path / "chunks.json").exists():
        index = DocIndex.load(index_path)
    else:
        chunks = ingest_dir(s.docs_dir, s.chunk_tokens, s.chunk_overlap)
        if not chunks:
            raise RuntimeError(f"no documents found in {s.docs_dir}")
        index = DocIndex.build(chunks, embedder)
        index.save(index_path)
    llm = None
    if s.use_llm and s.openai_key:
        from openai import OpenAI

        llm = OpenAI(api_key=s.openai_key)
    return QAEngine(index, embedder, llm, s.openai_chat_model, s.top_k)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.engine = build_engine()
    log.info("index ready: %d chunks, embedder=%s, llm=%s", len(app.state.engine.index.chunks),
             app.state.engine.embedder.name, bool(app.state.engine.llm))
    yield


app = FastAPI(title="Document Q&A (RAG)", version="1.0.0", lifespan=lifespan,
              description="Cited answers over the Zimbabwe Cyber and Data Protection Act [Chapter 12:07].")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=500, examples=["Who is the Data Protection Authority?"])
    top_k: int | None = Field(None, ge=1, le=10)


class Citation(BaseModel):
    n: int
    chunk_id: str
    doc: str
    pages: str
    sections: list[str]
    score: float
    dense: float
    bm25: float
    text: str


class AskResponse(BaseModel):
    question: str
    answer: str
    found: bool
    mode: str = Field(..., description="'llm', 'extractive' (no LLM configured) or 'refused'")
    citations: list[Citation]


class IngestResponse(BaseModel):
    file: str
    chunks_added: int
    total_chunks: int


class HealthResponse(BaseModel):
    status: str
    chunks: int
    documents: list[str]
    embedder: str
    llm: bool


@app.get("/", include_in_schema=False)
def demo_page():
    return FileResponse(STATIC / "index.html")


@app.get("/health", response_model=HealthResponse)
def health(request: Request):
    engine: QAEngine = request.app.state.engine
    docs = sorted({c.doc for c in engine.index.chunks})
    return HealthResponse(status="ok", chunks=len(engine.index.chunks), documents=docs,
                          embedder=engine.embedder.name, llm=engine.llm is not None)


@app.post("/ask", response_model=AskResponse)
@limiter.limit(settings.rate_limit_ask)
def ask(request: Request, body: AskRequest):
    engine: QAEngine = request.app.state.engine
    ans = engine.ask(body.question.strip(), body.top_k)
    return AskResponse(question=ans.question, answer=ans.answer, found=ans.found, mode=ans.mode,
                       citations=[Citation(**c) for c in ans.citations])


@app.post("/ingest", response_model=IngestResponse)
@limiter.limit(settings.rate_limit_ingest)
async def ingest(request: Request, file: UploadFile = File(...), x_admin_token: str | None = Header(default=None)):
    s = get_settings()
    if s.admin_token and x_admin_token != s.admin_token:
        raise HTTPException(status_code=403, detail="Ingest requires a valid X-Admin-Token header.")
    name = re.sub(r"[^\w.\-]+", "_", Path(file.filename or "upload").name)
    if Path(name).suffix.lower() not in SUPPORTED:
        raise HTTPException(status_code=400, detail=f"Unsupported file type. Use one of {sorted(SUPPORTED)}.")
    data = await file.read()
    if len(data) > 20 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="File larger than 20 MB.")
    engine: QAEngine = request.app.state.engine
    if any(c.doc == name for c in engine.index.chunks):
        raise HTTPException(status_code=409, detail=f"{name} is already indexed.")
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / name
        path.write_bytes(data)
        try:
            chunks = ingest_file(path, s.chunk_tokens, s.chunk_overlap)
        except Exception as e:  # unreadable PDF, wrong encoding, ...
            raise HTTPException(status_code=400, detail=f"Could not read file: {e}") from e
    if not chunks:
        raise HTTPException(status_code=400, detail="No extractable text found.")
    engine.index.add(chunks, engine.embedder)
    engine.index.save(s.index_dir / engine.embedder.name)
    return IngestResponse(file=name, chunks_added=len(chunks), total_chunks=len(engine.index.chunks))
