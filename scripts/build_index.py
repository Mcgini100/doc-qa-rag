"""Build (or rebuild) the index for the configured embedding backend.

    python scripts/build_index.py                       # local WordLlama embeddings
    OPENAI_API_KEY=... EMBEDDING_BACKEND=openai python scripts/build_index.py
"""
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_settings  # noqa: E402
from app.embeddings import make_embedder  # noqa: E402
from app.main import build_engine  # noqa: E402

if __name__ == "__main__":
    s = get_settings()
    name = make_embedder(s.resolved_backend, s.openai_embedding_model, s.openai_key).name
    shutil.rmtree(s.index_dir / name, ignore_errors=True)
    engine = build_engine()
    docs = sorted({c.doc for c in engine.index.chunks})
    print(f"built {s.index_dir / name}: {len(engine.index.chunks)} chunks from {docs}")
