import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
ACT_PDF = ROOT / "data" / "docs" / "Cyber_and_Data_Protection_Act_Cap_12-07_No_5_of_2021.pdf"


@pytest.fixture(scope="session")
def local_embedder():
    from app.embeddings import LocalEmbedder
    return LocalEmbedder()


@pytest.fixture(autouse=True)
def offline_env(monkeypatch):
    """Tests never call OpenAI: force the local embedder and extractive answers."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("EMBEDDING_BACKEND", "local")
    monkeypatch.delenv("ADMIN_TOKEN", raising=False)
