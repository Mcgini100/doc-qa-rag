import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("INDEX_DIR", str(tmp_path / "index"))
    from app.main import app, limiter
    limiter.reset()
    with TestClient(app) as c:
        yield c


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["chunks"] > 20
    assert body["embedder"] == "local-wordllama-256" and body["llm"] is False


def test_ask_returns_cited_answer(client):
    r = client.post("/ask", json={"question": "Which body is designated as the Data Protection Authority?"})
    assert r.status_code == 200
    body = r.json()
    assert body["found"] and body["mode"] == "extractive"
    assert body["citations"] and {"doc", "pages", "sections", "text"} <= body["citations"][0].keys()
    assert any("Postal and Telecommunications Regulatory Authority" in c["text"] for c in body["citations"])


def test_ask_off_topic_says_not_found(client):
    body = client.post("/ask", json={"question": "Who won the 2010 football world cup?"}).json()
    assert body == {"question": "Who won the 2010 football world cup?", "answer": "not found in the documents",
                    "found": False, "mode": "refused", "citations": []}


@pytest.mark.parametrize("payload", [{}, {"question": "hi"}, {"question": "x" * 501},
                                     {"question": "valid question", "top_k": 0}])
def test_ask_validation(client, payload):
    assert client.post("/ask", json=payload).status_code == 422


def test_ask_rate_limited(client):
    codes = [client.post("/ask", json={"question": "What is the object of this Act?"}).status_code for _ in range(11)]
    assert codes[:10] == [200] * 10 and codes[10] == 429


def test_ingest_adds_document(client):
    text = ("Telecom customer charter. " * 5 + "Customers may port their number within 48 hours. ") * 20
    r = client.post("/ingest", files={"file": ("charter.txt", text.encode(), "text/plain")})
    assert r.status_code == 200, r.text
    assert r.json()["chunks_added"] >= 1
    assert "charter.txt" in client.get("/health").json()["documents"]
    hits = client.post("/ask", json={"question": "Within how many hours can customers port their number?"}).json()
    assert any(c["doc"] == "charter.txt" for c in hits["citations"])
    assert client.post("/ingest", files={"file": ("charter.txt", b"again", "text/plain")}).status_code == 409


def test_ingest_rejects_bad_type(client):
    r = client.post("/ingest", files={"file": ("x.exe", b"MZ", "application/octet-stream")})
    assert r.status_code == 400


def test_ingest_requires_token_when_configured(client, monkeypatch):
    monkeypatch.setenv("ADMIN_TOKEN", "s3cret")
    files = {"file": ("a.txt", b"some text " * 50, "text/plain")}
    assert client.post("/ingest", files=files).status_code == 403
    assert client.post("/ingest", files=files, headers={"X-Admin-Token": "s3cret"}).status_code == 200


def test_demo_page(client):
    r = client.get("/")
    assert r.status_code == 200 and "<form" in r.text
