import numpy as np
import pytest

from app.index import DocIndex, bm25_tokens
from app.ingest import ingest_file
from app.qa import NOT_FOUND, QAEngine
from tests.conftest import ACT_PDF


@pytest.fixture(scope="module")
def act_index(local_embedder):
    return DocIndex.build(ingest_file(ACT_PDF), local_embedder)


@pytest.mark.parametrize("question,evidence", [
    ("Within how many hours must a breach be notified to the Authority?", "twenty-four (24) hours"),
    ("Which body is designated as the Data Protection Authority?", "designated as the data protection authority"),
    ("Where can a person appeal a decision of the Authority?", "administrative court"),
])
@pytest.mark.parametrize("mode", ["bm25", "hybrid"])
def test_relevant_chunk_in_top3(act_index, local_embedder, question, evidence, mode):
    hits = act_index.search(question, local_embedder, k=3, mode=mode)
    assert any(evidence in " ".join(h.chunk.text.lower().split()) for h in hits)


def test_hits_are_ranked_and_carry_metadata(act_index, local_embedder):
    hits = act_index.search("security breach notification", local_embedder, k=5)
    assert [h.rank for h in hits] == [1, 2, 3, 4, 5]
    assert all(hits[i].score >= hits[i + 1].score for i in range(4))
    assert all(h.chunk.doc.endswith(".pdf") and h.chunk.page_start >= 3 for h in hits)


def test_save_load_roundtrip(act_index, local_embedder, tmp_path):
    act_index.save(tmp_path)
    loaded = DocIndex.load(tmp_path)
    assert len(loaded.chunks) == len(act_index.chunks)
    np.testing.assert_allclose(loaded.vectors, act_index.vectors)
    q = "rights of a data subject"
    assert [h.chunk.id for h in loaded.search(q, local_embedder)] == [h.chunk.id for h in act_index.search(q, local_embedder)]


def test_off_topic_question_is_refused(act_index, local_embedder):
    ans = QAEngine(act_index, local_embedder).ask("Who won the 2010 football world cup?")
    assert ans.answer == NOT_FOUND and ans.found is False and ans.mode == "refused"


def test_bm25_tokens_drop_stopwords():
    assert bm25_tokens("What is the duty of the data controller?") == ["duty", "data", "controller"]


class _FakeLLM:
    """Stands in for openai.OpenAI: records the prompt and returns a canned reply."""

    def __init__(self, reply):
        self.reply, self.messages = reply, None
        self.chat = self
        self.completions = self

    def create(self, model, messages, temperature):
        self.messages = messages
        msg = type("M", (), {"content": self.reply})
        return type("R", (), {"choices": [type("C", (), {"message": msg})]})


def test_llm_answer_is_grounded_and_cited(act_index, local_embedder):
    llm = _FakeLLM("The controller must notify within 24 hours [2].")
    ans = QAEngine(act_index, local_embedder, llm=llm).ask("How fast must a security breach be notified?")
    system, user = llm.messages
    assert "ONLY the numbered context passages" in system["content"]
    assert "[1]" in user["content"] and "Question: How fast" in user["content"]
    assert ans.found and ans.mode == "llm"
    assert ans.citations[0]["n"] == 2  # the cited passage is listed first


def test_llm_not_found_reply_is_reported(act_index, local_embedder):
    llm = _FakeLLM(NOT_FOUND)
    ans = QAEngine(act_index, local_embedder, llm=llm).ask("What licence fee must a data controller pay?")
    assert ans.found is False and ans.citations == []
