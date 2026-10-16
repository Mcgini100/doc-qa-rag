"""Question answering: retrieve, gate on relevance, then answer only from the retrieved text."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.embeddings import Embedder
from app.index import DocIndex, Hit, bm25_tokens

NOT_FOUND = "not found in the documents"

# Minimum cosine similarity between the question and its best chunk before we even try
# to answer. The local value was set from six dev questions that are NOT in
# eval/questions.jsonl (in-scope 0.24-0.37, off-topic -0.07-0.10). The OpenAI value is a
# deliberately low, unmeasured guard; the LLM prompt does the fine-grained refusal.
DENSE_FLOOR = {"local-wordllama-256": 0.15, "openai-text-embedding-3-small": 0.20}

SYSTEM_PROMPT = f"""You answer questions about legal documents using ONLY the numbered context passages.
Rules:
- Use only facts stated in the passages. Do not use outside knowledge.
- Cite every claim with the passage number in square brackets, e.g. [2]. Cite section numbers when the passage gives them.
- If the passages do not contain the answer, reply with exactly: {NOT_FOUND}
- Be concise: at most 5 sentences or a short list."""


@dataclass
class Answer:
    question: str
    answer: str
    found: bool
    mode: str                       # "llm", "extractive" or "refused"
    citations: list[dict] = field(default_factory=list)


def citation(hit: Hit, n: int) -> dict:
    c = hit.chunk
    pages = f"{c.page_start}" if c.page_start == c.page_end else f"{c.page_start}-{c.page_end}"
    return {"n": n, "chunk_id": c.id, "doc": c.doc, "pages": pages, "sections": c.sections,
            "score": round(hit.score, 4), "dense": round(hit.dense, 4), "bm25": round(hit.bm25, 3),
            "text": c.text}


def is_relevant(question: str, hits: list[Hit], embedder_name: str) -> bool:
    """Refuse without calling the LLM when the best embedding similarity is below the
    backend's floor (clearly off-topic questions)."""
    if not hits:
        return False
    return max(h.dense for h in hits) >= DENSE_FLOOR.get(embedder_name, 0.2)


def build_context(hits: list[Hit]) -> str:
    parts = []
    for n, h in enumerate(hits, 1):
        c = h.chunk
        head = f"[{n}] {c.doc}, page {c.page_start}" + (f"-{c.page_end}" if c.page_end != c.page_start else "")
        if c.sections:
            head += " (" + "; ".join(c.sections) + ")"
        parts.append(head + "\n" + c.text)
    return "\n\n".join(parts)


def extractive_answer(question: str, hits: list[Hit], max_sentences: int = 3) -> str:
    """No-LLM fallback: the sentences from the retrieved passages that share the most
    content words with the question, each with its passage number."""
    q = set(bm25_tokens(question))
    scored = []
    for n, h in enumerate(hits, 1):
        for sent in re.split(r"(?<=[.;:])\s+", h.chunk.text):
            overlap = len(q & set(bm25_tokens(sent)))
            if overlap and len(sent) > 30:
                scored.append((overlap, -n, " ".join(sent.split()), n))
    scored.sort(reverse=True)
    picked = scored[:max_sentences]
    if not picked:
        return NOT_FOUND
    return " ".join(f"{s} [{n}]" for _, _, s, n in picked)


class QAEngine:
    def __init__(self, index: DocIndex, embedder: Embedder, llm=None, chat_model: str = "gpt-4o-mini",
                 top_k: int = 5) -> None:
        self.index = index
        self.embedder = embedder
        self.llm = llm  # an openai.OpenAI client, or None for extractive mode
        self.chat_model = chat_model
        self.top_k = top_k

    def retrieve(self, question: str, k: int | None = None, mode: str = "hybrid") -> list[Hit]:
        return self.index.search(question, self.embedder, k or self.top_k, mode)

    def ask(self, question: str, k: int | None = None) -> Answer:
        hits = self.retrieve(question, k)
        if not is_relevant(question, hits, self.embedder.name):
            return Answer(question, NOT_FOUND, False, "refused", [])
        cites = [citation(h, n) for n, h in enumerate(hits, 1)]
        if self.llm is None:
            text = extractive_answer(question, hits)
            return Answer(question, text, text != NOT_FOUND, "extractive", cites)
        resp = self.llm.chat.completions.create(
            model=self.chat_model,
            temperature=0,
            messages=[{"role": "system", "content": SYSTEM_PROMPT},
                      {"role": "user", "content": f"Context:\n{build_context(hits)}\n\nQuestion: {question}"}],
        )
        text = (resp.choices[0].message.content or "").strip()
        found = NOT_FOUND not in text.lower()
        used = {int(m) for m in re.findall(r"\[(\d+)\]", text)}
        # Return the passages the answer cites first, then the rest of the retrieved ones.
        cites.sort(key=lambda c: (c["n"] not in used, c["n"]))
        return Answer(question, text, found, "llm", cites if found else [])
