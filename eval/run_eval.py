"""Evaluate retrieval and answers on eval/questions.jsonl.

Retrieval hit@k: a question counts as a hit if any of the top-k chunks contains its
`evidence` string (a short verbatim quote from the Act that answers the question).
Measured for keyword (BM25), dense and hybrid retrieval.

Answer correctness (two judges, reported separately):
  * key-fact match (automatic, deterministic): every group in `key_facts` must have at
    least one alternative present in the answer, and the answer must not be a refusal.
    Unanswerable questions are correct only if the system refuses.
  * LLM judge (optional, --judge): gpt-4o-mini compares the answer with the reference
    answer and returns CORRECT / INCORRECT.

    python -m eval.run_eval                     # local embeddings, extractive answers
    OPENAI_API_KEY=... python -m eval.run_eval --judge   # OpenAI embeddings + LLM answers
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.main import build_engine  # noqa: E402
from app.qa import NOT_FOUND  # noqa: E402

KS = (1, 3, 5)


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).lower()


def key_fact_correct(q: dict, answer: str, found: bool) -> bool:
    if q["answer"] is None:
        return not found
    if not found:
        return False
    a = norm(answer)
    return all(any(norm(alt) in a for alt in group) for group in q["key_facts"])


JUDGE_PROMPT = """You grade answers to questions about a statute.
Question: {question}
Reference answer: {reference}
System answer: {answer}
Reply CORRECT if the system answer states the essential facts of the reference answer without
contradicting it (extra correct detail is fine). Reply INCORRECT otherwise. One word only."""


def llm_judge(client, model: str, q: dict, answer: str, found: bool) -> bool:
    if q["answer"] is None:
        return not found
    if not found:
        return False
    r = client.chat.completions.create(model=model, temperature=0, messages=[{"role": "user", "content":
        JUDGE_PROMPT.format(question=q["question"], reference=q["answer"], answer=answer)}])
    return (r.choices[0].message.content or "").strip().upper().startswith("CORRECT")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", default=str(ROOT / "eval" / "questions.jsonl"))
    ap.add_argument("--judge", action="store_true", help="also grade answers with an LLM judge (needs OPENAI_API_KEY)")
    ap.add_argument("--out", default=None, help="write per-question results as JSON")
    args = ap.parse_args()

    questions = [json.loads(l) for l in open(args.questions) if l.strip()]
    engine = build_engine()
    answerable = [q for q in questions if q["evidence"]]
    print(f"embedder: {engine.embedder.name} | answers: {'LLM ' + engine.chat_model if engine.llm else 'extractive (no LLM)'}")
    print(f"{len(questions)} questions ({len(answerable)} answerable, {len(questions) - len(answerable)} unanswerable)\n")

    summary = {"embedder": engine.embedder.name, "llm": engine.chat_model if engine.llm else None, "retrieval": {}}
    print("Retrieval hit rate (answerable questions):")
    for mode in ("bm25", "dense", "hybrid"):
        hits = {k: 0 for k in KS}
        for q in answerable:
            res = engine.retrieve(q["question"], max(KS), mode)
            first = next((i for i, h in enumerate(res) if norm(q["evidence"]) in norm(h.chunk.text)), None)
            for k in KS:
                hits[k] += first is not None and first < k
        rates = {f"hit@{k}": round(hits[k] / len(answerable), 3) for k in KS}
        summary["retrieval"][mode] = rates
        print(f"  {mode:7s} " + "  ".join(f"{k}={v:.2f}" for k, v in rates.items()))

    judge_client = None
    if args.judge:
        from openai import OpenAI
        judge_client = OpenAI()

    rows = []
    for q in questions:
        a = engine.ask(q["question"])
        res = engine.retrieve(q["question"], engine.top_k)
        hit = bool(q["evidence"]) and any(norm(q["evidence"]) in norm(h.chunk.text) for h in res)
        row = {"id": q["id"], "question": q["question"], "answerable": q["answer"] is not None,
               "retrieval_hit@k": hit if q["evidence"] else None, "mode": a.mode, "found": a.found,
               "answer": a.answer, "key_fact_correct": key_fact_correct(q, a.answer, a.found)}
        if judge_client:
            row["llm_judge_correct"] = llm_judge(judge_client, engine.chat_model, q, a.answer, a.found)
        rows.append(row)

    def rate(key, subset):
        vals = [r[key] for r in subset]
        return round(sum(vals) / len(vals), 3) if vals else None

    ans_rows = [r for r in rows if r["answerable"]]
    un_rows = [r for r in rows if not r["answerable"]]
    summary["answers"] = {
        "key_fact_correct_all": rate("key_fact_correct", rows),
        "key_fact_correct_answerable": rate("key_fact_correct", ans_rows),
        "refusal_correct_unanswerable": rate("key_fact_correct", un_rows),
        "false_refusals_on_answerable": sum(not r["found"] for r in ans_rows),
    }
    if judge_client:
        summary["answers"]["llm_judge_correct_all"] = rate("llm_judge_correct", rows)
        summary["answers"]["llm_judge_correct_answerable"] = rate("llm_judge_correct", ans_rows)

    print(f"\nAnswers (top_k={engine.top_k}):")
    for k, v in summary["answers"].items():
        print(f"  {k}: {v}")
    print("\nFailures (key-fact judge):")
    for r in rows:
        if not r["key_fact_correct"]:
            print(f"  {r['id']} [{r['mode']}] {r['question']}\n      -> {r['answer'][:220]}")
    if args.out:
        Path(args.out).write_text(json.dumps({"summary": summary, "questions": rows}, indent=2, ensure_ascii=False))
        print(f"\nwritten {args.out}")


if __name__ == "__main__":
    main()
