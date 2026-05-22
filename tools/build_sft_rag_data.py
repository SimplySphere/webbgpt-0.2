#!/usr/bin/env python3
from __future__ import annotations

from argparse import ArgumentParser
import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from rag.simple_index import load_chunks, normalize_text


QUESTION_TEMPLATES = [
    "What is the main point of this context?",
    "What information does this context provide?",
    "What should a student understand from this context?",
    "How would you answer using only this context?",
    "What should WebbGPT not infer beyond this context?",
    "What kind of question can this context support?",
]
MISSING_QUESTIONS = [
    "Who is the dean?",
    "What is the current phone policy?",
    "What is the admissions deadline?",
    "What are the dining hall hours?",
]


def split_sentences(text: str) -> list[str]:
    return [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", normalize_text(text))
        if len(sentence.split()) >= 8
    ]


def short_context(text: str, *, max_words: int = 180) -> str:
    words = normalize_text(text).split()
    return " ".join(words[:max_words])


RISK_RANK = {"low": 0, "medium": 1, "high": 2}


def grounded_answer(text: str, *, template_index: int = 0) -> str:
    sentences = split_sentences(text)
    sentence = sentences[template_index % len(sentences)] if sentences else " ".join(normalize_text(text).split()[:35])
    words = sentence.split()
    if len(words) > 48:
        sentence = " ".join(words[:48]).rstrip(",;:")
    if not sentence.endswith((".", "!", "?")):
        sentence = f"{sentence}."
    if template_index % 3 == 1:
        answer = f"Using only the context, the supported point is that {sentence[0].lower()}{sentence[1:]}"
    elif template_index % 3 == 2:
        answer = f"The answer should stay limited to this context: {sentence}"
    else:
        answer = f"The context says that {sentence[0].lower()}{sentence[1:]}"
    if len(answer.split()) < 15:
        answer = f"{answer} This answer stays limited to the provided context."
    return answer


def user_prompt(context: str, question: str) -> str:
    return (
        "Using only the context below, answer the question. If the context does not contain enough information, "
        "say so.\n\n"
        f"Context:\n{context}\n\n"
        f"Question:\n{question}"
    )


def row(row_id: str, chunk: dict, *, question: str, answer: str, split: str, mix: str) -> dict:
    context = short_context(str(chunk["text"]))
    return {
        "row_id": row_id,
        "messages": [
            {
                "role": "user",
                "content": user_prompt(context, question),
            },
            {
                "role": "assistant",
                "content": answer,
            },
        ],
        "mix": mix,
        "chunk_id": chunk["chunk_id"],
        "source_file": chunk.get("source_file"),
        "source_usage": "rag_context",
        "answer_word_count": len(answer.split()),
        "split": split,
    }


def is_allowed_chunk(chunk: dict, *, max_risk_level: str) -> bool:
    allowed_use = str(chunk.get("allowed_use", "")).lower()
    risk_level = str(chunk.get("risk_level", "")).lower()
    risk_allowed = RISK_RANK.get(risk_level, 99) <= RISK_RANK[max_risk_level]
    return "rag" in allowed_use and risk_allowed and len(str(chunk.get("text", "")).split()) >= 80


def build_rows(
    chunks: list[dict],
    *,
    train_target: int,
    validation_target: int,
    max_risk_level: str,
    examples_per_chunk: int,
) -> tuple[list[dict], list[dict]]:
    safe_chunks = [
        chunk
        for chunk in sorted(
            chunks,
            key=lambda item: (
                RISK_RANK.get(str(item.get("risk_level", "")).lower(), 99),
                str(item.get("source_file", "")),
                str(item["chunk_id"]),
            ),
        )
        if is_allowed_chunk(chunk, max_risk_level=max_risk_level)
    ]
    validation_chunk_ids = {chunk["chunk_id"] for index, chunk in enumerate(safe_chunks) if index % 5 == 0}
    train_chunks = [chunk for chunk in safe_chunks if chunk["chunk_id"] not in validation_chunk_ids]
    validation_chunks = [chunk for chunk in safe_chunks if chunk["chunk_id"] in validation_chunk_ids]

    train_rows: list[dict] = []
    validation_rows: list[dict] = []
    for split, target, source_chunks, sink in (
        ("train", train_target, train_chunks, train_rows),
        ("validation", validation_target, validation_chunks, validation_rows),
    ):
        for index, chunk in enumerate(source_chunks):
            for template_index in range(max(int(examples_per_chunk), 1)):
                if len(sink) >= target:
                    break
                question = QUESTION_TEMPLATES[(index + template_index) % len(QUESTION_TEMPLATES)]
                sink.append(
                    row(
                        f"sft-rag-v1-{split}-{len(sink):04d}",
                        chunk,
                        question=question,
                        answer=grounded_answer(str(chunk["text"]), template_index=template_index),
                        split=split,
                        mix="grounded_context_answer",
                    )
                )
            if len(sink) >= target:
                break
            if index % 3 == 0:
                missing_question = MISSING_QUESTIONS[index % len(MISSING_QUESTIONS)]
                sink.append(
                    row(
                        f"sft-rag-v1-{split}-{len(sink):04d}",
                        chunk,
                        question=missing_question,
                        answer=(
                            "The context does not contain enough information to answer that question, "
                            "so I should not invent the missing detail."
                        ),
                        split=split,
                        mix="missing_context_abstention",
                    )
                )
    return train_rows, validation_rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for item in rows:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")


def main() -> int:
    parser = ArgumentParser(description="Build SFT-RAG v1 rows from WebbGPT RAG chunks.")
    parser.add_argument("--chunks", type=Path, default=Path("data/rag/webbgpt_chunks.jsonl"))
    parser.add_argument("--train-output", type=Path, default=Path("data/posttrain/sft_rag_v1_train.jsonl"))
    parser.add_argument("--validation-output", type=Path, default=Path("data/posttrain/sft_rag_v1_validation.jsonl"))
    parser.add_argument("--train-target", type=int, default=180)
    parser.add_argument("--validation-target", type=int, default=40)
    parser.add_argument("--max-risk-level", choices=sorted(RISK_RANK), default="low")
    parser.add_argument("--examples-per-chunk", type=int, default=3)
    args = parser.parse_args()

    chunks = load_chunks(args.chunks)
    train_rows, validation_rows = build_rows(
        chunks,
        train_target=args.train_target,
        validation_target=args.validation_target,
        max_risk_level=args.max_risk_level,
        examples_per_chunk=args.examples_per_chunk,
    )
    train_chunk_ids = {row["chunk_id"] for row in train_rows}
    validation_chunk_ids = {row["chunk_id"] for row in validation_rows}
    overlap = train_chunk_ids & validation_chunk_ids
    if overlap:
        raise RuntimeError(f"Train/validation chunk overlap detected: {sorted(overlap)[:5]}")
    write_jsonl(args.train_output, train_rows)
    write_jsonl(args.validation_output, validation_rows)
    report = {
        "train_rows": len(train_rows),
        "validation_rows": len(validation_rows),
        "train_chunks": len(train_chunk_ids),
        "validation_chunks": len(validation_chunk_ids),
        "chunk_overlap": len(overlap),
        "train_missing_context_rows": sum(1 for row in train_rows if row["mix"] == "missing_context_abstention"),
        "validation_missing_context_rows": sum(
            1 for row in validation_rows if row["mix"] == "missing_context_abstention"
        ),
        "max_risk_level": args.max_risk_level,
        "examples_per_chunk": args.examples_per_chunk,
        "train_target_met": len(train_rows) >= args.train_target,
        "validation_target_met": len(validation_rows) >= args.validation_target,
        "train_output": str(args.train_output),
        "validation_output": str(args.validation_output),
    }
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
