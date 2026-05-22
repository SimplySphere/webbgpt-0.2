#!/usr/bin/env python3
from __future__ import annotations

from argparse import ArgumentParser
from collections import Counter
import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from config import ServeConfig
from rag.simple_index import LocalRagRetriever, format_rag_context
from serve.backends.native_backend import NativeCheckpointChatBackend
from tokenizer import format_chat


PROMPTS = [
    "At Webb, students often",
    "A course catalog helps students",
    "During a science project, the first step is",
    "The purpose of a boarding school community is",
    (
        "Using only this context, answer: Context: A prerequisite is required before taking a course. "
        "A recommendation is suggested preparation but not required. Question: What is the difference between a "
        "prerequisite and a recommendation?"
    ),
    (
        "Using only this context, answer: Context: This passage describes course planning but does not name any dean. "
        "Question: Who is the dean?"
    ),
    "What does the Webb context say about school community?",
    "What does the Webb context say about the current dean's phone number?",
]


def repeated_ngram(text: str, n: int = 4) -> bool:
    tokens = re.findall(r"[A-Za-z0-9']+", text.lower())
    if len(tokens) < n:
        return False
    counts = Counter(tuple(tokens[index : index + n]) for index in range(len(tokens) - n + 1))
    return max(counts.values(), default=0) >= 3


def assess(prompt: str, output: str, *, rag_hits: list[dict] | None = None) -> dict:
    lowered = output.lower()
    missing_prompt = "dean" in prompt.lower() and "does not name" in prompt.lower()
    prereq_prompt = "prerequisite" in prompt.lower() and "recommendation" in prompt.lower()
    return {
        "blank": not output.strip(),
        "fake_citation": bool(re.search(r"\[(source|citation):", output, flags=re.IGNORECASE)),
        "over_refusal": ("does not contain enough information" in lowered or "cannot answer" in lowered)
        and not missing_prompt,
        "semantic_repetition": repeated_ngram(output),
        "prompt_retention": any(word.lower() in lowered for word in re.findall(r"[A-Za-z]{5,}", prompt)[:3]),
        "context_grounded_correctness": (
            ("required" in lowered and ("suggested" in lowered or "not required" in lowered))
            if prereq_prompt
            else None
        ),
        "missing_context_honesty": (
            ("does not contain" in lowered or "not name" in lowered or "not enough" in lowered)
            if missing_prompt
            else None
        ),
        "retrieved_chunk_usage": bool(rag_hits),
        "copies_context_too_much": len(output.split()) > 90,
        "unsupported_fact_risk": any(term in lowered for term in ("dean is", "deadline is", "phone number is")),
    }


def build_prompt(prompt: str, retriever: LocalRagRetriever | None) -> tuple[str, list[dict]]:
    hits: list[dict] = []
    if retriever is not None:
        result = retriever.query(prompt)
        hits = list(result.get("hits") or [])
    messages = [{"role": "user", "content": prompt}]
    if hits:
        instruction = (
            "Use only the context below. If the context does not answer, say that the context does not contain enough "
            "information. Do not invent citations or unsupported facts."
        )
        messages.insert(
            0,
            {
                "role": "system",
                "content": f"{instruction}\n\nRetrieved context:\n{format_rag_context(hits)}",
            },
        )
    return format_chat(messages, add_generation_prompt=True), hits


def main() -> int:
    parser = ArgumentParser(description="Compare local WebbGPT checkpoints on fixed demo prompts.")
    parser.add_argument(
        "--checkpoint",
        action="append",
        nargs=2,
        metavar=("LABEL", "PATH"),
        default=None,
        help="Checkpoint label and path. May be supplied multiple times.",
    )
    parser.add_argument("--model-config", default="sample-configs/model-local-mvp.json")
    parser.add_argument("--tokenizer", default="artifacts/tokenizer/webbgpt-local-mvp.model")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.4)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--rag-index", default=None)
    parser.add_argument("--rag-chunks", default="data/rag/webbgpt_chunks.jsonl")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    checkpoint_rows = args.checkpoint or [
        ("pretrained", "artifacts/runs/local-mvp/checkpoints/pretrain/best-pretrain"),
        ("continue-small", "artifacts/runs/local-mvp/checkpoints/continue-small/best"),
        ("sft-rag-v1", "artifacts/runs/local-mvp/checkpoints/sft-rag-v1/best"),
    ]
    retriever = (
        LocalRagRetriever(index_path=args.rag_index, chunks_path=args.rag_chunks)
        if args.rag_index and Path(args.rag_index).exists()
        else None
    )
    results = []
    for label, checkpoint_path in checkpoint_rows:
        if not (Path(checkpoint_path) / "checkpoint.pt").exists():
            results.append({"label": label, "checkpoint": checkpoint_path, "skipped": True, "reason": "missing"})
            continue
        config = ServeConfig(
            checkpoint_path=checkpoint_path,
            tokenizer_path=args.tokenizer,
            model_config_path=args.model_config,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
        )
        backend = NativeCheckpointChatBackend(config)
        rows = []
        for prompt in PROMPTS:
            formatted_prompt, hits = build_prompt(prompt, retriever)
            output = backend.generate(
                formatted_prompt,
                max_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
            ).strip()
            rows.append(
                {
                    "prompt": prompt,
                    "output": output,
                    "rag_chunk_ids": [hit["chunk_id"] for hit in hits],
                    "metrics": assess(prompt, output, rag_hits=hits),
                }
            )
        aggregate = {
            "blank_output_count": sum(1 for row in rows if row["metrics"]["blank"]),
            "fake_citation_count": sum(1 for row in rows if row["metrics"]["fake_citation"]),
            "over_refusal_count": sum(1 for row in rows if row["metrics"]["over_refusal"]),
            "semantic_repetition_count": sum(1 for row in rows if row["metrics"]["semantic_repetition"]),
            "prompt_retention_count": sum(1 for row in rows if row["metrics"]["prompt_retention"]),
            "unsupported_fact_risk_count": sum(1 for row in rows if row["metrics"]["unsupported_fact_risk"]),
        }
        results.append({"label": label, "checkpoint": checkpoint_path, "skipped": False, "aggregate": aggregate, "rows": rows})
    payload = {"prompts": PROMPTS, "results": results}
    text = json.dumps(payload, indent=2, ensure_ascii=False)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
