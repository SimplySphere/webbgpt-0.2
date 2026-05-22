#!/usr/bin/env python3
from __future__ import annotations

from argparse import ArgumentParser
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from rag.simple_index import LocalRagRetriever


def main() -> int:
    parser = ArgumentParser(description="Query the lightweight WebbGPT RAG index.")
    parser.add_argument("query")
    parser.add_argument("--index", type=Path, default=Path("data/rag/webbgpt_index.json"))
    parser.add_argument("--chunks", type=Path, default=Path("data/rag/webbgpt_chunks.jsonl"))
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--min-score", type=float, default=0.05)
    parser.add_argument("--min-lexical-overlap", type=float, default=0.45)
    parser.add_argument("--min-matched-terms", type=int, default=2)
    parser.add_argument("--allow-missing-named-terms", action="store_true")
    parser.add_argument("--min-top-score-margin", type=float, default=0.0)
    args = parser.parse_args()

    retriever = LocalRagRetriever(
        index_path=args.index,
        chunks_path=args.chunks,
        top_k=args.top_k,
        min_score=args.min_score,
        min_lexical_overlap=args.min_lexical_overlap,
        min_matched_terms=args.min_matched_terms,
        require_named_terms=not args.allow_missing_named_terms,
        min_top_score_margin=args.min_top_score_margin,
    )
    print(json.dumps(retriever.query(args.query), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
