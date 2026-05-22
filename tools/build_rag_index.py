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

from rag.simple_index import build_index_payload, load_chunks


def main() -> int:
    parser = ArgumentParser(description="Build a lightweight lexical WebbGPT RAG index.")
    parser.add_argument("--chunks", type=Path, default=Path("data/rag/webbgpt_chunks.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("data/rag/webbgpt_index.json"))
    args = parser.parse_args()

    chunks = load_chunks(args.chunks)
    payload = build_index_payload(chunks, chunks_path=str(args.chunks))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"chunks": len(chunks), "vocab_size": payload["vocab_size"], "output": str(args.output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
