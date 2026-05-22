#!/usr/bin/env python3
from __future__ import annotations

from argparse import ArgumentParser
from datetime import datetime, timezone
import json
from pathlib import Path
import urllib.error
import urllib.request


PROMPTS = [
    "hi im harry potter",
    "hi im dr dzula",
    "What does the catalog say about prerequisites?",
    "What is the difference between a prerequisite and a recommendation?",
    "Who is the dean?",
    "A course catalog helps students",
    "During a science project, the first step is",
    "The purpose of a boarding school community is",
    "What is the Hogwarts dining policy?",
    (
        "Context: A prerequisite is required before taking a course. "
        "A recommendation is suggested preparation but not required. "
        "Question: What is the difference between a prerequisite and a recommendation?"
    ),
]


def post_json(url: str, payload: dict, *, timeout: float) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{url} returned HTTP {exc.code}: {body}") from exc


def summarize(prompt: str, payload: dict) -> dict:
    metadata = payload.get("metadata") or {}
    status = metadata.get("status") or {}
    routing = metadata.get("routing") or {}
    rag = metadata.get("rag") or {}
    quality = metadata.get("quality") or {}
    hits = list(rag.get("hits") or [])
    return {
        "prompt": prompt,
        "route_selected": routing.get("route") or routing.get("mode"),
        "retrieved_hit_count": len(hits),
        "retrieved_source_ids": [hit.get("chunk_id") for hit in hits],
        "source_display_correct": all(
            hit.get("chunk_id") and hit.get("source_file") and hit.get("text_preview") for hit in hits
        )
        if hits
        else True,
        "output_degenerate": bool(quality.get("degenerate")),
        "quality_reasons": list(quality.get("reasons") or []),
        "abstained": bool(status.get("abstained")),
        "answered": bool(status.get("answered")),
        "grounded": bool(status.get("grounded")),
        "fallback_triggered": bool(status.get("retrieved_context_fallback")),
        "final_label": status.get("final_label") or "Answered",
        "text": payload.get("text") or "",
        "rag_diagnostics": rag.get("diagnostics") or {},
    }


def main() -> int:
    parser = ArgumentParser(description="Run the fixed WebbGPT RAG reliability regression prompts against a server.")
    parser.add_argument("--url", default="http://127.0.0.1:8000/generate")
    parser.add_argument("--output", default="artifacts/runs/local-mvp/rag_reliability_regression.json")
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--top-p", type=float, default=0.92)
    args = parser.parse_args()

    rows = []
    for prompt in PROMPTS:
        response = post_json(
            args.url,
            {
                "prompt": prompt,
                "tools": True,
                "citations": True,
                "max_new_tokens": args.max_new_tokens,
                "temperature": args.temperature,
                "top_k": args.top_k,
                "top_p": args.top_p,
            },
            timeout=args.timeout,
        )
        rows.append(summarize(prompt, response))

    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "url": args.url,
        "decode": {
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_k": args.top_k,
            "top_p": args.top_p,
        },
        "rows": rows,
        "aggregate": {
            "prompts": len(rows),
            "abstained": sum(1 for row in rows if row["abstained"]),
            "answered": sum(1 for row in rows if row["answered"]),
            "fallback_triggered": sum(1 for row in rows if row["fallback_triggered"]),
            "degenerate": sum(1 for row in rows if row["output_degenerate"]),
            "retrieved_hit_prompts": sum(1 for row in rows if row["retrieved_hit_count"] > 0),
        },
    }
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(payload["aggregate"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
