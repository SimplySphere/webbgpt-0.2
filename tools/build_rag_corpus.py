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

from rag.simple_index import normalize_text, text_sha256, tokenize


SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
TITLE_RE = re.compile(r"^[A-Z][A-Za-z0-9 ,:;'-]{6,120}$")
HEADER_RE = re.compile(r"^(?P<key>Title|Risk level|Allowed use)\s*:\s*(?P<value>.+?)\s*$", re.IGNORECASE)


SOURCE_POLICY = {
    "rag_candidates": {
        "risk_level": "medium",
        "allowed_use": "RAG",
    },
    "continued_pretrain_candidates": {
        "risk_level": "medium",
        "allowed_use": "continued pretraining / RAG after chunking",
    },
    "needs_manual_review": {
        "risk_level": "high",
        "allowed_use": "manual review only",
    },
    "rejected_template_heavy": {
        "risk_level": "high",
        "allowed_use": "rejected",
    },
}


def iter_input_files(input_dirs: list[Path]) -> list[Path]:
    files: list[Path] = []
    for input_dir in input_dirs:
        if not input_dir.exists():
            continue
        files.extend(sorted(path for path in input_dir.rglob("*.txt") if path.is_file()))
    return files


def infer_title(path: Path, text: str) -> str:
    for raw_line in text.splitlines():
        line = normalize_text(raw_line)
        header = HEADER_RE.match(line)
        if header and header.group("key").lower() == "title":
            title = header.group("value").strip()
            if title:
                return title
        if 6 <= len(line) <= 120 and TITLE_RE.match(line):
            return line
    return path.stem.replace("_", " ").replace("-", " ").title()


def source_header_metadata(text: str) -> dict[str, str]:
    metadata: dict[str, str] = {}
    for raw_line in text.splitlines()[:12]:
        line = normalize_text(raw_line)
        match = HEADER_RE.match(line)
        if match:
            metadata[match.group("key").lower()] = match.group("value").strip()
    return metadata


def strip_source_headers(text: str) -> str:
    lines = []
    for raw_line in text.splitlines():
        line = normalize_text(raw_line)
        if HEADER_RE.match(line):
            continue
        if line.lower().startswith("provenance:"):
            continue
        lines.append(raw_line)
    return "\n".join(lines).strip()


def source_policy_for(path: Path, category: str, raw_text: str) -> dict[str, str]:
    policy = dict(SOURCE_POLICY.get(category, {"risk_level": "medium", "allowed_use": "manual review only"}))
    if policy["allowed_use"] in {"manual review only", "rejected"}:
        return policy
    metadata = source_header_metadata(raw_text)
    risk_level = metadata.get("risk level", "").lower()
    if risk_level in {"low", "medium", "high"}:
        policy["risk_level"] = risk_level
    allowed_use = metadata.get("allowed use", "")
    if allowed_use and "rag" in allowed_use.lower():
        policy["allowed_use"] = allowed_use
    return policy


def split_words(text: str) -> list[str]:
    return re.findall(r"\S+", normalize_text(text))


def ngram_repetition(words: list[str], n: int) -> int:
    if len(words) < n:
        return 0
    counts = Counter(tuple(word.lower() for word in words[index : index + n]) for index in range(len(words) - n + 1))
    return max(counts.values(), default=0)


def skip_reason(text: str) -> str | None:
    words = split_words(text)
    word_count = len(words)
    if word_count < 80:
        return "too_short"
    if word_count > 420:
        return "too_long"
    url_count = len(re.findall(r"https?://|www\.", text, flags=re.IGNORECASE))
    if url_count / max(word_count, 1) > 0.015:
        return "too_many_urls"
    separator_count = sum(text.count(symbol) for symbol in ("|", ">", "•", "-----", "====="))
    if separator_count / max(word_count, 1) > 0.08:
        return "too_many_separators"
    terms = tokenize(text)
    if terms:
        unique_ratio = len(set(terms)) / max(len(terms), 1)
        if unique_ratio < 0.28:
            return "low_unique_token_ratio"
    if ngram_repetition(words, 8) > 2 or ngram_repetition(words, 12) > 2 or ngram_repetition(words, 20) > 1:
        return "repeated_ngram"
    lowered = text.lower()
    boilerplate_hits = sum(
        lowered.count(phrase)
        for phrase in (
            "click here",
            "privacy policy",
            "all rights reserved",
            "subscribe",
            "navigation",
            "footer",
            "menu",
        )
    )
    if boilerplate_hits >= 3:
        return "boilerplate"
    return None


def chunk_words(words: list[str], *, chunk_words_target: int, overlap_words: int) -> list[list[str]]:
    chunks: list[list[str]] = []
    step = max(chunk_words_target - overlap_words, 1)
    for start in range(0, len(words), step):
        window = words[start : start + chunk_words_target]
        if len(window) < 80:
            break
        chunks.append(window)
    return chunks


def split_sentences(text: str) -> list[str]:
    normalized = normalize_text(text)
    sentences = [sentence.strip() for sentence in SENTENCE_RE.split(normalized) if sentence.strip()]
    return sentences or [normalized]


def trailing_overlap_sentences(sentences: list[str], *, overlap_words: int) -> list[str]:
    kept: list[str] = []
    total = 0
    for sentence in reversed(sentences):
        count = len(split_words(sentence))
        if kept and total + count > overlap_words:
            break
        kept.insert(0, sentence)
        total += count
    return kept


def chunk_sentence_windows(text: str, *, chunk_words_target: int, overlap_words: int) -> list[str]:
    windows: list[str] = []
    current: list[str] = []
    current_words = 0
    for sentence in split_sentences(text):
        sentence_words = len(split_words(sentence))
        if current and current_words + sentence_words > chunk_words_target and current_words >= 120:
            windows.append(normalize_text(" ".join(current)))
            current = trailing_overlap_sentences(current, overlap_words=overlap_words)
            current_words = sum(len(split_words(item)) for item in current)
        current.append(sentence)
        current_words += sentence_words
    if current_words >= 80:
        windows.append(normalize_text(" ".join(current)))
    return windows


def source_category_for(path: Path, source_root: Path) -> str:
    try:
        relative = path.relative_to(source_root)
        if len(relative.parts) > 1:
            return relative.parts[0]
    except ValueError:
        pass
    return path.parent.name


def build_chunks(
    files: list[Path],
    *,
    source_root: Path,
    chunk_words_target: int,
    overlap_words: int,
) -> tuple[list[dict], list[dict]]:
    chunks: list[dict] = []
    manifest_sources: list[dict] = []
    seen_hashes: set[str] = set()
    for file_path in files:
        category = source_category_for(file_path, source_root)
        raw_text = file_path.read_text(encoding="utf-8", errors="ignore")
        source_metadata = source_header_metadata(raw_text)
        policy = source_policy_for(file_path, category, raw_text)
        if policy["allowed_use"] in {"manual review only", "rejected"}:
            manifest_sources.append(
                {
                    "source_file": str(file_path),
                    "source_category": category,
                    "risk_level": policy["risk_level"],
                    "allowed_use": policy["allowed_use"],
                    "word_count": len(split_words(raw_text)),
                    "chunks_kept": 0,
                    "chunks_skipped": 0,
                    "skip_reasons": {"source_category_not_allowed": 1},
                    "source_metadata": source_metadata,
                }
            )
            continue
        title = infer_title(file_path, raw_text)
        chunkable_text = strip_source_headers(raw_text)
        words = split_words(chunkable_text)
        source_skip_reasons: Counter[str] = Counter()
        kept = 0
        skipped = 0
        for local_index, chunk_text in enumerate(
            chunk_sentence_windows(chunkable_text, chunk_words_target=chunk_words_target, overlap_words=overlap_words)
        ):
            reason = skip_reason(chunk_text)
            if reason is not None:
                source_skip_reasons[reason] += 1
                skipped += 1
                continue
            digest = text_sha256(chunk_text)
            if digest in seen_hashes:
                source_skip_reasons["near_duplicate_exact_text"] += 1
                skipped += 1
                continue
            seen_hashes.add(digest)
            chunk_id = f"rag-{text_sha256(str(file_path))[:10]}-{local_index:04d}"
            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "source_file": str(file_path),
                    "source_category": category,
                    "title": title,
                    "text": chunk_text,
                    "word_count": len(split_words(chunk_text)),
                    "sha256": digest,
                    "risk_level": policy["risk_level"],
                    "allowed_use": policy["allowed_use"],
                }
            )
            kept += 1
        manifest_sources.append(
            {
                "source_file": str(file_path),
                "source_category": category,
                "risk_level": policy["risk_level"],
                "allowed_use": policy["allowed_use"],
                "word_count": len(words),
                "chunks_kept": kept,
                "chunks_skipped": skipped,
                "skip_reasons": dict(source_skip_reasons),
                "source_metadata": source_metadata,
            }
        )
    return chunks, manifest_sources


def main() -> int:
    parser = ArgumentParser(description="Build a local WebbGPT RAG chunk corpus.")
    parser.add_argument("--input-dir", action="append", type=Path, default=None)
    parser.add_argument("--source-root", type=Path, default=Path("data/source_material"))
    parser.add_argument("--output", type=Path, default=Path("data/rag/webbgpt_chunks.jsonl"))
    parser.add_argument("--manifest", type=Path, default=Path("data/rag/webbgpt_sources_manifest.json"))
    parser.add_argument("--chunk-words", type=int, default=220)
    parser.add_argument("--overlap-words", type=int, default=45)
    args = parser.parse_args()

    input_dirs = args.input_dir or [Path("data/source_material/rag_candidates")]
    files = iter_input_files(input_dirs)
    chunks, manifest_sources = build_chunks(
        files,
        source_root=args.source_root,
        chunk_words_target=args.chunk_words,
        overlap_words=args.overlap_words,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for chunk in chunks:
            handle.write(json.dumps(chunk, ensure_ascii=False) + "\n")
    manifest = {
        "version": "1.0",
        "kind": "webbgpt_rag_sources_manifest",
        "chunk_output": str(args.output),
        "input_dirs": [str(path) for path in input_dirs],
        "chunk_words": args.chunk_words,
        "overlap_words": args.overlap_words,
        "chunk_count": len(chunks),
        "sources": manifest_sources,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"chunks": len(chunks), "manifest": str(args.manifest), "output": str(args.output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
