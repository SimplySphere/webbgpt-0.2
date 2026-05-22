from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any


TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9'-]{1,}")
STOPWORDS = {
    "about",
    "after",
    "also",
    "and",
    "are",
    "because",
    "before",
    "below",
    "between",
    "but",
    "can",
    "does",
    "for",
    "from",
    "has",
    "have",
    "hello",
    "hey",
    "hi",
    "how",
    "im",
    "i'm",
    "into",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "this",
    "to",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
    "would",
    "dr",
    "mr",
    "mrs",
    "ms",
}

ENTITY_ALLOWLIST = {"webb", "webbgpt"}


@dataclass(slots=True)
class RagHit:
    chunk_id: str
    score: float
    source_file: str
    source_category: str
    title: str
    text: str
    word_count: int
    sha256: str
    risk_level: str
    allowed_use: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "score": round(float(self.score), 6),
            "source_file": self.source_file,
            "source_category": self.source_category,
            "title": self.title,
            "text": self.text,
            "word_count": self.word_count,
            "sha256": self.sha256,
            "risk_level": self.risk_level,
            "allowed_use": self.allowed_use,
        }


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text: str) -> list[str]:
    return [
        token.lower().strip("'-")
        for token in TOKEN_RE.findall(text)
        if token.lower().strip("'-") and token.lower().strip("'-") not in STOPWORDS
    ]


def named_query_terms(text: str) -> set[str]:
    terms: set[str] = set()
    for match in re.finditer(r"\b[A-Z][A-Za-z0-9'-]{2,}\b", text):
        token = match.group(0).lower().strip("'-")
        if token and token not in STOPWORDS and token not in ENTITY_ALLOWLIST:
            terms.add(token)
    return terms


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_chunks(path: str | Path) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                chunk = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path} line {line_number}: {exc}") from exc
            chunks.append(chunk)
    return chunks


def build_index_payload(chunks: list[dict[str, Any]], *, chunks_path: str) -> dict[str, Any]:
    document_frequency: Counter[str] = Counter()
    chunk_terms: list[dict[str, Any]] = []
    for chunk in chunks:
        terms = tokenize(str(chunk.get("text", "")))
        counts = Counter(terms)
        document_frequency.update(counts.keys())
        chunk_terms.append(
            {
                "chunk_id": str(chunk["chunk_id"]),
                "length": max(sum(counts.values()), 1),
                "terms": dict(counts),
            }
        )

    chunk_count = max(len(chunks), 1)
    idf = {
        term: math.log((chunk_count + 1) / (df + 0.5)) + 1.0
        for term, df in sorted(document_frequency.items())
    }
    return {
        "version": "1.0",
        "kind": "webbgpt_lexical_rag_index",
        "chunks_path": chunks_path,
        "chunk_count": len(chunks),
        "vocab_size": len(idf),
        "idf": idf,
        "chunk_terms": chunk_terms,
    }


def load_index(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("kind") != "webbgpt_lexical_rag_index":
        raise ValueError(f"{path} is not a WebbGPT lexical RAG index.")
    return payload


def _score_terms(
    query_terms: Counter[str],
    chunk_terms: dict[str, int],
    *,
    idf: dict[str, float],
    chunk_length: int,
) -> float:
    if not query_terms or not chunk_terms:
        return 0.0
    score = 0.0
    matched = 0
    for term, query_count in query_terms.items():
        term_count = int(chunk_terms.get(term, 0))
        if term_count <= 0:
            continue
        matched += 1
        tf = term_count / max(chunk_length, 1)
        score += (1.0 + math.log(query_count)) * (1.0 + math.log(1 + term_count)) * tf * idf.get(term, 1.0)
    coverage = matched / max(len(query_terms), 1)
    return score * (0.5 + coverage)


def _overlap_stats(query_terms: Counter[str], chunk_terms: dict[str, int]) -> dict[str, Any]:
    query_vocab = set(query_terms)
    matched = sorted(term for term in query_vocab if int(chunk_terms.get(term, 0)) > 0)
    missing = sorted(query_vocab - set(matched))
    denominator = max(len(query_vocab), 1)
    return {
        "matched_terms": matched,
        "missing_terms": missing,
        "matched_term_count": len(matched),
        "query_term_count": len(query_vocab),
        "lexical_overlap": len(matched) / denominator,
    }


def query_index(
    query: str,
    *,
    index: dict[str, Any],
    chunks: list[dict[str, Any]],
    top_k: int = 3,
    min_score: float = 0.015,
    min_lexical_overlap: float = 0.45,
    min_matched_terms: int = 2,
    require_named_terms: bool = True,
    min_top_score_margin: float = 0.0,
) -> dict[str, Any]:
    chunk_by_id = {str(chunk["chunk_id"]): chunk for chunk in chunks}
    idf = {str(term): float(value) for term, value in dict(index.get("idf", {})).items()}
    query_terms = Counter(tokenize(query))
    required_named_terms = named_query_terms(query) if require_named_terms else set()
    scored: list[tuple[float, dict[str, Any], dict[str, Any]]] = []
    rejected: Counter[str] = Counter()
    for row in list(index.get("chunk_terms", [])):
        chunk_id = str(row.get("chunk_id"))
        chunk_terms = {str(term): int(count) for term, count in dict(row.get("terms", {})).items()}
        overlap = _overlap_stats(query_terms, chunk_terms)
        score = _score_terms(
            query_terms,
            chunk_terms,
            idf=idf,
            chunk_length=int(row.get("length") or 1),
        )
        if score < min_score:
            rejected["below_min_score"] += 1
            continue
        required_matches = min(max(int(min_matched_terms), 1), max(overlap["query_term_count"], 1))
        if overlap["matched_term_count"] < required_matches:
            rejected["too_few_matched_terms"] += 1
            continue
        if overlap["lexical_overlap"] < float(min_lexical_overlap):
            rejected["low_lexical_overlap"] += 1
            continue
        chunk = chunk_by_id.get(chunk_id)
        if chunk is None:
            rejected["missing_chunk"] += 1
            continue
        chunk_text_terms = set(tokenize(str(chunk.get("text", ""))))
        missing_named_terms = sorted(term for term in required_named_terms if term not in chunk_text_terms)
        if missing_named_terms:
            rejected["missing_named_query_terms"] += 1
            continue
        scored.append((score, chunk, overlap))
    scored.sort(key=lambda item: (-item[0], str(item[1].get("chunk_id", ""))))
    if scored and min_top_score_margin > 0 and len(scored) > 1:
        margin = scored[0][0] - scored[1][0]
        if margin < min_top_score_margin:
            rejected["top_score_margin_too_small"] += len(scored)
            scored = []
    hits = [
        RagHit(
            chunk_id=str(chunk["chunk_id"]),
            score=score,
            source_file=str(chunk.get("source_file", "")),
            source_category=str(chunk.get("source_category", "")),
            title=str(chunk.get("title", "")),
            text=str(chunk.get("text", "")),
            word_count=int(chunk.get("word_count") or 0),
            sha256=str(chunk.get("sha256", "")),
            risk_level=str(chunk.get("risk_level", "")),
            allowed_use=str(chunk.get("allowed_use", "")),
        ).to_dict()
        | {
            "text_preview": normalize_text(str(chunk.get("text", "")))[:420],
            "matched_terms": overlap["matched_terms"],
            "missing_terms": overlap["missing_terms"],
            "matched_term_count": overlap["matched_term_count"],
            "query_term_count": overlap["query_term_count"],
            "lexical_overlap": round(float(overlap["lexical_overlap"]), 4),
        }
        for score, chunk, overlap in scored[: max(int(top_k), 0)]
    ]
    return {
        "query": query,
        "top_k": int(top_k),
        "min_score": float(min_score),
        "min_lexical_overlap": float(min_lexical_overlap),
        "min_matched_terms": int(min_matched_terms),
        "require_named_terms": bool(require_named_terms),
        "required_named_terms": sorted(required_named_terms),
        "no_hit": not hits,
        "hits": hits,
        "diagnostics": {
            "query_terms": sorted(query_terms),
            "query_term_count": len(query_terms),
            "candidate_count": len(list(index.get("chunk_terms", []))),
            "accepted_count": len(hits),
            "rejected_count": sum(rejected.values()),
            "rejected_reasons": dict(rejected),
            "top_score": round(float(scored[0][0]), 6) if scored else 0.0,
            "top_lexical_overlap": round(float(scored[0][2]["lexical_overlap"]), 4) if scored else 0.0,
            "top_matched_terms": scored[0][2]["matched_terms"] if scored else [],
        },
    }


class LocalRagRetriever:
    def __init__(
        self,
        *,
        index_path: str | Path,
        chunks_path: str | Path | None = None,
        top_k: int = 3,
        min_score: float = 0.015,
        min_lexical_overlap: float = 0.45,
        min_matched_terms: int = 2,
        require_named_terms: bool = True,
        min_top_score_margin: float = 0.0,
    ):
        self.index_path = Path(index_path)
        self.index = load_index(self.index_path)
        resolved_chunks = chunks_path or self.index.get("chunks_path")
        if not resolved_chunks:
            raise ValueError("RAG index does not specify chunks_path and no chunks_path override was provided.")
        self.chunks_path = Path(str(resolved_chunks))
        self.chunks = load_chunks(self.chunks_path)
        self.top_k = int(top_k)
        self.min_score = float(min_score)
        self.min_lexical_overlap = float(min_lexical_overlap)
        self.min_matched_terms = int(min_matched_terms)
        self.require_named_terms = bool(require_named_terms)
        self.min_top_score_margin = float(min_top_score_margin)

    def query(self, query: str, *, top_k: int | None = None) -> dict[str, Any]:
        return query_index(
            query,
            index=self.index,
            chunks=self.chunks,
            top_k=self.top_k if top_k is None else int(top_k),
            min_score=self.min_score,
            min_lexical_overlap=self.min_lexical_overlap,
            min_matched_terms=self.min_matched_terms,
            require_named_terms=self.require_named_terms,
            min_top_score_margin=self.min_top_score_margin,
        )


def format_rag_context(hits: list[dict[str, Any]], *, max_chars_per_chunk: int = 1200) -> str:
    blocks = []
    for hit in hits:
        text = normalize_text(str(hit.get("text", "")))
        if len(text) > max_chars_per_chunk:
            text = text[:max_chars_per_chunk].rsplit(" ", 1)[0].strip()
        blocks.append(
            "\n".join(
                [
                    f"Chunk ID: {hit.get('chunk_id')}",
                    f"Source file: {hit.get('source_file')}",
                    f"Title: {hit.get('title')}",
                    f"Context: {text}",
                ]
            )
        )
    return "\n\n".join(blocks)
