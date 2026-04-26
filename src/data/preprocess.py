from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from config import DataConfig, DataSourceConfig
from data.schemas import DocumentRecord


EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
PHONE_RE = re.compile(r"(?:(?:\+?1[-.\s]*)?(?:\(\d{3}\)|\d{3})[-.\s]*)\d{3}[-.\s]*\d{4}")
SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
MULTISPACE_RE = re.compile(r"\s+")
URL_RE = re.compile(r"\b(?:https?://|www\.)\S+|\b\S+\.(?:com|org|net|edu|gov|io|co)\S*", re.IGNORECASE)
BULLET_OR_LIST_RE = re.compile(r"^\s*(?:[-*\u2022]+|\d+[.)]|[A-Z][.)])\s+")
LABEL_LINE_RE = re.compile(
    r"^\s*(?:"
    r"about|advertisement|archive|author|breadcrumb|byline|categories?|chapter|comments?|contact|contents?|"
    r"copyright|date|download|footer|header|home|image|keywords?|menu|more|navigation|next|page|posted|"
    r"previous|privacy|published|read more|related|section|share|source|subscribe|tags?|title|updated"
    r")\s*[:|.-]\s*",
    re.IGNORECASE,
)
TOC_LINE_RE = re.compile(r"\.{3,}\s*\d+\s*$")
DOT_LEADER_RE = re.compile(r"\.{5,}")
DOMAIN_SOURCE_SECTION_RE = re.compile(r"^\s*Source:\s*.+?\.\s*Section:\s*", re.IGNORECASE)

BROAD_LM_BOILERPLATE_PHRASES = (
    "available here",
    "back to top",
    "click here",
    "continue reading",
    "cookie policy",
    "go to next",
    "go to previous",
    "learn more",
    "log in",
    "privacy policy",
    "read more",
    "related articles",
    "share this",
    "sign in",
    "sign up",
    "skip to content",
    "subscribe",
    "terms of service",
)


@dataclass(slots=True)
class CleanResult:
    record: DocumentRecord | None
    dropped_reason: str | None = None


def normalize_whitespace(text: str) -> str:
    return MULTISPACE_RE.sub(" ", text.replace("\x00", " ")).strip()


def scrub_pii(text: str) -> str:
    text = EMAIL_RE.sub("[EMAIL]", text)
    text = PHONE_RE.sub("[PHONE]", text)
    return SSN_RE.sub("[SSN]", text)


def _safe_ratio(numerator: int, denominator: int) -> float:
    return numerator / max(denominator, 1)


def broad_lm_quality_filter_reason(text: str, raw_text: str | None = None) -> str | None:
    """Simple inspectable heuristics for noisy broad web LM documents."""
    lower_text = text.lower()
    words = text.split()
    url_count = len(URL_RE.findall(text))
    if url_count >= 3 or _safe_ratio(url_count, len(words)) > 0.015:
        return "broad_lm_url_heavy"

    boilerplate_hits = sum(1 for phrase in BROAD_LM_BOILERPLATE_PHRASES if phrase in lower_text)
    if boilerplate_hits >= 2:
        return "broad_lm_navigation_boilerplate"

    original = raw_text if raw_text is not None else text
    lines = [line.strip() for line in original.splitlines() if line.strip()]
    if len(lines) >= 4:
        bullet_lines = sum(1 for line in lines if BULLET_OR_LIST_RE.match(line))
        label_lines = sum(1 for line in lines if LABEL_LINE_RE.match(line))
        toc_lines = sum(1 for line in lines if TOC_LINE_RE.search(line))
        short_fragment_lines = sum(1 for line in lines if len(line.split()) <= 6)
        repeated_line_ratio = 1.0 - _safe_ratio(len(set(lines)), len(lines))

        if toc_lines >= 3:
            return "broad_lm_table_of_contents"
        if bullet_lines >= 4 and _safe_ratio(bullet_lines, len(lines)) >= 0.45:
            return "broad_lm_list_heavy"
        if label_lines >= 3 and _safe_ratio(label_lines, len(lines)) >= 0.25:
            return "broad_lm_metadata_heavy"
        if short_fragment_lines >= 6 and _safe_ratio(short_fragment_lines, len(lines)) >= 0.6:
            return "broad_lm_fragment_heavy"
        if len(lines) >= 8 and repeated_line_ratio >= 0.35:
            return "broad_lm_repeated_lines"

    dense_metadata_hits = len(
        re.findall(
            r"\b(?:posted|updated|categories?|tags?|related|read more|subscribe|share|comments?)\b",
            lower_text,
        )
    )
    if dense_metadata_hits >= 8:
        return "broad_lm_metadata_heavy"
    return None


def normalize_domain_lm_text(text: str) -> str:
    """Remove scrape provenance while keeping the domain text itself available for LM training."""
    text = DOMAIN_SOURCE_SECTION_RE.sub("", text).strip()
    text = re.sub(r"\s*\|\s*", ", ", text)
    text = re.sub(r"\s+-\s+", "; ", text)
    text = DOT_LEADER_RE.sub(" ", text)
    return normalize_whitespace(text)


def domain_lm_quality_filter_reason(text: str, raw_text: str | None = None) -> str | None:
    """Heuristics for Webb/catalog prose that should read like continuable text, not scraped chrome."""
    raw = raw_text if raw_text is not None else text
    raw_lower = raw.lower()
    words = text.split()
    if (
        "american heritage dictionary of the english language" in raw_lower
        or "curriculum detail" in raw_lower
        or "photo of " in raw_lower
    ):
        return "domain_lm_structured_source_junk"
    if "section: contents" in raw_lower or TOC_LINE_RE.search(raw) or DOT_LEADER_RE.search(raw):
        return "domain_lm_table_of_contents"
    if "top 40 colleges webb students matriculate to most" in raw_lower and len(words) < 20:
        return "domain_lm_list_fragment"
    if len(words) < 8:
        return "domain_lm_fragment"
    pipe_count = raw.count("|")
    if pipe_count >= 12 and len(words) < 80:
        return "domain_lm_dense_table_row"
    source_prefix_hits = len(re.findall(r"\bsource:\s*", raw_lower))
    section_prefix_hits = len(re.findall(r"\bsection:\s*", raw_lower))
    if source_prefix_hits + section_prefix_hits >= 3:
        return "domain_lm_metadata_heavy"
    return None


def quality_filter_reason(
    text: str,
    config: DataConfig,
    *,
    source_config: DataSourceConfig | None = None,
    raw_text: str | None = None,
) -> str | None:
    if len(text) < config.min_document_chars or len(text) > config.max_document_chars:
        return "too_short" if len(text) < config.min_document_chars else "too_long"
    alpha_chars = sum(char.isalpha() for char in text)
    if not text:
        return "empty"
    if alpha_chars / max(len(text), 1) < 0.2:
        return "low_alpha_ratio"
    if text.count("http") > 100:
        return "too_many_urls"
    if source_config is not None and source_config.quality_filter_mode == "domain_lm":
        return domain_lm_quality_filter_reason(text, raw_text=raw_text)
    if source_config is not None and source_config.quality_filter_mode == "broad_lm":
        return broad_lm_quality_filter_reason(text, raw_text=raw_text)
    return None


def quality_filter(text: str, config: DataConfig) -> bool:
    return quality_filter_reason(text, config) is None


def stable_document_hash(text: str) -> str:
    return hashlib.blake2b(text.encode("utf-8"), digest_size=16).hexdigest()


def clean_document(
    record: DocumentRecord,
    data_config: DataConfig,
    source_config: DataSourceConfig,
    seen_hashes: set[str] | None = None,
) -> CleanResult:
    raw_text = record.text
    text = normalize_whitespace(record.text)
    if source_config.pii_scrub:
        text = scrub_pii(text)
    if source_config.quality_filter_mode == "domain_lm":
        text = normalize_domain_lm_text(text)
    if source_config.quality_filter:
        drop_reason = quality_filter_reason(
            text,
            data_config,
            source_config=source_config,
            raw_text=raw_text,
        )
        if drop_reason is not None:
            return CleanResult(record=None, dropped_reason=drop_reason)
    doc_hash = stable_document_hash(text)
    if source_config.deduplicate and seen_hashes is not None:
        if doc_hash in seen_hashes:
            return CleanResult(record=None, dropped_reason="duplicate")
        seen_hashes.add(doc_hash)
    record.text = text
    record.document_id = record.document_id or doc_hash
    return CleanResult(record=record)
