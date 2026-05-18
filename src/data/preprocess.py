from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from functools import lru_cache

from config import DataConfig, DataSourceConfig
from data.schemas import DocumentRecord


EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
PHONE_RE = re.compile(r"(?:(?:\+?1[-.\s]*)?(?:\(\d{3}\)|\d{3})[-.\s]*)\d{3}[-.\s]*\d{4}")
SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
MULTISPACE_RE = re.compile(r"\s+")
URL_RE = re.compile(r"\b(?:https?://|www\.)\S+|\b\S+\.(?:com|org|net|edu|gov|io|co)\S*", re.IGNORECASE)
BULLET_OR_LIST_RE = re.compile(r"^\s*(?:[-*\u2022]+|\d+[.)]|[A-Z][.)])\s+")
SENTENCE_RE = re.compile(r"[.!?]+(?:\s+|$)")
WORD_RE = re.compile(r"[A-Za-z][A-Za-z']*")
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

BROAD_LM_MEDICAL_BODY_TERMS = (
    "blood",
    "bodies",
    "body",
    "child",
    "children",
    "diabetes",
    "diet",
    "disease",
    "doctor",
    "food",
    "health",
    "infection",
    "medical",
    "pain",
    "symptoms",
    "treatment",
    "viral",
    "virus",
)

BROAD_LM_NAVIGATION_TERMS = (
    "accept cookies",
    "advertisement",
    "back to top",
    "click here",
    "continue reading",
    "cookie policy",
    "footer",
    "go to next",
    "go to previous",
    "header",
    "home",
    "learn more",
    "log in",
    "menu",
    "privacy policy",
    "read more",
    "related articles",
    "search",
    "share this",
    "sign in",
    "sign up",
    "skip to content",
    "subscribe",
    "terms of service",
)

BROAD_LM_GENERIC_ARTICLE_FORMULAE = (
    "in the same time",
    "the first step",
    "the most important",
    "the process",
)

BROAD_LM_MALFORMED_FRAGMENT_TERMS = (
    "F-the-s",
    "Newth",
    "based-in",
    "in-in",
    "to-in",
)

BROAD_LM_PRODUCT_COMMERCIAL_TERMS = (
    "affiliate",
    "amazon",
    "best price",
    "buy now",
    "cart",
    "checkout",
    "coupon",
    "customer reviews",
    "discount",
    "free shipping",
    "lowest price",
    "order now",
    "price",
    "product",
    "products",
    "rating",
    "reviews",
    "sale",
    "shop",
    "shopping",
    "sponsored",
    "store",
    "warranty",
)

BROAD_LM_DICTIONARY_FRAGMENT_TERMS = (
    "adjective",
    "antonyms",
    "definition",
    "dictionary",
    "encyclopedia",
    "etymology",
    "noun",
    "plural",
    "pronunciation",
    "synonyms",
    "thesaurus",
    "verb",
    "word origin",
)

BROAD_LM_PAGE_BOILERPLATE_TERMS = (
    "all rights reserved",
    "archived from",
    "copyright",
    "edit this page",
    "follow us",
    "last modified",
    "page was last edited",
    "permalink",
    "printable version",
    "retrieved from",
    "submit",
    "view source",
)

BROAD_LM_PAGE_INSTRUCTION_PHRASES = (
    "add to this article",
    "audiencestart",
    "begin typing or use",
    "click submit",
    "discover the cosmos",
    "editing tools above",
    "editors will review what",
    "imcopy",
    "invited audience members",
    "publishing partner program",
    "remote presentation",
    "send the link below",
    "simply begin typing",
    "submitted and determine whether to revise",
)

CURATED_LM_FAMILYSEARCH_WIKI_COMPACT_PATTERNS = (
    "editthispagefromfamilysearchwiki",
    "familysearchwiki",
    "editthispage",
)

CURATED_LM_HTML_ARCHIVE_BOILERPLATE_PHRASES = (
    "the following html text is provided to enhance online readability",
    "provided to enhance online readability",
    "archived web page remains online",
    "this page will not be altered or updated",
    "this article has been archived from the now-defunct",
    "has been archived from the now-defunct",
    "this text is part of: table of contents:",
)

CURATED_LM_SCIENCE_FAIR_ENCYCLOPEDIA_PHRASES = (
    "science fair project encyclopedia",
)

CURATED_LM_LOOKUP_FRAGMENT_PHRASES = (
    "one of the mysteries of the english language finally explained",
    "in english, many things are named after a particular country",
    "view your list of saved words",
)

CURATED_LM_COOKIE_VIDEO_WIDGET_PHRASES = (
    "please activate cookies in your browser",
    "please enable cookies in your browser",
    "cookies are disabled in your browser",
    "by using this site you agree to our use of cookies",
    "rating is available when the video has been rented",
    "sign in to add this video",
    "sign in to report inappropriate content",
    "login to rate",
    "log in to rate",
    "watch queue queue",
    "add to want to watch this again later",
)

CURATED_LM_PRODUCT_WIDGET_PHRASES = (
    "the leading ebooks store online",
    "huge product rangeover",
    "140,000 books & equipment products rapid shipping",
)

CURATED_LM_NEWSLETTER_TAGLINE_PHRASES = (
    "the latest news from academia, regulators research labs",
    "latest news from academia, regulators research labs and other things of interest",
    "tech moves fast! stay ahead of the curve",
    "learn something new every day",
    "subscribe today to get the latest",
    "delivered right to your inbox",
    "right in your inbox. sign up for our email newsletter",
    "follow us on facebook",
    "follow us on twitter",
    "follow @universetoday",
    "want to stay on top of all the space news? follow",
    "grow your tree by exploring billions of historical records",
    "700 journals and 15,000,000 readers",
    "readership is 10 times more when compared to other subscription journals",
)

CURATED_LM_SEO_FRAGMENT_PHRASES = (
    "best essay writing service",
    "essay format generator",
    "essay quizlet",
    "what is 1 pages essay double spaced",
    "how many pages best essay",
    "top research paper topics good or bad",
)

CURATED_LM_RESIDUAL_WEB_BOILERPLATE_PHRASES = (
    "this preview has intentionally blurred sections",
    "sign up to view the full version",
    "this article is only available in the pdf format",
    "download the pdf to view the article",
    "it looks like you're using an ad blocker",
    "below are the first 10 and last 10 pages of uncorrected machine-read text",
    "search for native plants by scientific name, common name or family",
)

SEPARATOR_FRAGMENT_RE = re.compile(r"(?:[-_=*#]{4,}|[|]{2,}|/{3,}|>{3,}|<{3,})")
PIPE_CATEGORY_CHAIN_RE = re.compile(
    r"\b(?:individual differences|methods|statistics|clinical|educational|industrial|professional items|"
    r"world psychology)\b(?:\s*\|\s*\b(?:individual differences|methods|statistics|clinical|educational|"
    r"industrial|professional items|world psychology)\b){3,}",
    re.IGNORECASE,
)
QUESTION_START_RE = re.compile(r"(?:^|[.!?]\s+)(?:what|how|why|when|where|which|who|can|do|does|is|are)\b")

DOMAIN_LM_SYNTHETIC_SCAFFOLD_PHRASES = (
    "raw language-model pretraining",
    "course-planning paragraph",
    "strong continuation",
    "continuation target",
    "the model",
    "same planning principle",
    "in scenario",
    "student-facing explanation",
    "the same answer should switch",
    "catalog evidence",
    "should remain anchored",
    "useful catalog continuation",
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


def _word_count(text: str) -> int:
    return len(WORD_RE.findall(text))


@lru_cache(maxsize=None)
def _term_pattern(terms: tuple[str, ...]) -> re.Pattern[str]:
    alternatives = []
    for term in terms:
        parts = [re.escape(part) for part in term.lower().split()]
        alternatives.append(r"\b" + r"\s+".join(parts) + r"\b")
    return re.compile("|".join(alternatives))


def _term_count(lower_text: str, terms: tuple[str, ...]) -> int:
    return len(_term_pattern(terms).findall(lower_text))


def _compact_artifact_text(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def _contains_any_phrase(lower_text: str, raw_lower: str, phrases: tuple[str, ...]) -> bool:
    return any(phrase in lower_text or phrase in raw_lower for phrase in phrases)


def _ngram_repeat_signal(words: list[str], n: int) -> tuple[int, float]:
    if len(words) < n:
        return 0, 0.0
    counts: dict[tuple[str, ...], int] = {}
    for index in range(len(words) - n + 1):
        key = tuple(words[index : index + n])
        counts[key] = counts.get(key, 0) + 1
    max_count = max(counts.values(), default=0)
    repeated_extra = sum(count - 1 for count in counts.values() if count > 1)
    return max_count, _safe_ratio(repeated_extra, len(words) - n + 1)


def _line_shape_reason(lines: list[str]) -> str | None:
    if len(lines) < 4:
        return None
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
    return None


def _pipe_category_menu_reason(raw: str, word_count: int) -> str | None:
    pipe_count = raw.count("|")
    if pipe_count < 4:
        return None
    if PIPE_CATEGORY_CHAIN_RE.search(raw):
        return "curated_lm_category_menu_boilerplate"
    first_pipe_index = raw.find("|")
    if first_pipe_index > 450:
        return None
    segments = [segment.strip() for segment in raw.split("|") if segment.strip()]
    if len(segments) < 5:
        return None
    short_segments = sum(1 for segment in segments if 1 <= _word_count(segment) <= 4)
    pipe_density = _safe_ratio(pipe_count, word_count)
    if pipe_count >= 6 and _safe_ratio(short_segments, len(segments)) >= 0.6 and pipe_density > 0.01:
        return "curated_lm_category_menu_boilerplate"
    return None


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
    line_shape_reason = _line_shape_reason(lines)
    if line_shape_reason is not None:
        return line_shape_reason

    dense_metadata_hits = len(
        re.findall(
            r"\b(?:posted|updated|categories?|tags?|related|read more|subscribe|share|comments?)\b",
            lower_text,
        )
    )
    if dense_metadata_hits >= 8:
        return "broad_lm_metadata_heavy"
    return None


def curated_lm_quality_filter_reason(text: str, raw_text: str | None = None) -> str | None:
    """Stricter real-web selector for scalable broad LM pretraining."""
    broad_reason = broad_lm_quality_filter_reason(text, raw_text=raw_text)
    if broad_reason is not None:
        return broad_reason

    raw = raw_text if raw_text is not None else text
    lower_text = text.lower()
    raw_lower = raw.lower()
    words = [word.lower() for word in WORD_RE.findall(text)]
    word_count = len(words)
    compact_text = _compact_artifact_text(f"{text} {raw}")

    if any(pattern in compact_text for pattern in CURATED_LM_FAMILYSEARCH_WIKI_COMPACT_PATTERNS):
        return "curated_lm_familysearch_wiki_boilerplate"
    if _contains_any_phrase(lower_text, raw_lower, CURATED_LM_HTML_ARCHIVE_BOILERPLATE_PHRASES):
        return "curated_lm_html_archive_boilerplate"
    if _contains_any_phrase(lower_text, raw_lower, CURATED_LM_SCIENCE_FAIR_ENCYCLOPEDIA_PHRASES):
        return "curated_lm_science_fair_encyclopedia"
    if _contains_any_phrase(lower_text, raw_lower, CURATED_LM_LOOKUP_FRAGMENT_PHRASES):
        return "curated_lm_dictionary_or_encyclopedia_fragment"
    if _contains_any_phrase(lower_text, raw_lower, CURATED_LM_COOKIE_VIDEO_WIDGET_PHRASES):
        return "curated_lm_cookie_video_widget_boilerplate"
    if _contains_any_phrase(lower_text, raw_lower, CURATED_LM_PRODUCT_WIDGET_PHRASES):
        return "curated_lm_product_commercial_dense"
    if _contains_any_phrase(lower_text, raw_lower, CURATED_LM_NEWSLETTER_TAGLINE_PHRASES):
        return "curated_lm_newsletter_or_source_tagline"
    if _contains_any_phrase(lower_text, raw_lower, CURATED_LM_SEO_FRAGMENT_PHRASES):
        return "curated_lm_question_list_or_seo_fragments"
    if _contains_any_phrase(lower_text, raw_lower, CURATED_LM_RESIDUAL_WEB_BOILERPLATE_PHRASES):
        return "curated_lm_residual_web_boilerplate"
    pipe_menu_reason = _pipe_category_menu_reason(raw, word_count)
    if pipe_menu_reason is not None:
        return pipe_menu_reason

    if word_count < 80:
        return "curated_lm_too_few_words"

    sentence_count = len(SENTENCE_RE.findall(text))
    if sentence_count < 3:
        return "curated_lm_too_few_sentences"
    question_mark_count = text.count("?")
    question_start_count = len(QUESTION_START_RE.findall(lower_text))
    if (
        question_mark_count >= 5
        and question_start_count >= 4
        and _safe_ratio(question_mark_count, sentence_count) >= 0.35
    ):
        return "curated_lm_question_list_or_seo_fragments"

    avg_sentence_words = word_count / max(sentence_count, 1)
    if avg_sentence_words < 7:
        return "curated_lm_fragmentary_sentences"
    if avg_sentence_words > 48 and sentence_count < 6:
        return "curated_lm_runon_or_low_sentence_structure"

    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    line_shape_reason = _line_shape_reason(lines)
    if line_shape_reason is not None:
        return line_shape_reason

    url_count = len(URL_RE.findall(text))
    separator_count = len(SEPARATOR_FRAGMENT_RE.findall(raw))
    if separator_count >= 4 or _safe_ratio(separator_count, word_count) > 0.01:
        return "curated_lm_separator_heavy"
    if url_count >= 2 or _safe_ratio(url_count, word_count) > 0.008:
        return "curated_lm_url_or_metadata_heavy"

    medical_count = _term_count(lower_text, BROAD_LM_MEDICAL_BODY_TERMS)
    if medical_count >= 5 and _safe_ratio(medical_count, word_count) > 0.015:
        return "curated_lm_medical_body_dense"

    product_count = _term_count(lower_text, BROAD_LM_PRODUCT_COMMERCIAL_TERMS)
    if product_count >= 4 and _safe_ratio(product_count, word_count) > 0.012:
        return "curated_lm_product_commercial_dense"

    page_boilerplate_count = _term_count(lower_text, BROAD_LM_PAGE_BOILERPLATE_TERMS)
    if page_boilerplate_count >= 2 or _safe_ratio(page_boilerplate_count, word_count) > 0.012:
        return "curated_lm_page_boilerplate"
    if any(phrase in lower_text for phrase in BROAD_LM_PAGE_INSTRUCTION_PHRASES):
        return "curated_lm_page_instruction_boilerplate"

    dictionary_count = _term_count(lower_text, BROAD_LM_DICTIONARY_FRAGMENT_TERMS)
    dictionary_density = _safe_ratio(dictionary_count, word_count)
    if "american heritage" in lower_text and "dictionary" in lower_text:
        return "curated_lm_dictionary_or_encyclopedia_fragment"
    if dictionary_count >= 3 and (dictionary_density > 0.01 or sentence_count <= 5):
        return "curated_lm_dictionary_or_encyclopedia_fragment"

    fivegram_max, fivegram_repeat_rate = _ngram_repeat_signal(words, 5)
    eightgram_max, eightgram_repeat_rate = _ngram_repeat_signal(words, 8)
    if fivegram_max >= 4 or eightgram_max >= 3 or fivegram_repeat_rate > 0.06 or eightgram_repeat_rate > 0.035:
        return "curated_lm_repeated_ngram_heavy"

    prose_score = 0
    if sentence_count >= 5:
        prose_score += 1
    if 9 <= avg_sentence_words <= 34:
        prose_score += 1
    if any(marker in lower_text for marker in ("because", "therefore", "for example", "however", "although", "when ")):
        prose_score += 1
    if any(char in text for char in ",;:"):
        prose_score += 1
    if len(set(words)) / max(word_count, 1) >= 0.38:
        prose_score += 1
    if prose_score < 2:
        return "curated_lm_low_prose_quality_signal"

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
    lower_text = text.lower()
    words = text.split()
    if any(
        phrase in lower_text or phrase in raw_lower
        for phrase in DOMAIN_LM_SYNTHETIC_SCAFFOLD_PHRASES
    ):
        return "domain_lm_synthetic_training_scaffold"
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
    if source_config is not None and source_config.quality_filter_mode == "curated_lm":
        return curated_lm_quality_filter_reason(text, raw_text=raw_text)
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
