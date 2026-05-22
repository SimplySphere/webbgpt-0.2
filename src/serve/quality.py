from __future__ import annotations

import re
from typing import Any


_SEPARATOR_BURST_RE = re.compile(r"([,./>\-])\1{2,}")
_TOKEN_RE = re.compile(r"\S+")
_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9'-]{2,}")
_HYPHEN_LETTER_GARBAGE_RE = re.compile(r"\b(?:[A-Za-z]{1,3}-){3,}[A-Za-z]{1,3}\b")
_COPYRIGHT_YEAR_RE = re.compile(r"(?:©|copyright)\s*(?:19|20)\d{2}", re.IGNORECASE)
_PUNCT_ONLY_RE = re.compile(r"^[\W_]+$")
_ABSTENTION_RE = re.compile(
    r"(does not contain enough information|cannot answer|could not find|not enough information|not provided)",
    re.IGNORECASE,
)


def _content_terms(text: str) -> set[str]:
    stop = {
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
        "how",
        "into",
        "its",
        "that",
        "the",
        "their",
        "this",
        "what",
        "when",
        "where",
        "which",
        "while",
        "with",
    }
    return {token.lower().strip("'-") for token in _WORD_RE.findall(text) if token.lower().strip("'-") not in stop}


def _has_repeated_ngram(tokens: list[str], n: int, threshold: int) -> bool:
    if len(tokens) < n:
        return False
    counts: dict[tuple[str, ...], int] = {}
    for index in range(len(tokens) - n + 1):
        ngram = tuple(tokens[index : index + n])
        counts[ngram] = counts.get(ngram, 0) + 1
    return max(counts.values(), default=0) >= threshold


def analyze_generation(
    text: str,
    *,
    prompt: str | None = None,
    context: str | None = None,
    grounded: bool = False,
) -> dict[str, Any]:
    stripped = text.strip()
    nonspace_chars = [char for char in stripped if not char.isspace()]
    nonspace_count = len(nonspace_chars)
    alpha_count = sum(char.isalpha() for char in nonspace_chars)
    alpha_ratio = 0.0 if nonspace_count == 0 else alpha_count / float(nonspace_count)
    comma_ratio = 0.0 if nonspace_count == 0 else stripped.count(",") / float(nonspace_count)
    separator_bursts = len(_SEPARATOR_BURST_RE.findall(stripped))

    raw_tokens = _TOKEN_RE.findall(stripped)
    normalized_tokens = [re.sub(r"[^a-z0-9]+", "", token.lower()) for token in raw_tokens]
    nonempty_normalized = [token for token in normalized_tokens if token]
    content_terms = _content_terms(stripped)

    short_fragment_ratio = 0.0
    punctuation_suffix_ratio = 0.0
    unique_token_ratio = 0.0
    if raw_tokens:
        short_fragments = sum(1 for token in normalized_tokens if len(token) <= 2)
        punctuation_suffixes = sum(1 for token in raw_tokens if token[-1] in ",./->")
        short_fragment_ratio = short_fragments / float(len(raw_tokens))
        punctuation_suffix_ratio = punctuation_suffixes / float(len(raw_tokens))
    if nonempty_normalized:
        unique_token_ratio = len(set(nonempty_normalized)) / float(len(nonempty_normalized))

    repeated_token_run = 1
    current_run = 1
    previous = None
    for token in nonempty_normalized:
        if token == previous:
            current_run += 1
        else:
            current_run = 1
            previous = token
        repeated_token_run = max(repeated_token_run, current_run)

    reasons: list[str] = []
    if separator_bursts >= 2:
        reasons.append("separator_spam")
    if nonspace_count >= 40 and alpha_ratio < 0.45:
        reasons.append("low_alphabetic_ratio")
    if len(raw_tokens) >= 12 and short_fragment_ratio > 0.55:
        reasons.append("too_many_short_fragments")
    if len(raw_tokens) >= 12 and punctuation_suffix_ratio > 0.55:
        reasons.append("malformed_token_dump")
    if repeated_token_run >= 4:
        reasons.append("repeated_token_burst")
    if nonspace_count >= 40 and comma_ratio > 0.12:
        reasons.append("comma_spam")
    if len(nonempty_normalized) >= 12 and unique_token_ratio < 0.3:
        reasons.append("low_token_variety")
    if not stripped:
        reasons.append("blank_output")
    elif _PUNCT_ONLY_RE.match(stripped):
        reasons.append("punctuation_only")
    if nonspace_count <= 3:
        reasons.append("too_short_no_content")
    elif len(nonempty_normalized) <= 2 and len(content_terms) == 0:
        reasons.append("too_short_no_content")
    if _HYPHEN_LETTER_GARBAGE_RE.search(stripped):
        reasons.append("hyphen_letter_garbage")
    if _COPYRIGHT_YEAR_RE.search(stripped) and len(raw_tokens) <= 8:
        reasons.append("copyright_date_residue")
    if 0 < nonspace_count < 40 and alpha_ratio < 0.35:
        reasons.append("low_alphabetic_ratio")
    if _has_repeated_ngram(nonempty_normalized, 2, 4) or _has_repeated_ngram(nonempty_normalized, 3, 3):
        reasons.append("repeated_phrase")
    if len(nonempty_normalized) >= 10:
        token_counts: dict[str, int] = {}
        for token in nonempty_normalized:
            token_counts[token] = token_counts.get(token, 0) + 1
        most_common_count = max(token_counts.values(), default=0)
        if most_common_count >= 3 and most_common_count / len(nonempty_normalized) >= 0.18:
            reasons.append("dominant_repeated_token")
    if grounded and stripped.startswith((".", "?", "!", ",", ";", ":")):
        reasons.append("leading_punctuation_fragment")

    prompt_terms = _content_terms(prompt or "")
    context_terms = _content_terms(context or "")
    output_terms = _content_terms(stripped)
    prompt_overlap = sorted(output_terms & prompt_terms)
    context_overlap = sorted(output_terms & context_terms)
    abstention_like = bool(_ABSTENTION_RE.search(stripped))
    if grounded and len(raw_tokens) <= 3 and not abstention_like:
        reasons.append("grounded_answer_too_short")
    if grounded and context_terms and output_terms and not context_overlap and not abstention_like:
        reasons.append("ignores_provided_context")
    if grounded and prompt_terms and output_terms and not prompt_overlap and not context_overlap and not abstention_like:
        reasons.append("prompt_not_reflected")
    if grounded and len(prompt_terms) >= 3 and output_terms and len(prompt_overlap) < 2 and not abstention_like:
        reasons.append("low_prompt_retention")

    degenerate = (
        len(reasons) >= 2
        or separator_bursts >= 4
        or repeated_token_run >= 6
        or (len(raw_tokens) >= 16 and punctuation_suffix_ratio > 0.7 and short_fragment_ratio > 0.6)
        or any(
            reason in reasons
            for reason in (
                "blank_output",
                "punctuation_only",
                "hyphen_letter_garbage",
                "copyright_date_residue",
                "grounded_answer_too_short",
                "ignores_provided_context",
                "leading_punctuation_fragment",
                "low_prompt_retention",
            )
        )
        or (grounded and ("repeated_phrase" in reasons or "dominant_repeated_token" in reasons))
    )
    return {
        "degenerate": degenerate,
        "reasons": reasons,
        "metrics": {
            "nonspace_chars": nonspace_count,
            "alpha_ratio": round(alpha_ratio, 4),
            "comma_ratio": round(comma_ratio, 4),
            "separator_bursts": separator_bursts,
            "token_count": len(raw_tokens),
            "short_fragment_ratio": round(short_fragment_ratio, 4),
            "punctuation_suffix_ratio": round(punctuation_suffix_ratio, 4),
            "unique_token_ratio": round(unique_token_ratio, 4),
            "repeated_token_run": repeated_token_run,
            "content_term_count": len(output_terms),
            "prompt_overlap_terms": prompt_overlap[:12],
            "context_overlap_terms": context_overlap[:12],
        },
    }


def degenerate_output_message() -> str:
    return (
        "Weak generation. The local-MVP model produced an unreliable answer, so this response should be labeled low confidence."
    )
