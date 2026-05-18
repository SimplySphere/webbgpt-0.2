from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from config import DataConfig, DataSourceConfig, load_config  # noqa: E402
from data.preprocess import (  # noqa: E402
    BROAD_LM_DICTIONARY_FRAGMENT_TERMS,
    BROAD_LM_MEDICAL_BODY_TERMS,
    BROAD_LM_NAVIGATION_TERMS,
    BROAD_LM_PAGE_BOILERPLATE_TERMS,
    BROAD_LM_PAGE_INSTRUCTION_PHRASES,
    BROAD_LM_PRODUCT_COMMERCIAL_TERMS,
    CURATED_LM_COOKIE_VIDEO_WIDGET_PHRASES,
    CURATED_LM_FAMILYSEARCH_WIKI_COMPACT_PATTERNS,
    CURATED_LM_HTML_ARCHIVE_BOILERPLATE_PHRASES,
    CURATED_LM_LOOKUP_FRAGMENT_PHRASES,
    CURATED_LM_NEWSLETTER_TAGLINE_PHRASES,
    CURATED_LM_PRODUCT_WIDGET_PHRASES,
    CURATED_LM_RESIDUAL_WEB_BOILERPLATE_PHRASES,
    CURATED_LM_SCIENCE_FAIR_ENCYCLOPEDIA_PHRASES,
    CURATED_LM_SEO_FRAGMENT_PHRASES,
    DOMAIN_LM_SYNTHETIC_SCAFFOLD_PHRASES,
    SENTENCE_RE,
    SEPARATOR_FRAGMENT_RE,
    URL_RE,
    WORD_RE,
    _line_shape_reason,
    _ngram_repeat_signal,
    _safe_ratio,
    _term_count,
    clean_document,
    normalize_whitespace,
)
from data.schemas import DocumentRecord  # noqa: E402
from tokenizer import SentencePieceTokenizer  # noqa: E402


MAJOR_REJECTION_REASONS = (
    "curated_lm_medical_body_dense",
    "curated_lm_repeated_ngram_heavy",
    "broad_lm_url_heavy",
    "curated_lm_url_or_metadata_heavy",
    "curated_lm_separator_heavy",
    "broad_lm_navigation_boilerplate",
    "curated_lm_page_boilerplate",
    "curated_lm_product_commercial_dense",
    "curated_lm_dictionary_or_encyclopedia_fragment",
    "curated_lm_familysearch_wiki_boilerplate",
    "curated_lm_html_archive_boilerplate",
    "curated_lm_science_fair_encyclopedia",
    "curated_lm_cookie_video_widget_boilerplate",
    "curated_lm_newsletter_or_source_tagline",
    "curated_lm_category_menu_boilerplate",
    "curated_lm_question_list_or_seo_fragments",
    "curated_lm_residual_web_boilerplate",
)

HIDDEN_PATTERNS: dict[str, tuple[str, ...]] = {
    "medical_body_terms": BROAD_LM_MEDICAL_BODY_TERMS,
    "product_commercial_terms": BROAD_LM_PRODUCT_COMMERCIAL_TERMS,
    "dictionary_fragment_terms": BROAD_LM_DICTIONARY_FRAGMENT_TERMS,
    "edit_this_page": ("edit this page",),
    "follow_us": ("follow us",),
    "copyright": ("copyright",),
    "archived_web_page": ("archived web page", "archived from"),
    "list_menu_navigation_residue": BROAD_LM_NAVIGATION_TERMS,
    "learn_something_new_every_day": ("learn something new every day",),
    "familysearch_wiki_boilerplate": (
        "edit this page",
        "edit this page from familysearch wiki",
        "familysearch wiki",
        *CURATED_LM_FAMILYSEARCH_WIKI_COMPACT_PATTERNS,
    ),
    "html_archive_boilerplate": CURATED_LM_HTML_ARCHIVE_BOILERPLATE_PHRASES,
    "science_fair_encyclopedia": CURATED_LM_SCIENCE_FAIR_ENCYCLOPEDIA_PHRASES,
    "lookup_fragment_boilerplate": CURATED_LM_LOOKUP_FRAGMENT_PHRASES,
    "cookie_video_widget_boilerplate": CURATED_LM_COOKIE_VIDEO_WIDGET_PHRASES,
    "product_widget_boilerplate": CURATED_LM_PRODUCT_WIDGET_PHRASES,
    "newsletter_source_tagline": CURATED_LM_NEWSLETTER_TAGLINE_PHRASES,
    "seo_question_fragment": CURATED_LM_SEO_FRAGMENT_PHRASES,
    "residual_web_boilerplate": CURATED_LM_RESIDUAL_WEB_BOILERPLATE_PHRASES,
    "category_menu_chain": (
        "individual differences | methods | statistics | clinical",
        "methods | statistics | clinical | educational",
    ),
    "article_page_boilerplate": (
        *BROAD_LM_PAGE_BOILERPLATE_TERMS,
        *BROAD_LM_PAGE_INSTRUCTION_PHRASES,
    ),
    "generated_domain_remnants": (
        "local_mvp_v2_",
        "webb_catalog_large_lm_corpus",
        "webb_advising_large_lm_corpus",
        "webb_domain_large_lm_corpus",
        "webb_handbook_large_lm_corpus",
        "webb_school_context_large_lm_corpus",
        *DOMAIN_LM_SYNTHETIC_SCAFFOLD_PHRASES,
    ),
}

ACCEPTED_SAMPLE_ALIASES = {
    "fineweb_edu_curated_pretrain": "accepted_fineweb_edu_samples.txt",
    "fineweb_extension_curated_pretrain": "accepted_fineweb_extension_samples.txt",
    "local_mvp_curated_real_pretrain": "accepted_local_samples.txt",
    "fineweb_extension_corpus": "accepted_fineweb_extension_samples.txt",
    "local_mvp_pretrain_corpus": "accepted_local_samples.txt",
}


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _source_reports(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    reports = manifest.get("diagnostics", {}).get("per_source", [])
    return {
        str(report.get("source")): report
        for report in reports
        if isinstance(report, dict) and report.get("source")
    }


def _excerpt(text: str, *, limit: int = 1000) -> str:
    text = normalize_whitespace(text)
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def _context(text: str, index: int, *, radius: int = 360) -> str:
    start = max(index - radius, 0)
    end = min(index + radius, len(text))
    return _excerpt(text[start:end], limit=(2 * radius) + 80)


def _safe_mean(values: list[int]) -> float:
    return round(statistics.mean(values), 2) if values else 0.0


def _safe_median(values: list[int]) -> float:
    return round(statistics.median(values), 2) if values else 0.0


def _density(count: int, denominator: int) -> float:
    return round(count / max(denominator, 1), 6)


def _doc_diagnostics(text: str, raw_text: str | None = None) -> dict[str, Any]:
    raw = raw_text if raw_text is not None else text
    lower = text.lower()
    words = [word.lower() for word in WORD_RE.findall(text)]
    word_count = len(words)
    sentence_count = len(SENTENCE_RE.findall(text))
    line_shape_reason = _line_shape_reason([line.strip() for line in raw.splitlines() if line.strip()])
    fivegram_max, fivegram_repeat_rate = _ngram_repeat_signal(words, 5)
    eightgram_max, eightgram_repeat_rate = _ngram_repeat_signal(words, 8)
    medical_count = _term_count(lower, BROAD_LM_MEDICAL_BODY_TERMS)
    product_count = _term_count(lower, BROAD_LM_PRODUCT_COMMERCIAL_TERMS)
    dictionary_count = _term_count(lower, BROAD_LM_DICTIONARY_FRAGMENT_TERMS)
    page_boilerplate_count = _term_count(lower, BROAD_LM_PAGE_BOILERPLATE_TERMS)
    navigation_count = _term_count(lower, BROAD_LM_NAVIGATION_TERMS)
    separator_count = len(SEPARATOR_FRAGMENT_RE.findall(raw))
    url_count = len(URL_RE.findall(text))
    pipe_count = raw.count("|")
    question_mark_count = text.count("?")
    unique_word_ratio = len(set(words)) / max(word_count, 1)
    prose_score = 0
    if sentence_count >= 5:
        prose_score += 1
    avg_sentence_words = word_count / max(sentence_count, 1)
    if 9 <= avg_sentence_words <= 34:
        prose_score += 1
    if any(marker in lower for marker in ("because", "therefore", "for example", "however", "although", "when ")):
        prose_score += 1
    if any(char in text for char in ",;:"):
        prose_score += 1
    if unique_word_ratio >= 0.38:
        prose_score += 1
    return {
        "chars": len(text),
        "words": word_count,
        "sentences": sentence_count,
        "avg_sentence_words": round(avg_sentence_words, 2),
        "line_shape_reason": line_shape_reason,
        "url_count": url_count,
        "separator_count": separator_count,
        "pipe_count": pipe_count,
        "question_mark_count": question_mark_count,
        "medical_count": medical_count,
        "medical_density": _density(medical_count, word_count),
        "product_count": product_count,
        "product_density": _density(product_count, word_count),
        "dictionary_count": dictionary_count,
        "dictionary_density": _density(dictionary_count, word_count),
        "page_boilerplate_count": page_boilerplate_count,
        "page_boilerplate_density": _density(page_boilerplate_count, word_count),
        "navigation_count": navigation_count,
        "navigation_density": _density(navigation_count, word_count),
        "fivegram_max": fivegram_max,
        "fivegram_repeat_rate": round(fivegram_repeat_rate, 6),
        "eightgram_max": eightgram_max,
        "eightgram_repeat_rate": round(eightgram_repeat_rate, 6),
        "unique_word_ratio": round(unique_word_ratio, 6),
        "prose_score": prose_score,
    }


def _brief_diagnostics(diagnostics: dict[str, Any]) -> str:
    keys = (
        "words",
        "sentences",
        "avg_sentence_words",
        "url_count",
        "separator_count",
        "pipe_count",
        "question_mark_count",
        "medical_count",
        "medical_density",
        "product_count",
        "product_density",
        "dictionary_count",
        "dictionary_density",
        "page_boilerplate_count",
        "navigation_count",
        "fivegram_max",
        "fivegram_repeat_rate",
        "eightgram_max",
        "eightgram_repeat_rate",
        "line_shape_reason",
        "prose_score",
    )
    return json.dumps({key: diagnostics.get(key) for key in keys}, sort_keys=True)


def _even_sample(items: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    if len(items) <= limit:
        return list(items)
    if limit <= 1:
        return [items[0]]
    indexes = [
        round(index * (len(items) - 1) / (limit - 1))
        for index in range(limit)
    ]
    return [items[index] for index in indexes]


def _write_sample_file(path: Path, title: str, rows: list[dict[str, Any]], tokenizer: SentencePieceTokenizer) -> None:
    lines = [title, "=" * len(title), ""]
    if not rows:
        lines.append("No accepted samples captured for this source.")
    for number, row in enumerate(rows, start=1):
        text = str(row.get("text", ""))
        token_count = len(tokenizer.encode(text, add_bos=True, add_eos=True)) if text else row.get("token_count", 0)
        diagnostics = row.get("diagnostics", {})
        lines.extend(
            [
                f"## sample {number}",
                f"source: {row.get('source')}",
                f"record_index: {row.get('record_index')}",
                f"document_id: {row.get('document_id')}",
                f"tokens: {token_count}",
                f"words: {diagnostics.get('words')}",
                f"sentences: {diagnostics.get('sentences')}",
                f"diagnostics: {_brief_diagnostics(diagnostics)}",
                "excerpt:",
                _excerpt(text, limit=1400),
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def _source_slug(source_name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", source_name.lower()).strip("_") or "unknown_source"


def _write_accepted_sample_files(
    output_dir: Path,
    accepted_by_source: dict[str, list[dict[str, Any]]],
    tokenizer: SentencePieceTokenizer,
    source_names: list[str],
) -> None:
    accepted_dir = output_dir / "accepted_samples_by_source"
    accepted_dir.mkdir(parents=True, exist_ok=True)
    for source_name in sorted(set(source_names) | set(accepted_by_source)):
        rows = _even_sample(accepted_by_source.get(source_name, []), 30)
        title = f"Accepted {source_name} samples"
        _write_sample_file(
            accepted_dir / f"accepted_{_source_slug(source_name)}_samples.txt",
            title,
            rows,
            tokenizer,
        )
        alias = ACCEPTED_SAMPLE_ALIASES.get(source_name)
        if alias:
            _write_sample_file(output_dir / alias, title, rows, tokenizer)


def _add_packed_sample_fallbacks(
    accepted_by_source: dict[str, list[dict[str, Any]]],
    source_names: list[str],
    packed_samples: list[dict[str, Any]],
) -> None:
    missing_sources = {
        source_name
        for source_name in source_names
        if not accepted_by_source.get(source_name)
    }
    if not missing_sources:
        return
    for packed_sample in packed_samples:
        text = str(packed_sample.get("text", ""))
        if not text:
            continue
        sample_sources = set(str(source) for source in packed_sample.get("source_names", []))
        for source_name in sorted(missing_sources & sample_sources):
            if len(accepted_by_source[source_name]) >= 30:
                continue
            contributors = [
                contributor
                for contributor in packed_sample.get("contributors", [])
                if contributor.get("source") == source_name
            ]
            accepted_by_source[source_name].append(
                {
                    "source": source_name,
                    "record_index": f"packed_sequence:{packed_sample.get('global_index')}",
                    "document_id": ",".join(
                        str(contributor.get("document_id", ""))
                        for contributor in contributors[:5]
                        if contributor.get("document_id")
                    ),
                    "diagnostics": _doc_diagnostics(text),
                    "text": text,
                }
            )


def _write_rejected_samples(output_dir: Path, samples: dict[str, list[dict[str, Any]]]) -> None:
    rejected_dir = output_dir / "rejected_samples_by_reason"
    rejected_dir.mkdir(parents=True, exist_ok=True)
    for reason in MAJOR_REJECTION_REASONS:
        rows = samples.get(reason, [])
        lines = [reason, "=" * len(reason), ""]
        if not rows:
            lines.append("No samples captured for this rejection reason.")
        for number, row in enumerate(rows, start=1):
            lines.extend(
                [
                    f"## sample {number}",
                    f"source: {row.get('source')}",
                    f"record_index: {row.get('record_index')}",
                    f"rejection_reason: {row.get('reason')}",
                    f"diagnostics: {_brief_diagnostics(row.get('diagnostics', {}))}",
                    "excerpt:",
                    _excerpt(str(row.get("text", "")), limit=1000),
                    "",
                ]
            )
        (rejected_dir / f"{reason}.txt").write_text("\n".join(lines), encoding="utf-8")


def _metadata_counts(manifest: dict[str, Any], hidden_patterns: dict[str, tuple[str, ...]]) -> tuple[Counter[tuple[str, str]], Counter[str]]:
    contributor_counts: Counter[tuple[str, str]] = Counter()
    hidden_counts: Counter[str] = Counter()
    for shard in manifest.get("shards", []):
        metadata_path = Path(str(shard.get("metadata_path", "")))
        if not metadata_path.exists():
            continue
        with metadata_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                lower_line = line.lower()
                for label, patterns in hidden_patterns.items():
                    if any(pattern.lower() in lower_line for pattern in patterns):
                        hidden_counts[label] += 1
                row = json.loads(line)
                for contributor in row.get("contributors", []):
                    source = str(contributor.get("source", ""))
                    document_id = str(contributor.get("document_id", ""))
                    if source and document_id:
                        contributor_counts[(source, document_id)] += 1
    return contributor_counts, hidden_counts


def _packed_sequence_samples(
    manifest: dict[str, Any],
    tokenizer: SentencePieceTokenizer,
    *,
    limit: int = 30,
) -> list[dict[str, Any]]:
    import numpy as np

    total_rows = int(manifest.get("num_sequences", 0))
    if total_rows <= 0:
        return []
    indexes = [round(index * (total_rows - 1) / (limit - 1)) for index in range(limit)]
    cumulative: list[tuple[int, dict[str, Any]]] = []
    total = 0
    for shard in manifest.get("shards", []):
        rows = int(shard.get("rows", 0))
        cumulative.append((total + rows, shard))
        total += rows

    samples: list[dict[str, Any]] = []
    shard_cache: dict[str, Any] = {}
    metadata_cache: dict[str, list[dict[str, Any]]] = {}
    pad_token_id = int(manifest.get("pad_token_id", -1))
    for global_index in indexes:
        previous_end = 0
        selected_shard: dict[str, Any] | None = None
        for end, shard in cumulative:
            if global_index < end:
                selected_shard = shard
                break
            previous_end = end
        if selected_shard is None:
            continue
        local_index = global_index - previous_end
        shard_path = str(selected_shard["path"])
        metadata_path = str(selected_shard["metadata_path"])
        if shard_path not in shard_cache:
            shard_cache[shard_path] = np.load(shard_path, mmap_mode=None)
        if metadata_path not in metadata_cache:
            metadata_cache[metadata_path] = [
                json.loads(line)
                for line in Path(metadata_path).read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        row = shard_cache[shard_path][local_index].astype("int32").tolist()
        ids = [int(token) for token in row if int(token) != pad_token_id]
        metadata = metadata_cache[metadata_path][local_index]
        samples.append(
            {
                "global_index": global_index,
                "shard": shard_path,
                "row": local_index,
                "nonpad_tokens": len(ids),
                "source_names": metadata.get("source_names", []),
                "contributors": metadata.get("contributors", []),
                "text": tokenizer.decode(ids),
            }
        )
    return samples


def _write_packed_samples(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = ["Decoded packed sequence samples", "===============================", ""]
    for number, row in enumerate(rows, start=1):
        lines.extend(
            [
                f"## sequence {number}",
                f"global_index: {row.get('global_index')}",
                f"shard: {row.get('shard')}",
                f"row: {row.get('row')}",
                f"nonpad_tokens: {row.get('nonpad_tokens')}",
                f"source_names: {json.dumps(row.get('source_names', []), sort_keys=True)}",
                f"contributors: {json.dumps(row.get('contributors', []), sort_keys=True)}",
                "decoded_excerpt:",
                _excerpt(str(row.get("text", "")), limit=1400),
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def _scan_hidden_patterns(text: str, counters: Counter[str], examples: dict[str, list[dict[str, Any]]], row: dict[str, Any]) -> None:
    lower = text.lower()
    for label, patterns in HIDDEN_PATTERNS.items():
        hits = [pattern for pattern in patterns if pattern.lower() in lower]
        if not hits:
            continue
        counters[label] += 1
        if len(examples[label]) < 5:
            examples[label].append({**row, "matched_terms": hits[:8]})


def _format_hidden_scan(
    accepted_counts: Counter[str],
    accepted_examples: dict[str, list[dict[str, Any]]],
    metadata_counts: Counter[str],
    packed_counts: Counter[str],
    packed_examples: dict[str, list[dict[str, Any]]],
) -> str:
    lines = ["Hidden bad-pattern scan", "=======================", ""]
    lines.append("Accepted-document scan counts:")
    for label in sorted(HIDDEN_PATTERNS):
        lines.append(f"- {label}: {accepted_counts.get(label, 0)} accepted docs")
    lines.append("")
    lines.append("Prepared metadata scan counts:")
    for label in sorted(HIDDEN_PATTERNS):
        lines.append(f"- {label}: {metadata_counts.get(label, 0)} metadata rows")
    lines.append("")
    lines.append("Decoded packed-sequence sample scan counts:")
    for label in sorted(HIDDEN_PATTERNS):
        lines.append(f"- {label}: {packed_counts.get(label, 0)} sampled sequences")
    lines.append("")
    lines.append("Accepted-document examples:")
    for label in sorted(accepted_examples):
        lines.append(f"\n## {label}")
        for row in accepted_examples[label]:
            lines.append(
                f"source={row.get('source')} record_index={row.get('record_index')} "
                f"document_id={row.get('document_id')} matched_terms={row.get('matched_terms')}"
            )
            lines.append(_excerpt(str(row.get("text", "")), limit=700))
    lines.append("")
    lines.append("Decoded packed-sequence examples:")
    for label in sorted(packed_examples):
        lines.append(f"\n## {label}")
        for row in packed_examples[label]:
            lines.append(
                f"global_index={row.get('global_index')} source_names={row.get('source_names')} "
                f"matched_terms={row.get('matched_terms')}"
            )
            lines.append(_excerpt(str(row.get("text", "")), limit=700))
    lines.append("")
    return "\n".join(lines)


def _write_repeated_samples(
    path: Path,
    rows: list[dict[str, Any]],
    metadata_contributor_counts: Counter[tuple[str, str]],
) -> None:
    lines = ["Repeated accepted document samples", "==================================", ""]
    for number, row in enumerate(rows, start=1):
        source = str(row.get("source", ""))
        document_id = str(row.get("document_id", ""))
        metadata_count = metadata_contributor_counts.get((source, document_id), 0)
        lines.extend(
            [
                f"## repeated document {number}",
                f"source: {source}",
                f"record_index: {row.get('record_index')}",
                f"document_id: {document_id}",
                "document_stream_appearances: 2",
                "repeat_type: exact accepted-document repeat from bounded source restart",
                f"packed_metadata_contributor_appearances: {metadata_count}",
                f"diagnostics: {_brief_diagnostics(row.get('diagnostics', {}))}",
                "excerpt:",
                _excerpt(str(row.get("text", "")), limit=1200),
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_ngram_contexts(path: Path, contexts: dict[tuple[str, str], list[dict[str, Any]]], top_rows: list[tuple[str, dict[str, Any]]]) -> None:
    lines = ["Top repeated n-gram contexts", "============================", ""]
    for label, row in top_rows:
        phrase = str(row.get("phrase", ""))
        count = int(row.get("count", 0))
        lines.extend([f"## {label}: {phrase}", f"manifest_count: {count}", ""])
        rows = contexts.get((label, phrase), [])
        if not rows:
            lines.append("No surrounding context found in reconstructed accepted documents.")
            lines.append("")
            continue
        for number, context_row in enumerate(rows, start=1):
            lines.extend(
                [
                    f"### context {number}",
                    f"source: {context_row.get('source')}",
                    f"record_index: {context_row.get('record_index')}",
                    f"document_id: {context_row.get('document_id')}",
                    "excerpt:",
                    context_row.get("context", ""),
                    "",
                ]
            )
    path.write_text("\n".join(lines), encoding="utf-8")


def _source_config_by_name(config: DataConfig) -> dict[str, DataSourceConfig]:
    return {source.name: source for source in config.pretrain_sources}


def _iter_source_lines(source: DataSourceConfig):
    if source.paths:
        paths = [Path(path) for path in source.paths]
    elif source.path:
        paths = [Path(source.path)]
    else:
        return
    consumed = 0
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                consumed += 1
                if consumed <= int(source.skip_records):
                    continue
                if source.max_records is not None and consumed > int(source.skip_records) + int(source.max_records):
                    return
                yield consumed, line.rstrip("\n")


def _build_report(
    *,
    manifest_path: Path,
    manifest: dict[str, Any],
    data_config_path: Path,
    config: DataConfig,
    output_dir: Path,
    accepted_by_source: dict[str, list[dict[str, Any]]],
    repeated_by_source: dict[str, list[dict[str, Any]]],
    rejected_reasons_by_source: dict[str, Counter[str]],
    accepted_lengths_by_source: dict[str, list[int]],
    rejected_lengths_by_source: dict[str, list[int]],
    packed_samples: list[dict[str, Any]],
    metadata_hidden_counts: Counter[str],
    accepted_hidden_counts: Counter[str],
    packed_hidden_counts: Counter[str],
) -> str:
    diagnostics = manifest.get("diagnostics", {})
    per_source = _source_reports(manifest)
    total_tokens = int(manifest.get("num_tokens", 0))
    unique_docs = sum(
        int(report.get("kept_documents", 0)) - int(report.get("repeated_documents", 0))
        for report in per_source.values()
    )
    repeated_docs = sum(int(report.get("repeated_documents", 0)) for report in per_source.values())
    source_lines = []
    for source, report in sorted(per_source.items()):
        source_lines.append(
            f"- `{source}`: tokens={int(report.get('kept_tokens', 0)):,}, "
            f"share={float(report.get('token_share', 0.0)):.6f}, "
            f"unique_docs={int(report.get('kept_documents', 0)) - int(report.get('repeated_documents', 0)):,}, "
            f"repeated_docs={int(report.get('repeated_documents', 0)):,}, "
            f"repeat_rate={float(report.get('repeat_rate', 0.0)):.6f}"
        )
    reject_lines = []
    for source, counter in sorted(rejected_reasons_by_source.items()):
        reject_lines.append(f"### {source}")
        for reason, count in counter.most_common(12):
            reject_lines.append(f"- {reason}: {count:,}")
    accepted_stats = []
    for source in sorted(accepted_lengths_by_source):
        accepted_stats.append(
            f"- `{source}` accepted words: avg={_safe_mean(accepted_lengths_by_source[source])}, "
            f"median={_safe_median(accepted_lengths_by_source[source])}"
        )
    rejected_stats = []
    for source in sorted(rejected_lengths_by_source):
        rejected_stats.append(
            f"- `{source}` rejected words: avg={_safe_mean(rejected_lengths_by_source[source])}, "
            f"median={_safe_median(rejected_lengths_by_source[source])}"
        )

    repeat_policy = float(config.lm_max_source_repeat_rate)
    if repeated_docs <= 0 or repeat_policy <= 0.0:
        repeat_policy_lines = [
            f"`lm_max_source_repeat_rate` is currently `{repeat_policy}`.",
            "",
            "No controlled source restarts contributed repeated accepted documents to this manifest. Near-duplicate counts therefore reflect natural web duplication or source overlap, not deliberate repeat exposure.",
        ]
    else:
        repeat_policy_lines = [
            f"`lm_max_source_repeat_rate` is currently `{repeat_policy}`. Whole accepted source documents may be re-yielded by bounded source restarts only after passing the same filters and exact-dedupe pass.",
            "",
            "What is repeated: whole accepted source documents after filtering. Packed sequences are a downstream representation and are not directly selected for repetition. The repeated document is tokenized and packed again, so its content can contribute to additional packed windows.",
            "",
            "`repeat_rate` is `repeated_documents / kept_documents` per source. The unique document count is `kept_documents - repeated_documents`.",
            "",
            "Risk: repeated exposure can increase memorization or semantic-loop attraction if repeated documents contain boilerplate. The main mitigation is that the repeated pool is still filtered real prose, repeated n-grams are reported, and sanity probes should watch loop rate and copied boilerplate.",
        ]
    lines = [
        "# Curated Real-Data Pretrain Audit",
        "",
        f"manifest: `{manifest_path}`",
        f"data config: `{data_config_path}`",
        f"output dir: `{output_dir}`",
        "",
        "## Manifest summary",
        "",
        f"- tokens: {total_tokens:,}",
        f"- sequences: {int(manifest.get('num_sequences', 0)):,}",
        f"- unique accepted docs: {unique_docs:,}",
        f"- controlled repeated docs: {repeated_docs:,}",
        f"- max repeat rate: {diagnostics.get('max_repeat_rate')}",
        f"- corpus_quality_gate: {diagnostics.get('corpus_quality_gate', {}).get('passed')}",
        f"- broad_source_quality_gate: {diagnostics.get('broad_source_quality_gate', {}).get('passed')}",
        f"- domain gate mode: {diagnostics.get('domain_realization_gate', {}).get('mode')}",
        "",
        "## Source mix",
        "",
        *source_lines,
        "",
        "## Curated_lm selection process",
        "",
        "Filtering is applied before tokenization and packing. Each raw line from the configured real-text source is normalized, optionally scrubbed, filtered, exact-deduplicated by stable text hash, then tokenized. Only tokenized accepted documents enter the packed LM sequence builder.",
        "",
        "Filter order:",
        "",
        "1. Global text checks: min/max character count, alphabetic-character ratio, and extreme `http` count.",
        "2. Broad web checks reused by `curated_lm`: URL-heavy pages, repeated broad boilerplate phrases, line-shape checks for tables/lists/metadata/fragments/repeated lines, and dense metadata terms.",
        "3. Curated prose shape checks: minimum word count, minimum sentence count, fragmentary average sentence length, and very long sentence/run-on guard.",
        "4. Narrow residual boilerplate checks: FamilySearch edit headers, HTML/archive headers, science-fair encyclopedia residue, video/cookie widgets, repeated newsletter/source taglines, known SEO fragments, and question-list pages.",
        "5. Separator, URL/metadata, and menu-shape checks: dense pipes, separators, URLs, metadata residue, and pipe-separated category chains.",
        "6. Topic/artifact density checks: medical/body terms, product/commercial terms, page boilerplate, page-instruction boilerplate, and dictionary/encyclopedia fragments.",
        "7. Repetition check: 5-gram and 8-gram maximum counts and repeat-rate thresholds.",
        "8. Positive prose signal check: sentence count, normal sentence length, discourse markers, punctuation, and lexical variety must reach a small minimum score.",
        "",
        "Rejected documents are not tokenized or packed. Accepted documents can still produce multiple packed rows if long, and multiple short accepted documents can share one packed sequence.",
        "",
        "## Bounded repeat policy",
        "",
        *repeat_policy_lines,
        "",
        "## Accepted vs rejected quality",
        "",
        *accepted_stats,
        *rejected_stats,
        "",
        "Top rejection reasons by source:",
        "",
        *reject_lines,
        "",
        "Top repeated n-grams after filtering are listed with context in `top_repeated_ngram_contexts.txt`. Quality artifact densities and packed sequence samples are in this directory.",
        "",
        "## Hidden pattern scan summary",
        "",
        "Accepted-document counts:",
    ]
    for label in sorted(HIDDEN_PATTERNS):
        lines.append(f"- {label}: {accepted_hidden_counts.get(label, 0)}")
    lines.extend(["", "Prepared metadata counts:"])
    for label in sorted(HIDDEN_PATTERNS):
        lines.append(f"- {label}: {metadata_hidden_counts.get(label, 0)}")
    lines.extend(["", "Decoded packed sample counts:"])
    for label in sorted(HIDDEN_PATTERNS):
        lines.append(f"- {label}: {packed_hidden_counts.get(label, 0)}")
    lines.extend(
        [
            "",
            "## Files",
            "",
            "- `accepted_samples_by_source/*.txt`",
            "- `accepted_fineweb_edu_samples.txt`",
            "- `accepted_fineweb_extension_samples.txt`",
            "- `accepted_local_samples.txt`",
            "- `rejected_samples_by_reason/*.txt`",
            "- `repeated_doc_samples.txt`",
            "- `packed_sequence_samples.txt`",
            "- `top_repeated_ngram_contexts.txt`",
            "- `hidden_pattern_scan.txt`",
            "- `quality_summary.json`",
            "",
        ]
    )
    return "\n".join(lines)


def run_audit(args: argparse.Namespace) -> None:
    data_config_path = Path(args.data_config)
    manifest_path = Path(args.manifest)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rejected_dir = output_dir / "rejected_samples_by_reason"
    rejected_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(data_config_path, DataConfig)
    manifest = _load_json(manifest_path)
    source_reports = _source_reports(manifest)
    tokenizer = SentencePieceTokenizer(config.tokenizer_path)

    top_rows: list[tuple[str, dict[str, Any]]] = []
    diagnostics = manifest.get("diagnostics", {})
    for label, key in (
        ("8gram", "top_repeated_8grams"),
        ("12gram", "top_repeated_12grams"),
        ("20gram", "top_repeated_20grams"),
    ):
        for row in list(diagnostics.get(key) or [])[:5]:
            if isinstance(row, dict) and row.get("phrase"):
                top_rows.append((label, row))

    accepted_by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    repeated_by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    rejected_samples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    rejected_reasons_by_source: dict[str, Counter[str]] = defaultdict(Counter)
    accepted_lengths_by_source: dict[str, list[int]] = defaultdict(list)
    rejected_lengths_by_source: dict[str, list[int]] = defaultdict(list)
    accepted_hidden_counts: Counter[str] = Counter()
    accepted_hidden_examples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    ngram_contexts: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    seen_hashes: set[str] = set()
    source_names = [source.name for source in config.pretrain_sources]

    for source in config.pretrain_sources:
        source_report = source_reports.get(source.name, {})
        repeat_target = int(source_report.get("repeated_documents", 0))
        accepted_target = max(
            int(source_report.get("kept_documents", 0)),
            int(source_report.get("unique_documents", 0)),
        )
        accepted_count = 0
        for record_index, raw_text in _iter_source_lines(source):
            if accepted_target > 0 and accepted_count >= accepted_target:
                break
            if not raw_text.strip():
                continue
            normalized_text = normalize_whitespace(raw_text)
            diagnostics_for_doc = _doc_diagnostics(normalized_text, raw_text)
            clean_result = clean_document(
                DocumentRecord(text=raw_text, source=source.name),
                config,
                source,
                seen_hashes,
            )
            if clean_result.record is None:
                reason = clean_result.dropped_reason or "dropped"
                rejected_reasons_by_source[source.name][reason] += 1
                rejected_lengths_by_source[source.name].append(int(diagnostics_for_doc.get("words", 0)))
                if reason in MAJOR_REJECTION_REASONS and len(rejected_samples[reason]) < 10:
                    rejected_samples[reason].append(
                        {
                            "source": source.name,
                            "record_index": record_index,
                            "reason": reason,
                            "diagnostics": diagnostics_for_doc,
                            "text": normalized_text,
                        }
                    )
                continue

            accepted_count += 1
            text = clean_result.record.text
            diagnostics_for_doc = _doc_diagnostics(text, raw_text)
            document_id = clean_result.record.document_id or ""
            row = {
                "source": source.name,
                "record_index": record_index,
                "document_id": document_id,
                "diagnostics": diagnostics_for_doc,
                "text": text,
            }
            accepted_by_source[source.name].append(row)
            accepted_lengths_by_source[source.name].append(int(diagnostics_for_doc.get("words", 0)))
            if accepted_count <= repeat_target:
                repeated_by_source[source.name].append(row)
            _scan_hidden_patterns(text, accepted_hidden_counts, accepted_hidden_examples, row)

            lower_text = text.lower()
            for label, top_row in top_rows:
                phrase = str(top_row.get("phrase", ""))
                key = (label, phrase)
                if len(ngram_contexts[key]) >= 3:
                    continue
                index = lower_text.find(phrase.lower())
                if index == -1:
                    continue
                ngram_contexts[key].append(
                    {
                        "source": source.name,
                        "record_index": record_index,
                        "document_id": document_id,
                        "context": _context(text, index),
                    }
                )

    metadata_contributor_counts, metadata_hidden_counts = _metadata_counts(manifest, HIDDEN_PATTERNS)
    packed_samples = _packed_sequence_samples(manifest, tokenizer, limit=30)
    _add_packed_sample_fallbacks(accepted_by_source, source_names, packed_samples)
    packed_hidden_counts: Counter[str] = Counter()
    packed_hidden_examples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in packed_samples:
        _scan_hidden_patterns(str(row.get("text", "")), packed_hidden_counts, packed_hidden_examples, row)

    _write_accepted_sample_files(output_dir, accepted_by_source, tokenizer, source_names)
    _write_rejected_samples(output_dir, rejected_samples)

    repeated_rows: list[dict[str, Any]] = []
    for source_name in sorted(repeated_by_source):
        repeated_rows.extend(_even_sample(repeated_by_source[source_name], 15))
    repeated_rows = _even_sample(repeated_rows, 30)
    _write_repeated_samples(output_dir / "repeated_doc_samples.txt", repeated_rows, metadata_contributor_counts)
    _write_packed_samples(output_dir / "packed_sequence_samples.txt", packed_samples)
    _write_ngram_contexts(output_dir / "top_repeated_ngram_contexts.txt", ngram_contexts, top_rows)
    (output_dir / "hidden_pattern_scan.txt").write_text(
        _format_hidden_scan(
            accepted_hidden_counts,
            accepted_hidden_examples,
            metadata_hidden_counts,
            packed_hidden_counts,
            packed_hidden_examples,
        ),
        encoding="utf-8",
    )

    summary_payload = {
        "manifest": str(manifest_path),
        "data_config": str(data_config_path),
        "num_tokens": int(manifest.get("num_tokens", 0)),
        "num_sequences": int(manifest.get("num_sequences", 0)),
        "source_reports": source_reports,
        "accepted_document_stats": {
            source: {
                "count": len(lengths),
                "avg_words": _safe_mean(lengths),
                "median_words": _safe_median(lengths),
            }
            for source, lengths in accepted_lengths_by_source.items()
        },
        "rejected_document_stats": {
            source: {
                "count": len(lengths),
                "avg_words": _safe_mean(lengths),
                "median_words": _safe_median(lengths),
                "top_rejection_reasons": rejected_reasons_by_source[source].most_common(20),
            }
            for source, lengths in rejected_lengths_by_source.items()
        },
        "hidden_pattern_counts": {
            "accepted_documents": dict(sorted(accepted_hidden_counts.items())),
            "prepared_metadata_rows": dict(sorted(metadata_hidden_counts.items())),
            "packed_sequence_samples": dict(sorted(packed_hidden_counts.items())),
        },
        "top_repeated_8grams": diagnostics.get("top_repeated_8grams", [])[:10],
        "top_repeated_12grams": diagnostics.get("top_repeated_12grams", [])[:10],
        "top_repeated_20grams": diagnostics.get("top_repeated_20grams", [])[:10],
        "quality_artifact_densities": {
            key: diagnostics.get("broad_source_quality_gate", {}).get(f"max_{key}")
            for key in (
                "medical_body_density",
                "navigation_text_density",
                "malformed_fragment_density",
                "generic_article_formula_density",
                "product_commercial_density",
                "dictionary_fragment_density",
                "page_boilerplate_density",
            )
        },
    }
    (output_dir / "quality_summary.json").write_text(
        json.dumps(summary_payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    report = _build_report(
        manifest_path=manifest_path,
        manifest=manifest,
        data_config_path=data_config_path,
        config=config,
        output_dir=output_dir,
        accepted_by_source=accepted_by_source,
        repeated_by_source=repeated_by_source,
        rejected_reasons_by_source=rejected_reasons_by_source,
        accepted_lengths_by_source=accepted_lengths_by_source,
        rejected_lengths_by_source=rejected_lengths_by_source,
        packed_samples=packed_samples,
        metadata_hidden_counts=metadata_hidden_counts,
        accepted_hidden_counts=accepted_hidden_counts,
        packed_hidden_counts=packed_hidden_counts,
    )
    (output_dir / "audit_report.md").write_text(report, encoding="utf-8")

    print(f"wrote curated pretrain data audit to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit curated local-MVP pretrain data selection.")
    parser.add_argument(
        "--data-config",
        default="sample-configs/data-local-mvp.json",
    )
    parser.add_argument(
        "--manifest",
        default="artifacts/runs/local-mvp/prepared/pretrain.json",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/runs/local-mvp/reports/data_audit",
    )
    run_audit(parser.parse_args())


if __name__ == "__main__":
    main()
