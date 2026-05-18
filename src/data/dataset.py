from __future__ import annotations

import hashlib
import json
import math
import random
import re
import sys
import time
import traceback
from collections import Counter
from collections import deque
from concurrent.futures import Future, ProcessPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

from config import DataConfig, DataSourceConfig
from data.packing import PackedSequencePacker, pack_token_sequences
from data.prepared import (
    PreparedPackedDataset,
    PreparedPreferenceDataset,
    PreparedSFTDataset,
    append_hash_chunk,
    build_input_fingerprint,
    cleanup_prepare_outputs,
    encode_preference_example,
    encode_sft_messages,
    load_prepared_manifest,
    load_buffer_rows,
    load_metadata_rows,
    load_seen_hashes,
    prepared_resume_dir,
    prepared_resume_state_path,
    remove_resume_artifacts,
    save_buffer_rows,
    save_metadata_rows,
    save_prepared_manifest,
    save_resume_state,
    stage_has_partial_outputs,
    validate_prepared_manifest_artifacts,
    validate_resume_state_files,
)
from data.preprocess import BROAD_LM_GENERIC_ARTICLE_FORMULAE
from data.preprocess import BROAD_LM_MALFORMED_FRAGMENT_TERMS
from data.preprocess import BROAD_LM_MEDICAL_BODY_TERMS
from data.preprocess import BROAD_LM_NAVIGATION_TERMS
from data.preprocess import BROAD_LM_DICTIONARY_FRAGMENT_TERMS
from data.preprocess import BROAD_LM_PAGE_BOILERPLATE_TERMS
from data.preprocess import BROAD_LM_PAGE_INSTRUCTION_PHRASES
from data.preprocess import BROAD_LM_PRODUCT_COMMERCIAL_TERMS
from data.preprocess import DOMAIN_LM_SYNTHETIC_SCAFFOLD_PHRASES
from data.preprocess import clean_document
from data.schemas import DocumentRecord, PreferenceExample, SFTExample
from tokenizer import SentencePieceTokenizer


PREPARE_DOC_SNAPSHOT_INTERVAL = 1_000
PREPARE_EXAMPLE_SNAPSHOT_INTERVAL = 100
STANDARD_SFT_METADATA_FIELDS = ("behavior_bucket", "quality_tier")
STANDARD_PREFERENCE_METADATA_FIELDS = ("chosen_quality_tier", "negative_type")
LOCAL_MVP_PRETRAIN_DOMAIN_FAMILIES = {
    "advising_planning_prose",
    "catalog_grounding_prose",
    "handbook_policy_prose",
    "webb_domain_seed_prose",
}
LOCAL_MVP_PRETRAIN_DOMAIN_SOURCE_NAMES = {
    "advising_seed",
    "catalog_seed",
    "education_seed",
    "local_mvp_domain_seed",
    "philosophy_seed",
    "webb_domain_seed_mix",
    "webb_public_seed",
}
LOCAL_MVP_PRETRAIN_MIN_DOMAIN_TOKEN_SHARE = 0.05
LOCAL_MVP_PRETRAIN_MIN_DOMAIN_TOKENS = 5_000_000
LOCAL_MVP_PRETRAIN_MIN_DOMAIN_REALIZATION_RATIO = 0.5
LOCAL_MVP_PRETRAIN_MAX_GENERIC_SOURCE_SHARE = 0.50
LOCAL_MVP_PRETRAIN_REPEATED_PHRASE_WARN_COUNT = 100
LOCAL_MVP_PRETRAIN_MAX_REPEATED_4GRAM_COUNT = 8_000
LOCAL_MVP_PRETRAIN_MAX_REPEATED_8GRAM_COUNT = 4_000
LOCAL_MVP_PRETRAIN_MAX_REPEATED_12GRAM_COUNT = 2_500
LOCAL_MVP_PRETRAIN_MAX_DOMAIN_REPEATED_20GRAM_COUNT = 500
LOCAL_MVP_PRETRAIN_MAX_NEAR_DUPLICATE_RATIO = 0.12
LOCAL_MVP_PRETRAIN_MAX_TEMPLATE_FAMILY_DOMINANCE_SHARE = 0.20
LOCAL_MVP_PRETRAIN_TEMPLATE_FAMILY_DOMINANCE_MIN_DOCUMENTS = 16
LOCAL_MVP_PRETRAIN_BROAD_SYNTHETIC_META_ALLOWED_PHRASES = {"the model", "in scenario"}
LOCAL_MVP_PRETRAIN_NEAR_DUPLICATE_GATED_SOURCE_NAMES = {
    "catalog_domain_template_fixture",
    "domain_lm_large_fixture",
}
# The SFT planner trims overrepresented buckets globally after candidate collection.
# Targets are row-share goals first; label-token shares are diagnostics unless a bucket's
# token share drifts badly enough to surface in review.
SFT_BUCKET_TARGETS: dict[str, dict[str, float]] = {
    "constructive_direct": {
        "target_row_share": 0.68,
        "target_label_token_share": 0.78,
        "min_examples": 96,
    },
    "clarifying_question": {
        "target_row_share": 0.20,
        "target_label_token_share": 0.15,
        "min_examples": 48,
    },
    "informative_abstention": {
        "target_row_share": 0.08,
        "target_label_token_share": 0.05,
        "min_examples": 16,
    },
    "hard_refusal": {
        "target_row_share": 0.04,
        "target_label_token_share": 0.02,
        "min_examples": 8,
    },
}
SFT_BALANCING_WARMUP_EXAMPLES = 40
PROMOTABLE_DPO_CHOSEN_QUALITY_TIERS = {
    "human_curated",
    "human_edited_from_model",
    "approved_template",
}
PROMOTABLE_DPO_NEGATIVE_TYPES = {
    "hallucinated_specifics",
    "shallow_refusal",
    "vague_filler",
    "fake_citation",
    "repetitive_template_collapse",
    "overconfident_ungrounded",
    "generic_greeting",
    "missing_verification",
}
BROAD_LM_PAGE_ARTIFACT_TERMS = (
    *BROAD_LM_PAGE_BOILERPLATE_TERMS,
    *BROAD_LM_PAGE_INSTRUCTION_PHRASES,
)
LM_JUNK_PATTERNS: dict[str, re.Pattern[str]] = {
    "excessive_hyphen_fragments": re.compile(
        r"(?:\b[A-Za-z]{1,6}-){2,}[A-Za-z]{1,10}\b|-[A-Za-z]{1,4}-|"
        rf"\b(?:{'|'.join(re.escape(term) for term in BROAD_LM_MALFORMED_FRAGMENT_TERMS)})\b",
        re.IGNORECASE,
    ),
    "broken_quote_fragments": re.compile(r"(?:[\"'“”‘’]\s*){3,}|(?:\b[a-zA-Z]\s+[\"'“”‘’]\s*){2,}"),
    "navigation_like_text": re.compile(
        rf"\b(?:{'|'.join(re.escape(term) for term in BROAD_LM_NAVIGATION_TERMS)})\b",
        re.IGNORECASE,
    ),
    "generic_article_formula": re.compile(
        rf"\b(?:{'|'.join(re.escape(term) for term in BROAD_LM_GENERIC_ARTICLE_FORMULAE)})\b",
        re.IGNORECASE,
    ),
    "medical_body_health": re.compile(
        rf"\b(?:{'|'.join(re.escape(term) for term in BROAD_LM_MEDICAL_BODY_TERMS)})\b",
        re.IGNORECASE,
    ),
    "product_commercial": re.compile(
        rf"\b(?:{'|'.join(re.escape(term) for term in BROAD_LM_PRODUCT_COMMERCIAL_TERMS)})\b",
        re.IGNORECASE,
    ),
    "dictionary_fragment": re.compile(
        rf"\b(?:{'|'.join(re.escape(term) for term in BROAD_LM_DICTIONARY_FRAGMENT_TERMS)})\b",
        re.IGNORECASE,
    ),
    "page_boilerplate": re.compile(
        rf"\b(?:{'|'.join(re.escape(term) for term in BROAD_LM_PAGE_ARTIFACT_TERMS)})\b",
        re.IGNORECASE,
    ),
}
PARAGRAPH_SPLIT_RE = re.compile(r"\n\s*\n+")
NON_WORD_RE = re.compile(r"[^a-z\s]+")
SENTENCE_BOUNDARY_RE = re.compile(r"[.!?]+(?:\s+|$)")
LM_DOCUMENT_WORKER_CHUNK_SIZE = 16
LM_DOCUMENT_WORKER_PENDING_MULTIPLIER = 2


@dataclass(slots=True)
class LMDocumentProcessResult:
    record_index: int
    is_text: bool
    text: str = ""
    document_id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    dropped_reason: str | None = None
    token_ids: list[int] = field(default_factory=list)
    synthetic_meta_phrase_counts: dict[str, int] = field(default_factory=dict)
    audit_delta: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    traceback: str | None = None


_LM_DOCUMENT_WORKER_CONFIG: DataConfig | None = None
_LM_DOCUMENT_WORKER_SOURCE: DataSourceConfig | None = None
_LM_DOCUMENT_WORKER_TOKENIZER: SentencePieceTokenizer | None = None


def _progress(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def _require_datasets():
    try:
        from datasets import Dataset, IterableDataset, load_dataset  # type: ignore
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "datasets is required for data preparation. Install with `pip install datasets`."
        ) from exc
    return Dataset, IterableDataset, load_dataset


def _require_torch():
    try:
        import torch
        from torch.utils.data import ConcatDataset
    except ModuleNotFoundError as exc:
        raise RuntimeError("torch is required for dataset objects. Install with `pip install torch`.") from exc
    return torch, ConcatDataset


def _data_files_for_source(source: DataSourceConfig):
    if source.paths:
        return source.paths
    if source.path:
        return source.path
    raise ValueError(f"Source {source.name!r} does not define a local path or shard list.")


def _apply_record_window(dataset, source: DataSourceConfig):
    dataset_cls, iterable_cls, _ = _require_datasets()
    if isinstance(dataset, iterable_cls):
        if source.skip_records > 0:
            dataset = dataset.skip(source.skip_records)
        if source.max_records is not None:
            dataset = dataset.take(source.max_records)
        return dataset

    if isinstance(dataset, dataset_cls):
        start = min(source.skip_records, len(dataset))
        stop = len(dataset) if source.max_records is None else min(start + source.max_records, len(dataset))
        return dataset.select(range(start, stop))

    return dataset


def _coerce_prompt_messages(prompt: Any) -> list[dict[str, str]] | None:
    if isinstance(prompt, list):
        return prompt
    if isinstance(prompt, str) and prompt.strip():
        return [{"role": "user", "content": prompt.strip()}]
    return None


def _normalize_text(value: str) -> str:
    return " ".join(value.split()).strip()


def _stable_payload_hash(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _normalize_messages(
    messages: list[dict[str, str]],
    *,
    include_assistant: bool,
) -> list[dict[str, str]]:
    normalized: list[dict[str, str]] = []
    for raw_message in messages:
        role = str(raw_message.get("role", "")).strip()
        if not include_assistant and role == "assistant":
            continue
        content = raw_message.get("content", "")
        if not isinstance(content, str):
            content = str(content)
        normalized.append({"role": role, "content": _normalize_text(content)})
    return normalized


def _prompt_signature_text(messages: list[dict[str, str]]) -> str:
    return json.dumps(_normalize_messages(messages, include_assistant=False), sort_keys=True, separators=(",", ":"))


def _prompt_signature_hash(messages: list[dict[str, str]]) -> str:
    return _stable_payload_hash(_normalize_messages(messages, include_assistant=False))


def _coerce_optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = _normalize_text(str(value))
    return text or None


def _collect_standard_metadata(
    item: dict[str, Any],
    *,
    metadata_fields: list[str],
    extra_fields: tuple[str, ...],
) -> dict[str, Any]:
    metadata = {field: item.get(field) for field in metadata_fields}
    for field in extra_fields:
        if field not in metadata and field in item:
            metadata[field] = item.get(field)
    return metadata


def _counter_to_sorted_dict(counter: Counter[str]) -> dict[str, int]:
    return {key: int(counter[key]) for key in sorted(counter)}


def _share_dict(counter: Counter[str], total: int) -> dict[str, float]:
    if total <= 0:
        return {}
    return {key: round(counter[key] / total, 6) for key in sorted(counter)}


def _sft_bucket_targets(bucket: str) -> dict[str, float]:
    return SFT_BUCKET_TARGETS.get(bucket, SFT_BUCKET_TARGETS["constructive_direct"])


def _sft_target_row_share(bucket: str) -> float:
    return float(_sft_bucket_targets(bucket).get("target_row_share", 0.0))


def _sft_target_label_token_share(bucket: str) -> float:
    return float(_sft_bucket_targets(bucket).get("target_label_token_share", 0.0))


def _sft_min_examples(bucket: str) -> int:
    return int(_sft_bucket_targets(bucket).get("min_examples", 0))


def _coerce_behavior_bucket(metadata: dict[str, Any]) -> str:
    bucket = _coerce_optional_text(metadata.get("behavior_bucket"))
    return bucket or "unspecified"


def _coerce_quality_tier(metadata: dict[str, Any], *, default: str = "unspecified") -> str:
    tier = _coerce_optional_text(metadata.get("quality_tier"))
    return tier or default


def _assistant_response_text(messages: list[dict[str, str]]) -> str:
    for message in reversed(messages):
        if str(message.get("role", "")).strip() == "assistant":
            content = message.get("content", "")
            if isinstance(content, str):
                return _normalize_text(content)
            return _normalize_text(str(content))
    return ""


def _normalize_behavior_bucket(value: str | None) -> str | None:
    normalized = _coerce_optional_text(value)
    if normalized is None:
        return None
    alias_map = {
        "constructive": "constructive_direct",
        "constructive_direct": "constructive_direct",
        "direct_answer": "constructive_direct",
        "informative_abstention": "informative_abstention",
        "abstention": "informative_abstention",
        "clarifying_question": "clarifying_question",
        "clarifying": "clarifying_question",
        "comparison_question": "clarifying_question",
        "hard_refusal": "hard_refusal",
        "refusal": "hard_refusal",
    }
    return alias_map.get(normalized)


def _infer_behavior_bucket(messages: list[dict[str, str]], metadata: dict[str, Any]) -> str:
    explicit_bucket = _normalize_behavior_bucket(metadata.get("behavior_bucket"))
    if explicit_bucket is not None:
        return explicit_bucket
    response = _assistant_response_text(messages).lower()
    if not response:
        return "hard_refusal"
    if any(
        phrase in response
        for phrase in (
            "i can't help with that",
            "i cannot help with that",
            "i won't help with that",
            "i will not help with that",
            "i can't comply",
            "i cannot comply",
        )
    ):
        return "hard_refusal"
    if response.endswith("?") or any(
        phrase in response
        for phrase in (
            "tell me ",
            "share ",
            "which two",
            "what are your goals",
            "what kind of work",
            "what interests you",
        )
    ):
        return "clarifying_question"
    if any(
        phrase in response
        for phrase in (
            "could not verify",
            "couldn't verify",
            "could not find",
            "couldn't find",
            "do not want to guess",
            "don't want to guess",
            "avoid guessing",
            "what is missing",
            "not in the current",
            "check the current catalog",
            "check the latest catalog",
            "ask an advisor",
        )
    ):
        return "informative_abstention"
    return "constructive_direct"


def _coerce_chosen_quality_tier(metadata: dict[str, Any]) -> str:
    tier = _coerce_optional_text(metadata.get("chosen_quality_tier"))
    return tier or "model_unreviewed"


def _coerce_negative_type(metadata: dict[str, Any]) -> str:
    negative = _coerce_optional_text(metadata.get("negative_type"))
    return negative or "unspecified"


def _source_family(source: DataSourceConfig) -> str:
    family = _coerce_optional_text(source.family)
    return family or source.name


def _token_share(count: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return count / total


def _top_ngram_families(text: str, *, n: int = 4, limit: int = 6) -> list[str]:
    words = [part for part in text.split() if part]
    if len(words) < n:
        return []
    seen: list[str] = []
    seen_set: set[str] = set()
    for index in range(len(words) - n + 1):
        phrase = " ".join(words[index : index + n]).lower()
        if phrase in seen_set:
            continue
        seen.append(phrase)
        seen_set.add(phrase)
        if len(seen) >= limit:
            break
    return seen


def _top_counter_rows(counter: Counter[str], *, limit: int = 10) -> list[dict[str, int | str]]:
    return [
        {"phrase": phrase, "count": int(count)}
        for phrase, count in counter.most_common(limit)
    ]


def _duplicate_count(counter: Counter[str]) -> int:
    return sum(max(int(count) - 1, 0) for count in counter.values())


def _cluster_count(counter: Counter[str]) -> int:
    return sum(1 for count in counter.values() if int(count) > 1)


def _largest_cluster_size(counter: Counter[str]) -> int:
    return max((int(count) for count in counter.values()), default=0)


def _density(count: int, total: int) -> float:
    return round(count / max(total, 1), 6)


def _pattern_occurrence_count(pattern: re.Pattern[str], text: str) -> int:
    return len(pattern.findall(text))


def _paragraphs_for_duplicate_scan(text: str) -> list[str]:
    paragraphs = [
        _normalize_text(part)
        for part in PARAGRAPH_SPLIT_RE.split(text)
        if _normalize_text(part)
    ]
    return paragraphs or [_normalize_text(text)]


def _hash_text(text: str) -> str:
    return hashlib.blake2b(text.encode("utf-8"), digest_size=16).hexdigest()


def _normalized_paragraph_signature(text: str) -> str:
    normalized = text.lower()
    normalized = re.sub(r"\d+", " ", normalized)
    normalized = NON_WORD_RE.sub(" ", normalized)
    normalized = _normalize_text(normalized)
    return _hash_text(normalized) if normalized else ""


def _update_duplicate_counters(audit_state: dict[str, Any], text: str) -> None:
    for paragraph in _paragraphs_for_duplicate_scan(text):
        audit_state["exact_paragraph_counter"][_hash_text(paragraph)] += 1
        normalized_signature = _normalized_paragraph_signature(paragraph)
        if normalized_signature:
            audit_state["normalized_paragraph_counter"][normalized_signature] += 1


def _update_synthetic_meta_phrase_counts(audit_state: dict[str, Any], text: str) -> None:
    lower = text.lower()
    for phrase in DOMAIN_LM_SYNTHETIC_SCAFFOLD_PHRASES:
        count = lower.count(phrase)
        if count:
            audit_state["synthetic_meta_phrase_counter"][phrase] += count


def _document_shape_counts(text: str) -> dict[str, int]:
    paragraphs = _paragraphs_for_duplicate_scan(text)
    words = _normalize_text(text).split()
    sentences = SENTENCE_BOUNDARY_RE.findall(text)
    return {
        "chars": len(text),
        "words": len(words),
        "sentences": len(sentences),
        "paragraphs": len(paragraphs),
    }


def _synthetic_meta_phrase_counts(text: str) -> dict[str, int]:
    lower = text.lower()
    return {
        phrase: int(count)
        for phrase in DOMAIN_LM_SYNTHETIC_SCAFFOLD_PHRASES
        if (count := lower.count(phrase)) > 0
    }


def _lm_document_audit_delta(text: str) -> dict[str, Any]:
    exact_paragraph_counter: Counter[str] = Counter()
    normalized_paragraph_counter: Counter[str] = Counter()
    for paragraph in _paragraphs_for_duplicate_scan(text):
        exact_paragraph_counter[_hash_text(paragraph)] += 1
        normalized_signature = _normalized_paragraph_signature(paragraph)
        if normalized_signature:
            normalized_paragraph_counter[normalized_signature] += 1

    quality_artifact_counter: Counter[str] = Counter()
    quality_artifact_occurrence_counter: Counter[str] = Counter()
    for artifact_name, pattern in LM_JUNK_PATTERNS.items():
        artifact_count = _pattern_occurrence_count(pattern, text)
        if artifact_count > 0:
            quality_artifact_counter[artifact_name] += 1
            quality_artifact_occurrence_counter[artifact_name] += artifact_count

    return {
        "shape_counts": _document_shape_counts(text),
        "quality_diagnostic_token_count": len(text.split()),
        "phrase_counter": _top_ngram_families(text, n=4, limit=8),
        "phrase_counter_8": _top_ngram_families(text, n=8, limit=6),
        "phrase_counter_12": _top_ngram_families(text, n=12, limit=4),
        "phrase_counter_20": _top_ngram_families(text, n=20, limit=3),
        "exact_paragraph_counter": dict(exact_paragraph_counter),
        "normalized_paragraph_counter": dict(normalized_paragraph_counter),
        "quality_artifact_counter": dict(quality_artifact_counter),
        "quality_artifact_occurrence_counter": dict(quality_artifact_occurrence_counter),
    }


def _lm_document_payload_from_item(
    item: dict[str, Any],
    source: DataSourceConfig,
    record_index: int,
) -> dict[str, Any]:
    return {
        "record_index": record_index,
        "text": item.get(source.text_field, ""),
        "metadata": {field: item.get(field) for field in source.metadata_fields},
    }


def _process_lm_document_payload(
    payload: dict[str, Any],
    data_config: DataConfig,
    source_config: DataSourceConfig,
    tokenizer: SentencePieceTokenizer,
) -> LMDocumentProcessResult:
    record_index = int(payload.get("record_index", 0))
    raw_text = payload.get("text", "")
    if not isinstance(raw_text, str):
        return LMDocumentProcessResult(record_index=record_index, is_text=False)

    synthetic_counts = _synthetic_meta_phrase_counts(raw_text)
    record = DocumentRecord(
        text=raw_text,
        source=source_config.name,
        metadata=dict(payload.get("metadata", {})),
    )
    cleaned = clean_document(record, data_config, source_config, None)
    if cleaned.record is None:
        return LMDocumentProcessResult(
            record_index=record_index,
            is_text=True,
            dropped_reason=cleaned.dropped_reason or "dropped",
            synthetic_meta_phrase_counts=synthetic_counts,
        )

    token_ids = tokenizer.encode(cleaned.record.text, add_bos=True, add_eos=True)
    return LMDocumentProcessResult(
        record_index=record_index,
        is_text=True,
        text=cleaned.record.text,
        document_id=cleaned.record.document_id or "",
        metadata=dict(cleaned.record.metadata),
        token_ids=token_ids,
        synthetic_meta_phrase_counts=synthetic_counts,
        audit_delta=_lm_document_audit_delta(cleaned.record.text),
    )


def _init_lm_document_worker(
    data_config_payload: dict[str, Any],
    source_payload: dict[str, Any],
    tokenizer_path: str,
) -> None:
    global _LM_DOCUMENT_WORKER_CONFIG
    global _LM_DOCUMENT_WORKER_SOURCE
    global _LM_DOCUMENT_WORKER_TOKENIZER
    _LM_DOCUMENT_WORKER_CONFIG = DataConfig.from_dict(data_config_payload)
    _LM_DOCUMENT_WORKER_SOURCE = DataSourceConfig.from_dict(source_payload)
    _LM_DOCUMENT_WORKER_TOKENIZER = SentencePieceTokenizer(tokenizer_path)


def _process_lm_document_chunk(payloads: list[dict[str, Any]]) -> list[LMDocumentProcessResult]:
    if (
        _LM_DOCUMENT_WORKER_CONFIG is None
        or _LM_DOCUMENT_WORKER_SOURCE is None
        or _LM_DOCUMENT_WORKER_TOKENIZER is None
    ):
        raise RuntimeError("LM document worker was not initialized.")
    results: list[LMDocumentProcessResult] = []
    for payload in payloads:
        record_index = int(payload.get("record_index", 0))
        try:
            results.append(
                _process_lm_document_payload(
                    payload,
                    _LM_DOCUMENT_WORKER_CONFIG,
                    _LM_DOCUMENT_WORKER_SOURCE,
                    _LM_DOCUMENT_WORKER_TOKENIZER,
                )
            )
        except Exception as exc:  # pragma: no cover - defensive process boundary
            results.append(
                LMDocumentProcessResult(
                    record_index=record_index,
                    is_text=isinstance(payload.get("text", ""), str),
                    error=f"{type(exc).__name__}: {exc}",
                    traceback=traceback.format_exc(),
                )
            )
    return results


def _message_group_id(
    source: DataSourceConfig,
    item: dict[str, Any],
    messages: list[dict[str, str]],
) -> str:
    if source.group_field is not None:
        explicit_group = _coerce_optional_text(item.get(source.group_field))
        if explicit_group is not None:
            return explicit_group
    explicit_group = _coerce_optional_text(item.get("group_id"))
    if explicit_group is not None:
        return explicit_group
    conversation_id = _coerce_optional_text(item.get("conversation_id"))
    if conversation_id is not None:
        return conversation_id
    return _stable_payload_hash(
        {
            "source": source.name,
            "messages": _normalize_messages(messages, include_assistant=False),
        }
    )


def _message_example_id(
    source: DataSourceConfig,
    item: dict[str, Any],
    messages: list[dict[str, str]],
) -> str:
    if source.id_field is not None:
        explicit_id = _coerce_optional_text(item.get(source.id_field))
        if explicit_id is not None:
            return explicit_id
    explicit_id = _coerce_optional_text(item.get("example_id"))
    if explicit_id is not None:
        return explicit_id
    row_id = _coerce_optional_text(item.get("id"))
    if row_id is not None:
        return row_id
    return _stable_payload_hash(
        {
            "source": source.name,
            "messages": _normalize_messages(messages, include_assistant=True),
        }
    )


def _preference_example_id(
    source: DataSourceConfig,
    item: dict[str, Any],
    prompt_messages: list[dict[str, str]],
    chosen: str,
    rejected: str,
) -> str:
    if source.id_field is not None:
        explicit_id = _coerce_optional_text(item.get(source.id_field))
        if explicit_id is not None:
            return explicit_id
    explicit_id = _coerce_optional_text(item.get("example_id"))
    if explicit_id is not None:
        return explicit_id
    row_id = _coerce_optional_text(item.get("id"))
    if row_id is not None:
        return row_id
    return _stable_payload_hash(
        {
            "source": source.name,
            "prompt": _normalize_messages(prompt_messages, include_assistant=False),
            "chosen": _normalize_text(chosen),
            "rejected": _normalize_text(rejected),
        }
    )


def _source_location(source: DataSourceConfig) -> str:
    return source.dataset_name or source.path or ",".join(source.paths)


def _source_with_cursor(source: DataSourceConfig, raw_records_consumed: int) -> DataSourceConfig:
    if raw_records_consumed <= 0:
        return source
    updated = DataSourceConfig.from_dict(source.to_dict())
    updated.skip_records = source.skip_records + raw_records_consumed
    if source.max_records is not None:
        updated.max_records = max(source.max_records - raw_records_consumed, 0)
    return updated


class PackedSequenceDataset:
    def __init__(self, sequences: list[list[int]], pad_token_id: int):
        self.sequences = sequences
        self.pad_token_id = pad_token_id

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, index: int) -> dict[str, Any]:
        torch, _ = _require_torch()
        sequence = torch.tensor(self.sequences[index], dtype=torch.long)
        attention_mask = (sequence != self.pad_token_id).long()
        labels = sequence.clone()
        labels[attention_mask == 0] = -100
        return {"input_ids": sequence, "attention_mask": attention_mask, "labels": labels}


class SFTDataset:
    def __init__(self, examples: list[SFTExample], tokenizer_path: str, sequence_length: int):
        self.examples = examples
        self.tokenizer = SentencePieceTokenizer(tokenizer_path)
        self.sequence_length = sequence_length

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        torch, _ = _require_torch()
        example = self.examples[index]
        input_ids, labels = encode_sft_messages(example.messages, self.tokenizer, self.sequence_length)
        input_tensor = torch.tensor(input_ids, dtype=torch.long)
        labels_tensor = torch.tensor(labels, dtype=torch.long)
        attention_mask = (input_tensor != self.tokenizer.token_to_id("<pad>")).long()
        return {"input_ids": input_tensor, "attention_mask": attention_mask, "labels": labels_tensor}


class PreferenceDataset:
    def __init__(self, examples: list[PreferenceExample], tokenizer_path: str, sequence_length: int):
        self.examples = examples
        self.tokenizer = SentencePieceTokenizer(tokenizer_path)
        self.sequence_length = sequence_length

    def _encode(self, prompt: list[dict[str, str]], answer: str) -> list[int]:
        return encode_preference_example(prompt, answer, self.tokenizer, self.sequence_length)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        torch, _ = _require_torch()
        example = self.examples[index]
        chosen = self._encode(example.prompt, example.chosen)
        rejected = self._encode(example.prompt, example.rejected)
        pad_id = self.tokenizer.token_to_id("<pad>")
        return {
            "chosen_input_ids": torch.tensor(chosen, dtype=torch.long),
            "rejected_input_ids": torch.tensor(rejected, dtype=torch.long),
            "chosen_attention_mask": torch.tensor([token != pad_id for token in chosen], dtype=torch.long),
            "rejected_attention_mask": torch.tensor([token != pad_id for token in rejected], dtype=torch.long),
        }


class IndexedDataset:
    def __init__(self, dataset, indices: list[int]):
        self.dataset = dataset
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int):
        return self.dataset[self.indices[index]]


def _split_indices_by_group(
    items: list[Any],
    *,
    stage_name: str,
    seed: int,
    validation_fraction: float,
    validation_min_examples: int,
    allow_weak_validation: bool,
) -> tuple[list[int], list[int]]:
    total_examples = len(items)
    if validation_fraction <= 0:
        return list(range(total_examples)), []
    if total_examples < 2:
        raise RuntimeError(
            f"WebbGPT: {stage_name} stage needs at least 2 examples before it can reserve validation examples."
        )

    requested_val = round(total_examples * validation_fraction)
    target_val_examples = max(validation_min_examples, requested_val)
    grouped_indices: dict[str, list[int]] = {}
    for index, item in enumerate(items):
        group_id = getattr(item, "split_group_id", None) or getattr(item, "example_id", None) or f"{stage_name}:{index}"
        grouped_indices.setdefault(str(group_id), []).append(index)

    group_ids = list(grouped_indices)
    if len(group_ids) < 2:
        raise RuntimeError(
            f"WebbGPT: {stage_name} stage needs at least two split groups before it can create a held-out validation set."
        )

    random.Random(f"{seed}:{stage_name}:validation").shuffle(group_ids)
    validation_group_ids: list[str] = []
    validation_count = 0
    for group_id in group_ids:
        remaining_groups = len(group_ids) - len(validation_group_ids) - 1
        if validation_count >= target_val_examples and validation_group_ids:
            break
        if remaining_groups < 1:
            break
        validation_group_ids.append(group_id)
        validation_count += len(grouped_indices[group_id])

    validation_index_set = {
        index for group_id in validation_group_ids for index in grouped_indices[group_id]
    }
    train_indices = [index for index in range(total_examples) if index not in validation_index_set]
    validation_indices = [index for index in range(total_examples) if index in validation_index_set]

    if not train_indices or not validation_indices:
        raise RuntimeError(
            f"WebbGPT: {stage_name} stage could not create a non-empty grouped validation split. "
            "Provide explicit validation sources instead."
        )
    if len(validation_indices) < validation_min_examples:
        message = (
            f"WebbGPT: grouped auto-split for {stage_name} would produce only {len(validation_indices)} "
            f"validation examples, below the required minimum of {validation_min_examples}. "
            "Provide explicit validation sources instead."
        )
        if not allow_weak_validation:
            raise RuntimeError(message)
        _progress(f"{message} Continuing only because allow_weak_posttrain_validation=true.")

    _progress(
        f"WebbGPT: split {stage_name} stage into {len(train_indices)} train and {len(validation_indices)} validation examples "
        f"across {len(grouped_indices)} grouped prompts."
    )
    return train_indices, validation_indices


def split_dataset_for_validation(
    dataset,
    *,
    stage_name: str,
    seed: int,
    validation_fraction: float,
    validation_min_examples: int,
    allow_weak_validation: bool = False,
) -> tuple[object, object | None]:
    total_examples = len(dataset)
    if validation_fraction <= 0 or total_examples == 0:
        return dataset, None
    items = getattr(dataset, "examples", None)
    if items is None:
        items = [dataset[index] for index in range(total_examples)]
    train_indices, validation_indices = _split_indices_by_group(
        list(items),
        stage_name=stage_name,
        seed=seed,
        validation_fraction=validation_fraction,
        validation_min_examples=validation_min_examples,
        allow_weak_validation=allow_weak_validation,
    )
    return IndexedDataset(dataset, train_indices), IndexedDataset(dataset, validation_indices)


class DatasetBuilder:
    def __init__(self, config: DataConfig):
        self.config = config

    def _stage_sources(self, stage: str) -> list[DataSourceConfig]:
        mapping = {
            "pretrain": self.config.pretrain_sources,
            "continue": self.config.continued_pretrain_sources,
            "sft": self.config.sft_sources,
            "preference": self.config.preference_sources,
            "validation": self.config.validation_sources,
        }
        return mapping[stage]

    def _require_stage_sources(self, stage: str, sources: list[DataSourceConfig]) -> None:
        if sources:
            return
        stage_help = {
            "pretrain": "pretraining text sources",
            "continue": "continued-pretraining text sources",
            "sft": "SFT chat examples",
            "preference": "preference chosen/rejected examples",
            "validation": "validation text sources",
        }[stage]
        raise RuntimeError(
            f"No {stage_help} are configured in the current data config. "
            f"Populate the `{stage}` stage sources before running this command."
        )

    def _uses_prepared_sources(self, sources: list[DataSourceConfig]) -> bool:
        if not sources:
            return False
        prepared = [source.format == "prepared" for source in sources]
        if any(prepared) and not all(prepared):
            raise RuntimeError("Do not mix prepared-manifest sources with raw sources in the same stage.")
        return all(prepared)

    def _concat_datasets(self, datasets: list):
        if len(datasets) == 1:
            return datasets[0]
        _, concat_cls = _require_torch()
        return concat_cls(datasets)

    def _build_prepared_dataset(self, sources: list[DataSourceConfig], expected_kind: str):
        datasets = []
        for source in sources:
            manifest = validate_prepared_manifest_artifacts(source.path, expected_kind=expected_kind)
            kind = manifest.get("kind")
            if kind == "packed_lm":
                datasets.append(PreparedPackedDataset(source.path))
            elif kind == "sft":
                datasets.append(PreparedSFTDataset(source.path))
            elif kind == "preference":
                datasets.append(PreparedPreferenceDataset(source.path))
            else:
                raise RuntimeError(f"Unsupported prepared dataset kind {kind!r}.")
        return self._concat_datasets(datasets)

    def _initial_source_progress(self, sources: list[DataSourceConfig]) -> list[dict[str, Any]]:
        return [
            {
                "name": source.name,
                "raw_records_consumed": 0,
                "accepted_records": 0,
                "restart_count": 0,
            }
            for source in sources
        ]

    def _resolve_prepare_target(
        self,
        *,
        stage: str,
        kind: str,
        output_path: str,
        input_fingerprint: str,
        force_rebuild: bool,
    ) -> tuple[str, dict[str, Any] | None]:
        manifest_path = Path(output_path)
        resume_state_path = prepared_resume_state_path(manifest_path)
        resume_workspace = prepared_resume_dir(manifest_path)

        if force_rebuild:
            _progress(f"WebbGPT: force rebuilding prepared stage {stage}; clearing prior outputs first.")
            cleanup_prepare_outputs(manifest_path)

        if manifest_path.exists():
            if manifest_path.stat().st_size == 0:
                _progress(
                    f"WebbGPT: found empty prepared-manifest placeholder at {manifest_path}; "
                    "treating it as a fresh target."
                )
                manifest_path.unlink()
            else:
                manifest = load_prepared_manifest(manifest_path)
                manifest_fingerprint = manifest.get("input_fingerprint")
                if manifest_fingerprint is None:
                    raise RuntimeError(
                        f"Existing prepared manifest at {manifest_path} predates resumable metadata and is not safely reusable. "
                        "Re-run with --force-rebuild."
                    )
                if manifest_fingerprint != input_fingerprint:
                    raise RuntimeError(
                        f"Existing prepared manifest at {manifest_path} does not match the current {stage} inputs. "
                        "Re-run with --force-rebuild."
                    )
                if manifest.get("kind") != kind:
                    raise RuntimeError(
                        f"Existing prepared manifest at {manifest_path} has kind {manifest.get('kind')!r}, expected {kind!r}. "
                        "Re-run with --force-rebuild."
                    )
                validate_prepared_manifest_artifacts(manifest_path, expected_kind=kind)
                _progress(f"WebbGPT: reusing completed prepared stage {stage} from {manifest_path}.")
                remove_resume_artifacts(manifest_path)
                return "reuse", manifest

        if resume_state_path.exists():
            state = load_prepared_manifest(resume_state_path)
            if state.get("input_fingerprint") != input_fingerprint:
                raise RuntimeError(
                    f"Prepared-data resume state at {resume_state_path} does not match the current {stage} inputs. "
                    "Re-run with --force-rebuild."
                )
            if state.get("kind") != kind:
                raise RuntimeError(
                    f"Prepared-data resume state at {resume_state_path} has kind {state.get('kind')!r}, expected {kind!r}. "
                    "Re-run with --force-rebuild."
                )
            validate_resume_state_files(state)
            return "resume", state

        if stage_has_partial_outputs(manifest_path) or resume_workspace.exists():
            raise RuntimeError(
                f"Found partial prepared-data outputs for stage {stage!r} at {manifest_path.with_suffix('')} "
                "without resumable metadata. These legacy partial shards are not resumable. "
                "Re-run with --force-rebuild to discard them and rebuild safely."
            )

        return "fresh", None

    def _prepare_packed_stage(
        self,
        stage: str,
        sources: list[DataSourceConfig],
        output_path: str,
        *,
        force_rebuild: bool,
    ) -> dict[str, Any]:
        tokenizer = SentencePieceTokenizer(self.config.tokenizer_path)
        pad_token_id = tokenizer.token_to_id("<pad>")
        eos_token_id = tokenizer.token_to_id("</s>")
        token_budget = None
        if stage == "pretrain":
            token_budget = self.config.pretraining_token_budget
        elif stage == "continue":
            token_budget = self.config.continued_pretraining_token_budget
        source_snapshots = [source.to_dict() for source in sources]
        input_fingerprint = build_input_fingerprint(
            stage=stage,
            kind="packed_lm",
            tokenizer_path=self.config.tokenizer_path,
            sequence_length=self.config.sequence_length,
            rows_per_shard=self.config.prepared_shard_size,
            source_snapshots=source_snapshots,
            token_budget=token_budget,
            extra={
                "pad_token_id": pad_token_id,
                "eos_token_id": eos_token_id,
                "packing_version": "checkpointable-v3-min-3-nonpad",
            },
        )
        action, payload = self._resolve_prepare_target(
            stage=stage,
            kind="packed_lm",
            output_path=output_path,
            input_fingerprint=input_fingerprint,
            force_rebuild=force_rebuild,
        )
        if action == "reuse":
            return payload or {}

        manifest_path = Path(output_path)
        shard_dir = manifest_path.with_suffix("")
        shard_dir.mkdir(parents=True, exist_ok=True)
        resume_workspace = prepared_resume_dir(manifest_path)
        resume_workspace.mkdir(parents=True, exist_ok=True)
        resume_state_path = prepared_resume_state_path(manifest_path)
        rows_buffer_path = resume_workspace / "rows-buffer.npy"
        metadata_buffer_path = resume_workspace / "metadata-buffer.jsonl"
        stage_start_time = time.monotonic()

        if action == "resume":
            state = payload or {}
            source_progress = list(state.get("source_progress", []))
            if len(source_progress) != len(sources):
                raise RuntimeError(
                    f"Prepared-data resume state at {resume_state_path} no longer matches the configured source list. "
                    "Re-run with --force-rebuild."
                )
            rows = load_buffer_rows(state.get("rows_buffer_path"))
            metadata_rows = load_metadata_rows(state.get("metadata_buffer_path"))
            packer = PackedSequencePacker(
                sequence_length=self.config.sequence_length,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                current=(state.get("packer_state") or {}).get("current", []),
                current_metadata=(state.get("packer_state") or {}).get("current_metadata", []),
                dropped_short_windows=int(
                    (state.get("packer_state") or {}).get("dropped_short_windows", 0)
                ),
            )
            shards = list(state.get("shards", []))
            shard_index = int(state.get("next_shard_index", len(shards)))
            num_sequences = int(state.get("num_sequences", 0))
            num_tokens = int(state.get("num_tokens", 0))
            dedupe_hash_chunks = list(state.get("dedupe_hash_chunks", []))
            source_audit_states = [
                self._restore_lm_audit_state(source, snapshot)
                for source, snapshot in zip(
                    sources,
                    list(state.get("source_audit_states", [])),
                    strict=False,
                )
            ]
            if len(source_audit_states) != len(sources):
                source_audit_states = [self._new_lm_audit_state(source) for source in sources]
            seen_hashes = load_seen_hashes(dedupe_hash_chunks) if any(source.deduplicate for source in sources) else set()
        else:
            source_progress = self._initial_source_progress(sources)
            rows: list[list[int]] = []
            metadata_rows: list[dict[str, Any]] = []
            packer = PackedSequencePacker(
                sequence_length=self.config.sequence_length,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
            )
            shards = []
            shard_index = 0
            num_sequences = 0
            num_tokens = 0
            dedupe_hash_chunks: list[str] = []
            seen_hashes: set[str] = set()
            source_audit_states = [self._new_lm_audit_state(source) for source in sources]

        def _stage_progress(message: str) -> None:
            _progress(message)

        if action == "resume":
            _stage_progress(
                f"WebbGPT: resuming prepared stage {stage} "
                f"from {len(shards):,} shard(s) and {num_tokens:,} packed tokens."
            )
        else:
            _stage_progress(f"WebbGPT: starting fresh prepared stage {stage}.")

        pending_hashes: list[str] = []
        consumed_since_snapshot = 0

        def snapshot_state() -> None:
            nonlocal pending_hashes, consumed_since_snapshot
            if pending_hashes:
                chunk_path = append_hash_chunk(
                    resume_workspace / f"dedupe-{len(dedupe_hash_chunks):05d}.txt",
                    pending_hashes,
                )
                if chunk_path is not None:
                    dedupe_hash_chunks.append(chunk_path)
                pending_hashes = []
            buffer_path = save_buffer_rows(rows_buffer_path, rows)
            metadata_buffer = save_metadata_rows(metadata_buffer_path, metadata_rows)
            if buffer_path is None and rows_buffer_path.exists():
                rows_buffer_path.unlink()
            if metadata_buffer is None and metadata_buffer_path.exists():
                metadata_buffer_path.unlink()
            save_resume_state(
                resume_state_path,
                {
                    "version": "1.0",
                    "stage": stage,
                    "kind": "packed_lm",
                    "input_fingerprint": input_fingerprint,
                    "tokenizer_path": self.config.tokenizer_path,
                    "sequence_length": self.config.sequence_length,
                    "pad_token_id": pad_token_id,
                    "eos_token_id": eos_token_id,
                    "rows_per_shard": self.config.prepared_shard_size,
                    "token_budget": token_budget,
                    "source_snapshots": source_snapshots,
                    "source_progress": source_progress,
                    "source_audit_states": [
                        self._serialize_lm_audit_state(audit_state)
                        for audit_state in source_audit_states
                    ],
                    "shards": shards,
                    "next_shard_index": shard_index,
                    "num_sequences": num_sequences,
                    "num_tokens": num_tokens,
                    "packer_state": packer.state_dict(),
                    "rows_buffer_path": buffer_path,
                    "metadata_buffer_path": metadata_buffer,
                    "dedupe_hash_chunks": dedupe_hash_chunks,
                },
            )
            consumed_since_snapshot = 0

        def flush_completed_shard(*, final: bool = False) -> None:
            nonlocal rows, metadata_rows, shard_index
            if not rows:
                return
            shard_path = shard_dir / f"shard-{shard_index:05d}.npy"
            metadata_path = shard_dir / f"metadata-{shard_index:05d}.jsonl"
            save_buffer_rows(shard_path, rows)
            save_metadata_rows(metadata_path, metadata_rows)
            shards.append({"path": str(shard_path), "metadata_path": str(metadata_path), "rows": len(rows)})
            message_prefix = "final shard" if final else "shard"
            _stage_progress(
                f"WebbGPT: preparing {stage}: wrote {message_prefix} {shard_index + 1} "
                f"({num_sequences:,} sequences, {num_tokens:,} packed tokens so far)."
            )
            rows = []
            metadata_rows = []
            shard_index += 1
            snapshot_state()

        token_budget_reached = token_budget is not None and num_tokens >= token_budget
        use_weighted_mix = len(sources) > 1 and any(abs(float(source.weight) - 1.0) > 1e-6 for source in sources)
        if use_weighted_mix:
            _stage_progress(
                f"WebbGPT: preparing {stage} with weighted source mixing across {len(sources)} sources."
            )
            for source, cleaned_record, token_ids, _audit in self._iter_weighted_tokenized_documents(
                sources,
                tokenizer=tokenizer,
                source_progress=source_progress,
                source_audits=source_audit_states,
                seen_hashes=seen_hashes,
            ):
                consumed_since_snapshot += 1
                if cleaned_record.document_id and source.deduplicate:
                    pending_hashes.append(cleaned_record.document_id)
                sequence_metadata = {
                    "source": source.name,
                    "family": _source_family(source),
                    "document_id": cleaned_record.document_id or "",
                }
                for sequence, packed_metadata in packer.push_with_metadata(token_ids, sequence_metadata):
                    rows.append(sequence)
                    metadata_rows.append(packed_metadata)
                    num_sequences += 1
                    num_tokens += sum(token != pad_token_id for token in sequence)
                    if len(rows) >= self.config.prepared_shard_size:
                        flush_completed_shard()
                    if token_budget is not None and num_tokens >= token_budget:
                        token_budget_reached = True
                        break
                if consumed_since_snapshot >= PREPARE_DOC_SNAPSHOT_INTERVAL:
                    snapshot_state()
                if token_budget_reached:
                    break
            for source_index, source in enumerate(sources):
                kept_records = int(source_progress[source_index].get("accepted_records", 0))
                _stage_progress(
                    f"WebbGPT: preparing {stage} source {source.name}: "
                    f"finished with {kept_records:,} documents kept."
                )
        else:
            for source_index, source in enumerate(sources):
                progress = source_progress[source_index]
                kept_records = int(progress.get("accepted_records", 0))
                if token_budget_reached:
                    break
                _stage_progress(
                    f"WebbGPT: preparing {stage} source {source.name} "
                    f"({source.format}) from {_source_location(source)}."
                )
                for cleaned_record, token_ids in self._iter_tokenized_documents_for_source(
                    source,
                    tokenizer=tokenizer,
                    seen_hashes=seen_hashes,
                    raw_records_consumed=int(progress.get("raw_records_consumed", 0)),
                    audit_state=source_audit_states[source_index],
                    progress_state=progress,
                ):
                    consumed_since_snapshot += 1
                    if cleaned_record.document_id and source.deduplicate:
                        pending_hashes.append(cleaned_record.document_id)
                    kept_records = int(progress.get("accepted_records", 0))
                    sequence_metadata = {
                        "source": source.name,
                        "family": _source_family(source),
                        "document_id": cleaned_record.document_id or "",
                    }
                    for sequence, packed_metadata in packer.push_with_metadata(token_ids, sequence_metadata):
                        rows.append(sequence)
                        metadata_rows.append(packed_metadata)
                        num_sequences += 1
                        num_tokens += sum(token != pad_token_id for token in sequence)
                        if len(rows) >= self.config.prepared_shard_size:
                            flush_completed_shard()
                        if token_budget is not None and num_tokens >= token_budget:
                            token_budget_reached = True
                            break
                    if kept_records % 1000 == 0 and kept_records > 0:
                        _stage_progress(
                            f"WebbGPT: preparing {stage} source {source.name}: "
                            f"kept {kept_records:,} documents so far."
                        )
                    if consumed_since_snapshot >= PREPARE_DOC_SNAPSHOT_INTERVAL:
                        snapshot_state()
                    if token_budget_reached:
                        break
                _stage_progress(
                    f"WebbGPT: preparing {stage} source {source.name}: "
                    f"finished with {kept_records:,} documents kept."
                )

        if not token_budget_reached:
            for sequence, packed_metadata in packer.finish_with_metadata():
                rows.append(sequence)
                metadata_rows.append(packed_metadata)
                num_sequences += 1
                num_tokens += sum(token != pad_token_id for token in sequence)
                if len(rows) >= self.config.prepared_shard_size:
                    flush_completed_shard()

        flush_completed_shard(final=True)
        diagnostics = self._lm_source_diagnostics(
            source_audit_states,
            total_tokens=num_tokens,
            total_documents=sum(int(progress.get("accepted_records", 0)) for progress in source_progress),
            stage=stage,
        )
        diagnostics["too_short_packed_sequences"] = int(packer.dropped_short_windows)
        prepare_warnings: list[str] = []
        domain_realization_gate = diagnostics.get("domain_realization_gate") if stage == "pretrain" else None
        corpus_quality_gate = diagnostics.get("corpus_quality_gate") if stage == "pretrain" else None
        broad_source_quality_gate = diagnostics.get("broad_source_quality_gate") if stage == "pretrain" else None
        if isinstance(domain_realization_gate, dict) and not bool(domain_realization_gate.get("passed", True)):
            message = str(domain_realization_gate.get("message", "pretrain domain realization failed"))
            prepare_warnings.append(message)
            _stage_progress(f"WebbGPT: {message}")
        if isinstance(corpus_quality_gate, dict) and not bool(corpus_quality_gate.get("passed", True)):
            message = str(corpus_quality_gate.get("message", "pretrain corpus quality gate failed"))
            prepare_warnings.append(message)
            _stage_progress(f"WebbGPT: {message}")
        if isinstance(broad_source_quality_gate, dict) and not bool(
            broad_source_quality_gate.get("passed", True)
        ):
            message = str(
                broad_source_quality_gate.get("message", "pretrain broad source quality gate failed")
            )
            prepare_warnings.append(message)
            _stage_progress(f"WebbGPT: {message}")
        manifest = {
            "version": "2.0",
            "stage": stage,
            "kind": "packed_lm",
            "input_fingerprint": input_fingerprint,
            "tokenizer_path": self.config.tokenizer_path,
            "sequence_length": self.config.sequence_length,
            "pad_token_id": pad_token_id,
            "eos_token_id": eos_token_id,
            "num_sequences": num_sequences,
            "num_tokens": num_tokens,
            "source_snapshots": source_snapshots,
            "diagnostics": diagnostics,
            "prepare_warnings": prepare_warnings,
            "shards": shards,
        }
        if isinstance(domain_realization_gate, dict):
            manifest["domain_realization_gate"] = domain_realization_gate
        if isinstance(corpus_quality_gate, dict):
            manifest["corpus_quality_gate"] = corpus_quality_gate
        if isinstance(broad_source_quality_gate, dict):
            manifest["broad_source_quality_gate"] = broad_source_quality_gate
        save_prepared_manifest(manifest_path, manifest)
        remove_resume_artifacts(manifest_path)
        _stage_progress(
            f"WebbGPT: finished preparing {stage} "
            f"({num_sequences:,} sequences across {len(shards):,} shards, {num_tokens:,} packed tokens)."
        )
        if (
            isinstance(domain_realization_gate, dict)
            and not bool(domain_realization_gate.get("passed", True))
            and domain_realization_gate.get("mode") == "fail"
        ):
            raise RuntimeError(str(domain_realization_gate.get("message")))
        if (
            isinstance(corpus_quality_gate, dict)
            and not bool(corpus_quality_gate.get("passed", True))
            and corpus_quality_gate.get("mode") == "fail"
        ):
            raise RuntimeError(str(corpus_quality_gate.get("message")))
        if (
            isinstance(broad_source_quality_gate, dict)
            and not bool(broad_source_quality_gate.get("passed", True))
            and broad_source_quality_gate.get("mode") == "fail"
        ):
            raise RuntimeError(str(broad_source_quality_gate.get("message")))
        return manifest

    def _should_keep_sft_example(
        self,
        *,
        bucket: str,
        label_token_count: int,
        assistant_text: str,
    ) -> tuple[bool, str | None]:
        response = assistant_text.lower().strip()
        if bucket == "hard_refusal" and (
            response in {"i can't say that.", "i cant say that.", "i can’t say that."}
            or response in {"i can't help you.", "i can’t help you.", "i can't help you to help."}
        ):
            return False, "generic_refusal"
        if bucket == "informative_abstention" and label_token_count < 24:
            return False, "too_short_abstention"
        return True, None

    def _planned_sft_example_total(
        self,
        candidate_bucket_counts: Counter[str],
    ) -> int:
        total_candidates = sum(int(count) for count in candidate_bucket_counts.values())
        if total_candidates <= SFT_BALANCING_WARMUP_EXAMPLES:
            return total_candidates
        floor_total = sum(
            min(int(candidate_bucket_counts[bucket]), _sft_min_examples(bucket))
            for bucket in candidate_bucket_counts
        )
        feasible_limits: list[int] = []
        for bucket, available in candidate_bucket_counts.items():
            target_share = _sft_target_row_share(bucket)
            minimum_examples = _sft_min_examples(bucket)
            if target_share <= 0.0 or int(available) < max(minimum_examples, 1):
                continue
            feasible_limits.append(max(0, math.floor(int(available) / target_share)))
        if not feasible_limits:
            return total_candidates
        return min(total_candidates, max(floor_total, min(feasible_limits)))

    def _planned_sft_bucket_quotas(
        self,
        candidate_bucket_counts: Counter[str],
        *,
        planned_total_examples: int,
    ) -> Counter[str]:
        quotas: Counter[str] = Counter()
        if planned_total_examples <= 0:
            return quotas
        for bucket, available in candidate_bucket_counts.items():
            quotas[bucket] = min(int(available), _sft_min_examples(bucket))
        remaining_slots = planned_total_examples - sum(quotas.values())
        if remaining_slots < 0:
            overflow = -remaining_slots
            reducible = sorted(
                candidate_bucket_counts,
                key=lambda bucket: (_sft_target_row_share(bucket), bucket),
                reverse=True,
            )
            for bucket in reducible:
                minimum = min(int(candidate_bucket_counts[bucket]), _sft_min_examples(bucket))
                removable = max(0, int(quotas[bucket]) - minimum)
                if removable <= 0:
                    continue
                delta = min(removable, overflow)
                quotas[bucket] -= delta
                overflow -= delta
                if overflow <= 0:
                    break
            remaining_slots = planned_total_examples - sum(quotas.values())
        while remaining_slots > 0:
            eligible = [
                bucket
                for bucket, available in candidate_bucket_counts.items()
                if int(quotas[bucket]) < int(available)
            ]
            if not eligible:
                break
            selected_bucket = max(
                eligible,
                key=lambda bucket: (
                    _sft_target_row_share(bucket) - (quotas[bucket] / max(planned_total_examples, 1)),
                    _sft_target_row_share(bucket),
                    int(candidate_bucket_counts[bucket]) - int(quotas[bucket]),
                    bucket != "constructive_direct",
                    bucket,
                ),
            )
            quotas[selected_bucket] += 1
            remaining_slots -= 1
        return quotas

    def _ordered_sft_bucket_indices(
        self,
        bucket: str,
        indices: list[int],
    ) -> list[int]:
        if bucket == "informative_abstention":
            # Keep the planner quotas unchanged, but let the newest abstention
            # examples fill the bucket first so iterative prompt-adjacent fixes
            # are not crowded out by older verbose rows.
            return sorted(indices, reverse=True)
        return list(indices)

    def _select_sft_candidate_indices(
        self,
        candidate_metadata_rows: list[dict[str, Any]],
    ) -> tuple[list[int], dict[str, Any]]:
        candidate_bucket_indices: dict[str, list[int]] = {}
        candidate_bucket_counts: Counter[str] = Counter()
        candidate_bucket_label_tokens: Counter[str] = Counter()
        for index, metadata in enumerate(candidate_metadata_rows):
            bucket = str(metadata.get("behavior_bucket", "unspecified"))
            candidate_bucket_indices.setdefault(bucket, []).append(index)
            candidate_bucket_counts[bucket] += 1
            candidate_bucket_label_tokens[bucket] += int(metadata.get("label_token_count", 0))
        planned_total_examples = self._planned_sft_example_total(candidate_bucket_counts)
        quotas = self._planned_sft_bucket_quotas(
            candidate_bucket_counts,
            planned_total_examples=planned_total_examples,
        )
        selected_indices: list[int] = []
        selected_bucket_counts: Counter[str] = Counter()
        selected_bucket_label_tokens: Counter[str] = Counter()
        distribution_reject_counts: Counter[str] = Counter()
        for bucket in sorted(candidate_bucket_indices):
            indices = self._ordered_sft_bucket_indices(
                bucket,
                candidate_bucket_indices[bucket],
            )
            keep_count = min(len(indices), int(quotas[bucket]))
            selected_bucket_counts[bucket] = keep_count
            if keep_count > 0:
                kept_indices = indices[:keep_count]
                selected_indices.extend(kept_indices)
                for index in kept_indices:
                    selected_bucket_label_tokens[bucket] += int(
                        candidate_metadata_rows[index].get("label_token_count", 0)
                    )
            if keep_count < len(indices):
                distribution_reject_counts[bucket] = len(indices) - keep_count
        selected_indices.sort()
        selected_examples = len(selected_indices)
        selected_label_tokens = sum(selected_bucket_label_tokens.values())
        bucket_targets: dict[str, dict[str, float]] = {}
        observed_buckets = sorted(set(candidate_bucket_counts) | set(SFT_BUCKET_TARGETS))
        for bucket in observed_buckets:
            bucket_targets[bucket] = {
                "target_row_share": round(_sft_target_row_share(bucket), 6),
                "target_label_token_share": round(_sft_target_label_token_share(bucket), 6),
                "candidate_examples": int(candidate_bucket_counts[bucket]),
                "candidate_label_tokens": int(candidate_bucket_label_tokens[bucket]),
                "accepted_examples": int(selected_bucket_counts[bucket]),
                "accepted_label_tokens": int(selected_bucket_label_tokens[bucket]),
                "distribution_rejects": int(distribution_reject_counts[bucket]),
                "realized_row_share": round(
                    _token_share(int(selected_bucket_counts[bucket]), selected_examples),
                    6,
                ),
                "realized_label_token_share": round(
                    _token_share(int(selected_bucket_label_tokens[bucket]), selected_label_tokens),
                    6,
                ),
                "distribution_gap": round(
                    _sft_target_row_share(bucket)
                    - _token_share(int(selected_bucket_counts[bucket]), selected_examples),
                    6,
                ),
            }
        return selected_indices, {
            "planned_total_examples": planned_total_examples,
            "candidate_bucket_counts": candidate_bucket_counts,
            "candidate_bucket_label_tokens": candidate_bucket_label_tokens,
            "selected_bucket_counts": selected_bucket_counts,
            "selected_bucket_label_tokens": selected_bucket_label_tokens,
            "distribution_reject_counts": distribution_reject_counts,
            "bucket_targets": bucket_targets,
        }

    def _validate_preference_metadata(
        self,
        examples: list[PreferenceExample],
    ) -> dict[str, Any]:
        invalid_quality_tiers: Counter[str] = Counter()
        invalid_negative_types: Counter[str] = Counter()
        approved_template_count = 0
        for example in examples:
            quality_tier = _coerce_chosen_quality_tier(example.metadata)
            negative_type = _coerce_negative_type(example.metadata)
            if quality_tier not in PROMOTABLE_DPO_CHOSEN_QUALITY_TIERS:
                invalid_quality_tiers[quality_tier] += 1
            if negative_type not in PROMOTABLE_DPO_NEGATIVE_TYPES:
                invalid_negative_types[negative_type] += 1
            if quality_tier == "approved_template":
                approved_template_count += 1
        total_examples = len(examples)
        template_share = approved_template_count / max(total_examples, 1)
        blockers: list[str] = []
        if invalid_quality_tiers:
            blockers.append("invalid_chosen_quality_tier")
        if invalid_negative_types:
            blockers.append("invalid_negative_type")
        if template_share > 0.25:
            blockers.append("approved_template_share_too_high")
        return {
            "valid_for_promotion": not blockers,
            "promotion_blockers": blockers,
            "invalid_quality_tiers": _counter_to_sorted_dict(invalid_quality_tiers),
            "invalid_negative_types": _counter_to_sorted_dict(invalid_negative_types),
            "approved_template_count": approved_template_count,
            "approved_template_share": round(template_share, 6),
        }

    def _prepare_sft_stage(
        self,
        stage: str,
        sources: list[DataSourceConfig],
        output_path: str,
        *,
        force_rebuild: bool,
    ) -> dict[str, Any]:
        tokenizer = SentencePieceTokenizer(self.config.tokenizer_path)
        source_snapshots = [source.to_dict() for source in sources]
        input_fingerprint = build_input_fingerprint(
            stage=stage,
            kind="sft",
            tokenizer_path=self.config.tokenizer_path,
            sequence_length=self.config.sequence_length,
            rows_per_shard=self.config.prepared_shard_size,
            source_snapshots=source_snapshots,
            extra={"label_mode": "assistant_only_v1"},
        )
        action, payload = self._resolve_prepare_target(
            stage=stage,
            kind="sft",
            output_path=output_path,
            input_fingerprint=input_fingerprint,
            force_rebuild=force_rebuild,
        )
        if action == "reuse":
            return payload or {}

        manifest_path = Path(output_path)
        shard_dir = manifest_path.with_suffix("")
        shard_dir.mkdir(parents=True, exist_ok=True)
        resume_workspace = prepared_resume_dir(manifest_path)
        resume_workspace.mkdir(parents=True, exist_ok=True)
        resume_state_path = prepared_resume_state_path(manifest_path)
        input_buffer_path = resume_workspace / "input-buffer.npy"
        label_buffer_path = resume_workspace / "label-buffer.npy"
        metadata_buffer_path = resume_workspace / "metadata-buffer.jsonl"
        if action == "resume":
            state = payload or {}
            source_progress = list(state.get("source_progress", []))
            if len(source_progress) != len(sources):
                raise RuntimeError(
                    f"Prepared-data resume state at {resume_state_path} no longer matches the configured source list. "
                    "Re-run with --force-rebuild."
                )
            candidate_input_rows = load_buffer_rows(state.get("input_buffer_path"))
            candidate_label_rows = load_buffer_rows(state.get("label_buffer_path"))
            candidate_metadata_rows = load_metadata_rows(state.get("metadata_buffer_path"))
            shards = list(state.get("shards", []))
            shard_index = int(state.get("next_shard_index", len(shards)))
            candidate_source_row_counts = Counter(state.get("source_row_counts", {}))
            candidate_source_label_token_counts = Counter(state.get("source_label_token_counts", {}))
            candidate_bucket_row_counts = Counter(state.get("bucket_row_counts", {}))
            candidate_bucket_label_token_counts = Counter(state.get("bucket_label_token_counts", {}))
            skipped_bucket_counts = Counter(state.get("skipped_bucket_counts", {}))
            skipped_reason_counts = Counter(state.get("skipped_reason_counts", {}))
            candidate_prompt_signature_counts = Counter(state.get("prompt_signature_counts", {}))
            truncated_examples = int(state.get("truncated_examples", 0))
        else:
            source_progress = self._initial_source_progress(sources)
            candidate_input_rows = []
            candidate_label_rows = []
            candidate_metadata_rows = []
            shards = []
            shard_index = 0
            candidate_source_row_counts = Counter()
            candidate_source_label_token_counts = Counter()
            candidate_bucket_row_counts = Counter()
            candidate_bucket_label_token_counts = Counter()
            skipped_bucket_counts = Counter()
            skipped_reason_counts = Counter()
            candidate_prompt_signature_counts = Counter()
            truncated_examples = 0

        def _stage_progress(message: str) -> None:
            _progress(message)

        if action == "resume":
            _stage_progress(
                f"WebbGPT: resuming prepared stage {stage} "
                f"from {len(shards):,} shard(s) and {len(candidate_metadata_rows):,} collected candidates."
            )
        else:
            _stage_progress(f"WebbGPT: starting fresh prepared stage {stage}.")

        consumed_since_snapshot = 0

        def snapshot_state() -> None:
            nonlocal consumed_since_snapshot
            input_buffer = save_buffer_rows(input_buffer_path, candidate_input_rows)
            label_buffer = save_buffer_rows(label_buffer_path, candidate_label_rows)
            metadata_buffer = save_metadata_rows(metadata_buffer_path, candidate_metadata_rows)
            if input_buffer is None and input_buffer_path.exists():
                input_buffer_path.unlink()
            if label_buffer is None and label_buffer_path.exists():
                label_buffer_path.unlink()
            if metadata_buffer is None and metadata_buffer_path.exists():
                metadata_buffer_path.unlink()
            save_resume_state(
                resume_state_path,
                {
                    "version": "2.0",
                    "stage": stage,
                    "kind": "sft",
                    "input_fingerprint": input_fingerprint,
                    "tokenizer_path": self.config.tokenizer_path,
                    "sequence_length": self.config.sequence_length,
                    "pad_token_id": tokenizer.token_to_id("<pad>"),
                    "rows_per_shard": self.config.prepared_shard_size,
                    "source_snapshots": source_snapshots,
                    "source_progress": source_progress,
                    "shards": shards,
                    "next_shard_index": shard_index,
                    "num_examples": len(candidate_metadata_rows),
                    "num_label_tokens": sum(
                        int(row.get("label_token_count", 0))
                        for row in candidate_metadata_rows
                    ),
                    "input_buffer_path": input_buffer,
                    "label_buffer_path": label_buffer,
                    "metadata_buffer_path": metadata_buffer,
                    "source_row_counts": dict(candidate_source_row_counts),
                    "source_label_token_counts": dict(candidate_source_label_token_counts),
                    "bucket_row_counts": dict(candidate_bucket_row_counts),
                    "bucket_label_token_counts": dict(candidate_bucket_label_token_counts),
                    "skipped_bucket_counts": dict(skipped_bucket_counts),
                    "skipped_reason_counts": dict(skipped_reason_counts),
                    "prompt_signature_counts": dict(candidate_prompt_signature_counts),
                    "truncated_examples": truncated_examples,
                },
            )
            consumed_since_snapshot = 0

        def flush_completed_shard(
            shard_input_rows: list[list[int]],
            shard_label_rows: list[list[int]],
            shard_metadata_rows: list[dict[str, Any]],
            *,
            final: bool = False,
            num_examples: int,
            num_label_tokens: int,
        ) -> None:
            nonlocal shard_index
            if not shard_input_rows:
                return
            input_path = shard_dir / f"input_ids-{shard_index:05d}.npy"
            label_path = shard_dir / f"labels-{shard_index:05d}.npy"
            metadata_path = shard_dir / f"metadata-{shard_index:05d}.jsonl"
            save_buffer_rows(input_path, shard_input_rows)
            save_buffer_rows(label_path, shard_label_rows)
            save_metadata_rows(metadata_path, shard_metadata_rows)
            shards.append(
                {
                    "input_ids_path": str(input_path),
                    "labels_path": str(label_path),
                    "metadata_path": str(metadata_path),
                    "rows": len(shard_input_rows),
                }
            )
            message_prefix = "final shard" if final else "shard"
            _stage_progress(
                f"WebbGPT: preparing {stage}: wrote {message_prefix} {shard_index + 1} "
                f"({num_examples:,} examples, {num_label_tokens:,} supervised tokens so far)."
            )
            shard_index += 1

        for source_index, source in enumerate(sources):
            progress = source_progress[source_index]
            accepted_records = int(progress.get("accepted_records", 0))
            _stage_progress(f"WebbGPT: preparing {stage} source {source.name} ({source.format}).")
            for item in self._load_source_records(
                source,
                raw_records_consumed=int(progress.get("raw_records_consumed", 0)),
            ):
                progress["raw_records_consumed"] = int(progress.get("raw_records_consumed", 0)) + 1
                consumed_since_snapshot += 1
                messages = item.get(source.messages_field)
                if not isinstance(messages, list):
                    prompt_messages = _coerce_prompt_messages(item.get(source.prompt_field))
                    response = item.get(source.response_field)
                    if not isinstance(response, str):
                        response = item.get(source.chosen_field)
                    if prompt_messages is None or not isinstance(response, str):
                        if consumed_since_snapshot >= PREPARE_EXAMPLE_SNAPSHOT_INTERVAL:
                            snapshot_state()
                        continue
                    messages = [*prompt_messages, {"role": "assistant", "content": response}]
                input_ids, labels = encode_sft_messages(messages, tokenizer, self.config.sequence_length)
                metadata = _collect_standard_metadata(
                    item,
                    metadata_fields=source.metadata_fields,
                    extra_fields=STANDARD_SFT_METADATA_FIELDS,
                )
                prompt_hash = _prompt_signature_hash(messages)
                bucket = _infer_behavior_bucket(messages, metadata)
                label_token_count = sum(label != -100 for label in labels)
                truncated = sum(token != tokenizer.token_to_id("<pad>") for token in input_ids) >= self.config.sequence_length
                keep_example, skipped_reason = self._should_keep_sft_example(
                    bucket=bucket,
                    label_token_count=label_token_count,
                    assistant_text=_assistant_response_text(messages),
                )
                if not keep_example:
                    skipped_bucket_counts[bucket] += 1
                    skipped_reason_counts[skipped_reason or "filtered"] += 1
                    if consumed_since_snapshot >= PREPARE_EXAMPLE_SNAPSHOT_INTERVAL:
                        snapshot_state()
                    continue
                candidate_input_rows.append(input_ids)
                candidate_label_rows.append(labels)
                candidate_metadata_rows.append(
                    {
                        "example_id": _message_example_id(source, item, messages),
                        "split_group_id": _message_group_id(source, item, messages),
                        "source": source.name,
                        "prompt_signature_hash": prompt_hash,
                        "behavior_bucket": bucket,
                        "quality_tier": _coerce_quality_tier(metadata),
                        "label_token_count": label_token_count,
                    }
                )
                accepted_records += 1
                progress["accepted_records"] = accepted_records
                candidate_source_row_counts[source.name] += 1
                candidate_source_label_token_counts[source.name] += label_token_count
                candidate_bucket_row_counts[bucket] += 1
                candidate_bucket_label_token_counts[bucket] += label_token_count
                candidate_prompt_signature_counts[prompt_hash] += 1
                if truncated:
                    truncated_examples += 1
                if accepted_records % 500 == 0:
                    _stage_progress(
                        f"WebbGPT: preparing {stage} source {source.name}: "
                        f"collected {accepted_records:,} valid SFT candidates so far."
                    )
                if consumed_since_snapshot >= PREPARE_EXAMPLE_SNAPSHOT_INTERVAL:
                    snapshot_state()
            _stage_progress(
                f"WebbGPT: preparing {stage} source {source.name}: "
                f"finished with {accepted_records:,} valid SFT candidates."
            )
        selected_indices, planner = self._select_sft_candidate_indices(candidate_metadata_rows)
        selected_input_rows = [candidate_input_rows[index] for index in selected_indices]
        selected_label_rows = [candidate_label_rows[index] for index in selected_indices]
        selected_metadata_rows = [candidate_metadata_rows[index] for index in selected_indices]
        num_examples = len(selected_metadata_rows)
        num_label_tokens = sum(int(row.get("label_token_count", 0)) for row in selected_metadata_rows)
        source_row_counts = Counter(str(row.get("source", "prepared")) for row in selected_metadata_rows)
        source_label_token_counts: Counter[str] = Counter()
        bucket_row_counts = Counter(str(row.get("behavior_bucket", "unspecified")) for row in selected_metadata_rows)
        bucket_label_token_counts: Counter[str] = Counter()
        selected_prompt_signature_counts: Counter[str] = Counter()
        for row in selected_metadata_rows:
            source_name = str(row.get("source", "prepared"))
            bucket_name = str(row.get("behavior_bucket", "unspecified"))
            label_token_count = int(row.get("label_token_count", 0))
            source_label_token_counts[source_name] += label_token_count
            bucket_label_token_counts[bucket_name] += label_token_count
            prompt_hash = str(row.get("prompt_signature_hash", ""))
            if prompt_hash:
                selected_prompt_signature_counts[prompt_hash] += 1
        shard_input_rows: list[list[int]] = []
        shard_label_rows: list[list[int]] = []
        shard_metadata_rows: list[dict[str, Any]] = []
        emitted_examples = 0
        emitted_label_tokens = 0
        for input_ids, labels, metadata in zip(
            selected_input_rows,
            selected_label_rows,
            selected_metadata_rows,
            strict=False,
        ):
            shard_input_rows.append(input_ids)
            shard_label_rows.append(labels)
            shard_metadata_rows.append(metadata)
            emitted_examples += 1
            emitted_label_tokens += int(metadata.get("label_token_count", 0))
            if len(shard_input_rows) >= self.config.prepared_shard_size:
                flush_completed_shard(
                    shard_input_rows,
                    shard_label_rows,
                    shard_metadata_rows,
                    num_examples=emitted_examples,
                    num_label_tokens=emitted_label_tokens,
                )
                shard_input_rows = []
                shard_label_rows = []
                shard_metadata_rows = []
        flush_completed_shard(
            shard_input_rows,
            shard_label_rows,
            shard_metadata_rows,
            final=True,
            num_examples=emitted_examples,
            num_label_tokens=emitted_label_tokens,
        )
        duplicate_prompt_count = sum(count - 1 for count in selected_prompt_signature_counts.values() if count > 1)
        manifest = {
            "version": "2.0",
            "stage": stage,
            "kind": "sft",
            "input_fingerprint": input_fingerprint,
            "tokenizer_path": self.config.tokenizer_path,
            "sequence_length": self.config.sequence_length,
            "pad_token_id": tokenizer.token_to_id("<pad>"),
            "num_examples": num_examples,
            "num_label_tokens": num_label_tokens,
            "source_snapshots": source_snapshots,
            "diagnostics": {
                "planned_total_examples": int(planner["planned_total_examples"]),
                "candidate_examples": len(candidate_metadata_rows),
                "candidate_label_tokens": sum(
                    int(row.get("label_token_count", 0))
                    for row in candidate_metadata_rows
                ),
                "per_source_rows": _counter_to_sorted_dict(source_row_counts),
                "per_source_label_tokens": _counter_to_sorted_dict(source_label_token_counts),
                "per_source_row_share": _share_dict(source_row_counts, num_examples),
                "per_source_label_token_share": _share_dict(source_label_token_counts, num_label_tokens),
                "candidate_per_source_rows": _counter_to_sorted_dict(candidate_source_row_counts),
                "candidate_per_source_label_tokens": _counter_to_sorted_dict(candidate_source_label_token_counts),
                "candidate_per_source_row_share": _share_dict(candidate_source_row_counts, len(candidate_metadata_rows)),
                "candidate_per_source_label_token_share": _share_dict(
                    candidate_source_label_token_counts,
                    sum(
                        int(row.get("label_token_count", 0))
                        for row in candidate_metadata_rows
                    ),
                ),
                "per_bucket_rows": _counter_to_sorted_dict(bucket_row_counts),
                "per_bucket_label_tokens": _counter_to_sorted_dict(bucket_label_token_counts),
                "per_bucket_row_share": _share_dict(bucket_row_counts, num_examples),
                "per_bucket_label_token_share": _share_dict(bucket_label_token_counts, num_label_tokens),
                "candidate_per_bucket_rows": _counter_to_sorted_dict(planner["candidate_bucket_counts"]),
                "candidate_per_bucket_label_tokens": _counter_to_sorted_dict(planner["candidate_bucket_label_tokens"]),
                "candidate_per_bucket_row_share": _share_dict(
                    planner["candidate_bucket_counts"],
                    len(candidate_metadata_rows),
                ),
                "candidate_per_bucket_label_token_share": _share_dict(
                    planner["candidate_bucket_label_tokens"],
                    sum(
                        int(row.get("label_token_count", 0))
                        for row in candidate_metadata_rows
                    ),
                ),
                "bucket_planner_targets": planner["bucket_targets"],
                "distribution_reject_bucket_rows": _counter_to_sorted_dict(planner["distribution_reject_counts"]),
                "skipped_bucket_rows": _counter_to_sorted_dict(skipped_bucket_counts),
                "skipped_reason_rows": _counter_to_sorted_dict(skipped_reason_counts),
                "truncated_examples": truncated_examples,
                "truncation_rate": 0.0 if len(candidate_metadata_rows) <= 0 else round(truncated_examples / len(candidate_metadata_rows), 6),
                "prompt_signature_unique_count": len(selected_prompt_signature_counts),
                "prompt_signature_duplicate_count": duplicate_prompt_count,
            },
            "trust": {
                "artifact_status": "promotable",
                "promotion_blockers": [],
                "supports_prompt_overlap_check": True,
            },
            "shards": shards,
        }
        save_prepared_manifest(manifest_path, manifest)
        remove_resume_artifacts(manifest_path)
        _stage_progress(
            f"WebbGPT: finished preparing {stage} "
            f"({num_examples:,} examples across {len(shards):,} shards, {num_label_tokens:,} supervised tokens)."
        )
        return manifest

    def _prepare_preference_stage(
        self,
        stage: str,
        sources: list[DataSourceConfig],
        output_path: str,
        *,
        force_rebuild: bool,
    ) -> dict[str, Any]:
        tokenizer = SentencePieceTokenizer(self.config.tokenizer_path)
        source_snapshots = [source.to_dict() for source in sources]
        input_fingerprint = build_input_fingerprint(
            stage=stage,
            kind="preference",
            tokenizer_path=self.config.tokenizer_path,
            sequence_length=self.config.sequence_length,
            rows_per_shard=self.config.prepared_shard_size,
            source_snapshots=source_snapshots,
            extra={"preference_mode": "chosen_rejected_v1"},
        )
        action, payload = self._resolve_prepare_target(
            stage=stage,
            kind="preference",
            output_path=output_path,
            input_fingerprint=input_fingerprint,
            force_rebuild=force_rebuild,
        )
        if action == "reuse":
            return payload or {}

        manifest_path = Path(output_path)
        shard_dir = manifest_path.with_suffix("")
        shard_dir.mkdir(parents=True, exist_ok=True)
        resume_workspace = prepared_resume_dir(manifest_path)
        resume_workspace.mkdir(parents=True, exist_ok=True)
        resume_state_path = prepared_resume_state_path(manifest_path)
        chosen_buffer_path = resume_workspace / "chosen-buffer.npy"
        rejected_buffer_path = resume_workspace / "rejected-buffer.npy"
        metadata_buffer_path = resume_workspace / "metadata-buffer.jsonl"
        if action == "resume":
            state = payload or {}
            source_progress = list(state.get("source_progress", []))
            if len(source_progress) != len(sources):
                raise RuntimeError(
                    f"Prepared-data resume state at {resume_state_path} no longer matches the configured source list. "
                    "Re-run with --force-rebuild."
                )
            chosen_rows = load_buffer_rows(state.get("chosen_buffer_path"))
            rejected_rows = load_buffer_rows(state.get("rejected_buffer_path"))
            metadata_rows = load_metadata_rows(state.get("metadata_buffer_path"))
            shards = list(state.get("shards", []))
            shard_index = int(state.get("next_shard_index", len(shards)))
            num_examples = int(state.get("num_examples", 0))
            source_row_counts = Counter(state.get("source_row_counts", {}))
            source_token_counts = Counter(state.get("source_token_counts", {}))
            negative_type_counts = Counter(state.get("negative_type_counts", {}))
            prompt_signature_counts = Counter(state.get("prompt_signature_counts", {}))
            truncated_examples = int(state.get("truncated_examples", 0))
        else:
            source_progress = self._initial_source_progress(sources)
            chosen_rows = []
            rejected_rows = []
            metadata_rows = []
            shards = []
            shard_index = 0
            num_examples = 0
            source_row_counts = Counter()
            source_token_counts = Counter()
            negative_type_counts = Counter()
            prompt_signature_counts = Counter()
            truncated_examples = 0

        def _stage_progress(message: str) -> None:
            _progress(message)

        if action == "resume":
            _stage_progress(
                f"WebbGPT: resuming prepared stage {stage} "
                f"from {len(shards):,} shard(s) and {num_examples:,} preference examples."
            )
        else:
            _stage_progress(f"WebbGPT: starting fresh prepared stage {stage}.")

        consumed_since_snapshot = 0

        def snapshot_state() -> None:
            nonlocal consumed_since_snapshot
            chosen_buffer = save_buffer_rows(chosen_buffer_path, chosen_rows)
            rejected_buffer = save_buffer_rows(rejected_buffer_path, rejected_rows)
            metadata_buffer = save_metadata_rows(metadata_buffer_path, metadata_rows)
            if chosen_buffer is None and chosen_buffer_path.exists():
                chosen_buffer_path.unlink()
            if rejected_buffer is None and rejected_buffer_path.exists():
                rejected_buffer_path.unlink()
            if metadata_buffer is None and metadata_buffer_path.exists():
                metadata_buffer_path.unlink()
            save_resume_state(
                resume_state_path,
                {
                    "version": "2.0",
                    "stage": stage,
                    "kind": "preference",
                    "input_fingerprint": input_fingerprint,
                    "tokenizer_path": self.config.tokenizer_path,
                    "sequence_length": self.config.sequence_length,
                    "pad_token_id": tokenizer.token_to_id("<pad>"),
                    "rows_per_shard": self.config.prepared_shard_size,
                    "source_snapshots": source_snapshots,
                    "source_progress": source_progress,
                    "shards": shards,
                    "next_shard_index": shard_index,
                    "num_examples": num_examples,
                    "chosen_buffer_path": chosen_buffer,
                    "rejected_buffer_path": rejected_buffer,
                    "metadata_buffer_path": metadata_buffer,
                    "source_row_counts": dict(source_row_counts),
                    "source_token_counts": dict(source_token_counts),
                    "negative_type_counts": dict(negative_type_counts),
                    "prompt_signature_counts": dict(prompt_signature_counts),
                    "truncated_examples": truncated_examples,
                },
            )
            consumed_since_snapshot = 0

        def flush_completed_shard(*, final: bool = False) -> None:
            nonlocal chosen_rows, rejected_rows, metadata_rows, shard_index
            if not chosen_rows:
                return
            chosen_path = shard_dir / f"chosen_input_ids-{shard_index:05d}.npy"
            rejected_path = shard_dir / f"rejected_input_ids-{shard_index:05d}.npy"
            metadata_path = shard_dir / f"metadata-{shard_index:05d}.jsonl"
            save_buffer_rows(chosen_path, chosen_rows)
            save_buffer_rows(rejected_path, rejected_rows)
            save_metadata_rows(metadata_path, metadata_rows)
            shards.append(
                {
                    "chosen_input_ids_path": str(chosen_path),
                    "rejected_input_ids_path": str(rejected_path),
                    "metadata_path": str(metadata_path),
                    "rows": len(chosen_rows),
                }
            )
            message_prefix = "final shard" if final else "shard"
            _stage_progress(
                f"WebbGPT: preparing {stage}: wrote {message_prefix} {shard_index + 1} "
                f"({num_examples:,} preference examples so far)."
            )
            chosen_rows = []
            rejected_rows = []
            metadata_rows = []
            shard_index += 1
            snapshot_state()

        for source_index, source in enumerate(sources):
            progress = source_progress[source_index]
            accepted_records = int(progress.get("accepted_records", 0))
            _stage_progress(f"WebbGPT: preparing {stage} source {source.name} ({source.format}).")
            for item in self._load_source_records(
                source,
                raw_records_consumed=int(progress.get("raw_records_consumed", 0)),
            ):
                progress["raw_records_consumed"] = int(progress.get("raw_records_consumed", 0)) + 1
                consumed_since_snapshot += 1
                prompt = _coerce_prompt_messages(item.get(source.prompt_field))
                chosen = item.get(source.chosen_field)
                rejected = item.get(source.rejected_field)
                if prompt is None or not isinstance(chosen, str) or not isinstance(rejected, str):
                    if consumed_since_snapshot >= PREPARE_EXAMPLE_SNAPSHOT_INTERVAL:
                        snapshot_state()
                    continue
                chosen_input_ids = encode_preference_example(prompt, chosen, tokenizer, self.config.sequence_length)
                rejected_input_ids = encode_preference_example(prompt, rejected, tokenizer, self.config.sequence_length)
                metadata = _collect_standard_metadata(
                    item,
                    metadata_fields=source.metadata_fields,
                    extra_fields=STANDARD_PREFERENCE_METADATA_FIELDS,
                )
                prompt_hash = _prompt_signature_hash(prompt)
                chosen_token_count = sum(token != tokenizer.token_to_id("<pad>") for token in chosen_input_ids)
                rejected_token_count = sum(token != tokenizer.token_to_id("<pad>") for token in rejected_input_ids)
                truncated = (
                    chosen_token_count >= self.config.sequence_length
                    or rejected_token_count >= self.config.sequence_length
                )
                chosen_rows.append(chosen_input_ids)
                rejected_rows.append(rejected_input_ids)
                metadata_rows.append(
                    {
                        "example_id": _preference_example_id(source, item, prompt, chosen, rejected),
                        "split_group_id": _message_group_id(source, item, prompt),
                        "source": source.name,
                        "prompt_signature_hash": prompt_hash,
                        "chosen_quality_tier": _coerce_chosen_quality_tier(metadata),
                        "negative_type": _coerce_negative_type(metadata),
                        "chosen_token_count": chosen_token_count,
                        "rejected_token_count": rejected_token_count,
                    }
                )
                accepted_records += 1
                progress["accepted_records"] = accepted_records
                num_examples += 1
                source_row_counts[source.name] += 1
                source_token_counts[source.name] += chosen_token_count + rejected_token_count
                negative_type_counts[_coerce_negative_type(metadata)] += 1
                prompt_signature_counts[prompt_hash] += 1
                if truncated:
                    truncated_examples += 1
                if len(chosen_rows) >= self.config.prepared_shard_size:
                    flush_completed_shard()
                if accepted_records % 500 == 0:
                    _stage_progress(
                        f"WebbGPT: preparing {stage} source {source.name}: "
                        f"loaded {accepted_records:,} preference examples so far."
                    )
                if consumed_since_snapshot >= PREPARE_EXAMPLE_SNAPSHOT_INTERVAL:
                    snapshot_state()
            _stage_progress(
                f"WebbGPT: preparing {stage} source {source.name}: "
                f"finished with {accepted_records:,} preference examples."
            )

        flush_completed_shard(final=True)
        duplicate_prompt_count = sum(count - 1 for count in prompt_signature_counts.values() if count > 1)
        manifest = {
            "version": "2.0",
            "stage": stage,
            "kind": "preference",
            "input_fingerprint": input_fingerprint,
            "tokenizer_path": self.config.tokenizer_path,
            "sequence_length": self.config.sequence_length,
            "pad_token_id": tokenizer.token_to_id("<pad>"),
            "num_examples": num_examples,
            "source_snapshots": source_snapshots,
            "diagnostics": {
                "per_source_rows": _counter_to_sorted_dict(source_row_counts),
                "per_source_tokens": _counter_to_sorted_dict(source_token_counts),
                "per_source_row_share": _share_dict(source_row_counts, num_examples),
                "per_source_token_share": _share_dict(source_token_counts, sum(source_token_counts.values())),
                "per_negative_type_rows": _counter_to_sorted_dict(negative_type_counts),
                "per_negative_type_row_share": _share_dict(negative_type_counts, num_examples),
                "truncated_examples": truncated_examples,
                "truncation_rate": 0.0 if num_examples <= 0 else round(truncated_examples / num_examples, 6),
                "prompt_signature_unique_count": len(prompt_signature_counts),
                "prompt_signature_duplicate_count": duplicate_prompt_count,
            },
            "trust": {
                "artifact_status": "promotable",
                "promotion_blockers": [],
                "supports_prompt_overlap_check": True,
            },
            "shards": shards,
        }
        save_prepared_manifest(manifest_path, manifest)
        remove_resume_artifacts(manifest_path)
        _stage_progress(
            f"WebbGPT: finished preparing {stage} "
            f"({num_examples:,} examples across {len(shards):,} shards)."
        )
        return manifest

    def prepare_stage(self, stage: str, output_path: str, *, force_rebuild: bool = False) -> dict[str, Any]:
        sources = self._stage_sources(stage)
        self._require_stage_sources(stage, sources)
        _progress(
            f"WebbGPT: preparing stage {stage} from {len(sources)} source(s): "
            + ", ".join(source.name for source in sources)
        )
        if self._uses_prepared_sources(sources):
            manifest = validate_prepared_manifest_artifacts(sources[0].path)
            output = Path(output_path)
            output.parent.mkdir(parents=True, exist_ok=True)
            save_prepared_manifest(output, manifest)
            return manifest

        if stage in {"pretrain", "continue", "validation"}:
            return self._prepare_packed_stage(stage, sources, output_path, force_rebuild=force_rebuild)
        if stage == "sft":
            return self._prepare_sft_stage(stage, sources, output_path, force_rebuild=force_rebuild)
        if stage == "preference":
            return self._prepare_preference_stage(stage, sources, output_path, force_rebuild=force_rebuild)

        raise ValueError(f"Unsupported stage {stage!r}")

    def _load_source_records(
        self,
        source: DataSourceConfig,
        *,
        raw_records_consumed: int = 0,
    ) -> Iterable[dict[str, Any]]:
        dataset_cls, iterable_cls, load_dataset = _require_datasets()
        source = _source_with_cursor(source, raw_records_consumed)
        dataset = None
        if source.format == "hf":
            dataset_name = source.dataset_name or source.path
            if not dataset_name:
                raise ValueError(f"HF source {source.name!r} requires dataset_name or path.")
            dataset = load_dataset(
                dataset_name,
                source.dataset_config_name,
                split=source.split,
                revision=source.dataset_revision,
                streaming=bool(source.streaming),
            )
        elif source.format == "text":
            dataset = load_dataset(
                "text",
                data_files=_data_files_for_source(source),
                split=source.split,
                streaming=bool(source.streaming),
            )
        elif source.format == "jsonl":
            dataset = load_dataset(
                "json",
                data_files=_data_files_for_source(source),
                split=source.split,
                streaming=bool(source.streaming),
            )
        elif source.format == "parquet":
            dataset = load_dataset(
                "parquet",
                data_files=_data_files_for_source(source),
                split=source.split,
                streaming=bool(source.streaming),
            )
        elif source.format == "arrow":
            dataset = load_dataset(
                "arrow",
                data_files=_data_files_for_source(source),
                split=source.split,
                streaming=bool(source.streaming),
            )
        elif source.format == "prepared":
            raise RuntimeError("Prepared-manifest sources cannot be read as raw records.")
        else:
            raise ValueError(f"Unsupported source format {source.format}")

        dataset = _apply_record_window(dataset, source)
        if isinstance(dataset, (dataset_cls, iterable_cls)):
            return dataset
        return dataset

    def _new_lm_audit_state(self, source: DataSourceConfig) -> dict[str, Any]:
        return {
            "source": source.name,
            "family": _source_family(source),
            "weight": float(source.weight),
            "target_share": 0.0,
            "quality_filter_mode": source.quality_filter_mode,
            "raw_records": 0,
            "kept_documents": 0,
            "kept_tokens": 0,
            "dropped_reasons": Counter(),
            "repeated_documents": 0,
            "restart_count": 0,
            "unique_document_ids": set(),
            "phrase_counter": Counter(),
            "phrase_counter_8": Counter(),
            "phrase_counter_12": Counter(),
            "phrase_counter_20": Counter(),
            "exact_paragraph_counter": Counter(),
            "normalized_paragraph_counter": Counter(),
            "synthetic_meta_phrase_counter": Counter(),
            "quality_artifact_counter": Counter(),
            "quality_artifact_occurrence_counter": Counter(),
            "quality_diagnostic_token_count": 0,
            "document_char_count": 0,
            "document_word_count": 0,
            "document_sentence_count": 0,
            "document_paragraph_count": 0,
        }

    def _serialize_lm_audit_state(self, audit_state: dict[str, Any]) -> dict[str, Any]:
        return {
            "source": str(audit_state["source"]),
            "family": str(audit_state["family"]),
            "weight": float(audit_state["weight"]),
            "target_share": float(audit_state.get("target_share", 0.0)),
            "quality_filter_mode": str(audit_state.get("quality_filter_mode", "basic")),
            "raw_records": int(audit_state["raw_records"]),
            "kept_documents": int(audit_state["kept_documents"]),
            "kept_tokens": int(audit_state["kept_tokens"]),
            "dropped_reasons": dict(audit_state["dropped_reasons"]),
            "repeated_documents": int(audit_state["repeated_documents"]),
            "restart_count": int(audit_state.get("restart_count", 0)),
            "phrase_counter": dict(audit_state["phrase_counter"]),
            "phrase_counter_8": dict(audit_state.get("phrase_counter_8", {})),
            "phrase_counter_12": dict(audit_state.get("phrase_counter_12", {})),
            "phrase_counter_20": dict(audit_state.get("phrase_counter_20", {})),
            "exact_paragraph_counter": dict(audit_state.get("exact_paragraph_counter", {})),
            "normalized_paragraph_counter": dict(audit_state.get("normalized_paragraph_counter", {})),
            "synthetic_meta_phrase_counter": dict(audit_state.get("synthetic_meta_phrase_counter", {})),
            "quality_artifact_counter": dict(audit_state.get("quality_artifact_counter", {})),
            "quality_artifact_occurrence_counter": dict(
                audit_state.get("quality_artifact_occurrence_counter", {})
            ),
            "quality_diagnostic_token_count": int(audit_state.get("quality_diagnostic_token_count", 0)),
            "document_char_count": int(audit_state.get("document_char_count", 0)),
            "document_word_count": int(audit_state.get("document_word_count", 0)),
            "document_sentence_count": int(audit_state.get("document_sentence_count", 0)),
            "document_paragraph_count": int(audit_state.get("document_paragraph_count", 0)),
        }

    def _restore_lm_audit_state(
        self,
        source: DataSourceConfig,
        snapshot: dict[str, Any] | None,
    ) -> dict[str, Any]:
        audit_state = self._new_lm_audit_state(source)
        if snapshot is None:
            return audit_state
        audit_state["target_share"] = float(snapshot.get("target_share", audit_state["target_share"]))
        audit_state["raw_records"] = int(snapshot.get("raw_records", 0))
        audit_state["kept_documents"] = int(snapshot.get("kept_documents", 0))
        audit_state["kept_tokens"] = int(snapshot.get("kept_tokens", 0))
        audit_state["dropped_reasons"] = Counter(snapshot.get("dropped_reasons", {}))
        audit_state["repeated_documents"] = int(snapshot.get("repeated_documents", 0))
        audit_state["restart_count"] = int(snapshot.get("restart_count", 0))
        audit_state["phrase_counter"] = Counter(snapshot.get("phrase_counter", {}))
        audit_state["phrase_counter_8"] = Counter(snapshot.get("phrase_counter_8", {}))
        audit_state["phrase_counter_12"] = Counter(snapshot.get("phrase_counter_12", {}))
        audit_state["phrase_counter_20"] = Counter(snapshot.get("phrase_counter_20", {}))
        audit_state["exact_paragraph_counter"] = Counter(snapshot.get("exact_paragraph_counter", {}))
        audit_state["normalized_paragraph_counter"] = Counter(snapshot.get("normalized_paragraph_counter", {}))
        audit_state["synthetic_meta_phrase_counter"] = Counter(snapshot.get("synthetic_meta_phrase_counter", {}))
        audit_state["quality_artifact_counter"] = Counter(snapshot.get("quality_artifact_counter", {}))
        audit_state["quality_artifact_occurrence_counter"] = Counter(
            snapshot.get("quality_artifact_occurrence_counter", {})
        )
        audit_state["quality_diagnostic_token_count"] = int(
            snapshot.get("quality_diagnostic_token_count", 0)
        )
        audit_state["document_char_count"] = int(snapshot.get("document_char_count", 0))
        audit_state["document_word_count"] = int(snapshot.get("document_word_count", 0))
        audit_state["document_sentence_count"] = int(snapshot.get("document_sentence_count", 0))
        audit_state["document_paragraph_count"] = int(snapshot.get("document_paragraph_count", 0))
        return audit_state

    def _realizable_lm_token_capacity(
        self,
        *,
        kept_documents: int,
        kept_tokens: int,
        repeated_documents: int,
    ) -> int:
        if kept_documents <= 0 or kept_tokens <= 0:
            return 0
        repeat_cap = float(self.config.lm_max_source_repeat_rate)
        if repeat_cap >= 1.0:
            return kept_tokens
        unique_documents = max(kept_documents - repeated_documents, 0)
        if unique_documents <= 0:
            return kept_tokens
        if repeat_cap <= 0.0:
            max_documents = unique_documents
        else:
            max_documents = int(math.floor(unique_documents / max(1.0 - repeat_cap, 1e-8) + 1e-8))
        average_tokens = kept_tokens / max(kept_documents, 1)
        return max(kept_tokens, int(math.floor(average_tokens * max_documents + 1e-8)))

    def _lm_source_report_from_audit_state(self, audit_state: dict[str, Any]) -> dict[str, Any]:
        kept_documents = int(audit_state["kept_documents"])
        exact_counter = Counter(audit_state.get("exact_paragraph_counter", {}))
        normalized_counter = Counter(audit_state.get("normalized_paragraph_counter", {}))
        near_duplicate_cluster_count = _cluster_count(normalized_counter)
        largest_near_duplicate_cluster_size = _largest_cluster_size(normalized_counter)
        near_duplicate_documents = sum(
            int(count) for count in normalized_counter.values() if int(count) > 1
        )
        near_duplicate_ratio = near_duplicate_documents / max(kept_documents, 1)
        phrase_counter = Counter(audit_state["phrase_counter"])
        phrase_counter_8 = Counter(audit_state.get("phrase_counter_8", {}))
        phrase_counter_12 = Counter(audit_state.get("phrase_counter_12", {}))
        phrase_counter_20 = Counter(audit_state.get("phrase_counter_20", {}))
        quality_filter_mode = str(audit_state.get("quality_filter_mode", "basic"))
        quality_artifact_occurrences = Counter(audit_state.get("quality_artifact_occurrence_counter", {}))
        quality_diagnostic_token_count = int(audit_state.get("quality_diagnostic_token_count", 0))
        medical_body_density = _density(
            int(quality_artifact_occurrences.get("medical_body_health", 0)),
            quality_diagnostic_token_count,
        )
        navigation_text_density = _density(
            int(quality_artifact_occurrences.get("navigation_like_text", 0)),
            quality_diagnostic_token_count,
        )
        malformed_fragment_density = _density(
            int(quality_artifact_occurrences.get("excessive_hyphen_fragments", 0))
            + int(quality_artifact_occurrences.get("broken_quote_fragments", 0)),
            quality_diagnostic_token_count,
        )
        generic_article_formula_density = _density(
            int(quality_artifact_occurrences.get("generic_article_formula", 0)),
            quality_diagnostic_token_count,
        )
        product_commercial_density = _density(
            int(quality_artifact_occurrences.get("product_commercial", 0)),
            quality_diagnostic_token_count,
        )
        dictionary_fragment_density = _density(
            int(quality_artifact_occurrences.get("dictionary_fragment", 0)),
            quality_diagnostic_token_count,
        )
        page_boilerplate_density = _density(
            int(quality_artifact_occurrences.get("page_boilerplate", 0)),
            quality_diagnostic_token_count,
        )
        document_word_count = int(audit_state.get("document_word_count", 0))
        document_sentence_count = int(audit_state.get("document_sentence_count", 0))
        document_paragraph_count = int(audit_state.get("document_paragraph_count", 0))
        document_char_count = int(audit_state.get("document_char_count", 0))
        broad_source_junk_score = round(
            min(
                1.0,
                medical_body_density
                + (2.0 * navigation_text_density)
                + (4.0 * malformed_fragment_density)
                + (2.0 * generic_article_formula_density),
            ),
            6,
        )
        return {
            "source": str(audit_state["source"]),
            "family": str(audit_state["family"]),
            "weight": float(audit_state["weight"]),
            "target_share": round(float(audit_state.get("target_share", 0.0)), 6),
            "quality_filter_mode": quality_filter_mode,
            "is_broad_lm_source": quality_filter_mode in {"broad_lm", "curated_lm"},
            "raw_records": int(audit_state["raw_records"]),
            "kept_documents": kept_documents,
            "kept_tokens": int(audit_state["kept_tokens"]),
            "repeated_documents": int(audit_state["repeated_documents"]),
            "restart_count": int(audit_state.get("restart_count", 0)),
            "repeat_rate": round(
                int(audit_state["repeated_documents"]) / max(kept_documents, 1),
                6,
            ),
            "dropped_reasons": _counter_to_sorted_dict(audit_state["dropped_reasons"]),
            "quality_artifact_counts": _counter_to_sorted_dict(
                Counter(audit_state.get("quality_artifact_counter", {}))
            ),
            "quality_artifact_occurrence_counts": _counter_to_sorted_dict(quality_artifact_occurrences),
            "quality_diagnostic_token_count": quality_diagnostic_token_count,
            "broad_source_junk_score": broad_source_junk_score,
            "medical_body_density": medical_body_density,
            "navigation_text_density": navigation_text_density,
            "malformed_fragment_density": malformed_fragment_density,
            "generic_article_formula_density": generic_article_formula_density,
            "product_commercial_density": product_commercial_density,
            "dictionary_fragment_density": dictionary_fragment_density,
            "page_boilerplate_density": page_boilerplate_density,
            "avg_document_chars": round(document_char_count / max(kept_documents, 1), 2),
            "avg_document_words": round(document_word_count / max(kept_documents, 1), 2),
            "avg_document_tokens": round(int(audit_state["kept_tokens"]) / max(kept_documents, 1), 2),
            "avg_sentences_per_document": round(document_sentence_count / max(kept_documents, 1), 2),
            "avg_paragraphs_per_document": round(document_paragraph_count / max(kept_documents, 1), 2),
            "avg_sentence_words": round(document_word_count / max(document_sentence_count, 1), 2),
            "synthetic_meta_phrase_counts": _counter_to_sorted_dict(
                Counter(audit_state.get("synthetic_meta_phrase_counter", {}))
            ),
            "synthetic_meta_phrase_count": int(
                sum(Counter(audit_state.get("synthetic_meta_phrase_counter", {})).values())
            ),
            "exact_paragraph_duplicate_count": int(_duplicate_count(exact_counter)),
            "normalized_paragraph_duplicate_count": int(_duplicate_count(normalized_counter)),
            "near_duplicate_cluster_count": int(near_duplicate_cluster_count),
            "largest_near_duplicate_cluster_size": int(largest_near_duplicate_cluster_size),
            "near_duplicate_ratio": round(near_duplicate_ratio, 6),
            "near_duplicate_document_count": int(near_duplicate_documents),
            "repeated_4gram_counts": dict(phrase_counter.most_common(10)),
            "repeated_8gram_counts": dict(phrase_counter_8.most_common(10)),
            "repeated_12gram_counts": dict(phrase_counter_12.most_common(10)),
            "repeated_20gram_counts": dict(phrase_counter_20.most_common(10)),
            "top_repeated_phrases": _top_counter_rows(phrase_counter, limit=5),
            "top_repeated_8grams": _top_counter_rows(phrase_counter_8, limit=5),
            "top_repeated_12grams": _top_counter_rows(phrase_counter_12, limit=5),
            "top_repeated_20grams": _top_counter_rows(phrase_counter_20, limit=5),
        }

    def _effective_target_token_allocations(
        self,
        source_reports: list[dict[str, Any]],
        *,
        total_tokens: int,
    ) -> list[float]:
        if total_tokens <= 0 or not source_reports:
            return [0.0 for _ in source_reports]
        configured_targets = [max(float(report.get("target_share", 0.0)), 0.0) for report in source_reports]
        capacities = [
            float(
                self._realizable_lm_token_capacity(
                    kept_documents=int(report.get("kept_documents", 0)),
                    kept_tokens=int(report.get("kept_tokens", 0)),
                    repeated_documents=int(report.get("repeated_documents", 0)),
                )
            )
            for report in source_reports
        ]
        allocations = [0.0 for _ in source_reports]
        remaining = set(range(len(source_reports)))
        remaining_tokens = float(total_tokens)
        while remaining and remaining_tokens > 0.0:
            total_target = sum(configured_targets[index] for index in remaining)
            if total_target <= 0.0:
                break
            capped_any = False
            for index in list(remaining):
                desired_tokens = remaining_tokens * configured_targets[index] / total_target
                if desired_tokens > capacities[index] + 1e-8:
                    allocations[index] = capacities[index]
                    remaining_tokens -= capacities[index]
                    remaining.remove(index)
                    capped_any = True
            if not capped_any:
                for index in remaining:
                    allocations[index] = remaining_tokens * configured_targets[index] / total_target
                remaining.clear()
        if remaining and remaining_tokens > 0.0:
            equal_share = remaining_tokens / max(len(remaining), 1)
            for index in remaining:
                allocations[index] = min(equal_share, capacities[index])
        return allocations

    def _finalize_lm_source_reports(
        self,
        source_reports: list[dict[str, Any]],
        *,
        total_tokens: int,
        total_documents: int,
    ) -> None:
        effective_target_tokens = self._effective_target_token_allocations(
            source_reports,
            total_tokens=total_tokens,
        )
        for index, report in enumerate(source_reports):
            report["token_share"] = round(_token_share(int(report["kept_tokens"]), total_tokens), 6)
            report["document_share"] = round(_token_share(int(report["kept_documents"]), total_documents), 6)
            report["share_gap"] = round(float(report["target_share"]) - float(report["token_share"]), 6)
            realizable_capacity = self._realizable_lm_token_capacity(
                kept_documents=int(report.get("kept_documents", 0)),
                kept_tokens=int(report.get("kept_tokens", 0)),
                repeated_documents=int(report.get("repeated_documents", 0)),
            )
            effective_target_share = _token_share(int(round(effective_target_tokens[index])), total_tokens)
            configured_target_tokens = float(report["target_share"]) * max(total_tokens, 0)
            report["effective_target_share"] = round(effective_target_share, 6)
            report["effective_share_gap"] = round(
                float(report["effective_target_share"]) - float(report["token_share"]),
                6,
            )
            report["realizable_token_capacity"] = int(realizable_capacity)
            report["realizability_limited"] = realizable_capacity + 1e-8 < configured_target_tokens

    def _is_local_mvp_domain_source_report(self, report: dict[str, Any]) -> bool:
        source = str(report.get("source", ""))
        family = str(report.get("family", ""))
        return (
            source in LOCAL_MVP_PRETRAIN_DOMAIN_SOURCE_NAMES
            or family in LOCAL_MVP_PRETRAIN_DOMAIN_FAMILIES
        )

    def _is_local_mvp_near_duplicate_gated_source_report(self, report: dict[str, Any]) -> bool:
        source = str(report.get("source", ""))
        return source in LOCAL_MVP_PRETRAIN_NEAR_DUPLICATE_GATED_SOURCE_NAMES

    def _max_repeated_ngram_count_for_report(self, report: dict[str, Any], key: str) -> int:
        return max(
            (int(count) for count in dict(report.get(key, {})).values()),
            default=0,
        )

    def _template_family_dominance_for_report(self, report: dict[str, Any]) -> float:
        kept_documents = int(report.get("kept_documents", 0))
        if kept_documents < LOCAL_MVP_PRETRAIN_TEMPLATE_FAMILY_DOMINANCE_MIN_DOCUMENTS:
            return 0.0
        max_repeated_count = max(
            self._max_repeated_ngram_count_for_report(report, "repeated_8gram_counts"),
            self._max_repeated_ngram_count_for_report(report, "repeated_12gram_counts"),
            self._max_repeated_ngram_count_for_report(report, "repeated_20gram_counts"),
        )
        return max_repeated_count / max(kept_documents, 1)

    def _pretrain_domain_contribution_summary(
        self,
        source_reports: list[dict[str, Any]],
        *,
        total_tokens: int,
        total_documents: int,
    ) -> dict[str, Any]:
        domain_reports = [
            report for report in source_reports if self._is_local_mvp_domain_source_report(report)
        ]
        domain_tokens = sum(int(report.get("kept_tokens", 0)) for report in domain_reports)
        domain_documents = sum(int(report.get("kept_documents", 0)) for report in domain_reports)
        token_share = _token_share(domain_tokens, total_tokens)
        document_share = _token_share(domain_documents, total_documents)
        configured_target_share = sum(float(report.get("target_share", 0.0)) for report in domain_reports)
        effective_target_share = sum(float(report.get("effective_target_share", 0.0)) for report in domain_reports)
        family_tokens: Counter[str] = Counter()
        for report in domain_reports:
            family_tokens[str(report.get("family", ""))] += int(report.get("kept_tokens", 0))
        failures: list[str] = []
        if domain_tokens < LOCAL_MVP_PRETRAIN_MIN_DOMAIN_TOKENS:
            failures.append("domain_tokens_too_low")
        if token_share < LOCAL_MVP_PRETRAIN_MIN_DOMAIN_TOKEN_SHARE:
            failures.append("domain_share_too_low")
        realization_ratio = token_share / max(configured_target_share, 1e-8)
        if (
            configured_target_share >= LOCAL_MVP_PRETRAIN_MIN_DOMAIN_TOKEN_SHARE
            and realization_ratio < LOCAL_MVP_PRETRAIN_MIN_DOMAIN_REALIZATION_RATIO
        ):
            failures.append("domain_realization_ratio_too_low")
        domain_readiness_expected = not failures
        return {
            "passed": domain_readiness_expected,
            "domain_readiness_expected": domain_readiness_expected,
            "severity": "pass" if domain_readiness_expected else "warning",
            "failures": failures,
            "minimum_token_share": LOCAL_MVP_PRETRAIN_MIN_DOMAIN_TOKEN_SHARE,
            "minimum_tokens": LOCAL_MVP_PRETRAIN_MIN_DOMAIN_TOKENS,
            "minimum_domain_realization_ratio": LOCAL_MVP_PRETRAIN_MIN_DOMAIN_REALIZATION_RATIO,
            "minimum_recommended_domain_tokens_for_profile": LOCAL_MVP_PRETRAIN_MIN_DOMAIN_TOKENS,
            "domain_tokens": domain_tokens,
            "domain_documents": domain_documents,
            "token_share": round(token_share, 6),
            "document_share": round(document_share, 6),
            "configured_domain_share": round(configured_target_share, 6),
            "realized_domain_share": round(token_share, 6),
            "domain_realization_ratio": round(realization_ratio, 6),
            "configured_target_share": round(configured_target_share, 6),
            "configured_share_gap": round(configured_target_share - token_share, 6),
            "effective_target_share": round(effective_target_share, 6),
            "effective_share_gap": round(effective_target_share - token_share, 6),
            "source_count": len(domain_reports),
            "source_names": [str(report.get("source", "")) for report in domain_reports],
            "family_token_share": {
                family: round(_token_share(tokens, total_tokens), 6)
                for family, tokens in sorted(family_tokens.items())
            },
            "realizability_limited_sources": [
                str(report.get("source", ""))
                for report in domain_reports
                if bool(report.get("realizability_limited", False))
            ],
            "all_domain_sources_realizability_limited": bool(domain_reports)
            and all(bool(report.get("realizability_limited", False)) for report in domain_reports),
            "domain_sources_realizability_limited": [
                str(report.get("source", ""))
                for report in domain_reports
                if bool(report.get("realizability_limited", False))
            ],
        }

    def _pretrain_domain_realization_gate(self, contribution: dict[str, Any]) -> dict[str, Any]:
        failures = list(contribution.get("failures", []))
        mode = str(self.config.pretrain_domain_realization_gate_mode)
        informational = mode in {"off", "informational"}
        passed = True if informational else not failures
        message = "pretrain domain realization passed"
        if not passed:
            message = (
                "pretrain domain realization failed: "
                f"{', '.join(failures)} "
                f"(domain_tokens={contribution.get('domain_tokens')}, "
                f"token_share={contribution.get('token_share')}, "
                f"configured_domain_share={contribution.get('configured_domain_share')}, "
                f"domain_realization_ratio={contribution.get('domain_realization_ratio')}, "
                f"minimum_tokens={contribution.get('minimum_tokens')}, "
                f"minimum_token_share={contribution.get('minimum_token_share')}, "
                f"minimum_domain_realization_ratio={contribution.get('minimum_domain_realization_ratio')})"
            )
        elif informational and failures:
            message = (
                "pretrain domain realization tracked for information only: "
                f"{', '.join(failures)} "
                f"(domain_tokens={contribution.get('domain_tokens')}, "
                f"token_share={contribution.get('token_share')}, "
                f"configured_domain_share={contribution.get('configured_domain_share')})"
            )
        return {
            "passed": passed,
            "domain_readiness_expected": bool(contribution.get("domain_readiness_expected", passed)),
            "mode": mode,
            "severity": (
                "informational"
                if informational and failures
                else ("pass" if passed else ("error" if mode == "fail" else "warning"))
            ),
            "failures": failures,
            "message": message,
            "domain_tokens": contribution.get("domain_tokens"),
            "token_share": contribution.get("token_share"),
            "configured_domain_share": contribution.get("configured_domain_share"),
            "realized_domain_share": contribution.get("realized_domain_share"),
            "domain_realization_ratio": contribution.get("domain_realization_ratio"),
            "minimum_tokens": contribution.get("minimum_tokens"),
            "minimum_token_share": contribution.get("minimum_token_share"),
            "minimum_domain_realization_ratio": contribution.get("minimum_domain_realization_ratio"),
        }

    def _pretrain_corpus_quality_gate(self, source_reports: list[dict[str, Any]]) -> dict[str, Any]:
        failures: list[str] = []
        domain_reports = [
            report for report in source_reports if self._is_local_mvp_domain_source_report(report)
        ]
        synthetic_meta_phrase_count = sum(
            int(report.get("synthetic_meta_phrase_count", 0))
            for report in source_reports
        )
        synthetic_meta_phrase_gate_counts_by_source: dict[str, dict[str, int]] = {}
        domain_synthetic_meta_phrase_count = 0
        strict_broad_synthetic_meta_phrase_count = 0
        for report in source_reports:
            source = str(report.get("source", ""))
            phrase_counts = {
                str(phrase): int(count)
                for phrase, count in dict(report.get("synthetic_meta_phrase_counts", {})).items()
                if int(count) > 0
            }
            if not phrase_counts:
                continue
            if self._is_local_mvp_domain_source_report(report):
                gated_counts = phrase_counts
                domain_synthetic_meta_phrase_count += sum(gated_counts.values())
            else:
                gated_counts = {
                    phrase: count
                    for phrase, count in phrase_counts.items()
                    if phrase not in LOCAL_MVP_PRETRAIN_BROAD_SYNTHETIC_META_ALLOWED_PHRASES
                }
                strict_broad_synthetic_meta_phrase_count += sum(gated_counts.values())
            if gated_counts:
                synthetic_meta_phrase_gate_counts_by_source[source] = gated_counts

        synthetic_meta_phrase_gate_count = (
            domain_synthetic_meta_phrase_count + strict_broad_synthetic_meta_phrase_count
        )
        if synthetic_meta_phrase_gate_count > 0:
            failures.append("synthetic_meta_phrase_count_nonzero")

        repeated_ngram_counts_by_source = {
            str(report.get("source", "")): {
                "max_repeated_4gram_count": self._max_repeated_ngram_count_for_report(
                    report, "repeated_4gram_counts"
                ),
                "max_repeated_8gram_count": self._max_repeated_ngram_count_for_report(
                    report, "repeated_8gram_counts"
                ),
                "max_repeated_12gram_count": self._max_repeated_ngram_count_for_report(
                    report, "repeated_12gram_counts"
                ),
                "max_repeated_20gram_count": self._max_repeated_ngram_count_for_report(
                    report, "repeated_20gram_counts"
                ),
            }
            for report in source_reports
        }
        max_repeated_4gram_count = max(
            (
                counts["max_repeated_4gram_count"]
                for counts in repeated_ngram_counts_by_source.values()
            ),
            default=0,
        )
        if max_repeated_4gram_count > LOCAL_MVP_PRETRAIN_MAX_REPEATED_4GRAM_COUNT:
            failures.append("repeated_4gram_count_above_limit")
        max_repeated_8gram_count = max(
            (
                counts["max_repeated_8gram_count"]
                for counts in repeated_ngram_counts_by_source.values()
            ),
            default=0,
        )
        if max_repeated_8gram_count > LOCAL_MVP_PRETRAIN_MAX_REPEATED_8GRAM_COUNT:
            failures.append("repeated_8gram_count_above_limit")
        max_repeated_12gram_count = max(
            (
                counts["max_repeated_12gram_count"]
                for counts in repeated_ngram_counts_by_source.values()
            ),
            default=0,
        )
        if max_repeated_12gram_count > LOCAL_MVP_PRETRAIN_MAX_REPEATED_12GRAM_COUNT:
            failures.append("repeated_12gram_count_above_limit")
        template_family_dominance_by_source = {
            str(report.get("source", "")): round(self._template_family_dominance_for_report(report), 6)
            for report in source_reports
        }
        template_family_dominance_sources = [
            source
            for source, share in sorted(template_family_dominance_by_source.items())
            if share > LOCAL_MVP_PRETRAIN_MAX_TEMPLATE_FAMILY_DOMINANCE_SHARE
        ]
        max_template_family_dominance_share = max(
            template_family_dominance_by_source.values(),
            default=0.0,
        )
        if template_family_dominance_sources:
            failures.append("template_family_dominance_above_limit")

        max_near_duplicate_ratio = max(
            (float(report.get("near_duplicate_ratio", 0.0)) for report in source_reports),
            default=0.0,
        )
        max_domain_near_duplicate_ratio = max(
            (float(report.get("near_duplicate_ratio", 0.0)) for report in domain_reports),
            default=0.0,
        )
        near_duplicate_gated_reports = [
            report
            for report in source_reports
            if self._is_local_mvp_near_duplicate_gated_source_report(report)
        ]
        max_gated_domain_near_duplicate_ratio = max(
            (
                float(report.get("near_duplicate_ratio", 0.0))
                for report in near_duplicate_gated_reports
            ),
            default=0.0,
        )
        if max_gated_domain_near_duplicate_ratio > LOCAL_MVP_PRETRAIN_MAX_NEAR_DUPLICATE_RATIO:
            failures.append("near_duplicate_ratio_above_limit")
        max_domain_repeated_20gram_count = max(
            (
                self._max_repeated_ngram_count_for_report(report, "repeated_20gram_counts")
                for report in domain_reports
            ),
            default=0,
        )
        if max_domain_repeated_20gram_count > LOCAL_MVP_PRETRAIN_MAX_DOMAIN_REPEATED_20GRAM_COUNT:
            failures.append("domain_repeated_20gram_count_above_limit")

        mode = str(self.config.pretrain_domain_realization_gate_mode)
        passed = not failures
        message = "pretrain corpus quality gate passed"
        if not passed:
            message = (
                "pretrain corpus quality gate failed: "
                f"{', '.join(failures)} "
                f"(synthetic_meta_phrase_gate_count={synthetic_meta_phrase_gate_count}, "
                f"domain_synthetic_meta_phrase_count={domain_synthetic_meta_phrase_count}, "
                f"max_repeated_4gram_count={max_repeated_4gram_count}, "
                f"max_repeated_8gram_count={max_repeated_8gram_count}, "
                f"max_repeated_12gram_count={max_repeated_12gram_count}, "
                f"max_template_family_dominance_share={round(max_template_family_dominance_share, 6)}, "
                f"max_gated_domain_near_duplicate_ratio={round(max_gated_domain_near_duplicate_ratio, 6)}, "
                f"max_domain_repeated_20gram_count={max_domain_repeated_20gram_count})"
            )
        return {
            "passed": passed,
            "mode": mode,
            "severity": "pass" if passed else ("error" if mode == "fail" else "warning"),
            "failures": failures,
            "message": message,
            "thresholds": {
                "max_repeated_4gram_count": LOCAL_MVP_PRETRAIN_MAX_REPEATED_4GRAM_COUNT,
                "max_repeated_8gram_count": LOCAL_MVP_PRETRAIN_MAX_REPEATED_8GRAM_COUNT,
                "max_repeated_12gram_count": LOCAL_MVP_PRETRAIN_MAX_REPEATED_12GRAM_COUNT,
                "max_domain_repeated_20gram_count": LOCAL_MVP_PRETRAIN_MAX_DOMAIN_REPEATED_20GRAM_COUNT,
                "max_near_duplicate_ratio": LOCAL_MVP_PRETRAIN_MAX_NEAR_DUPLICATE_RATIO,
                "max_template_family_dominance_share": (
                    LOCAL_MVP_PRETRAIN_MAX_TEMPLATE_FAMILY_DOMINANCE_SHARE
                ),
                "template_family_dominance_min_documents": (
                    LOCAL_MVP_PRETRAIN_TEMPLATE_FAMILY_DOMINANCE_MIN_DOCUMENTS
                ),
                "max_synthetic_meta_phrase_count": 0,
            },
            "synthetic_meta_phrase_count": synthetic_meta_phrase_count,
            "synthetic_meta_phrase_gate_count": int(synthetic_meta_phrase_gate_count),
            "domain_synthetic_meta_phrase_count": int(domain_synthetic_meta_phrase_count),
            "strict_broad_synthetic_meta_phrase_count": int(strict_broad_synthetic_meta_phrase_count),
            "synthetic_meta_phrase_gate_counts_by_source": synthetic_meta_phrase_gate_counts_by_source,
            "repeated_ngram_counts_by_source": repeated_ngram_counts_by_source,
            "max_repeated_4gram_count": int(max_repeated_4gram_count),
            "max_repeated_8gram_count": int(max_repeated_8gram_count),
            "max_repeated_12gram_count": int(max_repeated_12gram_count),
            "template_family_dominance_by_source": template_family_dominance_by_source,
            "template_family_dominance_sources": template_family_dominance_sources,
            "max_template_family_dominance_share": round(
                max_template_family_dominance_share,
                6,
            ),
            "max_near_duplicate_ratio": round(max_near_duplicate_ratio, 6),
            "max_domain_near_duplicate_ratio": round(max_domain_near_duplicate_ratio, 6),
            "max_gated_domain_near_duplicate_ratio": round(
                max_gated_domain_near_duplicate_ratio,
                6,
            ),
            "near_duplicate_gated_sources": [
                str(report.get("source", "")) for report in near_duplicate_gated_reports
            ],
            "max_domain_repeated_20gram_count": int(max_domain_repeated_20gram_count),
        }

    def _pretrain_broad_source_quality_gate(self, source_reports: list[dict[str, Any]]) -> dict[str, Any]:
        broad_reports = [
            report for report in source_reports if bool(report.get("is_broad_lm_source", False))
        ]
        thresholds = {
            "max_broad_source_junk_score": float(self.config.pretrain_broad_max_junk_score),
            "max_medical_body_density": float(self.config.pretrain_broad_max_medical_body_density),
            "max_navigation_text_density": float(self.config.pretrain_broad_max_navigation_text_density),
            "max_malformed_fragment_density": float(
                self.config.pretrain_broad_max_malformed_fragment_density
            ),
            "max_generic_article_formula_density": float(
                self.config.pretrain_broad_max_generic_article_formula_density
            ),
            "max_product_commercial_density": float(
                self.config.pretrain_curated_max_product_commercial_density
            ),
            "max_dictionary_fragment_density": float(
                self.config.pretrain_curated_max_dictionary_fragment_density
            ),
            "max_page_boilerplate_density": float(
                self.config.pretrain_curated_max_page_boilerplate_density
            ),
        }
        per_source_failures: list[dict[str, Any]] = []
        failure_names: set[str] = set()
        for report in broad_reports:
            source_failures: list[str] = []
            if float(report.get("broad_source_junk_score", 0.0)) > thresholds["max_broad_source_junk_score"]:
                source_failures.append("broad_source_junk_score_above_limit")
            if float(report.get("medical_body_density", 0.0)) > thresholds["max_medical_body_density"]:
                source_failures.append("medical_body_density_above_limit")
            if float(report.get("navigation_text_density", 0.0)) > thresholds["max_navigation_text_density"]:
                source_failures.append("navigation_text_density_above_limit")
            if float(report.get("malformed_fragment_density", 0.0)) > thresholds["max_malformed_fragment_density"]:
                source_failures.append("malformed_fragment_density_above_limit")
            if (
                float(report.get("generic_article_formula_density", 0.0))
                > thresholds["max_generic_article_formula_density"]
            ):
                source_failures.append("generic_article_formula_density_above_limit")
            if (
                float(report.get("product_commercial_density", 0.0))
                > thresholds["max_product_commercial_density"]
            ):
                source_failures.append("product_commercial_density_above_limit")
            if (
                float(report.get("dictionary_fragment_density", 0.0))
                > thresholds["max_dictionary_fragment_density"]
            ):
                source_failures.append("dictionary_fragment_density_above_limit")
            if (
                float(report.get("page_boilerplate_density", 0.0))
                > thresholds["max_page_boilerplate_density"]
            ):
                source_failures.append("page_boilerplate_density_above_limit")
            if source_failures:
                failure_names.update(source_failures)
                per_source_failures.append(
                    {
                        "source": str(report.get("source", "")),
                        "failures": source_failures,
                        "broad_source_junk_score": float(report.get("broad_source_junk_score", 0.0)),
                        "medical_body_density": float(report.get("medical_body_density", 0.0)),
                        "navigation_text_density": float(report.get("navigation_text_density", 0.0)),
                        "malformed_fragment_density": float(report.get("malformed_fragment_density", 0.0)),
                        "generic_article_formula_density": float(
                            report.get("generic_article_formula_density", 0.0)
                        ),
                        "product_commercial_density": float(
                            report.get("product_commercial_density", 0.0)
                        ),
                        "dictionary_fragment_density": float(
                            report.get("dictionary_fragment_density", 0.0)
                        ),
                        "page_boilerplate_density": float(
                            report.get("page_boilerplate_density", 0.0)
                        ),
                        "quality_artifact_occurrence_counts": dict(
                            report.get("quality_artifact_occurrence_counts", {})
                        ),
                    }
                )

        mode = str(self.config.pretrain_broad_source_quality_gate_mode)
        passed = not per_source_failures
        max_scores = {
            "max_broad_source_junk_score": round(
                max((float(report.get("broad_source_junk_score", 0.0)) for report in broad_reports), default=0.0),
                6,
            ),
            "max_medical_body_density": round(
                max((float(report.get("medical_body_density", 0.0)) for report in broad_reports), default=0.0),
                6,
            ),
            "max_navigation_text_density": round(
                max((float(report.get("navigation_text_density", 0.0)) for report in broad_reports), default=0.0),
                6,
            ),
            "max_malformed_fragment_density": round(
                max((float(report.get("malformed_fragment_density", 0.0)) for report in broad_reports), default=0.0),
                6,
            ),
            "max_generic_article_formula_density": round(
                max(
                    (float(report.get("generic_article_formula_density", 0.0)) for report in broad_reports),
                    default=0.0,
                ),
                6,
            ),
            "max_product_commercial_density": round(
                max((float(report.get("product_commercial_density", 0.0)) for report in broad_reports), default=0.0),
                6,
            ),
            "max_dictionary_fragment_density": round(
                max((float(report.get("dictionary_fragment_density", 0.0)) for report in broad_reports), default=0.0),
                6,
            ),
            "max_page_boilerplate_density": round(
                max((float(report.get("page_boilerplate_density", 0.0)) for report in broad_reports), default=0.0),
                6,
            ),
        }
        message = "pretrain broad source quality gate passed"
        if not passed:
            failing_sources = ", ".join(row["source"] for row in per_source_failures)
            message = (
                "pretrain broad source quality gate failed: "
                f"{', '.join(sorted(failure_names))} "
                f"(sources={failing_sources}, thresholds={thresholds})"
            )
        return {
            "passed": passed,
            "mode": mode,
            "severity": "pass" if passed else ("error" if mode == "fail" else "warning"),
            "failures": sorted(failure_names),
            "message": message,
            "thresholds": thresholds,
            "source_count": len(broad_reports),
            "failing_source_count": len(per_source_failures),
            "per_source_failures": per_source_failures,
            **max_scores,
        }

    def _lm_source_diagnostics(
        self,
        source_audit_states: list[dict[str, Any]],
        *,
        total_tokens: int,
        total_documents: int,
        stage: str | None = None,
    ) -> dict[str, Any]:
        source_reports: list[dict[str, Any]] = []
        family_counts: Counter[str] = Counter()
        phrase_counter: Counter[str] = Counter()
        phrase_counter_8: Counter[str] = Counter()
        phrase_counter_12: Counter[str] = Counter()
        phrase_counter_20: Counter[str] = Counter()
        quality_artifact_counter: Counter[str] = Counter()
        quality_artifact_occurrence_counter: Counter[str] = Counter()
        synthetic_meta_phrase_counter: Counter[str] = Counter()
        exact_paragraph_duplicate_count = 0
        normalized_paragraph_duplicate_count = 0
        near_duplicate_cluster_count = 0
        largest_near_duplicate_cluster_size = 0
        near_duplicate_ratio_by_source: dict[str, float] = {}
        near_duplicate_ratio_by_source_family: dict[str, float] = {}
        synthetic_meta_phrase_counter_by_source_family: dict[str, Counter[str]] = {}
        for audit_state in source_audit_states:
            kept_documents = int(audit_state["kept_documents"])
            kept_tokens = int(audit_state["kept_tokens"])
            if kept_documents > 0:
                family_counts[str(audit_state["family"])] += 1
            phrase_counter.update(audit_state["phrase_counter"])
            phrase_counter_8.update(audit_state.get("phrase_counter_8", Counter()))
            phrase_counter_12.update(audit_state.get("phrase_counter_12", Counter()))
            phrase_counter_20.update(audit_state.get("phrase_counter_20", Counter()))
            quality_artifact_counter.update(audit_state.get("quality_artifact_counter", Counter()))
            quality_artifact_occurrence_counter.update(
                audit_state.get("quality_artifact_occurrence_counter", Counter())
            )
            synthetic_meta_phrase_counter.update(audit_state.get("synthetic_meta_phrase_counter", Counter()))
            report = self._lm_source_report_from_audit_state(audit_state)
            exact_paragraph_duplicate_count += int(report["exact_paragraph_duplicate_count"])
            normalized_paragraph_duplicate_count += int(report["normalized_paragraph_duplicate_count"])
            near_duplicate_cluster_count += int(report["near_duplicate_cluster_count"])
            largest_near_duplicate_cluster_size = max(
                largest_near_duplicate_cluster_size,
                int(report["largest_near_duplicate_cluster_size"]),
            )
            near_duplicate_ratio_by_source[str(report["source"])] = float(report["near_duplicate_ratio"])
            family = str(report.get("family", "unknown"))
            near_duplicate_ratio_by_source_family[family] = max(
                near_duplicate_ratio_by_source_family.get(family, 0.0),
                float(report["near_duplicate_ratio"]),
            )
            synthetic_meta_phrase_counter_by_source_family.setdefault(family, Counter()).update(
                dict(report.get("synthetic_meta_phrase_counts", {}))
            )
            source_reports.append(report)
        self._finalize_lm_source_reports(
            source_reports,
            total_tokens=total_tokens,
            total_documents=total_documents,
        )
        diagnostics = {
            "per_source": source_reports,
            "source_family_count": sum(1 for count in family_counts.values() if count > 0),
            "max_single_source_token_share": round(
                max((float(report["token_share"]) for report in source_reports), default=0.0),
                6,
            ),
            "max_repeat_rate": round(
                max((float(report["repeat_rate"]) for report in source_reports), default=0.0),
                6,
            ),
            "top_repeated_phrases": [
                {"phrase": phrase, "count": count}
                for phrase, count in phrase_counter.most_common(10)
            ],
            "top_repeated_8grams": _top_counter_rows(phrase_counter_8, limit=10),
            "top_repeated_12grams": _top_counter_rows(phrase_counter_12, limit=10),
            "top_repeated_20grams": _top_counter_rows(phrase_counter_20, limit=10),
            "repeated_4gram_counts": dict(phrase_counter.most_common(20)),
            "repeated_8gram_counts": dict(phrase_counter_8.most_common(20)),
            "repeated_12gram_counts": dict(phrase_counter_12.most_common(20)),
            "repeated_20gram_counts": dict(phrase_counter_20.most_common(20)),
            "quality_artifact_counts": _counter_to_sorted_dict(quality_artifact_counter),
            "quality_artifact_occurrence_counts": _counter_to_sorted_dict(
                quality_artifact_occurrence_counter
            ),
            "broad_source_quality_scores_by_source": {
                str(report["source"]): {
                    "broad_source_junk_score": float(report.get("broad_source_junk_score", 0.0)),
                    "medical_body_density": float(report.get("medical_body_density", 0.0)),
                    "navigation_text_density": float(report.get("navigation_text_density", 0.0)),
                    "malformed_fragment_density": float(report.get("malformed_fragment_density", 0.0)),
                    "generic_article_formula_density": float(
                        report.get("generic_article_formula_density", 0.0)
                    ),
                    "product_commercial_density": float(report.get("product_commercial_density", 0.0)),
                    "dictionary_fragment_density": float(report.get("dictionary_fragment_density", 0.0)),
                    "page_boilerplate_density": float(report.get("page_boilerplate_density", 0.0)),
                }
                for report in source_reports
                if bool(report.get("is_broad_lm_source", False))
            },
            "synthetic_meta_phrase_counts_by_source": {
                str(report["source"]): dict(report.get("synthetic_meta_phrase_counts", {}))
                for report in source_reports
                if int(report.get("synthetic_meta_phrase_count", 0)) > 0
            },
            "synthetic_meta_phrase_counts_by_source_family": {
                family: _counter_to_sorted_dict(counter)
                for family, counter in sorted(synthetic_meta_phrase_counter_by_source_family.items())
                if sum(counter.values()) > 0
            },
            "synthetic_meta_phrase_counts": _counter_to_sorted_dict(synthetic_meta_phrase_counter),
            "synthetic_meta_phrase_count": int(sum(synthetic_meta_phrase_counter.values())),
            "exact_paragraph_duplicate_count": int(exact_paragraph_duplicate_count),
            "normalized_paragraph_duplicate_count": int(normalized_paragraph_duplicate_count),
            "near_duplicate_cluster_count": int(near_duplicate_cluster_count),
            "largest_near_duplicate_cluster_size": int(largest_near_duplicate_cluster_size),
            "near_duplicate_ratio_by_source": {
                source: round(ratio, 6)
                for source, ratio in sorted(near_duplicate_ratio_by_source.items())
            },
            "near_duplicate_ratio_by_source_family": {
                family: round(ratio, 6)
                for family, ratio in sorted(near_duplicate_ratio_by_source_family.items())
            },
            "repeated_paragraph_hash_count": int(
                sum(int(report["repeated_documents"]) for report in source_reports)
            ),
            "document_shape": {
                "avg_document_chars": round(
                    sum(float(report.get("avg_document_chars", 0.0)) * int(report.get("kept_documents", 0)) for report in source_reports)
                    / max(total_documents, 1),
                    2,
                ),
                "avg_document_words": round(
                    sum(float(report.get("avg_document_words", 0.0)) * int(report.get("kept_documents", 0)) for report in source_reports)
                    / max(total_documents, 1),
                    2,
                ),
                "avg_document_tokens": round(total_tokens / max(total_documents, 1), 2),
                "avg_sentences_per_document": round(
                    sum(float(report.get("avg_sentences_per_document", 0.0)) * int(report.get("kept_documents", 0)) for report in source_reports)
                    / max(total_documents, 1),
                    2,
                ),
                "avg_paragraphs_per_document": round(
                    sum(float(report.get("avg_paragraphs_per_document", 0.0)) * int(report.get("kept_documents", 0)) for report in source_reports)
                    / max(total_documents, 1),
                    2,
                ),
                "avg_sentence_words": round(
                    sum(float(report.get("avg_sentence_words", 0.0)) * int(report.get("kept_documents", 0)) for report in source_reports)
                    / max(total_documents, 1),
                    2,
                ),
            },
        }
        warnings: list[str] = []
        if diagnostics["max_single_source_token_share"] > LOCAL_MVP_PRETRAIN_MAX_GENERIC_SOURCE_SHARE:
            warnings.append("single_source_token_share_above_local_mvp_limit")
        if any(
            int(row["count"]) > LOCAL_MVP_PRETRAIN_REPEATED_PHRASE_WARN_COUNT
            for row in diagnostics["top_repeated_phrases"]
        ):
            warnings.append("repeated_phrase_count_above_local_mvp_limit")
        if any(
            int(row["count"]) > LOCAL_MVP_PRETRAIN_MAX_REPEATED_8GRAM_COUNT
            for row in diagnostics["top_repeated_8grams"]
        ):
            warnings.append("repeated_8gram_count_above_local_mvp_limit")
        if any(
            int(row["count"]) > LOCAL_MVP_PRETRAIN_MAX_REPEATED_12GRAM_COUNT
            for row in diagnostics["top_repeated_12grams"]
        ):
            warnings.append("repeated_12gram_count_above_local_mvp_limit")
        if any(
            float(self._template_family_dominance_for_report(report))
            > LOCAL_MVP_PRETRAIN_MAX_TEMPLATE_FAMILY_DOMINANCE_SHARE
            for report in source_reports
        ):
            warnings.append("template_family_dominance_above_local_mvp_limit")
        if diagnostics["synthetic_meta_phrase_count"] > 0:
            warnings.append("synthetic_meta_phrase_observed")
        if diagnostics["largest_near_duplicate_cluster_size"] > 1:
            warnings.append("near_duplicate_clusters_detected")
        if any(int(count) > 0 for count in quality_artifact_counter.values()):
            warnings.append("quality_artifacts_detected")
        diagnostics["quality_warnings"] = warnings
        if stage == "pretrain":
            domain_contribution = self._pretrain_domain_contribution_summary(
                source_reports,
                total_tokens=total_tokens,
                total_documents=total_documents,
            )
            diagnostics["domain_contribution"] = domain_contribution
            diagnostics["domain_realization_gate"] = self._pretrain_domain_realization_gate(domain_contribution)
            corpus_quality_gate = self._pretrain_corpus_quality_gate(source_reports)
            diagnostics["corpus_quality_gate"] = corpus_quality_gate
            broad_source_quality_gate = self._pretrain_broad_source_quality_gate(source_reports)
            diagnostics["broad_source_quality_gate"] = broad_source_quality_gate
            if (
                str(self.config.pretrain_domain_realization_gate_mode) not in {"off", "informational"}
                and not domain_contribution.get("passed", True)
            ):
                diagnostics["quality_warnings"] = [
                    *diagnostics["quality_warnings"],
                    "domain_readiness_not_expected",
                ]
            if not corpus_quality_gate.get("passed", True):
                diagnostics["quality_warnings"] = [
                    *diagnostics["quality_warnings"],
                    "corpus_quality_gate_failed",
                ]
            if not broad_source_quality_gate.get("passed", True):
                diagnostics["quality_warnings"] = [
                    *diagnostics["quality_warnings"],
                    "broad_source_quality_gate_failed",
                ]
        return diagnostics

    def _audit_prepared_lm_stage(
        self,
        stage: str,
        sources: list[DataSourceConfig],
    ) -> dict[str, Any]:
        source_reports: list[dict[str, Any]] = []
        total_documents = 0
        total_tokens = 0
        family_counts: Counter[str] = Counter()
        phrase_counter: Counter[str] = Counter()
        for source in sources:
            manifest = load_prepared_manifest(source.path)
            kind = manifest.get("kind")
            if kind != "packed_lm":
                raise RuntimeError(
                    f"Prepared source {source.path} has kind {kind!r}, expected 'packed_lm'."
                )
            manifest_stage = str(manifest.get("stage", ""))
            if manifest_stage != stage:
                raise RuntimeError(
                    f"Prepared source {source.path} has stage {manifest_stage!r}, expected {stage!r}."
                )
            diagnostics = manifest.get("diagnostics", {})
            prepared_reports = diagnostics.get("per_source", [])
            if not isinstance(prepared_reports, list):
                raise RuntimeError(
                    f"Prepared source {source.path} is missing per-source LM diagnostics."
                )
            for report in prepared_reports:
                kept_documents = int(report.get("kept_documents", 0))
                kept_tokens = int(report.get("kept_tokens", 0))
                repeated_documents = int(report.get("repeated_documents", 0))
                family = str(report.get("family", source.name))
                total_documents += kept_documents
                total_tokens += kept_tokens
                if kept_documents > 0:
                    family_counts[family] += 1
                source_reports.append(
                    {
                        "source": str(report.get("source", source.name)),
                        "family": family,
                        "weight": float(report.get("weight", source.weight)),
                        "target_share": round(float(report.get("target_share", 0.0)), 6),
                        "quality_filter_mode": str(report.get("quality_filter_mode", "basic")),
                        "is_broad_lm_source": bool(report.get("is_broad_lm_source", False)),
                        "raw_records": int(report.get("raw_records", 0)),
                        "kept_documents": kept_documents,
                        "kept_tokens": kept_tokens,
                        "repeated_documents": repeated_documents,
                        "restart_count": int(report.get("restart_count", 0)),
                        "repeat_rate": round(repeated_documents / max(kept_documents, 1), 6),
                        "dropped_reasons": dict(report.get("dropped_reasons", {})),
                        "quality_artifact_counts": dict(report.get("quality_artifact_counts", {})),
                        "quality_artifact_occurrence_counts": dict(
                            report.get("quality_artifact_occurrence_counts", {})
                        ),
                        "quality_diagnostic_token_count": int(
                            report.get("quality_diagnostic_token_count", 0)
                        ),
                        "broad_source_junk_score": float(report.get("broad_source_junk_score", 0.0)),
                        "medical_body_density": float(report.get("medical_body_density", 0.0)),
                        "navigation_text_density": float(report.get("navigation_text_density", 0.0)),
                        "malformed_fragment_density": float(report.get("malformed_fragment_density", 0.0)),
                        "generic_article_formula_density": float(
                            report.get("generic_article_formula_density", 0.0)
                        ),
                        "product_commercial_density": float(
                            report.get("product_commercial_density", 0.0)
                        ),
                        "dictionary_fragment_density": float(
                            report.get("dictionary_fragment_density", 0.0)
                        ),
                        "page_boilerplate_density": float(
                            report.get("page_boilerplate_density", 0.0)
                        ),
                        "avg_document_chars": float(report.get("avg_document_chars", 0.0)),
                        "avg_document_words": float(report.get("avg_document_words", 0.0)),
                        "avg_document_tokens": float(report.get("avg_document_tokens", 0.0)),
                        "avg_sentences_per_document": float(
                            report.get("avg_sentences_per_document", 0.0)
                        ),
                        "avg_paragraphs_per_document": float(
                            report.get("avg_paragraphs_per_document", 0.0)
                        ),
                        "avg_sentence_words": float(report.get("avg_sentence_words", 0.0)),
                        "synthetic_meta_phrase_counts": dict(report.get("synthetic_meta_phrase_counts", {})),
                        "synthetic_meta_phrase_count": int(report.get("synthetic_meta_phrase_count", 0)),
                        "exact_paragraph_duplicate_count": int(
                            report.get("exact_paragraph_duplicate_count", 0)
                        ),
                        "normalized_paragraph_duplicate_count": int(
                            report.get("normalized_paragraph_duplicate_count", 0)
                        ),
                        "near_duplicate_cluster_count": int(report.get("near_duplicate_cluster_count", 0)),
                        "largest_near_duplicate_cluster_size": int(
                            report.get("largest_near_duplicate_cluster_size", 0)
                        ),
                        "near_duplicate_ratio": float(report.get("near_duplicate_ratio", 0.0)),
                        "near_duplicate_document_count": int(report.get("near_duplicate_document_count", 0)),
                        "repeated_4gram_counts": dict(report.get("repeated_4gram_counts", {})),
                        "repeated_8gram_counts": dict(report.get("repeated_8gram_counts", {})),
                        "repeated_12gram_counts": dict(report.get("repeated_12gram_counts", {})),
                        "repeated_20gram_counts": dict(report.get("repeated_20gram_counts", {})),
                    }
                )
            for row in diagnostics.get("top_repeated_phrases", []):
                phrase = row.get("phrase")
                if not isinstance(phrase, str) or not phrase:
                    continue
                phrase_counter[phrase] += int(row.get("count", 0))
        self._finalize_lm_source_reports(
            source_reports,
            total_tokens=total_tokens,
            total_documents=total_documents,
        )
        max_source_share = max((float(report["token_share"]) for report in source_reports), default=0.0)
        max_repeat_rate = max((float(report["repeat_rate"]) for report in source_reports), default=0.0)
        audit = {
            "stage": stage,
            "total_documents": total_documents,
            "total_clean_tokens": total_tokens,
            "source_reports": source_reports,
            "source_family_count": sum(1 for count in family_counts.values() if count > 0),
            "max_single_source_token_share": round(max_source_share, 6),
            "max_repeat_rate": round(max_repeat_rate, 6),
            "top_repeated_phrases": [
                {"phrase": phrase, "count": count}
                for phrase, count in phrase_counter.most_common(10)
            ],
            "repeated_paragraph_hash_count": int(
                sum(int(report["repeated_documents"]) for report in source_reports)
            ),
        }
        warnings: list[str] = []
        if max_source_share > LOCAL_MVP_PRETRAIN_MAX_GENERIC_SOURCE_SHARE:
            warnings.append("single_source_token_share_above_local_mvp_limit")
        if any(count > LOCAL_MVP_PRETRAIN_REPEATED_PHRASE_WARN_COUNT for _phrase, count in phrase_counter.items()):
            warnings.append("repeated_phrase_count_above_local_mvp_limit")
        audit["quality_warnings"] = warnings
        if stage == "pretrain":
            domain_contribution = self._pretrain_domain_contribution_summary(
                source_reports,
                total_tokens=total_tokens,
                total_documents=total_documents,
            )
            audit["domain_contribution"] = domain_contribution
            audit["domain_realization_gate"] = self._pretrain_domain_realization_gate(domain_contribution)
            corpus_quality_gate = self._pretrain_corpus_quality_gate(source_reports)
            audit["corpus_quality_gate"] = corpus_quality_gate
            broad_source_quality_gate = self._pretrain_broad_source_quality_gate(source_reports)
            audit["broad_source_quality_gate"] = broad_source_quality_gate
            if (
                str(self.config.pretrain_domain_realization_gate_mode) not in {"off", "informational"}
                and not domain_contribution.get("passed", True)
            ):
                audit["quality_warnings"] = [*audit["quality_warnings"], "domain_readiness_not_expected"]
            if not corpus_quality_gate.get("passed", True):
                audit["quality_warnings"] = [*audit["quality_warnings"], "corpus_quality_gate_failed"]
            if not broad_source_quality_gate.get("passed", True):
                audit["quality_warnings"] = [
                    *audit["quality_warnings"],
                    "broad_source_quality_gate_failed",
                ]
        return audit

    def _iter_tokenized_documents_for_source(
        self,
        source: DataSourceConfig,
        *,
        tokenizer: SentencePieceTokenizer,
        seen_hashes: set[str] | None,
        raw_records_consumed: int = 0,
        audit_state: dict[str, Any] | None = None,
        progress_state: dict[str, Any] | None = None,
        allow_reentry: bool = False,
        max_kept_documents: int | None = None,
        num_workers_override: int | None = None,
    ) -> Iterable[tuple[DocumentRecord, list[int]]]:
        kept_in_iterator = 0
        source_seen_ids = (
            set(audit_state.get("unique_document_ids", set()))
            if allow_reentry and audit_state is not None
            else set()
        )
        num_workers = self._bounded_lm_document_workers(num_workers_override=num_workers_override)
        if num_workers > 1:
            result_iter = self._iter_parallel_lm_document_results(
                source,
                raw_records_consumed=raw_records_consumed,
                num_workers=num_workers,
            )
        else:
            result_iter = self._iter_serial_lm_document_results(
                source,
                tokenizer=tokenizer,
                raw_records_consumed=raw_records_consumed,
            )

        for result in result_iter:
            if result.error is not None:
                self._raise_lm_document_worker_error(source, result)
            if audit_state is not None:
                audit_state["raw_records"] += 1
            if progress_state is not None:
                progress_state["raw_records_consumed"] = int(progress_state.get("raw_records_consumed", 0)) + 1
            if not result.is_text:
                if audit_state is not None:
                    audit_state["dropped_reasons"]["non_text"] += 1
                continue
            if audit_state is not None:
                self._merge_lm_counter(
                    audit_state["synthetic_meta_phrase_counter"],
                    result.synthetic_meta_phrase_counts,
                )
            if result.dropped_reason is not None:
                if audit_state is not None:
                    audit_state["dropped_reasons"][result.dropped_reason or "dropped"] += 1
                continue

            document_id = result.document_id
            if source.deduplicate and seen_hashes is not None:
                if allow_reentry:
                    # On a restart, allow repeats from this source while still blocking
                    # duplicates that were first introduced by other sources.
                    if document_id and document_id in seen_hashes and document_id not in source_seen_ids:
                        if audit_state is not None:
                            audit_state["dropped_reasons"]["duplicate"] += 1
                        continue
                    if document_id:
                        seen_hashes.add(document_id)
                else:
                    if document_id and document_id in seen_hashes:
                        if audit_state is not None:
                            audit_state["dropped_reasons"]["duplicate"] += 1
                        continue
                    if document_id:
                        seen_hashes.add(document_id)

            token_ids = list(result.token_ids)
            if len(token_ids) <= 1:
                if audit_state is not None:
                    audit_state["dropped_reasons"]["too_short_tokenized"] += 1
                continue
            if audit_state is not None:
                self._merge_lm_accepted_audit(audit_state, result, len(token_ids))
            if progress_state is not None:
                progress_state["accepted_records"] = int(progress_state.get("accepted_records", 0)) + 1
            kept_in_iterator += 1
            yield (
                DocumentRecord(
                    text=result.text,
                    source=source.name,
                    document_id=document_id,
                    metadata=dict(result.metadata),
                ),
                token_ids,
            )
            if max_kept_documents is not None and kept_in_iterator >= max_kept_documents:
                return

    def _weighted_lm_source_iterators(
        self,
        sources: list[DataSourceConfig],
        *,
        tokenizer: SentencePieceTokenizer,
        source_progress: list[dict[str, Any]] | None = None,
        source_audits: list[dict[str, Any]] | None = None,
        seen_hashes: set[str] | None = None,
    ) -> list[dict[str, Any]]:
        states: list[dict[str, Any]] = []
        total_weight = sum(max(float(source.weight), 1e-8) for source in sources)
        workers_per_source = self._bounded_lm_document_workers(active_source_count=len(sources))
        for index, source in enumerate(sources):
            audit_state = (
                self._new_lm_audit_state(source)
                if source_audits is None
                else source_audits[index]
            )
            audit_state["target_share"] = max(float(source.weight), 1e-8) / max(total_weight, 1e-8)
            progress_state = None if source_progress is None else source_progress[index]
            state = {
                "source": source,
                "audit": audit_state,
                "progress": progress_state,
            }

            def iterator_factory(
                *,
                _source: DataSourceConfig = source,
                _audit_state: dict[str, Any] = audit_state,
                _progress_state: dict[str, Any] | None = progress_state,
                raw_records_consumed: int,
                allow_reentry: bool,
                max_kept_documents: int | None = None,
            ) -> Iterable[tuple[DocumentRecord, list[int]]]:
                return self._iter_tokenized_documents_for_source(
                    _source,
                    tokenizer=tokenizer,
                    seen_hashes=seen_hashes,
                    raw_records_consumed=raw_records_consumed,
                    audit_state=_audit_state,
                    progress_state=_progress_state,
                    allow_reentry=allow_reentry,
                    max_kept_documents=max_kept_documents,
                    num_workers_override=workers_per_source,
                )

            restart_count = 0 if progress_state is None else int(progress_state.get("restart_count", 0))
            current_raw_records = 0 if progress_state is None else int(progress_state.get("raw_records_consumed", 0))
            reentry_limit = (
                self._remaining_weighted_restart_documents(state)
                if restart_count > 0
                else None
            )
            state["iterator_factory"] = iterator_factory
            state["iterator"] = iter(
                iterator_factory(
                    raw_records_consumed=current_raw_records,
                    allow_reentry=restart_count > 0,
                    max_kept_documents=reentry_limit,
                )
            )
            states.append(state)
        return states

    def _source_hits_share_cap(self, state: dict[str, Any], active_states: list[dict[str, Any]]) -> bool:
        total_tokens = sum(int(item["audit"]["kept_tokens"]) for item in active_states)
        if total_tokens < self.config.lm_weighted_source_token_budget:
            return False
        source_tokens = int(state["audit"]["kept_tokens"])
        share = _token_share(source_tokens, total_tokens)
        if share <= self.config.lm_max_source_token_share:
            return False
        other_tokens = [
            int(item["audit"]["kept_tokens"])
            for item in active_states
            if item is not state
        ]
        return any(tokens < source_tokens for tokens in other_tokens)

    def _source_hits_repeat_cap(self, state: dict[str, Any], active_states: list[dict[str, Any]]) -> bool:
        kept_documents = int(state["audit"]["kept_documents"])
        if kept_documents <= 0:
            return False
        repeat_rate = int(state["audit"]["repeated_documents"]) / max(kept_documents, 1)
        if repeat_rate <= self.config.lm_max_source_repeat_rate:
            return False
        other_documents = [
            int(item["audit"]["kept_documents"])
            for item in active_states
            if item is not state
        ]
        return any(documents > 0 for documents in other_documents)

    def _weighted_source_priority(
        self,
        state: dict[str, Any],
        all_states: list[dict[str, Any]],
    ) -> tuple[float, float, float, float]:
        source_tokens = int(state["audit"]["kept_tokens"])
        total_tokens = sum(int(item["audit"]["kept_tokens"]) for item in all_states)
        target_share = float(state["audit"].get("target_share", 0.0))
        if total_tokens <= 0 or target_share <= 0.0:
            return (0.0, target_share, 0.0, -float(source_tokens))
        realized_share = _token_share(source_tokens, total_tokens)
        deficit = target_share - realized_share
        weighted_ratio = realized_share / max(target_share, 1e-8)
        return (
            deficit,
            -weighted_ratio,
            target_share,
            -float(source_tokens),
        )

    def _weighted_source_share_gap_tokens(
        self,
        state: dict[str, Any],
        all_states: list[dict[str, Any]],
    ) -> float:
        source_tokens = int(state["audit"]["kept_tokens"])
        total_tokens = sum(int(item["audit"]["kept_tokens"]) for item in all_states)
        target_share = float(state["audit"].get("target_share", 0.0))
        if total_tokens <= 0 or target_share <= 0.0:
            return 0.0
        realized_share = _token_share(source_tokens, total_tokens)
        return max(0.0, (target_share - realized_share) * total_tokens)

    def _remaining_weighted_restart_documents(self, state: dict[str, Any]) -> int:
        kept_documents = int(state["audit"]["kept_documents"])
        repeated_documents = int(state["audit"]["repeated_documents"])
        if kept_documents <= 0:
            return 0
        repeat_cap = float(self.config.lm_max_source_repeat_rate)
        if repeat_cap <= 0.0:
            return 0
        if repeat_cap >= 1.0:
            return len(state["audit"].get("unique_document_ids", set())) or kept_documents
        remaining = (repeat_cap * kept_documents - repeated_documents) / max(1.0 - repeat_cap, 1e-8)
        if remaining <= 0.0:
            return 0
        cycle_size = len(state["audit"].get("unique_document_ids", set()))
        budget = int(math.floor(remaining + 1e-8))
        if cycle_size > 0:
            budget = min(budget, cycle_size)
        return max(budget, 0)

    def _source_can_restart_weighted(
        self,
        state: dict[str, Any],
        all_states: list[dict[str, Any]],
    ) -> bool:
        kept_documents = int(state["audit"]["kept_documents"])
        kept_tokens = int(state["audit"]["kept_tokens"])
        if kept_documents <= 0 or kept_tokens <= 0:
            return False
        if self._source_hits_share_cap(state, all_states):
            return False
        if self._source_hits_repeat_cap(state, all_states):
            return False
        restart_budget = self._remaining_weighted_restart_documents(state)
        if restart_budget <= 0:
            return False
        average_document_tokens = kept_tokens / max(kept_documents, 1)
        return self._weighted_source_share_gap_tokens(state, all_states) >= max(average_document_tokens, 1.0)

    def _restart_weighted_source(
        self,
        state: dict[str, Any],
        all_states: list[dict[str, Any]],
    ) -> bool:
        if not self._source_can_restart_weighted(state, all_states):
            return False
        restart_budget = self._remaining_weighted_restart_documents(state)
        if restart_budget <= 0:
            return False
        progress_state = state.get("progress")
        if progress_state is not None:
            progress_state["raw_records_consumed"] = 0
            progress_state["restart_count"] = int(progress_state.get("restart_count", 0)) + 1
        state["audit"]["restart_count"] = int(state["audit"].get("restart_count", 0)) + 1
        iterator_factory = state.get("iterator_factory")
        if iterator_factory is None:
            return False
        state["iterator"] = iter(
            iterator_factory(
                raw_records_consumed=0,
                allow_reentry=True,
                max_kept_documents=restart_budget,
            )
        )
        return True

    def _iter_weighted_tokenized_documents(
        self,
        sources: list[DataSourceConfig],
        *,
        tokenizer: SentencePieceTokenizer,
        source_progress: list[dict[str, Any]] | None = None,
        source_audits: list[dict[str, Any]] | None = None,
        seen_hashes: set[str] | None = None,
    ) -> Iterable[tuple[DataSourceConfig, DocumentRecord, list[int], dict[str, Any]]]:
        states = self._weighted_lm_source_iterators(
            sources,
            tokenizer=tokenizer,
            source_progress=source_progress,
            source_audits=source_audits,
            seen_hashes=seen_hashes,
        )
        active = list(states)
        dormant: list[dict[str, Any]] = []
        while active or dormant:
            for state in list(dormant):
                if self._restart_weighted_source(state, states):
                    dormant.remove(state)
                    active.append(state)
            if not active:
                break
            eligible = [
                state
                for state in active
                if not self._source_hits_share_cap(state, active)
                and not self._source_hits_repeat_cap(state, active)
            ]
            if not eligible:
                eligible = [state for state in active if not self._source_hits_repeat_cap(state, active)]
            if not eligible:
                eligible = list(active)
            selected = max(
                eligible,
                key=lambda state: self._weighted_source_priority(state, states),
            )
            try:
                document, token_ids = next(selected["iterator"])
            except StopIteration:
                active.remove(selected)
                dormant.append(selected)
                continue
            yield selected["source"], document, token_ids, selected["audit"]

    def audit_lm_stage(self, stage: str) -> dict[str, Any]:
        sources = self._stage_sources(stage)
        self._require_stage_sources(stage, sources)
        if self._uses_prepared_sources(sources):
            return self._audit_prepared_lm_stage(stage, sources)
        tokenizer = SentencePieceTokenizer(self.config.tokenizer_path)
        total_documents = 0
        total_tokens = 0
        source_audit_states: list[dict[str, Any]] = []
        shared_seen_hashes: set[str] = set()
        total_weight = sum(max(float(source.weight), 1e-8) for source in sources)
        for source in sources:
            audit_state = self._new_lm_audit_state(source)
            audit_state["target_share"] = max(float(source.weight), 1e-8) / max(total_weight, 1e-8)
            for _record, _token_ids in self._iter_tokenized_documents_for_source(
                source,
                tokenizer=tokenizer,
                seen_hashes=shared_seen_hashes,
                audit_state=audit_state,
                num_workers_override=self._configured_lm_audit_workers(),
            ):
                pass
            kept_documents = int(audit_state["kept_documents"])
            kept_tokens = int(audit_state["kept_tokens"])
            total_documents += kept_documents
            total_tokens += kept_tokens
            source_audit_states.append(audit_state)
        diagnostics = self._lm_source_diagnostics(
            source_audit_states,
            total_tokens=total_tokens,
            total_documents=total_documents,
            stage=stage,
        )
        return {
            "stage": stage,
            "total_documents": total_documents,
            "total_clean_tokens": total_tokens,
            "source_reports": diagnostics["per_source"],
            **diagnostics,
        }

    def _configured_lm_document_workers(self) -> int:
        configured = self.config.tokenizer_num_workers
        if configured is None:
            configured = self.config.preprocessing_num_workers
        if configured is None:
            configured = self.config.num_workers
        return max(int(configured or 0), 0)

    def _configured_lm_audit_workers(self) -> int:
        configured = self.config.audit_num_workers
        if int(configured or 0) <= 0:
            configured = self.config.num_workers
        return max(int(configured or 0), 0)

    def _bounded_lm_document_workers(
        self,
        *,
        num_workers_override: int | None = None,
        active_source_count: int = 1,
    ) -> int:
        configured = (
            self._configured_lm_document_workers()
            if num_workers_override is None
            else max(int(num_workers_override), 0)
        )
        if configured <= 1:
            return 0
        source_count = max(int(active_source_count), 1)
        return max(configured // source_count, 1)

    def _iter_serial_lm_document_results(
        self,
        source: DataSourceConfig,
        *,
        tokenizer: SentencePieceTokenizer,
        raw_records_consumed: int,
    ) -> Iterable[LMDocumentProcessResult]:
        record_index = int(raw_records_consumed)
        for item in self._load_source_records(source, raw_records_consumed=raw_records_consumed):
            record_index += 1
            payload = _lm_document_payload_from_item(item, source, record_index)
            try:
                yield _process_lm_document_payload(payload, self.config, source, tokenizer)
            except Exception as exc:
                yield LMDocumentProcessResult(
                    record_index=record_index,
                    is_text=isinstance(payload.get("text", ""), str),
                    error=f"{type(exc).__name__}: {exc}",
                    traceback=traceback.format_exc(),
                )

    def _iter_parallel_lm_document_results(
        self,
        source: DataSourceConfig,
        *,
        raw_records_consumed: int,
        num_workers: int,
    ) -> Iterable[LMDocumentProcessResult]:
        pending: deque[Future[list[LMDocumentProcessResult]]] = deque()
        max_pending = max(num_workers * LM_DOCUMENT_WORKER_PENDING_MULTIPLIER, 1)
        records = iter(self._load_source_records(source, raw_records_consumed=raw_records_consumed))
        next_record_index = int(raw_records_consumed)
        exhausted = False

        def next_chunk() -> list[dict[str, Any]]:
            nonlocal next_record_index
            chunk: list[dict[str, Any]] = []
            for _ in range(LM_DOCUMENT_WORKER_CHUNK_SIZE):
                try:
                    item = next(records)
                except StopIteration:
                    break
                next_record_index += 1
                chunk.append(_lm_document_payload_from_item(item, source, next_record_index))
            return chunk

        with ProcessPoolExecutor(
            max_workers=num_workers,
            initializer=_init_lm_document_worker,
            initargs=(self.config.to_dict(), source.to_dict(), self.config.tokenizer_path),
        ) as executor:
            while True:
                while len(pending) < max_pending and not exhausted:
                    chunk = next_chunk()
                    if not chunk:
                        exhausted = True
                        break
                    pending.append(executor.submit(_process_lm_document_chunk, chunk))
                if not pending:
                    break
                future = pending.popleft()
                try:
                    results = future.result()
                except Exception as exc:
                    raise RuntimeError(
                        f"LM document worker failed for source {source.name!r}: {type(exc).__name__}: {exc}"
                    ) from exc
                yield from results

    def _raise_lm_document_worker_error(
        self,
        source: DataSourceConfig,
        result: LMDocumentProcessResult,
    ) -> None:
        details = f"\n{result.traceback}" if result.traceback else ""
        raise RuntimeError(
            f"LM document worker failed for source {source.name!r} at raw record "
            f"{result.record_index}: {result.error}{details}"
        )

    def _merge_lm_counter(self, counter: Counter[str], values: dict[str, Any]) -> None:
        counter.update({str(key): int(value) for key, value in dict(values).items()})

    def _merge_lm_accepted_audit(
        self,
        audit_state: dict[str, Any],
        result: LMDocumentProcessResult,
        token_count: int,
    ) -> None:
        document_id = result.document_id
        if document_id and document_id in audit_state["unique_document_ids"]:
            audit_state["repeated_documents"] += 1
        if document_id:
            audit_state["unique_document_ids"].add(document_id)
        audit_state["kept_documents"] += 1
        audit_state["kept_tokens"] += token_count

        delta = result.audit_delta
        audit_state["quality_diagnostic_token_count"] += int(delta.get("quality_diagnostic_token_count", 0))
        shape_counts = dict(delta.get("shape_counts", {}))
        audit_state["document_char_count"] += int(shape_counts.get("chars", 0))
        audit_state["document_word_count"] += int(shape_counts.get("words", 0))
        audit_state["document_sentence_count"] += int(shape_counts.get("sentences", 0))
        audit_state["document_paragraph_count"] += int(shape_counts.get("paragraphs", 0))
        self._merge_lm_counter(
            audit_state["exact_paragraph_counter"],
            dict(delta.get("exact_paragraph_counter", {})),
        )
        self._merge_lm_counter(
            audit_state["normalized_paragraph_counter"],
            dict(delta.get("normalized_paragraph_counter", {})),
        )
        for phrase in list(delta.get("phrase_counter", [])):
            audit_state["phrase_counter"][str(phrase)] += 1
        for phrase in list(delta.get("phrase_counter_8", [])):
            audit_state["phrase_counter_8"][str(phrase)] += 1
        for phrase in list(delta.get("phrase_counter_12", [])):
            audit_state["phrase_counter_12"][str(phrase)] += 1
        for phrase in list(delta.get("phrase_counter_20", [])):
            audit_state["phrase_counter_20"][str(phrase)] += 1
        self._merge_lm_counter(
            audit_state["quality_artifact_counter"],
            dict(delta.get("quality_artifact_counter", {})),
        )
        self._merge_lm_counter(
            audit_state["quality_artifact_occurrence_counter"],
            dict(delta.get("quality_artifact_occurrence_counter", {})),
        )

    def assess_continue_readiness(self) -> dict[str, Any]:
        audit = self.audit_lm_stage("continue")
        clean_tokens = int(audit["total_clean_tokens"])
        required_tokens = int(
            max(
                0,
                round(
                    float(self.config.continue_readiness_min_clean_token_fraction)
                    * int(self.config.continued_pretraining_token_budget)
                ),
            )
        )
        failures: list[str] = []
        if clean_tokens < required_tokens:
            failures.append("insufficient_clean_tokens")
        if int(audit["total_documents"]) < self.config.continue_readiness_min_documents:
            failures.append("insufficient_documents")
        if int(audit["source_family_count"]) < self.config.continue_readiness_min_source_families:
            failures.append("insufficient_source_families")
        if float(audit["max_single_source_token_share"]) > self.config.continue_readiness_max_single_source_share:
            failures.append("single_source_share_too_high")
        if float(audit["max_repeat_rate"]) > self.config.continue_readiness_max_repeat_rate:
            failures.append("repeat_rate_too_high")
        return {
            "passed": not failures,
            "failures": failures,
            "required_clean_tokens": required_tokens,
            "audit": audit,
        }

    def _iter_documents(
        self, sources: list[DataSourceConfig], *, stage: str | None = None
    ) -> Iterable[DocumentRecord]:
        tokenizer = SentencePieceTokenizer(self.config.tokenizer_path)
        if len(sources) > 1 and any(abs(float(source.weight) - 1.0) > 1e-6 for source in sources):
            for _source, document, _token_ids, _audit in self._iter_weighted_tokenized_documents(
                sources,
                tokenizer=tokenizer,
            ):
                yield document
            return
        seen_hashes: set[str] = set()
        for source in sources:
            yielded_records = 0
            if stage is not None:
                location = source.dataset_name or source.path or ",".join(source.paths)
                _progress(
                    f"WebbGPT: preparing {stage} source {source.name} "
                    f"({source.format}) from {location}."
                )
            audit_state = self._new_lm_audit_state(source)
            for document, _token_ids in self._iter_tokenized_documents_for_source(
                source,
                tokenizer=tokenizer,
                seen_hashes=seen_hashes,
                audit_state=audit_state,
            ):
                yielded_records += 1
                if stage is not None and yielded_records % 1000 == 0:
                    _progress(
                        f"WebbGPT: preparing {stage} source {source.name}: "
                        f"kept {yielded_records:,} documents so far."
                    )
                yield document
            if stage is not None:
                _progress(
                    f"WebbGPT: preparing {stage} source {source.name}: "
                    f"finished with {yielded_records:,} documents kept."
                )

    def _iter_sft_examples(
        self, sources: list[DataSourceConfig], *, stage: str | None = None
    ) -> Iterable[SFTExample]:
        for source in sources:
            yielded_examples = 0
            if stage is not None:
                _progress(f"WebbGPT: preparing {stage} source {source.name} ({source.format}).")
            for item in self._load_source_records(source):
                messages = item.get(source.messages_field)
                if not isinstance(messages, list):
                    prompt_messages = _coerce_prompt_messages(item.get(source.prompt_field))
                    response = item.get(source.response_field)
                    if not isinstance(response, str):
                        response = item.get(source.chosen_field)
                    if prompt_messages is None or not isinstance(response, str):
                        continue
                    messages = [
                        *prompt_messages,
                        {"role": "assistant", "content": response},
                    ]
                yielded_examples += 1
                if stage is not None and yielded_examples % 500 == 0:
                    _progress(
                        f"WebbGPT: preparing {stage} source {source.name}: "
                        f"loaded {yielded_examples:,} SFT examples so far."
                    )
                metadata = _collect_standard_metadata(
                    item,
                    metadata_fields=source.metadata_fields,
                    extra_fields=STANDARD_SFT_METADATA_FIELDS,
                )
                yield SFTExample(
                    messages=messages,
                    source=source.name,
                    example_id=_message_example_id(source, item, messages),
                    split_group_id=_message_group_id(source, item, messages),
                    metadata={
                        **metadata,
                        "behavior_bucket": _infer_behavior_bucket(messages, metadata),
                        "quality_tier": _coerce_quality_tier(metadata),
                        "prompt_signature_hash": _prompt_signature_hash(messages),
                    },
                )
            if stage is not None:
                _progress(
                    f"WebbGPT: preparing {stage} source {source.name}: "
                    f"finished with {yielded_examples:,} SFT examples."
                )

    def _iter_preference_examples(
        self, sources: list[DataSourceConfig], *, stage: str | None = None
    ) -> Iterable[PreferenceExample]:
        for source in sources:
            yielded_examples = 0
            if stage is not None:
                _progress(f"WebbGPT: preparing {stage} source {source.name} ({source.format}).")
            for item in self._load_source_records(source):
                prompt = _coerce_prompt_messages(item.get(source.prompt_field))
                chosen = item.get(source.chosen_field)
                rejected = item.get(source.rejected_field)
                if prompt is None or not isinstance(chosen, str) or not isinstance(rejected, str):
                    continue
                yielded_examples += 1
                if stage is not None and yielded_examples % 500 == 0:
                    _progress(
                        f"WebbGPT: preparing {stage} source {source.name}: "
                        f"loaded {yielded_examples:,} preference examples so far."
                    )
                metadata = _collect_standard_metadata(
                    item,
                    metadata_fields=source.metadata_fields,
                    extra_fields=STANDARD_PREFERENCE_METADATA_FIELDS,
                )
                yield PreferenceExample(
                    prompt=prompt,
                    chosen=chosen,
                    rejected=rejected,
                    source=source.name,
                    example_id=_preference_example_id(source, item, prompt, chosen, rejected),
                    split_group_id=_message_group_id(source, item, prompt),
                    metadata={
                        **metadata,
                        "chosen_quality_tier": _coerce_chosen_quality_tier(metadata),
                        "negative_type": _coerce_negative_type(metadata),
                        "prompt_signature_hash": _prompt_signature_hash(prompt),
                    },
                )
            if stage is not None:
                _progress(
                    f"WebbGPT: preparing {stage} source {source.name}: "
                    f"finished with {yielded_examples:,} preference examples."
                )

    def build_pretrain(self):
        self._require_stage_sources("pretrain", self.config.pretrain_sources)
        if self._uses_prepared_sources(self.config.pretrain_sources):
            return self._build_prepared_dataset(self.config.pretrain_sources, expected_kind="packed_lm")
        tokenizer = SentencePieceTokenizer(self.config.tokenizer_path)
        tokenized = (
            tokenizer.encode(doc.text, add_bos=True, add_eos=True)
            for doc in self._iter_documents(self.config.pretrain_sources)
        )
        sequences = pack_token_sequences(
            tokenized,
            sequence_length=self.config.sequence_length,
            pad_token_id=tokenizer.token_to_id("<pad>"),
            eos_token_id=tokenizer.token_to_id("</s>"),
        )
        return PackedSequenceDataset(sequences, tokenizer.token_to_id("<pad>"))

    def build_continued_pretrain(self):
        self._require_stage_sources("continue", self.config.continued_pretrain_sources)
        if self._uses_prepared_sources(self.config.continued_pretrain_sources):
            return self._build_prepared_dataset(
                self.config.continued_pretrain_sources, expected_kind="packed_lm"
            )
        tokenizer = SentencePieceTokenizer(self.config.tokenizer_path)
        tokenized = (
            tokenizer.encode(doc.text, add_bos=True, add_eos=True)
            for doc in self._iter_documents(self.config.continued_pretrain_sources)
        )
        sequences = pack_token_sequences(
            tokenized,
            sequence_length=self.config.sequence_length,
            pad_token_id=tokenizer.token_to_id("<pad>"),
            eos_token_id=tokenizer.token_to_id("</s>"),
        )
        return PackedSequenceDataset(sequences, tokenizer.token_to_id("<pad>"))

    def _build_sft_dataset_from_sources(self, sources: list[DataSourceConfig]):
        if self._uses_prepared_sources(sources):
            return self._build_prepared_dataset(sources, expected_kind="sft")
        return SFTDataset(
            list(self._iter_sft_examples(sources)),
            self.config.tokenizer_path,
            self.config.sequence_length,
        )

    def build_sft(self):
        self._require_stage_sources("sft", self.config.sft_sources)
        return self._build_sft_dataset_from_sources(self.config.sft_sources)

    def build_sft_split(
        self,
        *,
        seed: int,
        validation_fraction: float,
        validation_min_examples: int,
        allow_weak_validation: bool,
        require_explicit_validation: bool = False,
    ):
        self._require_stage_sources("sft", self.config.sft_sources)
        if self.config.sft_validation_sources:
            return (
                self._build_sft_dataset_from_sources(self.config.sft_sources),
                self._build_sft_dataset_from_sources(self.config.sft_validation_sources),
            )
        if require_explicit_validation:
            raise RuntimeError(
                "WebbGPT: local-MVP SFT requires explicit sft_validation_sources by default. "
                "Only exploratory runs may rely on grouped auto-split."
            )
        if self._uses_prepared_sources(self.config.sft_sources):
            raise RuntimeError(
                "Prepared SFT training sources require explicit sft_validation_sources in v1 because grouped auto-splitting "
                "needs access to raw prompt metadata."
            )
        examples = list(self._iter_sft_examples(self.config.sft_sources))
        train_indices, validation_indices = _split_indices_by_group(
            examples,
            stage_name="sft",
            seed=seed,
            validation_fraction=validation_fraction,
            validation_min_examples=validation_min_examples,
            allow_weak_validation=allow_weak_validation,
        )
        train_examples = [examples[index] for index in train_indices]
        validation_examples = [examples[index] for index in validation_indices]
        return (
            SFTDataset(train_examples, self.config.tokenizer_path, self.config.sequence_length),
            SFTDataset(validation_examples, self.config.tokenizer_path, self.config.sequence_length),
        )

    def _build_preference_dataset_from_sources(self, sources: list[DataSourceConfig]):
        if self._uses_prepared_sources(sources):
            return self._build_prepared_dataset(
                sources, expected_kind="preference"
            )
        return PreferenceDataset(
            list(self._iter_preference_examples(sources)),
            self.config.tokenizer_path,
            self.config.sequence_length,
        )

    def build_preference(self):
        self._require_stage_sources("preference", self.config.preference_sources)
        return self._build_preference_dataset_from_sources(self.config.preference_sources)

    def build_preference_split(
        self,
        *,
        seed: int,
        validation_fraction: float,
        validation_min_examples: int,
        allow_weak_validation: bool,
        require_explicit_validation: bool = False,
    ):
        self._require_stage_sources("preference", self.config.preference_sources)
        if self.config.preference_validation_sources:
            return (
                self._build_preference_dataset_from_sources(self.config.preference_sources),
                self._build_preference_dataset_from_sources(self.config.preference_validation_sources),
            )
        if require_explicit_validation:
            raise RuntimeError(
                "WebbGPT: local-MVP DPO requires explicit preference_validation_sources by default. "
                "Only exploratory runs may rely on grouped auto-split."
            )
        if self._uses_prepared_sources(self.config.preference_sources):
            raise RuntimeError(
                "Prepared preference training sources require explicit preference_validation_sources in v1 because grouped "
                "auto-splitting needs access to raw prompt metadata."
            )
        examples = list(self._iter_preference_examples(self.config.preference_sources))
        train_indices, validation_indices = _split_indices_by_group(
            examples,
            stage_name="preference",
            seed=seed,
            validation_fraction=validation_fraction,
            validation_min_examples=validation_min_examples,
            allow_weak_validation=allow_weak_validation,
        )
        train_examples = [examples[index] for index in train_indices]
        validation_examples = [examples[index] for index in validation_indices]
        return (
            PreferenceDataset(train_examples, self.config.tokenizer_path, self.config.sequence_length),
            PreferenceDataset(validation_examples, self.config.tokenizer_path, self.config.sequence_length),
        )

    def validate_preference_datasets(self, *datasets) -> dict[str, Any]:
        combined_examples: list[PreferenceExample] = []
        for dataset in datasets:
            examples = getattr(dataset, "examples", None)
            if examples is None:
                return {
                    "valid_for_promotion": False,
                    "promotion_blockers": ["behavior_eval_untrusted"],
                    "invalid_quality_tiers": {},
                    "invalid_negative_types": {},
                    "approved_template_count": 0,
                    "approved_template_share": 0.0,
                }
            combined_examples.extend(examples)
        return self._validate_preference_metadata(combined_examples)

    def build_validation(self):
        self._require_stage_sources("validation", self.config.validation_sources)
        if self._uses_prepared_sources(self.config.validation_sources):
            return self._build_prepared_dataset(self.config.validation_sources, expected_kind="packed_lm")
        tokenizer = SentencePieceTokenizer(self.config.tokenizer_path)
        tokenized = (
            tokenizer.encode(doc.text, add_bos=True, add_eos=True)
            for doc in self._iter_documents(self.config.validation_sources)
        )
        sequences = pack_token_sequences(
            tokenized,
            sequence_length=self.config.sequence_length,
            pad_token_id=tokenizer.token_to_id("<pad>"),
            eos_token_id=tokenizer.token_to_id("</s>"),
        )
        return PackedSequenceDataset(sequences, tokenizer.token_to_id("<pad>"))

    def export_examples(self, stage: str, output_path: str) -> None:
        sources = self._stage_sources(stage)
        self._require_stage_sources(stage, sources)
        if self._uses_prepared_sources(sources):
            output = Path(output_path)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(load_prepared_manifest(sources[0].path), indent=2))
            return
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        if stage in {"pretrain", "continue", "validation"}:
            rows = [asdict(record) for record in self._iter_documents(sources)]
        elif stage == "sft":
            rows = [asdict(record) for record in self._iter_sft_examples(sources)]
        elif stage == "preference":
            rows = [asdict(record) for record in self._iter_preference_examples(sources)]
        else:
            raise ValueError(f"Unsupported stage {stage!r}")
        output.write_text("\n".join(json.dumps(row) for row in rows))
