from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re
import shutil
from typing import Any

from tokenizer import SentencePieceTokenizer, format_chat


POSTTRAIN_REGRESSION_PATH = "data/eval/posttrain_regression.jsonl"
PRETRAIN_REGRESSION_PATH = "data/eval/pretrain_general_regression.jsonl"
PRETRAIN_FAMILY_HOLDOUTS_PATH = "data/eval/pretrain_family_holdouts_general.json"
PRETRAIN_QUALITATIVE_RUBRIC = [
    {
        "perplexity_band": "300+",
        "expected_behavior": "topic drift and broken semantics expected",
    },
    {
        "perplexity_band": "220-260",
        "expected_behavior": "sentence-shaped text, weak coherence, some drift expected",
    },
    {
        "perplexity_band": "180-220",
        "expected_behavior": "should often stay on topic for one paragraph, still generic",
    },
    {
        "perplexity_band": "150-180",
        "expected_behavior": "should usually complete the thought with local coherence",
    },
    {
        "perplexity_band": "below 150",
        "expected_behavior": "stronger paragraph control expected",
    },
]
SPECIAL_TOKEN_PIECES = {
    "<s>",
    "</s>",
    "<pad>",
    "<|system|>",
    "<|user|>",
    "<|assistant|>",
    "<|tool|>",
}
GENERIC_REFUSAL_PATTERNS = (
    "i can't say that",
    "i cant say that",
    "i can’t say that",
    "i can't help you",
    "i can’t help you",
    "i can't help you to help",
)
ABSTENTION_PATTERNS = (
    "not listed",
    "not in the catalog",
    "not in the handbook",
    "not enough information",
    "cannot confirm",
    "can't confirm",
    "i don't see",
    "i do not see",
    "i can't verify",
    "i cannot verify",
)
CLARIFICATION_PATTERNS = (
    "what ",
    "which ",
    "can you share",
    "could you share",
    "tell me more",
    "before i recommend",
    "before recommending",
)
SOURCE_TAG_RE = re.compile(r"\[source:\s*([^\]]+)\]", re.IGNORECASE)
SOURCE_MENTION_PATTERNS: dict[str, tuple[str, ...]] = {
    "course catalog": (
        "course catalog says",
        "catalog says",
        "[source: course catalog]",
    ),
    "handbook": (
        "handbook says",
        "[source: handbook]",
    ),
}
RAW_LM_GENERIC_ATTRACTOR_CATEGORIES = {
    "word_language_text": {
        "word",
        "words",
        "language",
        "term",
        "terms",
        "text",
        "texts",
    },
    "book_story_article_essay": {
        "book",
        "books",
        "story",
        "stories",
        "article",
        "articles",
        "essay",
        "essays",
    },
    "water_fire_land_history": {
        "water",
        "fire",
        "river",
        "rivers",
        "land",
        "city",
        "cities",
        "war",
        "wars",
    },
    "problem_issue_solution": {
        "problem",
        "problems",
        "issue",
        "issues",
        "solution",
        "solutions",
    },
    "data_study_research_result": {
        "data",
        "study",
        "studies",
        "research",
        "result",
        "results",
    },
    "product_computer_video_project": {
        "product",
        "products",
        "computer",
        "computers",
        "video",
        "videos",
        "project",
        "projects",
    },
    "student_course_class_teacher": {
        "student",
        "students",
        "course",
        "courses",
        "class",
        "classes",
        "teacher",
        "teachers",
    },
    "health_body_child_medical": {
        "health",
        "body",
        "bodies",
        "child",
        "children",
        "virus",
        "viruses",
        "infection",
        "infections",
        "diet",
        "diets",
        "heart",
    },
    "generic_fillers": {
        "world",
        "time",
        "thing",
        "things",
        "same",
        "important",
        "process",
        "people",
    },
}
RAW_LM_GENERIC_ATTRACTORS = {
    term for terms in RAW_LM_GENERIC_ATTRACTOR_CATEGORIES.values() for term in terms
}
RAW_LM_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "because",
    "but",
    "by",
    "can",
    "for",
    "from",
    "has",
    "have",
    "help",
    "how",
    "if",
    "in",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "should",
    "that",
    "the",
    "their",
    "then",
    "this",
    "to",
    "when",
    "who",
    "will",
    "with",
    "what",
    "which",
    "why",
    "you",
    "your",
}
RAW_LM_DOMAIN_TERMS = {
    "advisor",
    "advising",
    "catalog",
    "class",
    "course",
    "credit",
    "department",
    "departmental",
    "description",
    "elective",
    "eligible",
    "enroll",
    "honors",
    "placement",
    "planning",
    "prerequisite",
    "prerequisites",
    "recommendation",
    "schedule",
    "semester",
    "webb",
    "workload",
}
RAW_LM_CATALOG_CONTEXT_TERMS = {
    "catalog",
    "course",
    "courses",
    "class",
    "classes",
}
RAW_LM_CATALOG_GROUNDING_TERMS = {
    "approval",
    "credit",
    "credits",
    "department",
    "departmental",
    "eligible",
    "eligibility",
    "enroll",
    "enrollment",
    "prerequisite",
    "prerequisites",
}
RAW_LM_RECOMMENDATION_GROUNDING_TERMS = {
    "background",
    "eligible",
    "eligibility",
    "prerequisite",
    "prerequisites",
    "recommendation",
    "recommendations",
    "required",
    "requirement",
    "requirements",
}
RAW_LM_CREDIT_GROUNDING_TERMS = {
    "credit",
    "credits",
    "eligible",
    "eligibility",
    "enroll",
    "enrollment",
    "junior",
    "juniors",
    "open",
    "senior",
    "seniors",
}
RAW_LM_DOMAIN_BOILERPLATE = (
    "catalog entry should be read",
    "read for any placement or departmental approval notes",
    "the catalog entry should",
    "a useful continuation",
    "grounded in the catalog",
)
RAW_LM_CATALOG_DRIFT_PATTERNS = (
    "course catalog",
    "catalog entry",
    "catalog facts",
    "catalog says",
    "department approval",
    "departmental approval",
    "one-half credit",
    "prerequisite",
    "prerequisites",
    "webb",
)
RAW_LM_NARRATIVE_DRIFT_PATTERNS = (
    "catalog",
    "course",
    "students",
    "the article",
    "this article",
    "the study",
    "the research",
    "the history of",
    "geography",
    "in the early",
    "in the late",
    "united states",
    "population",
    "century",
    "government",
    "largest city",
    "first country",
    "was first known",
    "was established",
)
RAW_LM_EVERYDAY_MEDICAL_DRIFT_TERMS = {
    "blood",
    "body",
    "child",
    "children",
    "diabetes",
    "diet",
    "doctor",
    "food",
    "health",
    "medical",
    "pain",
    "symptoms",
    "virus",
    "viruses",
    "infection",
    "infections",
    "heart",
}
RAW_LM_EVERYDAY_PRODUCT_RESEARCH_DRIFT_TERMS = {
    "computer",
    "computers",
    "data",
    "product",
    "products",
    "project",
    "projects",
    "research",
    "study",
    "studies",
    "video",
    "videos",
}
RAW_LM_MAJOR_FAILURE_REASONS = {
    "prompt_topic_retention_too_low",
    "domain_collapse",
    "boilerplate_repetition",
    "semantic_repetition",
    "narrative_to_expository_drift",
    "everyday_to_medical_drift",
    "catalog_advising_drift_into_unrelated_prompt",
    "malformed_token_rate_high",
}
RAW_WORD_RE = re.compile(r"[a-zA-Z][a-zA-Z']*")
WEIRD_HYPHEN_RE = re.compile(
    r"(?:\b[a-zA-Z]{1,6}-){2,}[a-zA-Z]{1,10}\b|-[a-zA-Z]{1,4}-|"
    r"\b(?:Newth|F-the-s|based-in|in-in|to-in|mvp-[a-z]{1,3})\b"
)
MALFORMED_WORD_RE = re.compile(
    r"\b(?:scheduleite|berled|inology|engagingable|pronting|unforgetable)\b|"
    r"\b[a-z]{4,}ingable\b|"
    r"\bover-the-[a-z]{2,4}\b",
    re.IGNORECASE,
)
QUOTE_SYLLABLE_LOOP_RE = re.compile(
    r"[\"'“”‘’][a-z]{1,4}[\"'“”‘’](?:\s+(?:or|and|is|means|of|in|as)\s+"
    r"[\"'“”‘’][a-z]{1,4}[\"'“”‘’]){2,}",
    re.IGNORECASE,
)


def _require_torch():
    import torch

    return torch


def _normalize_text(value: str) -> str:
    return " ".join(value.split()).strip()


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


def prompt_signature_text(messages: list[dict[str, str]]) -> str:
    return json.dumps(_normalize_messages(messages, include_assistant=False), sort_keys=True, separators=(",", ":"))


def prompt_signature_hash(messages: list[dict[str, str]]) -> str:
    encoded = prompt_signature_text(messages).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_posttrain_regression_records(
    path: str | Path = POSTTRAIN_REGRESSION_PATH,
    *,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    regression_path = Path(path)
    if not regression_path.exists():
        return records
    for line in regression_path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        messages = row.get("messages") or [{"role": "user", "content": row["prompt"]}]
        records.append(
            {
                "source_path": str(regression_path),
                "messages": messages,
                "tags": list(row.get("tags", [])),
                "expected_mode": row.get("expected_mode"),
                "allowed_source_labels": list(row.get("allowed_source_labels", [])),
                "forbidden_source_labels": list(row.get("forbidden_source_labels", [])),
                "requires_source_label": bool(row.get("requires_source_label", False)),
                "prompt_signature_text": prompt_signature_text(messages),
                "prompt_signature_hash": prompt_signature_hash(messages),
            }
        )
        if limit is not None and len(records) >= limit:
            break
    return records


def _raw_lm_prompt_from_record(row: dict[str, Any]) -> str:
    prompt = row.get("prompt")
    if isinstance(prompt, str):
        return prompt.strip()
    messages = row.get("messages")
    if isinstance(messages, list):
        for message in reversed(messages):
            if not isinstance(message, dict) or message.get("role") != "user":
                continue
            content = str(message.get("content") or "")
            prefix = "Continue this passage in one coherent paragraph:\n"
            if content.startswith(prefix):
                content = content.removeprefix(prefix)
            return content.strip()
    return ""


def load_raw_lm_regression_records(
    path: str | Path = PRETRAIN_REGRESSION_PATH,
    *,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    regression_path = Path(path)
    if not regression_path.exists():
        return records
    for line in regression_path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        prompt = _raw_lm_prompt_from_record(row)
        if not prompt:
            continue
        records.append(
            {
                "source_path": str(regression_path),
                "id": str(row.get("id") or f"raw_lm_probe_{len(records) + 1:02d}"),
                "prompt": prompt,
                "bucket": str(row.get("bucket") or "unspecified"),
                "probe_type": str(row.get("probe_type") or "unspecified"),
                "tags": list(row.get("tags", [])),
                "sample_mode": "raw_lm",
            }
        )
        if limit is not None and len(records) >= limit:
            break
    return records


def collect_prompt_signatures(examples: list[Any]) -> tuple[set[str], set[str]]:
    signature_texts: set[str] = set()
    signature_hashes: set[str] = set()
    for example in examples:
        metadata = getattr(example, "metadata", None)
        if isinstance(metadata, dict):
            prompt_hash = metadata.get("prompt_signature_hash")
            if isinstance(prompt_hash, str) and prompt_hash:
                signature_hashes.add(prompt_hash)
        if hasattr(example, "messages"):
            messages = getattr(example, "messages")
        elif hasattr(example, "prompt"):
            messages = getattr(example, "prompt")
        else:
            continue
        if not messages:
            continue
        text = prompt_signature_text(messages)
        signature_texts.add(text)
        signature_hashes.add(prompt_signature_hash(messages))
    return signature_texts, signature_hashes


def ensure_no_regression_prompt_overlap(
    *,
    stage_name: str,
    train_examples: list[Any],
    validation_examples: list[Any],
    regression_path: str | Path = POSTTRAIN_REGRESSION_PATH,
) -> None:
    regression_records = load_posttrain_regression_records(regression_path)
    if not regression_records:
        return
    train_texts, train_hashes = collect_prompt_signatures(train_examples)
    validation_texts, validation_hashes = collect_prompt_signatures(validation_examples)
    for record in regression_records:
        prompt_text = record["prompt_signature_text"]
        prompt_hash = record["prompt_signature_hash"]
        if prompt_text in train_texts or prompt_hash in train_hashes:
            raise RuntimeError(
                f"WebbGPT: {stage_name} regression prompt suite overlaps the training data. "
                f"Remove or rewrite the overlapping prompt from {record['source_path']} before training."
            )
        if prompt_text in validation_texts or prompt_hash in validation_hashes:
            raise RuntimeError(
                f"WebbGPT: {stage_name} regression prompt suite overlaps the validation data. "
                f"Remove or rewrite the overlapping prompt from {record['source_path']} before training."
            )


def append_eval_history(path: str | Path, payload: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def write_selection_metadata(best_dir: str | Path, payload: dict[str, Any]) -> None:
    target_dir = Path(best_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    (target_dir / "selection.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False))


def update_topk_candidates(
    output_dir: str | Path,
    *,
    candidate_path: str | Path,
    candidate_payload: dict[str, Any],
    metric_key: str,
    limit: int,
    lower_is_better: bool = True,
) -> list[dict[str, Any]]:
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    metadata_path = target / "topk.json"
    current: list[dict[str, Any]] = []
    if metadata_path.exists():
        current = json.loads(metadata_path.read_text())
    current = [entry for entry in current if entry.get("path") != str(candidate_path)]
    current.append(
        {
            "path": str(candidate_path),
            "metric_key": metric_key,
            "metric_value": candidate_payload.get("selection_value"),
            "step": candidate_payload.get("step"),
            "selection_metric": candidate_payload.get("selection_metric"),
            "metrics": candidate_payload.get("metrics"),
        }
    )
    current.sort(
        key=lambda entry: (
            float(entry.get("metric_value", float("inf"))) if lower_is_better else -float(entry.get("metric_value", float("-inf"))),
            int(entry.get("step", 0)),
        )
    )
    kept = current[: max(limit, 0)]
    kept_paths = {entry["path"] for entry in kept}
    for entry in current[max(limit, 0) :]:
        stale = Path(entry["path"])
        if stale.exists() and stale.is_dir() and str(stale) not in kept_paths:
            shutil.rmtree(stale, ignore_errors=True)
    metadata_path.write_text(json.dumps(kept, indent=2, ensure_ascii=False))
    return kept


def _clean_generated_response(tokenizer: SentencePieceTokenizer, token_ids: list[int]) -> tuple[str, str]:
    eos_token_id = tokenizer.token_to_id("</s>")
    raw_response = tokenizer.decode(token_ids).strip()
    clean_token_ids: list[int] = []
    for token_id in token_ids:
        if token_id == eos_token_id:
            break
        piece = tokenizer.id_to_token(token_id)
        if piece in SPECIAL_TOKEN_PIECES or (piece.startswith("<|") and piece.endswith("|>")):
            continue
        clean_token_ids.append(token_id)
    clean_response = tokenizer.decode(clean_token_ids).replace("</s>", "").strip()
    return raw_response, clean_response


def generate_qualitative_samples(
    model,
    tokenizer_path: str,
    *,
    regression_path: str | Path = POSTTRAIN_REGRESSION_PATH,
    limit: int = 3,
    max_new_tokens: int = 128,
    temperature: float = 0.0,
    top_p: float = 1.0,
    repetition_penalty: float = 1.05,
    no_repeat_ngram_size: int = 4,
    stop_strings: list[str] | None = None,
) -> list[dict[str, Any]]:
    torch = _require_torch()
    prompt_records = load_posttrain_regression_records(regression_path, limit=limit)
    if not prompt_records:
        return []

    tokenizer = SentencePieceTokenizer(tokenizer_path)
    stop_token_ids = [tokenizer.token_to_id("</s>")]
    if stop_strings:
        for value in stop_strings:
            try:
                token_id = tokenizer.token_to_id(value)
            except Exception:
                continue
            if token_id >= 0 and token_id not in stop_token_ids:
                stop_token_ids.append(token_id)
    generation_model = model.module if hasattr(model, "module") and hasattr(model.module, "generate") else model
    device = next(generation_model.parameters()).device
    previous_training = bool(getattr(model, "training", False))
    model.eval()
    try:
        samples: list[dict[str, Any]] = []
        for record in prompt_records:
            messages = record["messages"]
            rendered = format_chat(messages, add_generation_prompt=True)
            input_ids = torch.tensor(
                [tokenizer.encode(rendered, add_bos=True, add_eos=False)],
                dtype=torch.long,
                device=device,
            )
            attention_mask = torch.ones_like(input_ids)
            generated = generation_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
                stop_token_ids=stop_token_ids,
            )
            new_tokens = generated[0, input_ids.size(1) :].tolist()
            raw_response, clean_response = _clean_generated_response(tokenizer, new_tokens)
            prompt = next(
                (message["content"] for message in reversed(messages) if message.get("role") == "user"),
                messages[-1]["content"],
            )
            samples.append(
                {
                    "prompt": prompt,
                    "raw_response": raw_response,
                    "clean_response": clean_response,
                    "tags": record["tags"],
                    "expected_mode": record.get("expected_mode"),
                    "allowed_source_labels": list(record.get("allowed_source_labels", [])),
                    "forbidden_source_labels": list(record.get("forbidden_source_labels", [])),
                    "requires_source_label": bool(record.get("requires_source_label", False)),
                    "source_path": record["source_path"],
                }
            )
        return samples
    finally:
        model.train(previous_training)


def generate_raw_lm_qualitative_samples(
    model,
    tokenizer_path: str,
    *,
    regression_path: str | Path = PRETRAIN_REGRESSION_PATH,
    limit: int | None = 3,
    max_new_tokens: int = 128,
    temperature: float = 0.7,
    top_p: float = 0.95,
    repetition_penalty: float = 1.05,
    no_repeat_ngram_size: int = 4,
    stop_strings: list[str] | None = None,
) -> list[dict[str, Any]]:
    torch = _require_torch()
    prompt_records = load_raw_lm_regression_records(regression_path, limit=limit)
    if not prompt_records:
        return []

    tokenizer = SentencePieceTokenizer(tokenizer_path)
    stop_token_ids = [tokenizer.token_to_id("</s>")]
    if stop_strings:
        for value in stop_strings:
            try:
                token_id = tokenizer.token_to_id(value)
            except Exception:
                continue
            if token_id >= 0 and token_id not in stop_token_ids:
                stop_token_ids.append(token_id)
    generation_model = model.module if hasattr(model, "module") and hasattr(model.module, "generate") else model
    device = next(generation_model.parameters()).device
    previous_training = bool(getattr(model, "training", False))
    model.eval()
    try:
        samples: list[dict[str, Any]] = []
        for record in prompt_records:
            prompt = record["prompt"]
            input_ids = torch.tensor(
                [tokenizer.encode(prompt, add_bos=True, add_eos=False)],
                dtype=torch.long,
                device=device,
            )
            attention_mask = torch.ones_like(input_ids)
            generated = generation_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
                stop_token_ids=stop_token_ids,
            )
            new_tokens = generated[0, input_ids.size(1) :].tolist()
            raw_response, clean_response = _clean_generated_response(tokenizer, new_tokens)
            samples.append(
                {
                    "prompt": prompt,
                    "raw_response": raw_response,
                    "clean_response": clean_response,
                    "id": record["id"],
                    "bucket": record["bucket"],
                    "probe_type": record["probe_type"],
                    "tags": record["tags"],
                    "sample_mode": "raw_lm",
                    "source_path": record["source_path"],
                }
            )
        return samples
    finally:
        model.train(previous_training)


def _normalize_source_label(value: str) -> str:
    return _normalize_text(value).lower()


def _extract_source_mentions(response: str) -> set[str]:
    normalized = response.lower()
    mentions = {
        _normalize_source_label(match.group(1))
        for match in SOURCE_TAG_RE.finditer(response)
    }
    for label, patterns in SOURCE_MENTION_PATTERNS.items():
        if any(pattern in normalized for pattern in patterns):
            mentions.add(label)
    return mentions


def _looks_like_abstention(response: str) -> bool:
    normalized = response.lower()
    return any(pattern in normalized for pattern in ABSTENTION_PATTERNS)


def _looks_like_clarification(response: str) -> bool:
    normalized = response.lower()
    return "?" in response or any(pattern in normalized for pattern in CLARIFICATION_PATTERNS)


def _score_grounded_sample(sample: dict[str, Any]) -> dict[str, int]:
    response = _normalize_text(
        str(sample.get("clean_response") or sample.get("response") or "")
    )
    if not response:
        return {
            "unsupported_source_tag_count": 0,
            "wrong_source_attribution_count": 0,
            "grounded_abstention_fail_count": 0,
            "clarification_missing_count": 0,
            "fabricated_fact_pattern_count": 0,
        }
    mentions = _extract_source_mentions(response)
    allowed_sources = {
        _normalize_source_label(value)
        for value in sample.get("allowed_source_labels", [])
        if isinstance(value, str)
    }
    forbidden_sources = {
        _normalize_source_label(value)
        for value in sample.get("forbidden_source_labels", [])
        if isinstance(value, str)
    }
    expected_mode = _normalize_text(str(sample.get("expected_mode") or "")).lower()
    requires_source_label = bool(sample.get("requires_source_label", False))
    unsupported_source_tag_count = 0
    wrong_source_attribution_count = 0
    grounded_abstention_fail_count = 0
    clarification_missing_count = 0
    fabricated_fact_pattern_count = 0
    if mentions and not allowed_sources and not requires_source_label:
        unsupported_source_tag_count += 1
    wrong_source_detected = False
    if allowed_sources and mentions and not mentions.issubset(allowed_sources):
        wrong_source_detected = True
    if forbidden_sources and mentions.intersection(forbidden_sources):
        wrong_source_detected = True
    if wrong_source_detected:
        wrong_source_attribution_count += 1
    if requires_source_label and not mentions:
        unsupported_source_tag_count += 1
    if expected_mode == "abstain" and not _looks_like_abstention(response):
        grounded_abstention_fail_count += 1
        if mentions:
            fabricated_fact_pattern_count += 1
    if expected_mode == "clarify" and not _looks_like_clarification(response):
        clarification_missing_count += 1
    if expected_mode in {"abstain", "clarify"} and any(
        pattern in response.lower()
        for pattern in (
            "the handbook says",
            "the course catalog says",
            "catalog says",
            "handbook says",
        )
    ) and mentions:
        fabricated_fact_pattern_count += 1
    return {
        "unsupported_source_tag_count": unsupported_source_tag_count,
        "wrong_source_attribution_count": wrong_source_attribution_count,
        "grounded_abstention_fail_count": grounded_abstention_fail_count,
        "clarification_missing_count": clarification_missing_count,
        "fabricated_fact_pattern_count": fabricated_fact_pattern_count,
    }


def load_pretrain_family_holdouts(
    path: str | Path = PRETRAIN_FAMILY_HOLDOUTS_PATH,
) -> dict[str, list[str]]:
    manifest_path = Path(path)
    if not manifest_path.exists():
        return {}
    payload = json.loads(manifest_path.read_text())
    families: dict[str, list[str]] = {}
    for family, source_path in payload.items():
        family_path = Path(source_path)
        if not family_path.exists():
            continue
        texts = [_normalize_text(line) for line in family_path.read_text().splitlines() if _normalize_text(line)]
        if texts:
            families[str(family)] = texts
    return families


def _iter_holdout_windows(
    texts: list[str],
    *,
    tokenizer: SentencePieceTokenizer,
    sequence_length: int,
) -> list[list[int]]:
    pad_id = tokenizer.token_to_id("<pad>")
    windows: list[list[int]] = []
    for text in texts:
        token_ids = tokenizer.encode(text, add_bos=True, add_eos=True)
        if not token_ids:
            continue
        for start in range(0, len(token_ids), sequence_length):
            window = token_ids[start : start + sequence_length]
            if len(window) < sequence_length:
                window = window + [pad_id] * (sequence_length - len(window))
            windows.append(window)
    return windows


def evaluate_pretrain_family_holdouts(
    model,
    tokenizer_path: str,
    *,
    sequence_length: int,
    holdouts_path: str | Path = PRETRAIN_FAMILY_HOLDOUTS_PATH,
) -> dict[str, Any]:
    torch = _require_torch()
    family_texts = load_pretrain_family_holdouts(holdouts_path)
    if not family_texts:
        return {"families": {}, "best_family": None, "worst_family": None}
    tokenizer = SentencePieceTokenizer(tokenizer_path)
    device = next(model.parameters()).device
    previous_training = bool(getattr(model, "training", False))
    model.eval()
    try:
        family_metrics: dict[str, dict[str, float | int]] = {}
        for family, texts in family_texts.items():
            windows = _iter_holdout_windows(texts, tokenizer=tokenizer, sequence_length=sequence_length)
            if not windows:
                continue
            losses: list[float] = []
            with torch.no_grad():
                for window in windows:
                    input_ids = torch.tensor([window], dtype=torch.long, device=device)
                    attention_mask = (input_ids != tokenizer.token_to_id("<pad>")).long()
                    labels = input_ids.clone()
                    labels[attention_mask == 0] = -100
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                    )
                    losses.append(float(outputs.loss.item()))
            if not losses:
                continue
            average_loss = sum(losses) / len(losses)
            family_metrics[family] = {
                "loss": round(average_loss, 6),
                "perplexity": round(math.exp(min(average_loss, 20.0)), 6),
                "examples_evaluated": len(texts),
                "windows_evaluated": len(losses),
                "coverage_percent": 100.0,
            }
        if not family_metrics:
            return {"families": {}, "best_family": None, "worst_family": None}
        ranked = sorted(family_metrics.items(), key=lambda item: float(item[1]["loss"]))
        total_examples = sum(int(metrics["examples_evaluated"]) for metrics in family_metrics.values())
        total_windows = sum(int(metrics["windows_evaluated"]) for metrics in family_metrics.values())
        return {
            "families": family_metrics,
            "best_family": ranked[0][0],
            "worst_family": ranked[-1][0],
            "coverage": {
                "family_count": len(family_metrics),
                "total_examples_evaluated": total_examples,
                "total_windows_evaluated": total_windows,
                "coverage_percent": 100.0,
                "sequence_length": sequence_length,
            },
        }
    finally:
        model.train(previous_training)


def assess_sample_behavior(samples: list[dict[str, Any]]) -> dict[str, Any]:
    generic_refusal_count = 0
    blank_count = 0
    repetitive_count = 0
    unsupported_source_tag_count = 0
    wrong_source_attribution_count = 0
    grounded_abstention_fail_count = 0
    clarification_missing_count = 0
    fabricated_fact_pattern_count = 0
    for sample in samples:
        response = _normalize_text(
            str(sample.get("clean_response") or sample.get("response") or "")
        ).lower()
        if not response:
            blank_count += 1
            continue
        if any(pattern in response for pattern in GENERIC_REFUSAL_PATTERNS):
            generic_refusal_count += 1
        words = response.split()
        if len(words) >= 6:
            most_common = max((words.count(word) for word in set(words)), default=0)
            if most_common / max(len(words), 1) >= 0.35:
                repetitive_count += 1
        grounded_counts = _score_grounded_sample(sample)
        unsupported_source_tag_count += grounded_counts["unsupported_source_tag_count"]
        wrong_source_attribution_count += grounded_counts["wrong_source_attribution_count"]
        grounded_abstention_fail_count += grounded_counts["grounded_abstention_fail_count"]
        clarification_missing_count += grounded_counts["clarification_missing_count"]
        fabricated_fact_pattern_count += grounded_counts["fabricated_fact_pattern_count"]
    blockers: list[str] = []
    if blank_count > 0:
        blockers.append("blank_sample_output")
    if generic_refusal_count > 0:
        blockers.append("generic_refusal_collapse")
    if repetitive_count > 0:
        blockers.append("repetitive_sample_collapse")
    if unsupported_source_tag_count >= 2 or fabricated_fact_pattern_count >= 2:
        blockers.append("unsupported_grounded_claims")
    if grounded_abstention_fail_count > 0:
        blockers.append("grounded_abstention_failures")
    if wrong_source_attribution_count > 0:
        blockers.append("source_attribution_failures")
    return {
        "blank_count": blank_count,
        "generic_refusal_count": generic_refusal_count,
        "repetitive_count": repetitive_count,
        "unsupported_source_tag_count": unsupported_source_tag_count,
        "wrong_source_attribution_count": wrong_source_attribution_count,
        "grounded_abstention_fail_count": grounded_abstention_fail_count,
        "clarification_missing_count": clarification_missing_count,
        "fabricated_fact_pattern_count": fabricated_fact_pattern_count,
        "collapse_detected": bool(
            blank_count
            or generic_refusal_count
            or repetitive_count >= 2
            or wrong_source_attribution_count > 0
            or grounded_abstention_fail_count > 0
            or unsupported_source_tag_count >= 2
        ),
        "promotion_blockers": blockers,
    }


def _raw_words(text: str) -> list[str]:
    return [match.group(0).lower() for match in RAW_WORD_RE.finditer(text)]


def _content_keywords(text: str) -> set[str]:
    return {
        word
        for word in _raw_words(text)
        if len(word) >= 4 and word not in RAW_LM_STOPWORDS and word not in RAW_LM_GENERIC_ATTRACTORS
    }


def _semantic_content_words(words: list[str]) -> list[str]:
    return [word for word in words if len(word) >= 4 and word not in RAW_LM_STOPWORDS]


def _ngram_counts(words: list[str], n: int) -> dict[tuple[str, ...], int]:
    counts: dict[tuple[str, ...], int] = {}
    if len(words) < n:
        return counts
    for index in range(len(words) - n + 1):
        key = tuple(words[index : index + n])
        counts[key] = counts.get(key, 0) + 1
    return counts


def _legible_raw_lm_span(text: str, *, max_words: int | None = None) -> bool:
    words = _raw_words(text)
    if max_words is not None:
        words = words[:max_words]
    if len(words) < 8:
        return False
    span = " ".join(words)
    if len(set(words)) / max(len(words), 1) < 0.32:
        return False
    if WEIRD_HYPHEN_RE.search(text):
        weird_count = len(WEIRD_HYPHEN_RE.findall(text))
        if weird_count / max(len(words), 1) > 0.04:
            return False
    max_word_count = max((_count for _word, _count in ((word, words.count(word)) for word in set(words))), default=0)
    if max_word_count / max(len(words), 1) >= 0.22:
        return False
    return any(char in span for char in "abcdefghijklmnopqrstuvwxyz")


def _repeated_ngram_rate(words: list[str], n: int) -> float:
    counts = _ngram_counts(words, n)
    total = max(len(words) - n + 1, 1)
    repeated_extra = sum(count - 1 for count in counts.values() if count > 1)
    return repeated_extra / total


def _semantic_repetition_metrics(words: list[str]) -> dict[str, Any]:
    content_words = _semantic_content_words(words)
    content_total = len(content_words)
    content_counts = {word: content_words.count(word) for word in set(content_words)}
    dominant_content_word_rate = max(content_counts.values(), default=0) / max(content_total, 1)
    bigram_rate = _repeated_ngram_rate(content_words, 2)
    trigram_rate = _repeated_ngram_rate(content_words, 3)
    semantic_loop_detected = bool(
        content_total >= 8
        and (
            dominant_content_word_rate >= 0.24
            or bigram_rate >= 0.18
            or trigram_rate >= 0.12
            or max(_ngram_counts(content_words, 3).values(), default=0) >= 3
        )
    )
    return {
        "dominant_content_word_rate": dominant_content_word_rate,
        "repeated_content_bigram_rate": bigram_rate,
        "repeated_content_trigram_rate": trigram_rate,
        "semantic_loop_detected": semantic_loop_detected,
    }


def _generic_attractor_category_counts(words: list[str]) -> dict[str, int]:
    return {
        category: sum(1 for word in words if word in terms)
        for category, terms in RAW_LM_GENERIC_ATTRACTOR_CATEGORIES.items()
    }


def _malformed_token_hits(text: str) -> list[str]:
    hits = [match.group(0) for match in MALFORMED_WORD_RE.finditer(text)]
    hits.extend(match.group(0) for match in WEIRD_HYPHEN_RE.finditer(text))
    if QUOTE_SYLLABLE_LOOP_RE.search(text):
        hits.append("quote_syllable_loop")
    small_quoted_tokens = re.findall(r"[\"'“”‘’]([a-zA-Z]{1,4})[\"'“”‘’]", text)
    if len(small_quoted_tokens) >= 6 and len(set(token.lower() for token in small_quoted_tokens)) <= 4:
        hits.append("repeated_quoted_syllables")
    return hits


def _sample_topic_retained(sample: dict[str, Any], response: str) -> bool:
    prompt = str(sample.get("prompt") or "")
    prompt_keywords = _content_keywords(prompt)
    response_keywords = _content_keywords(response)
    if not prompt_keywords:
        return bool(response_keywords)
    overlap = prompt_keywords & response_keywords
    if overlap:
        return True
    bucket = str(sample.get("bucket") or "").lower()
    probe_type = str(sample.get("probe_type") or "").lower()
    if "domain" in probe_type or "catalog" in bucket:
        return bool(response_keywords & RAW_LM_DOMAIN_TERMS)
    return False


def _looks_unfinished(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return True
    if stripped[-1] in ".!?\"'”’)]":
        return False
    tail_words = _raw_words(stripped[-80:])
    if not tail_words:
        return True
    return tail_words[-1] in {
        "a",
        "an",
        "and",
        "as",
        "because",
        "but",
        "for",
        "from",
        "if",
        "in",
        "of",
        "or",
        "the",
        "to",
        "with",
    }


def _domain_phrase_accurate(sample: dict[str, Any], response: str) -> bool | None:
    bucket = str(sample.get("bucket") or "").lower()
    probe_type = str(sample.get("probe_type") or "").lower()
    prompt = str(sample.get("prompt") or "").lower()
    is_domain = "domain" in probe_type or "catalog" in bucket or "catalog" in prompt or "course" in prompt
    if not is_domain:
        return None
    words = set(_raw_words(response))
    if not words & RAW_LM_DOMAIN_TERMS:
        return False
    lower = response.lower()
    boilerplate_hits = sum(1 for phrase in RAW_LM_DOMAIN_BOILERPLATE if phrase in lower)
    if boilerplate_hits >= 2:
        return False
    prompt_words = set(_raw_words(prompt))
    if {"credit", "juniors", "seniors", "open"} & prompt_words or "one-half credit" in prompt:
        return bool(words & RAW_LM_CREDIT_GROUNDING_TERMS)
    if {"recommendation", "recommendations", "background"} & prompt_words or "different from recommendations" in prompt:
        return bool(words & RAW_LM_RECOMMENDATION_GROUNDING_TERMS)
    if "catalog" in bucket or "catalog" in prompt or "course" in prompt or "class" in prompt:
        has_course_context = bool(words & RAW_LM_CATALOG_CONTEXT_TERMS)
        has_grounding = bool(words & RAW_LM_CATALOG_GROUNDING_TERMS)
        return has_course_context and has_grounding
    if "domain" in probe_type:
        return bool(words & (RAW_LM_CATALOG_GROUNDING_TERMS | RAW_LM_RECOMMENDATION_GROUNDING_TERMS))
    return True


def _is_domain_tracking_sample(sample: dict[str, Any]) -> bool:
    bucket = str(sample.get("bucket") or "").lower()
    probe_type = str(sample.get("probe_type") or "").lower()
    prompt = str(sample.get("prompt") or "").lower()
    return "domain" in probe_type or "catalog" in bucket or "catalog" in prompt or "course" in prompt


def _catalog_or_advising_drifted_into_unrelated_prompt(sample: dict[str, Any], response: str) -> bool:
    if _is_domain_tracking_sample(sample):
        return False
    prompt = str(sample.get("prompt") or "").lower()
    lower = response.lower()
    if any(pattern in prompt for pattern in RAW_LM_CATALOG_DRIFT_PATTERNS):
        return False
    return any(pattern in lower for pattern in RAW_LM_CATALOG_DRIFT_PATTERNS)


def assess_raw_lm_sample_behavior(samples: list[dict[str, Any]]) -> dict[str, Any]:
    per_sample: list[dict[str, Any]] = []
    blank_count = 0
    first_40_legible = 0
    full_128_legible = 0
    topic_retained = 0
    repeated_phrase_samples = 0
    max_repeated_4gram_count = 0
    weird_hyphen_samples = 0
    weird_hyphen_total = 0
    unfinished_count = 0
    generic_attractor_samples = 0
    generic_attractor_category_totals = {category: 0 for category in RAW_LM_GENERIC_ATTRACTOR_CATEGORIES}
    domain_boilerplate_samples = 0
    domain_phrase_total = 0
    domain_phrase_accurate = 0
    catalog_domain_drift_count = 0
    narrative_drift_count = 0
    everyday_medical_drift_count = 0
    everyday_product_research_drift_count = 0
    semantic_loop_count = 0
    dominant_content_word_rates: list[float] = []
    repeated_content_bigram_rates: list[float] = []
    repeated_content_trigram_rates: list[float] = []
    malformed_token_samples = 0
    malformed_token_total = 0
    drift_scores: list[float] = []
    cross_sample_domain_boilerplate_counter: dict[str, int] = {}

    for index, sample in enumerate(samples):
        response = _normalize_text(str(sample.get("clean_response") or sample.get("response") or ""))
        lower = response.lower()
        words = _raw_words(response)
        word_count = len(words)
        if not response:
            blank_count += 1
        first_legible = _legible_raw_lm_span(response, max_words=40)
        full_legible = _legible_raw_lm_span(response, max_words=128)
        first_40_legible += int(first_legible)
        full_128_legible += int(full_legible)

        retained = _sample_topic_retained(sample, response)
        topic_retained += int(retained)

        fourgrams = _ngram_counts(words, 4)
        sample_max_4gram = max(fourgrams.values(), default=0)
        max_repeated_4gram_count = max(max_repeated_4gram_count, sample_max_4gram)
        repeated_phrase = sample_max_4gram >= 3
        repeated_phrase_samples += int(repeated_phrase)

        weird_count = len(WEIRD_HYPHEN_RE.findall(response))
        weird_hyphen_total += weird_count
        weird_hyphen_samples += int(weird_count > 0)
        malformed_hits = _malformed_token_hits(response)
        malformed_count = len(malformed_hits)
        malformed_token_total += malformed_count
        malformed_token_samples += int(malformed_count > 0)

        unfinished = _looks_unfinished(response)
        unfinished_count += int(unfinished)

        generic_count = sum(1 for word in words if word in RAW_LM_GENERIC_ATTRACTORS)
        generic_rate = generic_count / max(word_count, 1)
        generic_attractor = generic_rate >= 0.16 or generic_count >= 20
        generic_attractor_samples += int(generic_attractor)
        generic_category_counts = _generic_attractor_category_counts(words)
        for category, count in generic_category_counts.items():
            generic_attractor_category_totals[category] += count

        boilerplate_hits = [phrase for phrase in RAW_LM_DOMAIN_BOILERPLATE if phrase in lower]
        for phrase in boilerplate_hits:
            cross_sample_domain_boilerplate_counter[phrase] = cross_sample_domain_boilerplate_counter.get(phrase, 0) + 1
        domain_boilerplate = bool(boilerplate_hits)
        domain_boilerplate_samples += int(domain_boilerplate)

        phrase_accuracy = _domain_phrase_accurate(sample, response)
        if phrase_accuracy is not None:
            domain_phrase_total += 1
            domain_phrase_accurate += int(phrase_accuracy)

        bucket = str(sample.get("bucket") or "").lower()
        narrative_drift = "narrative" in bucket and any(pattern in lower for pattern in RAW_LM_NARRATIVE_DRIFT_PATTERNS)
        narrative_drift_count += int(narrative_drift)
        prompt_words = set(_raw_words(str(sample.get("prompt") or "")))
        everyday_medical_drift = (
            "everyday" in bucket
            and not (prompt_words & RAW_LM_EVERYDAY_MEDICAL_DRIFT_TERMS)
            and bool(set(words) & RAW_LM_EVERYDAY_MEDICAL_DRIFT_TERMS)
        )
        everyday_medical_drift_count += int(everyday_medical_drift)
        everyday_product_research_drift = (
            "everyday" in bucket
            and not (prompt_words & RAW_LM_EVERYDAY_PRODUCT_RESEARCH_DRIFT_TERMS)
            and sum(1 for word in words if word in RAW_LM_EVERYDAY_PRODUCT_RESEARCH_DRIFT_TERMS) >= 2
        )
        everyday_product_research_drift_count += int(everyday_product_research_drift)
        catalog_domain_drift = _catalog_or_advising_drifted_into_unrelated_prompt(sample, response)
        catalog_domain_drift_count += int(catalog_domain_drift)

        semantic_metrics = _semantic_repetition_metrics(words)
        dominant_content_word_rates.append(float(semantic_metrics["dominant_content_word_rate"]))
        repeated_content_bigram_rates.append(float(semantic_metrics["repeated_content_bigram_rate"]))
        repeated_content_trigram_rates.append(float(semantic_metrics["repeated_content_trigram_rate"]))
        semantic_loop = bool(semantic_metrics["semantic_loop_detected"])
        semantic_loop_count += int(semantic_loop)

        drift_score = 0.0
        drift_score += 0.35 if not retained else 0.0
        drift_score += 0.20 if generic_attractor else 0.0
        drift_score += 0.15 if repeated_phrase else 0.0
        drift_score += 0.15 if malformed_count > 0 or weird_count > 0 else 0.0
        drift_score += 0.15 if semantic_loop else 0.0
        drift_score += 0.10 if narrative_drift or everyday_medical_drift or everyday_product_research_drift or catalog_domain_drift else 0.0
        drift_score += 0.05 if unfinished else 0.0
        drift_scores.append(min(drift_score, 1.0))

        per_sample.append(
            {
                "index": index,
                "id": sample.get("id"),
                "bucket": sample.get("bucket"),
                "probe_type": sample.get("probe_type"),
                "blank": not bool(response),
                "word_count": word_count,
                "first_40_tokens_legible": first_legible,
                "full_128_tokens_legible": full_legible,
                "prompt_topic_retained": retained,
                "max_repeated_4gram_count": sample_max_4gram,
                "weird_hyphen_or_subword_count": weird_count,
                "malformed_token_hits": malformed_hits,
                "malformed_token_count": malformed_count,
                "unfinished_output": unfinished,
                "generic_attractor_rate": round(generic_rate, 6),
                "generic_attractor_category_counts": generic_category_counts,
                "domain_boilerplate_hits": boilerplate_hits,
                "domain_phrase_accurate": phrase_accuracy,
                "catalog_domain_drift": catalog_domain_drift,
                "narrative_expository_drift": narrative_drift,
                "everyday_medical_or_child_drift": everyday_medical_drift,
                "everyday_product_or_research_drift": everyday_product_research_drift,
                "dominant_content_word_rate": round(float(semantic_metrics["dominant_content_word_rate"]), 6),
                "repeated_content_bigram_rate": round(float(semantic_metrics["repeated_content_bigram_rate"]), 6),
                "repeated_content_trigram_rate": round(float(semantic_metrics["repeated_content_trigram_rate"]), 6),
                "semantic_loop_detected": semantic_loop,
                "semantic_drift_score": round(min(drift_score, 1.0), 6),
            }
        )

    sample_count = len(samples)
    domain_boilerplate_repetition_rate = domain_boilerplate_samples / max(sample_count, 1)
    domain_phrase_accuracy = (
        domain_phrase_accurate / max(domain_phrase_total, 1)
        if domain_phrase_total
        else None
    )
    repeated_phrase_rate = repeated_phrase_samples / max(sample_count, 1)
    weird_hyphen_or_subword_rate = weird_hyphen_total / max(sum(len(_raw_words(str(sample.get("clean_response") or sample.get("response") or ""))) for sample in samples), 1)
    malformed_token_rate = malformed_token_total / max(sum(len(_raw_words(str(sample.get("clean_response") or sample.get("response") or ""))) for sample in samples), 1)
    unfinished_output_rate = unfinished_count / max(sample_count, 1)
    generic_attractor_rate = generic_attractor_samples / max(sample_count, 1)
    semantic_drift_score = sum(drift_scores) / max(len(drift_scores), 1)
    max_dominant_content_word_rate = max(dominant_content_word_rates, default=0.0)
    max_repeated_content_bigram_rate = max(repeated_content_bigram_rates, default=0.0)
    max_repeated_content_trigram_rate = max(repeated_content_trigram_rates, default=0.0)
    semantic_loop_detected = semantic_loop_count > 0

    reasons: list[str] = []
    if blank_count:
        reasons.append("blank_sample_output")
    if first_40_legible < min(10, sample_count):
        reasons.append("first_40_tokens_not_legible_enough")
    if topic_retained < min(8, sample_count):
        reasons.append("prompt_topic_retention_too_low")
    if max_repeated_4gram_count >= 4:
        reasons.append("dominant_repeated_4gram")
    if weird_hyphen_or_subword_rate > 0.025 or weird_hyphen_samples >= max(2, sample_count // 5):
        reasons.append("weird_hyphen_or_subword_artifacts")
    if malformed_token_rate > 0.015 or malformed_token_samples >= max(1, sample_count // 6):
        reasons.append("malformed_token_rate_high")
    if domain_phrase_total and domain_boilerplate_samples >= domain_phrase_total:
        reasons.append("domain_boilerplate_repetition")
        reasons.append("boilerplate_repetition")
    if catalog_domain_drift_count > 0:
        reasons.append("catalog_advising_drift_into_unrelated_prompt")
    if semantic_loop_detected or max_repeated_content_bigram_rate >= 0.18 or max_repeated_content_trigram_rate >= 0.12:
        reasons.append("semantic_repetition")
    if narrative_drift_count >= 1:
        reasons.append("narrative_prompts_drift_to_expository_history")
        reasons.append("narrative_to_expository_drift")
    if everyday_medical_drift_count >= 1 or everyday_product_research_drift_count >= 1:
        reasons.append("everyday_prompts_drift_to_medical_body_child_content")
        reasons.append("everyday_to_medical_drift")
    if generic_attractor_rate > 0.4:
        reasons.append("generic_attractor_collapse")
    if semantic_drift_score > 0.45:
        reasons.append("semantic_drift_too_high")
    reasons = list(dict.fromkeys(reasons))

    failure_mode_counts = {
        "topic_drift": sample_count - topic_retained,
        "domain_misuse": (domain_phrase_total - domain_phrase_accurate) + catalog_domain_drift_count,
        "semantic_loop": semantic_loop_count,
        "malformed_generation": malformed_token_samples,
        "genre_collapse": narrative_drift_count + everyday_medical_drift_count + everyday_product_research_drift_count,
        "boilerplate_repetition": domain_boilerplate_samples,
    }

    aggregate = {
        "sample_count": sample_count,
        "blank_count": blank_count,
        "first_40_tokens_legible_count": first_40_legible,
        "first_40_tokens_legible_rate": first_40_legible / max(sample_count, 1),
        "full_128_tokens_legible_count": full_128_legible,
        "full_128_tokens_legible_rate": full_128_legible / max(sample_count, 1),
        "prompt_topic_retention_count": topic_retained,
        "prompt_topic_retention_rate": topic_retained / max(sample_count, 1),
        "repeated_phrase_rate": repeated_phrase_rate,
        "max_repeated_4gram_count": max_repeated_4gram_count,
        "weird_hyphen_or_subword_rate": weird_hyphen_or_subword_rate,
        "malformed_token_rate": malformed_token_rate,
        "malformed_token_sample_count": malformed_token_samples,
        "unfinished_output_rate": unfinished_output_rate,
        "generic_attractor_rate": generic_attractor_rate,
        "generic_attractor_category_counts": generic_attractor_category_totals,
        "domain_boilerplate_repetition_rate": domain_boilerplate_repetition_rate,
        "domain_phrase_accuracy": domain_phrase_accuracy,
        "catalog_domain_drift_count": catalog_domain_drift_count,
        "dominant_content_word_rate": max_dominant_content_word_rate,
        "mean_dominant_content_word_rate": sum(dominant_content_word_rates) / max(len(dominant_content_word_rates), 1),
        "repeated_content_bigram_rate": max_repeated_content_bigram_rate,
        "repeated_content_trigram_rate": max_repeated_content_trigram_rate,
        "semantic_loop_detected": semantic_loop_detected,
        "semantic_loop_sample_count": semantic_loop_count,
        "semantic_drift_score": semantic_drift_score,
        "narrative_expository_drift_count": narrative_drift_count,
        "everyday_medical_or_child_drift_count": everyday_medical_drift_count,
        "everyday_product_or_research_drift_count": everyday_product_research_drift_count,
        "failure_mode_counts": failure_mode_counts,
        "top_domain_boilerplate_phrases": [
            {"phrase": phrase, "count": count}
            for phrase, count in sorted(
                cross_sample_domain_boilerplate_counter.items(),
                key=lambda item: (-item[1], item[0]),
            )[:5]
        ],
    }
    return {
        "raw_lm_quality_gate_passed": not reasons,
        "raw_lm_quality_gate_reasons": reasons,
        "per_sample_quality": per_sample,
        "aggregate_quality_metrics": aggregate,
        **aggregate,
    }


def raw_lm_quality_status(
    short_stable_quality: dict[str, Any],
    long_stress_quality: dict[str, Any],
) -> str:
    short_reasons = set(short_stable_quality.get("raw_lm_quality_gate_reasons", []))
    long_reasons = set(long_stress_quality.get("raw_lm_quality_gate_reasons", []))
    has_major_failure = bool((short_reasons | long_reasons) & RAW_LM_MAJOR_FAILURE_REASONS)
    if has_major_failure:
        return "weak_raw_lm"
    if bool(short_stable_quality.get("raw_lm_quality_gate_passed")) and bool(
        long_stress_quality.get("raw_lm_quality_gate_passed")
    ):
        return "usable_raw_lm"
    short_metrics = short_stable_quality.get("aggregate_quality_metrics", short_stable_quality)
    long_metrics = long_stress_quality.get("aggregate_quality_metrics", long_stress_quality)
    if (
        float(short_metrics.get("first_40_tokens_legible_rate", 0.0)) >= 0.75
        and float(short_metrics.get("prompt_topic_retention_rate", 0.0)) >= 0.55
        and float(long_metrics.get("semantic_drift_score", 1.0)) <= 0.4
        and float(long_metrics.get("generic_attractor_rate", 1.0)) <= 0.3
    ):
        return "improving_raw_lm"
    return "weak_raw_lm"
