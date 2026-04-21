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
PRETRAIN_REGRESSION_PATH = "data/eval/pretrain_regression.jsonl"
PRETRAIN_FAMILY_HOLDOUTS_PATH = "data/eval/pretrain_family_holdouts.json"
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
                "prompt": prompt,
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
        handle.write(json.dumps(payload) + "\n")


def write_selection_metadata(best_dir: str | Path, payload: dict[str, Any]) -> None:
    target_dir = Path(best_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    (target_dir / "selection.json").write_text(json.dumps(payload, indent=2))


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
    metadata_path.write_text(json.dumps(kept, indent=2))
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
    limit: int = 3,
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
                "windows_evaluated": len(losses),
            }
        if not family_metrics:
            return {"families": {}, "best_family": None, "worst_family": None}
        ranked = sorted(family_metrics.items(), key=lambda item: float(item[1]["loss"]))
        return {
            "families": family_metrics,
            "best_family": ranked[0][0],
            "worst_family": ranked[-1][0],
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
