from __future__ import annotations

import json
import math
from typing import Any


def _precision_for_key_path(key_path: tuple[str, ...]) -> int:
    for key in reversed(key_path):
        normalized = key.lower()
        if normalized == "lr" or "learning_rate" in normalized or normalized.endswith("_lr"):
            return 8
        if "loss" in normalized:
            return 5
        if "margin" in normalized:
            return 3
    return 2


def _round_float(value: float, *, key_path: tuple[str, ...]) -> float:
    if math.isnan(value) or math.isinf(value):
        return value
    rounded = round(value, _precision_for_key_path(key_path))
    return 0.0 if rounded == 0 else rounded


def format_scalar(value: object, *, key: str | None = None) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"
        key_path = () if key is None else (key,)
        precision = _precision_for_key_path(key_path)
        rounded = _round_float(value, key_path=key_path)
        text = f"{rounded:.{precision}f}".rstrip("0").rstrip(".")
        if "." not in text:
            text += ".0"
        return text
    return str(value)


def round_output_numbers(value: Any, *, key_path: tuple[str, ...] = ()) -> Any:
    if isinstance(value, dict):
        return {key: round_output_numbers(inner, key_path=(*key_path, key)) for key, inner in value.items()}
    if isinstance(value, list):
        return [round_output_numbers(inner, key_path=key_path) for inner in value]
    if isinstance(value, float):
        return _round_float(value, key_path=key_path)
    return value


def dump_rounded_json(value: Any, *, indent: int | None = None) -> str:
    return json.dumps(round_output_numbers(value), indent=indent)


def single_line_text(value: object) -> str:
    text = str(value or "")
    return " ".join(text.split())


def simplify_samples(value: object, *, limit: int | None = 3) -> list[dict[str, str]]:
    if not isinstance(value, list):
        return []
    simplified: list[dict[str, str]] = []
    for sample in value:
        if not isinstance(sample, dict):
            continue
        response = sample.get("response")
        if response is None:
            response = sample.get("clean_response") or sample.get("raw_response", "")
        output = {
            "prompt": str(sample.get("prompt", "")),
            "response": str(response),
        }
        for key in ("id", "bucket", "probe_type"):
            if sample.get(key) is not None:
                output[key] = str(sample.get(key))
        simplified.append(output)
        if limit is not None and len(simplified) >= limit:
            break
    return simplified


def _print_samples_block(
    prefix: str,
    samples_value: object,
    *,
    closing_suffix: str = "",
    limit: int | None = 3,
) -> None:
    samples = simplify_samples(samples_value, limit=limit)
    if not samples:
        print(f"{prefix}; samples: []{closing_suffix}", flush=True)
        return
    print(f"{prefix}; samples: [", flush=True)
    for index, sample in enumerate(samples, start=1):
        prompt = json.dumps(single_line_text(sample.get("prompt", "")), ensure_ascii=True)
        response = json.dumps(single_line_text(sample.get("response", "")), ensure_ascii=True)
        metadata = ""
        if sample.get("id") or sample.get("bucket") or sample.get("probe_type"):
            metadata = (
                f"id: {json.dumps(single_line_text(sample.get('id', '')), ensure_ascii=True)}; "
                f"bucket: {json.dumps(single_line_text(sample.get('bucket', '')), ensure_ascii=True)}; "
                f"probe_type: {json.dumps(single_line_text(sample.get('probe_type', '')), ensure_ascii=True)}; "
            )
        suffix = ";" if index < len(samples) else ""
        print(f"  sample{index}: {{{metadata}prompt: {prompt}; response: {response}}}{suffix}", flush=True)
    print(f"]{closing_suffix}", flush=True)


def print_lm_train_event(payload: dict[str, object]) -> None:
    fields = [
        ("step", payload.get("step")),
        ("loss", payload.get("loss")),
        ("lr", payload.get("lr")),
        ("tokens_seen", payload.get("tokens_seen")),
        ("progress_percent", payload.get("progress_percent")),
        ("stage_elapsed_sec", payload.get("stage_elapsed_sec")),
        ("stage_eta_sec", payload.get("stage_eta_sec")),
    ]
    print("; ".join(f"{name}: {format_scalar(value, key=name)}" for name, value in fields), flush=True)


def print_lm_eval_event(payload: dict[str, object]) -> None:
    metrics = payload.get("eval") if isinstance(payload.get("eval"), dict) else {}
    prefix = (
        f"eval: {{loss: {format_scalar(metrics.get('loss') if isinstance(metrics, dict) else None, key='loss')}; "
        f"perplexity: {format_scalar(metrics.get('perplexity') if isinstance(metrics, dict) else None, key='perplexity')}}}; "
        f"progress_percent: {format_scalar(payload.get('progress_percent'), key='progress_percent')}; "
        f"stage_elapsed_sec: {format_scalar(payload.get('stage_elapsed_sec'), key='stage_elapsed_sec')}; "
        f"stage_eta_sec: {format_scalar(payload.get('stage_eta_sec'), key='stage_eta_sec')}"
    )
    sample_limit = None if payload.get("sample_mode") == "raw_lm" else 3
    _print_samples_block(prefix, payload.get("samples"), limit=sample_limit)


def print_sft_eval_event(payload: dict[str, object]) -> None:
    metrics = payload.get("eval") if isinstance(payload.get("eval"), dict) else {}
    prefix = (
        f"step: {format_scalar(payload.get('step'), key='step')}; "
        f"eval: {{loss: {format_scalar(metrics.get('loss') if isinstance(metrics, dict) else None, key='loss')}; "
        f"perplexity: {format_scalar(metrics.get('perplexity') if isinstance(metrics, dict) else None, key='perplexity')}; "
        f"batches_evaluated: {format_scalar(metrics.get('batches_evaluated') if isinstance(metrics, dict) else None, key='batches_evaluated')}; "
        f"examples_evaluated: {format_scalar(metrics.get('examples_evaluated') if isinstance(metrics, dict) else None, key='examples_evaluated')}}}; "
        f"approx_epoch: {format_scalar(payload.get('approx_epoch'), key='approx_epoch')}; "
        f"train_dataset_size: {format_scalar(payload.get('train_dataset_size'), key='train_dataset_size')}; "
        f"validation_dataset_size: {format_scalar(payload.get('validation_dataset_size'), key='validation_dataset_size')}; "
        f"train_examples_seen: {format_scalar(payload.get('train_examples_seen'), key='train_examples_seen')}; "
        f"validation_examples_evaluated: {format_scalar(payload.get('validation_examples_evaluated'), key='validation_examples_evaluated')}; "
        f"progress_percent: {format_scalar(payload.get('progress_percent'), key='progress_percent')}; "
        f"stage_elapsed_sec: {format_scalar(payload.get('stage_elapsed_sec'), key='stage_elapsed_sec')}; "
        f"stage_eta_sec: {format_scalar(payload.get('stage_eta_sec'), key='stage_eta_sec')}; "
        f"best_step: {format_scalar(payload.get('best_step_so_far'), key='best_step_so_far')}"
    )
    _print_samples_block(prefix, payload.get("samples"))


def print_dpo_train_event(payload: dict[str, object]) -> None:
    fields = [
        ("step", payload.get("step")),
        ("loss", payload.get("loss")),
        ("lr", payload.get("lr")),
        ("train_examples_seen", payload.get("train_examples_seen")),
        ("progress_percent", payload.get("progress_percent")),
        ("stage_elapsed_sec", payload.get("stage_elapsed_sec")),
        ("stage_eta_sec", payload.get("stage_eta_sec")),
    ]
    print("; ".join(f"{name}: {format_scalar(value, key=name)}" for name, value in fields), flush=True)


def print_dpo_eval_event(payload: dict[str, object]) -> None:
    metrics = payload.get("eval") if isinstance(payload.get("eval"), dict) else {}
    samples_value = payload.get("samples")
    if samples_value is None:
        samples_value = payload.get("qualitative_samples")
    prefix = (
        f"step: {format_scalar(payload.get('step'), key='step')}; "
        f"eval: {{loss: {format_scalar(metrics.get('val_dpo_loss', metrics.get('loss')) if isinstance(metrics, dict) else None, key='loss')}; "
        f"preference_accuracy: {format_scalar(metrics.get('preference_accuracy') if isinstance(metrics, dict) else None, key='preference_accuracy')}; "
        f"mean_margin: {format_scalar(metrics.get('mean_margin') if isinstance(metrics, dict) else None, key='mean_margin')}; "
        f"examples_evaluated: {format_scalar(metrics.get('examples_evaluated') if isinstance(metrics, dict) else None, key='examples_evaluated')}}}; "
        f"approx_epoch: {format_scalar(payload.get('approx_epoch'), key='approx_epoch')}; "
        f"train_dataset_size: {format_scalar(payload.get('train_dataset_size'), key='train_dataset_size')}; "
        f"validation_dataset_size: {format_scalar(payload.get('validation_dataset_size'), key='validation_dataset_size')}; "
        f"train_examples_seen: {format_scalar(payload.get('train_examples_seen'), key='train_examples_seen')}; "
        f"validation_examples_evaluated: {format_scalar(payload.get('validation_examples_evaluated'), key='validation_examples_evaluated')}; "
        f"progress_percent: {format_scalar(payload.get('progress_percent'), key='progress_percent')}; "
        f"stage_elapsed_sec: {format_scalar(payload.get('stage_elapsed_sec'), key='stage_elapsed_sec')}; "
        f"stage_eta_sec: {format_scalar(payload.get('stage_eta_sec'), key='stage_eta_sec')}"
    )
    lm_health = payload.get("lm_health") if isinstance(payload.get("lm_health"), dict) else None
    if isinstance(lm_health, dict):
        prefix += (
            f"; llm_health: {{loss: {format_scalar(lm_health.get('loss'), key='loss')}; "
            f"perplexity: {format_scalar(lm_health.get('perplexity'), key='perplexity')}}}"
        )
    prefix += f"; best_step: {format_scalar(payload.get('best_step_so_far'), key='best_step_so_far')}"
    _print_samples_block(prefix, samples_value)
