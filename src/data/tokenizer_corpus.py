from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

from config import TokenizerCorpusConfig
from progress import build_progress_snapshot
from train.console import format_scalar


WHITESPACE_RE = re.compile(r"\s+")


def _progress(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def _log_corpus_event(
    *,
    status: str,
    dataset_name: str,
    dataset_config_name: str | None,
    split: str,
    output_path: Path,
    documents_written: int,
    characters_written: int,
    progress_fraction: float | None,
    elapsed_seconds: float,
    remaining_seconds: float | None,
    max_documents: int,
    max_characters: int,
    metadata_path: Path | None = None,
    stop_reason: str | None = None,
) -> None:
    dataset_label = dataset_name if not dataset_config_name else f"{dataset_name}/{dataset_config_name}"
    fields: list[tuple[str, object]] = [
        ("status", status),
        ("dataset", dataset_label),
        ("split", split),
        ("documents_written", documents_written),
        ("characters_written", characters_written),
        ("max_documents", max_documents),
        ("max_characters", max_characters),
        ("progress_percent", None if progress_fraction is None else progress_fraction * 100.0),
        ("stage_elapsed_sec", elapsed_seconds),
        ("stage_eta_sec", remaining_seconds),
        ("output_path", str(output_path)),
    ]
    if metadata_path is not None:
        fields.append(("metadata_path", str(metadata_path)))
    if stop_reason is not None:
        fields.append(("stop_reason", stop_reason))
    rendered = "; ".join(
        f"{name}: {format_scalar(value, key=name)}"
        for name, value in fields
    )
    _progress(f"WebbGPT: tokenizer-corpus; {rendered}")


def _require_datasets():
    try:
        from datasets import load_dataset  # type: ignore[import-not-found]
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "datasets is required to build the tokenizer corpus. Install it with `pip install datasets`."
        ) from exc
    return load_dataset


def _normalize_text(text: str) -> str:
    return WHITESPACE_RE.sub(" ", text.replace("\x00", " ")).strip()


def build_tokenizer_corpus(config: TokenizerCorpusConfig) -> dict[str, int | str]:
    load_dataset = _require_datasets()
    output_path = Path(config.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    stage_start_time = time.monotonic()
    documents_written = 0
    characters_written = 0
    metadata_path = output_path.with_suffix(output_path.suffix + ".meta.json")
    stop_reason = "dataset_exhausted"

    initial_snapshot = build_progress_snapshot(
        0.0,
        (documents_written, config.max_documents),
        (characters_written, config.max_characters),
    )
    _log_corpus_event(
        status="starting",
        dataset_name=config.dataset_name,
        dataset_config_name=config.dataset_config_name,
        split=config.split,
        output_path=output_path,
        documents_written=documents_written,
        characters_written=characters_written,
        progress_fraction=initial_snapshot.fraction_complete,
        elapsed_seconds=initial_snapshot.elapsed_seconds,
        remaining_seconds=initial_snapshot.remaining_seconds,
        max_documents=config.max_documents,
        max_characters=config.max_characters,
    )

    dataset = load_dataset(
        config.dataset_name,
        config.dataset_config_name,
        split=config.split,
        streaming=config.streaming,
    )

    with output_path.open("w", encoding="utf-8") as handle:
        for row in dataset:
            text = row.get(config.text_field)
            if not isinstance(text, str):
                continue
            if config.normalize_whitespace:
                text = _normalize_text(text)
            if len(text) < config.min_document_chars:
                continue

            handle.write(text)
            handle.write("\n")
            documents_written += 1
            characters_written += len(text) + 1

            if documents_written % 10000 == 0:
                snapshot = build_progress_snapshot(
                    time.monotonic() - stage_start_time,
                    (documents_written, config.max_documents),
                    (characters_written, config.max_characters),
                )
                _log_corpus_event(
                    status="running",
                    dataset_name=config.dataset_name,
                    dataset_config_name=config.dataset_config_name,
                    split=config.split,
                    output_path=output_path,
                    documents_written=documents_written,
                    characters_written=characters_written,
                    progress_fraction=snapshot.fraction_complete,
                    elapsed_seconds=snapshot.elapsed_seconds,
                    remaining_seconds=snapshot.remaining_seconds,
                    max_documents=config.max_documents,
                    max_characters=config.max_characters,
                )

            if documents_written >= config.max_documents:
                stop_reason = "max_documents_reached"
                break
            if characters_written >= config.max_characters:
                stop_reason = "max_characters_reached"
                break

    metadata = {
        "config": config.to_dict(),
        "dataset_name": config.dataset_name,
        "dataset_config_name": config.dataset_config_name,
        "split": config.split,
        "text_field": config.text_field,
        "output_path": str(output_path),
        "streaming": config.streaming,
        "documents_written": documents_written,
        "characters_written": characters_written,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2))
    final_snapshot = build_progress_snapshot(
        time.monotonic() - stage_start_time,
        (documents_written, config.max_documents),
        (characters_written, config.max_characters),
    )
    _log_corpus_event(
        status="finished",
        dataset_name=config.dataset_name,
        dataset_config_name=config.dataset_config_name,
        split=config.split,
        output_path=output_path,
        documents_written=documents_written,
        characters_written=characters_written,
        progress_fraction=final_snapshot.fraction_complete,
        elapsed_seconds=final_snapshot.elapsed_seconds,
        remaining_seconds=final_snapshot.remaining_seconds,
        max_documents=config.max_documents,
        max_characters=config.max_characters,
        metadata_path=metadata_path,
        stop_reason=stop_reason,
    )
    return metadata
