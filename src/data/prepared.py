from __future__ import annotations

import bisect
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Iterable

from tokenizer import SentencePieceTokenizer, format_chat


def _progress(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def _require_numpy():
    try:
        import numpy as np
    except ModuleNotFoundError as exc:
        raise RuntimeError("numpy is required for prepared dataset artifacts.") from exc
    return np


def _require_torch():
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError("torch is required for prepared dataset loading.") from exc
    return torch


def load_prepared_manifest(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def _artifact_dir(manifest_path: str | Path) -> Path:
    manifest = Path(manifest_path)
    return manifest.with_suffix("")


def prepared_resume_state_path(manifest_path: str | Path) -> Path:
    manifest = Path(manifest_path)
    return manifest.with_suffix(".resume.json")


def prepared_resume_dir(manifest_path: str | Path) -> Path:
    manifest = Path(manifest_path)
    return manifest.with_suffix(".resume")


def build_input_fingerprint(
    *,
    stage: str,
    kind: str,
    tokenizer_path: str,
    sequence_length: int,
    rows_per_shard: int,
    source_snapshots: list[dict[str, Any]],
    token_budget: int | None = None,
    extra: dict[str, Any] | None = None,
) -> str:
    payload = {
        "stage": stage,
        "kind": kind,
        "tokenizer_path": tokenizer_path,
        "sequence_length": sequence_length,
        "rows_per_shard": rows_per_shard,
        "source_snapshots": source_snapshots,
        "token_budget": token_budget,
        "extra": extra or {},
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _atomic_write_text(path: str | Path, content: str) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_target = target.with_name(f"{target.name}.tmp")
    tmp_target.write_text(content)
    tmp_target.replace(target)


def _atomic_write_json(path: str | Path, payload: dict[str, Any]) -> None:
    _atomic_write_text(path, json.dumps(payload, indent=2))


def save_prepared_manifest(path: str | Path, payload: dict[str, Any]) -> None:
    _atomic_write_json(path, payload)


def save_resume_state(path: str | Path, payload: dict[str, Any]) -> None:
    _atomic_write_json(path, payload)


def _atomic_save_array(path: str | Path, rows: list[list[int]]) -> None:
    np = _require_numpy()
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_target = target.with_name(f"{target.stem}.tmp{target.suffix}")
    np.save(tmp_target, np.asarray(rows, dtype=np.int32), allow_pickle=False)
    tmp_target.replace(target)


def _atomic_write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_target = target.with_name(f"{target.name}.tmp")
    with tmp_target.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    tmp_target.replace(target)


def _save_buffer_rows(path: str | Path, rows: list[list[int]]) -> str | None:
    if not rows:
        return None
    _atomic_save_array(path, rows)
    return str(path)


def save_buffer_rows(path: str | Path, rows: list[list[int]]) -> str | None:
    return _save_buffer_rows(path, rows)


def load_buffer_rows(path: str | Path | None) -> list[list[int]]:
    if not path:
        return []
    np = _require_numpy()
    buffer_path = Path(path)
    if not buffer_path.exists():
        raise RuntimeError(f"Prepared-data resume buffer is missing: {buffer_path}")
    array = np.load(buffer_path, mmap_mode=None)
    return array.astype("int32").tolist()


def save_metadata_rows(path: str | Path, rows: list[dict[str, Any]]) -> str | None:
    if not rows:
        return None
    _atomic_write_jsonl(path, rows)
    return str(path)


def load_metadata_rows(path: str | Path | None) -> list[dict[str, Any]]:
    if not path:
        return []
    metadata_path = Path(path)
    if not metadata_path.exists():
        raise RuntimeError(f"Prepared-data metadata sidecar is missing: {metadata_path}")
    rows: list[dict[str, Any]] = []
    with metadata_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def append_hash_chunk(path: str | Path, hashes: list[str]) -> str | None:
    if not hashes:
        return None
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_target = target.with_name(f"{target.name}.tmp")
    tmp_target.write_text("".join(f"{value}\n" for value in hashes))
    tmp_target.replace(target)
    return str(target)


def load_seen_hashes(paths: list[str]) -> set[str]:
    seen: set[str] = set()
    for raw_path in paths:
        path = Path(raw_path)
        if not path.exists():
            raise RuntimeError(f"Prepared-data dedupe hash chunk is missing: {path}")
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                value = line.strip()
                if value:
                    seen.add(value)
    return seen


def cleanup_prepare_outputs(manifest_path: str | Path) -> None:
    manifest = Path(manifest_path)
    shard_dir = _artifact_dir(manifest)
    resume_state = prepared_resume_state_path(manifest)
    resume_dir = prepared_resume_dir(manifest)
    if manifest.exists():
        manifest.unlink()
    if resume_state.exists():
        resume_state.unlink()
    if shard_dir.exists():
        shutil.rmtree(shard_dir)
    if resume_dir.exists():
        shutil.rmtree(resume_dir)


def remove_resume_artifacts(manifest_path: str | Path) -> None:
    state_path = prepared_resume_state_path(manifest_path)
    resume_dir = prepared_resume_dir(manifest_path)
    if state_path.exists():
        state_path.unlink()
    if resume_dir.exists():
        shutil.rmtree(resume_dir)


def stage_has_partial_outputs(manifest_path: str | Path) -> bool:
    shard_dir = _artifact_dir(manifest_path)
    if not shard_dir.exists():
        return False
    return any(shard_dir.iterdir())


def validate_resume_state_files(state: dict[str, Any]) -> None:
    for shard in state.get("shards", []):
        for key in (
            "path",
            "input_ids_path",
            "labels_path",
            "chosen_input_ids_path",
            "rejected_input_ids_path",
            "metadata_path",
        ):
            raw_path = shard.get(key)
            if raw_path and not Path(raw_path).exists():
                raise RuntimeError(f"Prepared-data resume shard is missing: {raw_path}")
    for key in (
        "rows_buffer_path",
        "input_buffer_path",
        "label_buffer_path",
        "chosen_buffer_path",
        "rejected_buffer_path",
        "metadata_buffer_path",
    ):
        raw_path = state.get(key)
        if raw_path and not Path(raw_path).exists():
            raise RuntimeError(f"Prepared-data resume buffer is missing: {raw_path}")
    for raw_path in state.get("dedupe_hash_chunks", []):
        if not Path(raw_path).exists():
            raise RuntimeError(f"Prepared-data dedupe hash chunk is missing: {raw_path}")


def prepared_manifest_trust_flags(manifest: dict[str, Any]) -> list[str]:
    version = str(manifest.get("version", "1.0"))
    kind = manifest.get("kind")
    flags: list[str] = []
    if kind in {"sft", "preference"} and not version.startswith("2."):
        flags.extend(["behavior_eval_untrusted", "overlap_guard_skipped"])
    return flags


def prepared_manifest_supports_prompt_overlap(manifest: dict[str, Any]) -> bool:
    return not prepared_manifest_trust_flags(manifest)


def derive_artifact_status(
    blockers: list[str] | set[str],
    *,
    base_status: str = "promotable",
) -> str:
    blocker_set = {value for value in blockers if value}
    if base_status == "blocked":
        return "blocked"
    if {"lineage_ambiguous", "lm_health_regressed"} & blocker_set:
        return "blocked"
    if base_status == "dev_only" or blocker_set:
        return "dev_only"
    return "promotable"


def _flush_single_array_shard(
    rows: list[list[int]],
    *,
    shard_dir: Path,
    prefix: str,
    shard_index: int,
) -> dict[str, Any]:
    shard_path = shard_dir / f"{prefix}-{shard_index:05d}.npy"
    _atomic_save_array(shard_path, rows)
    return {"path": str(shard_path), "rows": len(rows)}


def _flush_double_array_shard(
    rows_a: list[list[int]],
    rows_b: list[list[int]],
    *,
    shard_dir: Path,
    prefix_a: str,
    prefix_b: str,
    shard_index: int,
) -> dict[str, Any]:
    shard_a = shard_dir / f"{prefix_a}-{shard_index:05d}.npy"
    shard_b = shard_dir / f"{prefix_b}-{shard_index:05d}.npy"
    _atomic_save_array(shard_a, rows_a)
    _atomic_save_array(shard_b, rows_b)
    return {"path_a": str(shard_a), "path_b": str(shard_b), "rows": len(rows_a)}


def _pad_ids(token_ids: list[int], *, sequence_length: int, pad_token_id: int) -> list[int]:
    if len(token_ids) >= sequence_length:
        return token_ids[:sequence_length]
    return token_ids + [pad_token_id] * (sequence_length - len(token_ids))


def _assistant_only_labels(token_ids: list[int], tokenizer: SentencePieceTokenizer) -> list[int]:
    assistant_prefix_ids = tokenizer.encode("<|assistant|>\n", add_bos=False, add_eos=False)
    assistant_id = tokenizer.token_to_id("<|assistant|>")
    eos_id = tokenizer.token_to_id("</s>")
    labels = [-100] * len(token_ids)
    index = 0
    while index < len(token_ids):
        if token_ids[index] != assistant_id:
            index += 1
            continue
        content_start = min(index + len(assistant_prefix_ids), len(token_ids))
        end = content_start
        while end < len(token_ids) and token_ids[end] != eos_id:
            end += 1
        if end < len(token_ids):
            end += 1
        for position in range(content_start, end):
            labels[position] = token_ids[position]
        index = max(end, index + 1)
    return labels


def encode_sft_messages(
    messages: list[dict[str, str]],
    tokenizer: SentencePieceTokenizer,
    sequence_length: int,
) -> tuple[list[int], list[int]]:
    pad_id = tokenizer.token_to_id("<pad>")
    rendered = format_chat(messages, add_generation_prompt=False)
    token_ids = tokenizer.encode(rendered, add_bos=True, add_eos=False)
    token_ids = token_ids[:sequence_length]
    labels = _assistant_only_labels(token_ids, tokenizer)[:sequence_length]
    return (
        _pad_ids(token_ids, sequence_length=sequence_length, pad_token_id=pad_id),
        labels + [-100] * (sequence_length - len(labels)),
    )


def encode_preference_example(
    prompt: list[dict[str, str]],
    answer: str,
    tokenizer: SentencePieceTokenizer,
    sequence_length: int,
) -> list[int]:
    pad_id = tokenizer.token_to_id("<pad>")
    prompt_text = format_chat(prompt, add_generation_prompt=True)
    full_text = f"{prompt_text}{answer.strip()}\n</s>"
    token_ids = tokenizer.encode(full_text, add_bos=True, add_eos=False)
    return _pad_ids(token_ids, sequence_length=sequence_length, pad_token_id=pad_id)


def write_packed_lm_artifacts(
    *,
    stage: str,
    token_sequences: Iterable[list[int]],
    tokenizer_path: str,
    sequence_length: int,
    pad_token_id: int,
    eos_token_id: int,
    output_path: str | Path,
    rows_per_shard: int,
    source_snapshots: list[dict[str, Any]],
    token_budget: int | None = None,
    input_fingerprint: str | None = None,
) -> dict[str, Any]:
    manifest_path = Path(output_path)
    shard_dir = _artifact_dir(manifest_path)
    shard_dir.mkdir(parents=True, exist_ok=True)

    rows: list[list[int]] = []
    shards: list[dict[str, Any]] = []
    shard_index = 0
    num_sequences = 0
    num_tokens = 0
    last_heartbeat = time.monotonic()
    budget_text = "unbounded" if token_budget is None else f"{token_budget:,} tokens"
    _progress(
        f"WebbGPT: preparing {stage} packed LM artifacts "
        f"(sequence_length={sequence_length}, rows_per_shard={rows_per_shard}, token_budget={budget_text})."
    )

    for sequence in token_sequences:
        rows.append(sequence)
        num_sequences += 1
        num_tokens += sum(token != pad_token_id for token in sequence)
        if len(rows) >= rows_per_shard:
            shards.append(
                _flush_single_array_shard(
                    rows,
                    shard_dir=shard_dir,
                    prefix="shard",
                    shard_index=shard_index,
                )
            )
            _progress(
                f"WebbGPT: preparing {stage}: wrote shard {shard_index + 1} "
                f"({num_sequences:,} sequences, {num_tokens:,} packed tokens so far)."
            )
            rows = []
            shard_index += 1
        if token_budget is not None and num_tokens >= token_budget:
            break
        now = time.monotonic()
        if now - last_heartbeat >= 30:
            _progress(
                f"WebbGPT: preparing {stage} is still running "
                f"({num_sequences:,} sequences, {num_tokens:,} packed tokens, {len(shards):,} shards written)."
            )
            last_heartbeat = now

    if rows:
        shards.append(
            _flush_single_array_shard(
                rows,
                shard_dir=shard_dir,
                prefix="shard",
                shard_index=shard_index,
            )
        )
        _progress(
            f"WebbGPT: preparing {stage}: wrote final shard {shard_index + 1} "
            f"({num_sequences:,} sequences, {num_tokens:,} packed tokens total)."
        )

    manifest = {
        "version": "1.0",
        "stage": stage,
        "kind": "packed_lm",
        "input_fingerprint": input_fingerprint,
        "tokenizer_path": tokenizer_path,
        "sequence_length": sequence_length,
        "pad_token_id": pad_token_id,
        "eos_token_id": eos_token_id,
        "num_sequences": num_sequences,
        "num_tokens": num_tokens,
        "source_snapshots": source_snapshots,
        "shards": shards,
    }
    save_prepared_manifest(manifest_path, manifest)
    _progress(
        f"WebbGPT: finished preparing {stage} "
        f"({num_sequences:,} sequences across {len(shards):,} shards, {num_tokens:,} packed tokens)."
    )
    return manifest


def write_sft_artifacts(
    *,
    stage: str,
    examples: Iterable[list[dict[str, str]]],
    tokenizer_path: str,
    sequence_length: int,
    output_path: str | Path,
    rows_per_shard: int,
    source_snapshots: list[dict[str, Any]],
    input_fingerprint: str | None = None,
) -> dict[str, Any]:
    manifest_path = Path(output_path)
    shard_dir = _artifact_dir(manifest_path)
    shard_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = SentencePieceTokenizer(tokenizer_path)

    input_rows: list[list[int]] = []
    label_rows: list[list[int]] = []
    shards: list[dict[str, Any]] = []
    shard_index = 0
    num_examples = 0
    num_label_tokens = 0
    last_heartbeat = time.monotonic()
    total_examples = len(examples) if hasattr(examples, "__len__") else None

    _progress(
        f"WebbGPT: preparing {stage} SFT artifacts "
        f"(sequence_length={sequence_length}, rows_per_shard={rows_per_shard})."
    )

    for messages in examples:
        input_ids, labels = encode_sft_messages(messages, tokenizer, sequence_length)
        input_rows.append(input_ids)
        label_rows.append(labels)
        num_examples += 1
        num_label_tokens += sum(label != -100 for label in labels)
        if len(input_rows) >= rows_per_shard:
            shard = _flush_double_array_shard(
                input_rows,
                label_rows,
                shard_dir=shard_dir,
                prefix_a="input_ids",
                prefix_b="labels",
                shard_index=shard_index,
            )
            shards.append(
                {
                    "input_ids_path": shard["path_a"],
                    "labels_path": shard["path_b"],
                    "rows": shard["rows"],
                }
            )
            _progress(
                f"WebbGPT: preparing {stage}: wrote shard {shard_index + 1} "
                f"({num_examples:,} examples, {num_label_tokens:,} supervised tokens so far)."
            )
            input_rows = []
            label_rows = []
            shard_index += 1
        now = time.monotonic()
        if now - last_heartbeat >= 30:
            _progress(
                f"WebbGPT: preparing {stage} is still running "
                f"({num_examples:,} examples, {num_label_tokens:,} supervised tokens, {len(shards):,} shards written)."
            )
            last_heartbeat = now

    if input_rows:
        shard = _flush_double_array_shard(
            input_rows,
            label_rows,
            shard_dir=shard_dir,
            prefix_a="input_ids",
            prefix_b="labels",
            shard_index=shard_index,
        )
        shards.append(
            {
                "input_ids_path": shard["path_a"],
                "labels_path": shard["path_b"],
                "rows": shard["rows"],
            }
        )
        _progress(
            f"WebbGPT: preparing {stage}: wrote final shard {shard_index + 1} "
            f"({num_examples:,} examples total)."
        )

    manifest = {
        "version": "1.0",
        "stage": stage,
        "kind": "sft",
        "input_fingerprint": input_fingerprint,
        "tokenizer_path": tokenizer_path,
        "sequence_length": sequence_length,
        "pad_token_id": tokenizer.token_to_id("<pad>"),
        "num_examples": num_examples,
        "num_label_tokens": num_label_tokens,
        "source_snapshots": source_snapshots,
        "shards": shards,
    }
    save_prepared_manifest(manifest_path, manifest)
    _progress(
        f"WebbGPT: finished preparing {stage} "
        f"({num_examples:,} examples across {len(shards):,} shards, {num_label_tokens:,} supervised tokens)."
    )
    return manifest


def write_preference_artifacts(
    *,
    stage: str,
    examples: Iterable[tuple[list[dict[str, str]], str, str]],
    tokenizer_path: str,
    sequence_length: int,
    output_path: str | Path,
    rows_per_shard: int,
    source_snapshots: list[dict[str, Any]],
    input_fingerprint: str | None = None,
) -> dict[str, Any]:
    manifest_path = Path(output_path)
    shard_dir = _artifact_dir(manifest_path)
    shard_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = SentencePieceTokenizer(tokenizer_path)

    chosen_rows: list[list[int]] = []
    rejected_rows: list[list[int]] = []
    shards: list[dict[str, Any]] = []
    shard_index = 0
    num_examples = 0
    last_heartbeat = time.monotonic()
    total_examples = len(examples) if hasattr(examples, "__len__") else None

    _progress(
        f"WebbGPT: preparing {stage} preference artifacts "
        f"(sequence_length={sequence_length}, rows_per_shard={rows_per_shard})."
    )

    for prompt, chosen, rejected in examples:
        chosen_rows.append(encode_preference_example(prompt, chosen, tokenizer, sequence_length))
        rejected_rows.append(encode_preference_example(prompt, rejected, tokenizer, sequence_length))
        num_examples += 1
        if len(chosen_rows) >= rows_per_shard:
            shard = _flush_double_array_shard(
                chosen_rows,
                rejected_rows,
                shard_dir=shard_dir,
                prefix_a="chosen_input_ids",
                prefix_b="rejected_input_ids",
                shard_index=shard_index,
            )
            shards.append(
                {
                    "chosen_input_ids_path": shard["path_a"],
                    "rejected_input_ids_path": shard["path_b"],
                    "rows": shard["rows"],
                }
            )
            _progress(
                f"WebbGPT: preparing {stage}: wrote shard {shard_index + 1} "
                f"({num_examples:,} preference examples so far)."
            )
            chosen_rows = []
            rejected_rows = []
            shard_index += 1
        now = time.monotonic()
        if now - last_heartbeat >= 30:
            _progress(
                f"WebbGPT: preparing {stage} is still running "
                f"({num_examples:,} preference examples, {len(shards):,} shards written)."
            )
            last_heartbeat = now

    if chosen_rows:
        shard = _flush_double_array_shard(
            chosen_rows,
            rejected_rows,
            shard_dir=shard_dir,
            prefix_a="chosen_input_ids",
            prefix_b="rejected_input_ids",
            shard_index=shard_index,
        )
        shards.append(
            {
                "chosen_input_ids_path": shard["path_a"],
                "rejected_input_ids_path": shard["path_b"],
                "rows": shard["rows"],
            }
        )
        _progress(
            f"WebbGPT: preparing {stage}: wrote final shard {shard_index + 1} "
            f"({num_examples:,} preference examples total)."
        )

    manifest = {
        "version": "1.0",
        "stage": stage,
        "kind": "preference",
        "input_fingerprint": input_fingerprint,
        "tokenizer_path": tokenizer_path,
        "sequence_length": sequence_length,
        "pad_token_id": tokenizer.token_to_id("<pad>"),
        "num_examples": num_examples,
        "source_snapshots": source_snapshots,
        "shards": shards,
    }
    save_prepared_manifest(manifest_path, manifest)
    _progress(
        f"WebbGPT: finished preparing {stage} "
        f"({num_examples:,} examples across {len(shards):,} shards)."
    )
    return manifest


class _PreparedDatasetBase:
    def __init__(self, manifest_path: str | Path):
        self.manifest_path = str(manifest_path)
        self.manifest = load_prepared_manifest(manifest_path)
        self.trust_flags = prepared_manifest_trust_flags(self.manifest)
        self.artifact_status = derive_artifact_status(self.trust_flags)
        self.sequence_length = int(self.manifest["sequence_length"])
        self.pad_token_id = int(self.manifest["pad_token_id"])
        self.shards = list(self.manifest["shards"])
        self._offsets: list[int] = []
        total = 0
        for shard in self.shards:
            self._offsets.append(total)
            total += int(shard["rows"])
        self._length = total

    def __len__(self) -> int:
        return self._length

    def _resolve_index(self, index: int) -> tuple[int, int]:
        if index < 0 or index >= self._length:
            raise IndexError(index)
        shard_index = bisect.bisect_right(self._offsets, index) - 1
        row_index = index - self._offsets[shard_index]
        return shard_index, row_index


class PreparedPackedDataset(_PreparedDatasetBase):
    def __init__(self, manifest_path: str | Path):
        super().__init__(manifest_path)
        if self.manifest.get("kind") != "packed_lm":
            raise ValueError(f"{manifest_path} is not a packed LM manifest.")
        self._arrays: dict[int, Any] = {}
        self._metadata_rows: dict[int, list[dict[str, Any]]] = {}

    def _array(self, shard_index: int):
        np = _require_numpy()
        if shard_index not in self._arrays:
            self._arrays[shard_index] = np.load(self.shards[shard_index]["path"], mmap_mode="r")
        return self._arrays[shard_index]

    def _metadata(self, shard_index: int) -> list[dict[str, Any]]:
        if shard_index not in self._metadata_rows:
            metadata_path = self.shards[shard_index].get("metadata_path")
            self._metadata_rows[shard_index] = load_metadata_rows(metadata_path)
        return self._metadata_rows[shard_index]

    def __getitem__(self, index: int) -> dict[str, Any]:
        torch = _require_torch()
        shard_index, row_index = self._resolve_index(index)
        sequence = torch.tensor(self._array(shard_index)[row_index], dtype=torch.long)
        attention_mask = (sequence != self.pad_token_id).long()
        labels = sequence.clone()
        labels[attention_mask == 0] = -100
        metadata = {}
        metadata_rows = self._metadata(shard_index)
        if row_index < len(metadata_rows):
            metadata = dict(metadata_rows[row_index])
        return {
            "input_ids": sequence,
            "attention_mask": attention_mask,
            "labels": labels,
            "provenance_json": json.dumps(
                {
                    "shard_index": shard_index,
                    "row_index": row_index,
                    "source_names": list(metadata.get("source_names", [])),
                    "contributors": list(metadata.get("contributors", [])),
                    "packed_document_count": int(metadata.get("packed_document_count", 0)),
                },
                sort_keys=True,
            ),
        }


class PreparedSFTDataset(_PreparedDatasetBase):
    def __init__(self, manifest_path: str | Path):
        super().__init__(manifest_path)
        if self.manifest.get("kind") != "sft":
            raise ValueError(f"{manifest_path} is not an SFT manifest.")
        self._input_arrays: dict[int, Any] = {}
        self._label_arrays: dict[int, Any] = {}
        self.examples = self._build_examples()

    def _build_examples(self):
        if not prepared_manifest_supports_prompt_overlap(self.manifest):
            return None
        from data.schemas import SFTExample

        examples = []
        for shard in self.shards:
            for row in load_metadata_rows(shard.get("metadata_path")):
                examples.append(
                    SFTExample(
                        messages=[],
                        source=str(row.get("source", "prepared")),
                        example_id=row.get("example_id"),
                        split_group_id=row.get("split_group_id"),
                        metadata=row,
                    )
                )
        return examples

    def _input_array(self, shard_index: int):
        np = _require_numpy()
        if shard_index not in self._input_arrays:
            self._input_arrays[shard_index] = np.load(
                self.shards[shard_index]["input_ids_path"], mmap_mode="r"
            )
        return self._input_arrays[shard_index]

    def _label_array(self, shard_index: int):
        np = _require_numpy()
        if shard_index not in self._label_arrays:
            self._label_arrays[shard_index] = np.load(
                self.shards[shard_index]["labels_path"], mmap_mode="r"
            )
        return self._label_arrays[shard_index]

    def __getitem__(self, index: int) -> dict[str, Any]:
        torch = _require_torch()
        shard_index, row_index = self._resolve_index(index)
        input_ids = torch.tensor(self._input_array(shard_index)[row_index], dtype=torch.long)
        labels = torch.tensor(self._label_array(shard_index)[row_index], dtype=torch.long)
        attention_mask = (input_ids != self.pad_token_id).long()
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


class PreparedPreferenceDataset(_PreparedDatasetBase):
    def __init__(self, manifest_path: str | Path):
        super().__init__(manifest_path)
        if self.manifest.get("kind") != "preference":
            raise ValueError(f"{manifest_path} is not a preference manifest.")
        self._chosen_arrays: dict[int, Any] = {}
        self._rejected_arrays: dict[int, Any] = {}
        self.examples = self._build_examples()

    def _build_examples(self):
        if not prepared_manifest_supports_prompt_overlap(self.manifest):
            return None
        from data.schemas import PreferenceExample

        examples = []
        for shard in self.shards:
            for row in load_metadata_rows(shard.get("metadata_path")):
                examples.append(
                    PreferenceExample(
                        prompt=[],
                        chosen="",
                        rejected="",
                        source=str(row.get("source", "prepared")),
                        example_id=row.get("example_id"),
                        split_group_id=row.get("split_group_id"),
                        metadata=row,
                    )
                )
        return examples

    def _chosen_array(self, shard_index: int):
        np = _require_numpy()
        if shard_index not in self._chosen_arrays:
            self._chosen_arrays[shard_index] = np.load(
                self.shards[shard_index]["chosen_input_ids_path"], mmap_mode="r"
            )
        return self._chosen_arrays[shard_index]

    def _rejected_array(self, shard_index: int):
        np = _require_numpy()
        if shard_index not in self._rejected_arrays:
            self._rejected_arrays[shard_index] = np.load(
                self.shards[shard_index]["rejected_input_ids_path"], mmap_mode="r"
            )
        return self._rejected_arrays[shard_index]

    def __getitem__(self, index: int) -> dict[str, Any]:
        torch = _require_torch()
        shard_index, row_index = self._resolve_index(index)
        chosen_input_ids = torch.tensor(self._chosen_array(shard_index)[row_index], dtype=torch.long)
        rejected_input_ids = torch.tensor(
            self._rejected_array(shard_index)[row_index], dtype=torch.long
        )
        return {
            "chosen_input_ids": chosen_input_ids,
            "rejected_input_ids": rejected_input_ids,
            "chosen_attention_mask": (chosen_input_ids != self.pad_token_id).long(),
            "rejected_attention_mask": (rejected_input_ids != self.pad_token_id).long(),
        }
