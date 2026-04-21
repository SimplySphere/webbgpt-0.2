from __future__ import annotations

import json
import sys
import threading
import time
from pathlib import Path
from typing import Iterable

from config import TokenizerConfig
from train.console import format_scalar


def _require_sentencepiece():
    try:
        import sentencepiece as spm  # type: ignore[import-not-found]
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "sentencepiece is required for tokenizer training and inference. "
            "Install it with `pip install sentencepiece`."
        ) from exc
    return spm


class SentencePieceTokenizer:
    def __init__(self, model_path: str | Path):
        spm = _require_sentencepiece()
        self.model_path = str(model_path)
        self.processor = spm.SentencePieceProcessor(model_file=self.model_path)
        self.special_tokens = self._load_special_tokens()
        self._special_token_to_id = {
            token: int(self.processor.piece_to_id(token))
            for token in self.special_tokens
            if int(self.processor.piece_to_id(token)) >= 0
        }
        self._special_tokens_sorted = sorted(self._special_token_to_id.keys(), key=len, reverse=True)

    @property
    def vocab_size(self) -> int:
        return int(self.processor.vocab_size())

    @property
    def bos_token_id(self) -> int:
        return int(self.processor.bos_id())

    @property
    def eos_token_id(self) -> int:
        return int(self.processor.eos_id())

    @property
    def pad_token_id(self) -> int:
        return int(self.processor.pad_id())

    def _load_special_tokens(self) -> list[str]:
        meta_path = Path(self.model_path).with_suffix(".tokenizer.json")
        if meta_path.exists():
            try:
                payload = json.loads(meta_path.read_text())
                special_tokens = list((payload.get("special_tokens") or {}).values())
                if special_tokens:
                    return special_tokens
            except Exception:
                pass
        return list(TokenizerConfig().special_tokens.values())

    def _encode_plain(self, text: str) -> list[int]:
        if not text:
            return []
        return list(self.processor.encode(text, out_type=int, add_bos=False, add_eos=False))

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        token_ids: list[int] = []
        if add_bos and self.bos_token_id >= 0:
            token_ids.append(self.bos_token_id)

        index = 0
        while index < len(text):
            matched = False
            for token in self._special_tokens_sorted:
                if text.startswith(token, index):
                    token_ids.append(self._special_token_to_id[token])
                    index += len(token)
                    matched = True
                    break
            if matched:
                continue

            next_special_index = min(
                (
                    position
                    for token in self._special_tokens_sorted
                    if (position := text.find(token, index)) != -1
                ),
                default=-1,
            )
            if next_special_index == -1:
                token_ids.extend(self._encode_plain(text[index:]))
                break
            token_ids.extend(self._encode_plain(text[index:next_special_index]))
            index = next_special_index

        if add_eos and self.eos_token_id >= 0:
            token_ids.append(self.eos_token_id)
        return token_ids

    def decode(self, ids: Iterable[int]) -> str:
        return str(self.processor.decode(list(ids)))

    def token_to_id(self, token: str) -> int:
        return int(self.processor.piece_to_id(token))

    def id_to_token(self, idx: int) -> str:
        return str(self.processor.id_to_piece(idx))


def _describe_input_files(input_files: list[str]) -> tuple[int, int]:
    existing_files = 0
    total_bytes = 0
    for raw_path in input_files:
        path = Path(raw_path)
        if not path.exists():
            continue
        existing_files += 1
        total_bytes += path.stat().st_size
    return existing_files, total_bytes


def _format_bytes(num_bytes: int) -> str:
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    size = float(num_bytes)
    for unit in units:
        if size < 1024 or unit == units[-1]:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{num_bytes} B"


def _log_tokenizer_event(
    *,
    status: str,
    input_file_count: int,
    input_bytes: int,
    vocab_size: int,
    model_type: str,
    model_path: Path,
    metadata_path: Path,
    elapsed_seconds: float,
    remaining_seconds: float | None,
) -> None:
    fields: list[tuple[str, object]] = [
        ("status", status),
        ("input_files", input_file_count),
        ("input_bytes", _format_bytes(input_bytes)),
        ("vocab_size", vocab_size),
        ("model_type", model_type),
        ("progress_percent", 100.0 if status == "finished" else None),
        ("stage_elapsed_sec", elapsed_seconds),
        ("stage_eta_sec", remaining_seconds),
        ("model_path", str(model_path)),
        ("metadata_path", str(metadata_path)),
    ]
    rendered = "; ".join(
        f"{name}: {format_scalar(value, key=name)}"
        for name, value in fields
    )
    print(f"WebbGPT: tokenizer; {rendered}", file=sys.stderr, flush=True)


def _heartbeat(
    stop_event: threading.Event,
    interval_seconds: float,
    start_time: float,
    *,
    input_file_count: int,
    input_bytes: int,
    vocab_size: int,
    model_type: str,
    model_path: Path,
    metadata_path: Path,
) -> None:
    while not stop_event.wait(interval_seconds):
        _log_tokenizer_event(
            status="running",
            input_file_count=input_file_count,
            input_bytes=input_bytes,
            vocab_size=vocab_size,
            model_type=model_type,
            model_path=model_path,
            metadata_path=metadata_path,
            elapsed_seconds=time.monotonic() - start_time,
            remaining_seconds=None,
        )


def train_tokenizer(
    input_files: list[str],
    config: TokenizerConfig,
    user_defined_symbols: list[str] | None = None,
) -> Path:
    spm = _require_sentencepiece()
    model_prefix = Path(config.model_prefix)
    model_prefix.parent.mkdir(parents=True, exist_ok=True)
    reserved_symbols = {
        config.special_tokens["unk_token"],
        config.special_tokens["bos_token"],
        config.special_tokens["eos_token"],
        config.special_tokens["pad_token"],
    }
    symbols = user_defined_symbols or [
        token for token in config.special_tokens.values() if token not in reserved_symbols
    ]
    args = {
        "input": ",".join(input_files),
        "model_prefix": str(model_prefix),
        "vocab_size": config.vocab_size,
        "model_type": config.model_type,
        "character_coverage": config.character_coverage,
        "byte_fallback": str(config.byte_fallback).lower(),
        "normalization_rule_name": config.normalization_rule_name,
        # SentencePiece expects `input_sentence_size`; keep the config field name
        # as-is and translate it here so older saved configs still work.
        "input_sentence_size": config.sample_input_sentence_size,
        "max_sentence_length": config.max_sentence_length,
        "train_extremely_large_corpus": str(config.train_extremely_large_corpus).lower(),
        "user_defined_symbols": ",".join(symbols),
        "unk_id": 0,
        "bos_id": 1,
        "eos_id": 2,
        "pad_id": 3,
        "unk_piece": config.special_tokens["unk_token"],
        "pad_piece": config.special_tokens["pad_token"],
        "bos_piece": config.special_tokens["bos_token"],
        "eos_piece": config.special_tokens["eos_token"],
    }
    command = " ".join(f"--{key}={value}" for key, value in args.items())
    existing_files, total_bytes = _describe_input_files(input_files)
    stage_start_time = time.monotonic()
    model_path = model_prefix.with_suffix(".model")
    meta_path = model_prefix.with_suffix(".tokenizer.json")
    _log_tokenizer_event(
        status="starting",
        input_file_count=existing_files,
        input_bytes=total_bytes,
        vocab_size=config.vocab_size,
        model_type=config.model_type,
        model_path=model_path,
        metadata_path=meta_path,
        elapsed_seconds=0.0,
        remaining_seconds=None,
    )
    stop_event = threading.Event()
    heartbeat = threading.Thread(
        target=_heartbeat,
        args=(stop_event, 20.0, stage_start_time),
        kwargs={
            "input_file_count": existing_files,
            "input_bytes": total_bytes,
            "vocab_size": config.vocab_size,
            "model_type": config.model_type,
            "model_path": model_path,
            "metadata_path": meta_path,
        },
        daemon=True,
    )
    heartbeat.start()
    try:
        spm.SentencePieceTrainer.Train(command)
    except RuntimeError as exc:
        message = str(exc)
        if "Vocabulary size too high" in message:
            total_chars = 0
            total_lines = 0
            total_words = 0
            for raw_path in input_files:
                path = Path(raw_path)
                if not path.exists():
                    continue
                text = path.read_text(encoding="utf-8", errors="ignore")
                total_chars += len(text)
                total_lines += len(text.splitlines())
                total_words += len(text.split())
            raise RuntimeError(
                "Tokenizer training failed because the requested vocab size is much larger than the "
                "available corpus can support. "
                f"Requested vocab size: {config.vocab_size}. "
                f"Observed corpus size: {total_lines} lines, {total_words} words, {total_chars} characters. "
                "If you want the real tokenizer at 50176, provide a much larger corpus file or corpus shard list."
            ) from exc
        raise
    finally:
        stop_event.set()
        heartbeat.join(timeout=1.0)
    meta_path.write_text(json.dumps(config.to_dict(), indent=2))
    _log_tokenizer_event(
        status="finished",
        input_file_count=existing_files,
        input_bytes=total_bytes,
        vocab_size=config.vocab_size,
        model_type=config.model_type,
        model_path=model_path,
        metadata_path=meta_path,
        elapsed_seconds=time.monotonic() - stage_start_time,
        remaining_seconds=0.0,
    )
    return model_path
