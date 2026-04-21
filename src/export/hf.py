from __future__ import annotations

import json
import shutil
from pathlib import Path

from config import ModelConfig
from generation import default_stop_strings
from provenance import checkpoint_manifest, tokenizer_manifest
from train.checkpoint import load_artifact_trust


def _require_torch():
    import torch

    return torch


def _infer_tokenizer_path(checkpoint_path: str) -> str | None:
    config_path = Path(checkpoint_path).parent / "configs" / "data.json"
    if not config_path.exists():
        return None
    payload = json.loads(config_path.read_text())
    return payload.get("tokenizer_path")


def _hf_config(model_config: ModelConfig) -> dict:
    return {
        "architectures": ["LlamaForCausalLM"],
        "model_type": "llama",
        "vocab_size": model_config.vocab_size,
        "hidden_size": model_config.hidden_size,
        "intermediate_size": model_config.intermediate_size,
        "num_hidden_layers": model_config.num_hidden_layers,
        "num_attention_heads": model_config.num_attention_heads,
        "num_key_value_heads": model_config.num_key_value_heads,
        "max_position_embeddings": model_config.max_position_embeddings,
        "rms_norm_eps": model_config.rms_norm_eps,
        "rope_theta": model_config.rope_theta,
        "bos_token_id": model_config.bos_token_id,
        "eos_token_id": model_config.eos_token_id,
        "pad_token_id": model_config.pad_token_id,
        "tie_word_embeddings": model_config.tie_word_embeddings,
        "torch_dtype": "bfloat16",
        "attention_bias": False,
        "hidden_act": "silu",
    }


def _normalize_key(key: str) -> str:
    prefixes = ("_orig_mod.", "module.")
    normalized = key
    for prefix in prefixes:
        if normalized.startswith(prefix):
            normalized = normalized[len(prefix) :]
    return normalized


def _hf_state_dict(state_dict: dict[str, object]) -> dict[str, object]:
    converted: dict[str, object] = {}
    for raw_key, value in state_dict.items():
        key = _normalize_key(raw_key)
        if key.startswith("embed_tokens."):
            converted[f"model.{key}"] = value
        elif key.startswith("layers."):
            converted[f"model.{key}"] = value
        elif key.startswith("norm."):
            converted[f"model.{key}"] = value
        elif key.startswith("lm_head."):
            converted[key] = value
        else:
            converted[key] = value
    return converted


def _dedupe_shared_tensors(state_dict: dict[str, object]) -> dict[str, object]:
    deduped: dict[str, object] = {}
    seen_ptrs: set[int] = set()
    for key, value in state_dict.items():
        ptr_getter = getattr(value, "untyped_storage", None)
        if callable(ptr_getter):
            ptr = value.untyped_storage().data_ptr()
            if ptr in seen_ptrs:
                deduped[key] = value.clone()
                continue
            seen_ptrs.add(ptr)
        deduped[key] = value
    return deduped


def export_hf_checkpoint(model_config: ModelConfig, checkpoint_path: str, output_dir: str) -> None:
    torch = _require_torch()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    payload = torch.load(Path(checkpoint_path) / "checkpoint.pt", map_location="cpu")
    state_dict = _dedupe_shared_tensors(_hf_state_dict(payload["model"]))

    try:
        from safetensors.torch import save_file  # type: ignore[import-not-found]
    except ModuleNotFoundError:
        torch.save(state_dict, output / "pytorch_model.bin")
    else:
        try:
            save_file(state_dict, str(output / "model.safetensors"))
        except RuntimeError:
            torch.save(state_dict, output / "pytorch_model.bin")

    (output / "config.json").write_text(json.dumps(_hf_config(model_config), indent=2))
    (output / "generation_config.json").write_text(
        json.dumps(
            {
                "max_new_tokens": 256,
                "do_sample": False,
                "temperature": 0.0,
                "top_p": 1.0,
                "repetition_penalty": 1.05,
                "no_repeat_ngram_size": 4,
                "bos_token_id": model_config.bos_token_id,
                "eos_token_id": model_config.eos_token_id,
                "pad_token_id": model_config.pad_token_id,
                "stop_strings": default_stop_strings(),
            },
            indent=2,
        )
    )

    tokenizer_path = _infer_tokenizer_path(checkpoint_path)
    if tokenizer_path:
        tokenizer_source = Path(tokenizer_path)
        if tokenizer_source.exists():
            shutil.copy2(tokenizer_source, output / tokenizer_source.name)
            shutil.copy2(tokenizer_source, output / "tokenizer.model")
        vocab_file = tokenizer_source.with_suffix(".vocab")
        tokenizer_meta = tokenizer_source.with_suffix(".tokenizer.json")
        if vocab_file.exists():
            shutil.copy2(vocab_file, output / vocab_file.name)
            shutil.copy2(vocab_file, output / "tokenizer.vocab")
        if tokenizer_meta.exists():
            shutil.copy2(tokenizer_meta, output / tokenizer_meta.name)
        (output / "tokenizer_config.json").write_text(
            json.dumps(
                {
                    "tokenizer_class": "LlamaTokenizer",
                    "legacy": False,
                    "byte_fallback": True,
                    "bos_token": "<s>",
                    "eos_token": "</s>",
                    "unk_token": "<unk>",
                    "pad_token": "<pad>",
                    "additional_special_tokens": [
                        "<|assistant|>",
                        "<|user|>",
                        "<|system|>",
                        "<|tool|>",
                        "<|citation|>",
                    ],
                },
                indent=2,
            )
        )
        (output / "special_tokens_map.json").write_text(
            json.dumps(
                {
                    "bos_token": "<s>",
                    "eos_token": "</s>",
                    "unk_token": "<unk>",
                    "pad_token": "<pad>",
                    "additional_special_tokens": [
                        "<|assistant|>",
                        "<|user|>",
                        "<|system|>",
                        "<|tool|>",
                        "<|citation|>",
                    ],
                },
                indent=2,
            )
        )
    (output / "provenance.json").write_text(
        json.dumps(
            {
                "checkpoint": checkpoint_manifest(checkpoint_path),
                "tokenizer": tokenizer_manifest(tokenizer_path) if tokenizer_path else None,
            },
            indent=2,
        )
    )
    for metadata_name in ("checkpoint_metadata.json", "stage_summary.json"):
        source_path = Path(checkpoint_path) / metadata_name
        if source_path.exists():
            shutil.copy2(source_path, output / metadata_name)
    (output / "artifact_trust.json").write_text(
        json.dumps(load_artifact_trust(checkpoint_path), indent=2)
    )
