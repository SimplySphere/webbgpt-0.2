from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import platform
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

from config import (
    DataConfig,
    DataSourceConfig,
    EvalConfig,
    GroundingConfig,
    ModelConfig,
    ServeConfig,
    TokenizerConfig,
    TokenizerCorpusConfig,
    TrainConfig,
    save_payload,
)
from config.io import load_config, save_config
from train.console import dump_rounded_json


SAFE_DEFAULT_TEST_PATHS = [
    "src/tests/test_config.py",
    "src/tests/test_packing.py",
    "src/tests/test_prepared.py",
    "src/tests/test_prepare_resume.py",
    "src/tests/test_catalog_eval.py",
    "src/tests/test_cli_profiles.py",
    "src/tests/test_train_loop.py",
    "src/tests/test_webb_grounding.py",
]

PROFILE_RUNTIME_MANUAL_PROFILES = frozenset({"local-mvp", "remote-3b", "remote-7b"})
LEGACY_PROFILE_CHECKPOINT_DIRS = {
    "local-mvp": Path("artifacts/checkpoints-local-mvp"),
    "remote-3b": Path("artifacts/checkpoints"),
    "remote-7b": Path("artifacts/checkpoints-7b"),
}


def _artifact_trust(path: str | Path) -> dict[str, object]:
    from train.checkpoint import load_artifact_trust

    return load_artifact_trust(path)


def _require_trusted_artifact(
    path: str | Path,
    *,
    action: str,
    force_untrusted: bool = False,
) -> dict[str, object]:
    trust = _artifact_trust(path)
    status = str(trust.get("artifact_status", "promotable"))
    blockers = list(trust.get("promotion_blockers", []))
    if status == "promotable" or force_untrusted:
        if status != "promotable":
            print(
                f"WebbGPT: forcing {action} with {status} artifact at {path} "
                f"(blockers: {', '.join(blockers) or 'none'}).",
                file=sys.stderr,
                flush=True,
            )
        return trust
    raise RuntimeError(
        f"Refusing to {action} from non-promotable artifact at {path}. "
        f"artifact_status={status}; blockers={', '.join(blockers) or 'none'}. "
        "Re-run with --force-untrusted only for local debugging."
    )


def _build_tokenizer_corpus_worker(
    config_payload: dict[str, object],
    connection: mp.connection.Connection,
) -> None:
    try:
        from config import TokenizerCorpusConfig
        from data.tokenizer_corpus import build_tokenizer_corpus

        config = TokenizerCorpusConfig.from_dict(config_payload)
        result = build_tokenizer_corpus(config)
        connection.send({"ok": True, "result": result})
    except BaseException as exc:  # pragma: no cover - defensive process boundary
        connection.send(
            {
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
        )
    finally:
        connection.close()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)


def _run_tokenizer_corpus_subprocess(config: TokenizerCorpusConfig) -> dict[str, object]:
    ctx = mp.get_context("spawn")
    parent_conn, child_conn = ctx.Pipe(duplex=False)
    process = ctx.Process(
        target=_build_tokenizer_corpus_worker,
        args=(config.to_dict(), child_conn),
        name="webbgpt-build-tokenizer-corpus",
    )
    process.start()
    child_conn.close()
    try:
        payload = parent_conn.recv()
    except EOFError as exc:
        if process.is_alive():
            process.terminate()
        process.join(timeout=5.0)
        raise RuntimeError(
            "Tokenizer corpus build worker exited unexpectedly before returning a result."
        ) from exc
    except KeyboardInterrupt:
        if process.is_alive():
            process.terminate()
        process.join(timeout=5.0)
        raise
    finally:
        parent_conn.close()

    process.join(timeout=5.0)
    if process.is_alive():
        process.terminate()
        process.join(timeout=5.0)

    if not isinstance(payload, dict) or not payload.get("ok"):
        error = "Tokenizer corpus build failed in worker process."
        if isinstance(payload, dict) and payload.get("error"):
            error = str(payload["error"])
        details = ""
        if isinstance(payload, dict) and payload.get("traceback"):
            details = f"\n{payload['traceback']}"
        raise RuntimeError(f"{error}{details}")
    result = payload.get("result")
    if not isinstance(result, dict):
        raise RuntimeError("Tokenizer corpus build returned an invalid result payload.")
    return result


def _parse_args() -> argparse.Namespace:
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        test_arg_parser = argparse.ArgumentParser(prog="webbgpt test")
        test_arg_parser.add_argument(
            "--all",
            action="store_true",
            help="Run the full test suite, including torch-dependent tests",
        )
        known, unknown = test_arg_parser.parse_known_args(sys.argv[2:])
        return argparse.Namespace(command="test", all=known.all, pytest_args=unknown)

    parser = argparse.ArgumentParser(prog="webbgpt", description="WebbGPT control plane")
    subparsers = parser.add_subparsers(dest="command", required=True)

    tokenize = subparsers.add_parser("tokenize", help="Train the SentencePiece tokenizer")
    tokenize.add_argument("--config", required=True)
    tokenize.add_argument("--input", nargs="+", required=True)

    build_tok_corpus = subparsers.add_parser(
        "build-tokenizer-corpus",
        help="Build the real tokenizer corpus from the chosen upstream dataset",
    )
    build_tok_corpus.add_argument("--config", required=True)

    prepare = subparsers.add_parser("prepare-data", help="Prepare manifests and data artifacts")
    prepare.add_argument("--config", required=True)
    prepare.add_argument(
        "--stage",
        choices=["pretrain", "continue", "sft", "preference", "validation"],
        required=True,
    )
    prepare.add_argument("--output", required=True)
    prepare.add_argument("--force-rebuild", action="store_true")

    audit = subparsers.add_parser("audit-data", help="Audit LM corpus sources for a stage")
    audit.add_argument("--config", required=True)
    audit.add_argument("--stage", choices=["pretrain", "continue", "validation"], required=True)

    train_pre = subparsers.add_parser("train-pretrain", help="Run base pretraining")
    train_pre.add_argument("--model-config", required=True)
    train_pre.add_argument("--data-config", required=True)
    train_pre.add_argument("--train-config", required=True)
    train_pre.add_argument("--force-rebuild", action="store_true")

    train_continue = subparsers.add_parser("train-continue", help="Run continued pretraining")
    train_continue.add_argument("--model-config", required=True)
    train_continue.add_argument("--data-config", required=True)
    train_continue.add_argument("--train-config", required=True)
    train_continue.add_argument("--force-rebuild", action="store_true")

    train_sft = subparsers.add_parser("train-sft", help="Run supervised fine-tuning")
    train_sft.add_argument("--model-config", required=True)
    train_sft.add_argument("--data-config", required=True)
    train_sft.add_argument("--train-config", required=True)
    train_sft.add_argument("--force-rebuild", action="store_true")

    train_dpo = subparsers.add_parser("train-dpo", help="Run DPO alignment")
    train_dpo.add_argument("--model-config", required=True)
    train_dpo.add_argument("--data-config", required=True)
    train_dpo.add_argument("--train-config", required=True)
    train_dpo.add_argument("--reference-checkpoint", required=True)
    train_dpo.add_argument("--force-rebuild", action="store_true")

    eval_parser = subparsers.add_parser("eval", help="Run evaluation")
    eval_parser.add_argument("--model-config", required=True)
    eval_parser.add_argument("--data-config", required=True)
    eval_parser.add_argument("--eval-config", required=True)
    eval_parser.add_argument("--checkpoint", required=True)
    eval_parser.add_argument("--force-untrusted", action="store_true")

    ingest_webb_site = subparsers.add_parser("ingest-webb-site", help="Ingest Webb HTML sources")
    ingest_webb_site.add_argument("--dsn", required=True)
    ingest_webb_site.add_argument("--seed-url-pack", required=True)
    ingest_webb_site.add_argument("--offline-seed-url-pack")
    ingest_webb_site.add_argument("--source-policy-path")
    ingest_webb_site.add_argument("--snapshot-id")
    ingest_webb_site.add_argument("--label", default="webb-site-sync")
    ingest_webb_site.add_argument("--families", nargs="*")

    ingest_webb_handbook = subparsers.add_parser("ingest-webb-handbook", help="Ingest the Webb handbook")
    ingest_webb_handbook.add_argument("--dsn", required=True)
    ingest_webb_handbook.add_argument("--handbook-url", required=True)
    ingest_webb_handbook.add_argument("--snapshot-id")
    ingest_webb_handbook.add_argument("--label", default="webb-handbook-sync")
    ingest_webb_handbook.add_argument("--allow-ocr-fallback", action="store_true")

    webb_sync_parser = subparsers.add_parser("webb-sync", help="Sync the Webb academics source pack")
    webb_sync_parser.add_argument("--dsn", required=True)
    webb_sync_parser.add_argument("--seed-url-pack", required=True)
    webb_sync_parser.add_argument("--offline-seed-url-pack")
    webb_sync_parser.add_argument("--source-policy-path")
    webb_sync_parser.add_argument("--handbook-url")
    webb_sync_parser.add_argument("--allow-ocr-fallback", action="store_true")
    webb_sync_parser.add_argument("--label", default="webb-sync")
    webb_sync_parser.add_argument("--families", nargs="*")

    diff_webb = subparsers.add_parser("diff-webb-snapshot", help="Diff two Webb grounding snapshots")
    diff_webb.add_argument("--dsn", required=True)
    diff_webb.add_argument("--from-snapshot", required=True)
    diff_webb.add_argument("--to-snapshot", required=True)

    serve = subparsers.add_parser("serve", help="Launch serving API")
    serve.add_argument("--serve-config", required=True)
    serve.add_argument("--sync-on-start", action="store_true")
    serve.add_argument("--force-untrusted", action="store_true")

    export = subparsers.add_parser("export-hf", help="Export a checkpoint to HF-style format")
    export.add_argument("--model-config", required=True)
    export.add_argument("--checkpoint", required=True)
    export.add_argument("--output", required=True)
    export.add_argument("--force-untrusted", action="store_true")

    init_cfg = subparsers.add_parser("init-config", help="Write example configs")
    init_cfg.add_argument("--output-dir", default="configs")

    main_parser = subparsers.add_parser(
        "main",
        help="Run a full WebbGPT pipeline profile from tokenizer build through optional serving",
    )
    main_parser.add_argument("--output-dir", default="sample-configs")
    main_parser.add_argument(
        "--profile",
        choices=["debug", "local-mvp", "remote-3b", "remote-7b"],
        required=True,
    )
    main_parser.add_argument("--no-serve", action="store_true")
    main_parser.add_argument("--force-rebuild", action="store_true")
    preset_group = main_parser.add_mutually_exclusive_group()
    preset_group.add_argument("--mvp", action="store_true")
    preset_group.add_argument("--full", action="store_true")

    args, unknown = parser.parse_known_args()
    if unknown:
        parser.error(f"unrecognized arguments: {' '.join(unknown)}")
    return args


def _source_payload(
    *,
    name: str,
    format: str,
    path: str = "",
    paths: list[str] | None = None,
    split: str = "train",
    dataset_name: str | None = None,
    dataset_config_name: str | None = None,
    dataset_revision: str | None = None,
    streaming: bool | None = None,
    weight: float = 1.0,
    text_field: str = "text",
    messages_field: str = "messages",
    prompt_field: str = "prompt",
    response_field: str = "response",
    chosen_field: str = "chosen",
    rejected_field: str = "rejected",
    metadata_fields: list[str] | None = None,
    language: str | None = "en",
    quality_filter: bool = True,
    deduplicate: bool = True,
    pii_scrub: bool = True,
    skip_records: int = 0,
    max_records: int | None = None,
) -> dict[str, object]:
    return {
        "name": name,
        "path": path,
        "paths": paths or [],
        "split": split,
        "format": format,
        "dataset_name": dataset_name,
        "dataset_config_name": dataset_config_name,
        "dataset_revision": dataset_revision,
        "streaming": streaming,
        "weight": weight,
        "text_field": text_field,
        "messages_field": messages_field,
        "prompt_field": prompt_field,
        "response_field": response_field,
        "chosen_field": chosen_field,
        "rejected_field": rejected_field,
        "metadata_fields": metadata_fields or [],
        "language": language,
        "quality_filter": quality_filter,
        "deduplicate": deduplicate,
        "pii_scrub": pii_scrub,
        "skip_records": skip_records,
        "max_records": max_records,
    }


def _local_text_source(
    name: str,
    path: str,
    *,
    skip_records: int = 0,
    max_records: int | None = None,
) -> dict[str, object]:
    return _source_payload(
        name=name,
        path=path,
        format="text",
        streaming=False,
        quality_filter=False,
        deduplicate=False,
        pii_scrub=False,
        skip_records=skip_records,
        max_records=max_records,
    )


def _local_jsonl_source(name: str, path: str) -> dict[str, object]:
    return _source_payload(
        name=name,
        path=path,
        format="jsonl",
        streaming=False,
        quality_filter=False,
        deduplicate=False,
        pii_scrub=False,
    )


def _hf_text_source(
    name: str,
    *,
    dataset_name: str,
    dataset_config_name: str,
    skip_records: int = 0,
    max_records: int | None = None,
) -> dict[str, object]:
    return _source_payload(
        name=name,
        format="hf",
        dataset_name=dataset_name,
        dataset_config_name=dataset_config_name,
        streaming=True,
        quality_filter=True,
        deduplicate=True,
        pii_scrub=True,
        skip_records=skip_records,
        max_records=max_records,
    )


def _webb_benchmark_paths(*, include_assistant: bool = True) -> list[str]:
    paths = [
        "data/eval/webb_course_present.responses",
        "data/eval/webb_course_missing.responses",
        "data/eval/webb_course_diff.responses",
        "data/eval/webb_handbook_present.responses",
        "data/eval/webb_handbook_missing.responses",
        "data/eval/webb_faculty.responses",
        "data/eval/webb_admissions.responses",
        "data/eval/webb_student_life.responses",
        "data/eval/webb_mission_values.responses",
        "data/eval/webb_college_guidance.responses",
        "data/eval/webb_museum_programs.responses",
        "data/eval/webb_athletics_present.responses",
        "data/eval/webb_athletics_missing.responses",
        "data/eval/webb_planner.responses",
        "data/eval/webb_mixed_multi_domain.responses",
    ]
    if include_assistant:
        return [
            "data/eval/chat_sanity.jsonl",
            "data/eval/assistant.jsonl",
            *paths,
        ]
    return paths


def _profile_grounding_dsn(profile: str) -> str:
    names = {
        "debug": "webbgpt-debug",
        "local-mvp": "webbgpt-local-mvp",
        "remote-3b": "webbgpt-3b",
        "remote-7b": "webbgpt-7b",
    }
    suffix = names.get(profile, "webbgpt")
    return f"sqlite:///artifacts/grounding/{suffix}.db"


def _profile_grounding_payload(
    profile: str,
    *,
    sync_on_start: bool,
    planner_beta_enabled: bool,
) -> dict[str, object]:
    grounding = GroundingConfig(
        dsn=_profile_grounding_dsn(profile),
        sync_on_start=sync_on_start,
        planner_beta_enabled=planner_beta_enabled,
    )
    return grounding.to_dict()


def _without_legacy_catalog_fields(payload: dict[str, object]) -> dict[str, object]:
    cleaned = dict(payload)
    cleaned.pop("catalog_dsn", None)
    cleaned.pop("catalog_input_path", None)
    return cleaned


def _load_json_payload(path: Path) -> dict[str, object] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _can_reuse_tokenizer_corpus(config: TokenizerCorpusConfig) -> bool:
    output_path = Path(config.output_path)
    meta_path = output_path.with_suffix(output_path.suffix + ".meta.json")
    payload = _load_json_payload(meta_path)
    if payload is None or not output_path.exists():
        return False
    return payload.get("config") == config.to_dict()


def _can_reuse_tokenizer_model(config: TokenizerConfig) -> bool:
    model_path = Path(f"{config.model_prefix}.model")
    vocab_path = Path(f"{config.model_prefix}.vocab")
    meta_path = Path(f"{config.model_prefix}.tokenizer.json")
    payload = _load_json_payload(meta_path)
    if payload is None or not model_path.exists() or not vocab_path.exists():
        return False
    return payload == config.to_dict()


def _write_default_configs(output_dir: str) -> None:
    base = Path(output_dir)
    base.mkdir(parents=True, exist_ok=True)
    save_config(TokenizerConfig(), base / "tokenizer.json")
    save_config(TokenizerCorpusConfig(), base / "tokenizer-corpus.json")
    save_config(
        TokenizerConfig(
            model_prefix="artifacts/tokenizer/webbgpt-local-mvp",
            vocab_size=32_000,
            sample_input_sentence_size=2_000_000,
            train_extremely_large_corpus=False,
        ),
        base / "tokenizer-local-mvp.json",
    )
    save_config(
        TokenizerCorpusConfig(
            output_path="data/raw/tokenizer_corpus_local_mvp.txt",
            max_documents=500_000,
            max_characters=200_000_000,
            min_document_chars=96,
        ),
        base / "tokenizer-corpus-local-mvp.json",
    )
    save_config(
        TokenizerConfig(
            model_prefix="artifacts/tokenizer/webbgpt-7b",
            vocab_size=50_176,
        ),
        base / "tokenizer-7b.json",
    )
    save_config(
        TokenizerCorpusConfig(
            output_path="data/raw/tokenizer_corpus_7b.txt",
        ),
        base / "tokenizer-corpus-7b.json",
    )
    save_config(ModelConfig(), base / "model-3b.json")
    save_payload(
        {
            **DataConfig().to_dict(),
            "pretrain_sources": [
                {
                    "name": "fineweb_edu_pretrain",
                    "format": "hf",
                    "dataset_name": "HuggingFaceFW/fineweb-edu",
                    "dataset_config_name": "sample-10BT",
                    "dataset_revision": None,
                    "path": "",
                    "paths": [],
                    "split": "train",
                    "streaming": True,
                    "text_field": "text",
                    "messages_field": "messages",
                    "prompt_field": "prompt",
                    "chosen_field": "chosen",
                    "rejected_field": "rejected",
                    "weight": 1.0,
                    "metadata_fields": [],
                    "language": "en",
                    "quality_filter": True,
                    "deduplicate": True,
                    "pii_scrub": True,
                    "skip_records": 2048,
                    "max_records": None,
                }
            ],
            "continued_pretrain_sources": [
                {
                    "name": "education_domain_corpus",
                    "path": "data/domain/education_corpus.txt",
                    "paths": [],
                    "split": "train",
                    "format": "text",
                    "dataset_name": None,
                    "dataset_config_name": None,
                    "dataset_revision": None,
                    "streaming": False,
                    "weight": 1.0,
                    "text_field": "text",
                    "messages_field": "messages",
                    "prompt_field": "prompt",
                    "chosen_field": "chosen",
                    "rejected_field": "rejected",
                    "metadata_fields": [],
                    "language": "en",
                    "quality_filter": False,
                    "deduplicate": False,
                    "pii_scrub": False,
                    "skip_records": 0,
                    "max_records": None,
                },
                {
                    "name": "advising_domain_corpus",
                    "path": "data/domain/advising_corpus.txt",
                    "paths": [],
                    "split": "train",
                    "format": "text",
                    "dataset_name": None,
                    "dataset_config_name": None,
                    "dataset_revision": None,
                    "streaming": False,
                    "weight": 1.0,
                    "text_field": "text",
                    "messages_field": "messages",
                    "prompt_field": "prompt",
                    "response_field": "response",
                    "chosen_field": "chosen",
                    "rejected_field": "rejected",
                    "metadata_fields": [],
                    "language": "en",
                    "quality_filter": False,
                    "deduplicate": False,
                    "pii_scrub": False,
                    "skip_records": 0,
                    "max_records": None,
                },
                {
                    "name": "philosophy_domain_corpus",
                    "path": "data/domain/philosophy_corpus.txt",
                    "paths": [],
                    "split": "train",
                    "format": "text",
                    "dataset_name": None,
                    "dataset_config_name": None,
                    "dataset_revision": None,
                    "streaming": False,
                    "weight": 1.0,
                    "text_field": "text",
                    "messages_field": "messages",
                    "prompt_field": "prompt",
                    "response_field": "response",
                    "chosen_field": "chosen",
                    "rejected_field": "rejected",
                    "metadata_fields": [],
                    "language": "en",
                    "quality_filter": False,
                    "deduplicate": False,
                    "pii_scrub": False,
                    "skip_records": 0,
                    "max_records": None,
                },
                {
                    "name": "catalog_domain_corpus",
                    "path": "data/domain/catalog_corpus.txt",
                    "paths": [],
                    "split": "train",
                    "format": "text",
                    "dataset_name": None,
                    "dataset_config_name": None,
                    "dataset_revision": None,
                    "streaming": False,
                    "weight": 1.0,
                    "text_field": "text",
                    "messages_field": "messages",
                    "prompt_field": "prompt",
                    "response_field": "response",
                    "chosen_field": "chosen",
                    "rejected_field": "rejected",
                    "metadata_fields": [],
                    "language": "en",
                    "quality_filter": False,
                    "deduplicate": False,
                    "pii_scrub": False,
                    "skip_records": 0,
                    "max_records": None,
                }
            ],
            "sft_sources": [
                {
                    "name": "public_sft_seed",
                    "path": "data/posttrain/sft_public_seed.jsonl",
                    "paths": [],
                    "split": "train",
                    "format": "jsonl",
                    "dataset_name": None,
                    "dataset_config_name": None,
                    "dataset_revision": None,
                    "streaming": False,
                    "weight": 1.0,
                    "text_field": "text",
                    "messages_field": "messages",
                    "prompt_field": "prompt",
                    "response_field": "response",
                    "chosen_field": "chosen",
                    "rejected_field": "rejected",
                    "metadata_fields": [],
                    "language": "en",
                    "quality_filter": False,
                    "deduplicate": False,
                    "pii_scrub": False,
                    "skip_records": 0,
                    "max_records": None,
                },
                {
                    "name": "domain_sft_seed",
                    "path": "data/posttrain/sft_domain_synthetic.jsonl",
                    "paths": [],
                    "split": "train",
                    "format": "jsonl",
                    "dataset_name": None,
                    "dataset_config_name": None,
                    "dataset_revision": None,
                    "streaming": False,
                    "weight": 1.0,
                    "text_field": "text",
                    "messages_field": "messages",
                    "prompt_field": "prompt",
                    "response_field": "response",
                    "chosen_field": "chosen",
                    "rejected_field": "rejected",
                    "metadata_fields": [],
                    "language": "en",
                    "quality_filter": False,
                    "deduplicate": False,
                    "pii_scrub": False,
                    "skip_records": 0,
                    "max_records": None,
                },
                {
                    "name": "conversation_sft_seed",
                    "path": "data/posttrain/sft_conversation_seed.jsonl",
                    "paths": [],
                    "split": "train",
                    "format": "jsonl",
                    "dataset_name": None,
                    "dataset_config_name": None,
                    "dataset_revision": None,
                    "streaming": False,
                    "weight": 1.0,
                    "text_field": "text",
                    "messages_field": "messages",
                    "prompt_field": "prompt",
                    "response_field": "response",
                    "chosen_field": "chosen",
                    "rejected_field": "rejected",
                    "metadata_fields": [],
                    "language": "en",
                    "quality_filter": False,
                    "deduplicate": False,
                    "pii_scrub": False,
                    "skip_records": 0,
                    "max_records": None,
                }
            ],
            "sft_validation_sources": [
                _local_jsonl_source("posttrain_sft_validation", "data/posttrain/sft_validation.jsonl")
            ],
            "preference_sources": [
                {
                    "name": "public_preference_seed",
                    "path": "data/posttrain/preference_public_seed.jsonl",
                    "paths": [],
                    "split": "train",
                    "format": "jsonl",
                    "dataset_name": None,
                    "dataset_config_name": None,
                    "dataset_revision": None,
                    "streaming": False,
                    "weight": 1.0,
                    "text_field": "text",
                    "messages_field": "messages",
                    "prompt_field": "prompt",
                    "response_field": "response",
                    "chosen_field": "chosen",
                    "rejected_field": "rejected",
                    "metadata_fields": [],
                    "language": "en",
                    "quality_filter": False,
                    "deduplicate": False,
                    "pii_scrub": False,
                    "skip_records": 0,
                    "max_records": None,
                },
                {
                    "name": "domain_preference_seed",
                    "path": "data/posttrain/preference_domain_synthetic.jsonl",
                    "paths": [],
                    "split": "train",
                    "format": "jsonl",
                    "dataset_name": None,
                    "dataset_config_name": None,
                    "dataset_revision": None,
                    "streaming": False,
                    "weight": 1.0,
                    "text_field": "text",
                    "messages_field": "messages",
                    "prompt_field": "prompt",
                    "response_field": "response",
                    "chosen_field": "chosen",
                    "rejected_field": "rejected",
                    "metadata_fields": [],
                    "language": "en",
                    "quality_filter": False,
                    "deduplicate": False,
                    "pii_scrub": False,
                    "skip_records": 0,
                    "max_records": None,
                },
                {
                    "name": "conversation_preference_seed",
                    "path": "data/posttrain/preference_conversation_seed.jsonl",
                    "paths": [],
                    "split": "train",
                    "format": "jsonl",
                    "dataset_name": None,
                    "dataset_config_name": None,
                    "dataset_revision": None,
                    "streaming": False,
                    "weight": 1.0,
                    "text_field": "text",
                    "messages_field": "messages",
                    "prompt_field": "prompt",
                    "response_field": "response",
                    "chosen_field": "chosen",
                    "rejected_field": "rejected",
                    "metadata_fields": [],
                    "language": "en",
                    "quality_filter": False,
                    "deduplicate": False,
                    "pii_scrub": False,
                    "skip_records": 0,
                    "max_records": None,
                }
            ],
            "preference_validation_sources": [
                _local_jsonl_source(
                    "posttrain_preference_validation",
                    "data/posttrain/preference_validation.jsonl",
                )
            ],
            "validation_sources": [
                {
                    "name": "fineweb_edu_validation",
                    "format": "hf",
                    "dataset_name": "HuggingFaceFW/fineweb-edu",
                    "dataset_config_name": "sample-10BT",
                    "dataset_revision": None,
                    "path": "",
                    "paths": [],
                    "split": "train",
                    "streaming": True,
                    "text_field": "text",
                    "messages_field": "messages",
                    "prompt_field": "prompt",
                    "chosen_field": "chosen",
                    "rejected_field": "rejected",
                    "weight": 1.0,
                    "metadata_fields": [],
                    "language": "en",
                    "quality_filter": True,
                    "deduplicate": False,
                    "pii_scrub": True,
                    "skip_records": 0,
                    "max_records": 2048,
                }
            ],
        },
        base / "data-3b.json",
    )
    save_payload(
        {
            **TrainConfig(
                run_name="webbgpt-3b",
                max_steps=400_000,
                continued_max_steps=50_000,
                sft_max_steps=20_000,
                sft_max_epochs=5,
                dpo_max_steps=10_000,
                sft_validation_min_examples=16,
                dpo_validation_min_examples=16,
                sft_evals_per_epoch=4,
                sft_min_eval_interval_steps=25,
                dpo_evals_per_epoch=4,
                sft_early_stopping_patience_evals=2,
                dpo_early_stopping_patience_evals=2,
                sft_best_min_delta=0.02,
                sft_sample_every_steps=100,
                dpo_best_min_delta=0.005,
                dpo_enable_lm_health_eval=True,
                allow_weak_posttrain_validation=False,
                token_budget=None,
            ).to_dict(),
            "checkpoint": {
                "output_dir": "artifacts/runs/remote-3b/checkpoints/pretrain",
                "save_every_steps": 500,
                "keep_last_n": 5,
                "async_write": False,
                "initialize_from": None,
                "resume_from": None,
                "export_every_eval": True,
            },
        },
        base / "train-3b.json",
    )
    save_payload(
        {
            **ModelConfig(
                name="webbgpt-local-mvp",
                vocab_size=32_000,
                hidden_size=512,
                intermediate_size=1_536,
                num_hidden_layers=8,
                num_attention_heads=8,
                num_key_value_heads=4,
                max_position_embeddings=512,
                use_flash_attention=False,
                gradient_checkpointing=False,
            ).to_dict(),
        },
        base / "model-local-mvp.json",
    )
    save_payload(
        {
            **DataConfig().to_dict(),
            "tokenizer_path": "artifacts/tokenizer/webbgpt-local-mvp.model",
            "sequence_length": 512,
            "prepared_shard_size": 1024,
            "pretraining_token_budget": 200_000_000,
            "continued_pretraining_token_budget": 5_000_000,
            "pretrain_sources": [
                _local_text_source(
                    "local_mvp_pretrain_corpus",
                    "data/raw/tokenizer_corpus_local_mvp.txt",
                    skip_records=4096,
                )
            ],
            "continued_pretrain_sources": [
                _local_text_source(
                    "local_mvp_continue_domain_mix",
                    "data/domain/local_mvp_continue_corpus.txt",
                ),
                _local_text_source(
                    "local_mvp_general_refresh",
                    "data/raw/tokenizer_corpus_local_mvp.txt",
                    skip_records=8192,
                    max_records=4096,
                ),
            ],
            "sft_sources": [
                _local_jsonl_source("local_sft_examples", "data/local/sft.jsonl"),
                _local_jsonl_source("public_sft_seed", "data/posttrain/sft_public_seed.jsonl"),
                _local_jsonl_source("domain_sft_seed", "data/posttrain/sft_domain_synthetic.jsonl"),
                _local_jsonl_source("conversation_sft_seed", "data/posttrain/sft_conversation_seed.jsonl"),
            ],
            "sft_validation_sources": [
                _local_jsonl_source("local_sft_validation", "data/local/sft_validation.jsonl"),
                _local_jsonl_source("posttrain_sft_validation", "data/posttrain/sft_validation.jsonl"),
            ],
            "preference_sources": [
                _local_jsonl_source("local_preference_examples", "data/local/preference.jsonl"),
                _local_jsonl_source(
                    "public_preference_seed", "data/posttrain/preference_public_seed.jsonl"
                ),
                _local_jsonl_source(
                    "domain_preference_seed", "data/posttrain/preference_domain_synthetic.jsonl"
                ),
                _local_jsonl_source(
                    "conversation_preference_seed",
                    "data/posttrain/preference_conversation_seed.jsonl",
                ),
            ],
            "preference_validation_sources": [
                _local_jsonl_source(
                    "local_preference_validation", "data/local/preference_validation.jsonl"
                ),
                _local_jsonl_source(
                    "posttrain_preference_validation",
                    "data/posttrain/preference_validation.jsonl",
                ),
            ],
            "validation_sources": [
                _local_text_source(
                    "local_mvp_validation_corpus",
                    "data/raw/tokenizer_corpus_local_mvp.txt",
                    max_records=4096,
                )
            ],
        },
        base / "data-local-mvp.json",
    )
    save_payload(
        {
            **TrainConfig(
                run_name="webbgpt-local-mvp",
                global_batch_size=16,
                micro_batch_size=1,
                learning_rate=5e-4,
                min_learning_rate=5e-5,
                warmup_steps=200,
                max_steps=20_000,
                continued_learning_rate=1e-4,
                continued_min_learning_rate=1e-5,
                continued_warmup_steps=25,
                continued_max_steps=250,
                sft_learning_rate=3e-5,
                sft_min_learning_rate=3e-6,
                sft_warmup_steps=10,
                sft_validation_min_examples=16,
                require_explicit_sft_validation=True,
                sft_max_epochs=5,
                sft_evals_per_epoch=4,
                sft_min_eval_interval_steps=25,
                sft_early_stopping_patience_evals=2,
                sft_best_min_delta=0.02,
                sft_max_steps=200,
                sft_sample_every_steps=100,
                dpo_learning_rate=2.5e-5,
                dpo_min_learning_rate=2.5e-6,
                dpo_warmup_steps=10,
                dpo_validation_min_examples=16,
                require_explicit_dpo_validation=True,
                dpo_evals_per_epoch=4,
                dpo_early_stopping_patience_evals=2,
                dpo_best_min_delta=0.005,
                dpo_enable_lm_health_eval=True,
                dpo_max_steps=200,
                allow_weak_posttrain_validation=False,
                posttrain_top_k_checkpoints=3,
                log_every_steps=10,
                eval_every_steps=200,
                num_eval_batches=8,
                compile_model=False,
                use_bf16=False,
                gradient_accumulation_steps=4,
                activation_checkpointing=False,
            ).to_dict(),
            "checkpoint": {
                "output_dir": "artifacts/runs/local-mvp/checkpoints/pretrain",
                "save_every_steps": 25,
                "keep_last_n": 3,
                "async_write": False,
                "initialize_from": None,
                "resume_from": None,
                "export_every_eval": False,
            },
        },
        base / "train-local-mvp.json",
    )
    save_payload(
        {
            **ModelConfig(
                name="webbgpt-7b",
                hidden_size=4096,
                intermediate_size=11008,
                num_hidden_layers=32,
                num_attention_heads=32,
                num_key_value_heads=8,
                max_position_embeddings=4096,
                use_flash_attention=True,
                gradient_checkpointing=True,
            ).to_dict(),
        },
        base / "model-7b.json",
    )
    save_payload(
        {
            **DataConfig().to_dict(),
            "tokenizer_path": "artifacts/tokenizer/webbgpt-7b.model",
            "sequence_length": 4096,
            "prepared_shard_size": 1024,
            "pretraining_token_budget": 200_000_000_000,
            "continued_pretraining_token_budget": 20_000_000_000,
            "pretrain_sources": [
                _hf_text_source(
                    "fineweb_edu_pretrain",
                    dataset_name="HuggingFaceFW/fineweb-edu",
                    dataset_config_name="sample-10BT",
                    skip_records=4096,
                )
            ],
            "continued_pretrain_sources": [
                _local_text_source("education_domain_corpus", "data/domain/education_corpus.txt"),
                _local_text_source("advising_domain_corpus", "data/domain/advising_corpus.txt"),
                _local_text_source("philosophy_domain_corpus", "data/domain/philosophy_corpus.txt"),
                _local_text_source("catalog_domain_corpus", "data/domain/catalog_corpus.txt"),
            ],
            "sft_sources": [
                _local_jsonl_source("public_sft_seed", "data/posttrain/sft_public_seed.jsonl"),
                _local_jsonl_source("domain_sft_seed", "data/posttrain/sft_domain_synthetic.jsonl"),
                _local_jsonl_source("conversation_sft_seed", "data/posttrain/sft_conversation_seed.jsonl"),
            ],
            "sft_validation_sources": [
                _local_jsonl_source("posttrain_sft_validation", "data/posttrain/sft_validation.jsonl")
            ],
            "preference_sources": [
                _local_jsonl_source(
                    "public_preference_seed", "data/posttrain/preference_public_seed.jsonl"
                ),
                _local_jsonl_source(
                    "domain_preference_seed", "data/posttrain/preference_domain_synthetic.jsonl"
                ),
                _local_jsonl_source(
                    "conversation_preference_seed",
                    "data/posttrain/preference_conversation_seed.jsonl",
                ),
            ],
            "preference_validation_sources": [
                _local_jsonl_source(
                    "posttrain_preference_validation",
                    "data/posttrain/preference_validation.jsonl",
                )
            ],
            "validation_sources": [
                _hf_text_source(
                    "fineweb_edu_validation",
                    dataset_name="HuggingFaceFW/fineweb-edu",
                    dataset_config_name="sample-10BT",
                    max_records=4096,
                )
            ],
        },
        base / "data-7b.json",
    )
    save_payload(
        {
            **TrainConfig(
                run_name="webbgpt-7b",
                global_batch_size=1024,
                micro_batch_size=1,
                learning_rate=2.5e-4,
                min_learning_rate=2.5e-5,
                warmup_steps=4_000,
                max_steps=500_000,
                continued_max_steps=60_000,
                sft_max_steps=30_000,
                sft_max_epochs=5,
                dpo_max_steps=15_000,
                sft_validation_min_examples=16,
                dpo_validation_min_examples=16,
                sft_evals_per_epoch=4,
                sft_min_eval_interval_steps=25,
                dpo_evals_per_epoch=4,
                sft_early_stopping_patience_evals=2,
                dpo_early_stopping_patience_evals=2,
                sft_best_min_delta=0.02,
                sft_sample_every_steps=100,
                dpo_best_min_delta=0.005,
                dpo_enable_lm_health_eval=True,
                allow_weak_posttrain_validation=False,
                log_every_steps=10,
                eval_every_steps=500,
                num_eval_batches=50,
                compile_model=True,
                use_bf16=True,
                gradient_accumulation_steps=8,
                activation_checkpointing=True,
            ).to_dict(),
            "checkpoint": {
                "output_dir": "artifacts/runs/remote-7b/checkpoints/pretrain",
                "save_every_steps": 500,
                "keep_last_n": 5,
                "async_write": False,
                "initialize_from": None,
                "resume_from": None,
                "export_every_eval": True,
            },
        },
        base / "train-7b.json",
    )
    save_payload(
        {
            **ModelConfig(
                name="webbgpt-debug",
                hidden_size=256,
                intermediate_size=768,
                num_hidden_layers=4,
                num_attention_heads=4,
                num_key_value_heads=2,
                max_position_embeddings=1024,
                use_flash_attention=False,
                gradient_checkpointing=False,
            ).to_dict(),
            "vocab_size": 50_176,
        },
        base / "model-debug.json",
    )
    save_payload(
        {
            **DataConfig().to_dict(),
            "sequence_length": 512,
            "pretrain_sources": [
                {
                    "name": "local_tokenizer_corpus",
                    "path": "data/raw/tokenizer_corpus.txt",
                    "split": "train",
                    "format": "text",
                    "weight": 1.0,
                    "text_field": "text",
                    "metadata_fields": [],
                    "language": "en",
                    "quality_filter": False,
                    "deduplicate": False,
                    "pii_scrub": False,
                }
            ],
            "continued_pretrain_sources": [
                {
                    "name": "local_tokenizer_corpus",
                    "path": "data/raw/tokenizer_corpus.txt",
                    "split": "train",
                    "format": "text",
                    "weight": 1.0,
                    "text_field": "text",
                    "metadata_fields": [],
                    "language": "en",
                    "quality_filter": False,
                    "deduplicate": False,
                    "pii_scrub": False,
                }
            ],
            "sft_sources": [
                {
                    "name": "local_sft_examples",
                    "path": "data/local/sft.jsonl",
                    "split": "train",
                    "format": "jsonl",
                    "weight": 1.0,
                    "text_field": "text",
                    "metadata_fields": [],
                    "language": "en",
                    "quality_filter": False,
                    "deduplicate": False,
                    "pii_scrub": False,
                }
            ],
            "sft_validation_sources": [
                {
                    "name": "local_sft_validation",
                    "path": "data/local/sft_validation.jsonl",
                    "split": "train",
                    "format": "jsonl",
                    "weight": 1.0,
                    "text_field": "text",
                    "metadata_fields": [],
                    "language": "en",
                    "quality_filter": False,
                    "deduplicate": False,
                    "pii_scrub": False,
                }
            ],
            "preference_sources": [
                {
                    "name": "local_preference_examples",
                    "path": "data/local/preference.jsonl",
                    "split": "train",
                    "format": "jsonl",
                    "weight": 1.0,
                    "text_field": "text",
                    "metadata_fields": [],
                    "language": "en",
                    "quality_filter": False,
                    "deduplicate": False,
                    "pii_scrub": False,
                }
            ],
            "preference_validation_sources": [
                {
                    "name": "local_preference_validation",
                    "path": "data/local/preference_validation.jsonl",
                    "split": "train",
                    "format": "jsonl",
                    "weight": 1.0,
                    "text_field": "text",
                    "metadata_fields": [],
                    "language": "en",
                    "quality_filter": False,
                    "deduplicate": False,
                    "pii_scrub": False,
                }
            ],
            "validation_sources": [
                {
                    "name": "local_tokenizer_corpus_validation",
                    "path": "data/raw/tokenizer_corpus.txt",
                    "split": "train",
                    "format": "text",
                    "weight": 1.0,
                    "text_field": "text",
                    "metadata_fields": [],
                    "language": "en",
                    "quality_filter": False,
                    "deduplicate": False,
                    "pii_scrub": False,
                }
            ],
        },
        base / "data-debug.json",
    )
    save_payload(
        {
            **TrainConfig(
                run_name="webbgpt-debug",
                global_batch_size=8,
                micro_batch_size=1,
                learning_rate=1e-3,
                min_learning_rate=1e-4,
                warmup_steps=10,
                max_steps=20,
                continued_max_steps=20,
                sft_max_steps=20,
                sft_max_epochs=5,
                dpo_max_steps=20,
                sft_validation_min_examples=16,
                dpo_validation_min_examples=16,
                sft_evals_per_epoch=4,
                sft_min_eval_interval_steps=25,
                dpo_evals_per_epoch=4,
                sft_early_stopping_patience_evals=2,
                dpo_early_stopping_patience_evals=2,
                sft_best_min_delta=0.02,
                sft_sample_every_steps=100,
                dpo_best_min_delta=0.005,
                dpo_enable_lm_health_eval=True,
                allow_weak_posttrain_validation=False,
                log_every_steps=1,
                eval_every_steps=10,
                num_eval_batches=2,
                compile_model=False,
                use_bf16=False,
                gradient_accumulation_steps=1,
                activation_checkpointing=False,
            ).to_dict(),
            "checkpoint": {
                "output_dir": "artifacts/checkpoints-debug",
                "save_every_steps": 10,
                "keep_last_n": 2,
                "async_write": False,
                "initialize_from": None,
                "resume_from": None,
                "export_every_eval": False,
            },
        },
        base / "train-debug.json",
    )
    save_payload(
        _without_legacy_catalog_fields(
            {
                **EvalConfig().to_dict(),
                "benchmark_paths": _webb_benchmark_paths(),
                "grounding": _profile_grounding_payload(
                    "debug",
                    sync_on_start=True,
                    planner_beta_enabled=True,
                ),
                "enforce_release_gates": False,
            }
        ),
        base / "eval-debug.json",
    )
    save_payload(
        _without_legacy_catalog_fields(
            {
                **EvalConfig().to_dict(),
                "run_name": "webbgpt-3b-eval",
                "benchmark_paths": _webb_benchmark_paths(),
                "grounding": _profile_grounding_payload(
                    "remote-3b",
                    sync_on_start=True,
                    planner_beta_enabled=True,
                ),
                "enforce_release_gates": True,
            }
        ),
        base / "eval-3b.json",
    )
    save_payload(
        _without_legacy_catalog_fields(
            {
                **EvalConfig().to_dict(),
                "run_name": "webbgpt-local-mvp-eval",
                "batch_size": 4,
                "max_new_tokens": 192,
                "decode_preset": "local-mvp-release-eval",
                "benchmark_paths": _webb_benchmark_paths(),
                "grounding": _profile_grounding_payload(
                    "local-mvp",
                    sync_on_start=True,
                    planner_beta_enabled=True,
                ),
                "enforce_release_gates": True,
                "repetition_penalty": 1.05,
                "no_repeat_ngram_size": 4,
                "release_gates": {
                    "assistant_pass_rate_min": 0.8,
                    "chat_sanity_pass_rate_min": 0.8,
                },
            }
        ),
        base / "eval-local-mvp.json",
    )
    save_payload(
        _without_legacy_catalog_fields(
            {
                **EvalConfig().to_dict(),
                "run_name": "webbgpt-7b-eval",
                "benchmark_paths": _webb_benchmark_paths(),
                "grounding": _profile_grounding_payload(
                    "remote-7b",
                    sync_on_start=True,
                    planner_beta_enabled=True,
                ),
                "enforce_release_gates": True,
            }
        ),
        base / "eval-7b.json",
    )
    save_payload(
        _without_legacy_catalog_fields(
            {
                **ServeConfig().to_dict(),
                "model_name": "webbgpt-debug",
                "checkpoint_path": "artifacts/runs/debug/export/final",
                "grounding": _profile_grounding_payload(
                    "debug",
                    sync_on_start=False,
                    planner_beta_enabled=False,
                ),
            }
        ),
        base / "serve-debug.json",
    )
    save_payload(
        _without_legacy_catalog_fields(
            {
                **ServeConfig().to_dict(),
                "model_name": "webbgpt-3b-instruct",
                "checkpoint_path": "artifacts/runs/remote-3b/export/final",
                "grounding": _profile_grounding_payload(
                    "remote-3b",
                    sync_on_start=False,
                    planner_beta_enabled=False,
                ),
            }
        ),
        base / "serve-3b.json",
    )
    save_payload(
        _without_legacy_catalog_fields(
            {
                **ServeConfig().to_dict(),
                "model_name": "webbgpt-local-mvp",
                "checkpoint_path": "artifacts/runs/local-mvp/export/final",
                "tokenizer_path": "artifacts/runs/local-mvp/export/final",
                "max_model_len": 512,
                "grounding": _profile_grounding_payload(
                    "local-mvp",
                    sync_on_start=False,
                    planner_beta_enabled=False,
                ),
                "decode_preset": "local-mvp-serve",
                "repetition_penalty": 1.05,
                "no_repeat_ngram_size": 4,
                "transcript_path": "artifacts/runs/local-mvp/manual_demos/chat_transcript.jsonl",
            }
        ),
        base / "serve-local-mvp.json",
    )
    save_payload(
        _without_legacy_catalog_fields(
            {
                **ServeConfig().to_dict(),
                "model_name": "webbgpt-7b-instruct",
                "checkpoint_path": "artifacts/runs/remote-7b/export/final",
                "tokenizer_path": "artifacts/tokenizer/webbgpt-7b.model",
                "max_model_len": 4096,
                "tensor_parallel_size": 8,
                "grounding": _profile_grounding_payload(
                    "remote-7b",
                    sync_on_start=False,
                    planner_beta_enabled=False,
                ),
            }
        ),
        base / "serve-7b.json",
    )
    _write_default_local_examples()


def _write_default_local_examples() -> None:
    def _write_seed_file(path: Path, content: str) -> None:
        if path.exists():
            return
        path.write_text(content)

    data_dir = Path("data/local")
    data_dir.mkdir(parents=True, exist_ok=True)
    sft_examples = [
        {
            "messages": [
                {"role": "system", "content": "You are WebbGPT, a helpful assistant."},
                {"role": "user", "content": "Good morning"},
                {"role": "assistant", "content": "Good morning, Harry. How can I help today?"},
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "How are you?"},
                {
                    "role": "assistant",
                    "content": "I am doing well and ready to help. What would you like to work on?",
                },
            ]
        },
        {
            "messages": [
                {"role": "system", "content": "You are WebbGPT, a helpful assistant."},
                {"role": "user", "content": "Explain what a course prerequisite is."},
                {
                    "role": "assistant",
                    "content": "A course prerequisite is a class or requirement you should complete before taking a more advanced course.",
                },
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "Compare AdvSt Chemistry and Technology and Innovation Seminar I as course options."},
                {
                    "role": "assistant",
                    "content": "AdvSt Chemistry emphasizes advanced laboratory chemistry and quantitative science work. Technology and Innovation Seminar I emphasizes design, technology, and applied problem solving tied to math placement. They overlap in problem solving but ask for different preparation and workload.",
                },
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "What should you do when information is missing?"},
                {
                    "role": "assistant",
                    "content": "I should say what is uncertain, avoid guessing, and explain how to verify the missing information.",
                },
            ]
        },
    ]
    preference_examples = [
        {
            "prompt": [{"role": "user", "content": "Help me choose between two intro courses."}],
            "chosen": "I can help compare them if you share your interests, workload constraints, and whether you prefer theory or hands-on work.",
            "rejected": "Take whatever sounds coolest.",
        },
        {
            "prompt": [{"role": "user", "content": "What should I do if catalog information is missing?"}],
            "chosen": "I should say what is missing, avoid guessing, and recommend checking the current catalog or advisor for confirmation.",
            "rejected": "I will just infer the missing requirements from memory.",
        },
    ]
    sft_validation_examples = [
        {
            "messages": [
                {"role": "user", "content": "What does it mean when a course is recommended but not required?"},
                {
                    "role": "assistant",
                    "content": "It means the course may help you prepare, but it is not a formal prerequisite you must complete before enrolling.",
                },
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "How should you answer if prerequisites are unclear?"},
                {
                    "role": "assistant",
                    "content": "I should say what is uncertain, avoid inventing requirements, and point the student to the current catalog or an advisor for confirmation.",
                },
            ]
        },
    ]
    preference_validation_examples = [
        {
            "prompt": [{"role": "user", "content": "Can you help me think through two introductory classes?"}],
            "chosen": "Yes. Tell me the two classes, what kind of work you enjoy, and any schedule or workload constraints, and I can help compare them.",
            "rejected": "Just take the easier one.",
        },
        {
            "prompt": [{"role": "user", "content": "What should you do if a course detail seems uncertain?"}],
            "chosen": "I should be honest about the uncertainty, avoid guessing, and recommend checking the latest catalog entry or asking an advisor.",
            "rejected": "I should make the best guess and present it as a fact.",
        },
    ]
    _write_seed_file(data_dir / "sft.jsonl", "\n".join(json.dumps(row) for row in sft_examples) + "\n")
    _write_seed_file(
        data_dir / "sft_validation.jsonl",
        "\n".join(json.dumps(row) for row in sft_validation_examples) + "\n"
    )
    _write_seed_file(
        data_dir / "preference.jsonl",
        "\n".join(json.dumps(row) for row in preference_examples) + "\n"
    )
    _write_seed_file(
        data_dir / "preference_validation.jsonl",
        "\n".join(json.dumps(row) for row in preference_validation_examples) + "\n"
    )

    domain_dir = Path("data/domain")
    domain_dir.mkdir(parents=True, exist_ok=True)
    _write_seed_file(
        domain_dir / "education_corpus.txt",
        "\n".join(
            [
                "Education writing should explain concepts clearly, use examples, and surface the assumptions behind a recommendation.",
                "A helpful academic assistant should clarify student goals, timeline, and constraints before suggesting next steps.",
                "Strong advising language avoids overconfidence and distinguishes policy from suggestion.",
            ]
        )
        + "\n"
    )
    _write_seed_file(
        domain_dir / "advising_corpus.txt",
        "\n".join(
            [
                "Academic advising should help a student compare options by interests, workload, scheduling constraints, and long-term goals.",
                "When information is missing, the assistant should say what is uncertain and recommend checking the current catalog or advisor.",
                "Program planning should separate required courses, recommended sequencing, and elective flexibility.",
            ]
        )
        + "\n"
    )
    _write_seed_file(
        domain_dir / "philosophy_corpus.txt",
        "\n".join(
            [
                "Philosophy training strengthens careful reading, argument reconstruction, objection handling, and conceptual clarity.",
                "Good explanations compare positions fairly before identifying important differences.",
                "A concise answer can still be rigorous if it defines terms and states the relevant distinction.",
            ]
        )
        + "\n"
    )
    _write_seed_file(
        domain_dir / "catalog_corpus.txt",
        "\n".join(
            [
                "Course catalogs describe programs, prerequisites, credits, terms, and sections, but they should not be treated as proof of live seat availability without current term data.",
                "A grounded catalog assistant should cite the course or program entry it relied on.",
                "If no matching course is found, the assistant should abstain instead of guessing the requirements.",
            ]
        )
        + "\n"
    )

    posttrain_dir = Path("data/posttrain")
    posttrain_dir.mkdir(parents=True, exist_ok=True)
    public_sft_examples = [
        {
            "messages": [
                {"role": "system", "content": "You are WebbGPT, a thoughtful and clear assistant."},
                {"role": "user", "content": "Good morning"},
                {"role": "assistant", "content": "Good morning. What would you like help with today?"},
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "How are you?"},
                {
                    "role": "assistant",
                    "content": "I am doing well and ready to help. What are you working on?",
                },
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "What is a course prerequisite?"},
                {
                    "role": "assistant",
                    "content": "A course prerequisite is a requirement you are expected to complete before enrolling in a more advanced course.",
                },
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "Compare AdvSt Chemistry and Technology and Innovation Seminar I as course options."},
                {
                    "role": "assistant",
                    "content": "AdvSt Chemistry is a lab-centered advanced science course with quantitative chemistry work. Technology and Innovation Seminar I is a design-oriented math and computer science option. A useful comparison should start with the student's interests, current math placement, workload tolerance, and goals.",
                },
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "How are you today?"},
                {
                    "role": "assistant",
                    "content": "I am doing well and ready to help. What would you like to work on today?",
                },
            ]
        },
    ]
    domain_sft_examples = [
        {
            "messages": [
                {"role": "user", "content": "What should you do when catalog information is missing?"},
                {
                    "role": "assistant",
                    "content": "I should say what is missing, avoid guessing, and recommend checking the current catalog or an advisor for confirmation.",
                },
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "Should I take AdvSt Chemistry or Technology and Innovation Seminar I first?"},
                {
                    "role": "assistant",
                    "content": "That depends on your goals, current math placement, and workload. AdvSt Chemistry is the better fit for advanced laboratory chemistry, while Technology and Innovation Seminar I is better for design and applied technology work.",
                },
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "How does AdvSt Chemistry work?"},
                {
                    "role": "assistant",
                    "content": "AdvSt Chemistry is an advanced Webb science course. I should answer with grounded catalog details and include a citation when catalog facts are used.",
                },
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "What if I ask about a course that is not in the catalog?"},
                {
                    "role": "assistant",
                    "content": "I should say that I could not find a matching catalog entry, avoid inventing requirements, and suggest checking the current catalog or an advisor.",
                },
            ]
        },
    ]
    conversation_sft_examples = [
        {
            "messages": [
                {"role": "user", "content": "Hello"},
                {
                    "role": "assistant",
                    "content": "Hello. I am here and ready to help. What would you like to work on?",
                },
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "Can you say hi back to me?"},
                {"role": "assistant", "content": "Hi, Harry. It is good to hear from you."},
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "Explain recursion in plain English."},
                {
                    "role": "assistant",
                    "content": "Recursion is when a function solves a problem by calling a smaller version of itself until it reaches a simple base case.",
                },
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "I am choosing between two majors. How should I think about it?"},
                {
                    "role": "assistant",
                    "content": "Start with the kind of work you enjoy, then compare the majors by required courses, workload style, and where you hope they lead.",
                },
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "What if you are not sure about the answer?"},
                {
                    "role": "assistant",
                    "content": "I should be explicit about the uncertainty, avoid inventing details, and suggest the best way to verify the answer.",
                },
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "How does Hon Museum Research work?"},
                {
                    "role": "assistant",
                    "content": "If I use catalog facts, I should explain them clearly in plain English and include a citation instead of answering from memory.",
                },
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "Give me a short answer: what is computer science?"},
                {
                    "role": "assistant",
                    "content": "Computer science is the study of computation, programming, systems, and how information can be represented and processed.",
                },
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "How should an advisor-like assistant sound?"},
                {
                    "role": "assistant",
                    "content": "It should sound calm, clear, and practical. It should ask useful follow-up questions instead of pretending it already knows everything.",
                },
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "What do you do if I ask about Honors Robotics Lab and it is not in the catalog?"},
                {
                    "role": "assistant",
                    "content": "I should say I could not find Honors Robotics Lab in the catalog, avoid guessing the prerequisites, and suggest checking the current catalog or an advisor.",
                },
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "How are you today?"},
                {
                    "role": "assistant",
                    "content": "I am doing well and ready to help. Tell me what you want to figure out.",
                },
            ]
        },
    ]
    posttrain_sft_validation_examples = [
        {
            "messages": [
                {"role": "user", "content": "How should a grounded assistant explain catalog facts?"},
                {
                    "role": "assistant",
                    "content": "It should explain the catalog details in plain English, separate facts from advice, and cite the catalog entry when it relies on those facts.",
                },
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "What should you ask before helping with major selection?"},
                {
                    "role": "assistant",
                    "content": "I should ask about the student’s interests, strengths, preferred workload, and what they hope to do after graduation before comparing majors.",
                },
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "Explain recursion with a simple analogy."},
                {
                    "role": "assistant",
                    "content": "Recursion is like opening a nested set of boxes: you solve the smallest box first, then use that result to finish the bigger ones.",
                },
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "What is the difference between a prerequisite and a recommendation?"},
                {
                    "role": "assistant",
                    "content": "A prerequisite is something you are expected to complete before a course, while a recommendation is advice that may help but is not a formal requirement.",
                },
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "How should an assistant respond to missing course data?"},
                {
                    "role": "assistant",
                    "content": "It should say what information is unavailable, avoid guessing, and suggest the current catalog or an advisor as the best way to verify the missing detail.",
                },
            ]
        },
    ]
    public_preference_examples = [
        {
            "prompt": [{"role": "user", "content": "Good morning"}],
            "chosen": "Good morning. How can I help today?",
            "rejected": ",,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,",
        },
        {
            "prompt": [{"role": "user", "content": "How are you?"}],
            "chosen": "I am doing well and ready to help. What would you like to work on?",
            "rejected": "or and or and and or or and and and or.",
        },
        {
            "prompt": [{"role": "user", "content": "What should you do when information is missing?"}],
            "chosen": "I should acknowledge uncertainty, avoid guessing, and explain how to verify the missing information.",
            "rejected": "I should invent the most likely answer and say it confidently.",
        },
    ]
    domain_preference_examples = [
        {
            "prompt": [{"role": "user", "content": "Help me compare two majors."}],
            "chosen": "I can compare them by interests, workload style, quantitative demands, and likely outcomes if you share the two majors.",
            "rejected": "Pick whichever one sounds cooler.",
        },
        {
            "prompt": [{"role": "user", "content": "How does AdvSt Chemistry work?"}],
            "chosen": "I should use the Webb catalog entry for AdvSt Chemistry, explain it in plain English, and include a citation instead of free-associating.",
            "rejected": "I should answer from vague memory without citing the catalog.",
        },
        {
            "prompt": [{"role": "user", "content": "What are the prerequisites for Honors Robotics Lab?"}],
            "chosen": "If the Webb catalog does not show Honors Robotics Lab, I should say I could not find it and avoid inventing a prerequisite chain.",
            "rejected": "I should infer a plausible prerequisite chain and state it confidently.",
        },
    ]
    conversation_preference_examples = [
        {
            "prompt": [{"role": "user", "content": "Good morning"}],
            "chosen": "Good morning. How can I help today?",
            "rejected": "and and and and and and and",
        },
        {
            "prompt": [{"role": "user", "content": "How are you today?"}],
            "chosen": "I am doing well and ready to help. What would you like to work on today?",
            "rejected": ",,,,,,,,,,,,,,,,,,,,,,,,,,,,,",
        },
        {
            "prompt": [{"role": "user", "content": "Say hi back"}],
            "chosen": "Hi, Harry. What would you like help with?",
            "rejected": "or and or and or and or and",
        },
        {
            "prompt": [{"role": "user", "content": "Explain recursion briefly."}],
            "chosen": "Recursion solves a problem by reducing it to smaller versions of the same problem until a base case is reached.",
            "rejected": "Recursion is recursion because recursion recursively recurses forever.",
        },
        {
            "prompt": [{"role": "user", "content": "I am unsure which major to choose."}],
            "chosen": "I can help if you share the majors, your interests, and whether you prefer theory, writing, or hands-on technical work.",
            "rejected": "Just pick the one with the coolest name.",
        },
        {
            "prompt": [{"role": "user", "content": "What should you do when information is missing?"}],
            "chosen": "I should say what is uncertain, avoid guessing, and explain how to verify the missing information.",
            "rejected": "I should fill in the gaps with whatever seems likely.",
        },
        {
            "prompt": [{"role": "user", "content": "How does Hon Museum Research work?"}],
            "chosen": "I should use the Webb catalog entry, explain it clearly, and include a citation if I rely on catalog facts.",
            "rejected": "I should answer from vague memory and skip the citation.",
        },
        {
            "prompt": [{"role": "user", "content": "What are the prerequisites for Technology and Innovation Seminar III?"}],
            "chosen": "If I cannot find Technology and Innovation Seminar III in the Webb catalog, I should say so and avoid inventing a prerequisite chain.",
            "rejected": "I should guess that it probably requires calculus and linear algebra.",
        },
    ]
    posttrain_preference_validation_examples = [
        {
            "prompt": [{"role": "user", "content": "How should you greet a student who asks for help?"}],
            "chosen": "I should greet them warmly, stay concise, and invite them to share what they want help with.",
            "rejected": "I should ramble until they figure out whether I am friendly.",
        },
        {
            "prompt": [{"role": "user", "content": "What should you ask before comparing two majors?"}],
            "chosen": "I should ask about interests, workload preferences, strengths, and goals before comparing two majors.",
            "rejected": "I should immediately pick one major without asking anything.",
        },
        {
            "prompt": [{"role": "user", "content": "What should you do if the Webb catalog does not list Honors Robotics Lab?"}],
            "chosen": "I should say I could not verify Honors Robotics Lab in the Webb catalog and avoid inventing prerequisites or course details.",
            "rejected": "I should make up a likely catalog description for Honors Robotics Lab.",
        },
        {
            "prompt": [{"role": "user", "content": "Explain a prerequisite briefly."}],
            "chosen": "A prerequisite is a requirement you usually complete before taking a course.",
            "rejected": "A prerequisite is whatever sounds useful, whether the catalog lists it or not.",
        },
        {
            "prompt": [{"role": "user", "content": "How should an assistant handle uncertainty?"}],
            "chosen": "It should say what is uncertain, avoid bluffing, and explain the best way to verify the answer.",
            "rejected": "It should sound confident even when it does not know.",
        },
    ]
    _write_seed_file(
        posttrain_dir / "sft_public_seed.jsonl",
        "\n".join(json.dumps(row) for row in public_sft_examples) + "\n"
    )
    _write_seed_file(
        posttrain_dir / "sft_domain_synthetic.jsonl",
        "\n".join(json.dumps(row) for row in domain_sft_examples) + "\n"
    )
    _write_seed_file(
        posttrain_dir / "sft_conversation_seed.jsonl",
        "\n".join(json.dumps(row) for row in conversation_sft_examples) + "\n"
    )
    _write_seed_file(
        posttrain_dir / "sft_validation.jsonl",
        "\n".join(json.dumps(row) for row in posttrain_sft_validation_examples) + "\n"
    )
    _write_seed_file(
        posttrain_dir / "preference_public_seed.jsonl",
        "\n".join(json.dumps(row) for row in public_preference_examples) + "\n"
    )
    _write_seed_file(
        posttrain_dir / "preference_domain_synthetic.jsonl",
        "\n".join(json.dumps(row) for row in domain_preference_examples) + "\n"
    )
    _write_seed_file(
        posttrain_dir / "preference_conversation_seed.jsonl",
        "\n".join(json.dumps(row) for row in conversation_preference_examples) + "\n"
    )
    _write_seed_file(
        posttrain_dir / "preference_validation.jsonl",
        "\n".join(json.dumps(row) for row in posttrain_preference_validation_examples) + "\n"
    )

    eval_dir = Path("data/eval")
    eval_dir.mkdir(parents=True, exist_ok=True)
    chat_sanity_benchmarks = [
        {
            "messages": [{"role": "user", "content": "Good morning"}],
            "expected_substrings": ["good morning"],
            "forbidden_substrings": [",,,,,", "or and", "what are you working on"],
            "pass_score": 1.0,
        },
        {
            "messages": [{"role": "user", "content": "How are you?"}],
            "expected_substrings": ["help", "well"],
            "forbidden_substrings": [",,,,,", "or and", "what are you working on"],
            "pass_score": 1.0,
        },
        {
            "messages": [{"role": "user", "content": "Explain what a prerequisite is in one sentence."}],
            "expected_substrings": ["prerequisite"],
            "forbidden_substrings": [",,,,,", "or and"],
            "pass_score": 1.0,
        },
        {
            "messages": [{"role": "user", "content": "What should you do when information is missing?"}],
            "expected_substrings": ["avoid", "guess"],
            "forbidden_substrings": [",,,,,", "or and"],
            "pass_score": 1.0,
        },
        {
            "messages": [{"role": "user", "content": "I missed class and need a plan for catching up this week."}],
            "expected_substrings": ["plan", "week"],
            "forbidden_substrings": ["what would you like to work on", "what are you working on"],
            "pass_score": 1.0,
        },
        {
            "messages": [{"role": "user", "content": "Explain recursion with a simple analogy."}],
            "expected_substrings": ["smaller", "problem"],
            "forbidden_substrings": [",,,,,", "or and"],
            "pass_score": 1.0,
        },
        {
            "messages": [{"role": "user", "content": "What should you ask before recommending between two majors?"}],
            "expected_substrings": ["interests", "goals"],
            "forbidden_substrings": ["what would you like to work on"],
            "pass_score": 1.0,
        },
        {
            "messages": [{"role": "user", "content": "I am overwhelmed by course planning. How should an assistant respond?"}],
            "expected_substrings": ["break", "step"],
            "forbidden_substrings": ["what are you working on"],
            "pass_score": 1.0,
        },
    ]
    assistant_benchmarks = [
        {
            "messages": [{"role": "user", "content": "What is a course prerequisite?"}],
            "expected_substrings": ["prerequisite"],
            "pass_score": 1.0,
        },
        {
            "messages": [{"role": "user", "content": "Compare AdvSt Chemistry and Technology and Innovation Seminar I."}],
            "expected_substrings": ["chemistry", "technology"],
            "pass_score": 1.0,
        },
        {
            "messages": [{"role": "user", "content": "How should an assistant respond when the catalog is missing data?"}],
            "expected_substrings": ["uncertain", "verify"],
            "pass_score": 1.0,
        },
        {
            "messages": [{"role": "user", "content": "Should I take AdvSt Chemistry or Technology and Innovation Seminar I first?"}],
            "expected_substrings": ["chemistry", "technology"],
            "forbidden_substrings": ["what would you like to work on"],
            "pass_score": 1.0,
        },
        {
            "messages": [{"role": "user", "content": "What is the difference between a prerequisite and a recommendation?"}],
            "expected_substrings": ["prerequisite", "recommendation"],
            "pass_score": 1.0,
        },
        {
            "messages": [{"role": "user", "content": "What should you ask before helping someone choose a major?"}],
            "expected_substrings": ["interests", "goals"],
            "pass_score": 1.0,
        },
        {
            "messages": [{"role": "user", "content": "A student works 20 hours a week and feels overwhelmed by planning. How should you respond?"}],
            "expected_substrings": ["constraints", "step"],
            "forbidden_substrings": ["what are you working on"],
            "pass_score": 1.0,
        },
        {
            "messages": [{"role": "user", "content": "Explain recursion briefly."}],
            "expected_substrings": ["smaller", "base case"],
            "pass_score": 1.0,
        },
    ]
    posttrain_regression_prompts = [
        {
            "messages": [{"role": "user", "content": "I missed class and need a plan for catching up this week."}],
            "tags": ["general", "supportive"],
        },
        {
            "messages": [{"role": "user", "content": "What should you ask before recommending between two majors?"}],
            "tags": ["domain", "advising"],
        },
        {
            "messages": [{"role": "user", "content": "If the Webb catalog does not list Honors Robotics Lab, how should you respond?"}],
            "tags": ["domain", "grounded"],
        },
        {
            "messages": [{"role": "user", "content": "Explain recursion using a simple everyday analogy."}],
            "tags": ["general", "reasoning"],
        },
        {
            "messages": [{"role": "user", "content": "How would you explain the difference between a prerequisite and a recommendation to a new student?"}],
            "tags": ["domain", "definition"],
        },
        {
            "messages": [{"role": "user", "content": "I am overwhelmed by course planning. How should an assistant respond?"}],
            "tags": ["off_distribution", "supportive"],
        },
    ]
    catalog_benchmarks = [
        {
            "messages": [{"role": "user", "content": "What does AdvSt Chemistry require?"}],
            "expected_course_codes": ["AdvSt Chemistry"],
            "requires_citation": True,
        },
        {
            "messages": [{"role": "user", "content": "What is Technology and Innovation Seminar I?"}],
            "expected_course_codes": ["Technology and Innovation Seminar I"],
            "requires_citation": True,
        },
        {
            "messages": [{"role": "user", "content": "Tell me about Hon Museum Research."}],
            "expected_course_codes": ["Hon Museum Research"],
            "requires_citation": True,
        },
        {
            "messages": [{"role": "user", "content": "Which Webb course covers advanced computer science with HMC?"}],
            "expected_course_codes": ["HMC / Webb Advanced Courses in Computer Science"],
            "requires_citation": True,
        },
        {
            "messages": [{"role": "user", "content": "What is AdvSt Paleontology?"}],
            "expected_course_codes": ["AdvSt Paleontology"],
            "requires_citation": True,
        },
        {
            "messages": [{"role": "user", "content": "Which Webb course follows Art AB in the visual arts sequence?"}],
            "expected_course_codes": ["Advanced Art"],
            "requires_citation": True,
        },
    ]
    missing_catalog_benchmarks = [
        {
            "messages": [{"role": "user", "content": "What are the prerequisites for Honors Robotics Lab?"}],
            "expected_course_codes": [],
            "expects_abstention": True,
            "requires_citation": False,
        },
        {
            "messages": [{"role": "user", "content": "Tell me the current seats available for AdvSt Chemistry."}],
            "expected_course_codes": [],
            "expects_abstention": True,
            "requires_citation": False,
        },
        {
            "messages": [{"role": "user", "content": "Does Technology and Innovation Seminar III exist in the Webb catalog?"}],
            "expected_course_codes": [],
            "expects_abstention": True,
            "requires_citation": False,
        },
        {
            "messages": [{"role": "user", "content": "What is Advanced Marine Biology Expedition at Webb?"}],
            "expected_course_codes": [],
            "expects_abstention": True,
            "requires_citation": False,
        },
        {
            "messages": [{"role": "user", "content": "Can you confirm whether Honors Robotics Lab is offered this term?"}],
            "expected_course_codes": [],
            "expects_abstention": True,
            "requires_citation": False,
        },
        {
            "messages": [{"role": "user", "content": "What are the prerequisites for Webb Entrepreneurship 404?"}],
            "expected_course_codes": [],
            "expects_abstention": True,
            "requires_citation": False,
        },
    ]
    _write_seed_file(
        eval_dir / "chat_sanity.jsonl",
        "\n".join(json.dumps(row) for row in chat_sanity_benchmarks) + "\n"
    )
    _write_seed_file(
        eval_dir / "assistant.jsonl",
        "\n".join(json.dumps(row) for row in assistant_benchmarks) + "\n"
    )
    _write_seed_file(
        eval_dir / "posttrain_regression.jsonl",
        "\n".join(json.dumps(row) for row in posttrain_regression_prompts) + "\n"
    )
    _write_seed_file(
        eval_dir / "catalog.responses",
        "\n".join(json.dumps(row) for row in catalog_benchmarks) + "\n"
    )
    _write_seed_file(
        eval_dir / "catalog_missing.responses",
        "\n".join(json.dumps(row) for row in missing_catalog_benchmarks) + "\n"
    )

    catalog_dir = Path("data/catalog")
    catalog_dir.mkdir(parents=True, exist_ok=True)
    catalog_payload = {
        "institutions": [
            {"id": "webb", "name": "The Webb Schools", "website": "https://www.webb.org"}
        ],
        "terms": [
            {
                "id": "webb-2026-27",
                "institution_id": "webb",
                "code": "2026-27",
                "title": "2026-27 Course Catalog",
                "starts_on": "2026-08-01",
                "ends_on": "2027-06-30",
            }
        ],
        "programs": [
            {
                "id": "webb-course-catalog",
                "institution_id": "webb",
                "code": "WEBB-CATALOG",
                "title": "Webb Course Catalog",
                "description": "Webb's course catalog describes graduation requirements, course planning, departments, course offerings, prerequisites, and workload expectations.",
                "requirements": {"minimum_credits": 20, "source": "data/webb/mock/course_catalog_2025_26.html"},
            }
        ],
        "courses": [
            {
                "id": "webb-advst-chemistry",
                "institution_id": "webb",
                "program_id": "webb-course-catalog",
                "code": "AdvSt Chemistry",
                "title": "AdvSt Chemistry",
                "description": "A project-based advanced chemistry course built around an in-depth study of general chemistry and a significant laboratory component. The catalog lists B+ or better in Integrated Math II or Honors Integrated Math II, with Precalculus concurrent, as the prerequisite.",
                "credits": 1.0,
                "prerequisites": {"text": "B+ or better in Integrated Math II or Honors Integrated Math II; Precalculus concurrent"},
                "attributes": {"department": "Science", "workload": "1.25", "source": "data/webb/mock/course_catalog_2025_26.html"},
            },
            {
                "id": "webb-technology-innovation-seminar-i",
                "institution_id": "webb",
                "program_id": "webb-course-catalog",
                "code": "Technology and Innovation Seminar I",
                "title": "Technology and Innovation Seminar I",
                "description": "A Webb mathematics and computer science course listed with a concurrent level-appropriate math course prerequisite and workload 1.",
                "credits": 1.0,
                "prerequisites": {"text": "Concurrent level-appropriate math course"},
                "attributes": {"department": "Mathematics & Computer Science", "workload": "1", "source": "data/webb/mock/course_catalog_2025_26.html"},
            },
            {
                "id": "webb-hon-museum-research",
                "institution_id": "webb",
                "program_id": "webb-course-catalog",
                "code": "Hon Museum Research",
                "title": "Hon Museum Research",
                "description": "This course builds on skills learned in Honors Paleontology and focuses on scientific study of fossils, research communication, fossil data interpretation, and formal scientific reporting using Alf Museum collections.",
                "credits": 1.0,
                "prerequisites": {"text": "Hon Paleontology or concurrent second science"},
                "attributes": {"department": "Science", "workload": "1", "source": "data/webb/mock/science_2026_27.html"},
            },
            {
                "id": "webb-hmc-advanced-cs",
                "institution_id": "webb",
                "program_id": "webb-course-catalog",
                "code": "HMC / Webb Advanced Courses in Computer Science",
                "title": "HMC / Webb Advanced Courses in Computer Science",
                "description": "An advanced mathematics and computer science offering in the Webb catalog. The catalog lists concurrent level-appropriate math as the prerequisite and workload 1.25.",
                "credits": 1.0,
                "prerequisites": {"text": "Concurrent level-appropriate math course"},
                "attributes": {"department": "Mathematics & Computer Science", "workload": "1.25", "source": "data/webb/mock/course_catalog_2025_26.html"},
            },
            {
                "id": "webb-advst-paleontology",
                "institution_id": "webb",
                "program_id": "webb-course-catalog",
                "code": "AdvSt Paleontology",
                "title": "AdvSt Paleontology",
                "description": "An advanced original-research course on fossils in the Alf Museum collection. Students review paleontological literature, collect and analyze original data, write formal manuscripts, and present their work.",
                "credits": 1.0,
                "prerequisites": {"text": "Hon Museum Research or concurrent second science"},
                "attributes": {"department": "Science", "workload": "1.25", "source": "data/webb/mock/science_2026_27.html"},
            },
            {
                "id": "webb-advanced-art",
                "institution_id": "webb",
                "program_id": "webb-course-catalog",
                "code": "Advanced Art",
                "title": "Advanced Art",
                "description": "A Webb fine arts course in the visual arts sequence. The catalog lists Art AB as the prerequisite and workload 1.",
                "credits": 1.0,
                "prerequisites": {"text": "Art AB"},
                "attributes": {"department": "Fine Arts", "workload": "1", "source": "data/webb/mock/course_catalog_2025_26.html"},
            },
        ],
        "sections": [],
    }
    _write_seed_file(catalog_dir / "webb_catalog.json", json.dumps(catalog_payload, indent=2) + "\n")

    continue_documents: list[str] = []
    for corpus_path in (
        domain_dir / "education_corpus.txt",
        domain_dir / "advising_corpus.txt",
        domain_dir / "philosophy_corpus.txt",
        domain_dir / "catalog_corpus.txt",
    ):
        for line in corpus_path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            continue_documents.append(
                "WebbGPT local-MVP domain guidance: "
                f"{line} This text should shape a calm, grounded academic assistant that explains clearly, "
                "states uncertainty honestly, and avoids guessing when catalog information is incomplete."
            )

    institutions = {item["id"]: item for item in catalog_payload["institutions"]}
    programs = {item["id"]: item for item in catalog_payload["programs"]}
    courses = {item["id"]: item for item in catalog_payload["courses"]}
    terms = {item["id"]: item for item in catalog_payload["terms"]}
    for program in catalog_payload["programs"]:
        institution = institutions.get(program["institution_id"], {})
        requirements = program.get("requirements") or {}
        minimum_credits = requirements.get("minimum_credits")
        continue_documents.append(
            f"{institution.get('name', 'The institution')} offers program {program['code']}, titled {program['title']}. "
            f"{program['description']} The program currently requires at least {minimum_credits} credits. "
            "A grounded assistant should present program requirements as catalog facts, separate facts from advice, "
            "and recommend checking the current catalog when a student needs term-specific confirmation."
        )
    for course in catalog_payload["courses"]:
        program = programs.get(course.get("program_id"))
        program_text = (
            f"The course belongs to the program {program['code']} titled {program['title']}."
            if program is not None
            else "The course is not tied to a specific demo program entry."
        )
        prerequisites = course.get("prerequisites") or "No prerequisite is listed in the demo catalog."
        level = (course.get("attributes") or {}).get("level", "unspecified")
        continue_documents.append(
            f"Catalog entry for {course['code']} {course['title']}: {course['description']} "
            f"It carries {course['credits']} credits and is labeled {level}. {program_text} "
            f"Prerequisite status: {prerequisites} A careful assistant should explain the course in plain English, "
            "cite the catalog when using these details, and avoid inventing requirements that are not shown."
        )
    for section in catalog_payload["sections"]:
        course = courses.get(section["course_id"], {})
        term = terms.get(section["term_id"], {})
        meeting_times = section.get("meeting_times") or {}
        meeting_days = ", ".join(meeting_times.get("days") or [])
        continue_documents.append(
            f"Section data for {course.get('code', 'the course')} in term {term.get('code', 'unknown term')}: "
            f"instructor {section['instructor']}, meeting on {meeting_days} at {meeting_times.get('time', 'unknown time')}, "
            f"with modality {section['modality']}. The section has {section['seats_available']} available seats out of "
            f"{section['seats_total']} total seats. A grounded assistant should treat section and seat data as term-specific "
            "catalog facts and should not generalize live availability beyond the listed term."
        )

    _write_seed_file(
        domain_dir / "local_mvp_continue_corpus.txt",
        "\n".join(document.strip() for document in continue_documents if document.strip()) + "\n"
    )


def _completed_checkpoint_dirs(output_dir: str | Path) -> list[Path]:
    return sorted(
        path
        for path in Path(output_dir).glob("step-*")
        if path.is_dir() and not path.name.endswith(".tmp")
    )


def _latest_checkpoint_dir(output_dir: str | Path) -> str:
    candidates = _completed_checkpoint_dirs(output_dir)
    if not candidates:
        raise RuntimeError(f"No completed checkpoints were found in {output_dir}.")
    return str(candidates[-1])


def _checkpoint_step(checkpoint_dir: str) -> int:
    name = Path(checkpoint_dir).name
    if not name.startswith("step-"):
        raise RuntimeError(f"Could not infer step number from checkpoint path {checkpoint_dir}.")
    return int(name.split("step-", 1)[1])


def _profile_config_paths(base: Path, profile: str) -> dict[str, Path]:
    if profile == "debug":
        return {
            "tokenizer": base / "tokenizer.json",
            "tokenizer_corpus": base / "tokenizer-corpus.json",
            "model": base / "model-debug.json",
            "data": base / "data-debug.json",
            "train": base / "train-debug.json",
            "eval": base / "eval-debug.json",
            "serve": base / "serve-debug.json",
        }
    if profile == "local-mvp":
        return {
            "tokenizer": base / "tokenizer-local-mvp.json",
            "tokenizer_corpus": base / "tokenizer-corpus-local-mvp.json",
            "model": base / "model-local-mvp.json",
            "data": base / "data-local-mvp.json",
            "train": base / "train-local-mvp.json",
            "eval": base / "eval-local-mvp.json",
            "serve": base / "serve-local-mvp.json",
        }
    if profile == "remote-3b":
        return {
            "tokenizer": base / "tokenizer.json",
            "tokenizer_corpus": base / "tokenizer-corpus.json",
            "model": base / "model-3b.json",
            "data": base / "data-3b.json",
            "train": base / "train-3b.json",
            "eval": base / "eval-3b.json",
            "serve": base / "serve-3b.json",
        }
    if profile == "remote-7b":
        return {
            "tokenizer": base / "tokenizer-7b.json",
            "tokenizer_corpus": base / "tokenizer-corpus-7b.json",
            "model": base / "model-7b.json",
            "data": base / "data-7b.json",
            "train": base / "train-7b.json",
            "eval": base / "eval-7b.json",
            "serve": base / "serve-7b.json",
        }
    raise ValueError(f"Unknown profile {profile!r}")


def _require_profile_configs(base: Path, profile: str) -> dict[str, Path]:
    config_paths = _profile_config_paths(base, profile)
    missing = [str(path) for path in config_paths.values() if not path.exists()]
    if missing:
        raise RuntimeError(
            "The selected profile is missing starter config files. "
            f"Run `webbgpt init-config --output-dir {base}` first.\nMissing:\n- "
            + "\n- ".join(missing)
        )
    return config_paths


def _profile_artifact_root(profile: str) -> Path:
    return Path("artifacts/runs") / profile


def _profile_eval_dir(profile: str) -> Path:
    return _profile_artifact_root(profile) / "eval"


def _profile_export_dir(profile: str) -> Path:
    return _profile_artifact_root(profile) / "export" / "final"


def _profile_checkpoint_dir(profile: str, stage: str) -> Path:
    return _profile_artifact_root(profile) / "checkpoints" / stage


def _profile_prepared_dir(profile: str) -> Path:
    return _profile_artifact_root(profile) / "prepared"


def _uses_profile_runtime_layout(profile: str | None) -> bool:
    return profile in PROFILE_RUNTIME_MANUAL_PROFILES


def _legacy_profile_checkpoint_dir(profile: str) -> Path | None:
    return LEGACY_PROFILE_CHECKPOINT_DIRS.get(profile)


def _shared_config_pack_profile(
    model_config_path: str | Path,
    data_config_path: str | Path,
    train_config_path: str | Path,
) -> str | None:
    filenames = (
        Path(model_config_path).name,
        Path(data_config_path).name,
        Path(train_config_path).name,
    )
    profile_filenames = {
        "debug": ("model-debug.json", "data-debug.json", "train-debug.json"),
        "local-mvp": ("model-local-mvp.json", "data-local-mvp.json", "train-local-mvp.json"),
        "remote-3b": ("model-3b.json", "data-3b.json", "train-3b.json"),
        "remote-7b": ("model-7b.json", "data-7b.json", "train-7b.json"),
    }
    for profile, expected in profile_filenames.items():
        expected_model, expected_data, expected_train = expected
        train_stem = Path(filenames[2]).stem
        expected_train_stem = Path(expected_train).stem
        train_matches = train_stem == expected_train_stem or train_stem.startswith(
            f"{expected_train_stem}-"
        )
        if filenames[0] == expected_model and filenames[1] == expected_data and train_matches:
            return profile
    return None


def _backup_existing_stage_output(output_dir: Path, *, label: str) -> Path | None:
    if not output_dir.exists():
        return None
    has_existing_artifacts = any(
        child.name != ".DS_Store" for child in output_dir.iterdir()
    )
    if not has_existing_artifacts:
        return None
    backup_dir = output_dir.with_name(f"{output_dir.name}.{label}-{time.strftime('%Y%m%d-%H%M%S')}")
    if backup_dir.exists():
        raise RuntimeError(
            f"Could not back up existing stage output at {output_dir} because {backup_dir} already exists."
        )
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(output_dir), str(backup_dir))
    return backup_dir


def _apply_continue_stage_overrides(train_config: TrainConfig) -> TrainConfig:
    stage_config = TrainConfig.from_dict(train_config.to_dict())
    if stage_config.continued_learning_rate is not None:
        stage_config.learning_rate = stage_config.continued_learning_rate
    if stage_config.continued_min_learning_rate is not None:
        stage_config.min_learning_rate = stage_config.continued_min_learning_rate
    if stage_config.continued_warmup_steps is not None:
        stage_config.warmup_steps = stage_config.continued_warmup_steps
    if stage_config.continued_max_steps is not None:
        stage_config.max_steps = stage_config.continued_max_steps
    return stage_config


def _apply_sft_stage_overrides(train_config: TrainConfig) -> TrainConfig:
    stage_config = TrainConfig.from_dict(train_config.to_dict())
    if stage_config.sft_learning_rate is not None:
        stage_config.learning_rate = stage_config.sft_learning_rate
    if stage_config.sft_min_learning_rate is not None:
        stage_config.min_learning_rate = stage_config.sft_min_learning_rate
    if stage_config.sft_warmup_steps is not None:
        stage_config.warmup_steps = stage_config.sft_warmup_steps
    if stage_config.sft_max_steps is not None:
        stage_config.max_steps = stage_config.sft_max_steps
    return stage_config


def _apply_dpo_stage_overrides(train_config: TrainConfig) -> TrainConfig:
    stage_config = TrainConfig.from_dict(train_config.to_dict())
    if stage_config.dpo_learning_rate is not None:
        stage_config.learning_rate = stage_config.dpo_learning_rate
    if stage_config.dpo_min_learning_rate is not None:
        stage_config.min_learning_rate = stage_config.dpo_min_learning_rate
    if stage_config.dpo_warmup_steps is not None:
        stage_config.warmup_steps = stage_config.dpo_warmup_steps
    if stage_config.dpo_max_steps is not None:
        stage_config.max_steps = stage_config.dpo_max_steps
    return stage_config


def _manual_profile_pretrain_fallback_dirs(profile: str, train_config: TrainConfig) -> list[Path]:
    fallback_dirs: list[Path] = []
    for candidate in (
        Path(train_config.checkpoint.output_dir),
        _legacy_profile_checkpoint_dir(profile),
    ):
        if candidate is None or candidate in fallback_dirs:
            continue
        fallback_dirs.append(candidate)
    return fallback_dirs


def _prepare_manual_stage_checkpoint_dir(
    *,
    profile: str | None,
    stage_name: str,
    output_dir: str,
    resume_from: str | None,
    backup_label: str = "stale",
) -> str:
    if not _uses_profile_runtime_layout(profile):
        return output_dir
    stage_dir = _profile_checkpoint_dir(profile, stage_name)
    if resume_from is None:
        backup_dir = _backup_existing_stage_output(stage_dir, label=backup_label)
        if backup_dir is not None:
            print(
                f"WebbGPT: moved existing {stage_name} checkpoints to {backup_dir} "
                f"before starting a fresh {profile} {stage_name} run.",
                file=sys.stderr,
                flush=True,
            )
    return str(stage_dir)


def _prepare_manual_pretrain_train_config(
    model_config_path: str,
    data_config_path: str,
    train_config_path: str,
    train_config: TrainConfig,
) -> TrainConfig:
    stage_config = TrainConfig.from_dict(train_config.to_dict())
    profile = _shared_config_pack_profile(model_config_path, data_config_path, train_config_path)
    if not _uses_profile_runtime_layout(profile):
        return stage_config
    pretrain_dir = _prepare_manual_stage_checkpoint_dir(
        profile=profile,
        stage_name="pretrain",
        output_dir=stage_config.checkpoint.output_dir,
        resume_from=stage_config.checkpoint.resume_from,
    )
    stage_config.checkpoint.output_dir = pretrain_dir
    print(
        f"WebbGPT: {profile} manual pretrain will write checkpoints under {pretrain_dir}.",
        file=sys.stderr,
        flush=True,
    )
    return stage_config


def _prepare_manual_continue_train_config(
    model_config_path: str,
    data_config_path: str,
    train_config_path: str,
    train_config: TrainConfig,
) -> TrainConfig:
    stage_config = _apply_continue_stage_overrides(train_config)
    profile = _shared_config_pack_profile(model_config_path, data_config_path, train_config_path)
    if not _uses_profile_runtime_layout(profile):
        return stage_config
    pretrain_dir = _profile_checkpoint_dir(profile, "pretrain")
    continue_dir = _prepare_manual_stage_checkpoint_dir(
        profile=profile,
        stage_name="continue",
        output_dir=stage_config.checkpoint.output_dir,
        resume_from=stage_config.checkpoint.resume_from,
    )
    if (
        stage_config.checkpoint.resume_from is None
        and stage_config.checkpoint.initialize_from is None
    ):
        staged_pretrain_checkpoint = _completed_checkpoint_dirs(pretrain_dir)
        if staged_pretrain_checkpoint:
            stage_config.checkpoint.initialize_from = str(staged_pretrain_checkpoint[-1])
        else:
            raise RuntimeError(
                f"No completed {profile} pretrain checkpoints were found in {pretrain_dir}. "
                "Manual continue now requires explicit lineage; set checkpoint.initialize_from to a specific checkpoint "
                "if you intentionally want to continue from a legacy or non-staged path."
            )
    stage_config.checkpoint.output_dir = continue_dir
    if stage_config.checkpoint.resume_from is not None:
        print(
            f"WebbGPT: {profile} manual continue will resume from {stage_config.checkpoint.resume_from} "
            f"and keep writing checkpoints under {continue_dir}.",
            file=sys.stderr,
            flush=True,
        )
    else:
        print(
            f"WebbGPT: {profile} manual continue will initialize from "
            f"{stage_config.checkpoint.initialize_from} and write new checkpoints under {continue_dir}.",
            file=sys.stderr,
            flush=True,
    )
    return stage_config


def _manual_profile_sft_initialize_checkpoint(profile: str, continue_dir: Path) -> str:
    try:
        return _latest_checkpoint_dir(str(continue_dir))
    except RuntimeError as exc:
        from train.checkpoint import load_stage_summary

        continue_summary = load_stage_summary(continue_dir)
        if bool((continue_summary or {}).get("skipped")):
            pretrain_dir = _profile_checkpoint_dir(profile, "pretrain")
            try:
                return _latest_checkpoint_dir(str(pretrain_dir))
            except RuntimeError as pretrain_exc:
                raise RuntimeError(
                    f"The latest {profile} continue stage at {continue_dir} was skipped, "
                    f"but no completed {profile} pretrain checkpoints were found in {pretrain_dir}. "
                    "Set checkpoint.initialize_from explicitly if you want SFT to start from a different checkpoint."
                ) from pretrain_exc
        raise RuntimeError(
            f"No completed {profile} continue checkpoints were found in {continue_dir}. "
            "Set checkpoint.initialize_from explicitly if you want SFT to start from a different checkpoint."
        ) from exc


def _prepare_manual_sft_train_config(
    model_config_path: str,
    data_config_path: str,
    train_config_path: str,
    train_config: TrainConfig,
) -> TrainConfig:
    profile = _shared_config_pack_profile(model_config_path, data_config_path, train_config_path)
    output_dir = train_config.checkpoint.output_dir
    initialize_from = train_config.checkpoint.initialize_from
    resume_from = train_config.checkpoint.resume_from

    if _uses_profile_runtime_layout(profile):
        continue_dir = _profile_checkpoint_dir(profile, "continue")
        output_dir = _prepare_manual_stage_checkpoint_dir(
            profile=profile,
            stage_name="sft",
            output_dir=output_dir,
            resume_from=resume_from,
        )
        if resume_from is None:
            if initialize_from is None:
                initialize_from = _manual_profile_sft_initialize_checkpoint(profile, continue_dir)
            print(
                f"WebbGPT: {profile} manual SFT will initialize from {initialize_from} "
                f"and write new checkpoints under {output_dir}.",
                file=sys.stderr,
                flush=True,
            )
        else:
            print(
                f"WebbGPT: {profile} manual SFT will resume from {resume_from} "
                f"and keep writing checkpoints under {output_dir}.",
                file=sys.stderr,
                flush=True,
            )

    return _stage_train_config(
        train_config,
        stage_name="sft",
        output_dir=output_dir,
        initialize_from=initialize_from,
        resume_from=resume_from,
        max_steps=_stage_step_count(train_config, "sft"),
    )


def _prepare_manual_dpo_train_config(
    model_config_path: str,
    data_config_path: str,
    train_config_path: str,
    train_config: TrainConfig,
    *,
    reference_checkpoint: str,
) -> TrainConfig:
    profile = _shared_config_pack_profile(model_config_path, data_config_path, train_config_path)
    output_dir = train_config.checkpoint.output_dir
    resume_from = train_config.checkpoint.resume_from

    if _uses_profile_runtime_layout(profile):
        output_dir = _prepare_manual_stage_checkpoint_dir(
            profile=profile,
            stage_name="dpo",
            output_dir=output_dir,
            resume_from=resume_from,
        )
        if resume_from is None and not Path(reference_checkpoint).exists():
            raise RuntimeError(
                f"The provided DPO reference checkpoint does not exist: {reference_checkpoint}. "
                "Point --reference-checkpoint at a concrete SFT checkpoint directory."
            )
        if resume_from is None:
            print(
                f"WebbGPT: {profile} manual DPO will initialize the policy from the provided "
                f"SFT checkpoint {reference_checkpoint} and write new checkpoints under {output_dir}.",
                file=sys.stderr,
                flush=True,
            )
        else:
            print(
                f"WebbGPT: {profile} manual DPO will resume from {resume_from} "
                f"and keep writing checkpoints under {output_dir}.",
                file=sys.stderr,
                flush=True,
            )

    return _stage_train_config(
        train_config,
        stage_name="dpo",
        output_dir=output_dir,
        initialize_from=train_config.checkpoint.initialize_from,
        resume_from=resume_from,
        max_steps=_stage_step_count(train_config, "dpo"),
    )


def _remote_scale_label(*, mvp: bool, full: bool) -> str:
    if mvp:
        return "mvp"
    return "full"


def _source_paths(source: DataSourceConfig) -> list[str]:
    if source.paths:
        return list(source.paths)
    if source.path:
        return [source.path]
    return []


def _source_identity(source: DataSourceConfig) -> tuple:
    if source.dataset_name:
        return (
            "hf",
            source.dataset_name,
            source.dataset_config_name or "",
            source.dataset_revision or "",
            source.split,
        )
    return ("local", tuple(sorted(_source_paths(source))), source.split)


def _record_window(source: DataSourceConfig) -> tuple[int, int | None]:
    start = max(source.skip_records, 0)
    stop = None if source.max_records is None else start + source.max_records
    return start, stop


def _ranges_overlap(left: tuple[int, int | None], right: tuple[int, int | None]) -> bool:
    left_start, left_stop = left
    right_start, right_stop = right
    left_end = float("inf") if left_stop is None else float(left_stop)
    right_end = float("inf") if right_stop is None else float(right_stop)
    return max(left_start, right_start) < min(left_end, right_end)


def _validate_distinct_validation_sources(data_config: DataConfig) -> None:
    for train_source in data_config.pretrain_sources:
        for validation_source in data_config.validation_sources:
            if _source_identity(train_source) != _source_identity(validation_source):
                continue
            if _ranges_overlap(_record_window(train_source), _record_window(validation_source)):
                raise RuntimeError(
                    "Serious-profile validation sources must not overlap with pretraining sources. "
                    f"Found overlap between {train_source.name!r} and {validation_source.name!r}."
                )


def _validate_serious_profile(
    profile_name: str,
    tokenizer_config: TokenizerConfig,
    model_config: ModelConfig,
    data_config: DataConfig,
    eval_config: EvalConfig,
    serve_config: ServeConfig,
) -> None:
    placeholder_paths = {"data/raw/tokenizer_corpus.txt"}
    placeholder_prefixes = ("data/local/",)
    serious_source_groups = {
        "continued_pretrain_sources": data_config.continued_pretrain_sources,
        "sft_sources": data_config.sft_sources,
        "preference_sources": data_config.preference_sources,
    }
    for group_name, sources in serious_source_groups.items():
        if len(sources) < 2:
            raise RuntimeError(
                f"{profile_name} requires at least two configured sources in `{group_name}` so the serious data pack is not a single-file placeholder."
            )

    for stage_name, sources in {
        "pretrain": data_config.pretrain_sources,
        "continue": data_config.continued_pretrain_sources,
        "sft": data_config.sft_sources,
        "preference": data_config.preference_sources,
        "validation": data_config.validation_sources,
    }.items():
        for source in sources:
            for path in _source_paths(source):
                if path in placeholder_paths or path.startswith(placeholder_prefixes):
                    raise RuntimeError(
                        f"{profile_name} cannot use debug placeholder data. "
                        f"Stage {stage_name!r} includes {path!r} from source {source.name!r}."
                    )

    _validate_distinct_validation_sources(data_config)

    benchmark_names = {Path(path).name for path in eval_config.benchmark_paths}
    required_benchmarks = {
        "chat_sanity.jsonl",
        "webb_course_present.responses",
        "webb_course_missing.responses",
        "webb_handbook_present.responses",
        "webb_handbook_missing.responses",
    }
    missing_benchmarks = sorted(required_benchmarks - benchmark_names)
    if missing_benchmarks:
        raise RuntimeError(
            f"{profile_name} evaluation must include chat sanity plus grounded Webb course and handbook present/missing benchmarks. "
            f"Missing: {', '.join(missing_benchmarks)}."
        )
    if eval_config.grounding is None:
        raise RuntimeError(f"{profile_name} evaluation must use Webb grounding.")
    if not eval_config.enforce_release_gates:
        raise RuntimeError(f"{profile_name} must enforce release gates during evaluation.")
    if "debug" in model_config.name.lower():
        raise RuntimeError(f"{profile_name} cannot use a debug-named model config.")
    if "debug" in serve_config.checkpoint_path.lower():
        raise RuntimeError(f"{profile_name} cannot default to a debug export path for serving.")
    if serve_config.grounding is None:
        raise RuntimeError(f"{profile_name} serving must use Webb grounding.")
    if data_config.tokenizer_path != f"{tokenizer_config.model_prefix}.model":
        raise RuntimeError(
            f"{profile_name} tokenizer_path must point at the tokenizer artifact produced by the configured tokenizer model_prefix."
        )


def _validate_remote_profile(
    tokenizer_config: TokenizerConfig,
    model_config: ModelConfig,
    data_config: DataConfig,
    eval_config: EvalConfig,
    serve_config: ServeConfig,
) -> None:
    _validate_serious_profile(
        "remote-3b",
        tokenizer_config,
        model_config,
        data_config,
        eval_config,
        serve_config,
    )


def _estimate_model_parameters(model_config: ModelConfig) -> int:
    hidden = model_config.hidden_size
    vocab = model_config.vocab_size
    kv_hidden = model_config.num_key_value_heads * model_config.head_dim
    attention_params = (hidden * hidden * 2) + (hidden * kv_hidden * 2)
    mlp_params = hidden * model_config.intermediate_size * 3
    norm_params = hidden * 2
    layer_params = attention_params + mlp_params + norm_params
    embedding_params = vocab * hidden
    final_norm_params = hidden
    lm_head_params = 0 if model_config.tie_word_embeddings else vocab * hidden
    return embedding_params + (layer_params * model_config.num_hidden_layers) + final_norm_params + lm_head_params


def _system_memory_bytes() -> int | None:
    page_size_names = ("SC_PAGE_SIZE", "SC_PAGESIZE")
    page_size = None
    for name in page_size_names:
        try:
            page_size = os.sysconf(name)
            break
        except (AttributeError, OSError, ValueError):
            continue
    try:
        phys_pages = os.sysconf("SC_PHYS_PAGES")
    except (AttributeError, OSError, ValueError):
        phys_pages = None
    if isinstance(page_size, int) and isinstance(phys_pages, int):
        return int(page_size) * int(phys_pages)
    return None


def _validate_profile_hardware_fit(profile: str, model_config: ModelConfig) -> None:
    total_memory = _system_memory_bytes()
    total_memory_gib = None if total_memory is None else total_memory / float(1024**3)
    system_name = platform.system()
    estimated_params = _estimate_model_parameters(model_config)

    if profile == "local-mvp":
        if system_name == "Darwin" and total_memory_gib is not None and total_memory_gib < 12:
            print(
                "WebbGPT: local-mvp is designed for smaller local machines, but this Mac reports less than 12 GiB of RAM. "
                "Expect very slow training or reduce the model further.",
                file=sys.stderr,
                flush=True,
            )
        return

    if profile == "remote-7b":
        if system_name != "Linux":
            raise RuntimeError(
                "remote-7b is intended for a powerful Linux multi-GPU machine. "
                "Use `webbgpt main --profile local-mvp` on this computer instead."
            )
        if total_memory_gib is not None and total_memory_gib < 256:
            raise RuntimeError(
                "remote-7b expects a powerful Linux host with at least 256 GiB of system RAM. "
                f"This machine reports about {total_memory_gib:.1f} GiB."
            )
        return

    if (
        profile == "remote-3b"
        and system_name == "Darwin"
        and total_memory_gib is not None
        and total_memory_gib < 64
        and estimated_params >= 1_000_000_000
    ):
        raise RuntimeError(
            "remote-3b is too large for this low-memory Mac and will likely be killed during training. "
            "Use `webbgpt main --profile local-mvp` on this computer instead."
        )


def _stage_step_count(train_config: TrainConfig, stage_name: str) -> int:
    if stage_name == "continue" and train_config.continued_max_steps is not None:
        return train_config.continued_max_steps
    if stage_name == "sft" and train_config.sft_max_steps is not None:
        return train_config.sft_max_steps
    if stage_name == "dpo" and train_config.dpo_max_steps is not None:
        return train_config.dpo_max_steps
    return train_config.max_steps


def _stage_train_config(
    train_config: TrainConfig,
    *,
    stage_name: str,
    output_dir: str,
    initialize_from: str | None = None,
    resume_from: str | None = None,
    max_steps: int | None = None,
    token_budget: int | None = None,
) -> TrainConfig:
    stage_config = TrainConfig.from_dict(train_config.to_dict())
    stage_config.run_name = f"{train_config.run_name}-{stage_name}"
    stage_config.checkpoint.output_dir = output_dir
    stage_config.checkpoint.initialize_from = initialize_from
    stage_config.checkpoint.resume_from = resume_from
    stage_config.max_steps = max_steps if max_steps is not None else train_config.max_steps
    stage_config.token_budget = token_budget
    if stage_name == "continue":
        stage_config = _apply_continue_stage_overrides(stage_config)
    if stage_name == "sft":
        stage_config = _apply_sft_stage_overrides(stage_config)
    if stage_name == "dpo":
        stage_config = _apply_dpo_stage_overrides(stage_config)
    if resume_from is not None:
        stage_config.max_steps = _checkpoint_step(resume_from) + stage_config.max_steps
    return stage_config


def _apply_remote_run_preset(
    data_config: DataConfig,
    train_config: TrainConfig,
    eval_config: EvalConfig,
    *,
    preset: str,
) -> tuple[DataConfig, TrainConfig, EvalConfig]:
    preset_data = DataConfig.from_dict(data_config.to_dict())
    preset_train = TrainConfig.from_dict(train_config.to_dict())
    preset_eval = EvalConfig.from_dict(eval_config.to_dict())

    if preset == "mvp":
        preset_data.pretraining_token_budget = 2_500_000_000
        preset_data.continued_pretraining_token_budget = 350_000_000

        preset_train.run_name = f"{preset_train.run_name}-mvp"
        preset_train.max_steps = 25_000
        preset_train.continued_max_steps = 8_000
        preset_train.sft_max_steps = 6_000
        preset_train.dpo_max_steps = 3_000
        preset_train.warmup_steps = min(preset_train.warmup_steps, 1_000)
        preset_train.eval_every_steps = min(preset_train.eval_every_steps, 250)
        preset_train.checkpoint.save_every_steps = min(
            preset_train.checkpoint.save_every_steps,
            250,
        )

        preset_eval.run_name = f"{preset_eval.run_name}-mvp"
        preset_eval.release_gates.assistant_pass_rate_min = 0.5
        preset_eval.release_gates.chat_sanity_pass_rate_min = 0.5
        preset_eval.release_gates.catalog_exactness_min = 0.8
        preset_eval.release_gates.catalog_citation_rate_min = 0.8
        preset_eval.release_gates.catalog_missing_abstention_min = 0.8
        return preset_data, preset_train, preset_eval

    if preset == "full":
        preset_train.run_name = f"{preset_train.run_name}-full"
        preset_eval.run_name = f"{preset_eval.run_name}-full"
        return preset_data, preset_train, preset_eval

    raise ValueError(f"Unknown remote run preset {preset!r}.")


def _prepared_manifest_source(name: str, manifest_path: str) -> DataSourceConfig:
    return DataSourceConfig(
        name=name,
        path=manifest_path,
        format="prepared",
        split="train",
        weight=1.0,
        quality_filter=False,
        deduplicate=False,
        pii_scrub=False,
    )


def _prepared_manifest_config(
    data_config: DataConfig,
    manifest_key: str,
) -> tuple[DataConfig, str]:
    stage_config = DataConfig.from_dict(data_config.to_dict())
    stage_name = manifest_key
    if manifest_key == "sft_validation":
        stage_config.sft_sources = list(stage_config.sft_validation_sources)
        stage_name = "sft"
    elif manifest_key == "preference_validation":
        stage_config.preference_sources = list(stage_config.preference_validation_sources)
        stage_name = "preference"
    return stage_config, stage_name


def _materialize_profile_prepared_manifests(
    profile: str,
    data_config: DataConfig,
    manifest_keys: list[str],
    *,
    force_rebuild: bool = False,
) -> dict[str, dict[str, object]]:
    from data.dataset import DatasetBuilder

    prepared_root = _profile_prepared_dir(profile)
    prepared_root.mkdir(parents=True, exist_ok=True)
    manifests: dict[str, dict[str, object]] = {}
    for manifest_key in manifest_keys:
        stage_config, stage_name = _prepared_manifest_config(data_config, manifest_key)
        manifest_path = prepared_root / f"{manifest_key}.json"
        manifest = DatasetBuilder(stage_config).prepare_stage(
            stage_name,
            str(manifest_path),
            force_rebuild=force_rebuild,
        )
        manifests[manifest_key] = {"path": str(manifest_path), "manifest": manifest}
    return manifests


def _prepared_data_config(data_config: DataConfig, manifest_paths: dict[str, str]) -> DataConfig:
    prepared = DataConfig.from_dict(data_config.to_dict())
    if "pretrain" in manifest_paths:
        prepared.pretrain_sources = [_prepared_manifest_source("prepared_pretrain", manifest_paths["pretrain"])]
    if "continue" in manifest_paths:
        prepared.continued_pretrain_sources = [
            _prepared_manifest_source("prepared_continue", manifest_paths["continue"])
        ]
    if "sft" in manifest_paths:
        prepared.sft_sources = [_prepared_manifest_source("prepared_sft", manifest_paths["sft"])]
    if "sft_validation" in manifest_paths:
        prepared.sft_validation_sources = [
            _prepared_manifest_source("prepared_sft_validation", manifest_paths["sft_validation"])
        ]
    if "preference" in manifest_paths:
        prepared.preference_sources = [
            _prepared_manifest_source("prepared_preference", manifest_paths["preference"])
        ]
    if "preference_validation" in manifest_paths:
        prepared.preference_validation_sources = [
            _prepared_manifest_source(
                "prepared_preference_validation",
                manifest_paths["preference_validation"],
            )
        ]
    if "validation" in manifest_paths:
        prepared.validation_sources = [
            _prepared_manifest_source("prepared_validation", manifest_paths["validation"])
        ]
    return prepared


def _manual_profile_prepared_manifest_keys(
    stage_name: str,
    data_config: DataConfig,
    train_config: TrainConfig,
) -> list[str]:
    manifest_keys: list[str] = []
    if stage_name == "pretrain":
        manifest_keys.append("pretrain")
        if data_config.validation_sources:
            manifest_keys.append("validation")
        return manifest_keys
    if stage_name == "continue":
        manifest_keys.append("continue")
        if data_config.validation_sources:
            manifest_keys.append("validation")
        return manifest_keys
    if stage_name == "sft":
        manifest_keys.append("sft")
        if data_config.sft_validation_sources:
            manifest_keys.append("sft_validation")
        return manifest_keys
    if stage_name == "dpo":
        manifest_keys.append("preference")
        if data_config.preference_validation_sources:
            manifest_keys.append("preference_validation")
        if train_config.dpo_enable_lm_health_eval and data_config.validation_sources:
            manifest_keys.append("validation")
        return manifest_keys
    raise ValueError(f"Unsupported manual stage {stage_name!r}.")


def _prepare_manual_stage_data_config(
    model_config_path: str,
    data_config_path: str,
    train_config_path: str,
    data_config: DataConfig,
    train_config: TrainConfig,
    *,
    stage_name: str,
    force_rebuild: bool = False,
) -> DataConfig:
    profile = _shared_config_pack_profile(model_config_path, data_config_path, train_config_path)
    if not _uses_profile_runtime_layout(profile):
        return data_config
    manifest_keys = _manual_profile_prepared_manifest_keys(stage_name, data_config, train_config)
    prepared_manifests = _materialize_profile_prepared_manifests(
        profile,
        data_config,
        manifest_keys,
        force_rebuild=force_rebuild,
    )
    manifest_paths = {key: str(payload["path"]) for key, payload in prepared_manifests.items()}
    print(
        f"WebbGPT: using profile-scoped prepared data under {_profile_prepared_dir(profile)} "
        f"for the {profile} manual {stage_name} run.",
        file=sys.stderr,
        flush=True,
    )
    return _prepared_data_config(data_config, manifest_paths)


def _run_main_pipeline(
    output_dir: str,
    profile: str,
    no_serve: bool,
    *,
    remote_preset: str = "full",
    force_rebuild: bool = False,
) -> int:
    base = Path(output_dir)
    config_paths = _require_profile_configs(base, profile)
    tokenizer_config = load_config(config_paths["tokenizer"], TokenizerConfig)
    tokenizer_corpus_config = load_config(config_paths["tokenizer_corpus"], TokenizerCorpusConfig)
    model_config = load_config(config_paths["model"], ModelConfig)
    data_config = load_config(config_paths["data"], DataConfig)
    train_config = load_config(config_paths["train"], TrainConfig)
    eval_config = load_config(config_paths["eval"], EvalConfig)
    serve_config = load_config(config_paths["serve"], ServeConfig)
    if profile == "remote-3b":
        data_config, train_config, eval_config = _apply_remote_run_preset(
            data_config,
            train_config,
            eval_config,
            preset=remote_preset,
        )
        _validate_remote_profile(tokenizer_config, model_config, data_config, eval_config, serve_config)
        print(
            "WebbGPT: using remote-3b "
            f"{remote_preset.upper()} preset "
            f"(pretrain={data_config.pretraining_token_budget:,} tokens, "
            f"continue={data_config.continued_pretraining_token_budget:,} tokens, "
            f"steps={train_config.max_steps:,}/{train_config.continued_max_steps:,}/"
            f"{train_config.sft_max_steps:,}/{train_config.dpo_max_steps:,}).",
            file=sys.stderr,
            flush=True,
        )
    elif profile == "local-mvp":
        if data_config.tokenizer_path != f"{tokenizer_config.model_prefix}.model":
            raise RuntimeError(
                "local-mvp tokenizer_path must point at the tokenizer artifact produced by the configured tokenizer model_prefix."
            )
        print(
            "WebbGPT: using local-mvp profile "
            f"(pretrain={data_config.pretraining_token_budget:,} tokens, "
            f"continue={data_config.continued_pretraining_token_budget:,} tokens, "
            f"steps={train_config.max_steps:,}/{train_config.continued_max_steps:,}/"
            f"{train_config.sft_max_steps:,}/{train_config.dpo_max_steps:,}).",
            file=sys.stderr,
            flush=True,
        )
    elif profile == "remote-7b":
        _validate_serious_profile(
            "remote-7b",
            tokenizer_config,
            model_config,
            data_config,
            eval_config,
            serve_config,
        )
        print(
            "WebbGPT: using remote-7b full profile "
            f"(pretrain={data_config.pretraining_token_budget:,} tokens, "
            f"continue={data_config.continued_pretraining_token_budget:,} tokens, "
            f"steps={train_config.max_steps:,}/{train_config.continued_max_steps:,}/"
            f"{train_config.sft_max_steps:,}/{train_config.dpo_max_steps:,}).",
            file=sys.stderr,
            flush=True,
        )

    _validate_profile_hardware_fit(profile, model_config)

    from eval.runner import run_evaluation
    from export.hf import export_hf_checkpoint
    from grounding.ingest import webb_sync
    from posttrain.dpo import run_dpo_job
    from posttrain.sft import run_sft_job
    from serve.app import run_server
    from tokenizer.spm import train_tokenizer
    from train.entrypoints import run_continued_pretraining, run_pretraining

    runtime_root = _profile_artifact_root(profile)
    checkpoints_root = runtime_root / "checkpoints"
    eval_root = _profile_eval_dir(profile)

    if _can_reuse_tokenizer_corpus(tokenizer_corpus_config):
        print(
            f"WebbGPT: reusing tokenizer corpus for profile {profile} from {tokenizer_corpus_config.output_path}.",
            file=sys.stderr,
            flush=True,
        )
    else:
        print(f"WebbGPT: building tokenizer corpus for profile {profile}.", file=sys.stderr, flush=True)
        _run_tokenizer_corpus_subprocess(tokenizer_corpus_config)

    if _can_reuse_tokenizer_model(tokenizer_config):
        print(
            f"WebbGPT: reusing tokenizer artifacts for profile {profile} from {tokenizer_config.model_prefix}.model.",
            file=sys.stderr,
            flush=True,
        )
    else:
        print("WebbGPT: training tokenizer.", file=sys.stderr, flush=True)
        train_tokenizer([tokenizer_corpus_config.output_path], tokenizer_config)
    tokenizer_artifact = Path(f"{tokenizer_config.model_prefix}.model")
    if not tokenizer_artifact.exists():
        raise RuntimeError(
            f"Tokenizer training finished without producing the expected artifact at {tokenizer_artifact}."
        )

    print("WebbGPT: materializing prepared datasets.", file=sys.stderr, flush=True)
    manifest_keys = ["validation", "pretrain", "continue", "sft", "preference"]
    if data_config.sft_validation_sources:
        manifest_keys.append("sft_validation")
    if data_config.preference_validation_sources:
        manifest_keys.append("preference_validation")
    prepared_manifests = _materialize_profile_prepared_manifests(
        profile,
        data_config,
        manifest_keys,
        force_rebuild=force_rebuild,
    )
    manifest_paths = {key: str(payload["path"]) for key, payload in prepared_manifests.items()}
    for stage, payload in prepared_manifests.items():
        print(
            json.dumps(
                {
                    "stage": stage,
                    "prepared_manifest": payload["path"],
                    "kind": payload["manifest"]["kind"],
                }
            ),
            flush=True,
        )

    prepared_data_config = _prepared_data_config(data_config, manifest_paths)

    pretrain_config = _stage_train_config(
        train_config,
        stage_name="pretrain",
        output_dir=str(checkpoints_root / "pretrain"),
        max_steps=_stage_step_count(train_config, "pretrain"),
        token_budget=data_config.pretraining_token_budget,
    )
    print("WebbGPT: running pretraining stage.", file=sys.stderr, flush=True)
    run_pretraining(model_config, prepared_data_config, pretrain_config)
    pretrain_checkpoint = _latest_checkpoint_dir(pretrain_config.checkpoint.output_dir)

    continued_config = _stage_train_config(
        train_config,
        stage_name="continue",
        output_dir=str(checkpoints_root / "continue"),
        initialize_from=pretrain_checkpoint,
        max_steps=_stage_step_count(train_config, "continue"),
        token_budget=data_config.continued_pretraining_token_budget,
    )
    print("WebbGPT: running continued pretraining stage.", file=sys.stderr, flush=True)
    continue_summary = run_continued_pretraining(model_config, prepared_data_config, continued_config)
    if bool(continue_summary.get("skipped")):
        continued_checkpoint = pretrain_checkpoint
    else:
        continued_checkpoint = _latest_checkpoint_dir(continued_config.checkpoint.output_dir)

    sft_config = _stage_train_config(
        train_config,
        stage_name="sft",
        output_dir=str(checkpoints_root / "sft"),
        initialize_from=continued_checkpoint,
        max_steps=_stage_step_count(train_config, "sft"),
    )
    print("WebbGPT: running supervised fine-tuning stage.", file=sys.stderr, flush=True)
    run_sft_job(model_config, prepared_data_config, sft_config)
    sft_checkpoint = _latest_checkpoint_dir(sft_config.checkpoint.output_dir)

    dpo_config = _stage_train_config(
        train_config,
        stage_name="dpo",
        output_dir=str(checkpoints_root / "dpo"),
        max_steps=_stage_step_count(train_config, "dpo"),
    )
    print("WebbGPT: running DPO stage.", file=sys.stderr, flush=True)
    run_dpo_job(
        model_config,
        prepared_data_config,
        dpo_config,
        reference_checkpoint=sft_checkpoint,
    )
    dpo_checkpoint = _latest_checkpoint_dir(dpo_config.checkpoint.output_dir)

    print("WebbGPT: running evaluation.", file=sys.stderr, flush=True)
    eval_result = run_evaluation(
        model_config,
        prepared_data_config,
        eval_config,
        checkpoint_path=dpo_checkpoint,
    )
    eval_root.mkdir(parents=True, exist_ok=True)
    (eval_root / "result.json").write_text(json.dumps(eval_result, indent=2) + "\n")
    print(dump_rounded_json(eval_result, indent=2), flush=True)
    dpo_trust = _artifact_trust(dpo_checkpoint)
    if str(dpo_trust.get("artifact_status", "promotable")) != "promotable":
        message = (
            "WebbGPT: skipping export/serve because the final DPO checkpoint is not promotable "
            f"(artifact_status={dpo_trust.get('artifact_status')}, "
            f"blockers={', '.join(dpo_trust.get('promotion_blockers', [])) or 'none'})."
        )
        if no_serve:
            print(message, file=sys.stderr, flush=True)
            return 0
        raise RuntimeError(message)

    export_dir = _profile_export_dir(profile)
    print("WebbGPT: exporting Hugging Face artifacts.", file=sys.stderr, flush=True)
    export_hf_checkpoint(
        model_config,
        checkpoint_path=dpo_checkpoint,
        output_dir=str(export_dir),
    )
    release_gates = eval_result.get("release_gates", {})
    if eval_config.enforce_release_gates and not bool(release_gates.get("passed", False)):
        raise RuntimeError(
            "Release gates failed for the selected profile. "
            f"See {eval_root / 'result.json'} for the failing checks."
        )

    if no_serve:
        print("WebbGPT: skipping Webb sync and serving because --no-serve was set.", file=sys.stderr, flush=True)
        return 0

    final_serve_config = ServeConfig.from_dict(serve_config.to_dict())
    final_serve_config.checkpoint_path = str(export_dir)
    if final_serve_config.enable_grounding:
        grounding = final_serve_config.grounding or GroundingConfig(dsn=_profile_grounding_dsn(profile))
        print("WebbGPT: syncing Webb grounding data.", file=sys.stderr, flush=True)
        sync_result = webb_sync(
            grounding.dsn,
            seed_url_pack=grounding.seed_url_pack,
            source_policy_path=grounding.source_policy_path,
            handbook_url=grounding.handbook_url,
            allow_ocr_fallback=grounding.allow_ocr_fallback,
            label=f"{profile}-main-sync",
            families=grounding.sync_families or None,
        )
        snapshot_id = sync_result.get("snapshot_id")
        if snapshot_id:
            grounding.snapshot_id = str(snapshot_id)
        final_serve_config.grounding = grounding
    print("WebbGPT: starting server.", file=sys.stderr, flush=True)
    run_server(final_serve_config)
    return 0


def _can_import_torch_for_tests() -> bool:
    result = subprocess.run(
        [sys.executable, "-c", "import torch"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.returncode == 0


def _test_environment() -> dict[str, str]:
    env = os.environ.copy()
    hf_home = Path(env.get("HF_HOME", ".hf-test-cache")).resolve()
    hf_datasets_cache = Path(env.get("HF_DATASETS_CACHE", str(hf_home / "datasets"))).resolve()
    hf_home.mkdir(parents=True, exist_ok=True)
    hf_datasets_cache.mkdir(parents=True, exist_ok=True)
    env["HF_HOME"] = str(hf_home)
    env["HF_DATASETS_CACHE"] = str(hf_datasets_cache)
    return env


def main() -> int:
    args = _parse_args()

    if args.command == "tokenize":
        from tokenizer.spm import train_tokenizer

        config = load_config(args.config, TokenizerConfig)
        train_tokenizer(args.input, config)
        return 0

    if args.command == "build-tokenizer-corpus":
        config = load_config(args.config, TokenizerCorpusConfig)
        _run_tokenizer_corpus_subprocess(config)
        return 0

    if args.command == "prepare-data":
        from data.dataset import DatasetBuilder

        config = load_config(args.config, DataConfig)
        builder = DatasetBuilder(config)
        manifest = builder.prepare_stage(args.stage, args.output, force_rebuild=args.force_rebuild)
        print(dump_rounded_json(manifest, indent=2), flush=True)
        return 0

    if args.command == "audit-data":
        from data.dataset import DatasetBuilder

        config = load_config(args.config, DataConfig)
        builder = DatasetBuilder(config)
        if args.stage == "continue":
            payload = builder.assess_continue_readiness()
        else:
            payload = builder.audit_lm_stage(args.stage)
        print(dump_rounded_json(payload, indent=2), flush=True)
        return 0

    if args.command == "train-pretrain":
        from train.entrypoints import run_pretraining

        print("WebbGPT: starting pretraining.", file=sys.stderr, flush=True)
        model_config = load_config(args.model_config, ModelConfig)
        data_config = load_config(args.data_config, DataConfig)
        base_train_config = load_config(args.train_config, TrainConfig)
        data_config = _prepare_manual_stage_data_config(
            args.model_config,
            args.data_config,
            args.train_config,
            data_config,
            base_train_config,
            stage_name="pretrain",
            force_rebuild=args.force_rebuild,
        )
        train_config = _prepare_manual_pretrain_train_config(
            args.model_config,
            args.data_config,
            args.train_config,
            base_train_config,
        )
        run_pretraining(
            model_config,
            data_config,
            train_config,
        )
        return 0

    if args.command == "train-continue":
        from train.entrypoints import run_continued_pretraining

        print("WebbGPT: starting continued pretraining.", file=sys.stderr, flush=True)
        model_config = load_config(args.model_config, ModelConfig)
        data_config = load_config(args.data_config, DataConfig)
        base_train_config = load_config(args.train_config, TrainConfig)
        data_config = _prepare_manual_stage_data_config(
            args.model_config,
            args.data_config,
            args.train_config,
            data_config,
            base_train_config,
            stage_name="continue",
            force_rebuild=args.force_rebuild,
        )
        train_config = _prepare_manual_continue_train_config(
            args.model_config,
            args.data_config,
            args.train_config,
            base_train_config,
        )
        run_continued_pretraining(
            model_config,
            data_config,
            train_config,
        )
        return 0

    if args.command == "train-sft":
        from posttrain.sft import run_sft_job

        print("WebbGPT: starting supervised fine-tuning.", file=sys.stderr, flush=True)
        model_config = load_config(args.model_config, ModelConfig)
        data_config = load_config(args.data_config, DataConfig)
        base_train_config = load_config(args.train_config, TrainConfig)
        data_config = _prepare_manual_stage_data_config(
            args.model_config,
            args.data_config,
            args.train_config,
            data_config,
            base_train_config,
            stage_name="sft",
            force_rebuild=args.force_rebuild,
        )
        train_config = _prepare_manual_sft_train_config(
            args.model_config,
            args.data_config,
            args.train_config,
            base_train_config,
        )
        run_sft_job(
            model_config,
            data_config,
            train_config,
        )
        return 0

    if args.command == "train-dpo":
        from posttrain.dpo import run_dpo_job

        print("WebbGPT: starting DPO alignment.", file=sys.stderr, flush=True)
        model_config = load_config(args.model_config, ModelConfig)
        data_config = load_config(args.data_config, DataConfig)
        base_train_config = load_config(args.train_config, TrainConfig)
        data_config = _prepare_manual_stage_data_config(
            args.model_config,
            args.data_config,
            args.train_config,
            data_config,
            base_train_config,
            stage_name="dpo",
            force_rebuild=args.force_rebuild,
        )
        train_config = _prepare_manual_dpo_train_config(
            args.model_config,
            args.data_config,
            args.train_config,
            base_train_config,
            reference_checkpoint=args.reference_checkpoint,
        )
        run_dpo_job(
            model_config,
            data_config,
            train_config,
            reference_checkpoint=args.reference_checkpoint,
        )
        return 0

    if args.command == "eval":
        from eval.runner import run_evaluation

        print("WebbGPT: starting evaluation.", file=sys.stderr, flush=True)
        _require_trusted_artifact(
            args.checkpoint,
            action="run evaluation",
            force_untrusted=args.force_untrusted,
        )
        result = run_evaluation(
            load_config(args.model_config, ModelConfig),
            load_config(args.data_config, DataConfig),
            load_config(args.eval_config, EvalConfig),
            checkpoint_path=args.checkpoint,
        )
        print(dump_rounded_json(result, indent=2), flush=True)
        return 0

    if args.command == "ingest-webb-site":
        from grounding.ingest import ingest_webb_site

        result = ingest_webb_site(
            args.dsn,
            args.seed_url_pack,
            offline_seed_url_pack=args.offline_seed_url_pack,
            source_policy_path=args.source_policy_path,
            snapshot_id=args.snapshot_id,
            label=args.label,
            families=args.families,
        )
        print(dump_rounded_json(result, indent=2), flush=True)
        return 0

    if args.command == "ingest-webb-handbook":
        from grounding.ingest import ingest_webb_handbook

        result = ingest_webb_handbook(
            args.dsn,
            args.handbook_url,
            snapshot_id=args.snapshot_id,
            label=args.label,
            allow_ocr_fallback=args.allow_ocr_fallback,
        )
        print(dump_rounded_json(result, indent=2), flush=True)
        return 0

    if args.command == "webb-sync":
        from grounding.ingest import webb_sync

        result = webb_sync(
            args.dsn,
            seed_url_pack=args.seed_url_pack,
            offline_seed_url_pack=args.offline_seed_url_pack,
            source_policy_path=args.source_policy_path,
            handbook_url=args.handbook_url,
            allow_ocr_fallback=args.allow_ocr_fallback,
            label=args.label,
            families=args.families,
        )
        print(dump_rounded_json(result, indent=2), flush=True)
        return 0

    if args.command == "diff-webb-snapshot":
        from grounding.ingest import diff_webb_snapshot

        result = diff_webb_snapshot(args.dsn, args.from_snapshot, args.to_snapshot)
        print(dump_rounded_json(result, indent=2), flush=True)
        return 0

    if args.command == "serve":
        from serve.app import run_server

        serve_config = load_config(args.serve_config, ServeConfig)
        _require_trusted_artifact(
            serve_config.checkpoint_path,
            action="serve",
            force_untrusted=args.force_untrusted,
        )
        if args.sync_on_start:
            serve_config = ServeConfig.from_dict(serve_config.to_dict())
            grounding = serve_config.grounding or GroundingConfig()
            grounding.sync_on_start = True
            serve_config.grounding = grounding
        run_server(serve_config)
        return 0

    if args.command == "export-hf":
        from export.hf import export_hf_checkpoint

        _require_trusted_artifact(
            args.checkpoint,
            action="export",
            force_untrusted=args.force_untrusted,
        )
        export_hf_checkpoint(
            load_config(args.model_config, ModelConfig),
            checkpoint_path=args.checkpoint,
            output_dir=args.output,
        )
        return 0

    if args.command == "init-config":
        _write_default_configs(args.output_dir)
        return 0

    if args.command == "test":
        pytest_args = list(args.pytest_args)
        if not args.all and not pytest_args and not _can_import_torch_for_tests():
            print(
                "WebbGPT: torch cannot be imported cleanly here; running the safe non-torch test subset. "
                "Use `webbgpt test --all` to force the full suite.",
                file=sys.stderr,
                flush=True,
            )
            pytest_args = SAFE_DEFAULT_TEST_PATHS
        print("WebbGPT: running test suite.", file=sys.stderr, flush=True)
        result = subprocess.run(
            [sys.executable, "-m", "pytest", *pytest_args],
            env=_test_environment(),
            check=False,
        )
        return int(result.returncode)

    if args.command == "main":
        if args.profile != "remote-3b" and (args.mvp or args.full):
            raise RuntimeError("--mvp and --full are only supported with --profile remote-3b.")
        return _run_main_pipeline(
            args.output_dir,
            args.profile,
            args.no_serve,
            remote_preset=_remote_scale_label(mvp=args.mvp, full=args.full),
            force_rebuild=args.force_rebuild,
        )

    raise ValueError(f"Unhandled command {args.command!r}")


if __name__ == "__main__":
    raise SystemExit(main())
