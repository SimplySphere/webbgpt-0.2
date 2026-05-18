from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from config import DataConfig, load_config  # noqa: E402
from data.dataset import DatasetBuilder  # noqa: E402


def _limited_config(config: DataConfig, *, max_records: int, token_budget: int | None) -> DataConfig:
    limited = DataConfig.from_dict(config.to_dict())
    for source in limited.pretrain_sources:
        source.max_records = (
            max_records
            if source.max_records is None
            else min(int(source.max_records), max_records)
        )
    if token_budget is not None:
        limited.pretraining_token_budget = int(token_budget)
    return limited


def _run_prepare(config: DataConfig, *, output_path: Path, workers: int) -> tuple[dict[str, Any], float]:
    run_config = DataConfig.from_dict(config.to_dict())
    run_config.num_workers = int(workers)
    run_config.preprocessing_num_workers = None
    run_config.tokenizer_num_workers = int(workers)
    started = time.perf_counter()
    manifest = DatasetBuilder(run_config).prepare_stage(
        "pretrain",
        str(output_path),
        force_rebuild=True,
    )
    return manifest, time.perf_counter() - started


def _manifest_summary(manifest: dict[str, Any]) -> dict[str, Any]:
    diagnostics = manifest.get("diagnostics", {})
    return {
        "input_fingerprint": manifest.get("input_fingerprint"),
        "num_tokens": int(manifest.get("num_tokens", 0)),
        "num_sequences": int(manifest.get("num_sequences", 0)),
        "source_token_shares": {
            str(row.get("source")): row.get("token_share")
            for row in diagnostics.get("per_source", [])
            if isinstance(row, dict)
        },
        "rejection_counts": {
            str(row.get("source")): dict(row.get("dropped_reasons", {}))
            for row in diagnostics.get("per_source", [])
            if isinstance(row, dict)
        },
        "quality_warnings": diagnostics.get("quality_warnings", []),
        "domain_realization_gate": diagnostics.get("domain_realization_gate"),
        "corpus_quality_gate": diagnostics.get("corpus_quality_gate"),
        "broad_source_quality_gate": diagnostics.get("broad_source_quality_gate"),
    }


def _read_metadata_lines(manifest: dict[str, Any]) -> list[str]:
    rows: list[str] = []
    for shard in manifest.get("shards", []):
        metadata_path = shard.get("metadata_path")
        if metadata_path:
            rows.extend(Path(str(metadata_path)).read_text(encoding="utf-8").splitlines())
    return rows


def _same_shards(left: dict[str, Any], right: dict[str, Any]) -> bool:
    import numpy as np

    left_shards = list(left.get("shards", []))
    right_shards = list(right.get("shards", []))
    if len(left_shards) != len(right_shards):
        return False
    for left_shard, right_shard in zip(left_shards, right_shards, strict=False):
        if int(left_shard.get("rows", 0)) != int(right_shard.get("rows", 0)):
            return False
        if not np.array_equal(np.load(str(left_shard["path"])), np.load(str(right_shard["path"]))):
            return False
    return _read_metadata_lines(left) == _read_metadata_lines(right)


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark serial vs parallel pretrain preparation on a bounded subset.")
    parser.add_argument("--config", default="sample-configs/data-local-mvp.json")
    parser.add_argument("--output-dir", default="artifacts/benchmarks/prepare_parallel")
    parser.add_argument("--max-records", type=int, default=1000)
    parser.add_argument("--token-budget", type=int)
    parser.add_argument("--serial-workers", type=int, default=1)
    parser.add_argument("--parallel-workers", type=int, default=4)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    base_config = load_config(args.config, DataConfig)
    limited = _limited_config(
        base_config,
        max_records=int(args.max_records),
        token_budget=args.token_budget,
    )

    serial_manifest, serial_seconds = _run_prepare(
        limited,
        output_path=output_dir / "pretrain-serial.json",
        workers=int(args.serial_workers),
    )
    parallel_manifest, parallel_seconds = _run_prepare(
        limited,
        output_path=output_dir / "pretrain-parallel.json",
        workers=int(args.parallel_workers),
    )

    summary_equal = _manifest_summary(serial_manifest) == _manifest_summary(parallel_manifest)
    shards_equal = _same_shards(serial_manifest, parallel_manifest)
    payload = {
        "config": str(args.config),
        "max_records_per_source": int(args.max_records),
        "serial_workers": int(args.serial_workers),
        "parallel_workers": int(args.parallel_workers),
        "serial_seconds": round(serial_seconds, 3),
        "parallel_seconds": round(parallel_seconds, 3),
        "speedup": round(serial_seconds / parallel_seconds, 3) if parallel_seconds > 0 else None,
        "manifest_summary_equal": summary_equal,
        "shards_and_metadata_equal": shards_equal,
        "serial": _manifest_summary(serial_manifest),
        "parallel": _manifest_summary(parallel_manifest),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if summary_equal and shards_equal else 1


if __name__ == "__main__":
    raise SystemExit(main())
