from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


DENSITY_KEYS = (
    "medical_body_density",
    "product_commercial_density",
    "navigation_text_density",
    "dictionary_fragment_density",
    "page_boilerplate_density",
    "malformed_fragment_density",
    "generic_article_formula_density",
)


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _fmt_float(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def _gate_line(name: str, gate: Any) -> str:
    if not isinstance(gate, dict):
        return f"{name}: not present"
    status = "pass" if gate.get("passed") is True else "not-pass"
    severity = gate.get("severity", "unknown")
    failures = ", ".join(gate.get("failures") or [])
    suffix = f"; failures={failures}" if failures else ""
    return f"{name}: {status}; severity={severity}; mode={gate.get('mode')}{suffix}"


def _per_source(payload: dict[str, Any]) -> list[dict[str, Any]]:
    reports = payload.get("diagnostics", {}).get("per_source", [])
    return reports if isinstance(reports, list) else []


def _source_tokens(per_source: list[dict[str, Any]]) -> int:
    return sum(int(row.get("kept_tokens", 0)) for row in per_source)


def _source_documents(per_source: list[dict[str, Any]]) -> int:
    return sum(int(row.get("kept_documents", 0)) for row in per_source)


def _density_summary(per_source: list[dict[str, Any]]) -> dict[str, float]:
    token_count = sum(int(row.get("quality_diagnostic_token_count", 0)) for row in per_source)
    if token_count <= 0:
        return {key: 0.0 for key in DENSITY_KEYS}
    summary: dict[str, float] = {}
    for key in DENSITY_KEYS:
        weighted = sum(
            float(row.get(key, 0.0)) * int(row.get("quality_diagnostic_token_count", 0))
            for row in per_source
        )
        summary[key] = round(weighted / token_count, 6)
    return summary


def _top_reject_reasons(per_source: list[dict[str, Any]]) -> Counter[str]:
    counter: Counter[str] = Counter()
    for row in per_source:
        counter.update({str(key): int(value) for key, value in dict(row.get("dropped_reasons", {})).items()})
    return counter


def _manifest_summary(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    diagnostics = payload.get("diagnostics", {})
    per_source = _per_source(payload)
    total_tokens = int(payload.get("num_tokens") or _source_tokens(per_source))
    total_docs = _source_documents(per_source)
    density = _density_summary(per_source)
    return {
        "path": str(path),
        "tokens": total_tokens,
        "sequences": int(payload.get("num_sequences", 0)),
        "documents": total_docs,
        "max_source_share": float(diagnostics.get("max_single_source_token_share", 0.0)),
        "max_repeat_rate": float(diagnostics.get("max_repeat_rate", 0.0)),
        "max_near_duplicate_ratio": max(
            (float(row.get("near_duplicate_ratio", 0.0)) for row in per_source),
            default=0.0,
        ),
        "largest_near_duplicate_cluster_size": int(
            diagnostics.get("largest_near_duplicate_cluster_size", 0)
        ),
        "broad_source_gate_passed": (
            diagnostics.get("broad_source_quality_gate", {}).get("passed")
            if isinstance(diagnostics.get("broad_source_quality_gate"), dict)
            else None
        ),
        "corpus_gate_passed": (
            diagnostics.get("corpus_quality_gate", {}).get("passed")
            if isinstance(diagnostics.get("corpus_quality_gate"), dict)
            else None
        ),
        **density,
    }


def _print_manifest(path: Path, payload: dict[str, Any]) -> None:
    diagnostics = payload.get("diagnostics", {})
    per_source = _per_source(payload)
    total_tokens = int(payload.get("num_tokens") or _source_tokens(per_source))
    total_docs = _source_documents(per_source)

    print(f"manifest: {path}")
    print(f"stage: {payload.get('stage')}")
    print(f"kind: {payload.get('kind')}")
    print(f"num_tokens: {total_tokens}")
    print(f"num_sequences: {payload.get('num_sequences')}")
    print(f"total_documents: {total_docs}")
    print()
    print(_gate_line("domain_realization_gate", diagnostics.get("domain_realization_gate")))
    print(_gate_line("corpus_quality_gate", diagnostics.get("corpus_quality_gate")))
    print(_gate_line("broad_source_quality_gate", diagnostics.get("broad_source_quality_gate")))
    print(f"quality_warnings: {', '.join(diagnostics.get('quality_warnings') or []) or 'none'}")
    print()

    family_tokens: Counter[str] = Counter()
    for row in per_source:
        family_tokens[str(row.get("family", "unknown"))] += int(row.get("kept_tokens", 0))

    print("family token share:")
    if not family_tokens:
        print("  (no source diagnostics)")
    for family, tokens in sorted(family_tokens.items()):
        share = tokens / max(total_tokens, 1)
        print(f"  {family}: {share:.6f} ({tokens} tokens)")
    print()

    print("source token share:")
    if not per_source:
        print("  (no source diagnostics)")
    for row in sorted(per_source, key=lambda item: float(item.get("token_share", 0.0)), reverse=True):
        print(
            "  "
            f"{row.get('source')}: share={_fmt_float(row.get('token_share'))}, "
            f"target={_fmt_float(row.get('target_share'))}, "
            f"effective_target={_fmt_float(row.get('effective_target_share'))}, "
            f"family={row.get('family')}, docs={row.get('kept_documents')}, "
            f"unique_docs={int(row.get('kept_documents', 0)) - int(row.get('repeated_documents', 0))}, "
            f"repeated_docs={row.get('repeated_documents', 0)}, "
            f"repeat_rate={_fmt_float(row.get('repeat_rate', 0.0))}, "
            f"restarts={row.get('restart_count', 0)}, "
            f"tokens={row.get('kept_tokens')}"
        )
    print()

    print("quality artifact densities:")
    density = _density_summary(per_source)
    for key in DENSITY_KEYS:
        print(f"  {key}: {density[key]:.6f}")
    print()

    print("document shape:")
    shape = diagnostics.get("document_shape", {})
    if isinstance(shape, dict) and shape:
        for key in (
            "avg_document_words",
            "avg_document_tokens",
            "avg_sentences_per_document",
            "avg_paragraphs_per_document",
            "avg_sentence_words",
        ):
            print(f"  {key}: {shape.get(key)}")
    else:
        print("  (not present)")
    print()

    print("top rejection reasons:")
    reject_reasons = _top_reject_reasons(per_source)
    if not reject_reasons:
        print("  none")
    for reason, count in reject_reasons.most_common(12):
        print(f"  {reason}: {count}")
    print()

    print("top repeated n-grams:")
    for label, key in (
        ("8gram", "top_repeated_8grams"),
        ("12gram", "top_repeated_12grams"),
        ("20gram", "top_repeated_20grams"),
    ):
        rows = diagnostics.get(key) or []
        print(f"  {label}:")
        if not rows:
            print("    none")
        for row in rows[:5]:
            print(f"    {row.get('count')}: {row.get('phrase')}")
    print()

    print("near duplicates:")
    print(f"  exact_paragraph_duplicate_count: {diagnostics.get('exact_paragraph_duplicate_count', 0)}")
    print(
        "  normalized_paragraph_duplicate_count: "
        f"{diagnostics.get('normalized_paragraph_duplicate_count', 0)}"
    )
    print(f"  near_duplicate_cluster_count: {diagnostics.get('near_duplicate_cluster_count', 0)}")
    print(
        "  largest_near_duplicate_cluster_size: "
        f"{diagnostics.get('largest_near_duplicate_cluster_size', 0)}"
    )
    for row in sorted(per_source, key=lambda item: float(item.get("near_duplicate_ratio", 0.0)), reverse=True):
        print(f"  {row.get('source')}: near_duplicate_ratio={_fmt_float(row.get('near_duplicate_ratio', 0.0))}")


def _print_comparison(
    current_path: Path,
    current_payload: dict[str, Any],
    compare_path: Path,
    compare_payload: dict[str, Any],
) -> None:
    current = _manifest_summary(current_path, current_payload)
    baseline = _manifest_summary(compare_path, compare_payload)
    print()
    print("comparison:")
    print(f"  current: {current_path}")
    print(f"  compare_to: {compare_path}")
    for key in (
        "tokens",
        "documents",
        "max_source_share",
        "max_repeat_rate",
        "max_near_duplicate_ratio",
        "largest_near_duplicate_cluster_size",
        "medical_body_density",
        "product_commercial_density",
        "navigation_text_density",
        "dictionary_fragment_density",
        "page_boilerplate_density",
        "malformed_fragment_density",
        "generic_article_formula_density",
        "corpus_gate_passed",
        "broad_source_gate_passed",
    ):
        print(f"  {key}: current={current.get(key)} compare_to={baseline.get(key)}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Print a compact pretrain prepared-manifest report.")
    parser.add_argument("manifest", help="Path to a prepared pretrain manifest JSON.")
    parser.add_argument("--compare-to", help="Optional prepared manifest to compare against.")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    payload = _load(manifest_path)
    _print_manifest(manifest_path, payload)
    if args.compare_to:
        compare_path = Path(args.compare_to)
        _print_comparison(manifest_path, payload, compare_path, _load(compare_path))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
