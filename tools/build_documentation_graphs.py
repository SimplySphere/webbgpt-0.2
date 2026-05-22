#!/usr/bin/env python3.12
"""Generate the WebbGPT documentation graphics set from local artifacts.

The script is intentionally read-only with respect to model artifacts: it reads
logs, JSON, JSONL, source manifests, and code files, then writes PNG/GIF files
under documentation/.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import re
import textwrap
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "documentation"

PRETRAIN_HISTORY = ROOT / "artifacts/runs/local-mvp/checkpoints/pretrain/eval_history.jsonl"
PRETRAIN_SUMMARY = ROOT / "artifacts/runs/local-mvp/checkpoints/pretrain/stage_summary.json"
PRETRAIN_RUN_METADATA = ROOT / "artifacts/runs/local-mvp/checkpoints/pretrain/run_metadata.json"
PRETRAIN_MANIFEST = ROOT / "artifacts/runs/local-mvp/prepared/pretrain.json"
VALIDATION_MANIFEST = ROOT / "artifacts/runs/local-mvp/prepared/validation.json"
PRETRAIN_LOG = ROOT / "artifacts/runs/local-mvp/pretrain_final_curated.log"
PREPARE_PRETRAIN_LOG = ROOT / "artifacts/runs/local-mvp/prepare_pretrain.log"

SFT_SUMMARIES = [
    ROOT / "artifacts/runs/local-mvp/checkpoints/sft-small/stage_summary.json",
    ROOT / "artifacts/runs/local-mvp/checkpoints/sft-v2/stage_summary.json",
    ROOT / "artifacts/runs/local-mvp/checkpoints/sft-v3/stage_summary.json",
]
SFT_LOGS = [
    ROOT / "artifacts/runs/local-mvp/sft0_plumbing.log",
    ROOT / "artifacts/runs/local-mvp/sft_v3.log",
]

RAG_BEFORE = ROOT / "artifacts/runs/local-mvp/rag_reliability_regression.json"
RAG_AFTER = ROOT / "artifacts/runs/local-mvp/rag_reliability_regression_after_source_expansion.json"
RAG_QUERY_CHECKS = ROOT / "artifacts/runs/local-mvp/rag_query_checks_after_source_expansion.json"
RAG_SERVING = ROOT / "artifacts/runs/local-mvp/rag_serving_verification_after_source_expansion.json"
RAG_CHUNKS = ROOT / "data/rag/webbgpt_chunks.jsonl"
RAG_INDEX = ROOT / "data/rag/webbgpt_index.json"
RAG_MANIFEST = ROOT / "data/rag/webbgpt_sources_manifest.json"
PREFLIGHT = ROOT / "artifacts/runs/local-mvp/final_continue_sft_rag_preflight.json"
MANUAL_DEMOS = ROOT / "artifacts/runs/local-mvp/manual_demos/chat_transcript.jsonl"

COMPARISON_FILES = [
    ROOT / "artifacts/runs/local-mvp/final_pretrained_vs_sft_v3_comparison.json",
    ROOT / "artifacts/runs/local-mvp/sft_v2_comparison_samples.json",
]

QUALITY_SUMMARIES = [
    ROOT / "artifacts/runs/scale-3b-smoke-100m/reports/data_audit/quality_summary.json",
    ROOT / "artifacts/runs/scale-3b-smoke-100m/reports/data_audit_cleanup/quality_summary.json",
]

README = ROOT / "README.md"
SERVE_APP = ROOT / "src/serve/app.py"
PLAYGROUND = ROOT / "src/serve/playground.py"
PYTEST_LASTFAILED = ROOT / ".pytest_cache/v/cache/lastfailed"

REQUESTED_VISUALS = [
    ("01_pretrain_loss_curve.png", "Pretraining Loss Over Time", "png"),
    ("02_pretrain_perplexity_curve.png", "Pretraining Perplexity Over Time", "png"),
    ("03_loss_and_perplexity_dual_axis.png", "Loss and Perplexity Move Together", "png"),
    ("04_experiment_timeline_chart.png", "How WebbGPT 0.2 Came Together", "png"),
    ("05_training_data_source_mix.png", "What Fed Pretraining?", "png"),
    ("06_data_filtering_funnel.png", "From Raw Text to Training Tokens", "png"),
    ("07_data_rejection_reasons.png", "What the Filters Removed", "png"),
    ("08_rag_chunk_distribution.png", "RAG Sources at a Glance", "png"),
    ("09_rag_regression_outcomes.png", "How the RAG Demo Responds", "png"),
    ("10_rag_prompt_outcome_heatmap.png", "RAG Prompt Outcomes", "png"),
    ("11_failure_mode_heatmap.png", "Where Outputs Failed", "png"),
    ("12_generation_length_distribution.png", "How Long Saved Outputs Were", "png"),
    ("13_repetition_score_distribution.png", "Repetition in Saved Outputs", "png"),
    ("14_retrieval_score_distribution.png", "Retrieval Scores", "png"),
    ("15_query_source_heatmap.png", "Which Sources Each Query Retrieved", "png"),
    ("16_rag_before_after_comparison.png", "RAG Source Expansion Changed Coverage", "png"),
    ("17_model_quality_radar.png", "Final System Behavior at a Glance", "png"),
    ("18_demo_readiness_scorecard.png", "Demo Readiness Checklist", "png"),
    ("19_training_loss_animation.gif", "Loss Curve Animation", "gif"),
    ("20_perplexity_animation.gif", "Perplexity Curve Animation", "gif"),
    ("21_rag_retrieval_heatmap_animation.gif", "Retrieval Heatmap Animation", "gif"),
    ("22_streaming_output_demo.gif", "Progressive Response Demo", "gif"),
    ("23_data_filtering_funnel_animation.gif", "Filtering Funnel Animation", "gif"),
]
CONTACT_SHEET = ("00_graph_index.png", "Graph Index", "png")
REQUESTED_FILENAMES = {name for name, _, _ in REQUESTED_VISUALS} | {CONTACT_SHEET[0]}

SUBTITLES = {
    "00_graph_index.png": "A clean catalog of the generated WebbGPT documentation visuals.",
    "01_pretrain_loss_curve.png": "Validation loss fell steadily during local-MVP pretraining.",
    "02_pretrain_perplexity_curve.png": "Perplexity dropped sharply as the model learned the corpus.",
    "03_loss_and_perplexity_dual_axis.png": "Both curves show the same training story from different angles.",
    "04_experiment_timeline_chart.png": "Major phases are positioned from saved artifact timestamps and README evidence.",
    "05_training_data_source_mix.png": "The prepared corpus balanced local curated prose with the FineWeb extension.",
    "06_data_filtering_funnel.png": "Filtering removed noisy inputs before packing the final token stream.",
    "07_data_rejection_reasons.png": "Medical, repeated, and URL-heavy documents drove most rejections.",
    "08_rag_chunk_distribution.png": "Most retrieval chunks come from the large Webb context corpus.",
    "09_rag_regression_outcomes.png": "Saved regression prompts show when the demo generated, abstained, or fell back.",
    "10_rag_prompt_outcome_heatmap.png": "Each row shows what happened for one final RAG regression prompt.",
    "11_failure_mode_heatmap.png": "Saved quality flags plus deterministic heuristics summarize failure patterns.",
    "12_generation_length_distribution.png": "Saved demo and evaluation outputs vary from short abstentions to long fallbacks.",
    "13_repetition_score_distribution.png": "A deterministic score highlights outputs with repeated wording.",
    "14_retrieval_score_distribution.png": "Higher scores indicate stronger matches between queries and source chunks.",
    "15_query_source_heatmap.png": "Rows are queries; columns are source files; color is max retrieval score.",
    "16_rag_before_after_comparison.png": "Source expansion improved source surfacing but weak generations still appear.",
    "17_model_quality_radar.png": "A heuristic rubric summarizes final system behavior for the demo.",
    "18_demo_readiness_scorecard.png": "The checklist is grounded in saved verification, repo state, and cached test status.",
    "19_training_loss_animation.gif": "The validation-loss curve draws in progressively.",
    "20_perplexity_animation.gif": "The perplexity curve draws in progressively.",
    "21_rag_retrieval_heatmap_animation.gif": "Rows appear one by one to show retrieval coverage building across queries.",
    "22_streaming_output_demo.gif": "Saved response text is revealed in a WebbGPT-style UI frame.",
    "23_data_filtering_funnel_animation.gif": "The filtering and packing funnel builds stage by stage.",
}

STYLE = {
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#D8DEE6",
    "axes.labelcolor": "#26323F",
    "xtick.color": "#536271",
    "ytick.color": "#536271",
    "text.color": "#26323F",
    "font.family": "DejaVu Sans",
    "font.size": 11.5,
    "axes.titleweight": "bold",
    "axes.titlesize": 18,
    "axes.labelsize": 12,
    "xtick.labelsize": 10.5,
    "ytick.labelsize": 10.5,
    "axes.grid": True,
    "grid.color": "#E9EDF2",
    "grid.linewidth": 0.9,
    "grid.alpha": 0.9,
    "legend.fontsize": 10.5,
}

PALETTE = {
    "navy": "#17324D",
    "blue": "#244C6F",
    "teal": "#2A9D8F",
    "orange": "#E9A23B",
    "coral": "#D96C5F",
    "gray": "#6B7785",
    "light_gray": "#EEF2F6",
    "mid_gray": "#D8DEE6",
    "pale_blue": "#E8F1F8",
    "pale_teal": "#E6F4F1",
    "pale_orange": "#FFF3DE",
    "pale_coral": "#FBEAE7",
    "white": "#FFFFFF",
    "green": "#2A9D8F",
    "purple": "#244C6F",
    "pink": "#E9A23B",
    "red": "#D96C5F",
    "light_blue": "#E8F1F8",
    "light_green": "#E6F4F1",
    "light_orange": "#FFF3DE",
    "light_red": "#FBEAE7",
}


@dataclass
class VisualRecord:
    filename: str
    title: str
    visual_type: str
    generated: bool = False
    filepath: str | None = None
    data_sources: list[str] = field(default_factory=list)
    grounding: str = "direct"
    caveats: list[str] = field(default_factory=list)
    skipped_reason: str | None = None
    key_numbers: dict[str, Any] = field(default_factory=dict)


def rel(path: Path | str) -> str:
    path = Path(path)
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def source_list(paths: Iterable[Path | str]) -> list[str]:
    return sorted({rel(path) for path in paths if Path(path).exists()})


def safe_json(path: Path) -> tuple[Any | None, str | None]:
    if not path.exists():
        return None, f"missing: {rel(path)}"
    try:
        return json.loads(path.read_text(encoding="utf-8")), None
    except Exception as exc:  # noqa: BLE001 - warnings are surfaced in manifest.
        return None, f"could not parse {rel(path)}: {exc}"


def safe_jsonl(path: Path) -> tuple[list[dict[str, Any]], str | None]:
    if not path.exists():
        return [], f"missing: {rel(path)}"
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                loaded = json.loads(line)
                if isinstance(loaded, dict):
                    rows.append(loaded)
        return rows, None
    except Exception as exc:  # noqa: BLE001
        return rows, f"could not parse {rel(path)} line {line_no}: {exc}"


def read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def numeric(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    if isinstance(value, str):
        try:
            parsed = float(value)
        except ValueError:
            return None
        return parsed if math.isfinite(parsed) else None
    return None


def short_label(value: str, width: int = 36) -> str:
    value = " ".join(str(value).split())
    if len(value) <= width:
        return value
    return value[: width - 1].rstrip() + "..."


def clean_name(value: str) -> str:
    stem = Path(value).stem
    return stem.replace("_safe", "").replace("_", " ").replace("-", " ").title()


def word_count(text: str) -> int:
    return len(re.findall(r"[A-Za-z0-9']+", text or ""))


def repetition_score(text: str, metrics: dict[str, Any] | None = None) -> float:
    metrics = metrics or {}
    unique_ratio = numeric(metrics.get("unique_token_ratio"))
    if unique_ratio is not None:
        return max(0.0, min(1.0, 1.0 - unique_ratio))
    tokens = re.findall(r"[a-z0-9']+", (text or "").lower())
    if not tokens:
        return 1.0
    counts = Counter(tokens)
    dominant = max(counts.values()) / len(tokens)
    if len(tokens) < 4:
        return dominant
    fourgrams = Counter(tuple(tokens[i : i + 4]) for i in range(len(tokens) - 3))
    repeated_4gram = max(fourgrams.values()) if fourgrams else 1
    return max(0.0, min(1.0, dominant + max(0, repeated_4gram - 1) * 0.12))


def max_repeated_ngram(text: str, n: int = 4) -> int:
    tokens = re.findall(r"[a-z0-9']+", (text or "").lower())
    if len(tokens) < n:
        return 0
    return max(Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)).values())


def setup_style() -> None:
    plt.rcParams.update(STYLE)


def make_fig(
    figsize: tuple[float, float] = (10.2, 6.0),
    *,
    polar: bool = False,
) -> tuple[plt.Figure, plt.Axes]:
    fig = plt.figure(figsize=figsize, constrained_layout=True)
    ax = fig.add_subplot(111, polar=polar)
    return fig, ax


def subtitle_for(filename: str) -> str:
    return SUBTITLES.get(filename, "")


def add_title(ax: plt.Axes, title: str, subtitle: str | None = None) -> None:
    ax.set_title(title, loc="left", pad=30, fontsize=18, fontweight="bold", color=PALETTE["navy"])
    if subtitle:
        ax.text(
            0.0,
            1.015,
            subtitle,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=11.5,
            color=PALETTE["gray"],
            wrap=True,
        )


def polish_axes(fig: plt.Figure) -> None:
    for ax in fig.axes:
        if getattr(ax, "name", "") == "polar":
            ax.grid(color=PALETTE["mid_gray"], linewidth=0.9, alpha=0.8)
            continue
        ax.set_axisbelow(True)
        ax.tick_params(axis="both", which="both", length=0, pad=7, colors=PALETTE["gray"])
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(PALETTE["mid_gray"])
            ax.spines[side].set_linewidth(0.9)


def add_callout(
    ax: plt.Axes,
    text: str,
    xy: tuple[float, float],
    xytext: tuple[float, float],
    *,
    color: str = "orange",
    arrow: bool = True,
) -> None:
    arrowprops = (
        {
            "arrowstyle": "-|>",
            "lw": 1.1,
            "color": PALETTE[color],
            "shrinkA": 4,
            "shrinkB": 4,
        }
        if arrow
        else None
    )
    ax.annotate(
        text,
        xy=xy,
        xytext=xytext,
        textcoords="offset points",
        fontsize=10,
        color=PALETTE["navy"],
        bbox={
            "boxstyle": "round,pad=0.45,rounding_size=0.12",
            "facecolor": PALETTE["pale_orange"] if color == "orange" else PALETTE["pale_teal"],
            "edgecolor": PALETTE[color],
            "linewidth": 0.8,
        },
        arrowprops=arrowprops,
    )


def display_source(value: str) -> str:
    replacements = {
        "local_mvp_pretrain_corpus": "Local curated prose",
        "fineweb_extension_corpus": "FineWeb extension",
        "local_mvp_curated_validation_corpus": "Validation prose",
        "webb_school_context_large_lm_corpus": "Webb context corpus",
    }
    return replacements.get(value, clean_name(value))


def compact_number(value: float | int) -> str:
    value = float(value)
    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if abs(value) >= 1_000:
        return f"{value / 1_000:.1f}K"
    return f"{value:.0f}"


def save_fig(fig: plt.Figure, output: Path) -> None:
    polish_axes(fig)
    fig.savefig(output, dpi=170, bbox_inches="tight", pad_inches=0.38, facecolor="white")
    plt.close(fig)


def add_bar_labels(ax: plt.Axes, bars: Iterable[Any], *, fmt: str = "{:,.0f}", x_pad: float = 0.01) -> None:
    xmax = ax.get_xlim()[1] if ax.get_xlim()[1] else 1.0
    for bar in bars:
        width = bar.get_width()
        y = bar.get_y() + bar.get_height() / 2
        ax.text(width + xmax * x_pad, y, fmt.format(width), va="center", fontsize=9.5, color=PALETTE["gray"])


def editorial_cmap(name: str, tone: str = "teal") -> LinearSegmentedColormap:
    if tone == "coral":
        colors = [PALETTE["white"], PALETTE["pale_coral"], PALETTE["coral"]]
    elif tone == "blue":
        colors = [PALETTE["white"], PALETTE["pale_blue"], PALETTE["blue"]]
    else:
        colors = [PALETTE["white"], PALETTE["pale_teal"], PALETTE["teal"]]
    return LinearSegmentedColormap.from_list(name, colors)


def skip(records: list[VisualRecord], record: VisualRecord, reason: str, caveat: str | None = None) -> None:
    record.generated = False
    record.skipped_reason = reason
    if caveat:
        record.caveats.append(caveat)
    records.append(record)


def finish(
    records: list[VisualRecord],
    record: VisualRecord,
    output: Path,
    sources: Iterable[Path | str],
    grounding: str = "direct",
    caveats: Iterable[str] = (),
    key_numbers: dict[str, Any] | None = None,
) -> None:
    record.generated = True
    record.filepath = rel(output)
    record.data_sources = source_list(Path(path) for path in sources)
    record.grounding = grounding
    record.caveats = [item for item in caveats if item]
    record.key_numbers = key_numbers or {}
    records.append(record)


def load_pretrain_series() -> list[dict[str, float]]:
    rows, _ = safe_jsonl(PRETRAIN_HISTORY)
    series: list[dict[str, float]] = []
    for index, row in enumerate(rows):
        eval_block = row.get("eval") if isinstance(row.get("eval"), dict) else {}
        step = numeric(row.get("step")) or float(index)
        loss = numeric(eval_block.get("loss") or row.get("validation_loss") or row.get("loss"))
        ppl = numeric(eval_block.get("perplexity") or row.get("perplexity"))
        if loss is None and ppl is None:
            continue
        point = {"step": step}
        if loss is not None:
            point["loss"] = loss
        if ppl is not None:
            point["perplexity"] = ppl
        elif loss is not None:
            point["perplexity"] = math.exp(min(loss, 13.8))
        series.append(point)
    return sorted(series, key=lambda item: item["step"])


def load_pretrain_manifest() -> dict[str, Any]:
    data, _ = safe_json(PRETRAIN_MANIFEST)
    return data if isinstance(data, dict) else {}


def manifest_per_source(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    diagnostics = manifest.get("diagnostics") if isinstance(manifest.get("diagnostics"), dict) else {}
    rows = diagnostics.get("per_source") if isinstance(diagnostics.get("per_source"), list) else []
    return [row for row in rows if isinstance(row, dict)]


def source_mix_rows() -> list[dict[str, Any]]:
    manifest = load_pretrain_manifest()
    rows = []
    for row in manifest_per_source(manifest):
        rows.append(
            {
                "source": str(row.get("source") or "unknown"),
                "family": str(row.get("family") or "unknown"),
                "kept_tokens": int(row.get("kept_tokens") or 0),
                "target_share": float(row.get("target_share") or row.get("weight") or 0.0),
                "kept_documents": int(row.get("kept_documents") or 0),
                "raw_records": int(row.get("raw_records") or 0),
            }
        )
    return rows


def rejection_counts() -> Counter[str]:
    manifest = load_pretrain_manifest()
    counts: Counter[str] = Counter()
    for row in manifest_per_source(manifest):
        dropped = row.get("dropped_reasons")
        if isinstance(dropped, dict):
            for reason, value in dropped.items():
                counts[str(reason)] += int(value or 0)
    if counts:
        return counts
    for path in QUALITY_SUMMARIES:
        data, _ = safe_json(path)
        if not isinstance(data, dict):
            continue
        rejected = data.get("rejected_document_stats")
        if not isinstance(rejected, dict):
            continue
        for stats in rejected.values():
            if not isinstance(stats, dict):
                continue
            for reason, value in stats.get("top_rejection_reasons") or []:
                counts[str(reason)] += int(value or 0)
    return counts


def funnel_counts() -> list[tuple[str, int, str]]:
    manifest = load_pretrain_manifest()
    per_source = manifest_per_source(manifest)
    raw_records = sum(int(row.get("raw_records") or 0) for row in per_source)
    dropped = sum(sum(int(value or 0) for value in (row.get("dropped_reasons") or {}).values()) for row in per_source)
    accepted = sum(int(row.get("kept_documents") or 0) for row in per_source)
    repeated = sum(int(row.get("repeated_documents") or 0) for row in per_source)
    sequences = int(manifest.get("num_sequences") or 0)
    tokens = int(manifest.get("num_tokens") or 0)
    cleaned = max(raw_records - dropped, 0) if raw_records else accepted
    return [
        ("raw records", raw_records, "records"),
        ("cleaned records", cleaned, "records after rejection filters"),
        ("accepted docs", accepted, "documents"),
        ("repeat-packed docs", repeated, "document reuses for source balance"),
        ("packed sequences", sequences, "512-token sequences"),
        ("final tokens", tokens, "tokens"),
    ]


def load_regression(path: Path) -> list[dict[str, Any]]:
    data, _ = safe_json(path)
    if isinstance(data, dict) and isinstance(data.get("rows"), list):
        return [row for row in data["rows"] if isinstance(row, dict)]
    return []


def load_query_checks() -> list[dict[str, Any]]:
    data, _ = safe_json(RAG_QUERY_CHECKS)
    if isinstance(data, dict) and isinstance(data.get("queries"), list):
        return [row for row in data["queries"] if isinstance(row, dict)]
    return []


def load_serving_rows() -> list[dict[str, Any]]:
    data, _ = safe_json(RAG_SERVING)
    if isinstance(data, dict) and isinstance(data.get("rows"), list):
        return [row for row in data["rows"] if isinstance(row, dict)]
    return []


def regression_text(row: dict[str, Any]) -> str:
    return str(row.get("text") or row.get("response") or "")


def row_quality_reasons(row: dict[str, Any]) -> list[str]:
    reasons = row.get("quality_reasons")
    if isinstance(reasons, list):
        return [str(item) for item in reasons]
    quality = row.get("quality") if isinstance(row.get("quality"), dict) else {}
    reasons = quality.get("reasons")
    if isinstance(reasons, list):
        return [str(item) for item in reasons]
    return []


def row_status(row: dict[str, Any]) -> dict[str, Any]:
    status = row.get("status")
    return status if isinstance(status, dict) else row


def row_hits(row: dict[str, Any]) -> list[dict[str, Any]]:
    if isinstance(row.get("hits"), list):
        return [hit for hit in row["hits"] if isinstance(hit, dict)]
    rag = row.get("rag") if isinstance(row.get("rag"), dict) else {}
    if isinstance(rag.get("hits"), list):
        return [hit for hit in rag["hits"] if isinstance(hit, dict)]
    return []


def retrieved_count(row: dict[str, Any]) -> int:
    if numeric(row.get("retrieved_hit_count")) is not None:
        return int(row.get("retrieved_hit_count") or 0)
    rag = row.get("rag") if isinstance(row.get("rag"), dict) else {}
    if numeric(rag.get("retrieved_hits")) is not None:
        return int(rag.get("retrieved_hits") or 0)
    return len(row_hits(row))


def is_abstained(row: dict[str, Any]) -> bool:
    status = row_status(row)
    text = regression_text(row).lower()
    return bool(status.get("abstained") or row.get("abstained") or "does not contain enough information" in text)


def is_weak(row: dict[str, Any]) -> bool:
    status = row_status(row)
    return bool(
        row.get("output_degenerate")
        or status.get("degenerate_output")
        or status.get("generation_failed")
        or row_quality_reasons(row)
    )


def is_fallback(row: dict[str, Any]) -> bool:
    status = row_status(row)
    label = str(row.get("final_label") or status.get("final_label") or "").lower()
    return bool(row.get("fallback_triggered") or status.get("retrieved_context_fallback") or "fallback" in label)


def generated_text(row: dict[str, Any]) -> bool:
    return bool(regression_text(row).strip()) and not is_abstained(row)


def source_attached(row: dict[str, Any]) -> bool:
    if retrieved_count(row) <= 0:
        return False
    if row.get("source_display_correct") is False:
        return False
    hits = row_hits(row)
    return not hits or any(hit.get("source_file") for hit in hits)


def rag_outcome_counts(rows: list[dict[str, Any]]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for row in rows:
        if generated_text(row):
            counts["Generated"] += 1
        if generated_text(row) and source_attached(row):
            counts["Generated with sources"] += 1
        if is_weak(row):
            counts["Weak generation"] += 1
        if is_abstained(row):
            counts["Abstained"] += 1
        if row_status(row).get("generation_failed") or (is_fallback(row) and is_weak(row)):
            counts["Generation failed"] += 1
    return counts


def collect_output_cases() -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for source, rows in [
        (RAG_AFTER, load_regression(RAG_AFTER)),
        (RAG_BEFORE, load_regression(RAG_BEFORE)),
        (RAG_SERVING, load_serving_rows()),
    ]:
        for row in rows:
            text = regression_text(row)
            if text:
                quality = row.get("quality") if isinstance(row.get("quality"), dict) else {}
                metrics = quality.get("metrics") if isinstance(quality.get("metrics"), dict) else {}
                cases.append(
                    {
                        "prompt": str(row.get("prompt") or "prompt"),
                        "text": text,
                        "metrics": metrics,
                        "source": source,
                        "reasons": row_quality_reasons(row),
                    }
                )
    rows, _ = safe_jsonl(MANUAL_DEMOS)
    for row in rows:
        response = row.get("response") if isinstance(row.get("response"), dict) else {}
        metadata = response.get("metadata") if isinstance(response.get("metadata"), dict) else {}
        quality = metadata.get("quality") if isinstance(metadata.get("quality"), dict) else {}
        metrics = quality.get("metrics") if isinstance(quality.get("metrics"), dict) else {}
        text = str(response.get("text") or "")
        if text:
            messages = row.get("request", {}).get("messages", []) if isinstance(row.get("request"), dict) else []
            prompt = messages[-1].get("content") if messages and isinstance(messages[-1], dict) else "manual demo"
            cases.append(
                {
                    "prompt": str(prompt),
                    "text": text,
                    "metrics": metrics,
                    "source": MANUAL_DEMOS,
                    "reasons": [str(item) for item in quality.get("reasons", [])] if isinstance(quality.get("reasons"), list) else [],
                }
            )
    for path in COMPARISON_FILES:
        data, _ = safe_json(path)
        if not isinstance(data, dict):
            continue
        samples = data.get("samples")
        if not isinstance(samples, dict):
            continue
        for checkpoint, sample_rows in samples.items():
            if not isinstance(sample_rows, list):
                continue
            for sample in sample_rows:
                if not isinstance(sample, dict):
                    continue
                text = str(sample.get("response") or "")
                if not text:
                    continue
                evaluation = sample.get("evaluation") if isinstance(sample.get("evaluation"), dict) else {}
                cases.append(
                    {
                        "prompt": str(sample.get("prompt_id") or sample.get("prompt") or checkpoint),
                        "text": text,
                        "metrics": {"unique_token_ratio": None},
                        "source": path,
                        "reasons": [
                            name
                            for name, flag in evaluation.items()
                            if isinstance(flag, bool) and flag and name in {"fake_citation", "semantic_repetition", "blank"}
                        ],
                    }
                )
    return cases


def retrieval_scores() -> list[tuple[str, str, float, Path]]:
    values: list[tuple[str, str, float, Path]] = []
    for query in load_query_checks():
        prompt = str(query.get("query") or "query")
        for hit in query.get("hits") or []:
            if isinstance(hit, dict) and numeric(hit.get("score")) is not None:
                values.append((prompt, str(hit.get("source_file") or "unknown"), float(hit["score"]), RAG_QUERY_CHECKS))
    for path in [RAG_AFTER, RAG_BEFORE, RAG_SERVING]:
        rows = load_regression(path) if path != RAG_SERVING else load_serving_rows()
        for row in rows:
            prompt = str(row.get("prompt") or "prompt")
            for hit in row_hits(row):
                if numeric(hit.get("score")) is not None:
                    values.append((prompt, str(hit.get("source_file") or "unknown"), float(hit["score"]), path))
    return values


def plot_loss(records: list[VisualRecord], output_dir: Path) -> None:
    filename, title, visual_type = REQUESTED_VISUALS[0]
    record = VisualRecord(filename, title, visual_type)
    series = load_pretrain_series()
    points = [(row["step"], row["loss"]) for row in series if "loss" in row]
    if not points:
        skip(records, record, "pretraining eval history did not contain validation loss")
        return
    steps, losses = zip(*points)
    fig, ax = make_fig(figsize=(10.4, 6.1))
    ax.plot(steps, losses, color=PALETTE["blue"], linewidth=3.0)
    ax.fill_between(steps, losses, max(losses), color=PALETTE["pale_blue"], alpha=0.55)
    ax.scatter([steps[-1]], [losses[-1]], s=62, color=PALETTE["teal"], edgecolor="white", linewidth=1.2, zorder=3)
    add_title(ax, title, subtitle_for(filename))
    ax.set_xlabel("Training step")
    ax.set_ylabel("Validation loss")
    drop = losses[0] - losses[-1]
    add_callout(
        ax,
        f"Final loss {losses[-1]:.3f}\n{drop:.3f} lower than start",
        (steps[-1], losses[-1]),
        (-140, 28),
        color="teal",
    )
    output = output_dir / filename
    save_fig(fig, output)
    finish(
        records,
        record,
        output,
        [PRETRAIN_HISTORY],
        "direct",
        key_numbers={"points": len(points), "final_loss": round(losses[-1], 6), "best_loss": round(min(losses), 6)},
    )


def plot_perplexity(records: list[VisualRecord], output_dir: Path) -> None:
    filename, title, visual_type = REQUESTED_VISUALS[1]
    record = VisualRecord(filename, title, visual_type)
    series = load_pretrain_series()
    points = [(row["step"], row["perplexity"]) for row in series if "perplexity" in row]
    if not points:
        skip(records, record, "pretraining eval history did not contain perplexity or loss")
        return
    steps, ppls = zip(*points)
    plotted = [min(value, 1000.0) for value in ppls]
    fig, ax = make_fig(figsize=(10.4, 6.1))
    ax.plot(steps, plotted, color=PALETTE["teal"], linewidth=3.0)
    ax.fill_between(steps, plotted, max(plotted), color=PALETTE["pale_teal"], alpha=0.65)
    ax.scatter([steps[-1]], [plotted[-1]], s=62, color=PALETTE["blue"], edgecolor="white", linewidth=1.2, zorder=3)
    add_title(ax, title, subtitle_for(filename))
    ax.set_xlabel("Training step")
    ax.set_ylabel("Perplexity (clipped at 1,000)")
    add_callout(ax, f"Final perplexity {ppls[-1]:.1f}", (steps[-1], plotted[-1]), (-132, 28), color="teal")
    output = output_dir / filename
    save_fig(fig, output)
    finish(
        records,
        record,
        output,
        [PRETRAIN_HISTORY],
        "direct",
        caveats=["Values above 1000 are clipped for readability."],
        key_numbers={"points": len(points), "final_perplexity": round(ppls[-1], 6), "best_perplexity": round(min(ppls), 6)},
    )


def plot_dual_axis(records: list[VisualRecord], output_dir: Path) -> None:
    filename, title, visual_type = REQUESTED_VISUALS[2]
    record = VisualRecord(filename, title, visual_type)
    series = load_pretrain_series()
    points = [(row["step"], row["loss"], row["perplexity"]) for row in series if "loss" in row and "perplexity" in row]
    if not points:
        skip(records, record, "pretraining eval history did not contain paired loss/perplexity points")
        return
    steps = [item[0] for item in points]
    losses = [item[1] for item in points]
    ppls = [min(item[2], 1000.0) for item in points]
    fig, ax_loss = make_fig(figsize=(10.6, 6.1))
    ax_ppl = ax_loss.twinx()
    ax_loss.plot(steps, losses, color=PALETTE["blue"], linewidth=3.0, label="Validation loss")
    ax_ppl.plot(steps, ppls, color=PALETTE["teal"], linewidth=3.0, label="Perplexity")
    add_title(ax_loss, title, subtitle_for(filename))
    ax_loss.set_xlabel("Training step")
    ax_loss.set_ylabel("Validation loss", color=PALETTE["blue"])
    ax_ppl.set_ylabel("Perplexity (clipped at 1,000)", color=PALETTE["teal"])
    ax_ppl.tick_params(axis="y", colors=PALETTE["teal"], length=0, pad=7)
    ax_ppl.spines["right"].set_color(PALETTE["mid_gray"])
    lines = ax_loss.get_lines() + ax_ppl.get_lines()
    ax_loss.legend(lines, [line.get_label() for line in lines], frameon=False, loc="upper right", bbox_to_anchor=(1.0, 1.01))
    output = output_dir / filename
    save_fig(fig, output)
    finish(records, record, output, [PRETRAIN_HISTORY], "direct", caveats=["Perplexity is clipped at 1000 for readability."])


def file_time(path: Path) -> dt.datetime | None:
    if not path.exists():
        return None
    return dt.datetime.fromtimestamp(path.stat().st_mtime, tz=dt.UTC)


def plot_timeline(records: list[VisualRecord], output_dir: Path) -> None:
    filename, title, visual_type = REQUESTED_VISUALS[3]
    record = VisualRecord(filename, title, visual_type)
    phases: list[dict[str, Any]] = []
    def add_phase(name: str, start_path: Path, end_path: Path, caveat: str = "") -> None:
        start = file_time(start_path)
        end = file_time(end_path)
        if start and end:
            if end < start:
                start, end = end, start
            phases.append({"name": name, "start": start, "end": end, "sources": [start_path, end_path], "caveat": caveat})

    add_phase("baseline pretraining", ROOT / "artifacts/runs/local-mvp/sanity_probe.log", ROOT / "artifacts/runs/local-mvp/checkpoints/sanity-probe/stage_summary.json")
    add_phase("data cleanup", QUALITY_SUMMARIES[0], QUALITY_SUMMARIES[1])
    add_phase("curated pretraining", PREPARE_PRETRAIN_LOG, PRETRAIN_SUMMARY)
    add_phase("SFT attempts", SFT_LOGS[0], SFT_SUMMARIES[-1])
    add_phase("RAG integration", RAG_BEFORE, RAG_SERVING)
    add_phase("WebbGPT 0.2 UI/demo", MANUAL_DEMOS, RAG_SERVING)
    readme_time = file_time(README)
    if readme_time and "DPO is archived as legacy" in read_text(README):
        phases.append(
            {
                "name": "DPO attempts",
                "start": readme_time,
                "end": readme_time,
                "sources": [README],
                "caveat": "README records DPO as archived legacy; no active DPO artifact was present.",
            }
        )
    if not phases:
        skip(records, record, "no project milestone timestamps were available")
        return
    phases.sort(key=lambda item: item["start"])
    fig, ax = make_fig(figsize=(11.4, 6.4))
    colors = [PALETTE["blue"], PALETTE["teal"], PALETTE["orange"], PALETTE["coral"], PALETTE["navy"], PALETTE["gray"], PALETTE["mid_gray"]]
    for idx, phase in enumerate(phases):
        start = phase["start"]
        end = phase["end"]
        if start == end:
            ax.scatter(start, idx, marker="D", s=95, color=colors[idx % len(colors)], edgecolor="white", linewidth=1.0, zorder=3)
        else:
            ax.barh(
                idx,
                (end - start).total_seconds() / 86400.0,
                left=start,
                height=0.52,
                color=colors[idx % len(colors)],
                alpha=0.9,
            )
    ax.set_yticks(range(len(phases)))
    ax.set_yticklabels([phase["name"].title() for phase in phases])
    add_title(ax, title, subtitle_for(filename))
    ax.set_xlabel("Artifact timestamp")
    ax.grid(axis="x", color=PALETTE["mid_gray"], alpha=0.75)
    ax.grid(axis="y", visible=False)
    fig.autofmt_xdate()
    output = output_dir / filename
    save_fig(fig, output)
    caveats = [phase["caveat"] for phase in phases if phase.get("caveat")]
    finish(
        records,
        record,
        output,
        [source for phase in phases for source in phase["sources"]],
        "derived",
        caveats=caveats + ["Phase dates are derived from artifact file modification times or README evidence."],
        key_numbers={"phase_count": len(phases)},
    )


def plot_source_mix(records: list[VisualRecord], output_dir: Path) -> None:
    filename, title, visual_type = REQUESTED_VISUALS[4]
    record = VisualRecord(filename, title, visual_type)
    rows = source_mix_rows()
    if not rows:
        skip(records, record, "pretraining prepared manifest did not contain per-source diagnostics")
        return
    labels = [display_source(row["source"]) for row in rows]
    kept_tokens = [row["kept_tokens"] for row in rows]
    target_tokens = [sum(kept_tokens) * row["target_share"] for row in rows]
    x = range(len(rows))
    fig, ax = make_fig(figsize=(10.5, 6.0))
    width = 0.34
    actual_bars = ax.bar([i - width / 2 for i in x], kept_tokens, width=width, color=PALETTE["blue"], label="Actual kept tokens")
    target_bars = ax.bar([i + width / 2 for i in x], target_tokens, width=width, color=PALETTE["teal"], alpha=0.82, label="Target share")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=12, ha="right")
    ax.set_ylabel("Tokens")
    add_title(ax, title, subtitle_for(filename))
    ax.legend(frameon=False, ncols=2, loc="upper right")
    for bars in (actual_bars, target_bars):
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + max(kept_tokens + target_tokens) * 0.025,
                compact_number(height),
                ha="center",
                va="bottom",
                fontsize=9,
                color=PALETTE["gray"],
            )
    output = output_dir / filename
    save_fig(fig, output)
    finish(
        records,
        record,
        output,
        [PRETRAIN_MANIFEST],
        "direct",
        key_numbers={row["source"]: row["kept_tokens"] for row in rows},
    )


def plot_funnel(records: list[VisualRecord], output_dir: Path) -> None:
    filename, title, visual_type = REQUESTED_VISUALS[5]
    record = VisualRecord(filename, title, visual_type)
    counts = [(name, count, unit) for name, count, unit in funnel_counts() if count > 0]
    if not counts:
        skip(records, record, "pretraining prepared manifest did not contain funnel counts")
        return
    labels = [name for name, _, _ in counts]
    values = [count for _, count, _ in counts]
    units = [unit for _, _, unit in counts]
    fig, ax = make_fig(figsize=(10.5, 6.2))
    ypos = list(range(len(values)))
    colors = [PALETTE["blue"], PALETTE["teal"], PALETTE["teal"], PALETTE["orange"], PALETTE["navy"], PALETTE["coral"]][: len(values)]
    bars = ax.barh(ypos, values, color=colors, alpha=0.92)
    ax.set_xscale("log")
    ax.set_yticks(ypos)
    ax.set_yticklabels([label.title() for label in labels])
    ax.invert_yaxis()
    ax.set_xlabel("Count (log scale)")
    add_title(ax, title, subtitle_for(filename))
    for y, value, unit in zip(ypos, values, units):
        ax.text(value * 1.08, y, f"{compact_number(value)} {unit}", va="center", fontsize=9.5, color=PALETTE["gray"])
    bars[0].set_alpha(1.0)
    output = output_dir / filename
    save_fig(fig, output)
    finish(
        records,
        record,
        output,
        [PRETRAIN_MANIFEST],
        "direct",
        caveats=["The funnel uses direct artifact counts with mixed units, so the x-axis is logarithmic."],
        key_numbers={label: value for label, value, _ in counts},
    )


def plot_rejection_reasons(records: list[VisualRecord], output_dir: Path) -> None:
    filename, title, visual_type = REQUESTED_VISUALS[6]
    record = VisualRecord(filename, title, visual_type)
    counts = rejection_counts()
    if not counts:
        skip(records, record, "no dropped_reasons or audit rejection reason counts were found")
        return
    items = counts.most_common(12)
    labels = [reason.replace("curated_lm_", "").replace("broad_lm_", "").replace("_", " ").title() for reason, _ in items]
    values = [value for _, value in items]
    fig, ax = make_fig(figsize=(10.6, 6.6))
    bars = ax.barh(range(len(items)), values, color=PALETTE["coral"], alpha=0.9)
    ax.set_yticks(range(len(items)))
    ax.set_yticklabels(labels, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("Rejected document count")
    add_title(ax, title, subtitle_for(filename))
    for index, value in enumerate(values):
        ax.text(value + max(values) * 0.015, index, compact_number(value), va="center", fontsize=9.5, color=PALETTE["gray"])
    if bars:
        bars[0].set_color(PALETTE["orange"])
    output = output_dir / filename
    save_fig(fig, output)
    finish(records, record, output, [PRETRAIN_MANIFEST, *QUALITY_SUMMARIES], "direct", key_numbers=dict(items))


def plot_rag_chunks(records: list[VisualRecord], output_dir: Path) -> None:
    filename, title, visual_type = REQUESTED_VISUALS[7]
    record = VisualRecord(filename, title, visual_type)
    rows, _ = safe_jsonl(RAG_CHUNKS)
    if not rows:
        skip(records, record, "RAG chunks JSONL was missing or empty")
        return
    counts = Counter(str(row.get("source_file") or "unknown") for row in rows)
    categories = Counter(str(row.get("source_category") or "unknown") for row in rows)
    items = counts.most_common()
    fig, ax = make_fig(figsize=(10.6, max(5.5, 1.35 + 0.56 * len(items))))
    labels = [clean_name(source) for source, _ in items]
    values = [count for _, count in items]
    bars = ax.barh(range(len(items)), values, color=PALETTE["blue"], alpha=0.9)
    ax.set_yticks(range(len(items)))
    ax.set_yticklabels(labels, fontsize=9.5)
    ax.invert_yaxis()
    ax.set_xlabel("Chunk count")
    add_title(ax, title, subtitle_for(filename))
    for index, value in enumerate(values):
        ax.text(value + max(values) * 0.015, index, str(value), va="center", fontsize=9.5, color=PALETTE["gray"])
    if bars:
        bars[0].set_color(PALETTE["teal"])
    output = output_dir / filename
    save_fig(fig, output)
    finish(
        records,
        record,
        output,
        [RAG_CHUNKS, RAG_MANIFEST],
        "direct",
        key_numbers={"total_chunks": len(rows), "source_count": len(counts), "source_categories": dict(categories)},
    )


def plot_rag_outcomes(records: list[VisualRecord], output_dir: Path) -> None:
    filename, title, visual_type = REQUESTED_VISUALS[8]
    record = VisualRecord(filename, title, visual_type)
    rows = load_regression(RAG_AFTER)
    if not rows:
        skip(records, record, "final RAG regression artifact was missing or empty")
        return
    counts = rag_outcome_counts(rows)
    categories = ["Generated", "Generated with sources", "Weak generation", "Abstained", "Generation failed"]
    values = [counts[category] for category in categories]
    fig, ax = make_fig(figsize=(10.4, 6.0))
    bars = ax.bar(categories, values, color=[PALETTE["teal"], PALETTE["blue"], PALETTE["orange"], PALETTE["gray"], PALETTE["coral"]], alpha=0.92)
    ax.set_ylabel("Prompt count")
    add_title(ax, title, subtitle_for(filename))
    ax.set_xticks(range(len(categories)))
    ax.set_xticklabels([textwrap.fill(label, 12) for label in categories])
    for index, value in enumerate(values):
        ax.text(index, value + 0.1, str(value), ha="center", fontsize=10, color=PALETTE["gray"])
    if values and max(values) > 0:
        ax.set_ylim(0, max(values) * 1.22)
    for bar in bars:
        bar.set_linewidth(0)
    output = output_dir / filename
    save_fig(fig, output)
    finish(records, record, output, [RAG_AFTER], "derived", caveats=["Outcome categories are non-exclusive counts derived from saved regression fields."], key_numbers=dict(counts))


def prompt_heatmap_values(rows: list[dict[str, Any]]) -> tuple[list[str], list[str], list[list[int]]]:
    columns = ["retrieved source", "generated text", "weak generation", "sources attached", "abstained"]
    values: list[list[int]] = []
    labels: list[str] = []
    for row in rows:
        labels.append(short_label(str(row.get("prompt") or "prompt"), 58))
        values.append(
            [
                int(retrieved_count(row) > 0),
                int(generated_text(row)),
                int(is_weak(row)),
                int(source_attached(row)),
                int(is_abstained(row)),
            ]
        )
    return labels, columns, values


def plot_rag_prompt_heatmap(records: list[VisualRecord], output_dir: Path) -> None:
    filename, title, visual_type = REQUESTED_VISUALS[9]
    record = VisualRecord(filename, title, visual_type)
    rows = load_regression(RAG_AFTER)
    if not rows:
        skip(records, record, "final RAG regression artifact was missing or empty")
        return
    labels, columns, values = prompt_heatmap_values(rows)
    fig, ax = make_fig(figsize=(10.8, max(5.8, 1.45 + len(labels) * 0.48)))
    ax.imshow(values, cmap=editorial_cmap("binary_teal", "teal"), aspect="auto", vmin=0, vmax=1)
    add_title(ax, title, subtitle_for(filename))
    ax.set_xticks(range(len(columns)))
    ax.set_xticklabels([col.title() for col in columns], rotation=24, ha="right")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.grid(False)
    ax.set_xticks([x - 0.5 for x in range(1, len(columns))], minor=True)
    ax.set_yticks([y - 0.5 for y in range(1, len(labels))], minor=True)
    ax.grid(which="minor", color=PALETTE["white"], linestyle="-", linewidth=2.0)
    for y, row_values in enumerate(values):
        for x, value in enumerate(row_values):
            label = "yes" if value else ""
            ax.text(x, y, label, ha="center", va="center", fontsize=9.2, color=PALETTE["navy"])
    output = output_dir / filename
    save_fig(fig, output)
    finish(records, record, output, [RAG_AFTER], "derived", key_numbers={"prompt_count": len(rows), "columns": columns})


def failure_flags(row: dict[str, Any]) -> list[int]:
    text = regression_text(row)
    lower = text.lower()
    reasons = set(row_quality_reasons(row))
    alpha_chars = sum(ch.isalpha() for ch in text)
    nonspace = sum(not ch.isspace() for ch in text)
    return [
        int("repeated_phrase" in reasons or "dominant_repeated_token" in reasons or max_repeated_ngram(text) > 2),
        int("low_prompt_retention" in reasons or "prompt_not_reflected" in reasons),
        int(bool(text.strip()) and alpha_chars == 0),
        int(nonspace > 0 and (alpha_chars / max(nonspace, 1)) < 0.45),
        int("ignores_provided_context" in reasons or "unsupported" in lower),
        int(retrieved_count(row) == 0 and not is_abstained(row)),
        int(is_weak(row)),
    ]


def plot_failure_heatmap(records: list[VisualRecord], output_dir: Path) -> None:
    filename, title, visual_type = REQUESTED_VISUALS[10]
    record = VisualRecord(filename, title, visual_type)
    rows = load_regression(RAG_AFTER)
    if not rows:
        skip(records, record, "final RAG regression artifact was missing or empty")
        return
    columns = ["repetition", "drift", "punctuation-only", "gibberish", "unsupported answer", "no source", "weak generation"]
    labels = [short_label(str(row.get("prompt") or "prompt"), 58) for row in rows]
    values = [failure_flags(row) for row in rows]
    fig, ax = make_fig(figsize=(11.0, max(5.8, 1.45 + len(labels) * 0.48)))
    ax.imshow(values, cmap=editorial_cmap("failures_coral", "coral"), aspect="auto", vmin=0, vmax=1)
    add_title(ax, title, subtitle_for(filename))
    ax.set_xticks(range(len(columns)))
    ax.set_xticklabels([textwrap.fill(col.title(), 12) for col in columns], rotation=24, ha="right")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.grid(False)
    ax.set_xticks([x - 0.5 for x in range(1, len(columns))], minor=True)
    ax.set_yticks([y - 0.5 for y in range(1, len(labels))], minor=True)
    ax.grid(which="minor", color=PALETTE["white"], linestyle="-", linewidth=2.0)
    for y, row_values in enumerate(values):
        for x, value in enumerate(row_values):
            label = "flag" if value else ""
            ax.text(x, y, label, ha="center", va="center", fontsize=8.8, color=PALETTE["navy"])
    output = output_dir / filename
    save_fig(fig, output)
    finish(records, record, output, [RAG_AFTER], "heuristic", caveats=["Failure types combine saved quality reasons with deterministic text heuristics."], key_numbers={"prompt_count": len(rows), "columns": columns})


def plot_generation_lengths(records: list[VisualRecord], output_dir: Path) -> None:
    filename, title, visual_type = REQUESTED_VISUALS[11]
    record = VisualRecord(filename, title, visual_type)
    cases = collect_output_cases()
    lengths = [word_count(case["text"]) for case in cases if case.get("text")]
    if not lengths:
        skip(records, record, "no saved generation outputs were found")
        return
    fig, ax = make_fig(figsize=(10.2, 6.0))
    bins = min(24, max(8, int(math.sqrt(len(lengths)))))
    ax.hist(lengths, bins=bins, color=PALETTE["blue"], edgecolor="white", linewidth=1.2, alpha=0.92)
    median = sorted(lengths)[len(lengths) // 2]
    ax.axvline(median, color=PALETTE["orange"], linewidth=2.0)
    add_title(ax, title, subtitle_for(filename))
    ax.set_xlabel("Word count")
    ax.set_ylabel("Saved outputs")
    add_callout(ax, f"Median length: {median} words", (median, ax.get_ylim()[1] * 0.72), (24, 0), color="orange", arrow=False)
    output = output_dir / filename
    save_fig(fig, output)
    finish(
        records,
        record,
        output,
        [RAG_AFTER, RAG_BEFORE, RAG_SERVING, MANUAL_DEMOS, *COMPARISON_FILES],
        "derived",
        key_numbers={"samples": len(lengths), "median_words": sorted(lengths)[len(lengths) // 2], "max_words": max(lengths)},
    )


def plot_repetition_distribution(records: list[VisualRecord], output_dir: Path) -> None:
    filename, title, visual_type = REQUESTED_VISUALS[12]
    record = VisualRecord(filename, title, visual_type)
    cases = collect_output_cases()
    scores = [repetition_score(case["text"], case.get("metrics")) for case in cases if case.get("text")]
    if not scores:
        skip(records, record, "no saved generation outputs were found")
        return
    fig, ax = make_fig(figsize=(10.2, 6.0))
    ax.hist(scores, bins=18, range=(0, 1), color=PALETTE["teal"], edgecolor="white", linewidth=1.2, alpha=0.92)
    mean_score = sum(scores) / len(scores)
    ax.axvline(mean_score, color=PALETTE["orange"], linewidth=2.0)
    add_title(ax, title, subtitle_for(filename))
    ax.set_xlabel("Heuristic repetition score (lower is better)")
    ax.set_ylabel("Saved outputs")
    add_callout(ax, f"Mean score: {mean_score:.2f}", (mean_score, ax.get_ylim()[1] * 0.72), (24, 0), color="orange", arrow=False)
    output = output_dir / filename
    save_fig(fig, output)
    finish(
        records,
        record,
        output,
        [RAG_AFTER, RAG_BEFORE, RAG_SERVING, MANUAL_DEMOS, *COMPARISON_FILES],
        "heuristic",
        caveats=["Score uses saved unique_token_ratio when present, otherwise token dominance and repeated 4-grams."],
        key_numbers={"samples": len(scores), "mean_score": round(sum(scores) / len(scores), 4)},
    )


def plot_retrieval_scores(records: list[VisualRecord], output_dir: Path) -> None:
    filename, title, visual_type = REQUESTED_VISUALS[13]
    record = VisualRecord(filename, title, visual_type)
    values = [score for _, _, score, _ in retrieval_scores()]
    if not values:
        skip(records, record, "no retrieval hit scores were found in RAG artifacts")
        return
    fig, ax = make_fig(figsize=(10.2, 6.0))
    ax.hist(values, bins=16, color=PALETTE["teal"], edgecolor="white", linewidth=1.2, alpha=0.92)
    add_title(ax, title, subtitle_for(filename))
    ax.set_xlabel("Retrieval score")
    ax.set_ylabel("Hits")
    add_callout(ax, f"Max score: {max(values):.2f}", (max(values), ax.get_ylim()[1] * 0.65), (-108, 0), color="teal", arrow=False)
    output = output_dir / filename
    save_fig(fig, output)
    finish(
        records,
        record,
        output,
        [RAG_QUERY_CHECKS, RAG_AFTER, RAG_BEFORE, RAG_SERVING],
        "direct",
        key_numbers={"hit_count": len(values), "max_score": round(max(values), 6), "min_score": round(min(values), 6)},
    )


def query_source_matrix() -> tuple[list[str], list[str], list[list[float]], list[Path]]:
    scores = retrieval_scores()
    query_order: list[str] = []
    source_order: list[str] = []
    matrix_map: dict[tuple[str, str], float] = defaultdict(float)
    sources_used: set[Path] = set()
    for query, source, score, path in scores:
        if query not in query_order:
            query_order.append(query)
        if source not in source_order:
            source_order.append(source)
        matrix_map[(query, source)] = max(matrix_map[(query, source)], score)
        sources_used.add(path)
    source_order.sort(key=lambda source: clean_name(source))
    matrix = [[matrix_map[(query, source)] for source in source_order] for query in query_order]
    return query_order, source_order, matrix, sorted(sources_used)


def plot_query_source_heatmap(records: list[VisualRecord], output_dir: Path) -> None:
    filename, title, visual_type = REQUESTED_VISUALS[14]
    record = VisualRecord(filename, title, visual_type)
    queries, sources, matrix, used = query_source_matrix()
    if not matrix:
        skip(records, record, "no query-source retrieval score matrix could be built")
        return
    fig, ax = make_fig(figsize=(11.2, max(6.0, 1.35 + len(queries) * 0.48)))
    im = ax.imshow(matrix, cmap=editorial_cmap("retrieval_blue", "blue"), aspect="auto")
    add_title(ax, title, subtitle_for(filename))
    ax.set_xticks(range(len(sources)))
    ax.set_xticklabels([clean_name(source) for source in sources], rotation=32, ha="right", fontsize=9)
    ax.set_yticks(range(len(queries)))
    ax.set_yticklabels([short_label(query, 52) for query in queries], fontsize=9)
    ax.grid(False)
    ax.set_xticks([x - 0.5 for x in range(1, len(sources))], minor=True)
    ax.set_yticks([y - 0.5 for y in range(1, len(queries))], minor=True)
    ax.grid(which="minor", color=PALETTE["white"], linestyle="-", linewidth=1.8)
    cbar = fig.colorbar(im, ax=ax, shrink=0.78, label="Max retrieval score")
    cbar.outline.set_visible(False)
    output = output_dir / filename
    save_fig(fig, output)
    finish(records, record, output, used, "direct", key_numbers={"query_count": len(queries), "source_count": len(sources)})


def plot_rag_before_after(records: list[VisualRecord], output_dir: Path) -> None:
    filename, title, visual_type = REQUESTED_VISUALS[15]
    record = VisualRecord(filename, title, visual_type)
    before = load_regression(RAG_BEFORE)
    after = load_regression(RAG_AFTER)
    if not before or not after:
        skip(records, record, "before and after RAG regression artifacts were both required")
        return
    metrics = ["bad answers", "abstentions", "source surfaced", "weak flagged", "grounded support"]
    def metric_values(rows: list[dict[str, Any]]) -> list[int]:
        bad = sum(int(is_weak(row) and not is_abstained(row)) for row in rows)
        abstentions = sum(int(is_abstained(row)) for row in rows)
        source_surfaced = sum(int(source_attached(row)) for row in rows)
        weak = sum(int(is_weak(row)) for row in rows)
        grounded = sum(int(retrieved_count(row) > 0 or is_abstained(row)) for row in rows)
        return [bad, abstentions, source_surfaced, weak, grounded]
    before_values = metric_values(before)
    after_values = metric_values(after)
    x = range(len(metrics))
    width = 0.36
    fig, ax = make_fig(figsize=(10.6, 6.0))
    before_bars = ax.bar([i - width / 2 for i in x], before_values, width=width, color=PALETTE["mid_gray"], label="Before source expansion")
    after_bars = ax.bar([i + width / 2 for i in x], after_values, width=width, color=PALETTE["blue"], label="After source expansion")
    ax.set_xticks(list(x))
    ax.set_xticklabels([textwrap.fill(metric.title(), 12) for metric in metrics])
    ax.set_ylabel("Prompt count")
    add_title(ax, title, subtitle_for(filename))
    ax.legend(frameon=False, ncols=2, loc="upper right")
    for bars in (before_bars, after_bars):
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.08, f"{bar.get_height():.0f}", ha="center", fontsize=9.5, color=PALETTE["gray"])
    ax.set_ylim(0, max(before_values + after_values) * 1.25 if before_values + after_values else 1)
    output = output_dir / filename
    save_fig(fig, output)
    finish(records, record, output, [RAG_BEFORE, RAG_AFTER], "derived", caveats=["Metrics are non-exclusive counts derived from two saved RAG regression runs."], key_numbers={"before": dict(zip(metrics, before_values)), "after": dict(zip(metrics, after_values))})


def radar_scores() -> dict[str, float]:
    rows = load_regression(RAG_AFTER)
    if not rows:
        return {}
    total = len(rows)
    reasons = [reason for row in rows for reason in row_quality_reasons(row)]
    no_hit_rows = [row for row in rows if retrieved_count(row) == 0]
    hit_rows = [row for row in rows if retrieved_count(row) > 0]
    return {
        "fluency": max(0.0, 1.0 - sum(int(is_weak(row)) for row in rows) / total),
        "prompt retention": max(0.0, 1.0 - sum(reason in {"low_prompt_retention", "prompt_not_reflected"} for reason in reasons) / total),
        "grounding support": sum(int(retrieved_count(row) > 0 or is_abstained(row)) for row in rows) / total,
        "abstention honesty": (sum(int(is_abstained(row)) for row in no_hit_rows) / len(no_hit_rows)) if no_hit_rows else 0.0,
        "repetition control": max(0.0, 1.0 - sum(reason in {"repeated_phrase", "dominant_repeated_token"} for reason in reasons) / total),
        "source surfacing": (sum(int(source_attached(row)) for row in hit_rows) / len(hit_rows)) if hit_rows else 0.0,
    }


def plot_quality_radar(records: list[VisualRecord], output_dir: Path) -> None:
    filename, title, visual_type = REQUESTED_VISUALS[16]
    record = VisualRecord(filename, title, visual_type)
    scores = radar_scores()
    if not scores:
        skip(records, record, "final RAG regression artifact was missing or empty")
        return
    labels = list(scores)
    values = [scores[label] for label in labels]
    angles = [2 * math.pi * i / len(labels) for i in range(len(labels))]
    values_closed = values + values[:1]
    angles_closed = angles + angles[:1]
    fig, ax = make_fig(figsize=(8.2, 7.0), polar=True)
    ax.plot(angles_closed, values_closed, color=PALETTE["blue"], linewidth=2.8)
    ax.fill(angles_closed, values_closed, color=PALETTE["pale_blue"], alpha=0.82)
    ax.set_xticks(angles)
    ax.set_xticklabels([textwrap.fill(label.title(), 12) for label in labels], fontsize=10)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["25%", "50%", "75%", "100%"], color=PALETTE["gray"], fontsize=8.5)
    ax.set_ylim(0, 1)
    ax.spines["polar"].set_color(PALETTE["mid_gray"])
    add_title(ax, title, subtitle_for(filename))
    output = output_dir / filename
    save_fig(fig, output)
    finish(records, record, output, [RAG_AFTER], "heuristic", caveats=["Scores summarize saved regression metadata with a deterministic rubric, not human grading."], key_numbers={key: round(value, 4) for key, value in scores.items()})


def readiness_checks(output_dir: Path) -> list[tuple[str, float, str, list[Path]]]:
    serving, _ = safe_json(RAG_SERVING)
    serving_status = serving.get("status") if isinstance(serving, dict) and isinstance(serving.get("status"), dict) else {}
    app_text = read_text(SERVE_APP)
    playground_text = read_text(PLAYGROUND)
    lastfailed, _ = safe_json(PYTEST_LASTFAILED)
    rows_after = load_regression(RAG_AFTER)
    hit_rows = [row for row in rows_after if retrieved_count(row) > 0]
    return [
        ("checkpoint artifact present", float((ROOT / "artifacts/runs/local-mvp/checkpoints/pretrain/best-pretrain/checkpoint.pt").exists()), "best-pretrain checkpoint file exists", [ROOT / "artifacts/runs/local-mvp/checkpoints/pretrain/best-pretrain/checkpoint.pt"]),
        ("server verification ok", float(serving_status.get("status") == "ok"), "saved serving verification status", [RAG_SERVING]),
        ("RAG enabled", float(bool(serving_status.get("rag", {}).get("enabled"))), "saved serving metadata", [RAG_SERVING]),
        ("sources surfaced", float(bool(hit_rows) and all(source_attached(row) for row in hit_rows)), "RAG rows with hits include source metadata", [RAG_AFTER]),
        ("streaming route present", float("/generate_stream" in app_text and "true_model_token_streaming" in app_text), "static FastAPI route check", [SERVE_APP]),
        ("UI source cards present", float("source-card" in playground_text and "Sources available" in playground_text and "Run details" in playground_text), "static playground UI check", [PLAYGROUND]),
        ("documentation generated", float((ROOT / "tools/build_documentation_graphs.py").exists() and output_dir.exists()), "this generator writes the requested output set in one run", [ROOT / "tools/build_documentation_graphs.py", output_dir]),
        ("core tests last run clean", float(isinstance(lastfailed, dict) and not lastfailed), "pytest cache lastfailed is empty", [PYTEST_LASTFAILED]),
    ]


def plot_readiness(records: list[VisualRecord], output_dir: Path) -> None:
    filename, title, visual_type = REQUESTED_VISUALS[17]
    record = VisualRecord(filename, title, visual_type)
    checks = readiness_checks(output_dir)
    labels = [item[0] for item in checks]
    values = [item[1] for item in checks]
    fig, ax = make_fig(figsize=(10.7, 6.2))
    colors = [PALETTE["teal"] if value >= 0.99 else PALETTE["orange"] if value > 0 else PALETTE["coral"] for value in values]
    bars = ax.barh(range(len(checks)), values, color=colors, alpha=0.94)
    ax.set_xlim(0, 1)
    ax.set_yticks(range(len(checks)))
    ax.set_yticklabels([label.title() for label in labels], fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("Readiness score")
    add_title(ax, title, subtitle_for(filename))
    ax.set_xticks([0, 0.5, 1.0])
    ax.set_xticklabels(["0%", "50%", "100%"])
    for index, value in enumerate(values):
        label = "PASS" if value >= 0.99 else f"{value:.0%}"
        ax.text(min(value + 0.035, 0.91), index, label, va="center", fontsize=9.5, color=PALETTE["gray"])
    for bar in bars:
        bar.set_linewidth(0)
    output = output_dir / filename
    save_fig(fig, output)
    finish(records, record, output, [source for *_, sources in checks for source in sources], "derived", caveats=["Core tests status is read from pytest's lastfailed cache; the generator does not run the full test suite."], key_numbers={label: value for label, value in zip(labels, values)})


def save_anim(anim: animation.FuncAnimation, output: Path, fps: int) -> None:
    polish_axes(anim._fig)  # noqa: SLF001
    writer = animation.PillowWriter(fps=fps)
    anim.save(output, writer=writer, dpi=115)
    plt.close(anim._fig)  # noqa: SLF001


def gif_possible() -> tuple[bool, str | None]:
    try:
        import PIL  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        return False, f"Pillow is unavailable: {exc}"
    return True, None


def animate_curve(
    records: list[VisualRecord],
    output_dir: Path,
    visual_index: int,
    y_key: str,
    color: str,
    ylabel: str,
    no_gifs: bool,
) -> None:
    filename, title, visual_type = REQUESTED_VISUALS[visual_index]
    record = VisualRecord(filename, title, visual_type)
    if no_gifs:
        skip(records, record, "GIF generation disabled by --no-gifs")
        return
    ok, reason = gif_possible()
    if not ok:
        skip(records, record, reason or "GIF writer unavailable")
        return
    series = load_pretrain_series()
    points = [(row["step"], row[y_key]) for row in series if y_key in row]
    if len(points) < 2:
        skip(records, record, f"not enough {y_key} points for animation")
        return
    steps, values_raw = zip(*points)
    values = [min(value, 1000.0) if y_key == "perplexity" else value for value in values_raw]
    fig, ax = make_fig(figsize=(8.2, 4.9))
    add_title(ax, title, subtitle_for(filename))
    ax.set_xlabel("Training step")
    ax.set_ylabel(ylabel)
    ax.set_xlim(min(steps), max(steps))
    pad = (max(values) - min(values)) * 0.12 or 1.0
    ax.set_ylim(min(values) - pad, max(values) + pad)
    (line,) = ax.plot([], [], color=color, linewidth=2.8)
    marker = ax.scatter([], [], color=PALETTE["teal"], edgecolor="white", linewidth=1.0, s=56, zorder=3)
    label = ax.text(
        0.98,
        0.08,
        "",
        transform=ax.transAxes,
        ha="right",
        color=PALETTE["navy"],
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.35,rounding_size=0.12", "facecolor": PALETTE["pale_blue"], "edgecolor": PALETTE["mid_gray"]},
    )
    frames = min(45, max(12, len(points)))

    def update(frame: int) -> tuple[Any, ...]:
        end = max(2, int(round((frame + 1) / frames * len(points))))
        line.set_data(steps[:end], values[:end])
        marker.set_offsets([[steps[end - 1], values[end - 1]]])
        label.set_text(f"{ylabel.title()}: {values_raw[end - 1]:.3g}")
        return line, marker, label

    anim = animation.FuncAnimation(fig, update, frames=frames, blit=False, repeat=True)
    output = output_dir / filename
    save_anim(anim, output, fps=4)
    caveat = ["Perplexity is clipped at 1000 for readability."] if y_key == "perplexity" else []
    finish(records, record, output, [PRETRAIN_HISTORY], "direct", caveats=caveat, key_numbers={"frames": frames, "points": len(points)})


def animate_retrieval_heatmap(records: list[VisualRecord], output_dir: Path, no_gifs: bool) -> None:
    filename, title, visual_type = REQUESTED_VISUALS[20]
    record = VisualRecord(filename, title, visual_type)
    if no_gifs:
        skip(records, record, "GIF generation disabled by --no-gifs")
        return
    ok, reason = gif_possible()
    if not ok:
        skip(records, record, reason or "GIF writer unavailable")
        return
    queries, sources, matrix, used = query_source_matrix()
    if not matrix:
        skip(records, record, "no query-source retrieval score matrix could be built")
        return
    fig, ax = make_fig(figsize=(9.0, 5.8))
    max_value = max(max(row) for row in matrix) or 1.0

    def update(frame: int) -> list[Any]:
        ax.clear()
        visible = [row[:] for row in matrix]
        for y in range(frame + 1, len(visible)):
            visible[y] = [0.0] * len(sources)
        im = ax.imshow(visible, cmap=editorial_cmap("retrieval_animation_blue", "blue"), aspect="auto", vmin=0, vmax=max_value)
        add_title(ax, title, subtitle_for(filename))
        ax.set_xticks(range(len(sources)))
        ax.set_xticklabels([clean_name(source) for source in sources], rotation=32, ha="right", fontsize=8)
        ax.set_yticks(range(len(queries)))
        ax.set_yticklabels([short_label(query, 42) for query in queries], fontsize=8)
        ax.set_xticks([x - 0.5 for x in range(1, len(sources))], minor=True)
        ax.set_yticks([y - 0.5 for y in range(1, len(queries))], minor=True)
        ax.grid(False)
        ax.grid(which="minor", color=PALETTE["white"], linewidth=1.6)
        polish_axes(fig)
        return [im]

    anim = animation.FuncAnimation(fig, update, frames=len(queries), blit=False, repeat=True)
    output = output_dir / filename
    save_anim(anim, output, fps=2)
    finish(records, record, output, used, "direct", key_numbers={"frames": len(queries), "query_count": len(queries)})


def animate_streaming_demo(records: list[VisualRecord], output_dir: Path, no_gifs: bool) -> None:
    filename, title, visual_type = REQUESTED_VISUALS[21]
    record = VisualRecord(filename, title, visual_type)
    if no_gifs:
        skip(records, record, "GIF generation disabled by --no-gifs")
        return
    ok, reason = gif_possible()
    if not ok:
        skip(records, record, reason or "GIF writer unavailable")
        return
    rows, _ = safe_jsonl(MANUAL_DEMOS)
    chosen = ""
    for row in reversed(rows):
        if row.get("api_route") == "/generate_stream":
            response = row.get("response") if isinstance(row.get("response"), dict) else {}
            text = str(response.get("text") or "")
            if word_count(text) >= 10:
                chosen = text
                break
    sample_text = "This is generated by the local-MVP model. Sources are available below when RAG finds relevant context."
    if not chosen:
        chosen = sample_text
    chosen = re.sub(r"^[^A-Za-z0-9]+", "", chosen).strip()
    sample_text_used = word_count(chosen) < 10 or not chosen[:1].isupper()
    if sample_text_used:
        chosen = sample_text
    words = chosen.split()
    frames = min(len(words), 42)
    fig, ax = plt.subplots(figsize=(8.2, 4.9), constrained_layout=True)
    fig.patch.set_facecolor(PALETTE["white"])

    def update(frame: int) -> list[Any]:
        ax.clear()
        ax.set_axis_off()
        ax.set_facecolor(PALETTE["white"])
        ax.text(0.05, 0.93, "WebbGPT 0.2", fontsize=15, weight="bold", ha="left", color=PALETTE["navy"])
        ax.text(0.95, 0.93, "Progressive rendering", fontsize=10.5, ha="right", color=PALETTE["teal"])
        ax.add_patch(
            FancyBboxPatch(
                (0.05, 0.16),
                0.90,
                0.68,
                boxstyle="round,pad=0.018,rounding_size=0.025",
                facecolor=PALETTE["white"],
                edgecolor=PALETTE["mid_gray"],
                linewidth=1.1,
            )
        )
        ax.add_patch(
            FancyBboxPatch(
                (0.08, 0.68),
                0.16,
                0.055,
                boxstyle="round,pad=0.01,rounding_size=0.018",
                facecolor=PALETTE["pale_teal"],
                edgecolor=PALETTE["teal"],
                linewidth=0.8,
            )
        )
        ax.text(0.16, 0.708, "Answer", fontsize=9.3, ha="center", va="center", color=PALETTE["navy"])
        visible_count = max(1, int(round((frame + 1) / frames * len(words))))
        wrapped = "\n".join(textwrap.wrap(" ".join(words[:visible_count]), width=70))
        ax.text(0.09, 0.64, wrapped, fontsize=11.2, ha="left", va="top", color=PALETTE["navy"], linespacing=1.38)
        ax.text(0.09, 0.25, "Sources available below, collapsed by default", fontsize=9.2, color=PALETTE["gray"])
        ax.text(
            0.09,
            0.19,
            "UI-level progressive rendering, not true model-token streaming.",
            fontsize=8.8,
            color=PALETTE["gray"],
        )
        return []

    anim = animation.FuncAnimation(fig, update, frames=frames, blit=False, repeat=True)
    output = output_dir / filename
    save_anim(anim, output, fps=5)
    caveat = "The GIF simulates UI progressive reveal; it is not a live browser capture."
    if sample_text_used:
        caveat += " It uses the documented sample sentence because the saved response text was not presentation-friendly."
    else:
        caveat += " It reveals saved response text from the manual demo transcript."
    finish(records, record, output, [MANUAL_DEMOS, SERVE_APP, PLAYGROUND], "derived", caveats=[caveat], key_numbers={"frames": frames, "sample_text_used": sample_text_used})


def animate_funnel(records: list[VisualRecord], output_dir: Path, no_gifs: bool) -> None:
    filename, title, visual_type = REQUESTED_VISUALS[22]
    record = VisualRecord(filename, title, visual_type)
    if no_gifs:
        skip(records, record, "GIF generation disabled by --no-gifs")
        return
    ok, reason = gif_possible()
    if not ok:
        skip(records, record, reason or "GIF writer unavailable")
        return
    counts = [(name, count, unit) for name, count, unit in funnel_counts() if count > 0]
    if not counts:
        skip(records, record, "pretraining prepared manifest did not contain funnel counts")
        return
    max_value = max(count for _, count, _ in counts)
    frames = len(counts)
    fig, ax = make_fig(figsize=(8.8, 5.2))

    def update(frame: int) -> list[Any]:
        ax.clear()
        shown = counts[: frame + 1]
        labels = [item[0] for item in shown]
        values = [item[1] for item in shown]
        units = [item[2] for item in shown]
        colors = [PALETTE["blue"], PALETTE["teal"], PALETTE["teal"], PALETTE["orange"], PALETTE["navy"], PALETTE["coral"]][: len(shown)]
        ax.barh(range(len(shown)), values, color=colors, alpha=0.93)
        ax.set_xscale("log")
        ax.set_xlim(1, max_value * 1.35)
        ax.set_yticks(range(len(shown)))
        ax.set_yticklabels([label.title() for label in labels])
        ax.invert_yaxis()
        add_title(ax, title, subtitle_for(filename))
        ax.set_xlabel("Count (log scale)")
        for y, value, unit in zip(range(len(shown)), values, units):
            ax.text(value * 1.08, y, f"{compact_number(value)} {unit}", va="center", fontsize=9, color=PALETTE["gray"])
        polish_axes(fig)
        return []

    anim = animation.FuncAnimation(fig, update, frames=frames, blit=False, repeat=True)
    output = output_dir / filename
    save_anim(anim, output, fps=1)
    finish(records, record, output, [PRETRAIN_MANIFEST], "direct", caveats=["The funnel uses direct artifact counts with mixed units, so the x-axis is logarithmic."], key_numbers={"frames": frames})


def thumbnail_array(path: Path) -> Any | None:
    try:
        if path.suffix.lower() == ".gif":
            from PIL import Image

            with Image.open(path) as image:
                image.seek(0)
                return np.asarray(image.convert("RGB"))
        return plt.imread(path)
    except Exception:  # noqa: BLE001 - contact sheet should not block graph generation.
        return None


def plot_contact_sheet(records: list[VisualRecord], output_dir: Path) -> None:
    filename, title, visual_type = CONTACT_SHEET
    record = VisualRecord(filename, title, visual_type)
    generated = [record for record in records if record.generated and record.filepath and record.filename != filename]
    if not generated:
        skip(records, record, "no generated visuals were available for the contact sheet")
        return
    generated.sort(key=lambda item: item.filename)
    ncols = 4
    nrows = math.ceil(len(generated) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(15.5, max(6.0, nrows * 3.35)), constrained_layout=False)
    fig.patch.set_facecolor(PALETTE["white"])
    fig.subplots_adjust(left=0.035, right=0.985, bottom=0.035, top=0.905, wspace=0.18, hspace=0.42)
    axes_list = list(np.ravel(axes))
    fig.suptitle(title, x=0.035, y=0.985, ha="left", fontsize=23, fontweight="bold", color=PALETTE["navy"])
    fig.text(0.035, 0.958, subtitle_for(filename), ha="left", va="top", fontsize=12.5, color=PALETTE["gray"])
    for ax, item in zip(axes_list, generated):
        ax.set_axis_off()
        path = output_dir / item.filename
        image = thumbnail_array(path)
        ax.add_patch(
            FancyBboxPatch(
                (0.0, 0.0),
                1.0,
                1.0,
                transform=ax.transAxes,
                boxstyle="round,pad=0.015,rounding_size=0.025",
                facecolor=PALETTE["white"],
                edgecolor=PALETTE["mid_gray"],
                linewidth=0.9,
                zorder=0,
            )
        )
        if image is not None:
            ax.imshow(image, extent=(0.035, 0.965, 0.22, 0.92), aspect="auto", zorder=1)
        ax.text(0.04, 0.14, item.filename, transform=ax.transAxes, fontsize=9.2, color=PALETTE["blue"], weight="bold")
        ax.text(
            0.04,
            0.065,
            short_label(item.title, 46),
            transform=ax.transAxes,
            fontsize=8.7,
            color=PALETTE["gray"],
        )
    for ax in axes_list[len(generated) :]:
        ax.set_axis_off()
    output = output_dir / filename
    save_fig(fig, output)
    finish(
        records,
        record,
        output,
        [output_dir / item.filename for item in generated],
        "derived",
        caveats=["Supplemental contact sheet summarizing the generated documentation graphics set."],
        key_numbers={"indexed_visuals": len(generated)},
    )


def write_manifest(records: list[VisualRecord], output_dir: Path, cleaned: list[str]) -> None:
    payload = {
        "generated_at": dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "main_command": "python3.12 tools/build_documentation_graphs.py",
        "requested_visual_count": len(REQUESTED_VISUALS),
        "supplemental_visual_count": sum(1 for record in records if record.filename == CONTACT_SHEET[0]),
        "generated_visual_count": sum(1 for record in records if record.generated),
        "cleaned_stale_images": cleaned,
        "style_system": {
            "name": "Clean editorial science-explainer",
            "background": "white",
            "palette": {key: value for key, value in PALETTE.items() if key in {"navy", "blue", "teal", "orange", "coral", "gray", "light_gray", "mid_gray"}},
            "notes": "Presentation-oriented matplotlib styling with larger titles, human-facing subtitles, light gridlines, soft axes, and padded saves.",
        },
        "visuals": [
            {
                "filename": record.filename,
                "title": record.title,
                "type": record.visual_type,
                "generated": record.generated,
                "filepath": record.filepath,
                "data_sources": record.data_sources,
                "grounding": record.grounding,
                "caveats": record.caveats,
                "skipped_reason": record.skipped_reason,
                "key_numbers": record.key_numbers,
            }
            for record in sorted(records, key=lambda item: item.filename)
        ],
    }
    (output_dir / "visual_manifest.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_readme(records: list[VisualRecord], output_dir: Path) -> None:
    lines = [
        "# WebbGPT Documentation Graphics",
        "",
        "This folder contains code-generated documentation visuals for the WebbGPT final presentation/poster. The graphics are regenerated from local repo artifacts, logs, JSON/JSONL outputs, prepared-data manifests, RAG assets, saved regressions, serving verification, and static route/UI evidence. The generator does not start training and does not modify model checkpoints.",
        "",
        "The current visual style is a clean editorial science-explainer system: white backgrounds, generous padding, larger human-facing titles, short explanatory subtitles, soft navy/teal/orange/coral accents, subtle gridlines, and lightweight callouts for the main takeaway. The chart wording is intended for a lay presentation audience while preserving the underlying artifact-derived values.",
        "",
        "## Regenerate",
        "",
        "```bash",
        "python3.12 tools/build_documentation_graphs.py",
        "```",
        "",
        "Optional flags: `--output-dir documentation`, `--no-gifs`, `--verbose`, `--keep-stale`.",
        "",
        "## Outputs",
        "",
    ]
    for record in sorted(records, key=lambda item: item.filename):
        status = "generated" if record.generated else f"skipped: {record.skipped_reason}"
        sources = ", ".join(record.data_sources) if record.data_sources else "none"
        caveats = " ".join(record.caveats) if record.caveats else "None."
        lines.append(f"- `{record.filename}` - {status}. Grounding: {record.grounding}. Sources: {sources}. Caveats: {caveats}")
    lines.extend(
        [
            "",
            "## Grounding Labels",
            "",
            "- `direct`: plotted directly from artifact values.",
            "- `derived`: computed from explicit saved artifact fields without model calls.",
            "- `heuristic`: computed with a deterministic rubric over saved metadata/text and labeled as heuristic in the visual title or caveat.",
        ]
    )
    (output_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def clean_stale_images(output_dir: Path, keep_stale: bool) -> list[str]:
    if keep_stale or not output_dir.exists():
        return []
    cleaned: list[str] = []
    for path in output_dir.iterdir():
        if path.is_file() and path.suffix.lower() in {".png", ".gif"} and path.name not in REQUESTED_FILENAMES:
            path.unlink()
            cleaned.append(rel(path))
    return sorted(cleaned)


def build_all(output_dir: Path, no_gifs: bool, keep_stale: bool, verbose: bool) -> list[VisualRecord]:
    setup_style()
    output_dir.mkdir(parents=True, exist_ok=True)
    cleaned = clean_stale_images(output_dir, keep_stale)
    records: list[VisualRecord] = []
    plot_loss(records, output_dir)
    plot_perplexity(records, output_dir)
    plot_dual_axis(records, output_dir)
    plot_timeline(records, output_dir)
    plot_source_mix(records, output_dir)
    plot_funnel(records, output_dir)
    plot_rejection_reasons(records, output_dir)
    plot_rag_chunks(records, output_dir)
    plot_rag_outcomes(records, output_dir)
    plot_rag_prompt_heatmap(records, output_dir)
    plot_failure_heatmap(records, output_dir)
    plot_generation_lengths(records, output_dir)
    plot_repetition_distribution(records, output_dir)
    plot_retrieval_scores(records, output_dir)
    plot_query_source_heatmap(records, output_dir)
    plot_rag_before_after(records, output_dir)
    plot_quality_radar(records, output_dir)
    plot_readiness(records, output_dir)
    animate_curve(records, output_dir, 18, "loss", PALETTE["blue"], "validation loss", no_gifs)
    animate_curve(records, output_dir, 19, "perplexity", PALETTE["teal"], "perplexity", no_gifs)
    animate_retrieval_heatmap(records, output_dir, no_gifs)
    animate_streaming_demo(records, output_dir, no_gifs)
    animate_funnel(records, output_dir, no_gifs)
    plot_contact_sheet(records, output_dir)
    write_manifest(records, output_dir, cleaned)
    write_readme(records, output_dir)
    if verbose:
        requested_records = [record for record in records if record.filename != CONTACT_SHEET[0]]
        supplemental_records = [record for record in records if record.filename == CONTACT_SHEET[0]]
        print(
            f"generated {sum(1 for record in requested_records if record.generated)} / {len(REQUESTED_VISUALS)} requested visuals; "
            f"supplemental generated: {sum(1 for record in supplemental_records if record.generated)} / {len(supplemental_records)}"
        )
        if cleaned:
            print("cleaned stale images:", cleaned)
        skipped = [record.filename for record in records if not record.generated]
        print("skipped:", skipped)
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate WebbGPT documentation graphs.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Output directory for generated documentation graphics.")
    parser.add_argument("--no-gifs", action="store_true", help="Skip GIF generation.")
    parser.add_argument("--verbose", action="store_true", help="Print generation summary.")
    parser.add_argument("--keep-stale", action="store_true", help="Do not remove old PNG/GIF files outside the requested 23-file set.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    build_all(Path(args.output_dir), no_gifs=args.no_gifs, keep_stale=args.keep_stale, verbose=args.verbose)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
