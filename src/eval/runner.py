from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from config import DataConfig, EvalConfig, ModelConfig
from data.dataset import DatasetBuilder
from eval.assistant import run_assistant_benchmark
from eval.catalog import evaluate_catalog_benchmark
from eval.perplexity import evaluate_perplexity
from eval.webb import evaluate_webb_benchmark
from grounding.store import WebbKnowledgeStore
from model.transformer import CausalTransformer
from provenance import (
    benchmark_manifest,
    catalog_snapshot_manifest,
    checkpoint_manifest,
    grounding_snapshot_manifest,
    release_gate_manifest,
    reliability_payload,
    scorer_manifest,
    tokenizer_manifest,
)
from repro import seed_everything
from torch_runtime import get_torch_device


def _require_torch():
    import torch

    return torch


def _load_checkpoint(model, checkpoint_path: str) -> None:
    torch = _require_torch()
    payload = torch.load(Path(checkpoint_path) / "checkpoint.pt", map_location="cpu")
    model.load_state_dict(payload["model"], strict=True)


def _summarize_assistant_results(results: list[dict]) -> dict[str, float]:
    total_examples = sum(int(result["examples"]) for result in results)
    if total_examples <= 0:
        return {
            "examples": 0,
            "average_score": 0.0,
            "pass_rate": 0.0,
            "exact_match_rate": 0.0,
            "citation_rate": 0.0,
        }
    weighted_average_score = sum(result["average_score"] * result["examples"] for result in results) / total_examples
    weighted_pass_rate = sum(result["pass_rate"] * result["examples"] for result in results) / total_examples
    weighted_exact = sum(result["exact_match_rate"] * result["examples"] for result in results) / total_examples
    weighted_citation = sum(result["citation_rate"] * result["examples"] for result in results) / total_examples
    return {
        "examples": total_examples,
        "average_score": weighted_average_score,
        "pass_rate": weighted_pass_rate,
        "exact_match_rate": weighted_exact,
        "citation_rate": weighted_citation,
    }


def _assistant_subset(results: list[dict], *, name_contains: str) -> dict[str, float]:
    subset = [result for result in results if name_contains in Path(result["path"]).stem]
    return _summarize_assistant_results(subset)


def _catalog_subset(results: list[dict], *, name_contains: str, invert: bool = False) -> list[dict]:
    subset = []
    for result in results:
        stem = Path(result["path"]).stem
        matches = name_contains in stem
        if invert:
            matches = not matches
        if matches:
            subset.append(result)
    return subset


def _average_metric(results: list[dict], metric: str) -> float:
    total_examples = sum(int(result["examples"]) for result in results)
    if total_examples <= 0:
        return 0.0
    return sum(float(result[metric]) * int(result["examples"]) for result in results) / total_examples


def _build_release_gates(
    eval_config: EvalConfig,
    assistant_results: list[dict],
    catalog_results: list[dict],
    webb_results: list[dict] | None = None,
    grounding_quality: dict[str, object] | None = None,
) -> dict[str, object]:
    chat_sanity = _assistant_subset(assistant_results, name_contains="chat_sanity")
    normal_catalog = _catalog_subset(catalog_results, name_contains="missing", invert=True)
    missing_catalog = _catalog_subset(catalog_results, name_contains="missing")
    checks: dict[str, dict[str, object]] = {}
    if assistant_results:
        checks["assistant_pass_rate"] = {
            "observed": float(_summarize_assistant_results(assistant_results)["pass_rate"]),
            "minimum": float(eval_config.release_gates.assistant_pass_rate_min),
        }
    if chat_sanity["examples"] > 0:
        checks["chat_sanity_pass_rate"] = {
            "observed": float(chat_sanity["pass_rate"]),
            "minimum": float(eval_config.release_gates.chat_sanity_pass_rate_min),
        }
    if normal_catalog:
        checks["catalog_exactness"] = {
            "observed": _average_metric(normal_catalog, "exactness"),
            "minimum": float(eval_config.release_gates.catalog_exactness_min),
        }
        checks["catalog_citation_rate"] = {
            "observed": _average_metric(normal_catalog, "citation_rate"),
            "minimum": float(eval_config.release_gates.catalog_citation_rate_min),
        }
    if missing_catalog:
        checks["catalog_missing_abstention_rate"] = {
            "observed": _average_metric(missing_catalog, "abstention_rate"),
            "minimum": float(eval_config.release_gates.catalog_missing_abstention_min),
        }
    webb_results = webb_results or []
    if webb_results:
        route_audit = _aggregate_route_audit(webb_results)
        course_present = _find_webb_result(webb_results, "course_present") or _find_webb_result(webb_results, "webb_course")
        course_missing = _find_webb_result(webb_results, "course_missing")
        handbook_present = _find_webb_result(webb_results, "handbook_present") or _find_webb_result(webb_results, "webb_handbook")
        handbook_missing = _find_webb_result(webb_results, "handbook_missing")
        faculty = _find_webb_result(webb_results, "faculty")
        admissions = _find_webb_result(webb_results, "admissions")
        student_life = _find_webb_result(webb_results, "student_life")
        mission_values = _find_webb_result(webb_results, "mission_values")
        college_guidance = _find_webb_result(webb_results, "college_guidance")
        museum_programs = _find_webb_result(webb_results, "museum_programs")
        athletics_present = _find_webb_result(webb_results, "athletics_present")
        athletics_missing = _find_webb_result(webb_results, "athletics_missing")
        if course_present:
            checks["webb_course_present_exactness"] = {
                "observed": float(course_present["exactness"]),
                "minimum": float(eval_config.release_gates.webb_course_present_exactness_min),
            }
            checks["webb_course_present_citation_rate"] = {
                "observed": float(course_present["citation_rate"]),
                "minimum": float(eval_config.release_gates.webb_course_present_citation_rate_min),
            }
        if course_missing:
            checks["webb_course_missing_abstention_rate"] = {
                "observed": float(course_missing["abstention_rate"]),
                "minimum": float(eval_config.release_gates.webb_course_missing_abstention_min),
            }
        if handbook_present:
            checks["webb_handbook_present_exactness"] = {
                "observed": float(handbook_present["exactness"]),
                "minimum": float(eval_config.release_gates.webb_handbook_present_exactness_min),
            }
            checks["webb_handbook_present_citation_rate"] = {
                "observed": float(handbook_present["citation_rate"]),
                "minimum": float(eval_config.release_gates.webb_handbook_present_citation_rate_min),
            }
        if handbook_missing:
            checks["webb_handbook_missing_abstention_rate"] = {
                "observed": float(handbook_missing["abstention_rate"]),
                "minimum": float(eval_config.release_gates.webb_handbook_missing_abstention_min),
            }
        if faculty:
            checks["webb_faculty_exactness"] = {
                "observed": float(faculty["exactness"]),
                "minimum": float(eval_config.release_gates.webb_faculty_exactness_min),
            }
        if admissions:
            checks["webb_admissions_exactness"] = {
                "observed": float(admissions["exactness"]),
                "minimum": float(eval_config.release_gates.webb_admissions_exactness_min),
            }
        if student_life:
            checks["webb_student_life_exactness"] = {
                "observed": float(student_life["exactness"]),
                "minimum": float(eval_config.release_gates.webb_student_life_exactness_min),
            }
            checks["webb_student_life_citation_rate"] = {
                "observed": float(student_life["citation_rate"]),
                "minimum": float(eval_config.release_gates.webb_student_life_citation_rate_min),
            }
        if mission_values:
            checks["webb_mission_values_exactness"] = {
                "observed": float(mission_values["exactness"]),
                "minimum": float(eval_config.release_gates.webb_mission_values_exactness_min),
            }
            checks["webb_mission_values_citation_rate"] = {
                "observed": float(mission_values["citation_rate"]),
                "minimum": float(eval_config.release_gates.webb_mission_values_citation_rate_min),
            }
        if college_guidance:
            checks["webb_college_guidance_exactness"] = {
                "observed": float(college_guidance["exactness"]),
                "minimum": float(eval_config.release_gates.webb_college_guidance_exactness_min),
            }
            checks["webb_college_guidance_citation_rate"] = {
                "observed": float(college_guidance["citation_rate"]),
                "minimum": float(eval_config.release_gates.webb_college_guidance_citation_rate_min),
            }
        if museum_programs:
            checks["webb_museum_programs_exactness"] = {
                "observed": float(museum_programs["exactness"]),
                "minimum": float(eval_config.release_gates.webb_museum_programs_exactness_min),
            }
            checks["webb_museum_programs_citation_rate"] = {
                "observed": float(museum_programs["citation_rate"]),
                "minimum": float(eval_config.release_gates.webb_museum_programs_citation_rate_min),
            }
        if athletics_present:
            checks["webb_athletics_present_exactness"] = {
                "observed": float(athletics_present["exactness"]),
                "minimum": float(eval_config.release_gates.webb_athletics_present_exactness_min),
            }
            checks["webb_athletics_present_citation_rate"] = {
                "observed": float(athletics_present["citation_rate"]),
                "minimum": float(eval_config.release_gates.webb_athletics_present_citation_rate_min),
            }
        if athletics_missing:
            checks["webb_athletics_missing_abstention_rate"] = {
                "observed": float(athletics_missing["abstention_rate"]),
                "minimum": float(eval_config.release_gates.webb_athletics_missing_abstention_min),
            }
        checks["webb_route_false_negative_rate"] = {
            "observed": float(route_audit.get("route_false_negative_rate", 1.0)),
            "maximum": float(eval_config.release_gates.webb_route_false_negative_rate_max),
        }
        if eval_config.release_gates.webb_require_citable_handbook:
            checks["webb_handbook_citable"] = {
                "observed": bool((grounding_quality or {}).get("handbook_citable", False)),
                "expected": True,
            }
    for payload in checks.values():
        if "minimum" in payload:
            payload["passed"] = float(payload["observed"]) >= float(payload["minimum"])
        elif "maximum" in payload:
            payload["passed"] = float(payload["observed"]) <= float(payload["maximum"])
        else:
            payload["passed"] = payload["observed"] == payload["expected"]
    return {
        "enforced": bool(eval_config.enforce_release_gates),
        "passed": all(bool(payload["passed"]) for payload in checks.values()) if checks else True,
        "checks": checks,
    }


def _with_reliability(result: dict) -> dict:
    enriched = dict(result)
    enriched["reliability"] = reliability_payload(int(result.get("examples", 0)))
    return enriched


def _aggregate_catalog_lanes(catalog_results: list[dict]) -> dict[str, dict[str, float]]:
    totals: dict[str, dict[str, float]] = {}
    for result in catalog_results:
        for lane_name, lane_metrics in result.get("attribution_lanes", {}).items():
            accumulator = totals.setdefault(
                lane_name,
                {"examples": 0.0, "exactness": 0.0, "citation_rate": 0.0, "abstention_rate": 0.0},
            )
            examples = float(lane_metrics.get("examples", 0))
            accumulator["examples"] += examples
            accumulator["exactness"] += float(lane_metrics.get("exactness", 0.0)) * examples
            accumulator["citation_rate"] += float(lane_metrics.get("citation_rate", 0.0)) * examples
            accumulator["abstention_rate"] += float(lane_metrics.get("abstention_rate", 0.0)) * examples
    aggregated: dict[str, dict[str, float]] = {}
    for lane_name, metrics in totals.items():
        examples = max(metrics["examples"], 1.0)
        aggregated[lane_name] = {
            "examples": int(metrics["examples"]),
            "exactness": metrics["exactness"] / examples,
            "citation_rate": metrics["citation_rate"] / examples,
            "abstention_rate": metrics["abstention_rate"] / examples,
        }
    return aggregated


def _aggregate_retrieval_audit(catalog_results: list[dict]) -> dict[str, float]:
    totals: dict[str, float] = {}
    for result in catalog_results:
        for key, value in result.get("retrieval_audit", {}).items():
            totals[key] = totals.get(key, 0.0) + float(value)
    if totals.get("queries", 0.0) > 0:
        totals["router_trigger_rate"] = totals.get("router_triggers", 0.0) / totals["queries"]
        totals["retrieval_hit_rate"] = totals.get("retrieval_hits", 0.0) / totals["queries"]
    if totals.get("present_queries", 0.0) > 0:
        totals["present_query_hit_rate"] = (
            (totals.get("present_queries", 0.0) - totals.get("retrieval_false_negatives", 0.0))
            / totals["present_queries"]
        )
        totals["retrieval_false_negative_rate"] = (
            totals.get("retrieval_false_negatives", 0.0) / totals["present_queries"]
        )
    if totals.get("missing_queries", 0.0) > 0:
        totals["true_missing_rate"] = totals.get("true_missing_answers", 0.0) / totals["missing_queries"]
        totals["model_hallucination_after_no_hit_rate"] = (
            totals.get("model_hallucinations_after_no_hit", 0.0) / totals["missing_queries"]
        )
    return totals


def _aggregate_route_audit(webb_results: list[dict]) -> dict[str, float]:
    totals: dict[str, float] = {}
    for result in webb_results:
        for key, value in result.get("route_audit", {}).items():
            totals[key] = totals.get(key, 0.0) + float(value)
    if totals.get("expected_grounded_queries", 0.0) > 0:
        totals["route_false_negative_rate"] = (
            totals.get("route_false_negatives", 0.0) / totals["expected_grounded_queries"]
        )
    return totals


def _summarize_webb_domains(webb_results: list[dict]) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    for result in webb_results:
        domain = str(result.get("domain") or Path(result["path"]).stem)
        summary[domain] = {
            "examples": int(result.get("examples", 0)),
            "exactness": float(result.get("exactness", 0.0)),
            "citation_rate": float(result.get("citation_rate", 0.0)),
            "abstention_rate": float(result.get("abstention_rate", 0.0)),
        }
    return summary


def _find_webb_result(webb_results: list[dict], stem_fragment: str) -> dict | None:
    for result in webb_results:
        if stem_fragment in Path(result["path"]).stem:
            return result
    return None


def _run_evaluation_once(
    model_config: ModelConfig,
    data_config: DataConfig,
    eval_config: EvalConfig,
    checkpoint_path: str,
) -> dict:
    seed_bundle = seed_everything(eval_config.seed)
    model = CausalTransformer(model_config)
    _load_checkpoint(model, checkpoint_path)
    device = get_torch_device()
    model = model.to(device)
    model.eval()

    builder = DatasetBuilder(data_config)
    results: dict[str, object] = {}
    if data_config.validation_sources:
        validation = builder.build_validation()
        from train.loop import build_dataloader

        dataloader = build_dataloader(validation, batch_size=eval_config.batch_size, shuffle=False)
        results["validation"] = evaluate_perplexity(model, dataloader, max_batches=50)

    assistant_results = []
    catalog_results = []
    webb_results = []
    for benchmark_path in eval_config.benchmark_paths:
        suffix = Path(benchmark_path).suffix.lower()
        if suffix == ".responses":
            if eval_config.grounding is not None:
                webb_results.append(
                    evaluate_webb_benchmark(
                        benchmark_path,
                        model=model,
                        tokenizer_path=data_config.tokenizer_path,
                        grounding_dsn=eval_config.grounding.dsn,
                        seed_url_pack=eval_config.grounding.seed_url_pack,
                        handbook_url=eval_config.grounding.handbook_url,
                        source_policy_path=eval_config.grounding.source_policy_path,
                        snapshot_id=eval_config.grounding.snapshot_id,
                        sync_on_start=eval_config.grounding.sync_on_start,
                        allow_ocr_fallback=eval_config.grounding.allow_ocr_fallback,
                        route_fanout_limit=eval_config.grounding.route_fanout_limit,
                        planner_beta_enabled=eval_config.grounding.planner_beta_enabled,
                        max_new_tokens=eval_config.max_new_tokens,
                        temperature=eval_config.temperature,
                        top_p=eval_config.top_p,
                        repetition_penalty=eval_config.repetition_penalty,
                        no_repeat_ngram_size=eval_config.no_repeat_ngram_size,
                        stop_strings=eval_config.stop_strings,
                    )
                )
            else:
                catalog_results.append(
                    evaluate_catalog_benchmark(
                        benchmark_path,
                        model=model,
                        tokenizer_path=data_config.tokenizer_path,
                        catalog_dsn=eval_config.catalog_dsn,
                        catalog_input_path=eval_config.catalog_input_path,
                        max_new_tokens=eval_config.max_new_tokens,
                        temperature=eval_config.temperature,
                        top_p=eval_config.top_p,
                        repetition_penalty=eval_config.repetition_penalty,
                        no_repeat_ngram_size=eval_config.no_repeat_ngram_size,
                        stop_strings=eval_config.stop_strings,
                    )
                )
        else:
            assistant_results.append(
                run_assistant_benchmark(
                    model=model,
                    tokenizer_path=data_config.tokenizer_path,
                    benchmark_path=benchmark_path,
                    max_new_tokens=eval_config.max_new_tokens,
                    temperature=eval_config.temperature,
                    top_p=eval_config.top_p,
                    repetition_penalty=eval_config.repetition_penalty,
                    no_repeat_ngram_size=eval_config.no_repeat_ngram_size,
                    stop_strings=eval_config.stop_strings,
                    require_citations=eval_config.require_citations,
                )
            )
    assistant_results = [_with_reliability(asdict(result)) for result in assistant_results]
    catalog_results = [_with_reliability(asdict(result)) for result in catalog_results]
    webb_results = [_with_reliability(asdict(result)) for result in webb_results]
    results["assistant_benchmarks"] = assistant_results
    results["assistant_summary"] = _summarize_assistant_results(assistant_results)
    results["catalog_benchmarks"] = catalog_results
    results["webb_benchmarks"] = webb_results
    results["webb_domain_summary"] = _summarize_webb_domains(webb_results)
    if webb_results:
        results["attribution_lanes"] = _aggregate_catalog_lanes(webb_results)
        results["retrieval_audit"] = _aggregate_retrieval_audit(webb_results)
        results["route_audit"] = _aggregate_route_audit(webb_results)
    else:
        results["attribution_lanes"] = _aggregate_catalog_lanes(catalog_results)
        results["retrieval_audit"] = _aggregate_retrieval_audit(catalog_results)
    benchmark_info = benchmark_manifest(eval_config.benchmark_paths)
    scorer_info = scorer_manifest()
    release_gate_info = release_gate_manifest(eval_config.release_gates.to_dict())
    grounding_quality = None
    resolved_snapshot_id = None
    if eval_config.grounding is not None:
        grounding_store = WebbKnowledgeStore(eval_config.grounding.dsn)
        resolved_snapshot_id = grounding_store.resolve_snapshot_id(eval_config.grounding.snapshot_id)
        grounding_quality = grounding_store.snapshot_quality(eval_config.grounding.snapshot_id)
        results["grounding_quality"] = grounding_quality
    snapshot_provenance = (
        grounding_snapshot_manifest(
            eval_config.grounding.dsn,
            snapshot_id=resolved_snapshot_id,
            seed_url_pack=eval_config.grounding.seed_url_pack,
            offline_seed_url_pack=eval_config.grounding.offline_seed_url_pack,
            handbook_url=eval_config.grounding.handbook_url,
            source_policy_path=eval_config.grounding.source_policy_path,
            catalog_input_path=eval_config.grounding.legacy_catalog_input_path,
        )
        if eval_config.grounding is not None
        else catalog_snapshot_manifest(
            eval_config.catalog_dsn,
            eval_config.catalog_input_path,
        )
    )
    results["benchmark_suite"] = benchmark_info
    results["provenance"] = {
        "checkpoint": checkpoint_manifest(checkpoint_path),
        "tokenizer": tokenizer_manifest(data_config.tokenizer_path),
        "export": None,
        "catalog_snapshot": snapshot_provenance,
        "grounding_snapshot": snapshot_provenance,
        "decode": {
            "preset": eval_config.decode_preset,
            "max_new_tokens": eval_config.max_new_tokens,
            "temperature": eval_config.temperature,
            "top_p": eval_config.top_p,
            "repetition_penalty": eval_config.repetition_penalty,
            "no_repeat_ngram_size": eval_config.no_repeat_ngram_size,
            "stop_strings": eval_config.stop_strings,
        },
        "seed_bundle": seed_bundle,
        "benchmark_suite_version": benchmark_info["version"],
        "scorer": scorer_info,
        "release_gates": release_gate_info,
    }
    results["release_gates"] = _build_release_gates(
        eval_config,
        assistant_results,
        catalog_results,
        webb_results=webb_results,
        grounding_quality=grounding_quality,
    )
    return results


def run_evaluation(
    model_config: ModelConfig,
    data_config: DataConfig,
    eval_config: EvalConfig,
    checkpoint_path: str,
) -> dict:
    current = _run_evaluation_once(model_config, data_config, eval_config, checkpoint_path)
    if not eval_config.compare_to_checkpoint:
        return current

    baseline = _run_evaluation_once(
        model_config,
        data_config,
        eval_config,
        eval_config.compare_to_checkpoint,
    )
    comparison: dict[str, object] = {
        "compare_to_checkpoint": eval_config.compare_to_checkpoint,
        "assistant_pass_rate_delta": current["assistant_summary"]["pass_rate"]
        - baseline["assistant_summary"]["pass_rate"],
        "assistant_average_score_delta": current["assistant_summary"]["average_score"]
        - baseline["assistant_summary"]["average_score"],
    }
    if "validation" in current and "validation" in baseline:
        comparison["validation_loss_delta"] = current["validation"]["loss"] - baseline["validation"]["loss"]
        comparison["validation_perplexity_delta"] = (
            current["validation"]["perplexity"] - baseline["validation"]["perplexity"]
        )
    current["comparison"] = comparison
    current["baseline"] = {
        "checkpoint": eval_config.compare_to_checkpoint,
        "validation": baseline.get("validation"),
        "assistant_summary": baseline.get("assistant_summary"),
        "catalog_benchmarks": baseline.get("catalog_benchmarks", []),
        "webb_benchmarks": baseline.get("webb_benchmarks", []),
    }
    return current
