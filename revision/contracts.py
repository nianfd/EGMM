from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from scripts.aggregate_batch_metrics import (
    compute_metrics as compute_original_metrics,
    load_trace_context,
    normalize_final_result,
)

from .evidence import build_exact_evidence_map
from .io_utils import canonical_ref, evidence_refs, node_id, node_text, normalize_final, read_json


ALLOWED_RELATIONS = {
    "directly_addresses",
    "partially_addresses",
    "evaluates",
    "motivates",
    # Accepted aliases are normalized by semantic evaluation.
    "addresses",
    "mitigates",
    "supports",
    "implements",
}
ALLOWED_LINK_TYPES = {"evidence_supported", "inferred", "unverified", ""}
NO_VISUAL_CONDITIONS = {
    "added_bm25_rag_text",
    "added_schema_text_only",
    "added_visual_masked",
}


def required_schema_valid(data: dict[str, Any]) -> bool:
    """Require one of the two schemas accepted by the original aggregator."""
    if "final_research_problems" in data or "final_methods" in data:
        return (
            isinstance(data.get("final_research_problems"), list)
            and isinstance(data.get("final_methods"), list)
            and isinstance(data.get("problem_method_links"), list)
        )
    return (
        isinstance(data.get("paper_research_problems"), list)
        and isinstance(data.get("paper_methods"), list)
        and isinstance(data.get("problem_method_links"), list)
    )


def condition_result_file(condition_dir: Path) -> Path:
    candidates = [
        condition_dir / "04_final_extraction.json",
        condition_dir / "result.json",
        condition_dir / "final_result.json",
    ]
    return next((path for path in candidates if path.exists()), candidates[0])


def validate_condition_output(
    condition_dir: Path,
    condition: str,
    *,
    allow_empty: bool = False,
) -> dict[str, Any]:
    """Validate that one added condition is consumable by both metric stacks.

    This validates machine readability and metric compatibility, not scientific
    correctness. Model-quality defects remain in the prediction and are
    reported as warnings so the metric stacks can score them as failures.
    Missing/unreadable files, an unusable top-level schema, provenance mismatch,
    or a failed metric engine remain hard errors.
    """
    errors: list[str] = []
    warnings: list[str] = []
    result_path = condition_result_file(condition_dir)
    l1_path = condition_dir / "01_l1_chunk_results.json"
    index_path = condition_dir / "02_evidence_index.json"
    info_path = condition_dir / "condition_info.json"

    if not result_path.exists():
        errors.append(f"missing result file: {result_path}")
        return _report(condition, condition_dir, result_path, errors, warnings, {})
    if not l1_path.exists():
        errors.append(f"missing self-contained L1 support: {l1_path}")
    if not index_path.exists():
        errors.append(f"missing self-contained evidence index: {index_path}")
    if not info_path.exists():
        errors.append(f"missing condition_info.json: {info_path}")

    try:
        raw = read_json(result_path)
    except Exception as exc:
        errors.append(f"invalid result JSON: {type(exc).__name__}: {exc}")
        return _report(condition, condition_dir, result_path, errors, warnings, {})
    dry_placeholder = bool(
        allow_empty and isinstance(raw, dict) and raw.get("dry_run") is True
    )
    if not isinstance(raw, dict) or (not required_schema_valid(raw) and not dry_placeholder):
        errors.append("result does not contain the required final_* or paper_* list schema")
    normalized = (
        {"problems": [], "methods": [], "links": []}
        if dry_placeholder
        else normalize_final(raw) if isinstance(raw, dict)
        else {"problems": [], "methods": [], "links": []}
    )
    problems = normalized["problems"]
    methods = normalized["methods"]
    links = normalized["links"]
    if not allow_empty and (not problems or not methods):
        warnings.append(
            f"non-dry-run result is empty/incomplete: problems={len(problems)}, methods={len(methods)}"
        )

    problem_ids = _validate_nodes(problems, "problem", errors, warnings)
    method_ids = _validate_nodes(methods, "method", errors, warnings)
    seen_links: set[tuple[str, str, str, str]] = set()
    for index, link in enumerate(links, start=1):
        problem_id = str(link.get("problem_id") or "").strip()
        method_id = str(link.get("method_id") or "").strip()
        relation = str(link.get("relation") or "").strip()
        kind = str(link.get("link_type") or "").strip()
        if problem_id not in problem_ids:
            warnings.append(f"link {index} has unknown problem_id={problem_id!r}")
        if method_id not in method_ids:
            warnings.append(f"link {index} has unknown method_id={method_id!r}")
        if relation not in ALLOWED_RELATIONS:
            warnings.append(f"link {index} has unsupported relation={relation!r}")
        if kind not in ALLOWED_LINK_TYPES:
            warnings.append(f"link {index} has unsupported link_type={kind!r}")
        key = (problem_id, method_id, relation, kind)
        if key in seen_links:
            warnings.append(f"duplicate link at position {index}: {key}")
        seen_links.add(key)
        refs = link.get("evidence_refs", [])
        if refs is not None and not isinstance(refs, list):
            warnings.append(f"link {index} evidence_refs should be a list")
        elif isinstance(refs, list):
            for ref in refs:
                if not isinstance(ref, str) or canonical_ref(ref) is None:
                    warnings.append(f"link {index} has malformed evidence ref: {ref!r}")
        if "confidence" in link:
            _validate_confidence(link.get("confidence"), f"link {index}", warnings)

    evidence_map: dict[str, Any] = {}
    if l1_path.exists() and index_path.exists():
        try:
            l1 = read_json(l1_path)
            index = read_json(index_path)
            if not isinstance(l1, list):
                errors.append("01_l1_chunk_results.json must be a JSON list")
            if not isinstance(index, dict):
                errors.append("02_evidence_index.json must be a JSON object")
            if isinstance(l1, list) and isinstance(index, dict):
                evidence_map = build_exact_evidence_map(l1, index)
        except Exception as exc:
            errors.append(f"support files are unreadable: {type(exc).__name__}: {exc}")

    for kind, items in (("problem", problems), ("method", methods)):
        for index, item in enumerate(items, start=1):
            refs = item.get("evidence_refs", [])
            if not isinstance(refs, list):
                warnings.append(f"{kind} {index} evidence_refs should be a list")
                continue
            for ref in refs:
                normalized_ref = canonical_ref(ref) if isinstance(ref, str) else None
                if normalized_ref is None:
                    warnings.append(f"{kind} {index} has malformed evidence ref: {ref!r}")
                elif evidence_map and normalized_ref not in evidence_map:
                    warnings.append(f"{kind} {index} has unknown evidence ref: {ref!r}")
                elif evidence_map and evidence_map[normalized_ref].atom_kind != kind:
                    warnings.append(
                        f"{kind} {index} evidence ref has wrong atom kind: {ref!r} -> "
                        f"{evidence_map[normalized_ref].atom_kind}"
                    )

    if evidence_map:
        for index, link in enumerate(links, start=1):
            refs = link.get("evidence_refs", [])
            if not isinstance(refs, list):
                continue
            for ref in refs:
                normalized_ref = canonical_ref(ref) if isinstance(ref, str) else None
                if normalized_ref is not None and normalized_ref not in evidence_map:
                    warnings.append(f"link {index} has unknown evidence ref: {ref!r}")

    metric_status: dict[str, Any] = {}
    try:
        final_data = (
            {
                "paper_research_problems": [],
                "paper_methods": [],
                "problem_method_links": [],
            }
            if dry_placeholder
            else normalize_final_result(raw)
        )
        if final_data is None:
            raise ValueError("original aggregator could not normalize the result schema")
        traceable_refs, visual_chunks = load_trace_context(
            condition_dir,
            force_no_visual=condition in NO_VISUAL_CONDITIONS,
        )
        original = compute_original_metrics(final_data, traceable_refs, visual_chunks)
        required = {
            "NET_node_evidence_traceability",
            "AMG_actual_multimodal_grounding",
            "ESGC_evidence_supported_graph_connectivity",
            "TLP_traceable_link_purity",
            "EGMR_evidence_grounded_method_reproducibility",
            "BEGQ_balanced_evidence_grounded_quality",
        }
        missing_metrics = sorted(required - set(original))
        if missing_metrics:
            errors.append(f"original metric engine omitted fields: {missing_metrics}")
        for name in required & set(original):
            try:
                value = float(original[name])
            except (TypeError, ValueError):
                errors.append(f"original metric {name} is not numeric: {original[name]!r}")
                continue
            if not math.isfinite(value) or value < 0.0 or value > 1.0:
                errors.append(f"original metric {name} is outside [0,1]: {value!r}")
        metric_status["original_metric_engine"] = "passed"
    except Exception as exc:
        errors.append(f"original metric engine failed: {type(exc).__name__}: {exc}")
        metric_status["original_metric_engine"] = "failed"

    metric_status["gold_metric_schema"] = "passed" if isinstance(normalized, dict) else "failed"
    metric_status["exact_evidence_record_count"] = len(evidence_map)
    if info_path.exists():
        try:
            info = read_json(info_path)
            if not isinstance(info, dict):
                raise ValueError("condition_info.json must contain a JSON object")
            if str(info.get("condition") or "") != condition:
                errors.append(
                    f"condition_info condition mismatch: expected {condition!r}, got {info.get('condition')!r}"
                )
            if str(info.get("model") or "").lower() != "qwen-vl-max":
                errors.append(f"condition_info model is not qwen-vl-max: {info.get('model')!r}")
            if info.get("dry_run") is True and not allow_empty:
                errors.append("condition_info marks this output as dry_run=true")
        except Exception as exc:
            errors.append(f"condition_info is unreadable: {type(exc).__name__}: {exc}")

    return _report(condition, condition_dir, result_path, errors, warnings, {
        "problem_count": len(problems),
        "method_count": len(methods),
        "link_count": len(links),
        **metric_status,
    })


def _validate_nodes(
    items: list[dict[str, Any]],
    kind: str,
    errors: list[str],
    warnings: list[str],
) -> set[str]:
    ids: set[str] = set()
    prefix = "RP" if kind == "problem" else "M"
    for index, item in enumerate(items, start=1):
        raw_id = item.get("id") or item.get("problem_id") or item.get("method_id")
        value = node_id(item, kind, index)
        if raw_id is None or not str(raw_id).strip():
            warnings.append(f"{kind} {index} has no ID")
        elif value in ids:
            warnings.append(f"duplicate {kind} ID: {value}")
        elif not value.startswith(prefix):
            warnings.append(f"noncanonical {kind} ID: {value}")
        ids.add(value)
        if not node_text(item):
            warnings.append(f"{kind} {value} has no claim text")
        if "confidence" not in item:
            warnings.append(f"{kind} {value} has no confidence")
        else:
            _validate_confidence(item.get("confidence"), f"{kind} {value}", warnings)
    return ids


def _validate_confidence(value: Any, label: str, warnings: list[str]) -> None:
    try:
        confidence = float(value)
        if not math.isfinite(confidence) or confidence < 0 or confidence > 1:
            warnings.append(f"{label} confidence is outside [0,1]")
    except (TypeError, ValueError):
        warnings.append(f"{label} confidence is not numeric")


def _report(
    condition: str,
    condition_dir: Path,
    result_path: Path,
    errors: list[str],
    warnings: list[str],
    counts: dict[str, Any],
) -> dict[str, Any]:
    return {
        "condition": condition,
        "condition_dir": str(condition_dir),
        "result_path": str(result_path),
        "valid": not errors,
        "errors": errors,
        "warnings": warnings,
        "counts_and_engines": counts,
        "contract_version": "2026-08-metric-compat-v3-quality-warnings",
        "quality_warning_count": len(warnings),
    }
