from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .config import PipelineConfig
from .io_utils import read_text, short_text, write_json
from .markdown_mineru import Chunk, make_chunks, parse_sections
from .prompts import (
    L1_SYSTEM,
    L2_SYSTEM,
    RELATION_COMPLETION_SYSTEM,
    l1_user_prompt,
    l2_user_prompt,
    relation_completion_user_prompt,
)
from .progress import progress
from .qwenvl_client import QwenVLClient


def run_pipeline(config: PipelineConfig) -> dict[str, Any]:
    progress("Pipeline started", config.verbose)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.cache_dir.mkdir(parents=True, exist_ok=True)
    progress(f"Output directory: {config.output_dir}", config.verbose)
    progress(f"Cache directory: {config.cache_dir}", config.verbose)

    markdown = read_text(config.markdown_path)
    progress(f"Loaded MinerU markdown: {config.markdown_path} ({len(markdown)} chars)", config.verbose)
    sections = parse_sections(markdown)
    progress(f"Parsed sections: {len(sections)}", config.verbose)
    chunks = make_chunks(
        sections=sections,
        image_dir=config.image_dir,
        max_chars=config.max_chars_per_chunk,
        overlap_chars=config.overlap_chars,
        max_images=config.max_images_per_chunk,
    )
    progress(f"Built multimodal chunks: {len(chunks)}", config.verbose)
    manifest = build_manifest(config, chunks)
    write_json(config.output_dir / "00_manifest.json", manifest)
    progress("Wrote 00_manifest.json", config.verbose)

    client = QwenVLClient(
        api_key=config.api_key,
        base_url=config.base_url,
        model=config.model,
        cache_dir=config.cache_dir,
        timeout=config.request_timeout,
        max_retries=config.max_retries,
        max_tokens=config.max_tokens,
        temperature=config.temperature,
        dry_run=config.dry_run,
        verbose=config.verbose,
    )

    l1_results = []
    for index, chunk in enumerate(chunks, start=1):
        image_names = [p.name for p in chunk.image_paths]
        progress(
            f"L1 chunk {index}/{len(chunks)}: {chunk.id}, section='{chunk.section_title}', "
            f"chars={len(chunk.text)}, images={len(image_names)}",
            config.verbose,
        )
        user_prompt = l1_user_prompt(
            chunk_id=chunk.id,
            section=chunk.section_title,
            text=chunk.text,
            image_names=image_names,
        )
        result = client.chat_json(
            stage="l1_chunk_extract",
            system_prompt=L1_SYSTEM,
            user_text=user_prompt,
            images=chunk.image_paths,
            extra_cache_key={"chunk_id": chunk.id},
        )
        result.setdefault("chunk_id", chunk.id)
        result.setdefault("section", chunk.section_title)
        l1_results.append(result)
        progress(f"L1 chunk {chunk.id} completed", config.verbose)
    write_json(config.output_dir / "01_l1_chunk_results.json", l1_results)
    progress("Wrote 01_l1_chunk_results.json", config.verbose)

    evidence_index = build_evidence_index(chunks, l1_results)
    write_json(config.output_dir / "02_evidence_index.json", evidence_index)
    progress("Wrote 02_evidence_index.json", config.verbose)

    progress("L2 paper-level merge started", config.verbose)
    l2_result = merge_l1_results(client, l1_results, config.max_chars_per_chunk * 2)
    write_json(config.output_dir / "03_l2_paper_merge.json", l2_result)
    progress("Wrote 03_l2_paper_merge.json", config.verbose)

    progress("L3 deterministic quality audit started", config.verbose)
    l3_result = deterministic_l3_audit(l2_result, evidence_index)
    if config.enable_relation_completion:
        progress("Relation completion started", config.verbose)
        l3_result = complete_missing_relations(client, l3_result)
        progress("Relation completion completed", config.verbose)
    else:
        progress("Relation completion skipped", config.verbose)
    write_json(config.output_dir / "04_final_extraction.json", l3_result)
    progress("Wrote 04_final_extraction.json", config.verbose)
    progress("Pipeline completed", config.verbose)
    return l3_result


def merge_l1_results(client: QwenVLClient, l1_results: list[dict[str, Any]], max_payload_chars: int) -> dict[str, Any]:
    payload = json.dumps(l1_results, ensure_ascii=False, indent=2)
    if len(payload) <= max_payload_chars:
        progress(f"L2 merge uses one payload ({len(payload)} chars)", client.verbose)
        return client.chat_json(
            stage="l2_paper_merge",
            system_prompt=L2_SYSTEM,
            user_text=l2_user_prompt(payload),
            images=[],
            extra_cache_key={"l1_count": len(l1_results), "batch": "all"},
        )

    batch_results: list[dict[str, Any]] = []
    batch: list[dict[str, Any]] = []
    batch_idx = 1
    progress(f"L2 merge payload is large ({len(payload)} chars); batching enabled", client.verbose)
    for item in l1_results:
        trial = batch + [item]
        if batch and len(json.dumps(trial, ensure_ascii=False, indent=2)) > max_payload_chars:
            progress(f"L2 batch {batch_idx} started with {len(batch)} L1 items", client.verbose)
            batch_results.append(_merge_l1_batch(client, batch, batch_idx))
            progress(f"L2 batch {batch_idx} completed", client.verbose)
            batch_idx += 1
            batch = [item]
        else:
            batch = trial
    if batch:
        progress(f"L2 batch {batch_idx} started with {len(batch)} L1 items", client.verbose)
        batch_results.append(_merge_l1_batch(client, batch, batch_idx))
        progress(f"L2 batch {batch_idx} completed", client.verbose)

    compact_batches = [compact_l2_result(item, f"B{idx:02d}") for idx, item in enumerate(batch_results, start=1)]
    progress(
        f"L2 final merge uses deterministic code merge for {len(compact_batches)} compact batch results",
        client.verbose,
    )
    return deterministic_l2_merge(compact_batches)


def _merge_l1_batch(client: QwenVLClient, batch: list[dict[str, Any]], batch_idx: int) -> dict[str, Any]:
    return client.chat_json(
        stage="l2_paper_merge_batch",
        system_prompt=L2_SYSTEM,
        user_text=l2_user_prompt(json.dumps(batch, ensure_ascii=False, indent=2)),
        images=[],
        extra_cache_key={"batch_idx": batch_idx, "l1_count": len(batch)},
    )


def compact_l2_result(result: dict[str, Any], batch_id: str) -> dict[str, Any]:
    return {
        "batch_id": batch_id,
        "paper_research_problems": [
            {
                "id": item.get("id"),
                "problem": item.get("problem"),
                "problem_type": item.get("problem_type"),
                "explicitness": item.get("explicitness"),
                "evidence_refs": item.get("evidence_refs", []),
                "confidence": item.get("confidence"),
            }
            for item in result.get("paper_research_problems", [])
            if isinstance(item, dict)
        ],
        "paper_methods": [
            {
                "id": item.get("id"),
                "method": item.get("method"),
                "method_type": item.get("method_type"),
                "inputs": item.get("inputs", []),
                "outputs": item.get("outputs", []),
                "evidence_refs": item.get("evidence_refs", []),
                "confidence": item.get("confidence"),
            }
            for item in result.get("paper_methods", [])
            if isinstance(item, dict)
        ],
        "problem_method_links": [
            {
                "problem_id": item.get("problem_id"),
                "method_id": item.get("method_id"),
                "relation": item.get("relation"),
                "rationale": item.get("rationale"),
            }
            for item in result.get("problem_method_links", [])
            if isinstance(item, dict)
        ],
        "unresolved_or_ambiguous": result.get("unresolved_or_ambiguous", []),
    }


def deterministic_l2_merge(batch_results: list[dict[str, Any]]) -> dict[str, Any]:
    problems_by_key: dict[str, dict[str, Any]] = {}
    methods_by_key: dict[str, dict[str, Any]] = {}
    problem_id_map: dict[tuple[str, str], str] = {}
    method_id_map: dict[tuple[str, str], str] = {}
    unresolved: list[str] = []

    for batch in batch_results:
        batch_id = str(batch.get("batch_id", "BXX"))
        for item in batch.get("paper_research_problems", []):
            if not isinstance(item, dict):
                continue
            text = str(item.get("problem") or "").strip()
            old_id = str(item.get("id") or "")
            if not text:
                continue
            key = normalize_merge_key(text)
            if key not in problems_by_key:
                new_id = f"RP{len(problems_by_key) + 1}"
                problems_by_key[key] = {
                    "id": new_id,
                    "problem": text,
                    "problem_type": item.get("problem_type") or "other",
                    "explicitness": item.get("explicitness") or "uncertain",
                    "evidence_refs": unique_list(item.get("evidence_refs", [])),
                    "confidence": safe_float(item.get("confidence")),
                    "merged_from": [f"{batch_id}:{old_id}"],
                }
            else:
                merged = problems_by_key[key]
                merged["evidence_refs"] = unique_list(merged.get("evidence_refs", []) + item.get("evidence_refs", []))
                merged["confidence"] = max(safe_float(merged.get("confidence")), safe_float(item.get("confidence")))
                merged["merged_from"] = unique_list(merged.get("merged_from", []) + [f"{batch_id}:{old_id}"])
            if old_id:
                problem_id_map[(batch_id, old_id)] = problems_by_key[key]["id"]

        for item in batch.get("paper_methods", []):
            if not isinstance(item, dict):
                continue
            text = str(item.get("method") or "").strip()
            old_id = str(item.get("id") or "")
            if not text:
                continue
            key = normalize_merge_key(text)
            if key not in methods_by_key:
                new_id = f"M{len(methods_by_key) + 1}"
                methods_by_key[key] = {
                    "id": new_id,
                    "method": text,
                    "method_type": item.get("method_type") or "other",
                    "inputs": unique_list(item.get("inputs", [])),
                    "outputs": unique_list(item.get("outputs", [])),
                    "evidence_refs": unique_list(item.get("evidence_refs", [])),
                    "confidence": safe_float(item.get("confidence")),
                    "merged_from": [f"{batch_id}:{old_id}"],
                }
            else:
                merged = methods_by_key[key]
                merged["inputs"] = unique_list(merged.get("inputs", []) + item.get("inputs", []))
                merged["outputs"] = unique_list(merged.get("outputs", []) + item.get("outputs", []))
                merged["evidence_refs"] = unique_list(merged.get("evidence_refs", []) + item.get("evidence_refs", []))
                merged["confidence"] = max(safe_float(merged.get("confidence")), safe_float(item.get("confidence")))
                merged["merged_from"] = unique_list(merged.get("merged_from", []) + [f"{batch_id}:{old_id}"])
            if old_id:
                method_id_map[(batch_id, old_id)] = methods_by_key[key]["id"]

        for item in batch.get("unresolved_or_ambiguous", []):
            if item:
                unresolved.append(str(item))

    links = []
    seen_links: set[tuple[str, str, str]] = set()
    for batch in batch_results:
        batch_id = str(batch.get("batch_id", "BXX"))
        for link in batch.get("problem_method_links", []):
            if not isinstance(link, dict):
                continue
            problem_id = problem_id_map.get((batch_id, str(link.get("problem_id") or "")))
            method_id = method_id_map.get((batch_id, str(link.get("method_id") or "")))
            relation = str(link.get("relation") or "partially_addresses")
            if not problem_id or not method_id:
                continue
            key = (problem_id, method_id, relation)
            if key in seen_links:
                continue
            seen_links.add(key)
            links.append(
                {
                    "problem_id": problem_id,
                    "method_id": method_id,
                    "relation": relation,
                    "rationale": str(link.get("rationale") or ""),
                }
            )

    return {
        "paper_research_problems": list(problems_by_key.values()),
        "paper_methods": list(methods_by_key.values()),
        "problem_method_links": links,
        "unresolved_or_ambiguous": unique_list(unresolved),
        "merge_note": (
            "Final L2 merge was performed deterministically in code to avoid long invalid JSON "
            "generation and to improve reproducibility. Semantic merging is performed in L2 batches; "
            "the final step normalizes IDs, exact-duplicate claims, evidence references, and links."
        ),
    }


def normalize_merge_key(text: str) -> str:
    return " ".join(text.lower().replace("-", " ").replace("_", " ").split())


def unique_list(values: list[Any]) -> list[Any]:
    result: list[Any] = []
    seen: set[str] = set()
    for value in values:
        key = json.dumps(value, ensure_ascii=False, sort_keys=True) if isinstance(value, (dict, list)) else str(value)
        if key in seen:
            continue
        seen.add(key)
        result.append(value)
    return result


def safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def deterministic_l3_audit(l2_result: dict[str, Any], evidence_index: dict[str, Any]) -> dict[str, Any]:
    problems = [
        finalize_problem(item)
        for item in l2_result.get("paper_research_problems", [])
        if isinstance(item, dict) and item.get("evidence_refs")
    ]
    methods = [
        finalize_method(item)
        for item in l2_result.get("paper_methods", [])
        if isinstance(item, dict) and item.get("evidence_refs")
    ]
    problem_ids = {item["id"] for item in problems}
    method_ids = {item["id"] for item in methods}
    links = []
    for link in l2_result.get("problem_method_links", []):
        if not isinstance(link, dict):
            continue
        if link.get("problem_id") not in problem_ids or link.get("method_id") not in method_ids:
            continue
        links.append(
            {
                "problem_id": link.get("problem_id"),
                "method_id": link.get("method_id"),
                "relation": link.get("relation") or "partially_addresses",
                "link_type": link.get("link_type") or "evidence_supported",
                "confidence": safe_float(link.get("confidence")) or 1.0,
                "rationale": link.get("rationale") or "",
            }
        )

    all_items = problems + methods
    visual_count = sum(1 for item in all_items if contains_visual_evidence(item.get("evidence_refs", [])))
    evidence_count = sum(1 for item in all_items if item.get("evidence_refs"))
    low_confidence = [
        item["id"]
        for item in all_items
        if safe_float(item.get("confidence")) < 0.5
    ]
    coarse_items = [
        item["id"]
        for item in all_items
        if item.get("granularity") == "coarse"
    ]

    return {
        "final_research_problems": problems,
        "final_methods": methods,
        "problem_method_links": links,
        "quality_report": {
            "evidence_coverage": qualitative_ratio(evidence_count, len(all_items)),
            "cross_modal_usage": qualitative_ratio(visual_count, len(all_items)),
            "main_limitations": build_limitations(problems, methods, links, low_confidence, coarse_items),
            "recommended_human_checks": build_human_checks(low_confidence, coarse_items, evidence_index),
        },
        "audit_note": (
            "L3 was performed deterministically in code to avoid invalid long JSON generation. "
            "The audit removes evidence-free items, normalizes final fields, estimates granularity, "
            "checks link validity, and produces a quality report."
        ),
    }


def complete_missing_relations(client: QwenVLClient, final_result: dict[str, Any]) -> dict[str, Any]:
    payload = build_relation_completion_payload(final_result)
    if not payload["candidate_problems"] or not payload["candidate_methods"]:
        final_result.setdefault("relation_completion_note", "No isolated candidate nodes were found.")
        return final_result

    try:
        completion = client.chat_json(
            stage="relation_completion",
            system_prompt=RELATION_COMPLETION_SYSTEM,
            user_text=relation_completion_user_prompt(json.dumps(payload, ensure_ascii=False, indent=2)),
            images=[],
            extra_cache_key={
                "problem_count": len(final_result.get("final_research_problems", [])),
                "method_count": len(final_result.get("final_methods", [])),
                "existing_link_count": len(final_result.get("problem_method_links", [])),
                "version": "v1",
            },
        )
    except Exception as exc:
        final_result["relation_completion_note"] = f"Relation completion failed and was skipped: {exc}"
        return final_result
    inferred = validate_inferred_links(completion, final_result)
    existing = final_result.get("problem_method_links", [])
    final_result["problem_method_links"] = existing + inferred
    final_result["relation_completion_note"] = (
        f"Added {len(inferred)} inferred links. These links are semantic completions and "
        "should be distinguished from evidence_supported links in analysis."
    )
    report = final_result.setdefault("quality_report", {})
    checks = report.setdefault("recommended_human_checks", [])
    checks.append("Manually verify inferred problem-method links before using them as evidence-supported findings.")
    return final_result


def build_relation_completion_payload(final_result: dict[str, Any]) -> dict[str, Any]:
    problems = final_result.get("final_research_problems", [])
    methods = final_result.get("final_methods", [])
    links = final_result.get("problem_method_links", [])
    connected_problems = {link.get("problem_id") for link in links if isinstance(link, dict)}
    connected_methods = {link.get("method_id") for link in links if isinstance(link, dict)}

    candidate_problems = [
        compact_final_problem(item)
        for item in problems
        if isinstance(item, dict) and item.get("id") not in connected_problems
    ]
    candidate_methods = [
        compact_final_method(item)
        for item in methods
        if isinstance(item, dict) and item.get("id") not in connected_methods
    ]
    if not candidate_problems:
        candidate_problems = [compact_final_problem(item) for item in problems if isinstance(item, dict)]
    if not candidate_methods:
        candidate_methods = [compact_final_method(item) for item in methods if isinstance(item, dict)]

    return {
        "candidate_problems": candidate_problems,
        "candidate_methods": candidate_methods,
        "existing_links": [
            {
                "problem_id": link.get("problem_id"),
                "method_id": link.get("method_id"),
                "relation": link.get("relation"),
                "link_type": link.get("link_type", "evidence_supported"),
            }
            for link in links
            if isinstance(link, dict)
        ],
    }


def compact_final_problem(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": item.get("id"),
        "problem": item.get("problem"),
        "problem_type": item.get("problem_type"),
        "granularity": item.get("granularity"),
        "explicitness": item.get("explicitness"),
        "confidence": item.get("confidence"),
    }


def compact_final_method(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": item.get("id"),
        "method": item.get("method"),
        "method_type": item.get("method_type"),
        "granularity": item.get("granularity"),
        "confidence": item.get("confidence"),
    }


def validate_inferred_links(completion: dict[str, Any], final_result: dict[str, Any]) -> list[dict[str, Any]]:
    problem_ids = {item.get("id") for item in final_result.get("final_research_problems", []) if isinstance(item, dict)}
    method_ids = {item.get("id") for item in final_result.get("final_methods", []) if isinstance(item, dict)}
    existing_keys = {
        (link.get("problem_id"), link.get("method_id"), link.get("relation"))
        for link in final_result.get("problem_method_links", [])
        if isinstance(link, dict)
    }
    allowed_relations = {"directly_addresses", "partially_addresses", "evaluates", "motivates"}
    inferred = []
    for link in completion.get("inferred_problem_method_links", []):
        if not isinstance(link, dict):
            continue
        problem_id = link.get("problem_id")
        method_id = link.get("method_id")
        relation = link.get("relation") or "partially_addresses"
        if problem_id not in problem_ids or method_id not in method_ids:
            continue
        if relation not in allowed_relations:
            relation = "partially_addresses"
        key = (problem_id, method_id, relation)
        if key in existing_keys:
            continue
        existing_keys.add(key)
        inferred.append(
            {
                "problem_id": problem_id,
                "method_id": method_id,
                "relation": relation,
                "link_type": "inferred",
                "confidence": safe_float(link.get("confidence")),
                "rationale": str(link.get("rationale") or ""),
            }
        )
    return inferred


def finalize_problem(item: dict[str, Any]) -> dict[str, Any]:
    text = str(item.get("problem") or "").strip()
    confidence = safe_float(item.get("confidence"))
    return {
        "id": item.get("id"),
        "problem": text,
        "problem_type": item.get("problem_type") or "other",
        "granularity": infer_granularity(text),
        "explicitness": item.get("explicitness") or "uncertain",
        "evidence_refs": unique_list(item.get("evidence_refs", [])),
        "confidence": confidence,
        "risk_note": infer_risk_note(text, confidence, item.get("evidence_refs", [])),
    }


def finalize_method(item: dict[str, Any]) -> dict[str, Any]:
    text = str(item.get("method") or "").strip()
    confidence = safe_float(item.get("confidence"))
    inputs = unique_list(item.get("inputs", []))
    outputs = unique_list(item.get("outputs", []))
    return {
        "id": item.get("id"),
        "method": text,
        "method_type": item.get("method_type") or "other",
        "reproducibility_fields": {
            "inputs": inputs,
            "outputs": outputs,
            "procedure": infer_procedure_steps(text),
            "objective_or_metric": infer_objective_or_metric(text),
        },
        "granularity": infer_granularity(text),
        "evidence_refs": unique_list(item.get("evidence_refs", [])),
        "confidence": confidence,
        "risk_note": infer_risk_note(text, confidence, item.get("evidence_refs", [])),
    }


def infer_granularity(text: str) -> str:
    lower = text.lower()
    cjk_chars = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    words = [word for word in lower.replace("-", " ").replace("_", " ").split() if word]
    fine_cues = [
        "loss",
        "module",
        "step",
        "input",
        "output",
        "training",
        "inference",
        "feature",
        "representation",
        "evaluation",
        "metric",
        "objective",
        "模块",
        "步骤",
        "输入",
        "输出",
        "训练",
        "推理",
        "特征",
        "表示",
        "损失",
        "目标",
        "评估",
        "指标",
        "协议",
        "算法",
        "矩阵",
        "匹配",
    ]
    if cjk_chars:
        if cjk_chars < 12:
            return "coarse"
        if cjk_chars <= 80 or any(cue in lower for cue in fine_cues):
            return "fine"
        return "medium"
    if len(words) <= 8:
        return "coarse"
    if len(words) <= 35 or any(cue in lower for cue in fine_cues):
        return "fine"
    return "medium"


def infer_risk_note(text: str, confidence: float, evidence_refs: list[Any]) -> str:
    notes = []
    if confidence < 0.5:
        notes.append("low model confidence")
    if not evidence_refs:
        notes.append("missing evidence reference")
    if infer_granularity(text) == "coarse":
        notes.append("claim may be too coarse")
    return "; ".join(notes)


def infer_procedure_steps(text: str) -> list[str]:
    clauses = []
    for sep in ["; ", ". ", " and then ", " then ", " followed by "]:
        if sep in text:
            clauses = [part.strip(" .;") for part in text.split(sep) if part.strip(" .;")]
            break
    if not clauses:
        clauses = [text.strip()]
    return clauses[:5]


def infer_objective_or_metric(text: str) -> list[str]:
    lower = text.lower()
    cues = ["loss", "objective", "optimize", "metric", "accuracy", "iou", "ap", "map", "f1"]
    if any(cue in lower for cue in cues):
        return [text]
    return []


def contains_visual_evidence(evidence_refs: list[Any]) -> bool:
    visual_terms = ("image", "figure", "fig", "table", "visual", "equation", "mixed")
    for ref in evidence_refs:
        text = json.dumps(ref, ensure_ascii=False).lower()
        if any(term in text for term in visual_terms):
            return True
    return False


def qualitative_ratio(count: int, total: int) -> str:
    if total <= 0:
        return "low"
    ratio = count / total
    if ratio >= 0.75:
        return "high"
    if ratio >= 0.4:
        return "medium"
    return "low"


def build_limitations(
    problems: list[dict[str, Any]],
    methods: list[dict[str, Any]],
    links: list[dict[str, Any]],
    low_confidence: list[str],
    coarse_items: list[str],
) -> list[str]:
    limitations = []
    if not problems:
        limitations.append("No evidence-supported research problem remained after deterministic audit.")
    if not methods:
        limitations.append("No evidence-supported method remained after deterministic audit.")
    if not links:
        limitations.append("No valid problem-method link remained after ID validation.")
    if low_confidence:
        limitations.append(f"Low-confidence items require human verification: {', '.join(low_confidence[:10])}.")
    if coarse_items:
        limitations.append(f"Potentially coarse-grained items require manual refinement: {', '.join(coarse_items[:10])}.")
    if not limitations:
        limitations.append("Automatic audit found no major structural issue, but semantic faithfulness still requires human evaluation.")
    return limitations


def build_human_checks(
    low_confidence: list[str],
    coarse_items: list[str],
    evidence_index: dict[str, Any],
) -> list[str]:
    checks = []
    if low_confidence:
        checks.append(f"Check whether low-confidence items are supported by the cited chunks: {', '.join(low_confidence[:10])}.")
    if coarse_items:
        checks.append(f"Check whether coarse items should be split into finer atoms: {', '.join(coarse_items[:10])}.")
    if evidence_index:
        checks.append("Verify a sample of evidence_refs against 02_evidence_index.json and the original MinerU markdown/images.")
    checks.append("Manually compare final items with the abstract, introduction, method, and experiment sections.")
    return checks


def build_manifest(config: PipelineConfig, chunks: list[Chunk]) -> dict[str, Any]:
    return {
        "paper_dir": str(config.paper_dir),
        "markdown_path": str(config.markdown_path),
        "image_dir": str(config.image_dir),
        "model": config.model,
        "base_url": config.base_url,
        "chunk_count": len(chunks),
        "chunks": [
            {
                "id": c.id,
                "section_title": c.section_title,
                "section_level": c.section_level,
                "start_line": c.start_line,
                "char_count": len(c.text),
                "text_preview": short_text(c.text, 500),
                "images": [str(p) for p in c.image_paths],
            }
            for c in chunks
        ],
    }


def build_evidence_index(chunks: list[Chunk], l1_results: list[dict[str, Any]]) -> dict[str, Any]:
    by_chunk = {c.id: c for c in chunks}
    index: dict[str, Any] = {}
    for result in l1_results:
        chunk_id = str(result.get("chunk_id", "unknown"))
        chunk = by_chunk.get(chunk_id)
        index[chunk_id] = {
            "section": result.get("section") or (chunk.section_title if chunk else ""),
            "text_preview": short_text(chunk.text, 800) if chunk else "",
            "images": [p.name for p in chunk.image_paths] if chunk else [],
            "research_problem_atom_ids": [
                atom.get("id") for atom in result.get("research_problem_atoms", []) if isinstance(atom, dict)
            ],
            "method_atom_ids": [
                atom.get("id") for atom in result.get("method_atoms", []) if isinstance(atom, dict)
            ],
        }
    return index
