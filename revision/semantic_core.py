from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from paper_mining.config import PipelineConfig
from paper_mining.io_utils import read_text, write_json
from paper_mining.markdown_mineru import parse_sections
from paper_mining.pipeline import deterministic_l3_audit
from paper_mining.progress import progress

from .evidence import attach_strict_audit_diagnostics, build_exact_evidence_map
from .io_utils import canonical_ref, file_sha256, normalize_final, read_json, unique_strings
from .resilient_qwenvl import ResilientQwenVLClient
from .semantic_core_prompts import SEMANTIC_CORE_SYSTEM, semantic_core_user_prompt


ALLOWED_RELATIONS = {
    "directly_addresses",
    "partially_addresses",
    "evaluates",
    "motivates",
}
RELATION_ALIASES = {
    "addresses": "directly_addresses",
    "mitigates": "partially_addresses",
    "supports": "partially_addresses",
    "implements": "directly_addresses",
}
PROBLEM_TYPES = {
    "task_gap", "data_gap", "method_gap", "evaluation_gap",
    "application_constraint", "other",
}
METHOD_TYPES = {
    "architecture", "algorithm_step", "representation", "training_objective",
    "data_processing", "evaluation_protocol", "implementation_detail", "other",
}


def _confidence(value: Any, default: float = 0.5) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = default
    return min(1.0, max(0.0, number))


def _node_id(item: dict[str, Any], kind: str, index: int) -> str:
    value = item.get("id") or item.get("problem_id") or item.get("method_id")
    return str(value).strip() if value else f"{'RP' if kind == 'problem' else 'M'}{index}"


def _node_claim(item: dict[str, Any], kind: str) -> str:
    key = "problem" if kind == "problem" else "method"
    return str(item.get(key) or item.get("claim") or "").strip()


def select_primary_context(markdown: str, max_chars: int = 18000) -> str:
    """Select source-only high-level sections without consulting the gold file."""
    buckets: dict[str, list[Any]] = {"front": [], "abstract": [], "introduction": [], "method": [], "conclusion": []}
    for position, section in enumerate(parse_sections(markdown)):
        title = section.title.lower().strip()
        if "reference" in title or "bibliograph" in title or "acknowledg" in title:
            continue
        if position == 0 or title == "front matter":
            buckets["front"].append(section)
        elif "abstract" in title:
            buckets["abstract"].append(section)
        elif "introduction" in title or re.search(r"\bintro\b", title):
            buckets["introduction"].append(section)
        elif any(token in title for token in ("method", "approach", "framework", "architecture", "model")):
            buckets["method"].append(section)
        elif any(token in title for token in ("conclusion", "discussion", "limitation")):
            buckets["conclusion"].append(section)

    caps = {
        "front": 1200,
        "abstract": 3000,
        "introduction": 5000,
        "method": 7000,
        "conclusion": 3000,
    }
    pieces: list[str] = []
    remaining = max_chars
    for bucket in ("front", "abstract", "introduction", "method", "conclusion"):
        budget = min(caps[bucket], remaining)
        if budget <= 0:
            break
        bucket_text = "\n\n".join(
            f"## {section.title}\n{section.text}" for section in buckets[bucket]
        )
        if bucket_text:
            pieces.append(bucket_text[:budget])
            remaining -= min(len(bucket_text), budget)
    if not pieces:
        return markdown[:max_chars]
    return "\n\n".join(pieces)[:max_chars]


def build_semantic_core_records(
    l2_result: dict[str, Any],
    l1_results: list[dict[str, Any]],
    evidence_index: dict[str, Any],
) -> list[dict[str, Any]]:
    normalized = normalize_final(l2_result)
    evidence_map = build_exact_evidence_map(l1_results, evidence_index)
    records: list[dict[str, Any]] = []
    for kind, items in (("problem", normalized["problems"]), ("method", normalized["methods"])):
        prefix = "P" if kind == "problem" else "M"
        for index, item in enumerate(items, start=1):
            claim = _node_claim(item, kind)
            if not claim:
                continue
            raw_refs = item.get("evidence_refs") or []
            if isinstance(raw_refs, str):
                raw_refs = [raw_refs]
            refs: list[str] = []
            sections: list[str] = []
            for raw_ref in raw_refs:
                ref = canonical_ref(raw_ref)
                record = evidence_map.get(ref or "")
                if record is None or record.atom_kind != kind:
                    continue
                if ref not in refs:
                    refs.append(ref)
                if record.section and record.section not in sections:
                    sections.append(record.section)
            if not refs:
                continue
            node_identifier = _node_id(item, kind, index)
            record = {
                "source_id": f"{prefix}:{node_identifier}",
                "kind": kind,
                "claim": claim,
                "type": str(item.get("problem_type") or item.get("method_type") or "other"),
                "sections": sections,
                "evidence_refs": refs,
                "confidence": _confidence(item.get("confidence")),
            }
            if kind == "method":
                reproducibility = item.get("reproducibility_fields") or {}
                record["inputs"] = unique_strings(item.get("inputs") or reproducibility.get("inputs") or [])
                record["outputs"] = unique_strings(item.get("outputs") or reproducibility.get("outputs") or [])
            records.append(record)
    return records


def _core_section_score(record: dict[str, Any]) -> tuple[int, float, int]:
    text = " ".join(str(value).lower() for value in record.get("sections", []))
    if "abstract" in text:
        rank = 5
    elif "introduction" in text:
        rank = 4
    elif any(value in text for value in ("method", "approach", "framework", "architecture")):
        rank = 3
    elif any(value in text for value in ("conclusion", "discussion")):
        rank = 2
    elif any(value in text for value in ("related", "reference", "experiment", "dataset")):
        rank = 0
    else:
        rank = 1
    return rank, _confidence(record.get("confidence")), -len(str(record.get("claim") or ""))


def _fallback_groups(records: list[dict[str, Any]], kind: str, target: int) -> list[dict[str, Any]]:
    prefix = "P" if kind == "problem" else "M"
    selected = sorted(
        (record for record in records if record.get("kind") == kind),
        key=_core_section_score,
        reverse=True,
    )[:target]
    type_key = "problem_type" if kind == "problem" else "method_type"
    return [
        {
            "group_id": f"{prefix}{index}",
            "members": [record["source_id"]],
            "representative": record["source_id"],
            "canonical_claim": record["claim"],
            type_key: record.get("type") or "other",
            "confidence": record.get("confidence", 0.5),
        }
        for index, record in enumerate(selected, start=1)
    ]


def materialize_semantic_core_plan(
    plan: dict[str, Any],
    records: list[dict[str, Any]],
    *,
    target_problems: int,
    target_methods: int,
    max_problems: int,
    max_methods: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate an ID-grounded model plan and deterministically build the L2 schema."""
    by_id = {str(record["source_id"]): record for record in records}
    used: dict[str, set[str]] = {"problem": set(), "method": set()}
    nodes: dict[str, list[dict[str, Any]]] = {"problem": [], "method": []}
    group_maps: dict[str, dict[str, str]] = {"problem": {}, "method": {}}
    group_members: dict[str, dict[str, set[str]]] = {"problem": {}, "method": {}}
    rejected: list[dict[str, Any]] = []

    specifications = (
        ("problem", "problem_groups", target_problems, max_problems, PROBLEM_TYPES),
        ("method", "method_groups", target_methods, max_methods, METHOD_TYPES),
    )
    for kind, plan_key, target, maximum, allowed_types in specifications:
        raw_groups = plan.get(plan_key, [])
        groups = list(raw_groups) if isinstance(raw_groups, list) else []
        if not groups:
            groups = _fallback_groups(records, kind, target)
        for group in groups:
            if len(nodes[kind]) >= maximum:
                break
            if not isinstance(group, dict):
                continue
            raw_group_id = str(group.get("group_id") or f"{kind}_{len(nodes[kind]) + 1}")
            if raw_group_id in group_maps[kind]:
                rejected.append({"kind": kind, "group_id": raw_group_id, "reason": "duplicate_group_id"})
                continue
            members = []
            for source_id in unique_strings(group.get("members") or []):
                record = by_id.get(source_id)
                if record is None or record.get("kind") != kind or source_id in used[kind]:
                    continue
                members.append(source_id)
            if not members:
                rejected.append({"kind": kind, "group_id": group.get("group_id"), "reason": "no_valid_unused_members"})
                continue
            representative_id = str(group.get("representative") or "")
            if representative_id not in members:
                representative_id = max(members, key=lambda value: _core_section_score(by_id[value]))
            representative = by_id[representative_id]
            canonical_claim = str(group.get("canonical_claim") or "").strip()
            if len(re.findall(r"[A-Za-z0-9]+", canonical_claim)) < 3:
                canonical_claim = str(representative.get("claim") or "").strip()
            refs = unique_strings(
                ref
                for source_id in members
                for ref in by_id[source_id].get("evidence_refs", [])
            )
            if not refs:
                rejected.append({"kind": kind, "group_id": group.get("group_id"), "reason": "no_type_consistent_evidence"})
                continue
            used[kind].update(members)
            node_identifier = f"RP{len(nodes[kind]) + 1}" if kind == "problem" else f"M{len(nodes[kind]) + 1}"
            group_maps[kind][raw_group_id] = node_identifier
            group_members[kind][raw_group_id] = set(members)
            type_key = "problem_type" if kind == "problem" else "method_type"
            value_type = str(group.get(type_key) or representative.get("type") or "other")
            if value_type not in allowed_types:
                value_type = "other"
            confidence = _confidence(
                group.get("confidence"),
                sum(_confidence(by_id[value].get("confidence")) for value in members) / len(members),
            )
            if kind == "problem":
                nodes[kind].append({
                    "id": node_identifier,
                    "problem": canonical_claim,
                    "problem_type": value_type,
                    "explicitness": "explicit",
                    "evidence_refs": refs,
                    "confidence": confidence,
                })
            else:
                nodes[kind].append({
                    "id": node_identifier,
                    "method": canonical_claim,
                    "method_type": value_type,
                    "inputs": unique_strings(value for member in members for value in by_id[member].get("inputs", [])),
                    "outputs": unique_strings(value for member in members for value in by_id[member].get("outputs", [])),
                    "evidence_refs": refs,
                    "confidence": confidence,
                })

    links: list[dict[str, Any]] = []
    seen_links: set[tuple[str, str, str]] = set()
    raw_links = plan.get("links", [])
    for link in raw_links if isinstance(raw_links, list) else []:
        if not isinstance(link, dict):
            continue
        problem_group = str(link.get("problem_group") or "")
        method_group = str(link.get("method_group") or "")
        problem_id = group_maps["problem"].get(problem_group)
        method_id = group_maps["method"].get(method_group)
        if not problem_id or not method_id:
            continue
        relation = str(link.get("relation") or "partially_addresses").strip().lower()
        relation = RELATION_ALIASES.get(relation, relation)
        if relation not in ALLOWED_RELATIONS:
            relation = "partially_addresses"
        relation_sources = set(unique_strings(link.get("relation_evidence_source_ids") or []))
        problem_sources = relation_sources & group_members["problem"].get(problem_group, set())
        method_sources = relation_sources & group_members["method"].get(method_group, set())
        if not problem_sources or not method_sources:
            continue
        evidence_refs = unique_strings(
            ref
            for source_id in sorted(problem_sources | method_sources)
            for ref in by_id[source_id].get("evidence_refs", [])
        )
        key = (problem_id, method_id, relation)
        if not evidence_refs or key in seen_links:
            continue
        seen_links.add(key)
        links.append({
            "problem_id": problem_id,
            "method_id": method_id,
            "relation": relation,
            "link_type": "evidence_supported",
            "rationale": "Qwen-VL-Max selected type-consistent source nodes as direct relation evidence.",
            "evidence_refs": evidence_refs,
            "confidence": _confidence(link.get("confidence")),
        })

    result = {
        "paper_research_problems": nodes["problem"],
        "paper_methods": nodes["method"],
        "problem_method_links": links,
        "unresolved_or_ambiguous": [],
    }
    selected_ids = used["problem"] | used["method"]
    audit = {
        "input_problem_count": sum(record.get("kind") == "problem" for record in records),
        "input_method_count": sum(record.get("kind") == "method" for record in records),
        "retained_problem_count": len(nodes["problem"]),
        "retained_method_count": len(nodes["method"]),
        "retained_link_count": len(links),
        "selected_source_ids": sorted(selected_ids),
        "dropped_source_ids": sorted(set(by_id) - selected_ids),
        "rejected_plan_items": rejected,
        "budgets": {
            "target_problems": target_problems,
            "target_methods": target_methods,
            "max_problems": max_problems,
            "max_methods": max_methods,
        },
        "policy": "Model selects and rewrites only supplied frozen-L2 candidates; code validates IDs, caps counts, preserves exact evidence refs, and never reads gold files.",
    }
    return result, audit


def _json_recovery_summary(cache_dir: Path) -> dict[str, Any]:
    files = sorted(cache_dir.rglob("*.recovery.json")) if cache_dir.exists() else []
    return {
        "recovered_response_count": len(files),
        "audit_files": [str(path.relative_to(cache_dir)) for path in files],
    }


def run_semantic_core_from_frozen_6000(
    config: PipelineConfig,
    source_condition_dir: Path,
    *,
    target_problems: int = 6,
    target_methods: int = 8,
    max_problems: int = 10,
    max_methods: int = 12,
    context_chars: int = 18000,
) -> Path:
    """Run one Qwen-VL-Max L2.5 call over the existing 6k multimodal result."""
    required = {
        "manifest": source_condition_dir / "00_manifest.json",
        "l1": source_condition_dir / "01_l1_chunk_results.json",
        "evidence": source_condition_dir / "02_evidence_index.json",
        "l2": source_condition_dir / "03_l2_paper_merge.json",
    }
    missing = [str(path) for key, path in required.items() if key != "manifest" and not path.exists()]
    if missing:
        raise FileNotFoundError(
            "The semantic-core condition reuses added_chunk_6000 and will not regenerate it. "
            f"Missing required files: {missing}"
        )
    if not (1 <= target_problems <= max_problems and 1 <= target_methods <= max_methods):
        raise ValueError("Semantic-core target counts must be positive and no larger than their maxima")

    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.cache_dir.mkdir(parents=True, exist_ok=True)
    l1 = read_json(required["l1"])
    evidence_index = read_json(required["evidence"])
    frozen_l2 = read_json(required["l2"])
    if not isinstance(l1, list) or not isinstance(evidence_index, dict) or not isinstance(frozen_l2, dict):
        raise ValueError("Frozen added_chunk_6000 support files have unexpected JSON types")

    records = build_semantic_core_records(frozen_l2, l1, evidence_index)
    problem_count = sum(record.get("kind") == "problem" for record in records)
    method_count = sum(record.get("kind") == "method" for record in records)
    if not problem_count or not method_count:
        raise ValueError(
            f"Frozen 6k L2 produced no usable evidence-grounded candidates: problems={problem_count}, methods={method_count}"
        )

    context = select_primary_context(read_text(config.markdown_path), max_chars=context_chars)
    client = ResilientQwenVLClient(
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
    progress(
        f"Semantic core L2.5 started: frozen candidates problems={problem_count}, methods={method_count}",
        config.verbose,
    )
    plan = client.chat_json(
        stage="semantic_core_compact_plan_v1",
        system_prompt=SEMANTIC_CORE_SYSTEM,
        user_text=semantic_core_user_prompt(
            context,
            json.dumps(records, ensure_ascii=False, separators=(",", ":")),
            target_problems=target_problems,
            target_methods=target_methods,
            max_problems=max_problems,
            max_methods=max_methods,
        ),
        images=[],
        extra_cache_key={
            "protocol": "semantic_core_v1",
            "source_condition": "added_chunk_6000",
            "target_problems": target_problems,
            "target_methods": target_methods,
            "max_problems": max_problems,
            "max_methods": max_methods,
            "context_chars": context_chars,
            "record_count": len(records),
        },
    )
    core_l2, audit = materialize_semantic_core_plan(
        plan,
        records,
        target_problems=target_problems,
        target_methods=target_methods,
        max_problems=max_problems,
        max_methods=max_methods,
    )

    if required["manifest"].exists():
        write_json(config.output_dir / "00_manifest.json", read_json(required["manifest"]))
    write_json(config.output_dir / "01_l1_chunk_results.json", l1)
    write_json(config.output_dir / "02_evidence_index.json", evidence_index)
    write_json(config.output_dir / "03_semantic_core_plan.json", plan)
    write_json(config.output_dir / "03_semantic_core_audit.json", audit)
    write_json(config.output_dir / "03_l2_paper_merge.json", core_l2)

    final = deterministic_l3_audit(core_l2, evidence_index)
    relation_refs = {
        (
            str(link.get("problem_id") or ""),
            str(link.get("method_id") or ""),
            str(link.get("relation") or "partially_addresses"),
        ): list(link.get("evidence_refs") or [])
        for link in core_l2.get("problem_method_links", [])
        if isinstance(link, dict)
    }
    for link in final.get("problem_method_links", []):
        key = (
            str(link.get("problem_id") or ""),
            str(link.get("method_id") or ""),
            str(link.get("relation") or "partially_addresses"),
        )
        link["evidence_refs"] = relation_refs.get(key, [])
    final = attach_strict_audit_diagnostics(final, build_exact_evidence_map(l1, evidence_index))
    write_json(config.output_dir / "04_final_extraction.json", final)
    write_json(config.output_dir / "condition_info.json", {
        "condition": "added_semantic_core_6000",
        "definition": "Frozen 6k multimodal L1/L2 output followed by one Qwen-VL-Max semantic canonicalization call and deterministic L3 audit.",
        "semantic_core_protocol": "semantic_core_v1",
        "source_condition": "added_chunk_6000",
        "source_l1_path": str(required["l1"]),
        "source_l2_path": str(required["l2"]),
        "source_l1_sha256": file_sha256(required["l1"]),
        "source_l2_sha256": file_sha256(required["l2"]),
        "model": config.model,
        "backend": "Qwen-VL-Max",
        "input_modality": "multimodal evidence inherited from frozen added_chunk_6000; L2.5 consumes text records and source context",
        "max_chars_per_chunk": 6000,
        "overlap_chars": config.overlap_chars,
        "max_images_per_chunk": config.max_images_per_chunk,
        "target_problems": target_problems,
        "target_methods": target_methods,
        "max_problems": max_problems,
        "max_methods": max_methods,
        "context_chars": context_chars,
        "relation_completion": False,
        "original_method_modified": True,
        "gold_file_read_by_method": False,
        "development_evaluation_scope": "papers 1-30 when run with --start 1 --end 30",
        "dry_run": config.dry_run,
        "semantic_core_audit": audit,
        "json_recovery": _json_recovery_summary(config.cache_dir),
    })
    progress(
        f"Semantic core completed: problems={audit['retained_problem_count']}, methods={audit['retained_method_count']}, links={audit['retained_link_count']}",
        config.verbose,
    )
    return config.output_dir / "04_final_extraction.json"
