from __future__ import annotations

import json
import math
import re
import shutil
from dataclasses import replace
from pathlib import Path
from typing import Any

from paper_mining.config import PipelineConfig
from paper_mining.io_utils import read_text, write_json
from paper_mining.markdown_mineru import make_chunks, parse_sections
from paper_mining import pipeline as protected_pipeline
from paper_mining.pipeline import build_evidence_index, build_manifest, complete_missing_relations, deterministic_l3_audit, merge_l1_results
from paper_mining.prompts import L1_SYSTEM, l1_user_prompt
from paper_mining.qwenvl_client import QwenVLClient

from .baseline_prompts import COMPACT_EG_PLAN_SYSTEM, compact_eg_plan_user
from .evidence import attach_strict_audit_diagnostics, build_exact_evidence_map
from .resilient_qwenvl import ResilientQwenVLClient


TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_-]+")


def json_recovery_summary(cache_dir: Path) -> dict[str, Any]:
    files = sorted(cache_dir.rglob("*.recovery.json")) if cache_dir.exists() else []
    return {
        "recovered_response_count": len(files),
        "audit_files": [str(path.relative_to(cache_dir)) for path in files],
        "policy": (
            "Revision-only syntax recovery retains valid parsed content or complete model-emitted "
            "objects, discards incomplete tail objects, and never guesses scientific fields."
        ),
    }


def run_l1(config: PipelineConfig) -> tuple[list[Any], list[dict[str, Any]], dict[str, Any], QwenVLClient]:
    markdown = read_text(config.markdown_path)
    chunks = make_chunks(
        parse_sections(markdown),
        config.image_dir,
        max_chars=config.max_chars_per_chunk,
        overlap_chars=config.overlap_chars,
        max_images=config.max_images_per_chunk,
    )
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
    l1_results: list[dict[str, Any]] = []
    for chunk in chunks:
        result = client.chat_json(
            stage="l1_chunk_extract",
            system_prompt=L1_SYSTEM,
            user_text=l1_user_prompt(chunk.id, chunk.section_title, chunk.text, [path.name for path in chunk.image_paths]),
            images=chunk.image_paths,
            extra_cache_key={"chunk_id": chunk.id},
        )
        result.setdefault("chunk_id", chunk.id)
        result.setdefault("section", chunk.section_title)
        l1_results.append(result)
    evidence_index = build_evidence_index(chunks, l1_results)
    return chunks, l1_results, evidence_index, client


def write_pipeline_intermediates(config: PipelineConfig, chunks: list[Any], l1: list[dict[str, Any]], evidence_index: dict[str, Any]) -> None:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(config.output_dir / "00_manifest.json", build_manifest(config, chunks))
    write_json(config.output_dir / "01_l1_chunk_results.json", l1)
    write_json(config.output_dir / "02_evidence_index.json", evidence_index)


def run_no_l3_from_frozen_l2(config: PipelineConfig, frozen_l2_path: Path, frozen_l1_path: Path, frozen_evidence_path: Path) -> Path:
    l2 = json.loads(frozen_l2_path.read_text(encoding="utf-8"))
    l1 = json.loads(frozen_l1_path.read_text(encoding="utf-8"))
    evidence_index = json.loads(frozen_evidence_path.read_text(encoding="utf-8"))
    config.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(config.output_dir / "01_l1_chunk_results.json", l1)
    write_json(config.output_dir / "02_evidence_index.json", evidence_index)
    write_json(config.output_dir / "03_l2_paper_merge.json", l2)
    # This is the genuine pre-L3 result. Evaluation code normalizes paper_* fields.
    write_json(config.output_dir / "result.json", l2)
    write_json(config.output_dir / "condition_info.json", {
        "condition": "added_l3_off_frozen_l2",
        "frozen_l2_path": str(frozen_l2_path),
        "definition": "Identical saved L2 output; no new L1/L2 model call and no L3-generated labels.",
        "backend": "frozen qwen-vl-max output; no new model call",
        "model": config.model,
        "dry_run": config.dry_run,
        "new_model_call": False,
        "original_method_modified": False,
    })
    return config.output_dir / "result.json"


def run_l3_on_frozen_l2(config: PipelineConfig, frozen_l2_path: Path, frozen_l1_path: Path, frozen_evidence_path: Path) -> Path:
    l2 = json.loads(frozen_l2_path.read_text(encoding="utf-8"))
    l1 = json.loads(frozen_l1_path.read_text(encoding="utf-8"))
    evidence_index = json.loads(frozen_evidence_path.read_text(encoding="utf-8"))
    config.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(config.output_dir / "01_l1_chunk_results.json", l1)
    write_json(config.output_dir / "02_evidence_index.json", evidence_index)
    write_json(config.output_dir / "03_l2_paper_merge.json", l2)
    result = deterministic_l3_audit(l2, evidence_index)
    write_json(config.output_dir / "04_final_extraction.json", result)
    write_json(config.output_dir / "condition_info.json", {
        "condition": "added_l3_on_frozen_l2",
        "frozen_l2_path": str(frozen_l2_path),
        "definition": "Identical saved L2 output followed by the unchanged original deterministic L3 implementation.",
        "backend": "frozen qwen-vl-max output followed by deterministic L3; no new model call",
        "model": config.model,
        "dry_run": config.dry_run,
        "new_model_call": False,
        "original_method_modified": False,
    })
    return config.output_dir / "04_final_extraction.json"


def l1_evidence_records(l1: list[dict[str, Any]], selected_chunks: set[str] | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for chunk in l1:
        chunk_id = str(chunk.get("chunk_id") or "")
        if selected_chunks is not None and chunk_id not in selected_chunks:
            continue
        for key, kind in (("research_problem_atoms", "problem"), ("method_atoms", "method")):
            for atom in chunk.get(key, []):
                if isinstance(atom, dict) and atom.get("id"):
                    rows.append({
                        "evidence_ref": f"{chunk_id}:{atom['id']}",
                        "kind": kind,
                        "claim": atom.get("claim"),
                        "type": atom.get("problem_type") or atom.get("method_type"),
                        "inputs": atom.get("inputs", []),
                        "outputs": atom.get("outputs", []),
                        "evidence": atom.get("evidence", []),
                        "section": chunk.get("section"),
                        "confidence": atom.get("confidence", 0.5),
                    })
    return rows


def bm25_rank(query: str, documents: list[tuple[str, str]], k: int) -> list[str]:
    query_tokens = [token.lower() for token in TOKEN_RE.findall(query)]
    doc_tokens = [[token.lower() for token in TOKEN_RE.findall(text)] for _, text in documents]
    n_docs = len(documents)
    avg_len = sum(len(tokens_) for tokens_ in doc_tokens) / max(n_docs, 1)
    dfs = {term: sum(1 for tokens_ in doc_tokens if term in set(tokens_)) for term in set(query_tokens)}
    scores = []
    for (doc_id, _), tokens_ in zip(documents, doc_tokens):
        frequencies = {term: tokens_.count(term) for term in set(query_tokens)}
        score = 0.0
        for term in set(query_tokens):
            df = dfs.get(term, 0)
            idf = math.log(1 + (n_docs - df + 0.5) / (df + 0.5))
            tf = frequencies.get(term, 0)
            denom = tf + 1.5 * (1 - 0.75 + 0.75 * len(tokens_) / max(avg_len, 1))
            score += idf * (tf * 2.5 / denom if denom else 0.0)
        scores.append((score, doc_id))
    scores.sort(reverse=True)
    return [doc_id for _, doc_id in scores[:k]]


def _unique_strings(values: list[Any]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "").strip()
        if text and text not in seen:
            seen.add(text)
            result.append(text)
    return result


def _confidence(value: Any, default: float = 0.5) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return min(1.0, max(0.0, number))


def compact_prompt_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Remove verbose evidence prose while preserving every model decision input."""
    return [
        {
            "evidence_ref": row["evidence_ref"],
            "kind": row["kind"],
            "claim": str(row.get("claim") or "").strip(),
            "type": str(row.get("type") or "").strip(),
            "inputs": _unique_strings(list(row.get("inputs") or [])),
            "outputs": _unique_strings(list(row.get("outputs") or [])),
            "section": str(row.get("section") or "").strip(),
            "confidence": _confidence(row.get("confidence")),
        }
        for row in records
    ]


def materialize_compact_merge_plan(
    plan: dict[str, Any],
    records: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Convert a short ID-only model plan into the full metric-compatible L2 schema."""
    by_ref = {str(row.get("evidence_ref") or ""): row for row in records if row.get("evidence_ref")}
    used: dict[str, set[str]] = {"problem": set(), "method": set()}
    group_maps: dict[str, dict[str, str]] = {"problem": {}, "method": {}}
    nodes: dict[str, list[dict[str, Any]]] = {"problem": [], "method": []}
    rejected_groups: list[dict[str, Any]] = []
    unresolved_refs = set(_unique_strings(list(plan.get("unresolved_refs") or [])))

    for kind, plan_key in (("problem", "problem_groups"), ("method", "method_groups")):
        groups = plan.get(plan_key, [])
        if not isinstance(groups, list):
            groups = []
        planned_members = {
            ref
            for group in groups
            if isinstance(group, dict)
            for ref in _unique_strings(list(group.get("members") or []))
            if ref in by_ref and by_ref[ref].get("kind") == kind
        }
        # A merge plan decides equivalence and relations, not whether valid L1
        # evidence silently disappears. Unmentioned, non-unresolved atoms are
        # therefore materialized as auditable singleton groups.
        for position, ref in enumerate(
            sorted(
                value for value, row in by_ref.items()
                if row.get("kind") == kind and value not in planned_members and value not in unresolved_refs
            ),
            start=1,
        ):
            groups.append({
                "group_id": f"AUTO_{kind}_{position}",
                "members": [ref],
                "representative": ref,
                "confidence": by_ref[ref].get("confidence", 0.5),
            })
        for group in groups:
            if not isinstance(group, dict):
                continue
            members = [
                ref for ref in _unique_strings(list(group.get("members") or []))
                if ref in by_ref and by_ref[ref].get("kind") == kind and ref not in used[kind]
            ]
            if not members:
                rejected_groups.append({"kind": kind, "group_id": group.get("group_id"), "reason": "no_valid_unused_members"})
                continue
            representative_ref = str(group.get("representative") or "")
            if representative_ref not in members:
                representative_ref = max(
                    members,
                    key=lambda ref: (_confidence(by_ref[ref].get("confidence")), len(str(by_ref[ref].get("claim") or ""))),
                )
            representative = by_ref[representative_ref]
            used[kind].update(members)
            node_id = f"RP{len(nodes[kind]) + 1}" if kind == "problem" else f"M{len(nodes[kind]) + 1}"
            raw_group_id = str(group.get("group_id") or f"{kind[0].upper()}{len(nodes[kind]) + 1}")
            if raw_group_id in group_maps[kind]:
                rejected_groups.append({"kind": kind, "group_id": raw_group_id, "reason": "duplicate_group_id_link_mapping_kept_first"})
            else:
                group_maps[kind][raw_group_id] = node_id
            group_confidence = _confidence(
                group.get("confidence"),
                default=sum(_confidence(by_ref[ref].get("confidence")) for ref in members) / len(members),
            )
            if kind == "problem":
                nodes[kind].append({
                    "id": node_id,
                    "problem": str(representative.get("claim") or "").strip(),
                    "problem_type": str(representative.get("type") or "uncertain").strip() or "uncertain",
                    "explicitness": "explicit",
                    "evidence_refs": members,
                    "confidence": group_confidence,
                })
            else:
                nodes[kind].append({
                    "id": node_id,
                    "method": str(representative.get("claim") or "").strip(),
                    "method_type": str(representative.get("type") or "uncertain").strip() or "uncertain",
                    "inputs": _unique_strings([value for ref in members for value in list(by_ref[ref].get("inputs") or [])]),
                    "outputs": _unique_strings([value for ref in members for value in list(by_ref[ref].get("outputs") or [])]),
                    "evidence_refs": members,
                    "confidence": group_confidence,
                })

    allowed_relations = {"directly_addresses", "partially_addresses", "evaluates", "motivates"}
    links: list[dict[str, Any]] = []
    seen_links: set[tuple[str, str, str]] = set()
    for link in plan.get("links", []) if isinstance(plan.get("links", []), list) else []:
        if not isinstance(link, dict):
            continue
        problem_id = group_maps["problem"].get(str(link.get("problem_group") or ""))
        method_id = group_maps["method"].get(str(link.get("method_group") or ""))
        relation = str(link.get("relation") or "partially_addresses")
        if relation not in allowed_relations:
            relation = "partially_addresses"
        refs = [ref for ref in _unique_strings(list(link.get("evidence_refs") or [])) if ref in by_ref]
        key = (problem_id or "", method_id or "", relation)
        if not problem_id or not method_id or not refs or key in seen_links:
            continue
        seen_links.add(key)
        links.append({
            "problem_id": problem_id,
            "method_id": method_id,
            "relation": relation,
            "rationale": "The compact merge plan selected these indexed records as relation evidence.",
            "evidence_refs": refs,
            "confidence": _confidence(link.get("confidence")),
        })

    result = {
        "paper_research_problems": nodes["problem"],
        "paper_methods": nodes["method"],
        "problem_method_links": links,
        "unresolved_or_ambiguous": sorted(unresolved_refs),
    }
    audit = {
        "input_record_count": len(records),
        "problem_group_count": len(nodes["problem"]),
        "method_group_count": len(nodes["method"]),
        "link_count": len(links),
        "unused_problem_refs": sorted(ref for ref, row in by_ref.items() if row.get("kind") == "problem" and ref not in used["problem"]),
        "unused_method_refs": sorted(ref for ref, row in by_ref.items() if row.get("kind") == "method" and ref not in used["method"]),
        "rejected_groups": rejected_groups,
    }
    return result, audit


def run_schema_matched_baseline(config: PipelineConfig, condition: str, top_k: int = 8) -> Path:
    # Global-EG/BM25-RAG-Text share the same L1 extraction schema and IDs. The tested factor is paper-level evidence access.
    text_config = replace(config, max_images_per_chunk=0)
    # Do not reuse an old main-method cache here. Every new model call is locked to qwen-vl-max.
    l1_config = config if condition == "added_global_eg_merge" else text_config
    chunks, l1, evidence_index, client = run_l1(l1_config)
    write_pipeline_intermediates(l1_config, chunks, l1, evidence_index)
    selected_chunks: set[str] | None = None
    if condition == "added_bm25_rag_text":
        query = "research problem limitation challenge gap proposed method architecture module algorithm training objective loss evaluation protocol"
        ranked = bm25_rank(query, [(chunk.id, chunk.text) for chunk in chunks], k=top_k)
        selected_chunks = set(ranked)
    records = l1_evidence_records(l1, selected_chunks=selected_chunks)
    compact_records = compact_prompt_records(records)
    plan = client.chat_json(
        stage=f"{condition}_compact_plan_v2",
        system_prompt=COMPACT_EG_PLAN_SYSTEM,
        user_text=compact_eg_plan_user(
            json.dumps(compact_records, ensure_ascii=False, separators=(",", ":")),
            f"condition={condition}; records={len(records)}; top_k={top_k if selected_chunks is not None else 'all'}; output_protocol=id_only_v2",
        ),
        images=[],
        extra_cache_key={
            "condition": condition,
            "record_count": len(records),
            "selected_chunks": sorted(selected_chunks or []),
            "compact_protocol": "id_only_v2",
        },
    )
    result, plan_audit = materialize_compact_merge_plan(plan, records)
    write_json(config.output_dir / "03_compact_merge_plan.json", plan)
    write_json(config.output_dir / "03_compact_merge_plan_audit.json", plan_audit)
    write_json(config.output_dir / "03_l2_paper_merge.json", result)
    # Apply the unchanged original L3 implementation, then attach non-mutating stricter diagnostics.
    final = deterministic_l3_audit(result, evidence_index)
    # FAIR_EXTRACTION_SYSTEM requires relation-level evidence_refs. The
    # unchanged historical deterministic L3 predates that field, so restore it
    # only for these new schema-matched baselines.
    relation_refs = {
        (
            str(link.get("problem_id") or ""),
            str(link.get("method_id") or ""),
            str(link.get("relation") or "partially_addresses"),
        ): list(link.get("evidence_refs") or [])
        for link in result.get("problem_method_links", [])
        if isinstance(link, dict)
    }
    for link in final.get("problem_method_links", []):
        key = (
            str(link.get("problem_id") or ""),
            str(link.get("method_id") or ""),
            str(link.get("relation") or "partially_addresses"),
        )
        link["evidence_refs"] = relation_refs.get(key, [])
    if config.enable_relation_completion:
        final = complete_missing_relations(client, final)
    strict = attach_strict_audit_diagnostics(final, build_exact_evidence_map(l1, evidence_index))
    write_json(config.output_dir / "04_final_extraction.json", strict)
    write_json(config.output_dir / "condition_info.json", {
        "condition": condition,
        "schema": "same indexed evidence IDs and graph schema as the proposed system",
        "paper_level_merge_protocol": "compact ID-only Qwen-VL merge plan followed by deterministic schema materialization",
        "compact_merge_protocol_version": "id_only_v2",
        "compact_plan_audit": plan_audit,
        "selected_chunks": sorted(selected_chunks or {chunk.id for chunk in chunks}),
        "evidence_record_count": len(records),
        "relation_completion": config.enable_relation_completion,
        "backend": "Qwen-VL",
        "original_method_modified": False,
        "model": config.model,
        "max_chars_per_chunk": config.max_chars_per_chunk,
        "overlap_chars": config.overlap_chars,
        "max_images_per_chunk": l1_config.max_images_per_chunk,
        "input_modality": "multimodal" if l1_config.max_images_per_chunk > 0 else "text_only",
        "temperature": config.temperature,
        "dry_run": config.dry_run,
        "json_recovery": json_recovery_summary(config.cache_dir),
    })
    return config.output_dir / "04_final_extraction.json"


def create_masked_image_dir(source_dir: Path, target_dir: Path) -> Path:
    try:
        from PIL import Image, ImageOps
    except ImportError as exc:
        raise RuntimeError("Pillow is required for visual_masked; run pip install -r requirements-revision.txt") from exc
    for source in source_dir.rglob("*"):
        if not source.is_file() or source.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}:
            continue
        relative = source.relative_to(source_dir)
        target = target_dir / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(source) as image:
            mode = "RGBA" if source.suffix.lower() == ".png" else "RGB"
            blank = Image.new(mode, image.size, (255, 255, 255, 255) if mode == "RGBA" else (255, 255, 255))
            blank.save(target)
    return target_dir


def run_controlled_pipeline(config: PipelineConfig, condition: str) -> Path:
    # Call the protected original pipeline unchanged, injecting only the
    # revision-only client that adds restartable JSON syntax recovery.  The
    # prompts, chunking, L1/L2/L3 logic, parameters and cache keys are unchanged.
    original_client_class = protected_pipeline.QwenVLClient
    protected_pipeline.QwenVLClient = ResilientQwenVLClient
    try:
        result = protected_pipeline.run_pipeline(config)
    finally:
        protected_pipeline.QwenVLClient = original_client_class
    write_json(config.output_dir / "condition_info.json", {
        "condition": condition,
        "model": config.model,
        "max_chars_per_chunk": config.max_chars_per_chunk,
        "overlap_chars": config.overlap_chars,
        "max_images_per_chunk": config.max_images_per_chunk,
        "temperature": config.temperature,
        "relation_completion": config.enable_relation_completion,
        "backend": "Qwen-VL",
        "original_method_modified": False,
        "dry_run": config.dry_run,
        "json_recovery": json_recovery_summary(config.cache_dir),
    })
    return config.output_dir / "04_final_extraction.json"


def run_relation_completion_from_frozen_l2(
    config: PipelineConfig,
    frozen_l2_path: Path,
    frozen_l1_path: Path,
    frozen_evidence_path: Path,
    enabled: bool,
) -> Path:
    """Rebuild the unchanged original L3 result, then toggle only Qwen-VL relation completion."""
    l2 = json.loads(frozen_l2_path.read_text(encoding="utf-8"))
    evidence_index = json.loads(frozen_evidence_path.read_text(encoding="utf-8"))
    result = deterministic_l3_audit(l2, evidence_index)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    # Keep every condition self-contained. Metric code must never fall back to
    # the protected original run's evidence index for a new Qwen-VL-Max result.
    write_json(config.output_dir / "01_l1_chunk_results.json", json.loads(frozen_l1_path.read_text(encoding="utf-8")))
    write_json(config.output_dir / "02_evidence_index.json", evidence_index)
    write_json(config.output_dir / "03_l2_paper_merge.json", l2)
    if enabled:
        client = ResilientQwenVLClient(
            api_key=config.api_key, base_url=config.base_url, model=config.model, cache_dir=config.cache_dir,
            timeout=config.request_timeout, max_retries=config.max_retries, max_tokens=config.max_tokens,
            temperature=config.temperature, dry_run=config.dry_run, verbose=config.verbose,
        )
        result = complete_missing_relations(client, result)
    write_json(config.output_dir / "04_final_extraction.json", result)
    write_json(config.output_dir / "condition_info.json", {
        "condition": "added_relation_completion_on" if enabled else "added_relation_completion_off",
        "frozen_l2_path": str(frozen_l2_path),
        "frozen_l1_path": str(frozen_l1_path),
        "frozen_evidence_path": str(frozen_evidence_path),
        "relation_completion": enabled,
        "backend": "qwen-vl-max relation-completion call on frozen graph" if enabled else "no model call; frozen qwen-vl-max graph",
        "model": config.model,
        "new_model_call": enabled,
        "original_method_modified": False,
        "dry_run": config.dry_run,
        "definition": "The graph is frozen; only optional semantic relation completion is toggled. Inferred links remain labeled inferred.",
        "json_recovery": json_recovery_summary(config.cache_dir),
    })
    return config.output_dir / "04_final_extraction.json"
