from __future__ import annotations

import hashlib
import inspect
from pathlib import Path
from typing import Any

import paper_mining.pipeline as main_pipeline
from paper_mining.config import PipelineConfig
from paper_mining.io_utils import read_text
from paper_mining.qwenvl_client import QwenVLClient

from .annotation_schema import (
    EvidenceAnchor,
    GoldLink,
    GoldMethod,
    GoldProblem,
    PaperGoldAnnotation,
    ReproducibilityFields,
)
from .evidence import EvidenceRecord, build_exact_evidence_map, resolve_refs
from .io_utils import read_json, stable_hash, write_json
from .qwen38_main_aligned_client import Qwen38MainAlignedClient


PROBLEM_TYPES = {
    "task_gap", "data_gap", "method_gap", "evaluation_gap",
    "application_constraint", "other",
}
METHOD_TYPES = {
    "architecture", "algorithm_step", "representation", "training_objective",
    "data_processing", "evaluation_protocol", "implementation_detail", "other",
}
RELATIONS = {"directly_addresses", "partially_addresses", "evaluates", "motivates"}


class _AlignedClientFactory:
    def __init__(self, progress_seconds: float) -> None:
        self.progress_seconds = progress_seconds
        self.instance: Qwen38MainAlignedClient | None = None

    def __call__(self, **kwargs: Any) -> Qwen38MainAlignedClient:
        self.instance = Qwen38MainAlignedClient(
            **kwargs,
            progress_seconds=self.progress_seconds,
        )
        return self.instance


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def runtime_alignment_report(config: PipelineConfig) -> dict[str, Any]:
    """Record the exact main-pipeline objects and parameters used by this run."""
    return {
        "alignment_definition": (
            "The unchanged paper_mining.pipeline.run_pipeline implementation is executed. "
            "Only its QwenVLClient constructor is injected with the Qwen3.8-Max transport adapter."
        ),
        "pipeline_function_module": main_pipeline.run_pipeline.__module__,
        "pipeline_function_name": main_pipeline.run_pipeline.__name__,
        "pipeline_source_sha256": _sha256_text(inspect.getsource(main_pipeline.run_pipeline)),
        "l1_system_prompt_sha256": _sha256_text(main_pipeline.L1_SYSTEM),
        "l2_system_prompt_sha256": _sha256_text(main_pipeline.L2_SYSTEM),
        "relation_completion_system_prompt_sha256": _sha256_text(
            main_pipeline.RELATION_COMPLETION_SYSTEM
        ),
        "l1_user_prompt_factory_module": main_pipeline.l1_user_prompt.__module__,
        "l2_user_prompt_factory_module": main_pipeline.l2_user_prompt.__module__,
        "relation_user_prompt_factory_module": main_pipeline.relation_completion_user_prompt.__module__,
        "original_chat_json_module": QwenVLClient.chat_json.__module__,
        "original_chat_json_inherited_unchanged": (
            Qwen38MainAlignedClient.chat_json is QwenVLClient.chat_json
        ),
        "model": config.model,
        "max_chars_per_chunk": config.max_chars_per_chunk,
        "overlap_chars": config.overlap_chars,
        "max_images_per_chunk": config.max_images_per_chunk,
        "max_tokens": config.max_tokens,
        "temperature": config.temperature,
        "enable_relation_completion": config.enable_relation_completion,
        "multimodal": True,
    }


def _record_to_anchor(record: EvidenceRecord) -> EvidenceAnchor:
    text_cues: list[str] = []
    visual_cues: list[str] = []
    for item in record.evidence:
        cue = str(item.get("quote_or_visual_cue") or "").strip()
        if not cue:
            continue
        if str(item.get("source") or "").strip().lower() == "image":
            visual_cues.append(cue)
        else:
            text_cues.append(cue)
    has_text = bool(text_cues)
    has_visual = bool(visual_cues and record.chunk_images)
    if has_text and has_visual:
        support_type = "mixed"
    elif has_visual:
        support_type = "visual"
    else:
        support_type = "text"
    quote = "; ".join(text_cues) or record.claim or "; ".join(visual_cues)
    return EvidenceAnchor(
        span_ids=[],
        quote=quote,
        visual_asset_ids=list(record.chunk_images) if has_visual else [],
        visual_cue="; ".join(visual_cues),
        support_type=support_type,
    )


def _anchors_from_refs(
    refs: list[str],
    evidence_map: dict[str, EvidenceRecord],
    expected_kind: str | None,
) -> list[EvidenceAnchor]:
    resolved, _ = resolve_refs(refs, evidence_map, expected_kind=expected_kind)
    anchors: list[EvidenceAnchor] = []
    seen: set[str] = set()
    for record in resolved:
        anchor = _record_to_anchor(record)
        key = stable_hash(anchor.model_dump(mode="json"))
        if key not in seen:
            seen.add(key)
            anchors.append(anchor)
    return anchors


def _visual_dependency(anchors: list[EvidenceAnchor]) -> str:
    has_visual = any(anchor.visual_asset_ids and anchor.visual_cue for anchor in anchors)
    has_text = any(anchor.support_type in {"text", "mixed"} and anchor.quote for anchor in anchors)
    if has_visual and has_text:
        return "mixed"
    if has_visual:
        return "visual_dependent"
    return "text_sufficient"


def _relation(value: Any) -> str:
    text = str(value or "").strip().lower()
    aliases = {
        "addresses": "directly_addresses",
        "implements": "directly_addresses",
        "mitigates": "partially_addresses",
        "supports": "partially_addresses",
    }
    text = aliases.get(text, text)
    return text if text in RELATIONS else "partially_addresses"


def main_output_to_gold(
    paper_id: str,
    final_result: dict[str, Any],
    l1_results: list[dict[str, Any]],
    evidence_index: dict[str, Any],
) -> PaperGoldAnnotation:
    """Package the Qwen3.8-Max main-pipeline result in the existing gold schema.

    Claims, node IDs, relation endpoints, relation labels, and reproducibility
    fields are taken from ``04_final_extraction.json``.  Evidence anchors are a
    deterministic projection of the exact L1 ``chunk_id:local_atom_id`` records.
    No extra model call and no semantic annotation rule is introduced here.
    """
    evidence_map = build_exact_evidence_map(l1_results, evidence_index)
    problems: list[GoldProblem] = []
    methods: list[GoldMethod] = []
    problem_id_map: dict[str, str] = {}
    method_id_map: dict[str, str] = {}

    for index, item in enumerate(final_result.get("final_research_problems", []), start=1):
        if not isinstance(item, dict):
            continue
        claim = str(item.get("problem") or "").strip()
        if not claim:
            continue
        old_id = str(item.get("id") or f"RP{index}").strip()
        node_id = old_id or f"RP{index}"
        refs = [str(value) for value in item.get("evidence_refs", []) if isinstance(value, str)]
        anchors = _anchors_from_refs(refs, evidence_map, expected_kind="problem")
        problem_type = str(item.get("problem_type") or "other").strip()
        if problem_type not in PROBLEM_TYPES:
            problem_type = "other"
        problems.append(GoldProblem(
            id=node_id,
            claim=claim,
            problem_type=problem_type,
            evidence=anchors,
            visual_dependency=_visual_dependency(anchors),
        ))
        problem_id_map[old_id] = node_id

    for index, item in enumerate(final_result.get("final_methods", []), start=1):
        if not isinstance(item, dict):
            continue
        claim = str(item.get("method") or "").strip()
        if not claim:
            continue
        old_id = str(item.get("id") or f"M{index}").strip()
        node_id = old_id or f"M{index}"
        refs = [str(value) for value in item.get("evidence_refs", []) if isinstance(value, str)]
        anchors = _anchors_from_refs(refs, evidence_map, expected_kind="method")
        method_type = str(item.get("method_type") or "other").strip()
        if method_type not in METHOD_TYPES:
            method_type = "other"
        raw_repro = item.get("reproducibility_fields")
        repro = raw_repro if isinstance(raw_repro, dict) else {}
        methods.append(GoldMethod(
            id=node_id,
            claim=claim,
            method_type=method_type,
            reproducibility_fields=ReproducibilityFields(
                inputs=[str(value) for value in repro.get("inputs", []) if value],
                outputs=[str(value) for value in repro.get("outputs", []) if value],
                procedure=[str(value) for value in repro.get("procedure", []) if value],
                objective_or_metric=[
                    str(value) for value in repro.get("objective_or_metric", []) if value
                ],
            ),
            evidence=anchors,
            visual_dependency=_visual_dependency(anchors),
        ))
        method_id_map[old_id] = node_id

    problem_evidence = {item.id: item.evidence for item in problems}
    method_evidence = {item.id: item.evidence for item in methods}
    links: list[GoldLink] = []
    seen_links: set[tuple[str, str, str]] = set()
    for item in final_result.get("problem_method_links", []):
        if not isinstance(item, dict):
            continue
        problem_id = problem_id_map.get(str(item.get("problem_id") or ""))
        method_id = method_id_map.get(str(item.get("method_id") or ""))
        if not problem_id or not method_id:
            continue
        relation = _relation(item.get("relation"))
        key = (problem_id, method_id, relation)
        if key in seen_links:
            continue
        seen_links.add(key)
        anchors = [*problem_evidence.get(problem_id, []), *method_evidence.get(method_id, [])]
        deduplicated: list[EvidenceAnchor] = []
        seen_anchors: set[str] = set()
        for anchor in anchors:
            anchor_key = stable_hash(anchor.model_dump(mode="json"))
            if anchor_key not in seen_anchors:
                seen_anchors.add(anchor_key)
                deduplicated.append(anchor)
        links.append(GoldLink(
            problem_id=problem_id,
            method_id=method_id,
            relation=relation,
            evidence=deduplicated,
            visual_dependency=_visual_dependency(deduplicated),
        ))

    quality = final_result.get("quality_report")
    difficult: list[str] = []
    if isinstance(quality, dict):
        for key in ("main_limitations", "recommended_human_checks"):
            values = quality.get(key, [])
            if isinstance(values, list):
                difficult.extend(str(value) for value in values if value)
    return PaperGoldAnnotation(
        paper_id=paper_id,
        research_problems=problems,
        methods=methods,
        problem_method_links=links,
        difficult_or_ambiguous_cases=difficult,
    )


def run_main_aligned_gold_pipeline(
    config: PipelineConfig,
    *,
    progress_seconds: float = 30.0,
) -> dict[str, Any]:
    """Run the exact main pipeline with the Qwen3.8-Max transport adapter."""
    if config.model.lower() != Qwen38MainAlignedClient.REQUIRED_MODEL:
        raise ValueError("Main-aligned gold generation requires model=qwen3.8-max")
    factory = _AlignedClientFactory(progress_seconds)
    original_client_class = main_pipeline.QwenVLClient
    main_pipeline.QwenVLClient = factory  # type: ignore[assignment]
    try:
        final_result = main_pipeline.run_pipeline(config)
    finally:
        main_pipeline.QwenVLClient = original_client_class

    l1_results = read_json(config.output_dir / "01_l1_chunk_results.json")
    evidence_index = read_json(config.output_dir / "02_evidence_index.json")
    gold = main_output_to_gold(config.paper_dir.name, final_result, l1_results, evidence_index)
    write_json(config.output_dir / "human_gold.json", gold.model_dump(mode="json"))

    manifest = read_json(config.output_dir / "00_manifest.json")
    submitted_images = sorted({
        Path(value).name
        for chunk in manifest.get("chunks", [])
        if isinstance(chunk, dict)
        for value in chunk.get("images", [])
        if value
    })
    image_files = sorted(
        path.name for path in config.image_dir.iterdir()
        if path.is_file()
    )
    alignment = runtime_alignment_report(config)
    metadata = {
        "annotation_status": "qwen38_max_main_method_aligned_gold",
        "annotation_backend": "qwen3.8-max",
        "annotation_definition": (
            "Same main-method pipeline, prompts, chunking, multimodal input rules, "
            "L2 merging, deterministic L3 audit, and relation completion; model changed to Qwen3.8-Max."
        ),
        "paper_id": config.paper_dir.name,
        "source_char_count": len(read_text(config.markdown_path)),
        "source_image_count": len(image_files),
        "pipeline_submitted_unique_image_count": len(submitted_images),
        "pipeline_submitted_unique_image_ids": submitted_images,
        "human_gold_path": str(config.output_dir / "human_gold.json"),
        "main_final_path": str(config.output_dir / "04_final_extraction.json"),
        "problem_count": len(gold.research_problems),
        "method_count": len(gold.methods),
        "link_count": len(gold.problem_method_links),
        "alignment": alignment,
        "api_calls": factory.instance.call_metadata if factory.instance else [],
        "dry_run": config.dry_run,
    }
    write_json(config.output_dir / "annotation_metadata.json", metadata)
    write_json(config.output_dir / "main_method_alignment.json", alignment)
    return metadata
