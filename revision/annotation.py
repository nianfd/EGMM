from __future__ import annotations

import json
import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from .annotation_prompts import ADJUDICATOR, ANNOTATOR_A, ANNOTATOR_B, adjudication_user_text, annotation_user_text
from .annotation_schema import PaperGoldAnnotation, empty_annotation
from .io_utils import stable_hash, write_json
from .dashscope_qwen38_client import Qwen38AnnotationClient


IMAGE_RE = re.compile(r"!\[[^\]]*]\(([^)]+)\)|<img\b[^>]*\bsrc=[\"']([^\"']+)[\"'][^>]*>", re.IGNORECASE)
HASH_LIKE_RE = re.compile(r"^[0-9a-f]{32,}$", re.IGNORECASE)
SUPPORTED_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}


def number_markdown(markdown: str) -> tuple[str, dict[str, str]]:
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", markdown) if part.strip()]
    numbered: list[str] = []
    span_map: dict[str, str] = {}
    for index, paragraph in enumerate(paragraphs, start=1):
        span_id = f"P{index:04d}"
        rendered = f"[{span_id}] {paragraph}"
        numbered.append(rendered)
        span_map[span_id] = paragraph
    return "\n\n".join(numbered), span_map


def images_referenced_in_markdown(markdown: str, image_dir: Path) -> list[Path]:
    supported = {".png", ".jpg", ".jpeg", ".webp"}
    by_name = {
        path.name: path for path in image_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in supported
    }
    selected: list[Path] = []
    for match in IMAGE_RE.finditer(markdown):
        raw = match.group(1) or match.group(2) or ""
        name = Path(raw.split("?", 1)[0].split("#", 1)[0]).name
        path = by_name.get(name)
        if path and path not in selected:
            selected.append(path)
    return selected


def select_images(markdown: str, image_dir: Path) -> list[Path]:
    referenced = images_referenced_in_markdown(markdown, image_dir)
    all_images = sorted(path for path in image_dir.rglob("*") if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"})
    ordered = referenced + [path for path in all_images if path not in referenced]
    return ordered


def normalize_visual_asset_ids(
    annotation: PaperGoldAnnotation,
    valid_image_ids: set[str],
    *,
    fuzzy_threshold: float = 0.94,
    fuzzy_margin: float = 0.10,
) -> dict[str, Any]:
    """Map model-produced visual IDs only onto the submitted image manifest.

    Exact filenames are preserved. A missing extension is completed only when
    the stem maps to one submitted image. Hash-like transcription errors are
    corrected only when one candidate is both highly similar and clearly
    separated from the runner-up. Anything else is left unresolved so the
    strict verifier still fails instead of silently inventing evidence.
    """
    exact = {value.casefold(): value for value in valid_image_ids}
    by_stem: dict[str, list[str]] = {}
    for value in valid_image_ids:
        by_stem.setdefault(Path(value).stem.casefold(), []).append(value)
    candidates = sorted(valid_image_ids)
    actions: list[dict[str, Any]] = []
    unresolved: list[str] = []

    def canonicalize(raw: str) -> tuple[str, str | None, float | None]:
        cleaned = str(raw).strip().strip('"\'')
        # Accept a model copying a Markdown path, but retain only a basename
        # that must subsequently resolve against this paper's manifest.
        basename = cleaned.replace("\\", "/").rsplit("/", 1)[-1]
        direct = exact.get(basename.casefold())
        if direct is not None:
            return direct, "case_or_path_normalization" if direct != raw else None, 1.0

        stem = Path(basename).stem if Path(basename).suffix.lower() in SUPPORTED_IMAGE_SUFFIXES else basename
        stem_matches = by_stem.get(stem.casefold(), [])
        if len(stem_matches) == 1:
            return stem_matches[0], "unique_exact_stem", 1.0

        if HASH_LIKE_RE.fullmatch(stem):
            ranked = sorted(
                (
                    SequenceMatcher(None, stem.casefold(), Path(value).stem.casefold()).ratio(),
                    value,
                )
                for value in candidates
                if HASH_LIKE_RE.fullmatch(Path(value).stem)
            )
            if ranked:
                best_score, best_value = ranked[-1]
                second_score = ranked[-2][0] if len(ranked) > 1 else 0.0
                if best_score >= fuzzy_threshold and best_score - second_score >= fuzzy_margin:
                    return best_value, "unique_high_similarity_hash", best_score
        return basename, None, None

    groups = (
        ("research_problem", annotation.research_problems),
        ("method", annotation.methods),
        ("problem_method_link", annotation.problem_method_links),
    )
    for group_name, items in groups:
        for item_index, item in enumerate(items, start=1):
            for evidence_index, anchor in enumerate(item.evidence, start=1):
                normalized: list[str] = []
                for raw in anchor.visual_asset_ids:
                    mapped, method, score = canonicalize(raw)
                    if mapped not in normalized:
                        normalized.append(mapped)
                    if mapped != raw:
                        actions.append({
                            "location": f"{group_name}[{item_index}].evidence[{evidence_index}]",
                            "original": raw,
                            "normalized": mapped,
                            "method": method,
                            "similarity": round(score, 6) if score is not None else None,
                        })
                    if mapped not in valid_image_ids:
                        unresolved.append(mapped)
                anchor.visual_asset_ids = normalized
    return {
        "policy": (
            "manifest-only: exact/case/path match; unique exact stem completion; "
            f"unique hash similarity >= {fuzzy_threshold:.2f} with margin >= {fuzzy_margin:.2f}"
        ),
        "action_count": len(actions),
        "actions": actions,
        "unresolved": sorted(set(unresolved)),
    }


def verify_annotation(annotation: PaperGoldAnnotation, span_map: dict[str, str], image_ids: set[str]) -> dict[str, Any]:
    problem_id_list = [item.id for item in annotation.research_problems]
    method_id_list = [item.id for item in annotation.methods]
    problem_ids = set(problem_id_list)
    method_ids = set(method_id_list)
    errors: list[str] = []
    warnings: list[str] = []
    if len(problem_ids) != len(problem_id_list):
        errors.append("duplicate research-problem IDs")
    if len(method_ids) != len(method_id_list):
        errors.append("duplicate method IDs")
    for expected, actual in enumerate(problem_id_list, start=1):
        if actual != f"RP{expected}":
            errors.append(f"problem IDs must be contiguous RP1..RPn; position {expected} has {actual}")
    for expected, actual in enumerate(method_id_list, start=1):
        if actual != f"M{expected}":
            errors.append(f"method IDs must be contiguous M1..Mn; position {expected} has {actual}")
    anchors = []
    for item in [*annotation.research_problems, *annotation.methods, *annotation.problem_method_links]:
        anchors.extend(item.evidence)
    for anchor in anchors:
        for span_id in anchor.span_ids:
            if span_id not in span_map:
                errors.append(f"unknown span_id: {span_id}")
            elif anchor.quote and anchor.quote.lower() not in span_map[span_id].lower():
                warnings.append(f"quote not found verbatim in {span_id}: {anchor.quote[:80]}")
        for image_id in anchor.visual_asset_ids:
            if image_id not in image_ids:
                errors.append(f"unknown visual_asset_id: {image_id}")
        if anchor.support_type in {"visual", "mixed"} and not anchor.visual_asset_ids:
            errors.append("visual/mixed anchor missing visual_asset_ids")
        if anchor.support_type == "text" and not anchor.span_ids:
            errors.append("text anchor missing span_ids")
    for link in annotation.problem_method_links:
        if link.problem_id not in problem_ids:
            errors.append(f"link has unknown problem_id: {link.problem_id}")
        if link.method_id not in method_ids:
            errors.append(f"link has unknown method_id: {link.method_id}")
    return {
        "valid": not errors,
        "errors": sorted(set(errors)),
        "warnings": sorted(set(warnings)),
        "counts": {
            "research_problems": len(annotation.research_problems),
            "methods": len(annotation.methods),
            "problem_method_links": len(annotation.problem_method_links),
            "evidence_anchors": len(anchors),
        },
    }


def run_assisted_annotation(
    paper_dir: Path,
    output_dir: Path,
    client: Qwen38AnnotationClient,
) -> dict[str, Any]:
    paper_id = paper_dir.name
    markdown = (paper_dir / "full.md").read_text(encoding="utf-8", errors="replace")
    numbered, span_map = number_markdown(markdown)
    images = select_images(markdown, paper_dir / "images")
    all_image_ids = sorted(path.name for path in (paper_dir / "images").rglob("*") if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"})
    image_ids = [path.name for path in images]
    if sorted(image_ids) != all_image_ids:
        raise RuntimeError(f"Full-image submission invariant failed for paper {paper_id}")
    print(
        f"  input audit: full_markdown=True source_chars={len(markdown)} "
        f"numbered_chars={len(numbered)} all_images=True images={len(images)} multimodal={bool(images)}",
        flush=True,
    )
    common_user = annotation_user_text(paper_id, numbered, image_ids)

    draft_a, meta_a = client.parse(
        stage="qwen38_annotation_a",
        instructions=ANNOTATOR_A,
        user_text=common_user,
        schema=PaperGoldAnnotation,
        images=images,
        extra={"paper_id": paper_id, "annotator": "A"},
        dry_run_value=empty_annotation(paper_id),
    )
    draft_b, meta_b = client.parse(
        stage="qwen38_annotation_b",
        instructions=ANNOTATOR_B,
        user_text=common_user,
        schema=PaperGoldAnnotation,
        images=images,
        extra={"paper_id": paper_id, "annotator": "B"},
        dry_run_value=empty_annotation(paper_id),
    )
    adjudicated, meta_j = client.parse(
        stage="qwen38_annotation_adjudication",
        instructions=ADJUDICATOR,
        user_text=adjudication_user_text(
            paper_id,
            numbered,
            image_ids,
            json.dumps(draft_a.model_dump(mode="json"), ensure_ascii=False, indent=2),
            json.dumps(draft_b.model_dump(mode="json"), ensure_ascii=False, indent=2),
        ),
        schema=PaperGoldAnnotation,
        images=images,
        extra={"paper_id": paper_id, "annotator": "adjudicator"},
        dry_run_value=empty_annotation(paper_id),
    )

    normalization_before_repair = normalize_visual_asset_ids(adjudicated, set(image_ids))
    verification = verify_annotation(adjudicated, span_map, set(image_ids))
    repair_meta = None
    normalization_after_repair = None
    if not verification["valid"]:
        repair_instructions = ADJUDICATOR + (
            "\n\nThe previous adjudicated JSON failed deterministic anchor/ID validation. "
            "Repair every listed error while preserving only supported content."
        )
        repair_text = adjudication_user_text(
            paper_id,
            numbered,
            image_ids,
            json.dumps(adjudicated.model_dump(mode="json"), ensure_ascii=False, indent=2),
            json.dumps({"validation_errors": verification["errors"]}, ensure_ascii=False, indent=2),
        )
        adjudicated, repair_meta = client.parse(
            stage="qwen38_annotation_anchor_repair",
            instructions=repair_instructions,
            user_text=repair_text,
            schema=PaperGoldAnnotation,
            images=images,
            extra={"paper_id": paper_id, "validation_errors": verification["errors"]},
            dry_run_value=empty_annotation(paper_id),
        )
        normalization_after_repair = normalize_visual_asset_ids(adjudicated, set(image_ids))
        verification = verify_annotation(adjudicated, span_map, set(image_ids))
        if not verification["valid"]:
            raise RuntimeError(
                f"Qwen3.8-Max reference annotation for paper {paper_id} still has invalid IDs/evidence anchors after repair: "
                + json.dumps(verification["errors"], ensure_ascii=False)
            )

    final_run_meta = repair_meta if repair_meta is not None else meta_j
    final_schema_normalization = final_run_meta.get("local_schema_normalization") if isinstance(final_run_meta, dict) else None
    dropped_final_links = (
        final_schema_normalization.get("dropped_links", [])
        if isinstance(final_schema_normalization, dict)
        else []
    )
    if dropped_final_links:
        note = (
            f"LOCAL_SCHEMA_AUDIT: dropped {len(dropped_final_links)} incomplete or invalid model-generated "
            "problem-method link(s) without inferring missing relation/evidence fields; human reviewer should "
            "check whether supported relations need to be added manually."
        )
        if note not in adjudicated.difficult_or_ambiguous_cases:
            adjudicated.difficult_or_ambiguous_cases.append(note)

    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "qwen38_annotation_a.json", draft_a.model_dump(mode="json"))
    write_json(output_dir / "qwen38_annotation_b.json", draft_b.model_dump(mode="json"))
    write_json(output_dir / "human_gold.json", adjudicated.model_dump(mode="json"))
    metadata = {
        "paper_id": paper_id,
        "annotation_status": "qwen38_max_generated_model_gold",
        "gold_file": "human_gold.json",
        "annotation_backend": "qwen3.8-max",
        "experimental_backend": None,
        "input_policy": "complete MinerU Markdown plus every supported image; no text/image cap",
        "full_markdown_submitted": True,
        "all_images_submitted": True,
        "multimodal": bool(images),
        "source_markdown_sha256": stable_hash(markdown),
        "numbered_text_sha256": stable_hash(numbered),
        "numbered_span_count": len(span_map),
        "source_char_count": len(markdown),
        "submitted_char_count": len(numbered),
        "source_image_count": len(all_image_ids),
        "submitted_image_count": len(image_ids),
        "selected_visual_asset_ids": image_ids,
        "source_visual_asset_ids": all_image_ids,
        "visual_sampling_used": False,
        "visual_asset_id_normalization": {
            "before_model_repair": normalization_before_repair,
            "after_model_repair": normalization_after_repair,
        },
        "relation_schema_normalization": {
            "annotator_a": meta_a.get("local_schema_normalization") if isinstance(meta_a, dict) else None,
            "annotator_b": meta_b.get("local_schema_normalization") if isinstance(meta_b, dict) else None,
            "adjudicator": meta_j.get("local_schema_normalization") if isinstance(meta_j, dict) else None,
            "anchor_repair": repair_meta.get("local_schema_normalization") if isinstance(repair_meta, dict) else None,
            "final_dropped_link_count": len(dropped_final_links),
        },
        "runs": {"annotator_a": meta_a, "annotator_b": meta_b, "adjudicator": meta_j, "anchor_repair": repair_meta},
        "verification": verification,
    }
    write_json(output_dir / "annotation_metadata.json", metadata)
    (output_dir / "numbered_source.md").write_text(numbered, encoding="utf-8")
    return metadata
