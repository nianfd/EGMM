from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from revision.experiments import (
    run_controlled_pipeline,
    create_masked_image_dir,
    run_l3_on_frozen_l2,
    run_no_l3_from_frozen_l2,
    run_relation_completion_from_frozen_l2,
    run_schema_matched_baseline,
)
from revision.io_utils import discover_papers, read_json, write_json
from revision.runner import build_qwen_config, qwen_key
from revision.contracts import validate_condition_output
from revision.semantic_core import run_semantic_core_from_frozen_6000


AVAILABLE_CONDITIONS = [
    "added_global_eg_merge",
    "added_bm25_rag_text",
    "added_schema_text_only",
    "added_visual_masked",
    "added_chunk_6000",
    "added_chunk_9000",
    "added_chunk_12000",
    "added_l3_off_frozen_l2",
    "added_l3_on_frozen_l2",
    "added_relation_completion_off",
    "added_relation_completion_on",
    "added_semantic_core_6000",
]

# In the revised paper the complete 6,000-character multimodal pipeline is the
# proposed method.  A bare command therefore targets that condition only.  The
# semantic-core variant remains selectable as a compression ablation, but it is
# no longer treated as the proposed configuration.
DEFAULT_CONDITIONS = ["added_chunk_6000"]

PROTECTED_ORIGINAL_CONDITIONS = {
    "original_main",
    "original_baseline_oneshot_mineru",
    "original_ablation_text_only",
    "original_ablation_no_l3",
    "original_ablation_large_chunk",
    "proposed_full",
    "baseline_oneshot_mineru",
    "ablation_text_only",
    "ablation_no_l3",
    "ablation_large_chunk",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run fair baselines and single-variable major-revision experiments.")
    parser.add_argument("--root-dir", required=True)
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--end", type=int, default=100)
    parser.add_argument("--conditions", nargs="*", default=DEFAULT_CONDITIONS)
    parser.add_argument("--api-key")
    parser.add_argument("--base-url", default="https://dashscope.aliyuncs.com/compatible-mode/v1")
    parser.add_argument("--model", default="qwen-vl-max", choices=["qwen-vl-max"])
    parser.add_argument("--overlap-chars", type=int, default=900)
    parser.add_argument("--max-images-per-chunk", type=int, default=4)
    parser.add_argument("--request-timeout", type=int, default=600)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--rag-top-k", type=int, default=8)
    parser.add_argument("--semantic-core-target-problems", type=int, default=6)
    parser.add_argument("--semantic-core-target-methods", type=int, default=8)
    parser.add_argument("--semantic-core-max-problems", type=int, default=10)
    parser.add_argument("--semantic-core-max-methods", type=int, default=12)
    parser.add_argument("--semantic-core-context-chars", type=int, default=18000)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def parse_conditions(values: list[str]) -> list[str]:
    result = []
    for value in values:
        result.extend(item.strip() for item in value.split(",") if item.strip())
    return result


def ordered_conditions(values: list[str]) -> list[str]:
    result = list(values)
    frozen_dependents = {
        "added_l3_off_frozen_l2", "added_l3_on_frozen_l2",
        "added_relation_completion_off", "added_relation_completion_on",
    }
    if set(result) & frozen_dependents and "added_chunk_9000" not in result:
        result.insert(0, "added_chunk_9000")
        print("Added prerequisite condition: added_chunk_9000", flush=True)
    elif "added_chunk_9000" in result:
        result.remove("added_chunk_9000")
        result.insert(0, "added_chunk_9000")
    return result


def require_qwen_vl_max_provenance(
    condition_dir: Path,
    condition: str,
    *,
    allow_dry_run: bool = False,
) -> None:
    info_path = condition_dir / "condition_info.json"
    if not info_path.exists():
        raise RuntimeError(
            f"Refusing to reuse existing {condition}: missing {info_path}. Remove/rename this stale addition and rerun."
        )
    model = str(read_json(info_path).get("model") or "").lower()
    if model != "qwen-vl-max":
        raise RuntimeError(
            f"Refusing to reuse existing {condition}: recorded model is {model or 'UNKNOWN'}, not qwen-vl-max. "
            "This includes earlier qwen3-vl-plus pilot outputs. Keep them archived or move them aside, "
            "then rerun this addition with the original-paper backend."
        )
    if read_json(info_path).get("dry_run") is True and not allow_dry_run:
        raise RuntimeError(
            f"Refusing to reuse existing {condition}: it is marked dry_run=true. "
            "Preflight outputs are not experimental results."
        )


def report_quality_warnings(condition: str, validation: dict) -> int:
    """Log non-blocking model-quality defects without changing predictions."""
    warnings = validation.get("warnings", [])
    count = len(warnings) if isinstance(warnings, list) else 0
    if count:
        print(
            f"  quality warnings {condition}: {count} "
            "(recorded; the metric engines will score the saved prediction as-is)",
            flush=True,
        )
    return count


def main() -> int:
    args = parse_args()
    conditions = ordered_conditions(parse_conditions(args.conditions))
    protected = sorted(set(conditions) & PROTECTED_ORIGINAL_CONDITIONS)
    if protected:
        raise SystemExit(
            "Refusing to rerun/overwrite protected original Qwen-VL conditions: "
            f"{protected}. Use the original commands and outputs/comparison_experiments paths."
        )
    unknown = sorted(set(conditions) - set(AVAILABLE_CONDITIONS))
    if unknown:
        raise SystemExit(f"Unknown conditions: {unknown}")
    api_key = qwen_key(args.api_key)
    if not api_key and not args.dry_run:
        raise SystemExit("Set DASHSCOPE_API_KEY or QWEN_API_KEY; plaintext key files are not supported")
    root = Path(args.root_dir).resolve()
    papers = discover_papers(root, args.start, args.end)
    addition_dir_name = "major_revision_preflight" if args.dry_run else "major_revision_additions"
    summary = []
    for paper_pos, paper_dir in enumerate(papers, start=1):
        print(f"Paper {paper_pos}/{len(papers)}: {paper_dir.name}", flush=True)
        original = paper_dir / "outputs"
        for condition in conditions:
            output_dir = original / addition_dir_name / condition
            final_candidates = [output_dir / "04_final_extraction.json", output_dir / "result.json"]
            if args.skip_existing and any(path.exists() for path in final_candidates):
                require_qwen_vl_max_provenance(
                    output_dir,
                    condition,
                    allow_dry_run=args.dry_run,
                )
                validation = validate_condition_output(
                    output_dir,
                    condition,
                    allow_empty=args.dry_run,
                )
                if not validation["valid"]:
                    raise RuntimeError(
                        f"Existing {condition} is metric-incompatible: {validation['errors']}"
                    )
                write_json(output_dir / "metric_compatibility.json", validation)
                quality_warning_count = report_quality_warnings(condition, validation)
                summary.append({
                    "paper_id": paper_dir.name,
                    "condition": condition,
                    "result": str(next(path for path in final_candidates if path.exists())),
                    "metric_compatibility": "passed",
                    "quality_warning_count": quality_warning_count,
                    "run_status": "skipped_existing",
                })
                print(f"  skip {condition}", flush=True)
                continue
            config = build_qwen_config(
                paper_dir,
                output_dir,
                api_key=api_key or "dry-run",
                base_url=args.base_url,
                model=args.model,
                max_chars_per_chunk=9000,
                overlap_chars=args.overlap_chars,
                max_images_per_chunk=args.max_images_per_chunk,
                request_timeout=args.request_timeout,
                max_retries=args.max_retries,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                dry_run=args.dry_run,
                quiet=args.quiet,
            )
            print(f"  run {condition}", flush=True)
            if condition in {"added_l3_off_frozen_l2", "added_l3_on_frozen_l2"}:
                control = original / addition_dir_name / "added_chunk_9000"
                require_qwen_vl_max_provenance(
                    control, "added_chunk_9000 control", allow_dry_run=args.dry_run
                )
                files = [control / "03_l2_paper_merge.json", control / "01_l1_chunk_results.json", control / "02_evidence_index.json"]
                if not all(path.exists() for path in files):
                    raise FileNotFoundError(f"Run added_chunk_9000 first; its Qwen-VL-Max L2/L1/evidence outputs are required for {condition}: {files}")
                function = run_no_l3_from_frozen_l2 if condition == "added_l3_off_frozen_l2" else run_l3_on_frozen_l2
                result_path = function(config, *files)
            elif condition in {"added_relation_completion_off", "added_relation_completion_on"}:
                control = original / addition_dir_name / "added_chunk_9000"
                require_qwen_vl_max_provenance(
                    control, "added_chunk_9000 control", allow_dry_run=args.dry_run
                )
                frozen_l2 = control / "03_l2_paper_merge.json"
                frozen_l1 = control / "01_l1_chunk_results.json"
                frozen_evidence = control / "02_evidence_index.json"
                if not frozen_l2.exists() or not frozen_l1.exists() or not frozen_evidence.exists():
                    raise FileNotFoundError("Run added_chunk_9000 first; its Qwen-VL-Max L1, L2 and evidence index are required")
                result_path = run_relation_completion_from_frozen_l2(
                    config, frozen_l2, frozen_l1, frozen_evidence, condition.endswith("_on")
                )
            elif condition == "added_semantic_core_6000":
                # Reuse the already generated 6k multimodal L1/L2 files.  This
                # condition makes one new Qwen-VL-Max L2.5 call per paper and
                # never reads the Qwen3.8-Max gold file.
                source = original / "major_revision_additions" / "added_chunk_6000"
                require_qwen_vl_max_provenance(
                    source, "added_chunk_6000 source", allow_dry_run=False
                )
                result_path = run_semantic_core_from_frozen_6000(
                    replace(config, max_chars_per_chunk=6000),
                    source,
                    target_problems=args.semantic_core_target_problems,
                    target_methods=args.semantic_core_target_methods,
                    max_problems=args.semantic_core_max_problems,
                    max_methods=args.semantic_core_max_methods,
                    context_chars=args.semantic_core_context_chars,
                )
            elif condition in {"added_global_eg_merge", "added_bm25_rag_text"}:
                result_path = run_schema_matched_baseline(config, condition, top_k=args.rag_top_k)
            elif condition == "added_schema_text_only":
                result_path = run_controlled_pipeline(replace(config, max_images_per_chunk=0), condition)
            elif condition == "added_visual_masked":
                masked_dir = create_masked_image_dir(paper_dir / "images", output_dir / "masked_images")
                result_path = run_controlled_pipeline(replace(config, image_dir=masked_dir), condition)
            else:
                chunk_size = int(condition.rsplit("_", 1)[1])
                result_path = run_controlled_pipeline(replace(config, max_chars_per_chunk=chunk_size), condition)
            validation = validate_condition_output(output_dir, condition, allow_empty=args.dry_run)
            write_json(output_dir / "metric_compatibility.json", validation)
            if not validation["valid"]:
                raise RuntimeError(
                    f"{paper_dir.name}:{condition} produced a metric-incompatible output: {validation['errors']}"
                )
            quality_warning_count = report_quality_warnings(condition, validation)
            summary.append({
                "paper_id": paper_dir.name,
                "condition": condition,
                "result": str(result_path),
                "metric_compatibility": "passed",
                "quality_warning_count": quality_warning_count,
                "run_status": "generated",
            })
    summary_name = "revision_experiment_preflight_outputs.json" if args.dry_run else "revision_experiment_outputs.json"
    write_json(root / summary_name, {"runs": summary, "resolved_args": vars(args)})
    print(json.dumps({"completed_runs": len(summary)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
