from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from revision.annotation_schema import PaperGoldAnnotation
from revision.io_utils import discover_papers, read_json, write_csv, write_json


CONDITIONS = {
    "original_main": {
        "result": "outputs/04_final_extraction.json",
        "manifest": "outputs/00_manifest.json",
        "marker": None,
        "log": None,
    },
    "original_baseline_oneshot_mineru": {
        "result": "outputs/comparison_experiments/baseline_oneshot_mineru/baseline_oneshot_mineru.json",
        "manifest": "outputs/comparison_experiments/baseline_oneshot_mineru/baseline_oneshot_mineru_manifest.json",
        "marker": "outputs/.batch_experiments/baseline_oneshot_mineru.done",
        "log": "outputs/.batch_experiments/logs/baseline_oneshot_mineru.log",
    },
    "original_ablation_text_only": {
        "result": "outputs/comparison_experiments/ablations/ablation_text_only/04_final_extraction.json",
        "manifest": "outputs/comparison_experiments/ablations/ablation_text_only/00_manifest.json",
        "marker": "outputs/.batch_experiments/ablation_text_only.done",
        "log": "outputs/.batch_experiments/logs/ablation_text_only.log",
    },
    "original_ablation_no_l3": {
        "result": "outputs/comparison_experiments/ablations/ablation_no_l3/ablation_no_l3_result.json",
        "manifest": "outputs/comparison_experiments/ablations/ablation_no_l3/00_manifest.json",
        "marker": "outputs/.batch_experiments/ablation_no_l3.done",
        "log": "outputs/.batch_experiments/logs/ablation_no_l3.log",
    },
    "original_ablation_large_chunk": {
        "result": "outputs/comparison_experiments/ablations/ablation_large_chunk/04_final_extraction.json",
        "manifest": "outputs/comparison_experiments/ablations/ablation_large_chunk/00_manifest.json",
        "marker": "outputs/.batch_experiments/ablation_large_chunk.done",
        "log": "outputs/.batch_experiments/logs/ablation_large_chunk.log",
    },
}


def recorded_model(path: Path) -> str:
    if not path.exists():
        return "UNKNOWN"
    data = read_json(path)
    if not isinstance(data, dict):
        return "UNKNOWN"
    for key in ("model", "model_name", "requested_model"):
        if data.get(key):
            return str(data[key])
    config = data.get("config") or data.get("resolved_args") or {}
    return str(config.get("model") or config.get("model_name") or "UNKNOWN")


def first_command(log_path: Path) -> str:
    if not log_path.exists():
        return ""
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("COMMAND:"):
            return line.partition(":")[2].strip()
    return ""


def valid_json(path: Path) -> tuple[bool, str]:
    try:
        read_json(path)
        return True, ""
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Read-only preflight for MinerU inputs, Qwen3.8-Max human_gold files, and frozen original experiment outputs."
    )
    parser.add_argument("--root-dir", required=True)
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--end", type=int, default=30)
    parser.add_argument("--require-gold", action="store_true")
    parser.add_argument("--output-dir")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    root = Path(args.root_dir).resolve()
    output = Path(args.output_dir).resolve() if args.output_dir else root / "revision_metrics" / "input_validation"
    papers = discover_papers(root, args.start, args.end)
    paper_rows: list[dict[str, Any]] = []
    condition_rows: list[dict[str, Any]] = []
    problems: list[dict[str, Any]] = []
    expected_count = args.end - args.start + 1
    if len(papers) != expected_count:
        problems.append({
            "paper_id": "*",
            "scope": "dataset",
            "status": "paper_count_mismatch",
            "detail": f"selected {len(papers)} folders; expected {expected_count} for {args.start}..{args.end}",
        })

    for paper in papers:
        markdown = paper / "full.md"
        markdown_text = markdown.read_text(encoding="utf-8", errors="replace")
        images = sorted(
            path for path in (paper / "images").rglob("*")
            if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
        )
        gold = paper / "outputs" / "revision_annotations" / "human_gold.json"
        metadata_path = gold.parent / "annotation_metadata.json"
        numbered_path = gold.parent / "numbered_source.md"
        gold_status = "missing"
        gold_error = ""
        full_input_audit = False
        if gold.exists():
            try:
                annotation = PaperGoldAnnotation.model_validate(read_json(gold))
                if annotation.paper_id != paper.name:
                    raise ValueError(f"paper_id={annotation.paper_id!r}")
                gold_status = "valid"
                if metadata_path.exists():
                    metadata = read_json(metadata_path)
                    full_input_audit = (
                        metadata.get("full_markdown_submitted") is True
                        and metadata.get("all_images_submitted") is True
                        and int(metadata.get("source_char_count", -1)) == len(markdown_text)
                        and numbered_path.exists()
                        and int(metadata.get("submitted_char_count", -1))
                        == len(numbered_path.read_text(encoding="utf-8", errors="replace"))
                        and int(metadata.get("source_image_count", -1)) == len(images)
                        and int(metadata.get("submitted_image_count", -2)) == len(images)
                    )
                    if not full_input_audit:
                        gold_status = "metadata_input_mismatch"
                else:
                    gold_status = "missing_metadata"
            except Exception as exc:
                gold_status = "invalid"
                gold_error = f"{type(exc).__name__}: {exc}"
        elif args.require_gold:
            gold_error = "human_gold.json is required"

        paper_rows.append({
            "paper_id": paper.name,
            "markdown_chars": len(markdown_text),
            "image_count": len(images),
            "multimodal": bool(images),
            "gold_status": gold_status,
            "full_input_audit": full_input_audit,
            "gold_path": str(gold),
            "gold_error": gold_error,
        })
        if args.require_gold and gold_status != "valid":
            problems.append({"paper_id": paper.name, "scope": "gold", "status": gold_status, "detail": gold_error})
        if not images:
            problems.append({
                "paper_id": paper.name,
                "scope": "mineru_input",
                "status": "no_supported_images",
                "detail": "images/ contains no PNG/JPEG/WebP file, so the requested gold call would not be multimodal",
            })
        for image in images:
            if image.stat().st_size > 20 * 1024 * 1024:
                problems.append({
                    "paper_id": paper.name,
                    "scope": "mineru_input",
                    "status": "image_exceeds_20mb",
                    "detail": str(image),
                })

        for condition, spec in CONDITIONS.items():
            result_path = paper / str(spec["result"])
            manifest_path = paper / str(spec["manifest"])
            marker_path = paper / str(spec["marker"]) if spec["marker"] else None
            log_path = paper / str(spec["log"]) if spec["log"] else None
            marker_exists = bool(marker_path and marker_path.exists())
            command = first_command(log_path) if log_path else ""
            if result_path.exists():
                okay, error = valid_json(result_path)
                status = "ok" if okay else "invalid_json"
            elif marker_exists:
                status = "stale_done_marker"
                error = "done marker exists but the expected result file is absent"
            else:
                status = "missing_result"
                error = "expected result file is absent"
            if status != "ok":
                problems.append({"paper_id": paper.name, "scope": condition, "status": status, "detail": error})
            condition_rows.append({
                "paper_id": paper.name,
                "condition": condition,
                "status": status,
                "recorded_model": recorded_model(manifest_path),
                "result_path": str(result_path),
                "manifest_path": str(manifest_path),
                "done_marker": str(marker_path) if marker_path else "",
                "first_logged_command": command,
                "logged_explicit_output_path": ("--output-dir" in command or "--output-root" in command) if command else "UNKNOWN",
                "detail": error,
            })

    report = {
        "root_dir": str(root),
        "selected_paper_count": len(papers),
        "expected_paper_count": expected_count,
        "condition_count": len(CONDITIONS),
        "problem_count": len(problems),
        "problems": problems,
        "paper_rows": paper_rows,
        "condition_rows": condition_rows,
        "read_only": True,
    }
    write_json(output / "input_validation.json", report)
    write_csv(output / "paper_inputs.csv", paper_rows)
    write_csv(output / "original_condition_files.csv", condition_rows)
    print(json.dumps({
        "papers": len(papers),
        "problem_count": len(problems),
        "report": str(output / "input_validation.json"),
    }, ensure_ascii=False, indent=2))
    return 2 if args.strict and problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
