from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from revision.contracts import validate_condition_output
from revision.io_utils import discover_papers, write_csv, write_json


DEFAULT_CONDITIONS = [
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


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate saved added-experiment outputs against both the original and gold metric contracts."
    )
    parser.add_argument("--root-dir", required=True)
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--end", type=int, default=100)
    parser.add_argument("--conditions", nargs="*", default=DEFAULT_CONDITIONS)
    parser.add_argument("--output-dir")
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    root = Path(args.root_dir).resolve()
    output = Path(args.output_dir).resolve() if args.output_dir else root / "revision_metrics" / "metric_compatibility"
    rows = []
    reports = []
    for paper in discover_papers(root, args.start, args.end):
        for condition in args.conditions:
            directory = paper / "outputs" / "major_revision_additions" / condition
            if not directory.exists():
                report = {
                    "condition": condition,
                    "condition_dir": str(directory),
                    "result_path": "",
                    "valid": False,
                    "errors": ["condition directory is missing"],
                    "warnings": [],
                    "counts_and_engines": {},
                }
                if not args.require_complete:
                    continue
            else:
                report = validate_condition_output(directory, condition, allow_empty=False)
            report["paper_id"] = paper.name
            reports.append(report)
            counts = report.get("counts_and_engines", {})
            rows.append({
                "paper_id": paper.name,
                "condition": condition,
                "valid": report["valid"],
                "problem_count": counts.get("problem_count", ""),
                "method_count": counts.get("method_count", ""),
                "link_count": counts.get("link_count", ""),
                "original_metric_engine": counts.get("original_metric_engine", ""),
                "gold_metric_schema": counts.get("gold_metric_schema", ""),
                "error_count": len(report.get("errors", [])),
                "warning_count": len(report.get("warnings", [])),
                "errors": " | ".join(report.get("errors", [])),
                "warnings": " | ".join(report.get("warnings", [])),
                "result_path": report.get("result_path", ""),
            })

    failed = [report for report in reports if not report["valid"]]
    with_warnings = [report for report in reports if report.get("warnings")]
    total_warnings = sum(len(report.get("warnings", [])) for report in reports)
    write_json(output / "metric_compatibility.json", {
        "valid": not failed,
        "checked_count": len(reports),
        "failed_count": len(failed),
        "reports_with_quality_warnings": len(with_warnings),
        "quality_warning_count": total_warnings,
        "reports": reports,
    })
    write_csv(output / "metric_compatibility.csv", rows)
    print(json.dumps({
        "checked": len(reports),
        "failed": len(failed),
        "reports_with_quality_warnings": len(with_warnings),
        "quality_warnings": total_warnings,
        "report": str(output / "metric_compatibility.json"),
    }, ensure_ascii=False, indent=2))
    return 2 if args.strict and failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
