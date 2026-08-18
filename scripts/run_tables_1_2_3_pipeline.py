from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
STRUCTURAL_CONDITIONS = [
    "proposed_full",
    "baseline_oneshot_mineru",
    "added_global_eg_merge",
    "added_bm25_rag_text",
    "added_chunk_6000",
    "added_chunk_9000",
    "added_chunk_12000",
]
ADDED_CONDITIONS = [
    "added_global_eg_merge",
    "added_bm25_rag_text",
    "added_chunk_6000",
    "added_chunk_9000",
    "added_chunk_12000",
]


def _run(command: list[str]) -> None:
    print("RUN:", " ".join(command), flush=True)
    subprocess.run(command, check=True, cwd=ROOT)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Recompute 100-paper structural summaries and combine them with the existing "
            "30-paper semantic evaluation to build manuscript Tables I-III. No model API is called."
        )
    )
    parser.add_argument("--root-dir", required=True)
    parser.add_argument("--semantic-metrics-dir", required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--end", type=int, default=100)
    parser.add_argument("--expected-semantic-papers", type=int, default=30)
    parser.add_argument("--skip-compatibility-validation", action="store_true")
    args = parser.parse_args()

    root = Path(args.root_dir).resolve()
    semantic_dir = Path(args.semantic_metrics_dir).resolve()
    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir else root / "revision_metrics" / "paper_tables_1_2_3_v10"
    )
    structural_dir = output_dir / "structural_100"
    tables_dir = output_dir / "tables"
    expected_structural = args.end - args.start + 1
    if expected_structural != 100:
        raise SystemExit(
            f"Tables I and III are defined on 100 papers, but the selected range contains {expected_structural}. "
            "Use --start 1 --end 100 for the manuscript run."
        )
    if not (semantic_dir / "summary.json").exists() or not (semantic_dir / "matching_protocol.json").exists():
        raise SystemExit(
            f"Missing semantic summary.json or matching_protocol.json in {semantic_dir}. "
            "Point --semantic-metrics-dir to the completed revised6000_semantic30 directory."
        )

    if not args.skip_compatibility_validation:
        _run([
            sys.executable, "scripts/validate_metric_compatibility.py",
            "--root-dir", str(root),
            "--start", str(args.start), "--end", str(args.end),
            "--conditions", *ADDED_CONDITIONS,
            "--require-complete", "--strict",
            "--output-dir", str(output_dir / "metric_compatibility_100"),
        ])

    _run([
        sys.executable, "scripts/aggregate_batch_metrics.py",
        "--root-dir", str(root),
        "--start", str(args.start), "--end", str(args.end),
        "--conditions", *STRUCTURAL_CONDITIONS,
        "--output-dir", str(structural_dir),
    ])
    structural_summary = structural_dir / "metrics_summary_core.csv"
    _run([
        sys.executable, "scripts/build_paper_tables_1_2_3.py",
        "--structural-summary-csv", str(structural_summary),
        "--semantic-metrics-dir", str(semantic_dir),
        "--output-dir", str(tables_dir),
        "--expected-structural-papers", "100",
        "--expected-semantic-papers", str(args.expected_semantic_papers),
        "--expected-problem-threshold", "0.70",
        "--expected-method-threshold", "0.70",
    ])
    result = {
        "status": "complete",
        "model_api_calls": 0,
        "table_1_and_3_papers": 100,
        "table_2_reference_papers": args.expected_semantic_papers,
        "structural_summary": str(structural_summary),
        "tables_dir": str(tables_dir),
        "combined_latex": str(tables_dir / "article_tables_1_2_3.tex"),
    }
    (output_dir / "table_pipeline_summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
