from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run(command: list[str]) -> None:
    print("RUN:", " ".join(command), flush=True)
    subprocess.run(command, check=True, cwd=ROOT)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run semantic gold-based metrics, significance tests, confidence calibration, "
            "and manuscript-table export for the revised 6k main method."
        )
    )
    parser.add_argument("--root-dir", required=True)
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--end", type=int, default=30)
    parser.add_argument("--output-dir")
    parser.add_argument("--conditions", nargs="*")
    parser.add_argument("--annotation-dir-name", default="revision_annotations_main_aligned")
    parser.add_argument("--embedding-model", default="BAAI/bge-m3")
    parser.add_argument("--embedding-device", default="auto")
    parser.add_argument("--embedding-batch-size", type=int, default=32)
    parser.add_argument("--embedding-cache-file")
    parser.add_argument("--model-cache-dir")
    parser.add_argument("--problem-similarity-threshold", type=float, default=0.70)
    parser.add_argument("--method-similarity-threshold", type=float, default=0.70)
    parser.add_argument("--thresholds-file")
    parser.add_argument("--verified-match-dir")
    parser.add_argument(
        "--require-verified-matches", action="store_true",
        help="Require externally reviewed node-pair files instead of automatic semantic matches",
    )
    args = parser.parse_args()
    root = Path(args.root_dir).resolve()
    output = Path(args.output_dir).resolve() if args.output_dir else root / "revision_metrics"
    evaluate = [
        sys.executable, "scripts/evaluate_revision_experiments.py",
        "--root-dir", str(root), "--start", str(args.start), "--end", str(args.end),
        "--output-dir", str(output),
        "--annotation-dir-name", args.annotation_dir_name,
        "--matcher", "semantic",
        "--embedding-model", args.embedding_model,
        "--embedding-device", args.embedding_device,
        "--embedding-batch-size", str(args.embedding_batch_size),
        "--problem-similarity-threshold", str(args.problem_similarity_threshold),
        "--method-similarity-threshold", str(args.method_similarity_threshold),
        "--primary-condition", "added_chunk_6000",
        "--require-complete",
    ]
    if args.conditions:
        evaluate += ["--conditions", *args.conditions]
    if args.embedding_cache_file:
        evaluate += ["--embedding-cache-file", str(Path(args.embedding_cache_file).resolve())]
    if args.model_cache_dir:
        evaluate += ["--model-cache-dir", str(Path(args.model_cache_dir).resolve())]
    if args.thresholds_file:
        evaluate += ["--thresholds-file", str(Path(args.thresholds_file).resolve())]
    if args.verified_match_dir:
        evaluate += ["--verified-match-dir", str(Path(args.verified_match_dir).resolve())]
    if args.require_verified_matches:
        if not args.verified_match_dir:
            raise SystemExit("--require-verified-matches also requires --verified-match-dir")
        evaluate += ["--require-verified-matches"]
    run(evaluate)
    run([
        sys.executable, "scripts/calibrate_confidence.py",
        "--metrics-json", str(output / "per_paper_metrics.json"),
        "--condition", "added_chunk_6000", "--folds", "5", "--bins", "10",
        "--output-dir", str(output / "calibration"),
    ])
    run([
        sys.executable, "scripts/build_manuscript_tables.py",
        "--metrics-dir", str(output), "--output-dir", str(output / "manuscript_tables"),
    ])
    evaluated_conditions = set(args.conditions or [])
    if not args.conditions or {
        "added_chunk_6000", "original_main", "original_baseline_oneshot_mineru",
        "added_global_eg_merge", "added_bm25_rag_text",
    }.issubset(evaluated_conditions):
        report = [
            sys.executable, "scripts/report_revised_6000_results.py",
            "--metrics-dir", str(output),
            "--output-dir", str(output / "revised_6000_report"),
            "--expected-papers", str(args.end - args.start + 1),
        ]
        if not args.conditions or "added_semantic_core_6000" in evaluated_conditions:
            report.append("--include-compression-ablation")
        run(report)
    print(f"Complete. Quantitative tables: {output / 'manuscript_tables'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
