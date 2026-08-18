from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from revision.io_utils import discover_papers, read_json, write_csv, write_json


ORIGINAL_MANIFESTS = {
    "original_main": "outputs/00_manifest.json",
    "original_ablation_text_only": "outputs/comparison_experiments/ablations/ablation_text_only/00_manifest.json",
    "original_ablation_no_l3": "outputs/comparison_experiments/ablations/ablation_no_l3/00_manifest.json",
    "original_ablation_large_chunk": "outputs/comparison_experiments/ablations/ablation_large_chunk/00_manifest.json",
    "original_baseline_oneshot_mineru": "outputs/comparison_experiments/baseline_oneshot_mineru/baseline_oneshot_mineru_manifest.json",
}

ORIGINAL_RESULT_FILES = {
    "original_main": "outputs/04_final_extraction.json",
    "original_ablation_text_only": "outputs/comparison_experiments/ablations/ablation_text_only/04_final_extraction.json",
    "original_ablation_no_l3": "outputs/comparison_experiments/ablations/ablation_no_l3/ablation_no_l3_result.json",
    "original_ablation_large_chunk": "outputs/comparison_experiments/ablations/ablation_large_chunk/04_final_extraction.json",
    "original_baseline_oneshot_mineru": "outputs/comparison_experiments/baseline_oneshot_mineru/baseline_oneshot_mineru.json",
}

ADDED_RESULT_NAMES = ("04_final_extraction.json", "result.json", "final_result.json")


def nested_model(data: Any) -> str:
    if not isinstance(data, dict):
        return ""
    for key in ("model", "model_name", "requested_model"):
        if data.get(key):
            return str(data[key])
    for key in ("resolved_args", "config", "request", "metadata"):
        value = nested_model(data.get(key))
        if value:
            return value
    return ""


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit the model actually recorded by every saved original/additional result.")
    parser.add_argument("--root-dir", required=True)
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--end", type=int, default=100)
    parser.add_argument("--expected-original-model", default="qwen-vl-max", choices=["qwen-vl-max"])
    parser.add_argument("--expected-added-model", default="qwen-vl-max", choices=["qwen-vl-max"])
    parser.add_argument("--output-dir")
    parser.add_argument("--strict", action="store_true", help="Exit nonzero on a recorded mismatch or missing model provenance")
    args = parser.parse_args()
    root = Path(args.root_dir).resolve()
    output = Path(args.output_dir).resolve() if args.output_dir else root / "revision_metrics" / "backend_audit"
    rows: list[dict[str, Any]] = []
    for paper in discover_papers(root, args.start, args.end):
        for condition, relative in ORIGINAL_MANIFESTS.items():
            path = paper / relative
            model = nested_model(read_json(path)) if path.exists() else ""
            result_path = paper / ORIGINAL_RESULT_FILES[condition]
            if not result_path.exists():
                status = "missing_result"
            elif model.lower() == args.expected_original_model.lower():
                status = "ok"
            elif path.exists() and not model:
                status = "model_not_recorded"
            elif not path.exists():
                status = "missing_manifest"
            else:
                status = "mismatch"
            rows.append({
                "paper_id": paper.name,
                "family": "protected_original",
                "condition": condition,
                "expected_model": args.expected_original_model,
                "recorded_model": model or "UNKNOWN",
                "status": status,
                "provenance_file": str(path),
                "result_file": str(result_path),
            })
        additions = paper / "outputs" / "major_revision_additions"
        if additions.is_dir():
            for condition_dir in sorted(path for path in additions.iterdir() if path.is_dir()):
                path = condition_dir / "condition_info.json"
                model = nested_model(read_json(path)) if path.exists() else ""
                result_path = next((condition_dir / name for name in ADDED_RESULT_NAMES if (condition_dir / name).exists()), None)
                # Frozen conditions may legitimately have no new model call; record that separately.
                if result_path is None:
                    status = "missing_result"
                elif model:
                    status = "ok" if model.lower() == args.expected_added_model.lower() else "mismatch"
                else:
                    status = "frozen_no_new_call" if path.exists() else "missing_manifest"
                rows.append({
                    "paper_id": paper.name,
                    "family": "major_revision_addition",
                    "condition": condition_dir.name,
                    "expected_model": args.expected_added_model,
                    "recorded_model": model or "NO_NEW_MODEL_RECORDED",
                    "status": status,
                    "provenance_file": str(path),
                    "result_file": str(result_path) if result_path else "MISSING",
                })
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    report = {
        "expected_original_model": args.expected_original_model,
        "expected_added_model": args.expected_added_model,
        "counts": counts,
        "rows": rows,
        "interpretation": {
            "mismatch": "The manuscript model claim does not match the saved manifest; do not relabel the result.",
            "model_not_recorded": "Model provenance is absent; recover it from immutable logs/request caches before submission.",
            "frozen_no_new_call": "Controlled condition reused a frozen upstream output and made no new model call at that stage.",
        },
    }
    write_json(output / "backend_audit.json", report)
    write_csv(output / "backend_audit.csv", rows)
    print(f"Wrote {output / 'backend_audit.json'}; counts={counts}")
    bad = sum(counts.get(key, 0) for key in ("mismatch", "model_not_recorded", "missing_manifest", "missing_result"))
    return 2 if args.strict and bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
