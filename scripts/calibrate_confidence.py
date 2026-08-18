from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from revision.calibration import calibration_metrics, grouped_cross_validated_platt
from revision.io_utils import read_json, write_csv, write_json


def main() -> int:
    parser = argparse.ArgumentParser(description="Paper-grouped Platt calibration, ECE, Brier score, and reliability bins.")
    parser.add_argument("--metrics-json", required=True, help="per_paper_metrics.json from evaluate_revision_experiments.py")
    parser.add_argument("--condition", default="original_main")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--bins", type=int, default=10)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    paper_rows = read_json(Path(args.metrics_json))
    records = [record for row in paper_rows if row.get("condition") == args.condition for record in row.get("calibration_records", [])]
    calibrated, model = grouped_cross_validated_platt(records, folds=args.folds)
    overall = {
        "uncalibrated": calibration_metrics(calibrated, "confidence", bins=args.bins),
        "paper_grouped_cross_validated": calibration_metrics(calibrated, "calibrated_confidence", bins=args.bins),
    }
    by_section: dict[str, list[dict]] = defaultdict(list)
    for row in calibrated:
        by_section[row.get("section") or "unknown"].append(row)
    section_report = {
        section: {
            "uncalibrated": calibration_metrics(items, "confidence", bins=args.bins),
            "calibrated": calibration_metrics(items, "calibrated_confidence", bins=args.bins),
        }
        for section, items in sorted(by_section.items())
    }
    output_dir = Path(args.output_dir).resolve()
    write_json(output_dir / "calibration_report.json", {"condition": args.condition, "model": model, "overall": overall, "by_section": section_report})
    write_csv(output_dir / "calibration_records.csv", calibrated)
    write_csv(output_dir / "reliability_bins.csv", overall["paper_grouped_cross_validated"]["reliability_bins"])
    print(json.dumps({"records": len(calibrated), "output_dir": str(output_dir)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
