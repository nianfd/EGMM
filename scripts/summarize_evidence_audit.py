from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from revision.evaluation import cohens_kappa


ALLOWED = {"fully_supported", "partially_supported", "not_supported", "unjudgeable"}


def load(path: Path) -> dict[str, dict]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = {row["blind_item_id"]: row for row in csv.DictReader(handle)}
    invalid = [key for key, row in rows.items() if row.get("support_label") not in ALLOWED]
    if invalid:
        raise ValueError(f"Missing/invalid support labels in {path}: {invalid[:10]}")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize two completed expert evidence audits and agreement.")
    parser.add_argument("--reviewer-1", required=True)
    parser.add_argument("--reviewer-2", required=True)
    parser.add_argument("--adjudicated", help="Optional completed packet with one final support_label per blind_item_id")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    left, right = load(Path(args.reviewer_1)), load(Path(args.reviewer_2))
    common = sorted(set(left) & set(right))
    if set(left) != set(right):
        raise SystemExit("Reviewer packet IDs differ")
    labels_left = [left[key]["support_label"] for key in common]
    labels_right = [right[key]["support_label"] for key in common]
    by_condition = defaultdict(list)
    for key in common:
        by_condition[left[key]["condition"]].append(key)
    report = {
        "item_count": len(common),
        "cohens_kappa": cohens_kappa(labels_left, labels_right),
        "raw_agreement": sum(a == b for a, b in zip(labels_left, labels_right)) / len(common) if common else 0.0,
        "by_condition": {},
        "note": "Resolve disagreements in an adjudicated copy before reporting final faithfulness accuracy.",
    }
    for condition, ids in sorted(by_condition.items()):
        counts_1 = Counter(left[key]["support_label"] for key in ids)
        counts_2 = Counter(right[key]["support_label"] for key in ids)
        report["by_condition"][condition] = {"reviewer_1": dict(counts_1), "reviewer_2": dict(counts_2)}
    if args.adjudicated:
        final = load(Path(args.adjudicated))
        if set(final) != set(left):
            raise SystemExit("Adjudicated packet IDs differ")
        final_by_condition: dict[str, list[str]] = defaultdict(list)
        for key in common:
            final_by_condition[final[key]["condition"]].append(final[key]["support_label"])
        report["adjudicated"] = {"overall": {}, "by_condition": {}}
        all_labels = [final[key]["support_label"] for key in common]
        def accuracy(labels: list[str]) -> dict[str, float | int | dict[str, int]]:
            counts = Counter(labels)
            total = len(labels)
            return {
                "count": total,
                "label_counts": dict(counts),
                "fully_supported_rate": counts["fully_supported"] / total if total else 0.0,
                "at_least_partially_supported_rate": (counts["fully_supported"] + counts["partially_supported"]) / total if total else 0.0,
                "not_supported_rate": counts["not_supported"] / total if total else 0.0,
                "unjudgeable_rate": counts["unjudgeable"] / total if total else 0.0,
            }
        report["adjudicated"]["overall"] = accuracy(all_labels)
        report["adjudicated"]["by_condition"] = {
            condition: accuracy(labels) for condition, labels in sorted(final_by_condition.items())
        }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
