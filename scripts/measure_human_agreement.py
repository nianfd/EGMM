from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from revision.agreement import annotation_agreement
from revision.annotation_schema import PaperGoldAnnotation
from revision.io_utils import discover_papers, mean, read_json, write_json


def main() -> int:
    parser = argparse.ArgumentParser(description="Measure agreement between two real, independently produced human annotation files.")
    parser.add_argument("--root-dir", required=True)
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--end", type=int, default=30)
    parser.add_argument("--annotation-dir-name", default="revision_annotations")
    parser.add_argument("--threshold", type=float, default=0.45)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    rows = []
    for paper in discover_papers(Path(args.root_dir), args.start, args.end):
        directory = paper / "outputs" / args.annotation_dir_name
        left, right = directory / "human_annotator_1.json", directory / "human_annotator_2.json"
        if not left.exists() or not right.exists():
            print(f"skip {paper.name}: independent human files missing")
            continue
        rows.append(annotation_agreement(PaperGoldAnnotation.model_validate(read_json(left)), PaperGoldAnnotation.model_validate(read_json(right)), args.threshold))
    aggregate = {
        "paper_count": len(rows),
        "macro_problem_boundary_f1": mean([row["problem_nodes"]["node_boundary_f1"] for row in rows]),
        "macro_method_boundary_f1": mean([row["method_nodes"]["node_boundary_f1"] for row in rows]),
        "macro_relation_f1": mean([row["relation_f1_on_matched_endpoints"] for row in rows]),
        "macro_visual_dependency_kappa": mean([row["visual_dependency_kappa_on_matched_nodes"] for row in rows]),
    }
    write_json(Path(args.output), {"per_paper": rows, "aggregate": aggregate})
    print(json.dumps(aggregate, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
