from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from revision.annotation_schema import PaperGoldAnnotation
from revision.evaluation import greedy_match
from revision.io_utils import condition_result_path, discover_papers, node_id, node_text, normalize_final, read_json, write_json


def main() -> int:
    parser = argparse.ArgumentParser(description="Export node lists and lexical suggestions for blinded human match confirmation.")
    parser.add_argument("--root-dir", required=True)
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--end", type=int, default=30)
    parser.add_argument("--conditions", nargs="+", default=[
        "original_main", "original_baseline_oneshot_mineru", "original_ablation_text_only",
        "original_ablation_no_l3", "original_ablation_large_chunk",
        "added_global_eg_merge", "added_bm25_rag_text", "added_schema_text_only", "added_visual_masked",
        "added_chunk_6000", "added_chunk_9000", "added_chunk_12000",
        "added_l3_off_frozen_l2", "added_l3_on_frozen_l2",
        "added_relation_completion_off", "added_relation_completion_on",
    ])
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--threshold", type=float, default=0.45)
    args = parser.parse_args()
    output_dir = Path(args.output_dir).resolve()
    for paper in discover_papers(Path(args.root_dir), args.start, args.end):
        gold_path = paper / "outputs/revision_annotations/human_gold.json"
        if not gold_path.exists():
            continue
        gold = PaperGoldAnnotation.model_validate(read_json(gold_path))
        for condition in args.conditions:
            result_path = condition_result_path(paper, condition)
            if not result_path.exists():
                continue
            prediction = normalize_final(read_json(result_path))
            payload = {
                "paper_id": paper.name,
                "condition": condition,
                "status": "PENDING_HUMAN_REVIEW",
                "reviewer_name": "",
                "instructions": "Review semantic equivalence blinded to scores; edit *_pairs to the accepted one-to-one matches, then set status=verified_human and enter reviewer_name.",
                "predicted_problems": [{"id": node_id(item, "problem", i + 1), "text": node_text(item)} for i, item in enumerate(prediction["problems"])],
                "gold_problems": [{"id": item.id, "text": item.claim} for item in gold.research_problems],
                "predicted_methods": [{"id": node_id(item, "method", i + 1), "text": node_text(item)} for i, item in enumerate(prediction["methods"])],
                "gold_methods": [{"id": item.id, "text": item.claim} for item in gold.methods],
            }
            for kind, predicted, target in (("problem", prediction["problems"], gold.research_problems), ("method", prediction["methods"], gold.methods)):
                match = greedy_match([node_text(item) for item in predicted], [item.claim for item in target], args.threshold)
                payload[f"suggested_{kind}_pairs"] = [[node_id(predicted[p], kind, p + 1), target[g].id, score] for p, g, score in match.pairs]
                payload[f"{kind}_pairs"] = [[node_id(predicted[p], kind, p + 1), target[g].id] for p, g, _ in match.pairs]
            write_json(output_dir / f"{paper.name}__{condition}.json", payload)
    print(f"Matching packets written to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
