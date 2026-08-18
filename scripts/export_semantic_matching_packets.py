from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from revision.annotation_schema import PaperGoldAnnotation
from revision.io_utils import condition_result_path, discover_papers, node_id, node_text, normalize_final, read_json, write_json
from revision.semantic_matching import SentenceEmbeddingMatcher


DEFAULT_CONDITIONS = [
    "added_chunk_6000", "original_main", "original_baseline_oneshot_mineru",
    "added_global_eg_merge", "added_bm25_rag_text",
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Export semantic Hungarian node-match packets for optional verification")
    parser.add_argument("--root-dir", required=True)
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--end", type=int, default=5)
    parser.add_argument("--conditions", nargs="*", default=DEFAULT_CONDITIONS)
    parser.add_argument("--annotation-dir-name", default="revision_annotations_main_aligned")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--embedding-model", default="BAAI/bge-m3")
    parser.add_argument("--embedding-device", default="auto")
    parser.add_argument("--embedding-batch-size", type=int, default=32)
    parser.add_argument("--embedding-cache-file")
    parser.add_argument("--problem-threshold", type=float, default=0.70)
    parser.add_argument("--method-threshold", type=float, default=0.70)
    args = parser.parse_args()

    root = Path(args.root_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    cache_path = (
        Path(args.embedding_cache_file).resolve()
        if args.embedding_cache_file else output_dir / "semantic_embeddings.sqlite3"
    )
    matcher = SentenceEmbeddingMatcher(
        args.embedding_model,
        device=args.embedding_device,
        batch_size=args.embedding_batch_size,
        cache_path=cache_path,
    )
    written = 0
    try:
        for paper in discover_papers(root, args.start, args.end):
            gold_path = paper / "outputs" / args.annotation_dir_name / "human_gold.json"
            if not gold_path.exists():
                continue
            gold = PaperGoldAnnotation.model_validate(read_json(gold_path))
            for condition in args.conditions:
                result_path = condition_result_path(paper, condition)
                if not result_path.exists():
                    continue
                prediction = normalize_final(read_json(result_path))
                pred_problems = [
                    {"id": node_id(item, "problem", index + 1), "text": node_text(item)}
                    for index, item in enumerate(prediction["problems"])
                ]
                pred_methods = [
                    {"id": node_id(item, "method", index + 1), "text": node_text(item)}
                    for index, item in enumerate(prediction["methods"])
                ]
                gold_problems = [{"id": item.id, "text": item.claim} for item in gold.research_problems]
                gold_methods = [{"id": item.id, "text": item.claim} for item in gold.methods]
                problem_match = matcher.match(
                    [item["text"] for item in pred_problems],
                    [item["text"] for item in gold_problems],
                    args.problem_threshold,
                )
                method_match = matcher.match(
                    [item["text"] for item in pred_methods],
                    [item["text"] for item in gold_methods],
                    args.method_threshold,
                )
                problem_pairs = [
                    [pred_problems[p]["id"], gold_problems[g]["id"]]
                    for p, g, _ in problem_match.pairs
                ]
                method_pairs = [
                    [pred_methods[p]["id"], gold_methods[g]["id"]]
                    for p, g, _ in method_match.pairs
                ]
                payload = {
                    "paper_id": paper.name,
                    "condition": condition,
                    "status": "PENDING_HUMAN_REVIEW",
                    "reviewer_name": "",
                    "instructions": (
                        "Review semantic equivalence without looking at aggregate scores. Edit problem_pairs and "
                        "method_pairs to the accepted one-to-one mappings, enter reviewer_name, then set "
                        "status=verified_human. Keep IDs exactly as listed."
                    ),
                    "matching_protocol": matcher.provenance(),
                    "problem_threshold": args.problem_threshold,
                    "method_threshold": args.method_threshold,
                    "predicted_problems": pred_problems,
                    "gold_problems": gold_problems,
                    "predicted_methods": pred_methods,
                    "gold_methods": gold_methods,
                    "suggested_problem_pairs_with_scores": [
                        [pred_problems[p]["id"], gold_problems[g]["id"], score]
                        for p, g, score in problem_match.pairs
                    ],
                    "suggested_method_pairs_with_scores": [
                        [pred_methods[p]["id"], gold_methods[g]["id"], score]
                        for p, g, score in method_match.pairs
                    ],
                    "problem_pairs": problem_pairs,
                    "method_pairs": method_pairs,
                }
                write_json(output_dir / f"{paper.name}__{condition}.json", payload)
                written += 1
    finally:
        matcher.close()
    print(f"Wrote {written} semantic matching packets to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
