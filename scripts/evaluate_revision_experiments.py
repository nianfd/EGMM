from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from revision.evaluation import bootstrap_ci, evaluate_paper, mean, micro_aggregate, paired_permutation
from revision.io_utils import discover_papers, read_json, write_csv, write_json
from revision.semantic_matching import SentenceEmbeddingMatcher


DEFAULT_CONDITIONS = [
    "original_main", "original_baseline_oneshot_mineru", "original_ablation_text_only",
    "original_ablation_no_l3", "original_ablation_large_chunk",
    "added_global_eg_merge", "added_bm25_rag_text", "added_schema_text_only", "added_visual_masked",
    "added_chunk_6000", "added_chunk_9000", "added_chunk_12000",
    "added_l3_off_frozen_l2", "added_l3_on_frozen_l2",
    "added_relation_completion_off", "added_relation_completion_on",
    "added_semantic_core_6000",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root-dir", required=True)
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--end", type=int, default=30)
    parser.add_argument("--conditions", nargs="*", default=DEFAULT_CONDITIONS)
    parser.add_argument("--gold-name", default="human_gold.json")
    parser.add_argument("--annotation-dir-name", default="revision_annotations_main_aligned")
    parser.add_argument(
        "--matcher", choices=["semantic", "lexical"], default="semantic",
        help="Primary revised evaluator is semantic; lexical is retained only for legacy sensitivity analysis.",
    )
    parser.add_argument("--embedding-model", default="BAAI/bge-m3")
    parser.add_argument("--embedding-device", default="auto", help="auto, cpu, cuda, or cuda:0")
    parser.add_argument("--embedding-batch-size", type=int, default=32)
    parser.add_argument("--embedding-cache-file")
    parser.add_argument("--model-cache-dir")
    parser.add_argument("--problem-similarity-threshold", type=float, default=0.70)
    parser.add_argument("--method-similarity-threshold", type=float, default=0.70)
    parser.add_argument(
        "--thresholds-file",
        help="Optional JSON produced by calibrate_semantic_matcher.py; overrides model and thresholds.",
    )
    parser.add_argument("--similarity-threshold", type=float, default=0.45, help="Legacy lexical matcher only")
    parser.add_argument("--output-dir")
    parser.add_argument("--verified-match-dir", help="Optional directory containing paper__condition.json reviewed match files")
    parser.add_argument("--require-verified-matches", action="store_true")
    parser.add_argument("--require-complete", action="store_true", help="Fail unless every selected paper/condition is evaluated")
    parser.add_argument(
        "--primary-condition",
        default="added_chunk_6000",
        help="Condition used as the left-hand side of paired tests; revised main method is added_chunk_6000.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.root_dir).resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else root / "revision_metrics"
    if args.thresholds_file:
        thresholds = read_json(Path(args.thresholds_file).resolve())
        args.embedding_model = str(thresholds.get("embedding_model") or args.embedding_model)
        args.problem_similarity_threshold = float(
            thresholds.get("problem_similarity_threshold", args.problem_similarity_threshold)
        )
        args.method_similarity_threshold = float(
            thresholds.get("method_similarity_threshold", args.method_similarity_threshold)
        )
    for name, value in (
        ("problem", args.problem_similarity_threshold),
        ("method", args.method_similarity_threshold),
    ):
        if not 0.0 <= value <= 1.0:
            raise SystemExit(f"{name} semantic threshold must be within [0,1]")
    cache_path = (
        Path(args.embedding_cache_file).resolve()
        if args.embedding_cache_file
        else output_dir / "semantic_embeddings.sqlite3"
    )
    semantic_matcher = None
    if args.matcher == "semantic":
        semantic_matcher = SentenceEmbeddingMatcher(
            args.embedding_model,
            device=args.embedding_device,
            batch_size=args.embedding_batch_size,
            cache_path=cache_path,
            model_cache_dir=Path(args.model_cache_dir).resolve() if args.model_cache_dir else None,
        )
    rows = []
    missing = []
    selected_papers = discover_papers(root, args.start, args.end)
    for paper_dir in selected_papers:
        gold_path = paper_dir / "outputs" / args.annotation_dir_name / args.gold_name
        if not gold_path.exists():
            print(f"skip {paper_dir.name}: missing {gold_path}")
            missing.extend({"paper_id": paper_dir.name, "condition": condition, "reason": "missing_gold", "path": str(gold_path)} for condition in args.conditions)
            continue
        for condition in args.conditions:
            try:
                match_path = Path(args.verified_match_dir) / f"{paper_dir.name}__{condition}.json" if args.verified_match_dir else None
                if args.require_verified_matches and (match_path is None or not match_path.exists()):
                    print(f"skip {paper_dir.name}:{condition}: verified match file required")
                    missing.append({"paper_id": paper_dir.name, "condition": condition, "reason": "missing_verified_match", "path": str(match_path) if match_path else ""})
                    continue
                rows.append(evaluate_paper(
                    paper_dir, condition, gold_path,
                    similarity_threshold=args.similarity_threshold,
                    verified_match_path=match_path,
                    matching_method=args.matcher,
                    semantic_matcher=semantic_matcher,
                    problem_similarity_threshold=args.problem_similarity_threshold,
                    method_similarity_threshold=args.method_similarity_threshold,
                ))
            except FileNotFoundError:
                print(f"skip {paper_dir.name}:{condition}: missing result")
                missing.append({"paper_id": paper_dir.name, "condition": condition, "reason": "missing_prediction"})
    write_json(output_dir / "missing_inputs.json", missing)
    if args.require_complete and (missing or len(rows) != len(selected_papers) * len(args.conditions)):
        raise SystemExit(
            f"Incomplete evaluation: evaluated {len(rows)}/{len(selected_papers) * len(args.conditions)} paper-condition pairs; "
            f"see {output_dir / 'missing_inputs.json'}"
        )
    if semantic_matcher is not None:
        matcher_provenance = semantic_matcher.provenance()
        semantic_matcher.close()
    else:
        matcher_provenance = {
            "matcher": "legacy_token_set_jaccard_greedy",
            "similarity_threshold": args.similarity_threshold,
        }
    matcher_provenance.update({
        "problem_similarity_threshold": (
            args.problem_similarity_threshold if args.matcher == "semantic" else None
        ),
        "method_similarity_threshold": (
            args.method_similarity_threshold if args.matcher == "semantic" else None
        ),
        "thresholds_file": str(Path(args.thresholds_file).resolve()) if args.thresholds_file else None,
    })
    write_json(output_dir / "matching_protocol.json", matcher_provenance)

    by_condition = defaultdict(list)
    for row in rows:
        by_condition[row["condition"]].append(row)
    summary = []
    for condition, items in by_condition.items():
        output = {"condition": condition, "paper_count": len(items), "gold_statuses": sorted({item["gold_status"] for item in items})}
        for metric in [
            "problem", "method", "link", "link_including_inferred",
            "link_conditional_on_matched_nodes",
            "link_including_inferred_conditional_on_matched_nodes",
            "visual_dependent",
        ]:
            output[f"{metric}_micro"] = micro_aggregate(items, metric)
            f1s = [float(item[metric]["f1"]) for item in items]
            output[f"{metric}_macro_f1"] = mean(f1s)
            output[f"{metric}_macro_f1_ci95"] = bootstrap_ci(f1s)
        for metric in ["exact_node_reference_rate", "strict_visual_evidence_rate", "evidence_supported_link_rate", "strict_relation_evidence_rate"]:
            values = [float(item["structural_diagnostics"][metric]) for item in items]
            output[metric] = mean(values)
            output[f"{metric}_ci95"] = bootstrap_ci(values)
        for count_name in (
            "predicted_problems", "gold_problems", "matched_problems",
            "predicted_to_gold_problem_ratio", "predicted_methods", "gold_methods",
            "matched_methods", "predicted_to_gold_method_ratio",
        ):
            output[f"mean_{count_name}"] = mean([
                float(item["node_counts"][count_name]) for item in items
            ])
        summary.append(output)
    paired = []
    proposed_by_paper = {row["paper_id"]: row for row in by_condition.get(args.primary_condition, [])}
    if not proposed_by_paper:
        raise SystemExit(f"Primary condition has no evaluated papers: {args.primary_condition}")
    for condition, items in by_condition.items():
        if condition == args.primary_condition:
            continue
        other = {row["paper_id"]: row for row in items}
        common = sorted(set(proposed_by_paper) & set(other))
        for metric in [
            "problem", "method", "link", "link_including_inferred",
            "link_conditional_on_matched_nodes", "visual_dependent",
        ]:
            a = [float(proposed_by_paper[paper][metric]["f1"]) for paper in common]
            b = [float(other[paper][metric]["f1"]) for paper in common]
            paired.append({"comparison": f"{args.primary_condition}-vs-{condition}", "metric": f"{metric}_f1", "paper_count": len(common), "mean_difference": mean([x-y for x,y in zip(a,b)]), "paired_permutation_p": paired_permutation(a,b)})
    write_json(output_dir / "per_paper_metrics.json", rows)
    write_json(output_dir / "summary.json", summary)
    write_json(output_dir / "paired_tests.json", paired)
    flat = []
    for item in summary:
        row = {"condition": item["condition"], "paper_count": item["paper_count"], "gold_statuses": ";".join(item["gold_statuses"])}
        for metric in [
            "problem", "method", "link", "link_including_inferred",
            "link_conditional_on_matched_nodes",
            "link_including_inferred_conditional_on_matched_nodes",
            "visual_dependent",
        ]:
            row[f"{metric}_micro_precision"] = item[f"{metric}_micro"]["precision"]
            row[f"{metric}_micro_recall"] = item[f"{metric}_micro"]["recall"]
            row[f"{metric}_micro_f1"] = item[f"{metric}_micro"]["f1"]
            row[f"{metric}_macro_f1"] = item[f"{metric}_macro_f1"]
            row[f"{metric}_macro_ci_low"], row[f"{metric}_macro_ci_high"] = item[f"{metric}_macro_f1_ci95"]
        row["exact_node_reference_rate"] = item["exact_node_reference_rate"]
        row["strict_visual_evidence_rate"] = item["strict_visual_evidence_rate"]
        row["evidence_supported_link_rate"] = item["evidence_supported_link_rate"]
        row["strict_relation_evidence_rate"] = item["strict_relation_evidence_rate"]
        for count_name in (
            "predicted_problems", "gold_problems", "matched_problems",
            "predicted_to_gold_problem_ratio", "predicted_methods", "gold_methods",
            "matched_methods", "predicted_to_gold_method_ratio",
        ):
            row[f"mean_{count_name}"] = item[f"mean_{count_name}"]
        flat.append(row)
    write_csv(output_dir / "summary.csv", flat)
    count_rows = []
    for item in rows:
        count_rows.append({
            "paper_id": item["paper_id"],
            "condition": item["condition"],
            **item["node_counts"],
        })
    write_csv(output_dir / "node_count_diagnostics.csv", count_rows)
    print(json.dumps({
        "papers_conditions": len(rows),
        "matcher": args.matcher,
        "embedding_model": args.embedding_model if args.matcher == "semantic" else None,
        "problem_threshold": args.problem_similarity_threshold if args.matcher == "semantic" else None,
        "method_threshold": args.method_similarity_threshold if args.matcher == "semantic" else None,
        "output_dir": str(output_dir),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
