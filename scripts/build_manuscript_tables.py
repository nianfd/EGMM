from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from revision.io_utils import read_json, write_csv, write_json


DISPLAY = {
    "original_main": "Original main method (frozen)",
    "original_baseline_oneshot_mineru": "Original one-shot MinerU baseline (frozen)",
    "original_ablation_text_only": "Original text-only ablation (frozen)",
    "original_ablation_no_l3": "Original no-L3 ablation (frozen)",
    "original_ablation_large_chunk": "Original large-chunk ablation (frozen)",
    "added_global_eg_merge": "Global evidence-graph merge baseline",
    "added_bm25_rag_text": "BM25-RAG text baseline",
    "added_schema_text_only": "Schema-matched text-only",
    "added_visual_masked": "Visual-masked control",
    "added_chunk_6000": "Revised proposed method (6k multimodal)",
    "added_chunk_9000": "Chunk size 9,000 (control)",
    "added_chunk_12000": "Chunk size 12,000",
    "added_l3_off_frozen_l2": "L3 off (frozen L2)",
    "added_l3_on_frozen_l2": "L3 on (frozen L2)",
    "added_relation_completion_off": "Relation completion off",
    "added_relation_completion_on": "Relation completion on",
    "added_semantic_core_6000": "Semantic-core compression ablation",
}

GROUPS = {
    "table_original_results_re_evaluated_on_gold.csv": [
        "original_main", "original_baseline_oneshot_mineru", "original_ablation_text_only",
        "original_ablation_no_l3", "original_ablation_large_chunk",
    ],
    "table_revised_main_comparison.csv": [
        "added_chunk_6000", "original_main", "original_baseline_oneshot_mineru",
        "added_global_eg_merge", "added_bm25_rag_text",
    ],
    "table_controlled_ablations.csv": [
        "added_chunk_9000", "added_schema_text_only", "added_visual_masked",
        "added_chunk_6000", "added_chunk_12000",
        "added_l3_off_frozen_l2", "added_l3_on_frozen_l2",
        "added_relation_completion_off", "added_relation_completion_on",
    ],
    "table_compression_ablation.csv": [
        "added_chunk_6000", "added_semantic_core_6000",
    ],
}


def flatten(item: dict[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {
        "method": DISPLAY.get(item["condition"], item["condition"]),
        "condition": item["condition"],
        "n_papers": item["paper_count"],
        "gold_status": ";".join(item.get("gold_statuses", [])),
    }
    for metric in (
        "problem", "method", "link", "link_including_inferred",
        "link_conditional_on_matched_nodes",
        "link_including_inferred_conditional_on_matched_nodes",
        "visual_dependent",
    ):
        row[f"{metric}_precision"] = item[f"{metric}_micro"]["precision"]
        row[f"{metric}_recall"] = item[f"{metric}_micro"]["recall"]
        row[f"{metric}_f1"] = item[f"{metric}_micro"]["f1"]
        row[f"{metric}_macro_f1"] = item[f"{metric}_macro_f1"]
        row[f"{metric}_macro_ci95_low"] = item[f"{metric}_macro_f1_ci95"][0]
        row[f"{metric}_macro_ci95_high"] = item[f"{metric}_macro_f1_ci95"][1]
    for metric in ("exact_node_reference_rate", "strict_visual_evidence_rate", "evidence_supported_link_rate", "strict_relation_evidence_rate"):
        row[metric] = item[metric]
    row["exact_node_reference_applicable"] = item["condition"] != "original_baseline_oneshot_mineru"
    if not row["exact_node_reference_applicable"]:
        row["exact_node_reference_rate"] = "N/A"
    for count_name in (
        "predicted_problems", "gold_problems", "matched_problems",
        "predicted_to_gold_problem_ratio", "predicted_methods", "gold_methods",
        "matched_methods", "predicted_to_gold_method_ratio",
    ):
        row[f"mean_{count_name}"] = item.get(f"mean_{count_name}")
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert evaluation JSON into manuscript-ready CSV tables.")
    parser.add_argument("--metrics-dir", required=True)
    parser.add_argument("--output-dir")
    args = parser.parse_args()
    metrics_dir = Path(args.metrics_dir).resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else metrics_dir / "manuscript_tables"
    summary = read_json(metrics_dir / "summary.json")
    by_condition = {item["condition"]: item for item in summary}
    written: list[str] = []
    for filename, conditions in GROUPS.items():
        rows = [flatten(by_condition[name]) for name in conditions if name in by_condition]
        write_csv(output_dir / filename, rows)
        written.append(filename)
    paired = read_json(metrics_dir / "paired_tests.json")
    write_csv(output_dir / "table_paired_significance.csv", paired)
    mapping = {
        "model_gold_semantic_correctness": {
            "reviewer_concern": "Lack of a direct semantic-reference evaluation",
            "result_files": ["table_revised_main_comparison.csv", "table_original_results_re_evaluated_on_gold.csv", "table_paired_significance.csv"],
            "manuscript_location": "Experiments > Frozen Qwen3.8-Max 9k-reference protocol and revised 6k Results",
        },
        "stronger_baselines": {
            "reviewer_concern": "Need stronger and schema-matched baselines",
            "result_files": ["table_revised_main_comparison.csv"],
            "manuscript_location": "Experiments > Baselines; Results > Main comparison",
        },
        "semantic_matching": {
            "reviewer_concern": "Lexically different but semantically equivalent nodes must not be counted as errors",
            "result_files": ["table_revised_main_comparison.csv", "table_paired_significance.csv"],
            "manuscript_location": "Experiments > Semantic matching protocol",
        },
        "visual_grounding": {
            "reviewer_concern": "Need stricter evidence that images materially contribute",
            "result_files": ["table_controlled_ablations.csv"],
            "compare": ["added_chunk_9000", "added_schema_text_only", "added_visual_masked"],
            "manuscript_location": "Experiments > Controlled ablations; Results > Visual grounding",
        },
        "chunk_and_hierarchy": {
            "reviewer_concern": "Ablations confound chunk size and hierarchical stages",
            "result_files": ["table_controlled_ablations.csv"],
            "compare": ["added_chunk_6000", "added_chunk_9000", "added_chunk_12000", "added_l3_off_frozen_l2", "added_l3_on_frozen_l2"],
            "manuscript_location": "Experiments > Controlled ablations",
        },
        "relation_completion": {
            "reviewer_concern": "Need to isolate relation completion",
            "result_files": ["table_controlled_ablations.csv"],
            "compare": ["added_relation_completion_off", "added_relation_completion_on"],
            "manuscript_location": "Experiments > Relation-completion ablation",
        },
        "semantic_core_compression": {
            "reviewer_concern": "Aggressive semantic-core selection changes the task from exhaustive extraction to compression",
            "result_files": ["table_compression_ablation.csv"],
            "manuscript_location": "Ablation > Semantic-core compression",
        },
    }
    write_json(output_dir / "reviewer_result_mapping.json", mapping)
    print(f"Wrote {len(written) + 2} manuscript artifacts to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
