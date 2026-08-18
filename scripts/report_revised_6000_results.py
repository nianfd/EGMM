from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from revision.io_utils import read_json, write_csv, write_json


MAIN_CONDITIONS = [
    "added_chunk_6000",
    "original_main",
    "original_baseline_oneshot_mineru",
    "added_global_eg_merge",
    "added_bm25_rag_text",
]
DISPLAY = {
    "added_chunk_6000": "Revised proposed method (6k multimodal)",
    "original_main": "Submitted main configuration (9k, frozen)",
    "original_baseline_oneshot_mineru": "One-shot MinerU (frozen)",
    "added_global_eg_merge": "Global evidence-graph merge",
    "added_bm25_rag_text": "BM25-RAG text",
    "added_semantic_core_6000": "Semantic-core compression ablation",
}


def _metric_row(item: dict[str, Any]) -> dict[str, Any]:
    condition = str(item["condition"])
    row: dict[str, Any] = {
        "method": DISPLAY.get(condition, condition),
        "condition": condition,
        "n_papers": item["paper_count"],
        "gold_status": ";".join(item.get("gold_statuses", [])),
    }
    for metric in ("problem", "method", "link", "link_conditional_on_matched_nodes"):
        micro = item[f"{metric}_micro"]
        row[f"{metric}_precision"] = float(micro["precision"])
        row[f"{metric}_recall"] = float(micro["recall"])
        row[f"{metric}_f1"] = float(micro["f1"])
        row[f"{metric}_macro_f1"] = float(item[f"{metric}_macro_f1"])
        interval = item[f"{metric}_macro_f1_ci95"]
        row[f"{metric}_macro_ci95_low"] = float(interval[0])
        row[f"{metric}_macro_ci95_high"] = float(interval[1])
    row["exact_node_reference_rate"] = (
        "N/A" if condition == "original_baseline_oneshot_mineru"
        else float(item["exact_node_reference_rate"])
    )
    for count_name in (
        "predicted_problems", "gold_problems", "matched_problems",
        "predicted_to_gold_problem_ratio", "predicted_methods", "gold_methods",
        "matched_methods", "predicted_to_gold_method_ratio",
    ):
        row[f"mean_{count_name}"] = float(item[f"mean_{count_name}"])
    return row


def _escape(text: str) -> str:
    for old, new in (("\\", r"\textbackslash{}"), ("_", r"\_"), ("%", r"\%"), ("&", r"\&")):
        text = text.replace(old, new)
    return text


def _render_percent(value: Any, *, best: float | None = None) -> str:
    if value == "N/A":
        return "N/A"
    number = float(value)
    rendered = f"{100.0 * number:.2f}"
    if best is not None and abs(number - best) < 1e-12:
        return rf"\textbf{{{rendered}}}"
    return rendered


def _write_latex(path: Path, rows: list[dict[str, Any]]) -> None:
    keys = (
        "problem_precision", "problem_recall", "problem_f1",
        "method_f1", "link_f1", "link_conditional_on_matched_nodes_f1",
        "exact_node_reference_rate",
    )
    best = {
        key: max(float(row[key]) for row in rows if row[key] != "N/A")
        for key in keys
    }
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Thirty-paper semantic evaluation against the frozen Qwen3.8-Max reference generated with the submitted 9,000-character pipeline. The revised proposed method is the 6,000-character Qwen-VL-Max pipeline. Nodes are paired by BGE-M3 cosine similarity and one-to-one Hungarian assignment. E2E-L evaluates links end to end; C-L evaluates relation labels conditional on correctly matched endpoint nodes.}",
        r"\label{tab:revised_main_semantic30}",
        r"\begin{tabular}{lccccccc}",
        r"\toprule",
        r"Method & P-P & P-R & P-F1 & M-F1 & E2E-L & C-L & ExactRef \\",
        r"\midrule",
    ]
    for row in rows:
        values = [_render_percent(row[key], best=best[key]) for key in keys]
        lines.append(_escape(str(row["method"])) + " & " + " & ".join(values) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Report the revised 6k main method under semantic matching")
    parser.add_argument("--metrics-dir", required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--expected-papers", type=int, default=30)
    parser.add_argument("--include-compression-ablation", action="store_true")
    args = parser.parse_args()

    metrics_dir = Path(args.metrics_dir).resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else metrics_dir / "revised_6000_report"
    summary = read_json(metrics_dir / "summary.json")
    by_condition = {str(item["condition"]): item for item in summary}
    conditions = list(MAIN_CONDITIONS)
    if args.include_compression_ablation:
        conditions.append("added_semantic_core_6000")
    missing = [condition for condition in conditions if condition not in by_condition]
    if missing:
        raise SystemExit(f"Missing evaluated conditions: {missing}")
    rows = [_metric_row(by_condition[condition]) for condition in conditions]
    wrong_counts = {
        row["condition"]: row["n_papers"]
        for row in rows if int(row["n_papers"]) != args.expected_papers
    }
    if wrong_counts:
        raise SystemExit(f"Unexpected paper counts: {wrong_counts}")

    write_csv(output_dir / "revised_6000_semantic_comparison.csv", rows)
    _write_latex(output_dir / "revised_6000_semantic_comparison.tex", rows)
    diagnostic_rows = [{
        "method": row["method"],
        "condition": row["condition"],
        "mean_predicted_problems": row["mean_predicted_problems"],
        "mean_gold_problems": row["mean_gold_problems"],
        "problem_output_ratio": row["mean_predicted_to_gold_problem_ratio"],
        "mean_predicted_methods": row["mean_predicted_methods"],
        "mean_gold_methods": row["mean_gold_methods"],
        "method_output_ratio": row["mean_predicted_to_gold_method_ratio"],
    } for row in rows]
    write_csv(output_dir / "node_count_and_compression_diagnostics.csv", diagnostic_rows)

    protocol = read_json(metrics_dir / "matching_protocol.json")
    paired = read_json(metrics_dir / "paired_tests.json")
    selected_paired = [
        item for item in paired
        if str(item.get("comparison") or "").startswith("added_chunk_6000-vs-")
    ]
    write_csv(output_dir / "revised_6000_paired_tests.csv", selected_paired)
    manifest = {
        "revised_main_condition": "added_chunk_6000",
        "submitted_frozen_condition": "original_main",
        "semantic_core_role": "compression ablation, not the revised main method",
        "matching_protocol": protocol,
        "paper_count": args.expected_papers,
        "files": [
            "revised_6000_semantic_comparison.csv",
            "revised_6000_semantic_comparison.tex",
            "node_count_and_compression_diagnostics.csv",
            "revised_6000_paired_tests.csv",
        ],
    }
    write_json(output_dir / "revised_method_report_manifest.json", manifest)
    print(json.dumps({
        "comparison_csv": str(output_dir / "revised_6000_semantic_comparison.csv"),
        "latex_table": str(output_dir / "revised_6000_semantic_comparison.tex"),
        "diagnostics": str(output_dir / "node_count_and_compression_diagnostics.csv"),
        "manifest": str(output_dir / "revised_method_report_manifest.json"),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
