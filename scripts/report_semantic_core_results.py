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


DEFAULT_BASELINES = [
    "original_main",
    "original_baseline_oneshot_mineru",
    "added_global_eg_merge",
    "added_bm25_rag_text",
    "added_chunk_6000",
]
DISPLAY = {
    "original_main": "Original main method (frozen)",
    "original_baseline_oneshot_mineru": "Original one-shot MinerU (frozen)",
    "added_global_eg_merge": "Global evidence-graph merge",
    "added_bm25_rag_text": "BM25-RAG text",
    "added_chunk_6000": "6,000-character multimodal pipeline",
    "added_semantic_core_6000": "Proposed-SC (6k + semantic core)",
}
CORE_METRICS = ("problem", "method", "link")


def _number(value: Any) -> float:
    return float(value)


def _metric_row(item: dict[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {
        "method": DISPLAY.get(item["condition"], item["condition"]),
        "condition": item["condition"],
        "n_papers": item["paper_count"],
        "gold_status": ";".join(item.get("gold_statuses", [])),
    }
    for metric in CORE_METRICS:
        micro = item[f"{metric}_micro"]
        row[f"{metric}_precision"] = _number(micro["precision"])
        row[f"{metric}_recall"] = _number(micro["recall"])
        row[f"{metric}_f1"] = _number(micro["f1"])
        row[f"{metric}_macro_f1"] = _number(item[f"{metric}_macro_f1"])
        interval = item[f"{metric}_macro_f1_ci95"]
        row[f"{metric}_macro_ci95_low"] = _number(interval[0])
        row[f"{metric}_macro_ci95_high"] = _number(interval[1])
    for metric in (
        "exact_node_reference_rate",
        "strict_visual_evidence_rate",
        "evidence_supported_link_rate",
        "strict_relation_evidence_rate",
    ):
        row[metric] = _number(item[metric])
    return row


def _latex_escape(text: str) -> str:
    for old, new in (("\\", r"\textbackslash{}"), ("_", r"\_"), ("%", r"\%"), ("&", r"\&")):
        text = text.replace(old, new)
    return text


def _write_latex(path: Path, rows: list[dict[str, Any]], best: dict[str, float]) -> None:
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Thirty-paper comparison against the frozen Qwen3.8-Max reference generated with the submitted 9,000-character pipeline. The complete 6,000-character Qwen-VL-Max pipeline is the revised proposed method. P, M, and L denote research-problem, method, and problem--method-link extraction.}",
        r"\label{tab:semantic_core_dev30}",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"Method & P-P & P-R & P-F1 & M-F1 & L-F1 & ExactRef \\",
        r"\midrule",
    ]
    for row in rows:
        values = []
        for key in ("problem_precision", "problem_recall", "problem_f1", "method_f1", "link_f1", "exact_node_reference_rate"):
            rendered = f"{100.0 * float(row[key]):.2f}"
            if key in best and abs(float(row[key]) - best[key]) < 1e-12:
                rendered = rf"\textbf{{{rendered}}}"
            values.append(rendered)
        lines.append(_latex_escape(str(row["method"])) + " & " + " & ".join(values) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a 30-paper comparison against the main-method-aligned Qwen3.8-Max reference."
    )
    parser.add_argument("--metrics-dir", required=True, help="Directory produced by evaluate_revision_experiments.py")
    parser.add_argument("--output-dir")
    parser.add_argument("--target-condition", default="added_semantic_core_6000")
    parser.add_argument("--baselines", nargs="*", default=DEFAULT_BASELINES)
    parser.add_argument("--expected-papers", type=int, default=30)
    parser.add_argument("--compatibility-report", help="Optional metric_compatibility.json from strict validation")
    args = parser.parse_args()

    metrics_dir = Path(args.metrics_dir).resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else metrics_dir / "semantic_core_report"
    summary = read_json(metrics_dir / "summary.json")
    by_condition = {str(item["condition"]): item for item in summary}
    selected = [*args.baselines, args.target_condition]
    missing = [condition for condition in selected if condition not in by_condition]
    if missing:
        raise SystemExit(f"Missing evaluated conditions in summary.json: {missing}")

    rows = [_metric_row(by_condition[condition]) for condition in selected]
    wrong_counts = {
        row["condition"]: row["n_papers"]
        for row in rows
        if int(row["n_papers"]) != args.expected_papers
    }
    baseline_rows = [row for row in rows if row["condition"] in args.baselines]
    target = next(row for row in rows if row["condition"] == args.target_condition)
    comparisons: dict[str, Any] = {}
    strict_wins = []
    noninferior = []
    for metric in CORE_METRICS:
        key = f"{metric}_f1"
        best_baseline_row = max(baseline_rows, key=lambda row: float(row[key]))
        difference = float(target[key]) - float(best_baseline_row[key])
        comparisons[metric] = {
            "target_f1": float(target[key]),
            "best_baseline_condition": best_baseline_row["condition"],
            "best_baseline_f1": float(best_baseline_row[key]),
            "absolute_difference": difference,
            "strictly_best": difference > 1e-12,
            "best_or_tied": difference >= -1e-12,
        }
        strict_wins.append(difference > 1e-12)
        noninferior.append(difference >= -1e-12)

    compatibility_valid: bool | None = None
    compatibility_detail = "not supplied"
    if args.compatibility_report:
        report = read_json(Path(args.compatibility_report).resolve())
        compatibility_valid = bool(report.get("valid")) and int(report.get("failed_count", 0)) == 0
        compatibility_detail = str(Path(args.compatibility_report).resolve())

    gold_statuses = sorted({status for row in rows for status in str(row["gold_status"]).split(";") if status})
    complete = not wrong_counts
    strictly_best_all = all(strict_wins)
    best_or_tied_all = all(noninferior)
    eligible = complete and strictly_best_all and compatibility_valid is not False
    gate = {
        "target_condition": args.target_condition,
        "evaluation_role": "30-paper overall evaluation requested for the revision experiments",
        "reference_standard": (
            "Qwen3.8-Max main-method-aligned reference stored under the backward-compatible "
            "filename human_gold.json"
        ),
        "gold_statuses": gold_statuses,
        "expected_papers_per_condition": args.expected_papers,
        "paper_counts_complete": complete,
        "wrong_paper_counts": wrong_counts,
        "metric_comparisons": comparisons,
        "strictly_best_on_problem_method_and_link_f1": strictly_best_all,
        "best_or_tied_on_problem_method_and_link_f1": best_or_tied_all,
        "metric_compatibility_valid": compatibility_valid,
        "metric_compatibility_source": compatibility_detail,
        "recommended_to_expand_structural_run_to_100": eligible,
        "decision_rule": (
            "Expand only if all selected conditions have the expected 30 papers, Proposed-SC is strictly higher "
            "than every listed baseline on micro problem-F1, method-F1, and link-F1, and any supplied strict "
            "metric-compatibility report has zero failures. This script reports results; it never changes predictions."
        ),
    }

    write_csv(output_dir / "semantic_core_30_comparison.csv", rows)
    best = {
        key: max(float(row[key]) for row in rows)
        for key in ("problem_precision", "problem_recall", "problem_f1", "method_f1", "link_f1", "exact_node_reference_rate")
    }
    _write_latex(output_dir / "semantic_core_30_comparison.tex", rows, best)
    write_json(output_dir / "semantic_core_gate.json", gate)
    print(json.dumps({
        "comparison_csv": str(output_dir / "semantic_core_30_comparison.csv"),
        "latex_table": str(output_dir / "semantic_core_30_comparison.tex"),
        "gate_report": str(output_dir / "semantic_core_gate.json"),
        "recommended_to_expand_structural_run_to_100": eligible,
    }, ensure_ascii=False, indent=2))
    # Reporting must remain scriptable even when the target is not strictly
    # best; the decision is recorded in semantic_core_gate.json instead of
    # being encoded as a process failure.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
