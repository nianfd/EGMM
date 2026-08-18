from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from revision.io_utils import file_sha256, read_json, write_csv, write_json

REVISED_MAIN_CONDITION = "added_chunk_6000"
SUBMITTED_MAIN_STRUCTURAL_CONDITION = "proposed_full"
SUBMITTED_MAIN_SEMANTIC_CONDITION = "original_main"
REFERENCE_STATUS = "qwen38_max_main_method_aligned_gold"
REFERENCE_CHUNK_CHARS = 9000


STRUCTURAL_MAIN = [
    ("baseline_oneshot_mineru", "One-shot MinerU", False),
    ("added_bm25_rag_text", "BM25-RAG text", True),
    ("added_global_eg_merge", "Global evidence-graph merge", True),
    (SUBMITTED_MAIN_STRUCTURAL_CONDITION, "Submitted main configuration (9k, frozen)", True),
    (REVISED_MAIN_CONDITION, "Revised proposed method (6k multimodal)", True),
]

SEMANTIC_MAIN = [
    (REVISED_MAIN_CONDITION, "Revised proposed method (6k multimodal)"),
    (SUBMITTED_MAIN_SEMANTIC_CONDITION, "Submitted main configuration (9k, frozen)"),
    ("original_baseline_oneshot_mineru", "One-shot MinerU (frozen)"),
    ("added_global_eg_merge", "Global evidence-graph merge"),
    ("added_bm25_rag_text", "BM25-RAG text"),
    ("added_semantic_core_6000", "Semantic-core compression ablation"),
]

CHUNK_SWEEP = [
    (REVISED_MAIN_CONDITION, "6,000 (proposed)"),
    ("added_chunk_9000", "9,000"),
    ("added_chunk_12000", "12,000"),
]

STRUCTURAL_METRICS = ("NET", "AMG", "ESGC", "TLP", "EGMR", "BEGQ")
CHUNK_METRICS = ("ESGC", "TLP", "EGMR", "BEGQ")
SEMANTIC_METRICS = ("problem_precision", "problem_recall", "problem_f1", "method_f1", "link_f1", "conditional_link_f1", "exact_reference")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _number(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} is not numeric: {value!r}") from exc
    if not math.isfinite(result):
        raise ValueError(f"{label} is not finite: {value!r}")
    return result


def _structural_value(row: dict[str, str], metric: str) -> float:
    for key in (metric, f"{metric}_mean"):
        if key in row and str(row[key]).strip() != "":
            return _number(row[key], f"{row.get('condition')}:{key}")
    raise KeyError(f"Missing structural metric {metric} for {row.get('condition')}")


def _paper_count(row: dict[str, Any], *keys: str) -> int:
    for key in keys:
        if key in row and str(row[key]).strip() != "":
            return int(float(row[key]))
    raise KeyError(f"Missing paper count in row: {row}")


def _semantic_value(item: dict[str, Any], metric: str) -> float | str:
    if metric == "problem_precision":
        return float(item["problem_micro"]["precision"])
    if metric == "problem_recall":
        return float(item["problem_micro"]["recall"])
    if metric == "problem_f1":
        return float(item["problem_micro"]["f1"])
    if metric == "method_f1":
        return float(item["method_micro"]["f1"])
    if metric == "link_f1":
        return float(item["link_micro"]["f1"])
    if metric == "conditional_link_f1":
        return float(item["link_conditional_on_matched_nodes_micro"]["f1"])
    if metric == "exact_reference":
        if item["condition"] == "original_baseline_oneshot_mineru":
            return "N/A"
        return float(item["exact_node_reference_rate"])
    raise KeyError(metric)


def _latex_escape(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}", "&": r"\&", "%": r"\%",
        "_": r"\_", "#": r"\#", "$": r"\$",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def _best(rows: list[dict[str, Any]], metrics: tuple[str, ...]) -> dict[str, float]:
    return {
        metric: max(float(row[metric]) for row in rows if row[metric] != "N/A")
        for metric in metrics
    }


def _fmt(value: Any, best: float | None, *, percent: bool = False) -> str:
    if value == "N/A":
        return "N/A"
    number = float(value)
    rendered = f"{100.0 * number:.2f}" if percent else f"{number:.3f}"
    if best is not None and abs(number - best) < 1e-12:
        return rf"\textbf{{{rendered}}}"
    return rendered


def _render_table1(rows: list[dict[str, Any]]) -> str:
    best = _best(rows, STRUCTURAL_METRICS)
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Structural evidence metrics on all 100 papers. The complete 6,000-character multimodal pipeline is the revised proposed method; the submitted 9,000-character configuration is retained as a frozen comparator. These metrics evaluate provenance and schema compliance, not semantic correctness. One-shot MinerU does not produce the staged evidence-index schema, so its structural entries are not applicable.}",
        r"\label{tab:structural_main}",
        r"\setlength{\tabcolsep}{4.2pt}",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"Method & NET & AMG & ESGC & TLP & EGMR & BEGQ \\",
        r"\midrule",
    ]
    for row in rows:
        values = [_fmt(row[metric], best[metric]) for metric in STRUCTURAL_METRICS]
        lines.append(_latex_escape(row["method"]) + " & " + " & ".join(values) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}"])
    return "\n".join(lines)


def _render_table2(rows: list[dict[str, Any]]) -> str:
    best = _best(rows, SEMANTIC_METRICS)
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Thirty-paper semantic evaluation against the frozen Qwen3.8-Max reference generated with the submitted 9,000-character pipeline. The revised proposed method is the 6,000-character Qwen-VL-Max pipeline. Nodes are paired by BGE-M3 cosine similarity and one-to-one Hungarian assignment. E2E-L evaluates links end to end; C-L evaluates relation labels conditional on correctly matched endpoint nodes.}",
        r"\label{tab:gold_semantic}",
        r"\setlength{\tabcolsep}{4.2pt}",
        r"\begin{tabular}{lccccccc}",
        r"\toprule",
        r"Method & P-P & P-R & P-F1 & M-F1 & E2E-L & C-L & ExactRef \\",
        r"\midrule",
    ]
    for row in rows:
        values = [_fmt(row[metric], best[metric], percent=True) for metric in SEMANTIC_METRICS]
        lines.append(_latex_escape(row["method"]) + " & " + " & ".join(values) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}"])
    return "\n".join(lines)


def _render_table3(rows: list[dict[str, Any]]) -> str:
    best = _best(rows, CHUNK_METRICS)
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Controlled chunk-size sweep on all 100 papers. The 6,000-character setting is selected as the revised proposed configuration; 9,000 and 12,000 characters are controlled alternatives.}",
        r"\label{tab:chunk_control}",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Limit & ESGC & TLP & EGMR & BEGQ \\",
        r"\midrule",
    ]
    for row in rows:
        values = [_fmt(row[metric], best[metric]) for metric in CHUNK_METRICS]
        lines.append(_latex_escape(row["limit"]) + " & " + " & ".join(values) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    return "\n".join(lines)


def build_tables(
    structural_summary_csv: Path,
    semantic_metrics_dir: Path,
    output_dir: Path,
    *,
    expected_structural_papers: int = 100,
    expected_semantic_papers: int = 30,
    expected_problem_threshold: float = 0.70,
    expected_method_threshold: float = 0.70,
) -> dict[str, Any]:
    structural_source = _read_csv(structural_summary_csv)
    structural_by_condition = {row["condition"]: row for row in structural_source}
    required_structural = {condition for condition, _, _ in STRUCTURAL_MAIN} | {
        condition for condition, _ in CHUNK_SWEEP
    }
    missing_structural = sorted(required_structural - set(structural_by_condition))
    if missing_structural:
        raise ValueError(f"Structural summary is missing conditions: {missing_structural}")
    for condition in required_structural:
        count = _paper_count(structural_by_condition[condition], "num_successful_papers", "paper_count")
        if count != expected_structural_papers:
            raise ValueError(
                f"Structural condition {condition} has {count} papers; expected {expected_structural_papers}"
            )

    summary_path = semantic_metrics_dir / "summary.json"
    protocol_path = semantic_metrics_dir / "matching_protocol.json"
    semantic_source = read_json(summary_path)
    protocol = read_json(protocol_path)
    if "hungarian" not in str(protocol.get("matcher") or "").lower():
        raise ValueError(f"Semantic metrics are not from the Hungarian evaluator: {protocol}")
    for key, expected in (
        ("problem_similarity_threshold", expected_problem_threshold),
        ("method_similarity_threshold", expected_method_threshold),
    ):
        actual = _number(protocol.get(key), key)
        if abs(actual - expected) > 1e-12:
            raise ValueError(f"{key}={actual}; expected the prespecified primary value {expected}")
    semantic_by_condition = {str(item["condition"]): item for item in semantic_source}
    missing_semantic = sorted({condition for condition, _ in SEMANTIC_MAIN} - set(semantic_by_condition))
    if missing_semantic:
        raise ValueError(f"Semantic summary is missing conditions: {missing_semantic}")
    for condition, _ in SEMANTIC_MAIN:
        count = _paper_count(semantic_by_condition[condition], "paper_count")
        if count != expected_semantic_papers:
            raise ValueError(
                f"Semantic condition {condition} has {count} papers; expected {expected_semantic_papers}"
            )
        statuses = semantic_by_condition[condition].get("gold_statuses")
        if statuses != [REFERENCE_STATUS]:
            raise ValueError(
                f"Semantic condition {condition} has gold_statuses={statuses!r}; "
                f"expected the frozen 9k reference status {[REFERENCE_STATUS]!r}"
            )

    table1_rows: list[dict[str, Any]] = []
    for condition, display, applicable in STRUCTURAL_MAIN:
        source = structural_by_condition[condition]
        row: dict[str, Any] = {
            "method": display, "condition": condition,
            "n_papers": expected_structural_papers,
            "structural_metrics_applicable": applicable,
        }
        for metric in STRUCTURAL_METRICS:
            row[metric] = _structural_value(source, metric) if applicable else "N/A"
        table1_rows.append(row)

    table2_rows: list[dict[str, Any]] = []
    for condition, display in SEMANTIC_MAIN:
        source = semantic_by_condition[condition]
        row = {
            "method": display, "condition": condition,
            "n_papers": expected_semantic_papers,
        }
        for metric in SEMANTIC_METRICS:
            row[metric] = _semantic_value(source, metric)
        table2_rows.append(row)

    table3_rows: list[dict[str, Any]] = []
    for condition, limit in CHUNK_SWEEP:
        source = structural_by_condition[condition]
        row = {
            "limit": limit, "condition": condition,
            "n_papers": expected_structural_papers,
        }
        for metric in CHUNK_METRICS:
            row[metric] = _structural_value(source, metric)
        table3_rows.append(row)

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "table1_structural_100.csv", table1_rows)
    write_csv(output_dir / "table2_semantic_30.csv", table2_rows)
    write_csv(output_dir / "table3_chunk_100.csv", table3_rows)
    table1_tex = _render_table1(table1_rows)
    table2_tex = _render_table2(table2_rows)
    table3_tex = _render_table3(table3_rows)
    (output_dir / "table1_structural_100.tex").write_text(table1_tex + "\n", encoding="utf-8")
    (output_dir / "table2_semantic_30.tex").write_text(table2_tex + "\n", encoding="utf-8")
    (output_dir / "table3_chunk_100.tex").write_text(table3_tex + "\n", encoding="utf-8")
    (output_dir / "article_tables_1_2_3.tex").write_text(
        table1_tex + "\n\n" + table2_tex + "\n\n" + table3_tex + "\n",
        encoding="utf-8",
    )
    manifest = {
        "status": "complete",
        "cohorts": {
            "table_1_structural": expected_structural_papers,
            "table_2_semantic_reference": expected_semantic_papers,
            "table_3_chunk_ablation": expected_structural_papers,
        },
        "revised_main_condition": REVISED_MAIN_CONDITION,
        "revised_main_definition": "complete 6,000-character multimodal Qwen-VL-Max pipeline",
        "submitted_frozen_condition": (
            f"{SUBMITTED_MAIN_STRUCTURAL_CONDITION} / {SUBMITTED_MAIN_SEMANTIC_CONDITION}"
        ),
        "semantic_reference": {
            "status": REFERENCE_STATUS,
            "model": "qwen3.8-max",
            "chunk_chars": REFERENCE_CHUNK_CHARS,
            "regeneration_required": False,
        },
        "semantic_core_role": "compression ablation",
        "semantic_matching": protocol,
        "sources": {
            "structural_summary_csv": str(structural_summary_csv),
            "structural_summary_sha256": file_sha256(structural_summary_csv),
            "semantic_summary_json": str(summary_path),
            "semantic_summary_sha256": file_sha256(summary_path),
            "matching_protocol_json": str(protocol_path),
            "matching_protocol_sha256": file_sha256(protocol_path),
        },
    }
    write_json(output_dir / "tables_1_2_3_manifest.json", manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build revised manuscript Tables I-III from 100-paper structural and 30-paper semantic results."
    )
    parser.add_argument("--structural-summary-csv", required=True)
    parser.add_argument("--semantic-metrics-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-structural-papers", type=int, default=100)
    parser.add_argument("--expected-semantic-papers", type=int, default=30)
    parser.add_argument("--expected-problem-threshold", type=float, default=0.70)
    parser.add_argument("--expected-method-threshold", type=float, default=0.70)
    args = parser.parse_args()
    manifest = build_tables(
        Path(args.structural_summary_csv).resolve(),
        Path(args.semantic_metrics_dir).resolve(),
        Path(args.output_dir).resolve(),
        expected_structural_papers=args.expected_structural_papers,
        expected_semantic_papers=args.expected_semantic_papers,
        expected_problem_threshold=args.expected_problem_threshold,
        expected_method_threshold=args.expected_method_threshold,
    )
    print(json.dumps({
        "status": manifest["status"],
        "cohorts": manifest["cohorts"],
        "output_dir": str(Path(args.output_dir).resolve()),
        "combined_latex": str(Path(args.output_dir).resolve() / "article_tables_1_2_3.tex"),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
