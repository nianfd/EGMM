from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


HIGHER_IS_BETTER = {
    "BEGQ_balanced_evidence_grounded_quality",
    "EGR_evidence_graph_reliability",
    "RMR_reliable_method_reproducibility",
    "NET_node_evidence_traceability",
    "AMG_actual_multimodal_grounding",
    "ESGC_evidence_supported_graph_connectivity",
    "TLP_traceable_link_purity",
    "EGMR_evidence_grounded_method_reproducibility",
    "valid_link_ratio",
    "evidence_supported_link_ratio",
    "linked_node_ratio",
    "items_with_evidence_ratio",
    "avg_evidence_refs_per_item",
    "resolved_evidence_ref_ratio",
    "indexed_evidence_ref_ratio",
    "visual_evidence_ratio_strict",
    "visual_evidence_ratio_relaxed",
    "actual_visual_context_ratio",
    "fine_granularity_ratio",
    "auto_fine_granularity_ratio",
    "avg_confidence",
    "avg_method_reproducibility_score",
}

LOWER_IS_BETTER = {
    "isolated_node_ratio",
    "inferred_link_ratio",
}

COUNT_METRICS = {
    "num_research_problems",
    "num_methods",
    "num_links",
}

MAIN_METRICS = [
    "BEGQ_balanced_evidence_grounded_quality",
    "NET_node_evidence_traceability",
    "AMG_actual_multimodal_grounding",
    "ESGC_evidence_supported_graph_connectivity",
    "TLP_traceable_link_purity",
    "EGMR_evidence_grounded_method_reproducibility",
]


def read_rows(path: Path) -> list[dict[str, str]]:
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8-sig"))
        if isinstance(data, list):
            return [{k: str(v) for k, v in row.items()} for row in data if isinstance(row, dict)]
        if isinstance(data, dict):
            rows = data.get("rows") or data.get("metrics") or data.get("data")
            if isinstance(rows, list):
                return [{k: str(v) for k, v in row.items()} for row in rows if isinstance(row, dict)]
            return [{k: str(v) for k, v in data.items()}]
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def to_float(value: str) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def get_row(rows: list[dict[str, str]], condition: str) -> dict[str, str] | None:
    for row in rows:
        if row.get("condition") == condition:
            return row
    return None


def fmt(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value:.4f}"


def compare_metric(proposed: dict[str, str], other: dict[str, str], metric: str) -> tuple[float | None, float | None, float | None]:
    p = to_float(proposed.get(metric, ""))
    o = to_float(other.get(metric, ""))
    if p is None or o is None:
        return p, o, None
    return p, o, p - o


def best_condition(rows: list[dict[str, str]], metric: str, higher: bool = True) -> tuple[str, float] | None:
    values = []
    for row in rows:
        value = to_float(row.get(metric, ""))
        if value is not None:
            values.append((row.get("condition", "unknown"), value))
    if not values:
        return None
    return max(values, key=lambda x: x[1]) if higher else min(values, key=lambda x: x[1])


def generate_report(rows: list[dict[str, str]]) -> str:
    proposed = get_row(rows, "proposed_full")
    if proposed is None:
        return "没有找到 `proposed_full` 行，无法生成方法优劣分析。"

    lines = []
    lines.append("# Metrics Analysis Report")
    lines.append("")
    lines.append("## 总体结论")
    lines.append("")
    lines.append("以下结论由自动指标表生成。数量类指标只反映输出规模，不能单独代表质量；证据链、图连通性、视觉证据和方法可复现性指标更适合支撑本文方法优势。")
    lines.append("")

    lines.append("## Proposed Method 核心指标")
    lines.append("")
    for metric in MAIN_METRICS:
        if metric in proposed:
            lines.append(f"- `{metric}`: {proposed.get(metric)}")
    lines.append("")

    lines.append("## 与 Baseline / Ablation 的对比")
    lines.append("")
    for other_name in [
        "baseline_oneshot_mineru",
        "ablation_text_only",
        "ablation_no_l3",
        "ablation_large_chunk",
    ]:
        other = get_row(rows, other_name)
        if other is None:
            continue
        lines.append(f"### proposed_full vs {other_name}")
        for metric in MAIN_METRICS:
            if metric not in proposed or metric not in other:
                continue
            p, o, diff = compare_metric(proposed, other, metric)
            if diff is None:
                continue
            direction = "更高" if diff > 0 else "更低" if diff < 0 else "相同"
            lines.append(f"- `{metric}`: proposed={fmt(p)}, {other_name}={fmt(o)}, 差值={fmt(diff)}，proposed {direction}。")
        lines.append("")
        lines.append("补充指标：")
        for metric in sorted(HIGHER_IS_BETTER - set(MAIN_METRICS)):
            if metric not in proposed or metric not in other:
                continue
            p, o, diff = compare_metric(proposed, other, metric)
            if diff is None:
                continue
            direction = "更高" if diff > 0 else "更低" if diff < 0 else "相同"
            lines.append(f"- `{metric}`: proposed={fmt(p)}, {other_name}={fmt(o)}, 差值={fmt(diff)}，proposed {direction}。")
        for metric in sorted(LOWER_IS_BETTER):
            if metric not in proposed or metric not in other:
                continue
            p, o, diff = compare_metric(proposed, other, metric)
            if diff is None:
                continue
            direction = "更优" if diff < 0 else "更差" if diff > 0 else "相同"
            lines.append(f"- `{metric}`: proposed={fmt(p)}, {other_name}={fmt(o)}, 差值={fmt(diff)}，该指标越低越好，proposed {direction}。")
        lines.append("")

    lines.append("## 各指标最优条件")
    lines.append("")
    for metric in MAIN_METRICS:
        best = best_condition(rows, metric, higher=True)
        if best:
            lines.append(f"- `{metric}` 最优: `{best[0]}` = {best[1]:.4f}")
    lines.append("")
    lines.append("补充指标最优条件：")
    for metric in sorted(HIGHER_IS_BETTER - set(MAIN_METRICS)):
        best = best_condition(rows, metric, higher=True)
        if best:
            lines.append(f"- `{metric}` 最优: `{best[0]}` = {best[1]:.4f}")
    for metric in sorted(LOWER_IS_BETTER):
        best = best_condition(rows, metric, higher=False)
        if best:
            lines.append(f"- `{metric}` 最优: `{best[0]}` = {best[1]:.4f}，该指标越低越好。")
    lines.append("")

    lines.append("## 论文可写结论模板")
    lines.append("")
    lines.append("可以优先从以下角度写结论：")
    lines.append("")
    lines.append("1. 如果 `NET_node_evidence_traceability` 较高，说明 proposed method 的证据链更完整、更可追溯。")
    lines.append("2. 如果 `AMG_actual_multimodal_grounding` 高于 text-only/baseline，说明真实图文上下文对抽取有贡献。")
    lines.append("3. 如果 `ESGC_evidence_supported_graph_connectivity` 较高，说明更多节点进入了证据支持的问题-方法图结构。")
    lines.append("4. 如果 `TLP_traceable_link_purity` 较高，说明图中的有效边更多来自证据支持关系，而不是后验补全。")
    lines.append("5. 如果 `EGMR_evidence_grounded_method_reproducibility` 高于 no-L3，说明 L3 审校提高了有证据支撑的方法可复现性。")
    lines.append("")
    lines.append("需要谨慎：`num_research_problems`、`num_methods`、`num_links` 不能单独证明方法更好，它们需要结合人工 faithfulness/granularity 评价解释。")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze metrics_summary.csv/json and generate a Chinese report.")
    parser.add_argument("--csv", default="experiments/results/paper1/metrics_summary.csv")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    csv_path = Path(args.csv).resolve()
    rows = read_rows(csv_path)
    report = generate_report(rows)
    output = Path(args.output).resolve() if args.output else csv_path.with_name("metrics_analysis_report.md")
    output.write_text(report, encoding="utf-8")
    print(report)
    print(f"\nReport written to: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
