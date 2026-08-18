from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


CONDITIONS = [
    "proposed_full",
    "baseline_oneshot_mineru",
    "ablation_text_only",
    "ablation_no_l3",
    "ablation_large_chunk",
]

ADDED_CONDITIONS = {
    "added_global_eg_merge",
    "added_bm25_rag_text",
    "added_schema_text_only",
    "added_visual_masked",
    "added_chunk_6000",
    "added_chunk_9000",
    "added_chunk_12000",
    "added_l3_off_frozen_l2",
    "added_l3_on_frozen_l2",
    "added_relation_completion_off",
    "added_relation_completion_on",
    "added_semantic_core_6000",
}

CORE_METRICS = [
    "NET_node_evidence_traceability",
    "AMG_actual_multimodal_grounding",
    "ESGC_evidence_supported_graph_connectivity",
    "TLP_traceable_link_purity",
    "EGMR_evidence_grounded_method_reproducibility",
    "BEGQ_balanced_evidence_grounded_quality",
]

CORE_METRIC_LABELS = {
    "NET_node_evidence_traceability": "NET",
    "AMG_actual_multimodal_grounding": "AMG",
    "ESGC_evidence_supported_graph_connectivity": "ESGC",
    "TLP_traceable_link_purity": "TLP",
    "EGMR_evidence_grounded_method_reproducibility": "EGMR",
    "BEGQ_balanced_evidence_grounded_quality": "BEGQ",
}

METHOD_LABELS = {
    "proposed_full": "Proposed",
    "baseline_oneshot_mineru": "One-shot MinerU",
    "ablation_text_only": "w/o visual context",
    "ablation_no_l3": "w/o L3 audit",
    "ablation_large_chunk": "large chunking",
    "added_global_eg_merge": "Global evidence-graph merge",
    "added_bm25_rag_text": "BM25-RAG text",
    "added_schema_text_only": "Schema-matched text-only",
    "added_visual_masked": "Visual-masked",
    "added_chunk_6000": "Chunk 6,000",
    "added_chunk_9000": "Chunk 9,000",
    "added_chunk_12000": "Chunk 12,000",
    "added_l3_off_frozen_l2": "L3 off (frozen L2)",
    "added_l3_on_frozen_l2": "L3 on (frozen L2)",
    "added_relation_completion_off": "Relation completion off",
    "added_relation_completion_on": "Relation completion on",
    "added_semantic_core_6000": "Proposed-SC (6k + semantic core)",
}

COUNT_FIELDS = [
    "num_research_problems",
    "num_methods",
    "num_nodes",
    "num_links",
    "num_valid_links",
    "num_evidence_supported_valid_links",
]

REF_RE = re.compile(
    r"([A-Za-z]?\d+_C\d+):(RP|M)-([A-Za-z0-9][A-Za-z0-9._-]*)(?![A-Za-z0-9._-])"
)
CHUNK_RE = re.compile(r"[A-Za-z]?\d+_C\d+")
VISUAL_KEYS = {
    "image",
    "images",
    "image_id",
    "image_ids",
    "image_path",
    "image_paths",
    "figures",
    "figure",
    "visual",
    "visual_cues",
    "visual_context",
    "table_image",
}
REPRO_FIELDS = ["inputs", "outputs", "procedure", "objective_or_metric"]


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def numeric_key(path: Path) -> tuple[int, str]:
    if path.name.isdigit():
        return int(path.name), path.name
    return 10**9, path.name


def nonempty(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, tuple, set, dict)):
        return bool(value)
    return True


def normalize_ref(ref: Any) -> str | None:
    if not isinstance(ref, str):
        return None
    text = ref.strip()
    if not text:
        return None
    match = REF_RE.search(text)
    if match:
        chunk, kind, local = match.groups()
        parts = re.split(r"([._-])", local)
        normalized_local = "".join(str(int(part)) if part.isdigit() else part for part in parts)
        return f"{chunk}:{kind}-{normalized_local}"
    return text


def ref_chunk(ref: Any) -> str | None:
    if not isinstance(ref, str):
        return None
    match = CHUNK_RE.search(ref)
    return match.group(0) if match else None


def canonical_id(value: Any, fallback_prefix: str, index: int) -> str:
    if isinstance(value, str) and value.strip():
        return value.strip()
    if isinstance(value, (int, float)):
        return str(value)
    return f"{fallback_prefix}{index}"


def get_first(item: dict[str, Any], keys: list[str]) -> Any:
    for key in keys:
        if key in item and item[key] not in (None, ""):
            return item[key]
    return None


def normalize_final_result(data: Any) -> dict[str, Any] | None:
    if isinstance(data, dict):
        final = data.get("final")
        if isinstance(final, dict) and has_final_keys(final):
            return final
        if has_final_keys(data):
            return data
        for key in ["result", "output", "data"]:
            value = data.get(key)
            if isinstance(value, dict) and has_final_keys(value):
                return value
    return None


def has_final_keys(data: dict[str, Any]) -> bool:
    return any(
        key in data
        for key in [
            "final_research_problems",
            "research_problems",
            "merged_research_problems",
            "paper_research_problems",
            "global_research_problems",
            "research_problem_candidates",
            "final_methods",
            "methods",
            "merged_methods",
            "paper_methods",
            "global_methods",
            "method_candidates",
            "problem_method_links",
            "links",
            "merged_links",
        ]
    )


def find_result_file(condition_dir: Path) -> Path | None:
    if not condition_dir.exists():
        return None

    priority = [
        "04_final_extraction.json",
        "final_extraction.json",
        "final.json",
        "result.json",
        "03_l2_paper_merge.json",
        "02_l2_paper_merge.json",
        "baseline_oneshot_mineru.json",
    ]
    for name in priority:
        path = condition_dir / name
        if path.exists():
            try:
                if normalize_final_result(load_json(path)) is not None:
                    return path
            except Exception:
                pass

    for path in sorted(condition_dir.rglob("*.json")):
        if path.name in {"experiment_outputs.json", "metrics_summary.json", "metrics_summary_by_paper.json"}:
            continue
        try:
            if normalize_final_result(load_json(path)) is not None:
                return path
        except Exception:
            continue
    return None


def condition_dir(paper_dir: Path, condition: str) -> Path:
    outputs = paper_dir / "outputs"
    if condition in ADDED_CONDITIONS:
        return outputs / "major_revision_additions" / condition
    if condition == "proposed_full":
        return outputs
    if condition == "baseline_oneshot_mineru":
        return outputs / "comparison_experiments" / "baseline_oneshot_mineru"
    return outputs / "comparison_experiments" / "ablations" / condition


def iter_strings(obj: Any):
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for value in obj.values():
            yield from iter_strings(value)
    elif isinstance(obj, list):
        for value in obj:
            yield from iter_strings(value)


def object_has_visual(obj: Any) -> bool:
    if isinstance(obj, dict):
        for key, value in obj.items():
            key_lower = str(key).lower()
            if any(token in key_lower for token in VISUAL_KEYS) and nonempty(value):
                return True
            if object_has_visual(value):
                return True
    elif isinstance(obj, list):
        return any(object_has_visual(value) for value in obj)
    return False


def collect_evidence_from_l1(l1_data: Any) -> tuple[set[str], set[str]]:
    refs: set[str] = set()
    visual_chunks: set[str] = set()

    def visit(obj: Any, current_chunk: str | None = None, visual_context: bool = False) -> None:
        if isinstance(obj, dict):
            chunk = current_chunk
            for key in ["chunk_id", "source_chunk_id", "chunk", "id"]:
                value = obj.get(key)
                if isinstance(value, str) and CHUNK_RE.fullmatch(value.strip()):
                    chunk = value.strip()
                    break

            local_visual = visual_context or object_has_visual(obj)
            if chunk and local_visual:
                visual_chunks.add(chunk)

            for key in [
                "local_id",
                "source_problem_local_id",
                "source_method_local_id",
                "problem_local_id",
                "method_local_id",
                "id",
            ]:
                value = obj.get(key)
                if isinstance(value, str) and chunk:
                    normalized = normalize_ref(f"{chunk}:{value}")
                    if normalized:
                        refs.add(normalized)

            for value in obj.values():
                visit(value, chunk, local_visual)
        elif isinstance(obj, list):
            for value in obj:
                visit(value, current_chunk, visual_context)

    visit(l1_data)
    return refs, visual_chunks


def collect_evidence_from_index(index_data: Any) -> tuple[set[str], set[str]]:
    refs: set[str] = set()
    visual_chunks: set[str] = set()

    for text in iter_strings(index_data):
        normalized = normalize_ref(text)
        if normalized and normalized != text or REF_RE.search(text):
            refs.add(normalized or text)
        chunk = ref_chunk(text)
        if chunk and any(token in text.lower() for token in ["image", "figure", "visual", ".png", ".jpg", ".jpeg"]):
            visual_chunks.add(chunk)

    def visit(obj: Any, current_chunk: str | None = None) -> None:
        if isinstance(obj, dict):
            chunk = current_chunk
            for value in obj.values():
                if isinstance(value, str):
                    candidate = ref_chunk(value)
                    if candidate:
                        chunk = candidate
                        break
            if chunk and object_has_visual(obj):
                visual_chunks.add(chunk)
            for value in obj.values():
                visit(value, chunk)
        elif isinstance(obj, list):
            for value in obj:
                visit(value, current_chunk)

    visit(index_data)
    return refs, visual_chunks


def context_dir_for_condition(paper_dir: Path, condition: str) -> Path:
    if condition in ADDED_CONDITIONS:
        return paper_dir / "outputs" / "major_revision_additions" / condition
    if condition == "proposed_full":
        return paper_dir / "outputs"
    if condition == "baseline_oneshot_mineru":
        return paper_dir / "outputs" / "comparison_experiments" / "baseline_oneshot_mineru"
    return paper_dir / "outputs" / "comparison_experiments" / "ablations" / condition


def load_trace_context(context_dir: Path, force_no_visual: bool = False) -> tuple[set[str], set[str]]:
    outputs = context_dir
    refs: set[str] = set()
    visual_chunks: set[str] = set()

    l1_path = outputs / "01_l1_chunk_results.json"
    if l1_path.exists():
        try:
            l1_refs, l1_visual = collect_evidence_from_l1(load_json(l1_path))
            refs.update(l1_refs)
            visual_chunks.update(l1_visual)
        except Exception:
            pass

    index_path = outputs / "02_evidence_index.json"
    if index_path.exists():
        try:
            index_refs, index_visual = collect_evidence_from_index(load_json(index_path))
            refs.update(index_refs)
            visual_chunks.update(index_visual)
        except Exception:
            pass

    if force_no_visual:
        visual_chunks = set()
    return refs, visual_chunks


def node_evidence_refs(node: dict[str, Any]) -> list[str]:
    values: list[Any] = []
    for key in ["evidence_refs", "evidence_references", "evidence_ref", "evidence"]:
        if key in node:
            values.append(node[key])
    refs: list[str] = []
    for value in values:
        if isinstance(value, str):
            refs.append(value)
        elif isinstance(value, list):
            refs.extend(item for item in value if isinstance(item, str))
        elif isinstance(value, dict):
            refs.extend(item for item in iter_strings(value) if REF_RE.search(item))
    return refs


def is_traceable(node: dict[str, Any], traceable_refs: set[str]) -> bool:
    for ref in node_evidence_refs(node):
        normalized = normalize_ref(ref)
        if normalized in traceable_refs:
            return True
    return False


def is_visual_grounded(node: dict[str, Any], traceable_refs: set[str], visual_chunks: set[str]) -> bool:
    for ref in node_evidence_refs(node):
        normalized = normalize_ref(ref)
        chunk = ref_chunk(ref)
        if normalized in traceable_refs and chunk in visual_chunks:
            return True
    return False


def get_nodes(final_data: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rps = get_first(
        final_data,
        [
            "final_research_problems",
            "research_problems",
            "merged_research_problems",
            "paper_research_problems",
            "global_research_problems",
            "research_problem_candidates",
            "research_problem_atoms",
        ],
    )
    methods = get_first(
        final_data,
        [
            "final_methods",
            "methods",
            "merged_methods",
            "paper_methods",
            "global_methods",
            "method_candidates",
            "method_atoms",
        ],
    )
    if rps is None:
        rps = []
    if methods is None:
        methods = []
    if not isinstance(rps, list):
        rps = []
    if not isinstance(methods, list):
        methods = []
    return [item for item in rps if isinstance(item, dict)], [item for item in methods if isinstance(item, dict)]


def get_links(final_data: dict[str, Any]) -> list[dict[str, Any]]:
    links = get_first(final_data, ["problem_method_links", "links", "merged_links", "relations"])
    if links is None:
        links = []
    if not isinstance(links, list):
        return []
    return [item for item in links if isinstance(item, dict)]


def rp_id(node: dict[str, Any], index: int) -> str:
    return canonical_id(
        get_first(node, ["id", "global_id", "final_id", "research_problem_id", "problem_id", "rp_id"]),
        "RP",
        index,
    )


def method_id(node: dict[str, Any], index: int) -> str:
    return canonical_id(
        get_first(node, ["id", "global_id", "final_id", "method_id", "m_id"]),
        "M",
        index,
    )


def link_source(link: dict[str, Any]) -> str | None:
    value = get_first(
        link,
        [
            "research_problem_id",
            "problem_id",
            "source_research_problem_id",
            "source_problem_id",
            "source_id",
            "source",
            "from",
        ],
    )
    return str(value).strip() if value is not None else None


def link_target(link: dict[str, Any]) -> str | None:
    value = get_first(
        link,
        [
            "method_id",
            "target_method_id",
            "target_id",
            "target",
            "to",
        ],
    )
    return str(value).strip() if value is not None else None


def link_type(link: dict[str, Any]) -> str:
    value = get_first(link, ["link_type", "type", "relation_type", "support_type"])
    return str(value).strip() if value is not None else ""


def field_value(method: dict[str, Any], field: str) -> Any:
    for container_key in ["reproducibility", "reproducibility_fields", "structured_fields"]:
        container = method.get(container_key)
        if isinstance(container, dict) and field in container:
            return container[field]
    return method.get(field)


def compute_metrics(final_data: dict[str, Any], traceable_refs: set[str], visual_chunks: set[str]) -> dict[str, Any]:
    rps, methods = get_nodes(final_data)
    links = get_links(final_data)

    rp_ids = {rp_id(node, idx + 1): node for idx, node in enumerate(rps)}
    method_ids = {method_id(node, idx + 1): node for idx, node in enumerate(methods)}
    nodes = list(rp_ids.values()) + list(method_ids.values())

    traceable_by_id: dict[str, int] = {}
    visual_by_id: dict[str, int] = {}
    for node_id, node in {**rp_ids, **method_ids}.items():
        traceable_by_id[node_id] = 1 if is_traceable(node, traceable_refs) else 0
        visual_by_id[node_id] = 1 if is_visual_grounded(node, traceable_refs, visual_chunks) else 0

    valid_links: list[tuple[str, str, dict[str, Any]]] = []
    for link in links:
        source = link_source(link)
        target = link_target(link)
        if source in rp_ids and target in method_ids:
            valid_links.append((source, target, link))

    evidence_supported_valid_links: list[tuple[str, str, dict[str, Any]]] = []
    incident_es_nodes: set[str] = set()
    for source, target, link in valid_links:
        is_es = (
            link_type(link) == "evidence_supported"
            and traceable_by_id.get(source, 0) == 1
            and traceable_by_id.get(target, 0) == 1
        )
        if is_es:
            evidence_supported_valid_links.append((source, target, link))
            incident_es_nodes.add(source)
            incident_es_nodes.add(target)

    num_nodes = len(nodes)
    num_methods = len(methods)
    num_valid_links = len(valid_links)

    net = sum(traceable_by_id.values()) / num_nodes if num_nodes else 0.0
    amg = sum(visual_by_id.values()) / num_nodes if num_nodes else 0.0
    esgc = len(incident_es_nodes) / num_nodes if num_nodes else 0.0
    tlp = len(evidence_supported_valid_links) / num_valid_links if num_valid_links else 0.0

    method_scores: list[float] = []
    for idx, method in enumerate(methods, start=1):
        mid = method_id(method, idx)
        filled = sum(1 for field in REPRO_FIELDS if nonempty(field_value(method, field)))
        method_scores.append(traceable_by_id.get(mid, 0) * filled / len(REPRO_FIELDS))
    egmr = sum(method_scores) / num_methods if num_methods else 0.0

    begq = 0.15 * net + 0.15 * amg + 0.25 * esgc + 0.25 * tlp + 0.20 * egmr

    return {
        "NET_node_evidence_traceability": net,
        "AMG_actual_multimodal_grounding": amg,
        "ESGC_evidence_supported_graph_connectivity": esgc,
        "TLP_traceable_link_purity": tlp,
        "EGMR_evidence_grounded_method_reproducibility": egmr,
        "BEGQ_balanced_evidence_grounded_quality": begq,
        "num_research_problems": len(rps),
        "num_methods": len(methods),
        "num_nodes": num_nodes,
        "num_links": len(links),
        "num_valid_links": num_valid_links,
        "num_evidence_supported_valid_links": len(evidence_supported_valid_links),
    }


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def std(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    m = mean(values)
    return math.sqrt(sum((value - m) ** 2 for value in values) / (len(values) - 1))


def summarize(rows: list[dict[str, Any]], conditions: list[str]) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    by_condition: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("status") == "ok":
            by_condition[row["condition"]].append(row)

    for condition in conditions:
        items = by_condition.get(condition, [])
        output: dict[str, Any] = {
            "condition": condition,
            "num_successful_papers": len(items),
        }
        for key in CORE_METRICS + COUNT_FIELDS:
            values = [float(item[key]) for item in items if key in item and item[key] != ""]
            output[f"{key}_mean"] = mean(values)
            output[f"{key}_std"] = std(values)
            output[f"{key}_min"] = min(values) if values else 0.0
            output[f"{key}_max"] = max(values) if values else 0.0
        summary.append(output)
    return summary


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def build_core_mean_rows(summary: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in summary:
        row: dict[str, Any] = {
            "condition": item["condition"],
            "num_successful_papers": item["num_successful_papers"],
        }
        for metric in CORE_METRICS:
            row[CORE_METRIC_LABELS[metric]] = item.get(f"{metric}_mean", 0.0)
        rows.append(row)
    return rows


def build_core_mean_std_rows(summary: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in summary:
        row: dict[str, Any] = {
            "condition": item["condition"],
            "num_successful_papers": item["num_successful_papers"],
        }
        for metric in CORE_METRICS:
            label = CORE_METRIC_LABELS[metric]
            row[f"{label}_mean"] = item.get(f"{metric}_mean", 0.0)
            row[f"{label}_std"] = item.get(f"{metric}_std", 0.0)
        rows.append(row)
    return rows


def write_latex_core_table(path: Path, summary: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    metric_labels = [CORE_METRIC_LABELS[metric] for metric in CORE_METRICS]
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Automatic evaluation results averaged over 100 CV papers. Higher values indicate better performance for all metrics.}",
        r"\label{tab:auto_metrics}",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        "Method & " + " & ".join(metric_labels) + r" \\",
        r"\midrule",
    ]
    for item in summary:
        values = []
        for metric in CORE_METRICS:
            value = float(item.get(f"{metric}_mean", 0.0))
            values.append(f"{value:.3f}")
        method = str(item["condition"]).replace("_", r"\_")
        lines.append(method + " & " + " & ".join(values) + r" \\")
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def latex_escape(text: str) -> str:
    return text.replace("_", r"\_")


def metric_mean(item: dict[str, Any], metric: str) -> float:
    return float(item.get(f"{metric}_mean", 0.0))


def write_latex_selected_table(
    path: Path,
    summary: list[dict[str, Any]],
    selected_conditions: list[str],
    caption: str,
    label: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    by_condition = {str(item["condition"]): item for item in summary}
    rows = [by_condition[name] for name in selected_conditions if name in by_condition]
    metric_labels = [CORE_METRIC_LABELS[metric] for metric in CORE_METRICS]
    best_values = {
        metric: max((metric_mean(item, metric) for item in rows), default=0.0)
        for metric in CORE_METRICS
    }

    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\setlength{\tabcolsep}{5pt}",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        "Method & " + " & ".join([f"{label}$\\uparrow$" for label in metric_labels]) + r" \\",
        r"\midrule",
    ]

    for item in rows:
        condition = str(item["condition"])
        method_name = METHOD_LABELS.get(condition, condition)
        values: list[str] = []
        for metric in CORE_METRICS:
            value = metric_mean(item, metric)
            text = f"{value:.3f}"
            if abs(value - best_values[metric]) < 1e-12:
                text = rf"\textbf{{{text}}}"
            values.append(text)
        lines.append(latex_escape(method_name) + " & " + " & ".join(values) + r" \\")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root-dir", required=True, help="Folder containing paper subfolders, e.g. data/output.")
    parser.add_argument("--output-dir", help="Where to write aggregate metric files.")
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--end", type=int)
    parser.add_argument("--conditions", nargs="*", default=CONDITIONS)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root_dir = Path(args.root_dir).resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else root_dir / "aggregate_metrics"
    conditions = [item for value in args.conditions for item in value.split(",") if item]

    papers = sorted([path for path in root_dir.iterdir() if path.is_dir()], key=numeric_key)
    if args.end is None:
        end = len(papers)
    else:
        end = args.end
    selected_papers = papers[max(args.start - 1, 0): min(end, len(papers))]

    rows: list[dict[str, Any]] = []
    missing: list[dict[str, str]] = []

    for paper_dir in selected_papers:
        for condition in conditions:
            cdir = condition_dir(paper_dir, condition)
            result_path = find_result_file(cdir)
            no_actual_visual = condition in {
                "baseline_oneshot_mineru",
                "ablation_text_only",
                "added_bm25_rag_text",
                "added_schema_text_only",
                "added_visual_masked",
            }
            traceable_refs, visual_chunks = load_trace_context(
                context_dir_for_condition(paper_dir, condition),
                force_no_visual=no_actual_visual,
            )
            base_row: dict[str, Any] = {
                "paper_id": paper_dir.name,
                "condition": condition,
                "result_path": str(result_path) if result_path else "",
            }
            if result_path is None:
                row = {**base_row, "status": "missing"}
                rows.append(row)
                missing.append({"paper_id": paper_dir.name, "condition": condition, "reason": "result_json_not_found"})
                continue

            try:
                final_data = normalize_final_result(load_json(result_path))
                if final_data is None:
                    raise ValueError("No final extraction schema was found.")
                metrics = compute_metrics(final_data, traceable_refs, visual_chunks)
                rows.append({**base_row, "status": "ok", **metrics})
            except Exception as exc:
                row = {**base_row, "status": "error", "error": str(exc)}
                rows.append(row)
                missing.append({"paper_id": paper_dir.name, "condition": condition, "reason": str(exc)})

    summary = summarize(rows, conditions)

    by_paper_fields = ["paper_id", "condition", "status", "result_path", "error"] + CORE_METRICS + COUNT_FIELDS
    summary_fields = ["condition", "num_successful_papers"]
    for key in CORE_METRICS + COUNT_FIELDS:
        summary_fields.extend([f"{key}_mean", f"{key}_std", f"{key}_min", f"{key}_max"])

    write_csv(output_dir / "metrics_by_paper.csv", rows, by_paper_fields)
    write_csv(output_dir / "metrics_summary.csv", summary, summary_fields)
    core_mean_rows = build_core_mean_rows(summary)
    core_mean_std_rows = build_core_mean_std_rows(summary)
    core_mean_fields = ["condition", "num_successful_papers"] + [
        CORE_METRIC_LABELS[metric] for metric in CORE_METRICS
    ]
    core_mean_std_fields = ["condition", "num_successful_papers"]
    for metric in CORE_METRICS:
        label = CORE_METRIC_LABELS[metric]
        core_mean_std_fields.extend([f"{label}_mean", f"{label}_std"])
    write_csv(output_dir / "metrics_summary_core.csv", core_mean_rows, core_mean_fields)
    write_csv(output_dir / "metrics_summary_core_mean_std.csv", core_mean_std_rows, core_mean_std_fields)
    write_latex_core_table(output_dir / "metrics_summary_core_latex.tex", summary)
    write_latex_selected_table(
        output_dir / "table_comparison_core.tex",
        summary,
        ["baseline_oneshot_mineru", "proposed_full"],
        "Comparison with the one-shot multimodal baseline on 100 CV papers. Higher values indicate better performance for all metrics.",
        "tab:comparison_core_metrics",
    )
    write_latex_selected_table(
        output_dir / "table_ablation_core.tex",
        summary,
        ["proposed_full", "ablation_text_only", "ablation_no_l3", "ablation_large_chunk"],
        "Ablation study on 100 CV papers. Higher values indicate better performance for all metrics.",
        "tab:ablation_core_metrics",
    )
    write_json(output_dir / "metrics_by_paper.json", rows)
    write_json(output_dir / "metrics_summary.json", summary)
    write_json(output_dir / "missing_or_error_results.json", missing)

    print(f"Wrote per-paper metrics: {output_dir / 'metrics_by_paper.csv'}")
    print(f"Wrote aggregate summary: {output_dir / 'metrics_summary.csv'}")
    print(f"Wrote core metric summary: {output_dir / 'metrics_summary_core.csv'}")
    print(f"Wrote LaTeX table: {output_dir / 'metrics_summary_core_latex.tex'}")
    print(f"Wrote comparison table: {output_dir / 'table_comparison_core.tex'}")
    print(f"Wrote ablation table: {output_dir / 'table_ablation_core.tex'}")
    print(f"Wrote missing/error report: {output_dir / 'missing_or_error_results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
