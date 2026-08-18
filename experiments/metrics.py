from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from common import read_json, write_json


def normalize_result(data: dict[str, Any]) -> dict[str, Any]:
    if "final_research_problems" in data or "final_methods" in data:
        return {
            "research_problems": data.get("final_research_problems", []),
            "methods": data.get("final_methods", []),
            "links": data.get("problem_method_links", []),
            "quality_report": data.get("quality_report", {}),
        }
    return {
        "research_problems": data.get("paper_research_problems", []),
        "methods": data.get("paper_methods", []),
        "links": data.get("problem_method_links", []),
        "quality_report": {},
    }


def load_optional_json(path: Path) -> Any | None:
    if path.exists():
        return read_json(path)
    return None


def load_support_files(result_path: Path) -> dict[str, Any]:
    directory = result_path.parent
    l1 = load_optional_json(directory / "01_l1_chunk_results.json")
    evidence_index = load_optional_json(directory / "02_evidence_index.json")
    return {
        "l1_results": l1,
        "evidence_index": evidence_index,
        "atom_index": build_atom_index(l1),
    }


def build_atom_index(l1_results: Any | None) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    if not isinstance(l1_results, list):
        return index
    for chunk in l1_results:
        if not isinstance(chunk, dict):
            continue
        chunk_id = str(chunk.get("chunk_id") or "")
        if not chunk_id:
            continue
        for atom in chunk.get("research_problem_atoms", []):
            if isinstance(atom, dict) and atom.get("id"):
                index[f"{chunk_id}:{atom.get('id')}"] = {
                    **atom,
                    "chunk_id": chunk_id,
                    "atom_kind": "research_problem",
                    "section": chunk.get("section"),
                }
        for atom in chunk.get("method_atoms", []):
            if isinstance(atom, dict) and atom.get("id"):
                index[f"{chunk_id}:{atom.get('id')}"] = {
                    **atom,
                    "chunk_id": chunk_id,
                    "atom_kind": "method",
                    "section": chunk.get("section"),
                }
    return index


def evidence_refs(item: dict[str, Any]) -> list[Any]:
    refs = item.get("evidence_refs", [])
    if isinstance(refs, list):
        return refs
    return []


def node_is_traceable(item: dict[str, Any], support: dict[str, Any]) -> bool:
    return bool(resolve_indexed_evidence(evidence_refs(item), support))


def node_has_actual_multimodal_grounding(item: dict[str, Any], support: dict[str, Any]) -> bool:
    return has_actual_visual_context(evidence_refs(item), support)


def has_visual_evidence(refs: list[Any], support: dict[str, Any] | None = None, relaxed: bool = False) -> bool:
    resolved = resolve_evidence(refs, support or {})
    if resolved:
        return any(is_visual_resolved_evidence(item, relaxed=relaxed) for item in resolved)
    return any(is_visual_text(json.dumps(ref, ensure_ascii=False).lower()) for ref in refs)


def is_visual_text(text: str) -> bool:
    visual_terms = ("image", "figure", "fig", "table", "visual", "equation", "mixed")
    return any(term in text for term in visual_terms)


def resolve_evidence(refs: list[Any], support: dict[str, Any]) -> list[dict[str, Any]]:
    atom_index = support.get("atom_index") or {}
    evidence_index = support.get("evidence_index") or {}
    resolved: list[dict[str, Any]] = []
    for ref in refs:
        if isinstance(ref, dict):
            resolved.append({"kind": "direct", "raw": ref})
            continue
        if not isinstance(ref, str):
            continue
        chunk_id = ref.split(":", 1)[0]
        atom = atom_index.get(ref)
        chunk = evidence_index.get(chunk_id) if isinstance(evidence_index, dict) else None
        if atom or chunk:
            resolved.append(
                {
                    "kind": "indexed",
                    "ref": ref,
                    "atom": atom,
                    "chunk": chunk,
                    "atom_evidence": atom.get("evidence", []) if isinstance(atom, dict) else [],
                }
            )
    return resolved


def resolve_indexed_evidence(refs: list[Any], support: dict[str, Any]) -> list[dict[str, Any]]:
    """Resolve only evidence references that map to our chunk/L1 evidence chain.

    Direct evidence dictionaries produced by a one-shot baseline are evidence text,
    but they are not traceable to the staged MinerU chunk/L1 atom index.
    """
    atom_index = support.get("atom_index") or {}
    evidence_index = support.get("evidence_index") or {}
    resolved: list[dict[str, Any]] = []
    for ref in refs:
        if not isinstance(ref, str):
            continue
        chunk_id = ref.split(":", 1)[0]
        atom = atom_index.get(ref)
        chunk = evidence_index.get(chunk_id) if isinstance(evidence_index, dict) else None
        if atom or chunk:
            resolved.append(
                {
                    "kind": "indexed",
                    "ref": ref,
                    "atom": atom,
                    "chunk": chunk,
                    "atom_evidence": atom.get("evidence", []) if isinstance(atom, dict) else [],
                }
            )
    return resolved


def is_visual_resolved_evidence(item: dict[str, Any], relaxed: bool = False) -> bool:
    if item.get("kind") == "direct":
        return is_visual_text(json.dumps(item.get("raw"), ensure_ascii=False).lower())
    atom_evidence = item.get("atom_evidence") or []
    for evidence in atom_evidence:
        text = json.dumps(evidence, ensure_ascii=False).lower()
        source = str(evidence.get("source", "")).lower() if isinstance(evidence, dict) else ""
        if source and source != "text":
            return True
        if is_visual_text(text):
            return True
    if relaxed:
        chunk = item.get("chunk")
        if isinstance(chunk, dict) and chunk.get("images"):
            return True
    return False


def evidence_resolution_stats(all_items: list[dict[str, Any]], support: dict[str, Any]) -> tuple[int, int]:
    total_refs = 0
    resolved_refs = 0
    for item in all_items:
        refs = evidence_refs(item)
        total_refs += len(refs)
        resolved_refs += len(resolve_indexed_evidence(refs, support))
    return resolved_refs, total_refs


def indexed_evidence_stats(all_items: list[dict[str, Any]], support: dict[str, Any]) -> tuple[int, int]:
    total_refs = 0
    indexed_refs = 0
    atom_index = support.get("atom_index") or {}
    evidence_index = support.get("evidence_index") or {}
    for item in all_items:
        for ref in evidence_refs(item):
            total_refs += 1
            if not isinstance(ref, str):
                continue
            chunk_id = ref.split(":", 1)[0]
            if ref in atom_index or (isinstance(evidence_index, dict) and chunk_id in evidence_index):
                indexed_refs += 1
    return indexed_refs, total_refs


def has_actual_visual_context(refs: list[Any], support: dict[str, Any]) -> bool:
    resolved = resolve_evidence(refs, support)
    for item in resolved:
        chunk = item.get("chunk")
        if isinstance(chunk, dict) and chunk.get("images"):
            return True
    return False


def item_text(item: dict[str, Any]) -> str:
    return str(item.get("problem") or item.get("method") or "")


def auto_granularity(item: dict[str, Any]) -> str:
    text = item_text(item).strip()
    if not text:
        return "coarse"
    lower = text.lower()
    cjk_chars = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    tokens = [tok for tok in lower.replace("-", " ").replace("_", " ").split() if tok]
    fine_cues = [
        "loss",
        "module",
        "step",
        "input",
        "output",
        "training",
        "inference",
        "feature",
        "representation",
        "evaluation",
        "metric",
        "objective",
        "模块",
        "步骤",
        "输入",
        "输出",
        "训练",
        "推理",
        "特征",
        "表示",
        "损失",
        "目标",
        "评估",
        "指标",
        "协议",
        "算法",
        "矩阵",
        "匹配",
    ]
    if cjk_chars:
        if cjk_chars < 12:
            return "coarse"
        if cjk_chars <= 80 or any(cue in lower for cue in fine_cues):
            return "fine"
        return "medium"
    if len(tokens) <= 8:
        return "coarse"
    if len(tokens) <= 35 or any(cue in lower for cue in fine_cues):
        return "fine"
    return "medium"


def confidence(item: dict[str, Any]) -> float:
    value = item.get("confidence")
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def method_reproducibility_score(method: dict[str, Any]) -> float:
    fields = method.get("reproducibility_fields")
    if isinstance(fields, dict):
        keys = ["inputs", "outputs", "procedure", "objective_or_metric"]
        present = 0
        for key in keys:
            value = fields.get(key)
            if isinstance(value, list) and any(str(x).strip() for x in value):
                present += 1
            elif isinstance(value, str) and value.strip():
                present += 1
        return present / len(keys)
    present = 0
    for key in ["inputs", "outputs"]:
        value = method.get(key)
        if isinstance(value, list) and any(str(x).strip() for x in value):
            present += 1
    return present / 4


def required_schema_valid(data: dict[str, Any]) -> bool:
    if "final_research_problems" in data or "final_methods" in data:
        return (
            isinstance(data.get("final_research_problems"), list)
            and isinstance(data.get("final_methods"), list)
            and isinstance(data.get("problem_method_links"), list)
        )
    return (
        isinstance(data.get("paper_research_problems"), list)
        and isinstance(data.get("paper_methods"), list)
        and isinstance(data.get("problem_method_links"), list)
    )


def compute_metrics(data: dict[str, Any], support: dict[str, Any] | None = None) -> dict[str, Any]:
    support = support or {}
    normalized = normalize_result(data)
    problems = [x for x in normalized["research_problems"] if isinstance(x, dict)]
    methods = [x for x in normalized["methods"] if isinstance(x, dict)]
    links = [x for x in normalized["links"] if isinstance(x, dict)]
    all_items = problems + methods
    evidence_counts = [len(evidence_refs(item)) for item in all_items]
    visual_count = sum(1 for item in all_items if has_visual_evidence(evidence_refs(item), support, relaxed=False))
    relaxed_visual_count = sum(1 for item in all_items if has_visual_evidence(evidence_refs(item), support, relaxed=True))
    actual_visual_context_count = sum(1 for item in all_items if has_actual_visual_context(evidence_refs(item), support))
    fine_count = sum(1 for item in all_items if item.get("granularity") == "fine")
    auto_fine_count = sum(1 for item in all_items if auto_granularity(item) == "fine")
    confidence_values = [confidence(item) for item in all_items]
    reproducibility = [method_reproducibility_score(item) for item in methods]
    problem_ids = {item.get("id") for item in problems}
    method_ids = {item.get("id") for item in methods}
    valid_links = [
        link
        for link in links
        if link.get("problem_id") in problem_ids and link.get("method_id") in method_ids
    ]
    linked_problem_ids = {link.get("problem_id") for link in valid_links}
    linked_method_ids = {link.get("method_id") for link in valid_links}
    linked_node_count = len(linked_problem_ids) + len(linked_method_ids)
    evidence_supported_links = [link for link in valid_links if link.get("link_type", "evidence_supported") == "evidence_supported"]
    inferred_links = [link for link in valid_links if link.get("link_type") == "inferred"]
    resolved_refs, total_refs = evidence_resolution_stats(all_items, support)
    indexed_refs, indexed_total_refs = indexed_evidence_stats(all_items, support)

    traceable_node_ids = {
        item.get("id")
        for item in all_items
        if item.get("id") and node_is_traceable(item, support)
    }
    net = len(traceable_node_ids) / len(all_items) if all_items else 0.0
    amg = (
        sum(1 for item in all_items if node_has_actual_multimodal_grounding(item, support)) / len(all_items)
        if all_items
        else 0.0
    )
    evidence_supported_traceable_links = [
        link
        for link in valid_links
        if link.get("link_type", "evidence_supported") == "evidence_supported"
        and link.get("problem_id") in traceable_node_ids
        and link.get("method_id") in traceable_node_ids
    ]
    es_nodes = set()
    for link in evidence_supported_traceable_links:
        es_nodes.add(link.get("problem_id"))
        es_nodes.add(link.get("method_id"))
    esgc = len(es_nodes) / len(all_items) if all_items else 0.0
    tlp = len(evidence_supported_traceable_links) / len(valid_links) if valid_links else 0.0
    egmr_values = [
        method_reproducibility_score(item) if item.get("id") in traceable_node_ids else 0.0
        for item in methods
    ]
    egmr = sum(egmr_values) / len(egmr_values) if egmr_values else 0.0
    begq = (
        0.15 * net
        + 0.15 * amg
        + 0.25 * esgc
        + 0.25 * tlp
        + 0.20 * egmr
    )
    egr = esgc * tlp
    rmr = tlp * egmr
    return {
        "BEGQ_balanced_evidence_grounded_quality": begq,
        "NET_node_evidence_traceability": net,
        "AMG_actual_multimodal_grounding": amg,
        "ESGC_evidence_supported_graph_connectivity": esgc,
        "TLP_traceable_link_purity": tlp,
        "EGMR_evidence_grounded_method_reproducibility": egmr,
        "EGR_evidence_graph_reliability": egr,
        "RMR_reliable_method_reproducibility": rmr,
        "num_research_problems": len(problems),
        "num_methods": len(methods),
        "num_links": len(links),
        "valid_link_ratio": len(valid_links) / len(links) if links else 0.0,
        "evidence_supported_link_ratio": len(evidence_supported_links) / len(valid_links) if valid_links else 0.0,
        "inferred_link_ratio": len(inferred_links) / len(valid_links) if valid_links else 0.0,
        "linked_node_ratio": linked_node_count / len(all_items) if all_items else 0.0,
        "isolated_node_ratio": 1.0 - (linked_node_count / len(all_items)) if all_items else 0.0,
        "items_with_evidence_ratio": sum(1 for n in evidence_counts if n > 0) / len(all_items) if all_items else 0.0,
        "avg_evidence_refs_per_item": sum(evidence_counts) / len(evidence_counts) if evidence_counts else 0.0,
        "resolved_evidence_ref_ratio": resolved_refs / total_refs if total_refs else 0.0,
        "indexed_evidence_ref_ratio": indexed_refs / indexed_total_refs if indexed_total_refs else 0.0,
        "visual_evidence_ratio_strict": visual_count / len(all_items) if all_items else 0.0,
        "visual_evidence_ratio_relaxed": relaxed_visual_count / len(all_items) if all_items else 0.0,
        "actual_visual_context_ratio": actual_visual_context_count / len(all_items) if all_items else 0.0,
        "fine_granularity_ratio": fine_count / len(all_items) if all_items else 0.0,
        "auto_fine_granularity_ratio": auto_fine_count / len(all_items) if all_items else 0.0,
        "avg_confidence": sum(confidence_values) / len(confidence_values) if confidence_values else 0.0,
        "avg_method_reproducibility_score": sum(reproducibility) / len(reproducibility) if reproducibility else 0.0,
    }


def find_result_files(root: Path) -> list[Path]:
    names = {
        "04_final_extraction.json",
        "baseline_oneshot_mineru.json",
        "ablation_no_l3_result.json",
    }
    return sorted(path for path in root.rglob("*.json") if path.name in names)


def main() -> int:
    parser = argparse.ArgumentParser(description="Compute automatic comparison metrics.")
    parser.add_argument("--result-root", default="experiments/results/paper1")
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--output-csv", default=None)
    args = parser.parse_args()
    root = Path(args.result_root).resolve()
    rows = []
    for path in find_result_files(root):
        condition = path.parent.name
        support = load_support_files(path)
        metrics = compute_metrics(read_json(path), support=support)
        rows.append({"condition": condition, "file": str(path), **metrics})
    output_json = Path(args.output_json).resolve() if args.output_json else root / "metrics_summary.json"
    output_csv = Path(args.output_csv).resolve() if args.output_csv else root / "metrics_summary.csv"
    write_json(output_json, rows)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        if rows:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps({"json": str(output_json), "csv": str(output_csv), "rows": rows}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
