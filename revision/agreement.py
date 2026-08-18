from __future__ import annotations

from typing import Any

from .annotation_schema import PaperGoldAnnotation
from .evaluation import cohens_kappa, greedy_match


def annotation_agreement(left: PaperGoldAnnotation, right: PaperGoldAnnotation, threshold: float = 0.45) -> dict[str, Any]:
    if left.paper_id != right.paper_id:
        raise ValueError("Annotator files have different paper_id values")

    def node_agreement(left_items: list[Any], right_items: list[Any], type_attr: str) -> tuple[dict[str, Any], dict[str, str], dict[str, str]]:
        match = greedy_match([item.claim for item in left_items], [item.claim for item in right_items], threshold)
        denominator = len(left_items) + len(right_items)
        boundary_f1 = 2 * len(match.pairs) / denominator if denominator else 1.0
        labels_left = [str(getattr(left_items[i], type_attr)) for i, _, _ in match.pairs]
        labels_right = [str(getattr(right_items[j], type_attr)) for _, j, _ in match.pairs]
        mapping_left = {left_items[i].id: right_items[j].id for i, j, _ in match.pairs}
        mapping_right = {right_items[j].id: left_items[i].id for i, j, _ in match.pairs}
        return ({
            "annotator_1_count": len(left_items),
            "annotator_2_count": len(right_items),
            "matched_count": len(match.pairs),
            "node_boundary_f1": boundary_f1,
            "type_kappa_on_matched_nodes": cohens_kappa(labels_left, labels_right) if labels_left else 0.0,
            "pairs": match.pairs,
        }, mapping_left, mapping_right)

    problem, problem_1_to_2, _ = node_agreement(left.research_problems, right.research_problems, "problem_type")
    method, method_1_to_2, _ = node_agreement(left.methods, right.methods, "method_type")
    relation_1 = {(problem_1_to_2.get(item.problem_id), method_1_to_2.get(item.method_id), item.relation) for item in left.problem_method_links}
    relation_1 = {item for item in relation_1 if item[0] and item[1]}
    relation_2 = {(item.problem_id, item.method_id, item.relation) for item in right.problem_method_links}
    overlap = relation_1 & relation_2
    relation_f1 = 2 * len(overlap) / (len(relation_1) + len(relation_2)) if relation_1 or relation_2 else 1.0

    support_labels_1, support_labels_2 = [], []
    for left_index, right_index, _ in problem["pairs"]:
        support_labels_1.append(left.research_problems[left_index].visual_dependency)
        support_labels_2.append(right.research_problems[right_index].visual_dependency)
    for left_index, right_index, _ in method["pairs"]:
        support_labels_1.append(left.methods[left_index].visual_dependency)
        support_labels_2.append(right.methods[right_index].visual_dependency)
    return {
        "paper_id": left.paper_id,
        "similarity_threshold": threshold,
        "problem_nodes": problem,
        "method_nodes": method,
        "relation_f1_on_matched_endpoints": relation_f1,
        "matched_relation_count": len(overlap),
        "visual_dependency_kappa_on_matched_nodes": cohens_kappa(support_labels_1, support_labels_2) if support_labels_1 else 0.0,
    }
