from __future__ import annotations

import json
import math
import random
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .annotation_schema import PaperGoldAnnotation
from .evidence import build_exact_evidence_map, relation_evidence_support, resolve_refs, strict_visual_support
from .io_utils import canonical_ref, condition_result_path, evidence_refs, node_id, node_text, normalize_final, read_json


WORD_RE = re.compile(r"[A-Za-z0-9]+")


def tokens(text: str) -> set[str]:
    return {value.lower() for value in WORD_RE.findall(text) if len(value) > 1}


def lexical_similarity(a: str, b: str) -> float:
    left, right = tokens(a), tokens(b)
    if not left and not right:
        return 1.0
    return len(left & right) / max(len(left | right), 1)


def normalize_relation(value: Any) -> str:
    text = str(value or "").lower().strip()
    aliases = {
        "addresses": "directly_addresses",
        "mitigates": "partially_addresses",
        "supports": "partially_addresses",
        "implements": "directly_addresses",
    }
    return aliases.get(text, text)


@dataclass
class MatchResult:
    pairs: list[tuple[int, int, float]]
    unmatched_predictions: list[int]
    unmatched_gold: list[int]


def greedy_match(predictions: list[str], gold: list[str], threshold: float) -> MatchResult:
    candidates = []
    for pred_index, pred in enumerate(predictions):
        for gold_index, target in enumerate(gold):
            score = lexical_similarity(pred, target)
            if score >= threshold:
                candidates.append((score, pred_index, gold_index))
    candidates.sort(reverse=True)
    used_pred: set[int] = set()
    used_gold: set[int] = set()
    pairs = []
    for score, pred_index, gold_index in candidates:
        if pred_index in used_pred or gold_index in used_gold:
            continue
        used_pred.add(pred_index)
        used_gold.add(gold_index)
        pairs.append((pred_index, gold_index, score))
    return MatchResult(
        pairs=pairs,
        unmatched_predictions=[index for index in range(len(predictions)) if index not in used_pred],
        unmatched_gold=[index for index in range(len(gold)) if index not in used_gold],
    )


def prf(tp: int, fp: int, fn: int) -> dict[str, float | int]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def evaluate_paper(
    paper_dir: Path,
    condition: str,
    gold_path: Path,
    similarity_threshold: float = 0.45,
    verified_match_path: Path | None = None,
    *,
    matching_method: str = "lexical",
    semantic_matcher: Any | None = None,
    problem_similarity_threshold: float = 0.70,
    method_similarity_threshold: float = 0.70,
) -> dict[str, Any]:
    if gold_path.name != "human_gold.json":
        raise ValueError(f"Expected human_gold.json, got {gold_path}")
    gold = PaperGoldAnnotation.model_validate(read_json(gold_path))
    # The filename is retained for backward compatibility with the v4.x
    # scripts, but in this study it is a Qwen3.8-Max model-generated reference,
    # not a human annotation.  Never present it as human-verified unless a
    # separate verified protocol is actually supplied.
    metadata_path = gold_path.parent / "annotation_metadata.json"
    gold_status = "qwen38_max_generated_model_gold"
    if metadata_path.exists():
        try:
            metadata = read_json(metadata_path)
            backend = str(metadata.get("annotation_backend") or "").lower()
            declared_status = str(metadata.get("annotation_status") or "").strip()
            if declared_status:
                gold_status = declared_status
            elif backend and backend != "qwen3.8-max":
                gold_status = f"model_gold_backend_{backend}"
        except Exception:
            gold_status = "qwen38_max_generated_model_gold_metadata_unreadable"
    if gold.paper_id != paper_dir.name:
        raise ValueError(f"Gold paper_id={gold.paper_id!r} does not match folder {paper_dir.name!r}")
    result_path = condition_result_path(paper_dir, condition)
    prediction = normalize_final(read_json(result_path))
    condition_dir = result_path.parent
    l1_path = condition_dir / "01_l1_chunk_results.json"
    index_path = condition_dir / "02_evidence_index.json"
    if l1_path.exists() and index_path.exists():
        evidence_map = build_exact_evidence_map(read_json(l1_path), read_json(index_path))
    elif condition == "original_baseline_oneshot_mineru":
        # The one-shot baseline has direct prose evidence but no staged L1/index
        # chain. It must not borrow the proposed method's support files.
        evidence_map = {}
    else:
        raise FileNotFoundError(
            f"Condition {condition} is missing its own evidence support files: {l1_path}, {index_path}"
        )

    predicted_problems = prediction["problems"]
    predicted_methods = prediction["methods"]
    predicted_problem_texts = [node_text(item) for item in predicted_problems]
    gold_problem_texts = [item.claim for item in gold.research_problems]
    predicted_method_texts = [node_text(item) for item in predicted_methods]
    gold_method_texts = [item.claim for item in gold.methods]
    if matching_method == "semantic":
        if semantic_matcher is None:
            raise ValueError("semantic_matcher is required when matching_method='semantic'")
        problem_match = semantic_matcher.match(
            predicted_problem_texts,
            gold_problem_texts,
            problem_similarity_threshold,
        )
        method_match = semantic_matcher.match(
            predicted_method_texts,
            gold_method_texts,
            method_similarity_threshold,
        )
        matching_status = "semantic_embedding_cosine_hungarian"
    elif matching_method == "lexical":
        problem_match = greedy_match(predicted_problem_texts, gold_problem_texts, similarity_threshold)
        method_match = greedy_match(predicted_method_texts, gold_method_texts, similarity_threshold)
        matching_status = "automated_lexical_legacy"
    else:
        raise ValueError(f"Unknown matching method: {matching_method!r}")
    if verified_match_path is not None and verified_match_path.exists():
        match_data = read_json(verified_match_path)
        if match_data.get("status") != "verified_human" or not str(match_data.get("reviewer_name") or "").strip():
            raise ValueError(f"Match file is not verified by a named human reviewer: {verified_match_path}")
        def reviewed_match(items: list[dict[str, Any]], gold_items: list[Any], kind: str) -> MatchResult:
            pred_lookup = {node_id(item, kind, index + 1): index for index, item in enumerate(items)}
            gold_lookup = {item.id: index for index, item in enumerate(gold_items)}
            pairs = []
            seen_pred: set[int] = set()
            seen_gold: set[int] = set()
            for pair in match_data.get(f"{kind}_pairs", []):
                if len(pair) != 2 or pair[0] not in pred_lookup or pair[1] not in gold_lookup:
                    raise ValueError(f"Invalid reviewed {kind} pair {pair} in {verified_match_path}")
                pred_index = pred_lookup[pair[0]]
                gold_index = gold_lookup[pair[1]]
                if pred_index in seen_pred or gold_index in seen_gold:
                    raise ValueError(
                        f"Reviewed {kind} matches must be one-to-one; duplicate pair target {pair} in {verified_match_path}"
                    )
                seen_pred.add(pred_index)
                seen_gold.add(gold_index)
                pairs.append((pred_index, gold_index, 1.0))
            used_pred, used_gold = {p for p, _, _ in pairs}, {g for _, g, _ in pairs}
            return MatchResult(pairs, [i for i in range(len(items)) if i not in used_pred], [i for i in range(len(gold_items)) if i not in used_gold])
        problem_match = reviewed_match(predicted_problems, gold.research_problems, "problem")
        method_match = reviewed_match(predicted_methods, gold.methods, "method")
        matching_status = "verified_human"
    problem_scores = prf(len(problem_match.pairs), len(problem_match.unmatched_predictions), len(problem_match.unmatched_gold))
    method_scores = prf(len(method_match.pairs), len(method_match.unmatched_predictions), len(method_match.unmatched_gold))

    pred_problem_to_gold = {node_id(predicted_problems[p], "problem", p + 1): gold.research_problems[g].id for p, g, _ in problem_match.pairs}
    pred_method_to_gold = {node_id(predicted_methods[p], "method", p + 1): gold.methods[g].id for p, g, _ in method_match.pairs}
    gold_links = {(item.problem_id, item.method_id, item.relation) for item in gold.problem_method_links}
    matched_gold_problem_ids = set(pred_problem_to_gold.values())
    matched_gold_method_ids = set(pred_method_to_gold.values())

    def score_links(include_inferred: bool, *, conditional: bool = False) -> dict[str, float | int]:
        predicted_link_keys = []
        predicted_total = 0
        for link in prediction["links"]:
            if not include_inferred and str(link.get("link_type") or "evidence_supported") == "inferred":
                continue
            problem_id = pred_problem_to_gold.get(str(link.get("problem_id")))
            method_id = pred_method_to_gold.get(str(link.get("method_id")))
            if conditional and (not problem_id or not method_id):
                continue
            predicted_total += 1
            if problem_id and method_id:
                predicted_link_keys.append((problem_id, method_id, normalize_relation(link.get("relation"))))
        target_gold_links = gold_links
        if conditional:
            target_gold_links = {
                link for link in gold_links
                if link[0] in matched_gold_problem_ids and link[1] in matched_gold_method_ids
            }
        matched_links = set(predicted_link_keys) & target_gold_links
        # Links attached to unmatched/invalid nodes are false positives rather
        # than silently disappearing from the relation denominator.
        return prf(
            len(matched_links),
            predicted_total - len(matched_links),
            len(target_gold_links) - len(matched_links),
        )

    link_scores = score_links(include_inferred=False)
    all_link_scores = score_links(include_inferred=True)
    conditional_link_scores = score_links(include_inferred=False, conditional=True)
    conditional_all_link_scores = score_links(include_inferred=True, conditional=True)

    all_nodes = predicted_problems + predicted_methods
    exact_supported = 0
    strict_visual = 0
    for kind, items in (("problem", predicted_problems), ("method", predicted_methods)):
        for item in items:
            refs = evidence_refs(item)
            resolved, invalid = resolve_refs(refs, evidence_map, expected_kind=kind)
            if refs and not invalid and len(resolved) == len(refs):
                exact_supported += 1
            if strict_visual_support(item, evidence_map, expected_kind=kind)["has_strict_visual_support"]:
                strict_visual += 1
    structural = {
        "exact_node_reference_rate": exact_supported / len(all_nodes) if all_nodes else 0.0,
        "strict_visual_evidence_rate": strict_visual / len(all_nodes) if all_nodes else 0.0,
        "evidence_supported_link_rate": sum(str(link.get("link_type") or "") == "evidence_supported" for link in prediction["links"]) / len(prediction["links"]) if prediction["links"] else 0.0,
        "strict_relation_evidence_rate": sum(relation_evidence_support(link, evidence_map)["supported"] for link in prediction["links"]) / len(prediction["links"]) if prediction["links"] else 0.0,
        "inferred_link_count": sum(str(link.get("link_type") or "") == "inferred" for link in prediction["links"]),
        "unverified_link_count": sum(str(link.get("link_type") or "") == "unverified" for link in prediction["links"]),
    }
    visual_gold_ids = {
        item.id for item in [*gold.research_problems, *gold.methods] if item.visual_dependency in {"visual_dependent", "mixed"}
    }
    visual_tp = sum(1 for pred_index, gold_index, _ in problem_match.pairs if gold.research_problems[gold_index].id in visual_gold_ids and strict_visual_support(predicted_problems[pred_index], evidence_map, expected_kind="problem")["has_strict_visual_support"])
    visual_tp += sum(1 for pred_index, gold_index, _ in method_match.pairs if gold.methods[gold_index].id in visual_gold_ids and strict_visual_support(predicted_methods[pred_index], evidence_map, expected_kind="method")["has_strict_visual_support"])
    visual_pred = sum(1 for item in predicted_problems if strict_visual_support(item, evidence_map, expected_kind="problem")["has_strict_visual_support"])
    visual_pred += sum(1 for item in predicted_methods if strict_visual_support(item, evidence_map, expected_kind="method")["has_strict_visual_support"])
    visual_scores = prf(visual_tp, max(visual_pred - visual_tp, 0), max(len(visual_gold_ids) - visual_tp, 0))

    problem_correct = {pred_index for pred_index, _, _ in problem_match.pairs}
    method_correct = {pred_index for pred_index, _, _ in method_match.pairs}
    calibration_records = []
    for kind, items, correct_indices in (
        ("problem", predicted_problems, problem_correct),
        ("method", predicted_methods, method_correct),
    ):
        for index, item in enumerate(items):
            refs = evidence_refs(item)
            canonical_refs = [canonical_ref(ref) for ref in refs]
            sections = sorted({evidence_map[ref].section for ref in canonical_refs if ref in evidence_map and evidence_map[ref].section})
            try:
                confidence = min(1.0, max(0.0, float(item.get("confidence", 0.0))))
            except (TypeError, ValueError):
                confidence = 0.0
            calibration_records.append({
                "paper_id": paper_dir.name,
                "condition": condition,
                "kind": kind,
                "node_id": node_id(item, kind, index + 1),
                "section": "; ".join(sections) if sections else "unknown",
                "confidence": confidence,
                "correct": 1 if index in correct_indices else 0,
            })

    return {
        "paper_id": paper_dir.name,
        "condition": condition,
        "result_path": str(result_path),
        "gold_path": str(gold_path),
        "gold_status": gold_status,
        "problem": problem_scores,
        "method": method_scores,
        "link": link_scores,
        "link_including_inferred": all_link_scores,
        "link_conditional_on_matched_nodes": conditional_link_scores,
        "link_including_inferred_conditional_on_matched_nodes": conditional_all_link_scores,
        "visual_dependent": visual_scores,
        "node_counts": {
            "predicted_problems": len(predicted_problems),
            "gold_problems": len(gold.research_problems),
            "matched_problems": len(problem_match.pairs),
            "predicted_to_gold_problem_ratio": (
                len(predicted_problems) / len(gold.research_problems)
                if gold.research_problems else 0.0
            ),
            "predicted_methods": len(predicted_methods),
            "gold_methods": len(gold.methods),
            "matched_methods": len(method_match.pairs),
            "predicted_to_gold_method_ratio": (
                len(predicted_methods) / len(gold.methods)
                if gold.methods else 0.0
            ),
        },
        "structural_diagnostics": structural,
        "matching": {
            "status": matching_status,
            "legacy_lexical_threshold": similarity_threshold if matching_method == "lexical" else None,
            "problem_similarity_threshold": (
                problem_similarity_threshold if matching_method == "semantic" else None
            ),
            "method_similarity_threshold": (
                method_similarity_threshold if matching_method == "semantic" else None
            ),
            "problem_pairs": problem_match.pairs,
            "method_pairs": method_match.pairs,
            "warning": (
                "Dense semantic similarity with explicit unmatched dummies and global one-to-one "
                "Hungarian assignment is the primary automatic matcher; optional verified match "
                "files override it."
                if matching_method == "semantic"
                else "Legacy token-set Jaccard matching; do not use as the revised primary result."
            ),
        },
        "calibration_records": calibration_records,
    }


def micro_aggregate(rows: list[dict[str, Any]], metric: str) -> dict[str, float | int]:
    return prf(
        sum(int(row[metric]["tp"]) for row in rows),
        sum(int(row[metric]["fp"]) for row in rows),
        sum(int(row[metric]["fn"]) for row in rows),
    )


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def bootstrap_ci(values: list[float], seed: int = 42, replicates: int = 5000) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    rng = random.Random(seed)
    samples = []
    for _ in range(replicates):
        samples.append(mean([values[rng.randrange(len(values))] for _ in values]))
    samples.sort()
    return samples[int(0.025 * (len(samples) - 1))], samples[int(0.975 * (len(samples) - 1))]


def paired_permutation(values_a: list[float], values_b: list[float], seed: int = 42, replicates: int = 20000) -> float:
    differences = [a - b for a, b in zip(values_a, values_b)]
    if not differences:
        return 1.0
    observed = abs(mean(differences))
    rng = random.Random(seed)
    exceed = 0
    for _ in range(replicates):
        statistic = abs(mean([value if rng.random() < 0.5 else -value for value in differences]))
        if statistic >= observed:
            exceed += 1
    return (exceed + 1) / (replicates + 1)


def cohens_kappa(labels_a: list[str], labels_b: list[str]) -> float:
    if len(labels_a) != len(labels_b) or not labels_a:
        return 0.0
    observed = sum(a == b for a, b in zip(labels_a, labels_b)) / len(labels_a)
    counts_a, counts_b = Counter(labels_a), Counter(labels_b)
    labels = set(counts_a) | set(counts_b)
    expected = sum((counts_a[label] / len(labels_a)) * (counts_b[label] / len(labels_b)) for label in labels)
    return (observed - expected) / (1 - expected) if expected < 1 else 1.0
