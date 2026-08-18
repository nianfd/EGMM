from __future__ import annotations

import math
from collections import defaultdict
from typing import Any


def sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-min(value, 50.0))
        return 1.0 / (1.0 + z)
    z = math.exp(max(value, -50.0))
    return z / (1.0 + z)


def fit_platt(rows: list[dict[str, Any]], steps: int = 4000, learning_rate: float = 0.02, l2: float = 0.001) -> tuple[float, float]:
    """Fit P(correct)=sigmoid(a*logit(confidence)+b) without external ML packages."""
    if not rows:
        return 1.0, 0.0
    a, b = 1.0, 0.0
    features = []
    for row in rows:
        confidence = min(1 - 1e-6, max(1e-6, float(row["confidence"])))
        features.append((math.log(confidence / (1 - confidence)), int(row["correct"])))
    for _ in range(steps):
        grad_a = grad_b = 0.0
        for value, target in features:
            error = sigmoid(a * value + b) - target
            grad_a += error * value
            grad_b += error
        scale = 1.0 / len(features)
        a -= learning_rate * (grad_a * scale + l2 * a)
        b -= learning_rate * grad_b * scale
    return a, b


def apply_platt(confidence: float, parameters: tuple[float, float]) -> float:
    confidence = min(1 - 1e-6, max(1e-6, confidence))
    value = math.log(confidence / (1 - confidence))
    return sigmoid(parameters[0] * value + parameters[1])


def calibration_metrics(rows: list[dict[str, Any]], probability_key: str, bins: int = 10) -> dict[str, Any]:
    if not rows:
        return {"count": 0, "brier": 0.0, "ece": 0.0, "reliability_bins": []}
    brier = sum((float(row[probability_key]) - int(row["correct"])) ** 2 for row in rows) / len(rows)
    bucketed: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        probability = min(1.0, max(0.0, float(row[probability_key])))
        bucketed[min(bins - 1, int(probability * bins))].append(row)
    details, ece = [], 0.0
    for index in range(bins):
        items = bucketed.get(index, [])
        if not items:
            continue
        mean_confidence = sum(float(item[probability_key]) for item in items) / len(items)
        accuracy = sum(int(item["correct"]) for item in items) / len(items)
        ece += len(items) / len(rows) * abs(mean_confidence - accuracy)
        details.append({
            "bin": index,
            "lower": index / bins,
            "upper": (index + 1) / bins,
            "count": len(items),
            "mean_confidence": mean_confidence,
            "empirical_accuracy": accuracy,
        })
    return {"count": len(rows), "brier": brier, "ece": ece, "reliability_bins": details}


def grouped_cross_validated_platt(rows: list[dict[str, Any]], folds: int = 5) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    papers = sorted({str(row["paper_id"]) for row in rows})
    if len(papers) < 2:
        raise ValueError("Calibration needs at least two papers")
    folds = min(max(2, folds), len(papers))
    assignments = {paper: index % folds for index, paper in enumerate(papers)}
    output = []
    fold_parameters = []
    for fold in range(folds):
        train = [row for row in rows if assignments[str(row["paper_id"])] != fold]
        test = [row for row in rows if assignments[str(row["paper_id"])] == fold]
        parameters = fit_platt(train)
        fold_parameters.append({"fold": fold, "a": parameters[0], "b": parameters[1], "train_count": len(train), "test_count": len(test)})
        for row in test:
            copied = dict(row)
            copied["calibrated_confidence"] = apply_platt(float(row["confidence"]), parameters)
            copied["fold"] = fold
            output.append(copied)
    final_parameters = fit_platt(rows)
    return output, {
        "fold_count": folds,
        "grouping": "paper_id",
        "fold_parameters": fold_parameters,
        "deployment_parameters_all_labeled_papers": {"a": final_parameters[0], "b": final_parameters[1]},
    }
