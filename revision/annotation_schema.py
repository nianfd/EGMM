from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError


ProblemType = Literal["task_gap", "data_gap", "method_gap", "evaluation_gap", "application_constraint", "other"]
MethodType = Literal["architecture", "algorithm_step", "representation", "training_objective", "data_processing", "evaluation_protocol", "implementation_detail", "other"]
RelationType = Literal["directly_addresses", "partially_addresses", "evaluates", "motivates"]
VisualDependency = Literal["text_sufficient", "visual_dependent", "mixed"]


class EvidenceAnchor(BaseModel):
    span_ids: list[str] = Field(default_factory=list)
    quote: str
    visual_asset_ids: list[str] = Field(default_factory=list)
    visual_cue: str = ""
    support_type: Literal["text", "visual", "mixed"]


class GoldProblem(BaseModel):
    id: str
    claim: str
    problem_type: ProblemType
    evidence: list[EvidenceAnchor]
    visual_dependency: VisualDependency


class ReproducibilityFields(BaseModel):
    inputs: list[str] = Field(default_factory=list)
    outputs: list[str] = Field(default_factory=list)
    procedure: list[str] = Field(default_factory=list)
    objective_or_metric: list[str] = Field(default_factory=list)


class GoldMethod(BaseModel):
    id: str
    claim: str
    method_type: MethodType
    reproducibility_fields: ReproducibilityFields
    evidence: list[EvidenceAnchor]
    visual_dependency: VisualDependency


class GoldLink(BaseModel):
    problem_id: str
    method_id: str
    relation: RelationType
    evidence: list[EvidenceAnchor]
    visual_dependency: VisualDependency


class PaperGoldAnnotation(BaseModel):
    paper_id: str
    research_problems: list[GoldProblem]
    methods: list[GoldMethod]
    problem_method_links: list[GoldLink]
    difficult_or_ambiguous_cases: list[str] = Field(default_factory=list)


def normalize_incomplete_gold_links(payload: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Conservatively remove only schema-invalid relation entries.

    No missing relation label, evidence, or visual dependency is inferred.
    Endpoint punctuation is normalized only when it yields an exact node ID
    already present in this same payload. Every action is returned for audit.
    """
    copied = dict(payload)
    raw_links = copied.get("problem_method_links")
    audit: dict[str, Any] = {
        "policy": "drop schema-invalid links; never infer missing relation/evidence/visual_dependency",
        "input_link_count": len(raw_links) if isinstance(raw_links, list) else None,
        "output_link_count": None,
        "endpoint_normalizations": [],
        "dropped_links": [],
        "action_count": 0,
    }
    if not isinstance(raw_links, list):
        return copied, audit

    problem_ids = {
        str(item.get("id")).strip()
        for item in copied.get("research_problems", [])
        if isinstance(item, dict) and item.get("id") is not None
    }
    method_ids = {
        str(item.get("id")).strip()
        for item in copied.get("methods", [])
        if isinstance(item, dict) and item.get("id") is not None
    }
    retained: list[dict[str, Any]] = []
    required = {"problem_id", "method_id", "relation", "evidence", "visual_dependency"}
    for index, raw in enumerate(raw_links):
        if not isinstance(raw, dict):
            audit["dropped_links"].append({
                "index": index,
                "reason": "link is not a JSON object",
            })
            continue
        link = dict(raw)
        missing = sorted(key for key in required if key not in link)
        if missing:
            audit["dropped_links"].append({
                "index": index,
                "problem_id": link.get("problem_id"),
                "method_id": link.get("method_id"),
                "reason": "missing required fields",
                "missing_fields": missing,
            })
            continue

        for field, valid_ids in (("problem_id", problem_ids), ("method_id", method_ids)):
            raw_id = str(link.get(field) or "").strip()
            cleaned = raw_id.rstrip(" ,;:")
            if raw_id not in valid_ids and cleaned in valid_ids:
                link[field] = cleaned
                audit["endpoint_normalizations"].append({
                    "index": index,
                    "field": field,
                    "original": raw_id,
                    "normalized": cleaned,
                    "method": "strip_trailing_punctuation_to_exact_existing_id",
                })
        if str(link.get("problem_id") or "") not in problem_ids or str(link.get("method_id") or "") not in method_ids:
            audit["dropped_links"].append({
                "index": index,
                "problem_id": link.get("problem_id"),
                "method_id": link.get("method_id"),
                "reason": "endpoint does not resolve to an existing node ID",
            })
            continue
        try:
            validated = GoldLink.model_validate(link)
        except ValidationError as exc:
            audit["dropped_links"].append({
                "index": index,
                "problem_id": link.get("problem_id"),
                "method_id": link.get("method_id"),
                "reason": "link failed GoldLink schema validation",
                "validation_errors": [
                    {
                        "location": ".".join(str(value) for value in item.get("loc", [])),
                        "message": item.get("msg"),
                        "type": item.get("type"),
                    }
                    for item in exc.errors()
                ],
            })
            continue
        retained.append(validated.model_dump(mode="json"))

    copied["problem_method_links"] = retained
    audit["output_link_count"] = len(retained)
    audit["action_count"] = len(audit["endpoint_normalizations"]) + len(audit["dropped_links"])
    return copied, audit


def empty_annotation(paper_id: str) -> PaperGoldAnnotation:
    return PaperGoldAnnotation(
        paper_id=paper_id,
        research_problems=[],
        methods=[],
        problem_method_links=[],
        difficult_or_ambiguous_cases=["Dry run: no API annotation was generated."],
    )
