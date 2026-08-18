from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .io_utils import canonical_ref, evidence_refs, read_json


@dataclass(frozen=True)
class EvidenceRecord:
    ref: str
    chunk_id: str
    atom_id: str
    atom_kind: str
    section: str
    claim: str
    evidence: list[dict[str, Any]]
    chunk_images: tuple[str, ...]

    @property
    def image_evidence(self) -> list[dict[str, Any]]:
        return [item for item in self.evidence if str(item.get("source", "")).lower() == "image"]

    @property
    def visual_cues(self) -> list[str]:
        return [
            str(item.get("quote_or_visual_cue") or "").strip()
            for item in self.image_evidence
            if str(item.get("quote_or_visual_cue") or "").strip()
        ]


def build_exact_evidence_map(l1_results: list[dict[str, Any]], evidence_index: dict[str, Any]) -> dict[str, EvidenceRecord]:
    result: dict[str, EvidenceRecord] = {}
    for chunk in l1_results:
        if not isinstance(chunk, dict):
            continue
        chunk_id = str(chunk.get("chunk_id") or "").strip()
        if not chunk_id:
            continue
        chunk_meta = evidence_index.get(chunk_id, {}) if isinstance(evidence_index, dict) else {}
        images = tuple(str(value) for value in chunk_meta.get("images", []) if value)
        section = str(chunk.get("section") or chunk_meta.get("section") or "")
        for key, kind in (("research_problem_atoms", "problem"), ("method_atoms", "method")):
            for atom in chunk.get(key, []):
                if not isinstance(atom, dict) or not atom.get("id"):
                    continue
                ref = canonical_ref(f"{chunk_id}:{atom['id']}")
                if ref is None:
                    continue
                evidence = [item for item in atom.get("evidence", []) if isinstance(item, dict)]
                result[ref] = EvidenceRecord(
                    ref=ref,
                    chunk_id=chunk_id,
                    atom_id=str(atom["id"]),
                    atom_kind=kind,
                    section=section,
                    claim=str(atom.get("claim") or ""),
                    evidence=evidence,
                    chunk_images=images,
                )
    return result


def load_exact_evidence_map(output_dir: Path) -> dict[str, EvidenceRecord]:
    l1 = read_json(output_dir / "01_l1_chunk_results.json")
    index = read_json(output_dir / "02_evidence_index.json")
    return build_exact_evidence_map(l1, index)


def resolve_refs(refs: list[str], evidence_map: dict[str, EvidenceRecord], expected_kind: str | None = None) -> tuple[list[EvidenceRecord], list[str]]:
    resolved: list[EvidenceRecord] = []
    invalid: list[str] = []
    for raw in refs:
        ref = canonical_ref(raw)
        record = evidence_map.get(ref or "")
        if record is None or (expected_kind and record.atom_kind != expected_kind):
            invalid.append(str(raw))
        else:
            resolved.append(record)
    return resolved, invalid


def strict_visual_support(
    item: dict[str, Any],
    evidence_map: dict[str, EvidenceRecord],
    expected_kind: str | None = None,
) -> dict[str, Any]:
    resolved, invalid = resolve_refs(evidence_refs(item), evidence_map, expected_kind=expected_kind)
    visual_records = [record for record in resolved if record.image_evidence and record.visual_cues and record.chunk_images]
    return {
        "has_strict_visual_support": bool(visual_records),
        "visual_asset_ids": sorted({image for record in visual_records for image in record.chunk_images}),
        "visual_cues": sorted({cue for record in visual_records for cue in record.visual_cues}),
        "invalid_refs": invalid,
    }


def relation_evidence_support(link: dict[str, Any], evidence_map: dict[str, EvidenceRecord]) -> dict[str, Any]:
    """Conservatively verify that relation evidence reaches both endpoint types."""
    resolved, invalid = resolve_refs(evidence_refs(link), evidence_map)
    kinds = {record.atom_kind for record in resolved}
    return {
        "supported": {"problem", "method"}.issubset(kinds),
        "resolved_refs": [record.ref for record in resolved],
        "resolved_kinds": sorted(kinds),
        "invalid_refs": invalid,
    }


def audit_final_result(data: dict[str, Any], evidence_map: dict[str, EvidenceRecord], keep_invalid_nodes: bool = False) -> dict[str, Any]:
    problems_in = data.get("final_research_problems") or data.get("paper_research_problems") or []
    methods_in = data.get("final_methods") or data.get("paper_methods") or []
    problem_out: list[dict[str, Any]] = []
    method_out: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []

    for kind, items, target in (("problem", problems_in, problem_out), ("method", methods_in, method_out)):
        for item in items:
            if not isinstance(item, dict):
                continue
            resolved, invalid = resolve_refs(evidence_refs(item), evidence_map, expected_kind=kind)
            copied = dict(item)
            copied["evidence_refs"] = [record.ref for record in resolved]
            copied["evidence_audit"] = {
                "resolved_ref_count": len(resolved),
                "invalid_refs": invalid,
                **strict_visual_support(copied, evidence_map),
            }
            if resolved or keep_invalid_nodes:
                target.append(copied)
            else:
                rejected.append({"kind": kind, "id": item.get("id"), "reason": "no exact type-consistent evidence ref", "invalid_refs": invalid})

    problem_ids = {item.get("id") for item in problem_out}
    method_ids = {item.get("id") for item in method_out}
    links_out: list[dict[str, Any]] = []
    for link in data.get("problem_method_links", []):
        if not isinstance(link, dict):
            continue
        copied = dict(link)
        if copied.get("problem_id") not in problem_ids or copied.get("method_id") not in method_ids:
            rejected.append({"kind": "link", "problem_id": copied.get("problem_id"), "method_id": copied.get("method_id"), "reason": "invalid endpoint"})
            continue
        link_refs = evidence_refs(copied)
        relation_audit = relation_evidence_support(copied, evidence_map)
        resolved_refs = relation_audit["resolved_refs"]
        invalid = relation_audit["invalid_refs"]
        if str(copied.get("link_type") or "evidence_supported") == "inferred":
            copied["link_type"] = "inferred"
            copied["evidence_refs"] = []
            copied["evidence_audit"] = {"resolved_ref_count": 0, "invalid_refs": invalid}
            links_out.append(copied)
        elif relation_audit["supported"]:
            copied["link_type"] = "evidence_supported"
            copied["evidence_refs"] = resolved_refs
            copied["evidence_audit"] = {"resolved_ref_count": len(resolved_refs), **relation_audit}
            links_out.append(copied)
        else:
            copied["link_type"] = "unverified"
            copied["evidence_refs"] = resolved_refs
            copied["evidence_audit"] = {"resolved_ref_count": len(resolved_refs), **relation_audit}
            links_out.append(copied)

    return {
        "final_research_problems": problem_out,
        "final_methods": method_out,
        "problem_method_links": links_out,
        "strict_audit_report": {
            "input_problem_count": len(problems_in),
            "input_method_count": len(methods_in),
            "retained_problem_count": len(problem_out),
            "retained_method_count": len(method_out),
            "rejected": rejected,
            "definition": "Exact chunk_id:local_atom_id resolution with problem/method type checking; relation evidence is audited separately.",
        },
    }


def attach_strict_audit_diagnostics(data: dict[str, Any], evidence_map: dict[str, EvidenceRecord]) -> dict[str, Any]:
    """Add diagnostics without filtering or rewriting original Qwen-VL predictions."""
    copied = {
        key: value
        for key, value in data.items()
    }
    diagnostics = audit_final_result(data, evidence_map, keep_invalid_nodes=True)
    copied["strict_audit_diagnostics"] = diagnostics["strict_audit_report"]
    copied["strict_audit_diagnostics"]["non_mutating"] = True
    copied["strict_audit_diagnostics"]["note"] = "Predicted nodes/links above are unchanged; this block is evaluation-only diagnostics."
    return copied
