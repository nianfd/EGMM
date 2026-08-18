from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from revision.evidence import build_exact_evidence_map
from revision.io_utils import condition_result_path, discover_papers, evidence_refs, node_id, node_text, normalize_final, read_json, write_csv


def main() -> int:
    parser = argparse.ArgumentParser(description="Export blinded node/link evidence packets for real expert faithfulness ratings.")
    parser.add_argument("--root-dir", required=True)
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--end", type=int, default=30)
    parser.add_argument("--conditions", nargs="+", default=["original_main"])
    parser.add_argument("--sample-per-condition", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    rows = []
    for paper in discover_papers(Path(args.root_dir), args.start, args.end):
        evidence_map = build_exact_evidence_map(read_json(paper / "outputs/01_l1_chunk_results.json"), read_json(paper / "outputs/02_evidence_index.json"))
        for condition in args.conditions:
            path = condition_result_path(paper, condition)
            if not path.exists():
                continue
            condition_l1 = path.parent / "01_l1_chunk_results.json"
            condition_index = path.parent / "02_evidence_index.json"
            if condition_l1.exists() and condition_index.exists():
                evidence_map = build_exact_evidence_map(read_json(condition_l1), read_json(condition_index))
            result = normalize_final(read_json(path))
            problem_by_id = {node_id(item, "problem", index + 1): item for index, item in enumerate(result["problems"])}
            method_by_id = {node_id(item, "method", index + 1): item for index, item in enumerate(result["methods"])}
            for kind, items in (("problem", result["problems"]), ("method", result["methods"]), ("relation", result["links"])):
                for index, item in enumerate(items):
                    refs = evidence_refs(item)
                    evidence_scope = "direct_item_evidence"
                    if kind == "relation" and not refs:
                        problem = problem_by_id.get(str(item.get("problem_id") or ""), {})
                        method = method_by_id.get(str(item.get("method_id") or ""), {})
                        refs = evidence_refs(problem) + evidence_refs(method)
                        evidence_scope = "endpoint_node_evidence_only_not_direct_relation_evidence"
                    evidence_text = []
                    visual_assets = set()
                    for ref in refs:
                        record = evidence_map.get(ref)
                        if record:
                            evidence_text.append(f"{ref} | atom={record.claim} | source_evidence={record.evidence}")
                            visual_assets.update(record.chunk_images)
                    rows.append({
                        "paper_id": paper.name,
                        "condition": condition,
                        "item_kind": kind,
                        "item_id": node_id(item, kind, index + 1) if kind != "relation" else f"{item.get('problem_id')}->{item.get('method_id')}",
                        "claim_or_relation": node_text(item) if kind != "relation" else f"{item.get('relation')}: {item.get('rationale', '')}",
                        "cited_evidence": " || ".join(evidence_text),
                        "visual_asset_ids": ";".join(sorted(visual_assets)),
                        "evidence_scope": evidence_scope,
                        "support_label": "",
                        "reviewer_confidence": "",
                        "reviewer_note": "",
                    })
    rng = random.Random(args.seed)
    sampled = []
    for condition in args.conditions:
        candidates = [row for row in rows if row["condition"] == condition]
        rng.shuffle(candidates)
        sampled.extend(candidates[: args.sample_per_condition])
    rng.shuffle(sampled)
    for index, row in enumerate(sampled, start=1):
        row["blind_item_id"] = f"E{index:05d}"
    write_csv(Path(args.output), sampled, fieldnames=["blind_item_id", "paper_id", "condition", "item_kind", "item_id", "claim_or_relation", "cited_evidence", "visual_asset_ids", "evidence_scope", "support_label", "reviewer_confidence", "reviewer_note"])
    print(f"Wrote {len(sampled)} blinded items. Allowed support_label: fully_supported, partially_supported, not_supported, unjudgeable.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
