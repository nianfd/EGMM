from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from revision.contracts import validate_condition_output
from revision.evaluation import evaluate_paper
from revision.io_utils import write_json
from scripts.aggregate_batch_metrics import (
    ADDED_CONDITIONS,
    compute_metrics as compute_original_batch_metrics,
    condition_dir,
    context_dir_for_condition,
    find_result_file,
    load_json,
    load_trace_context,
    normalize_final_result,
)


def fixture_l1() -> list[dict]:
    return [{
        "chunk_id": "S001_C01",
        "section": "Method",
        "research_problem_atoms": [{
            "id": "RP-1",
            "claim": "lack of robust visual evidence",
            "problem_type": "method_gap",
            "evidence": [{"source": "text", "quote_or_visual_cue": "lack of robust visual evidence", "explanation": "explicit gap"}],
            "confidence": 0.9,
        }],
        "method_atoms": [{
            "id": "M-1",
            "claim": "a multimodal evidence module",
            "method_type": "architecture",
            "inputs": ["paper text", "figures"],
            "outputs": ["grounded graph"],
            "evidence": [{"source": "image", "quote_or_visual_cue": "two-branch architecture", "explanation": "architecture figure"}],
            "confidence": 0.9,
        }],
    }]


def fixture_final() -> dict:
    return {
        "final_research_problems": [{
            "id": "RP1",
            "problem": "lack of robust visual evidence",
            "problem_type": "method_gap",
            "granularity": "fine",
            "explicitness": "explicit",
            "evidence_refs": ["S001_C01:RP-1"],
            "confidence": 0.9,
            "risk_note": "",
        }],
        "final_methods": [{
            "id": "M1",
            "method": "a multimodal evidence module",
            "method_type": "architecture",
            "reproducibility_fields": {
                "inputs": ["paper text", "figures"],
                "outputs": ["grounded graph"],
                "procedure": ["fuse evidence"],
                "objective_or_metric": ["evidence coverage"],
            },
            "granularity": "fine",
            "evidence_refs": ["S001_C01:M-1"],
            "confidence": 0.9,
            "risk_note": "",
        }],
        "problem_method_links": [{
            "problem_id": "RP1",
            "method_id": "M1",
            "relation": "directly_addresses",
            "link_type": "evidence_supported",
            "evidence_refs": ["S001_C01:RP-1", "S001_C01:M-1"],
            "confidence": 0.9,
            "rationale": "the module addresses the evidence gap",
        }],
        "quality_report": {},
    }


def as_l2(final: dict) -> dict:
    return {
        "paper_research_problems": [{
            "id": item["id"], "problem": item["problem"], "problem_type": item["problem_type"],
            "explicitness": item["explicitness"], "evidence_refs": item["evidence_refs"], "confidence": item["confidence"],
        } for item in final["final_research_problems"]],
        "paper_methods": [{
            "id": item["id"], "method": item["method"], "method_type": item["method_type"],
            "inputs": item["reproducibility_fields"]["inputs"], "outputs": item["reproducibility_fields"]["outputs"],
            "evidence_refs": item["evidence_refs"], "confidence": item["confidence"],
        } for item in final["final_methods"]],
        "problem_method_links": final["problem_method_links"],
    }


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="papermining_metric_contract_") as temp:
        root = Path(temp)
        paper = root / "1"
        (paper / "images").mkdir(parents=True)
        (paper / "full.md").write_text("# Test\n\nA metric-contract fixture.", encoding="utf-8")
        (paper / "images" / "fig1.png").write_bytes(b"fixture")
        l1 = fixture_l1()
        evidence_index = {"S001_C01": {"section": "Method", "images": ["fig1.png"]}}
        final = fixture_final()
        gold_dir = paper / "outputs" / "revision_annotations"
        write_json(gold_dir / "human_gold.json", {
            "paper_id": "1",
            "research_problems": [{
                "id": "RP1", "claim": "lack of robust visual evidence", "problem_type": "method_gap",
                "evidence": [{"span_ids": ["P0001"], "quote": "lack of robust visual evidence", "visual_asset_ids": [], "visual_cue": "", "support_type": "text"}],
                "visual_dependency": "text_sufficient",
            }],
            "methods": [{
                "id": "M1", "claim": "a multimodal evidence module", "method_type": "architecture",
                "reproducibility_fields": {"inputs": ["paper text", "figures"], "outputs": ["grounded graph"], "procedure": ["fuse evidence"], "objective_or_metric": ["evidence coverage"]},
                "evidence": [{"span_ids": [], "quote": "", "visual_asset_ids": ["fig1.png"], "visual_cue": "two-branch architecture", "support_type": "visual"}],
                "visual_dependency": "visual_dependent",
            }],
            "problem_method_links": [{
                "problem_id": "RP1", "method_id": "M1", "relation": "directly_addresses",
                "evidence": [{"span_ids": ["P0001"], "quote": "addresses", "visual_asset_ids": ["fig1.png"], "visual_cue": "two-branch architecture", "support_type": "mixed"}],
                "visual_dependency": "mixed",
            }],
            "difficult_or_ambiguous_cases": [],
        })

        reports = []
        for condition in sorted(ADDED_CONDITIONS):
            directory = paper / "outputs" / "major_revision_additions" / condition
            write_json(directory / "01_l1_chunk_results.json", l1)
            write_json(directory / "02_evidence_index.json", evidence_index)
            write_json(directory / "03_l2_paper_merge.json", as_l2(final))
            result_name = "result.json" if condition == "added_l3_off_frozen_l2" else "04_final_extraction.json"
            write_json(directory / result_name, as_l2(final) if result_name == "result.json" else final)
            write_json(directory / "condition_info.json", {"condition": condition, "model": "qwen-vl-max", "dry_run": False})

            contract = validate_condition_output(directory, condition, allow_empty=False)
            if not contract["valid"]:
                raise AssertionError(f"contract failed for {condition}: {contract['errors']}")
            semantic = evaluate_paper(paper, condition, gold_dir / "human_gold.json")
            if semantic["problem"]["f1"] != 1.0 or semantic["method"]["f1"] != 1.0 or semantic["link"]["f1"] != 1.0:
                raise AssertionError(f"gold metric mismatch for {condition}: {semantic}")
            if semantic["structural_diagnostics"]["strict_relation_evidence_rate"] != 1.0:
                raise AssertionError(f"strict relation evidence failed for {condition}")

            resolved_dir = condition_dir(paper, condition)
            if resolved_dir != directory:
                raise AssertionError(f"original aggregator path mismatch for {condition}: {resolved_dir}")
            result_path = find_result_file(resolved_dir)
            if result_path is None:
                raise AssertionError(f"original aggregator found no result for {condition}")
            traceable, visual = load_trace_context(context_dir_for_condition(paper, condition))
            original = compute_original_batch_metrics(normalize_final_result(load_json(result_path)), traceable, visual)
            if original["NET_node_evidence_traceability"] != 1.0:
                raise AssertionError(f"original NET metric mismatch for {condition}: {original}")
            reports.append({
                "condition": condition,
                "contract": "passed",
                "gold_metrics": "passed",
                "original_metrics": "passed",
            })

        aggregate_dir = root / "aggregate_cli"
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "aggregate_batch_metrics.py"),
                "--root-dir", str(root),
                "--start", "1",
                "--end", "1",
                "--conditions", *sorted(ADDED_CONDITIONS),
                "--output-dir", str(aggregate_dir),
            ],
            check=True,
            cwd=ROOT,
            stdout=subprocess.DEVNULL,
        )
        with (aggregate_dir / "metrics_by_paper.csv").open(encoding="utf-8-sig", newline="") as handle:
            cli_rows = list(csv.DictReader(handle))
        if len(cli_rows) != len(ADDED_CONDITIONS) or any(row.get("status") != "ok" for row in cli_rows):
            raise AssertionError(f"original aggregate CLI did not consume all additions: {cli_rows}")

        print(json.dumps({
            "passed": len(reports),
            "conditions": reports,
            "original_aggregate_cli_rows": len(cli_rows),
        }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
