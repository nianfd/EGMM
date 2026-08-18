from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from revision.annotation_schema import PaperGoldAnnotation
from revision.evaluation import normalize_relation
from revision.io_utils import discover_papers, node_text, normalize_final, read_json, write_json


REQUIRED_FILES = (
    "00_manifest.json",
    "01_l1_chunk_results.json",
    "02_evidence_index.json",
    "03_l2_paper_merge.json",
    "04_final_extraction.json",
    "human_gold.json",
    "annotation_metadata.json",
    "main_method_alignment.json",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate Qwen3.8-Max main-method-aligned gold outputs")
    parser.add_argument("--root-dir", required=True)
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--end", type=int, default=30)
    parser.add_argument("--output-name", default="revision_annotations_main_aligned")
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument("--report")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.root_dir).resolve()
    papers = discover_papers(root, args.start, args.end)
    reports: list[dict[str, object]] = []
    failed = 0
    for paper_dir in papers:
        output_dir = paper_dir / "outputs" / args.output_name
        errors: list[str] = []
        missing = [name for name in REQUIRED_FILES if not (output_dir / name).is_file()]
        if missing:
            errors.append(f"missing files: {missing}")
        problem_count = method_count = link_count = None
        if not missing:
            try:
                gold = PaperGoldAnnotation.model_validate(read_json(output_dir / "human_gold.json"))
                final = normalize_final(read_json(output_dir / "04_final_extraction.json"))
                metadata = read_json(output_dir / "annotation_metadata.json")
                alignment = read_json(output_dir / "main_method_alignment.json")
                if gold.paper_id != paper_dir.name:
                    errors.append(f"paper_id mismatch: {gold.paper_id!r}")
                if metadata.get("annotation_status") != "qwen38_max_main_method_aligned_gold":
                    errors.append("annotation_status is not qwen38_max_main_method_aligned_gold")
                if str(metadata.get("annotation_backend") or "").lower() != "qwen3.8-max":
                    errors.append("annotation_backend is not qwen3.8-max")
                if str(alignment.get("model") or "").lower() != "qwen3.8-max":
                    errors.append("alignment model is not qwen3.8-max")
                for key, expected in (
                    ("max_chars_per_chunk", 9000),
                    ("overlap_chars", 900),
                    ("max_images_per_chunk", 4),
                    ("max_tokens", 8192),
                ):
                    if alignment.get(key) != expected:
                        errors.append(f"main alignment {key}={alignment.get(key)!r}; expected {expected!r}")

                gold_problem_pairs = [(item.id, item.claim) for item in gold.research_problems]
                final_problem_pairs = [
                    (str(item.get("id") or ""), node_text(item)) for item in final["problems"] if node_text(item)
                ]
                gold_method_pairs = [(item.id, item.claim) for item in gold.methods]
                final_method_pairs = [
                    (str(item.get("id") or ""), node_text(item)) for item in final["methods"] if node_text(item)
                ]
                if gold_problem_pairs != final_problem_pairs:
                    errors.append("gold problem IDs/claims differ from 04_final_extraction.json")
                if gold_method_pairs != final_method_pairs:
                    errors.append("gold method IDs/claims differ from 04_final_extraction.json")
                final_links = {
                    (
                        str(item.get("problem_id") or ""),
                        str(item.get("method_id") or ""),
                        normalize_relation(item.get("relation")),
                    )
                    for item in final["links"]
                }
                gold_links = {
                    (item.problem_id, item.method_id, item.relation)
                    for item in gold.problem_method_links
                }
                if not gold_links.issubset(final_links):
                    errors.append("gold contains a link not present in 04_final_extraction.json")
                problem_count = len(gold.research_problems)
                method_count = len(gold.methods)
                link_count = len(gold.problem_method_links)
            except Exception as exc:
                errors.append(f"validation exception: {type(exc).__name__}: {exc}")
        valid = not errors
        failed += 0 if valid else 1
        reports.append({
            "paper_id": paper_dir.name,
            "valid": valid,
            "errors": errors,
            "problem_count": problem_count,
            "method_count": method_count,
            "link_count": link_count,
            "output_dir": str(output_dir),
        })

    expected = args.end - args.start + 1
    if args.require_complete and len(papers) != expected:
        failed += expected - len(papers)
        reports.append({
            "paper_id": "__range__",
            "valid": False,
            "errors": [f"discovered {len(papers)} papers; expected {expected}"],
        })
    report_path = (
        Path(args.report).resolve()
        if args.report
        else root / "revision_metrics" / "main_aligned_gold_validation_30.json"
    )
    payload = {
        "status": "passed" if failed == 0 else "failed",
        "checked": len(papers),
        "failed": failed,
        "output_name": args.output_name,
        "papers": reports,
    }
    write_json(report_path, payload)
    print(json.dumps({
        "status": payload["status"],
        "checked": len(papers),
        "failed": failed,
        "report": str(report_path),
    }, ensure_ascii=False, indent=2))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
