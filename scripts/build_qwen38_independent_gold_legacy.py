from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from revision.annotation import run_assisted_annotation
from revision.dashscope_qwen38_client import Qwen38AnnotationClient
from revision.io_utils import discover_papers, read_json, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create the backward-compatible human_gold.json filename with Qwen3.8-Max from the complete "
            "MinerU Markdown and every parsed image. This is model-generated gold, not human-verified gold. "
            "No experimental predictions are exposed to the gold generator."
        )
    )
    parser.add_argument("--root-dir", required=True, help="data/output containing paper folders")
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--end", type=int, default=30)
    parser.add_argument("--output-name", default="revision_annotations")
    parser.add_argument("--model", default="qwen3.8-max", choices=["qwen3.8-max"])
    parser.add_argument("--base-url", default="https://dashscope.aliyuncs.com/compatible-mode/v1")
    parser.add_argument("--max-output-tokens", type=int, default=24000)
    parser.add_argument(
        "--request-timeout", type=float, default=1200,
        help="Maximum seconds for one A/B/adjudication API attempt",
    )
    parser.add_argument(
        "--progress-seconds", type=float, default=30,
        help="Print a heartbeat without exposing reasoning text",
    )
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    api_key = os.getenv("DASHSCOPE_API_KEY") or os.getenv("QWEN_API_KEY") or ""
    if not api_key and not args.dry_run:
        raise SystemExit("Set DASHSCOPE_API_KEY (recommended) or QWEN_API_KEY")
    root = Path(args.root_dir).resolve()
    papers = discover_papers(root, args.start, args.end)
    if len(papers) != args.end - args.start + 1:
        print(f"WARNING: selected {len(papers)} paper folders for numeric range {args.start}..{args.end}")
    summary = []
    for index, paper_dir in enumerate(papers, start=1):
        output_dir = paper_dir / "outputs" / args.output_name
        if args.skip_existing and (output_dir / "human_gold.json").exists():
            print(f"[{index}/{len(papers)}] skip {paper_dir.name}", flush=True)
            metadata_path = output_dir / "annotation_metadata.json"
            metadata = read_json(metadata_path) if metadata_path.exists() else {}
            summary.append({
                "paper_id": paper_dir.name,
                "human_gold": str(output_dir / "human_gold.json"),
                "model": metadata.get("annotation_backend", "qwen3.8-max"),
                "source_char_count": metadata.get("source_char_count"),
                "submitted_char_count": metadata.get("submitted_char_count"),
                "source_image_count": metadata.get("source_image_count"),
                "submitted_image_count": metadata.get("submitted_image_count"),
                "verification": metadata.get("verification"),
                "run_status": "skipped_existing",
            })
            continue
        print(f"[{index}/{len(papers)}] Qwen3.8-Max annotate {paper_dir.name}", flush=True)
        client = Qwen38AnnotationClient(
            api_key=api_key or "dry-run",
            model=args.model,
            base_url=args.base_url,
            max_output_tokens=args.max_output_tokens,
            max_retries=args.max_retries,
            request_timeout=args.request_timeout,
            progress_seconds=args.progress_seconds,
            cache_dir=output_dir / "cache",
            dry_run=args.dry_run,
        )
        metadata = run_assisted_annotation(
            paper_dir,
            output_dir,
            client,
        )
        summary.append({
            "paper_id": paper_dir.name,
            "human_gold": str(output_dir / "human_gold.json"),
            "model": "qwen3.8-max",
            "source_char_count": metadata["source_char_count"],
            "submitted_char_count": metadata["submitted_char_count"],
            "source_image_count": metadata["source_image_count"],
            "submitted_image_count": metadata["submitted_image_count"],
            "verification": metadata["verification"],
            "run_status": "generated_or_recovered_from_cache",
        })
    write_json(root / "qwen38_gold_generation_summary.json", {
        "status": "qwen38_max_generated_model_gold",
        "input_policy": "complete MinerU Markdown plus every supported image; no text/image cap",
        "papers": summary,
    })
    print(f"Completed {len(summary)} papers. Each selected paper now has outputs/{args.output_name}/human_gold.json.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
