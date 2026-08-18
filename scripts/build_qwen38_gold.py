from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paper_mining.config import PipelineConfig
from revision.io_utils import discover_papers, read_json, write_json
from revision.qwen38_main_aligned_gold import run_main_aligned_gold_pipeline


REQUIRED_OUTPUTS = (
    "00_manifest.json",
    "01_l1_chunk_results.json",
    "02_evidence_index.json",
    "03_l2_paper_merge.json",
    "04_final_extraction.json",
    "human_gold.json",
    "annotation_metadata.json",
    "main_method_alignment.json",
)

# This is the already-generated 30-paper reference protocol. It is intentionally
# kept at the submitted 9k setting and is independent of the revised 6k method.
REFERENCE_OUTPUT_NAME = "revision_annotations_main_aligned"
REFERENCE_CHUNK_CHARS = 9000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate human_gold.json by running the unchanged main-method pipeline and prompts "
            "with Qwen3.8-Max instead of Qwen-VL-Max."
        )
    )
    parser.add_argument("--root-dir", required=True, help="data/output containing numbered paper folders")
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--end", type=int, default=30)
    parser.add_argument("--output-name", default=REFERENCE_OUTPUT_NAME)
    parser.add_argument("--model", default="qwen3.8-max", choices=["qwen3.8-max"])
    parser.add_argument("--base-url", default="https://dashscope.aliyuncs.com/compatible-mode/v1")
    parser.add_argument("--api-key", help="Prefer DASHSCOPE_API_KEY or --api-key-file")
    parser.add_argument("--api-key-file", help="Optional text file containing the DashScope API key")

    # Reference-generation defaults are frozen for reproducibility. The revised
    # evaluated method is added_chunk_6000; it does not redefine this reference.
    parser.add_argument("--max-chars-per-chunk", type=int, default=REFERENCE_CHUNK_CHARS)
    parser.add_argument("--overlap-chars", type=int, default=900)
    parser.add_argument("--max-images-per-chunk", type=int, default=4)
    parser.add_argument("--max-output-tokens", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--request-timeout", type=int, default=1200)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--progress-seconds", type=float, default=30)
    parser.add_argument("--skip-relation-completion", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def _api_key(args: argparse.Namespace) -> str:
    if args.api_key:
        return str(args.api_key).strip()
    if args.api_key_file:
        path = Path(args.api_key_file).resolve()
        if not path.exists():
            raise SystemExit(f"API key file not found: {path}")
        return path.read_text(encoding="utf-8-sig").strip()
    return (os.getenv("DASHSCOPE_API_KEY") or os.getenv("QWEN_API_KEY") or "").strip()


def _is_complete(output_dir: Path) -> bool:
    return all((output_dir / name).is_file() for name in REQUIRED_OUTPUTS)


def main() -> int:
    args = parse_args()
    if args.end < args.start:
        raise SystemExit("--end must be greater than or equal to --start")
    if args.max_chars_per_chunk <= 0 or args.overlap_chars < 0 or args.max_images_per_chunk <= 0:
        raise SystemExit("Chunk size and image count must be positive; overlap must be non-negative")
    api_key = _api_key(args)
    if not api_key and not args.dry_run:
        raise SystemExit(
            "Set DASHSCOPE_API_KEY, pass --api-key-file qwen_key.txt, or pass --api-key"
        )

    root = Path(args.root_dir).resolve()
    papers = discover_papers(root, args.start, args.end)
    expected = args.end - args.start + 1
    if len(papers) != expected:
        print(
            f"WARNING: discovered {len(papers)} paper folders for numeric range "
            f"{args.start}..{args.end} (expected {expected}).",
            flush=True,
        )

    summary: list[dict[str, object]] = []
    for position, paper_dir in enumerate(papers, start=1):
        output_dir = paper_dir / "outputs" / args.output_name
        if args.skip_existing and _is_complete(output_dir):
            metadata = read_json(output_dir / "annotation_metadata.json")
            print(f"[{position}/{len(papers)}] skip complete paper {paper_dir.name}", flush=True)
            summary.append({
                "paper_id": paper_dir.name,
                "status": "skipped_complete",
                "output_dir": str(output_dir),
                "problem_count": metadata.get("problem_count"),
                "method_count": metadata.get("method_count"),
                "link_count": metadata.get("link_count"),
            })
            continue

        print(
            f"[{position}/{len(papers)}] main-aligned Qwen3.8-Max gold: {paper_dir.name}",
            flush=True,
        )
        config = PipelineConfig(
            paper_dir=paper_dir.resolve(),
            markdown_path=(paper_dir / "full.md").resolve(),
            image_dir=(paper_dir / "images").resolve(),
            output_dir=output_dir.resolve(),
            cache_dir=(output_dir / "cache").resolve(),
            api_key=api_key or "dry-run",
            base_url=args.base_url,
            model=args.model,
            max_chars_per_chunk=args.max_chars_per_chunk,
            overlap_chars=args.overlap_chars,
            max_images_per_chunk=args.max_images_per_chunk,
            request_timeout=args.request_timeout,
            max_retries=args.max_retries,
            max_tokens=args.max_output_tokens,
            temperature=args.temperature,
            dry_run=args.dry_run,
            verbose=not args.quiet,
            enable_relation_completion=not args.skip_relation_completion,
        )
        metadata = run_main_aligned_gold_pipeline(
            config,
            progress_seconds=args.progress_seconds,
        )
        summary.append({
            "paper_id": paper_dir.name,
            "status": "generated_or_resumed_from_stage_cache",
            "output_dir": str(output_dir),
            "problem_count": metadata["problem_count"],
            "method_count": metadata["method_count"],
            "link_count": metadata["link_count"],
        })

    summary_path = root / "qwen38_main_aligned_gold_generation_summary.json"
    write_json(summary_path, {
        "status": "qwen38_max_main_method_aligned_gold",
        "definition": (
            "Exact current main-method pipeline and prompts; only the model backend is Qwen3.8-Max."
        ),
        "output_name": args.output_name,
        "selected_range": [args.start, args.end],
        "main_method_parameters": {
            "max_chars_per_chunk": args.max_chars_per_chunk,
            "overlap_chars": args.overlap_chars,
            "max_images_per_chunk": args.max_images_per_chunk,
            "max_output_tokens": args.max_output_tokens,
            "temperature": args.temperature,
            "relation_completion": not args.skip_relation_completion,
        },
        "papers": summary,
    })
    print(f"Completed {len(summary)} papers.", flush=True)
    print(f"Summary: {summary_path}", flush=True)
    print(
        f"Gold files: <paper>\\outputs\\{args.output_name}\\human_gold.json",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
