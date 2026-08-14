from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import PipelineConfig
from .pipeline import run_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract fine-grained research problems and methods from MinerU markdown/images with Qwen-VL."
    )
    parser.add_argument("--paper-dir", default="data/paper1", help="Directory containing full.md and images/.")
    parser.add_argument("--markdown", default=None, help="Path to MinerU full.md. Defaults to paper-dir/full.md.")
    parser.add_argument("--images", default=None, help="Path to MinerU images directory. Defaults to paper-dir/images.")
    parser.add_argument("--output-dir", default=None, help="Output directory. Defaults to paper-dir/outputs.")
    parser.add_argument("--cache-dir", default=None, help="Local JSON cache directory. Defaults to output-dir/cache.")
    parser.add_argument("--api-key", default="", help="API key. Prefer env DASHSCOPE_API_KEY or QWEN_API_KEY.")
    parser.add_argument("--base-url", default="https://dashscope.aliyuncs.com/compatible-mode/v1")
    parser.add_argument("--model", default="qwen-vl-max")
    parser.add_argument("--max-chars-per-chunk", type=int, default=9000)
    parser.add_argument("--overlap-chars", type=int, default=900)
    parser.add_argument("--max-images-per-chunk", type=int, default=4)
    parser.add_argument("--request-timeout", type=int, default=600)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--dry-run", action="store_true", help="Build chunks/cache skeleton without calling API.")
    parser.add_argument("--quiet", action="store_true", help="Disable step-by-step progress logs.")
    parser.add_argument(
        "--skip-relation-completion",
        action="store_true",
        help="Disable inferred problem-method relation completion after L3.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.markdown is None:
        args.markdown = str(Path(args.paper_dir) / "full.md")
    if args.images is None:
        args.images = str(Path(args.paper_dir) / "images")
    if args.output_dir is None:
        args.output_dir = str(Path(args.paper_dir) / "outputs")
    if args.cache_dir is None:
        args.cache_dir = str(Path(args.output_dir) / "cache")
    try:
        config = PipelineConfig.from_args(args)
        result = run_pipeline(config)
    except Exception as exc:
        print(f"[paper_mining] failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({"output": str(config.output_dir), "final": result}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
