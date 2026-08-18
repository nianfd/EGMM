from __future__ import annotations

import argparse
from pathlib import Path

from common import api_key_from_env, ensure_dir, truncate_text, write_json
from baseline_prompts import DIRECT_EXTRACTION_SYSTEM, DIRECT_EXTRACTION_USER_TEMPLATE
from paper_mining.io_utils import read_text
from paper_mining.markdown_mineru import discover_images
from paper_mining.qwenvl_client import QwenVLClient


def select_images(image_dir: Path, max_images: int) -> list[Path]:
    images = discover_images(image_dir)
    if max_images <= 0:
        return []
    if len(images) <= max_images:
        return images
    if max_images == 1:
        return [images[0]]
    step = max(1, len(images) // max_images)
    sampled = images[::step][:max_images]
    return sampled


def run_one_shot_baseline(
    paper_dir: Path,
    output_dir: Path,
    model: str,
    base_url: str,
    api_key: str,
    max_text_chars: int,
    max_images: int,
    timeout: int,
    max_tokens: int,
    dry_run: bool,
) -> dict:
    markdown_path = paper_dir / "full.md"
    image_dir = paper_dir / "images"
    output_dir = ensure_dir(output_dir)
    cache_dir = ensure_dir(output_dir / "cache")

    paper_text = truncate_text(read_text(markdown_path), max_text_chars)
    images = select_images(image_dir, max_images)
    user_prompt = DIRECT_EXTRACTION_USER_TEMPLATE.format(
        paper_text=paper_text,
        image_names=[p.name for p in images],
    )
    client = QwenVLClient(
        api_key=api_key,
        base_url=base_url,
        model=model,
        cache_dir=cache_dir,
        timeout=timeout,
        max_tokens=max_tokens,
        dry_run=dry_run,
    )
    result = client.chat_json(
        stage="baseline_oneshot_mineru",
        system_prompt=DIRECT_EXTRACTION_SYSTEM,
        user_text=user_prompt,
        images=images,
        extra_cache_key={
            "paper_dir": str(paper_dir),
            "max_text_chars": max_text_chars,
            "max_images": max_images,
        },
    )
    write_json(output_dir / "baseline_oneshot_mineru.json", result)
    write_json(
        output_dir / "baseline_oneshot_mineru_manifest.json",
        {
            "paper_dir": str(paper_dir),
            "markdown_path": str(markdown_path),
            "image_dir": str(image_dir),
            "max_text_chars": max_text_chars,
            "selected_images": [str(p) for p in images],
            "output": str(output_dir / "baseline_oneshot_mineru.json"),
        },
    )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one-shot MinerU baseline.")
    parser.add_argument("--paper-dir", default="data/paper1")
    parser.add_argument("--output-dir", default="experiments/results/paper1/baseline_oneshot_mineru")
    parser.add_argument("--base-url", default="https://dashscope.aliyuncs.com/compatible-mode/v1")
    parser.add_argument("--model", default="qwen-vl-max")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--max-text-chars", type=int, default=60000)
    parser.add_argument("--max-images", type=int, default=8)
    parser.add_argument("--request-timeout", type=int, default=600)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    api_key = args.api_key or api_key_from_env()
    if not api_key and not args.dry_run:
        raise SystemExit("Missing API key. Set DASHSCOPE_API_KEY or QWEN_API_KEY.")
    run_one_shot_baseline(
        paper_dir=Path(args.paper_dir).resolve(),
        output_dir=Path(args.output_dir).resolve(),
        model=args.model,
        base_url=args.base_url,
        api_key=api_key,
        max_text_chars=args.max_text_chars,
        max_images=args.max_images,
        timeout=args.request_timeout,
        max_tokens=args.max_tokens,
        dry_run=args.dry_run,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
