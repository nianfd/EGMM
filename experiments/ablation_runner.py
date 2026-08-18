from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from common import api_key_from_env, ensure_dir, write_json
from one_shot_mineru_baseline import run_one_shot_baseline
from paper_mining.config import PipelineConfig
from paper_mining.io_utils import read_text
from paper_mining.markdown_mineru import make_chunks, parse_sections
from paper_mining.pipeline import build_evidence_index, build_manifest, merge_l1_results, run_pipeline
from paper_mining.progress import progress
from paper_mining.prompts import L1_SYSTEM, l1_user_prompt
from paper_mining.qwenvl_client import QwenVLClient


CONDITIONS = {
    "proposed_full",
    "baseline_oneshot_mineru",
    "ablation_text_only",
    "ablation_no_l3",
    "ablation_large_chunk",
}


def make_config(args: argparse.Namespace, condition: str) -> PipelineConfig:
    paper_dir = Path(args.paper_dir).resolve()
    output_dir = Path(args.output_root).resolve() / condition
    cache_dir = output_dir / "cache"
    class Args:
        pass
    obj = Args()
    obj.paper_dir = str(paper_dir)
    obj.markdown = str(paper_dir / "full.md")
    obj.images = str(paper_dir / "images")
    obj.output_dir = str(output_dir)
    obj.cache_dir = str(cache_dir)
    obj.api_key = args.api_key or api_key_from_env()
    obj.base_url = args.base_url
    obj.model = args.model
    obj.max_chars_per_chunk = args.max_chars_per_chunk
    obj.overlap_chars = args.overlap_chars
    obj.max_images_per_chunk = args.max_images_per_chunk
    obj.request_timeout = args.request_timeout
    obj.max_retries = args.max_retries
    obj.max_tokens = args.max_tokens
    obj.temperature = args.temperature
    obj.dry_run = args.dry_run
    obj.quiet = args.quiet
    return PipelineConfig.from_args(obj)


def run_l1_l2_only(config: PipelineConfig) -> dict[str, Any]:
    progress("Ablation no_l3 started", config.verbose)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.cache_dir.mkdir(parents=True, exist_ok=True)
    markdown = read_text(config.markdown_path)
    sections = parse_sections(markdown)
    chunks = make_chunks(
        sections=sections,
        image_dir=config.image_dir,
        max_chars=config.max_chars_per_chunk,
        overlap_chars=config.overlap_chars,
        max_images=config.max_images_per_chunk,
    )
    write_json(config.output_dir / "00_manifest.json", build_manifest(config, chunks))

    client = QwenVLClient(
        api_key=config.api_key,
        base_url=config.base_url,
        model=config.model,
        cache_dir=config.cache_dir,
        timeout=config.request_timeout,
        max_retries=config.max_retries,
        max_tokens=config.max_tokens,
        temperature=config.temperature,
        dry_run=config.dry_run,
        verbose=config.verbose,
    )

    l1_results = []
    for index, chunk in enumerate(chunks, start=1):
        progress(f"no_l3 L1 chunk {index}/{len(chunks)}: {chunk.id}", config.verbose)
        result = client.chat_json(
            stage="no_l3_l1_chunk_extract",
            system_prompt=L1_SYSTEM,
            user_text=l1_user_prompt(
                chunk_id=chunk.id,
                section=chunk.section_title,
                text=chunk.text,
                image_names=[p.name for p in chunk.image_paths],
            ),
            images=chunk.image_paths,
            extra_cache_key={"chunk_id": chunk.id},
        )
        result.setdefault("chunk_id", chunk.id)
        result.setdefault("section", chunk.section_title)
        l1_results.append(result)
    write_json(config.output_dir / "01_l1_chunk_results.json", l1_results)
    write_json(config.output_dir / "02_evidence_index.json", build_evidence_index(chunks, l1_results))
    l2 = merge_l1_results(client, l1_results, config.max_chars_per_chunk * 2)
    write_json(config.output_dir / "ablation_no_l3_result.json", l2)
    return l2


def run_condition(args: argparse.Namespace, condition: str) -> Path:
    config = make_config(args, condition)
    if not config.api_key and not config.dry_run:
        raise ValueError("Missing API key. Set DASHSCOPE_API_KEY/QWEN_API_KEY or pass --api-key.")

    if condition == "proposed_full":
        result = run_pipeline(config)
        output = config.output_dir / "04_final_extraction.json"
    elif condition == "baseline_oneshot_mineru":
        run_one_shot_baseline(
            paper_dir=config.paper_dir,
            output_dir=config.output_dir,
            model=config.model,
            base_url=config.base_url,
            api_key=config.api_key,
            max_text_chars=args.oneshot_max_text_chars,
            max_images=args.oneshot_max_images,
            timeout=config.request_timeout,
            max_tokens=config.max_tokens,
            dry_run=config.dry_run,
        )
        output = config.output_dir / "baseline_oneshot_mineru.json"
    elif condition == "ablation_text_only":
        result = run_pipeline(replace(config, max_images_per_chunk=0))
        output = config.output_dir / "04_final_extraction.json"
    elif condition == "ablation_no_l3":
        result = run_l1_l2_only(config)
        output = config.output_dir / "ablation_no_l3_result.json"
    elif condition == "ablation_large_chunk":
        large = replace(
            config,
            max_chars_per_chunk=max(args.large_chunk_chars, config.max_chars_per_chunk),
            overlap_chars=min(config.overlap_chars, 300),
            max_images_per_chunk=max(config.max_images_per_chunk, args.large_chunk_images),
        )
        result = run_pipeline(large)
        output = config.output_dir / "04_final_extraction.json"
    else:
        raise ValueError(f"Unknown condition: {condition}")

    write_json(
        config.output_dir / "condition_info.json",
        {
            "condition": condition,
            "paper_dir": str(config.paper_dir),
            "output_file": str(output),
            "description": describe_condition(condition),
        },
    )
    return output


def describe_condition(condition: str) -> str:
    descriptions = {
        "proposed_full": "Full proposed L1-L2-L3 multimodal streaming pipeline.",
        "baseline_oneshot_mineru": "One-shot baseline using MinerU text and sampled images in a single model call.",
        "ablation_text_only": "Removes image inputs to test cross-modal evidence contribution.",
        "ablation_no_l3": "Removes the final quality-audit stage to test L3 refinement contribution.",
        "ablation_large_chunk": "Uses much larger chunks to weaken section-aware streaming.",
    }
    return descriptions[condition]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run comparison and ablation experiments.")
    parser.add_argument("--paper-dir", default="data/paper1")
    parser.add_argument("--output-root", default="experiments/results/paper1")
    parser.add_argument(
        "--conditions",
        default="proposed_full,baseline_oneshot_mineru,ablation_text_only,ablation_no_l3,ablation_large_chunk",
        help="Comma-separated conditions.",
    )
    parser.add_argument("--base-url", default="https://dashscope.aliyuncs.com/compatible-mode/v1")
    parser.add_argument("--model", default="qwen-vl-max")
    # Security-only change: keep the experiment identical while sourcing the same Qwen key from CLI/env.
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--max-chars-per-chunk", type=int, default=9000)
    parser.add_argument("--overlap-chars", type=int, default=900)
    parser.add_argument("--max-images-per-chunk", type=int, default=4)
    parser.add_argument("--request-timeout", type=int, default=600)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--oneshot-max-text-chars", type=int, default=60000)
    parser.add_argument("--oneshot-max-images", type=int, default=8)
    parser.add_argument("--large-chunk-chars", type=int, default=50000)
    parser.add_argument("--large-chunk-images", type=int, default=8)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    selected = [item.strip() for item in args.conditions.split(",") if item.strip()]
    unknown = sorted(set(selected) - CONDITIONS)
    if unknown:
        raise SystemExit(f"Unknown conditions: {unknown}")
    outputs = {}
    for condition in selected:
        progress(f"Experiment condition started: {condition}", not args.quiet)
        outputs[condition] = str(run_condition(args, condition))
        progress(f"Experiment condition completed: {condition}", not args.quiet)
    summary_path = Path(args.output_root).resolve() / "experiment_outputs.json"
    write_json(summary_path, outputs)
    print(json.dumps({"outputs": outputs, "summary": str(summary_path)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
