from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

from paper_mining.config import PipelineConfig


def build_qwen_config(
    paper_dir: Path,
    output_dir: Path,
    *,
    api_key: str,
    base_url: str,
    model: str,
    max_chars_per_chunk: int,
    overlap_chars: int,
    max_images_per_chunk: int,
    request_timeout: int,
    max_retries: int,
    max_tokens: int,
    temperature: float,
    dry_run: bool,
    quiet: bool,
) -> PipelineConfig:
    if model.lower() != "qwen-vl-max":
        raise ValueError(
            f"Major-revision extraction experiments are locked to 'qwen-vl-max'; received {model!r}. "
            "This matches the frozen original-paper experimental backend. "
            "Qwen3.8-Max is reserved exclusively for reference-gold generation."
        )
    args = SimpleNamespace(
        paper_dir=str(paper_dir),
        markdown=str(paper_dir / "full.md"),
        images=str(paper_dir / "images"),
        output_dir=str(output_dir),
        cache_dir=str(output_dir / "cache"),
        api_key=api_key,
        base_url=base_url,
        model=model,
        max_chars_per_chunk=max_chars_per_chunk,
        overlap_chars=overlap_chars,
        max_images_per_chunk=max_images_per_chunk,
        request_timeout=request_timeout,
        max_retries=max_retries,
        max_tokens=max_tokens,
        temperature=temperature,
        dry_run=dry_run,
        quiet=quiet,
        # Preserve the original main-method default for controlled Qwen-VL additions.
        skip_relation_completion=False,
    )
    return PipelineConfig.from_args(args)


def qwen_key(explicit: str | None = None) -> str:
    return explicit or os.getenv("DASHSCOPE_API_KEY") or os.getenv("QWEN_API_KEY") or ""
