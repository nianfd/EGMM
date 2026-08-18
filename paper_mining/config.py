from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PipelineConfig:
    paper_dir: Path
    markdown_path: Path
    image_dir: Path
    output_dir: Path
    cache_dir: Path
    api_key: str
    base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    model: str = "qwen3-vl-plus"
    max_chars_per_chunk: int = 9000
    overlap_chars: int = 900
    max_images_per_chunk: int = 4
    request_timeout: int = 600
    max_retries: int = 5
    max_tokens: int = 8192
    temperature: float = 0.1
    dry_run: bool = False
    verbose: bool = True
    enable_relation_completion: bool = True

    @classmethod
    def from_args(cls, args: object) -> "PipelineConfig":
        paper_dir = Path(getattr(args, "paper_dir")).resolve()
        markdown_arg = getattr(args, "markdown", None) or (paper_dir / "full.md")
        image_arg = getattr(args, "images", None) or (paper_dir / "images")
        output_arg = getattr(args, "output_dir", None) or (paper_dir / "outputs")
        markdown_path = Path(markdown_arg).resolve()
        image_dir = Path(image_arg).resolve()
        output_dir = Path(output_arg).resolve()
        cache_arg = getattr(args, "cache_dir", None) or (output_dir / "cache")
        cache_dir = Path(cache_arg).resolve()
        # Security-only hardening: same Qwen backend, model and parameters; no OpenAI-key fallback.
        api_key = (
            getattr(args, "api_key", None)
            or os.getenv("DASHSCOPE_API_KEY")
            or os.getenv("QWEN_API_KEY")
            or ""
        )
        dry_run = bool(getattr(args, "dry_run", False))
        if not api_key and not dry_run:
            raise ValueError(
                "Missing API key. Set DASHSCOPE_API_KEY or QWEN_API_KEY, "
                "or pass --api-key. Do not hard-code keys in source files."
            )
        if not markdown_path.exists():
            raise FileNotFoundError(f"Markdown file not found: {markdown_path}")
        if not image_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {image_dir}")
        return cls(
            paper_dir=paper_dir,
            markdown_path=markdown_path,
            image_dir=image_dir,
            output_dir=output_dir,
            cache_dir=cache_dir,
            api_key=api_key,
            base_url=getattr(args, "base_url", cls.base_url),
            model=getattr(args, "model", cls.model),
            max_chars_per_chunk=int(getattr(args, "max_chars_per_chunk", cls.max_chars_per_chunk)),
            overlap_chars=int(getattr(args, "overlap_chars", cls.overlap_chars)),
            max_images_per_chunk=int(getattr(args, "max_images_per_chunk", cls.max_images_per_chunk)),
            request_timeout=int(getattr(args, "request_timeout", cls.request_timeout)),
            max_retries=int(getattr(args, "max_retries", cls.max_retries)),
            max_tokens=int(getattr(args, "max_tokens", cls.max_tokens)),
            temperature=float(getattr(args, "temperature", cls.temperature)),
            dry_run=dry_run,
            verbose=not bool(getattr(args, "quiet", False)),
            enable_relation_completion=not bool(getattr(args, "skip_relation_completion", False)),
        )
