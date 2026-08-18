from __future__ import annotations

import base64
import json
import mimetypes
import random
import re
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel

from .io_utils import file_sha256, read_json, stable_hash, write_json


T = TypeVar("T", bound=BaseModel)
JSON_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.IGNORECASE | re.DOTALL)
MAX_DATA_URI_SOURCE_BYTES = 20 * 1024 * 1024


def image_data_url(path: Path) -> str:
    size = path.stat().st_size
    if size > MAX_DATA_URI_SOURCE_BYTES:
        raise ValueError(
            f"{path} is {size} bytes. Qwen3.8-Max OpenAI-compatible Base64 input is limited "
            "to 20 MB before encoding; resize this image or provide a public URL."
        )
    mime = mimetypes.guess_type(str(path))[0] or "image/jpeg"
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"


def _json_object(text: str) -> dict[str, Any]:
    candidate = text.strip()
    fenced = JSON_FENCE_RE.match(candidate)
    if fenced:
        candidate = fenced.group(1).strip()
    try:
        value = json.loads(candidate)
    except json.JSONDecodeError:
        start, end = candidate.find("{"), candidate.rfind("}")
        if start < 0 or end <= start:
            raise
        value = json.loads(candidate[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("Qwen3.8-Max response must be one JSON object")
    return value


def _dump_sdk_value(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if isinstance(value, (str, int, float, bool, list, dict)):
        return value
    return str(value)


def _validate_with_conservative_gold_recovery(
    schema: type[T],
    payload: dict[str, Any],
) -> tuple[T, dict[str, Any] | None]:
    """Validate normally, then allow only an auditable invalid-link drop."""
    from .annotation_schema import PaperGoldAnnotation, normalize_incomplete_gold_links

    if schema is not PaperGoldAnnotation:
        return schema.model_validate(payload), None
    normalized, audit = normalize_incomplete_gold_links(payload)
    if not audit.get("action_count"):
        return schema.model_validate(payload), None
    try:
        parsed = schema.model_validate(normalized)
    except Exception:
        # Surface the unmodified provider payload's validation error when the
        # conservative link-only normalization cannot make the full object
        # schema-valid. Other missing node/top-level content is never hidden.
        return schema.model_validate(payload), None
    audit["applied"] = True
    try:
        schema.model_validate(payload)
    except Exception as original_error:
        audit["trigger_validation_error"] = str(original_error)[:3000]
    else:
        audit["trigger_validation_error"] = None
    return parsed, audit


class Qwen38AnnotationClient:
    """Schema-validated Qwen3.8-Max client for reference-annotation generation only."""

    REQUIRED_MODEL = "qwen3.8-max"

    def __init__(
        self,
        api_key: str,
        model: str = REQUIRED_MODEL,
        base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
        max_output_tokens: int = 24000,
        max_retries: int = 5,
        request_timeout: float = 1200.0,
        progress_seconds: float = 30.0,
        cache_dir: Path | None = None,
        dry_run: bool = False,
    ) -> None:
        if model.lower() != self.REQUIRED_MODEL:
            raise ValueError(
                f"Gold generation is locked to {self.REQUIRED_MODEL!r}; received {model!r}. "
                "Do not substitute the Qwen-VL-Max experimental backend here."
            )
        if request_timeout <= 0:
            raise ValueError("request_timeout must be positive")
        if progress_seconds <= 0:
            raise ValueError("progress_seconds must be positive")
        self.client = None
        if not dry_run:
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise RuntimeError("Install dependencies with: pip install -r requirements-revision.txt") from exc
            # Disable SDK-level retries because this class already performs
            # schema-aware retries and reports every attempt explicitly.
            self.client = OpenAI(
                api_key=api_key,
                base_url=base_url,
                timeout=request_timeout,
                max_retries=0,
            )
        self.model = self.REQUIRED_MODEL
        self.base_url = base_url
        self.max_output_tokens = max_output_tokens
        self.max_retries = max_retries
        self.request_timeout = request_timeout
        self.progress_seconds = progress_seconds
        self.cache_dir = cache_dir
        self.dry_run = dry_run

    def parse(
        self,
        stage: str,
        instructions: str,
        user_text: str,
        schema: type[T],
        images: list[Path] | None = None,
        extra: dict[str, Any] | None = None,
        dry_run_value: T | None = None,
    ) -> tuple[T, dict[str, Any]]:
        images = images or []
        image_manifest = [
            {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": file_sha256(path)}
            for path in images
        ]
        cache_key = {
            "stage": stage,
            "model": self.model,
            "enable_thinking": True,
            "response_format": "json_object",
            "instructions": instructions,
            "user_text": user_text,
            "images": image_manifest,
            "schema": schema.model_json_schema(),
            "extra": extra or {},
        }
        cache_path = self.cache_dir / stage / f"{stable_hash(cache_key)}.json" if self.cache_dir else None
        if cache_path and cache_path.exists():
            cached = read_json(cache_path)
            print(f"    {stage}: cache hit", flush=True)
            return schema.model_validate(cached["parsed"]), cached["metadata"]
        raw_answer_path = cache_path.with_suffix(".answer.txt") if cache_path else None
        raw_reasoning_path = cache_path.with_suffix(".reasoning.txt") if cache_path else None
        if raw_answer_path and raw_answer_path.exists():
            try:
                answer = raw_answer_path.read_text(encoding="utf-8")
                reasoning = raw_reasoning_path.read_text(encoding="utf-8") if raw_reasoning_path and raw_reasoning_path.exists() else ""
                parsed, local_normalization = _validate_with_conservative_gold_recovery(
                    schema, _json_object(answer)
                )
                metadata = self._metadata(
                    response_id=None,
                    returned_model=self.model,
                    usage=None,
                    reasoning=reasoning,
                    answer=answer,
                    cache_key=cache_key,
                    images=images,
                    dry_run=False,
                )
                metadata["recovered_from_saved_raw_answer"] = True
                metadata["local_schema_normalization"] = local_normalization
                self._write_cache(cache_path, parsed, metadata)
                print(f"    {stage}: recovered saved raw answer with conservative schema normalization", flush=True)
                return parsed, metadata
            except Exception as exc:
                print(
                    f"    {stage}: saved raw answer is not locally recoverable; a new API response is required: "
                    f"{type(exc).__name__}: {str(exc)[:300]}",
                    flush=True,
                )
        if self.dry_run:
            if dry_run_value is None:
                raise ValueError("dry_run_value is required in dry-run mode")
            metadata = self._metadata(
                response_id=None,
                returned_model=self.model,
                usage=None,
                reasoning="",
                answer=json.dumps(dry_run_value.model_dump(mode="json"), ensure_ascii=False),
                cache_key=cache_key,
                images=images,
                dry_run=True,
            )
            self._write_cache(cache_path, dry_run_value, metadata)
            return dry_run_value, metadata

        schema_instruction = (
            instructions
            + "\n\nReturn exactly one JSON object conforming to this JSON Schema. Do not add Markdown fences or prose:\n"
            + json.dumps(schema.model_json_schema(), ensure_ascii=False)
        )
        content: list[dict[str, Any]] = [{"type": "text", "text": user_text}]
        for path in images:
            content.append({"type": "text", "text": f"The next image has visual_asset_id={path.name}."})
            content.append({"type": "image_url", "image_url": {"url": image_data_url(path)}})

        last_error: Exception | None = None
        correction = ""
        for attempt in range(1, self.max_retries + 1):
            started = time.monotonic()
            progress_state: dict[str, Any] = {
                "phase": "waiting_for_first_stream_chunk",
                "reasoning_chars": 0,
                "answer_chars": 0,
            }
            heartbeat_stop = threading.Event()

            def heartbeat() -> None:
                while not heartbeat_stop.wait(self.progress_seconds):
                    elapsed = int(time.monotonic() - started)
                    print(
                        f"    {stage}: alive elapsed={elapsed}s phase={progress_state['phase']} "
                        f"reasoning_chars={progress_state['reasoning_chars']} "
                        f"answer_chars={progress_state['answer_chars']}",
                        flush=True,
                    )

            print(
                f"    {stage}: request attempt {attempt}/{self.max_retries}, "
                f"images={len(images)}, timeout={self.request_timeout:g}s",
                flush=True,
            )
            heartbeat_thread = threading.Thread(target=heartbeat, daemon=True)
            heartbeat_thread.start()
            try:
                if self.client is None:
                    raise RuntimeError("DashScope client is unavailable outside dry-run mode")
                request_content = list(content)
                if correction:
                    request_content.append({
                        "type": "text",
                        "text": "Previous output failed local JSON/schema validation. Correct it in this fresh response. " + correction,
                    })
                stream = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": schema_instruction},
                        {"role": "user", "content": request_content},
                    ],
                    extra_body={"enable_thinking": True},
                    stream=True,
                    stream_options={"include_usage": True},
                    response_format={"type": "json_object"},
                    max_tokens=self.max_output_tokens,
                )
                reasoning_parts: list[str] = []
                answer_parts: list[str] = []
                response_id = None
                returned_model = None
                usage = None
                for chunk in stream:
                    response_id = response_id or getattr(chunk, "id", None)
                    returned_model = returned_model or getattr(chunk, "model", None)
                    if getattr(chunk, "usage", None) is not None:
                        usage = _dump_sdk_value(chunk.usage)
                    if not getattr(chunk, "choices", None):
                        continue
                    delta = chunk.choices[0].delta
                    reasoning_piece = getattr(delta, "reasoning_content", None)
                    answer_piece = getattr(delta, "content", None)
                    if reasoning_piece:
                        reasoning_parts.append(str(reasoning_piece))
                        progress_state["phase"] = "thinking"
                        progress_state["reasoning_chars"] += len(str(reasoning_piece))
                    if answer_piece:
                        answer_parts.append(str(answer_piece))
                        progress_state["phase"] = "answering"
                        progress_state["answer_chars"] += len(str(answer_piece))
                answer = "".join(answer_parts)
                reasoning = "".join(reasoning_parts)
                if returned_model and str(returned_model).lower() != self.model:
                    raise RuntimeError(
                        f"Requested {self.model!r}, but API returned {returned_model!r}; refusing silent substitution"
                    )
                if cache_path is not None:
                    raw_dir = cache_path.parent
                    raw_dir.mkdir(parents=True, exist_ok=True)
                    cache_stem = cache_path.stem
                    # Save the provider response before Pydantic validation so
                    # an interrupted run can recover a conservatively repairable
                    # incomplete link without paying for the same stage again.
                    (raw_dir / f"{cache_stem}.answer.txt").write_text(answer, encoding="utf-8")
                    (raw_dir / f"{cache_stem}.reasoning.txt").write_text(reasoning, encoding="utf-8")
                parsed, local_normalization = _validate_with_conservative_gold_recovery(
                    schema, _json_object(answer)
                )
                metadata = self._metadata(
                    response_id=response_id,
                    returned_model=returned_model,
                    usage=usage,
                    reasoning=reasoning,
                    answer=answer,
                    cache_key=cache_key,
                    images=images,
                    dry_run=False,
                )
                metadata["local_schema_normalization"] = local_normalization
                self._write_cache(cache_path, parsed, metadata)
                elapsed = int(time.monotonic() - started)
                print(
                    f"    {stage}: completed elapsed={elapsed}s "
                    f"reasoning_chars={len(reasoning)} answer_chars={len(answer)}",
                    flush=True,
                )
                return parsed, metadata
            except Exception as exc:
                last_error = exc
                correction = f"Validation error: {str(exc)[:1500]}"
                elapsed = int(time.monotonic() - started)
                print(
                    f"    {stage}: attempt {attempt} failed after {elapsed}s: "
                    f"{type(exc).__name__}: {str(exc)[:500]}",
                    flush=True,
                )
                if attempt >= self.max_retries:
                    break
                time.sleep(min(60.0, (2 ** (attempt - 1)) + random.random()))
            finally:
                heartbeat_stop.set()
                heartbeat_thread.join(timeout=1.0)
        raise RuntimeError(f"Qwen3.8-Max structured request failed after {self.max_retries} attempts: {last_error}")

    def _metadata(
        self,
        *,
        response_id: str | None,
        returned_model: str | None,
        usage: Any,
        reasoning: str,
        answer: str,
        cache_key: dict[str, Any],
        images: list[Path],
        dry_run: bool,
    ) -> dict[str, Any]:
        return {
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "response_id": response_id,
            "provider": "Alibaba Cloud Model Studio (DashScope OpenAI-compatible API)",
            "model_requested": self.model,
            "model_returned": returned_model,
            "base_url": self.base_url,
            "enable_thinking": True,
            "response_format": "json_object",
            "stream": True,
            "request_timeout_seconds": self.request_timeout,
            "progress_seconds": self.progress_seconds,
            "run_hash": stable_hash(cache_key),
            "image_ids": [path.name for path in images],
            "reasoning_char_count": len(reasoning),
            "reasoning_sha256": stable_hash(reasoning),
            "answer_char_count": len(answer),
            "answer_sha256": stable_hash(answer),
            "usage": usage,
            "dry_run": dry_run,
        }

    @staticmethod
    def _write_cache(cache_path: Path | None, parsed: BaseModel, metadata: dict[str, Any]) -> None:
        if cache_path is not None:
            write_json(cache_path, {"parsed": parsed.model_dump(mode="json"), "metadata": metadata})
