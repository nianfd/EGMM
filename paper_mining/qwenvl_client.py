from __future__ import annotations

import base64
import json
import re
import mimetypes
import random
import re
import time
import urllib.error
import urllib.request


_INVALID_JSON_ESCAPE_RE = re.compile(r'\\(?!["\\/bfnrt]|u[0-9a-fA-F]{4})')


def _loads_json_lenient(content: str):
    """Parse JSON while tolerating invalid backslash escapes from model output."""
    try:
        return json.loads(content)
    except json.JSONDecodeError as original_error:
        repaired = _INVALID_JSON_ESCAPE_RE.sub(r"\\\\", content)
        if repaired != content:
            try:
                return json.loads(repaired)
            except json.JSONDecodeError:
                pass
        raise original_error
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .io_utils import read_json, stable_hash, write_json
from .progress import progress


@dataclass
class QwenVLClient:
    api_key: str
    base_url: str
    model: str
    cache_dir: Path
    timeout: int = 600
    max_retries: int = 5
    max_tokens: int = 8192
    temperature: float = 0.1
    dry_run: bool = False
    verbose: bool = True

    def chat_json(
        self,
        stage: str,
        system_prompt: str,
        user_text: str,
        images: list[Path] | None = None,
        extra_cache_key: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        images = images or []
        cache_payload = {
            "stage": stage,
            "model": self.model,
            "system": system_prompt,
            "user": user_text,
            "images": [str(p.resolve()) for p in images],
            "extra": extra_cache_key or {},
        }
        cache_path = self.cache_dir / stage / f"{stable_hash(cache_payload)}.json"
        if cache_path.exists():
            progress(f"{stage}: cache hit -> {cache_path.name}", self.verbose)
            return read_json(cache_path)
        if self.dry_run:
            progress(f"{stage}: dry-run response generated", self.verbose)
            data = self._dry_run_response(stage, user_text, images)
            write_json(cache_path, data)
            return data

        progress(f"{stage}: API request, images={len(images)}, cache={cache_path.name}", self.verbose)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": self._multimodal_content(user_text, images)},
        ]
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "response_format": {"type": "json_object"},
        }
        raw = self._post_json("/chat/completions", payload)
        content = raw["choices"][0]["message"]["content"]
        try:
            parsed = self._parse_json_content(content)
        except json.JSONDecodeError as exc:
            raw_path = cache_path.with_suffix(".raw.txt")
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            raw_path.write_text(content, encoding="utf-8", errors="replace")
            progress(
                f"{stage}: model returned invalid JSON ({exc}); raw response saved -> {raw_path.name}",
                self.verbose,
            )
            parsed = self._repair_json_content(stage, content)
        write_json(cache_path, parsed)
        return parsed

    def _post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        url = self.base_url.rstrip("/") + path
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            request = urllib.request.Request(url, data=body, headers=headers, method="POST")
            try:
                progress(f"Qwen-VL request attempt {attempt + 1}/{self.max_retries}", self.verbose)
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    parsed = json.loads(response.read().decode("utf-8"))
                    progress("Qwen-VL request succeeded", self.verbose)
                    return parsed
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")
                last_error = RuntimeError(f"HTTP {exc.code}: {detail}")
                progress(f"Qwen-VL request failed with HTTP {exc.code}", self.verbose)
                if exc.code < 500 and exc.code not in {408, 409, 429}:
                    raise last_error
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
                last_error = exc
                progress(f"Qwen-VL request failed: {exc}", self.verbose)
            sleep_seconds = min(60.0, (2**attempt) + random.random())
            progress(f"Retrying after {sleep_seconds:.1f}s", self.verbose)
            time.sleep(sleep_seconds)
        raise RuntimeError(f"QwenVL request failed after retries: {last_error}")

    def _multimodal_content(self, text: str, images: list[Path]) -> list[dict[str, Any]]:
        content: list[dict[str, Any]] = [{"type": "text", "text": text}]
        for image in images:
            content.append({"type": "image_url", "image_url": {"url": image_to_data_url(image)}})
        return content

    @staticmethod
    def _parse_json_content(content: str) -> dict[str, Any]:
        content = strip_markdown_fences(content)
        try:
            data = _loads_json_lenient(content)
        except json.JSONDecodeError:
            start = content.find("{")
            end = content.rfind("}")
            if start == -1 or end == -1 or end <= start:
                raise
            candidate = repair_common_json_issues(content[start : end + 1])
            data = _loads_json_lenient(candidate)
        if not isinstance(data, dict):
            raise ValueError("Model returned JSON that is not an object.")
        return data

    def _repair_json_content(self, stage: str, broken_content: str) -> dict[str, Any]:
        heuristic = repair_common_json_issues(strip_markdown_fences(broken_content))
        try:
            return self._parse_json_content(heuristic)
        except json.JSONDecodeError:
            pass

        progress(f"{stage}: requesting JSON repair from model", self.verbose)
        repair_payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You repair malformed JSON. Return only one valid JSON object. "
                        "Do not add new facts, do not summarize, and do not include Markdown fences."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "The following content is intended to be a JSON object but is malformed. "
                        "Repair only syntax errors such as missing commas, trailing commas, bad fences, "
                        "or unescaped line breaks. Preserve all keys and values as much as possible.\n\n"
                        f"{broken_content}"
                    ),
                },
            ],
            "temperature": 0,
            "max_tokens": self.max_tokens,
            "response_format": {"type": "json_object"},
        }
        repaired_raw = self._post_json("/chat/completions", repair_payload)
        repaired_content = repaired_raw["choices"][0]["message"]["content"]
        return self._parse_json_content(repaired_content)

    @staticmethod
    def _dry_run_response(stage: str, user_text: str, images: list[Path]) -> dict[str, Any]:
        return {
            "stage": stage,
            "dry_run": True,
            "note": "No API call was made. Set DASHSCOPE_API_KEY/QWEN_API_KEY and remove --dry-run.",
            "image_count": len(images),
            "text_preview": user_text[:600],
        }


def image_to_data_url(path: Path) -> str:
    mime = mimetypes.guess_type(str(path))[0] or "image/png"
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{data}"


def strip_markdown_fences(content: str) -> str:
    text = content.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def repair_common_json_issues(content: str) -> str:
    text = content.strip()
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        text = text[start : end + 1]
    text = re.sub(r",\s*([}\]])", r"\1", text)
    text = text.replace("\u0000", "")
    return text
