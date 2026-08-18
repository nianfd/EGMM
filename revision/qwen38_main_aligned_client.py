from __future__ import annotations

import random
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from paper_mining.progress import progress
from paper_mining.qwenvl_client import QwenVLClient


@dataclass
class Qwen38MainAlignedClient(QwenVLClient):
    """Qwen3.8-Max transport adapter for the unchanged main-method client contract.

    ``QwenVLClient.chat_json`` remains the implementation that constructs the
    system/user messages, multimodal content, cache key, JSON repair request,
    temperature, and maximum-output-token settings.  This subclass replaces
    only the HTTP transport so that Qwen3.8-Max thinking output can be consumed
    through DashScope's OpenAI-compatible streaming API.
    """

    progress_seconds: float = 30.0
    call_metadata: list[dict[str, Any]] = field(default_factory=list, init=False)
    _sdk_client: Any = field(default=None, init=False, repr=False)

    REQUIRED_MODEL = "qwen3.8-max"

    def __post_init__(self) -> None:
        if self.model.lower() != self.REQUIRED_MODEL:
            raise ValueError(
                f"The main-aligned reference backend is locked to {self.REQUIRED_MODEL!r}; "
                f"received {self.model!r}."
            )
        if self.timeout <= 0:
            raise ValueError("timeout must be positive")
        if self.progress_seconds <= 0:
            raise ValueError("progress_seconds must be positive")
        if self.dry_run:
            return
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "Install dependencies with: pip install -r requirements-revision.txt"
            ) from exc
        self._sdk_client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout,
            max_retries=0,
        )

    def _post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Execute the payload assembled by the original Qwen client.

        The messages and decoding values in ``payload`` are not rewritten.
        ``enable_thinking`` and streaming are provider-transport controls, not
        task-prompt changes.
        """
        if path != "/chat/completions":
            raise ValueError(f"Unsupported OpenAI-compatible path: {path}")
        if self._sdk_client is None:
            raise RuntimeError("DashScope client is unavailable outside dry-run mode")

        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            started = time.monotonic()
            state: dict[str, Any] = {
                "phase": "waiting_for_first_stream_chunk",
                "reasoning_chars": 0,
                "answer_chars": 0,
            }
            heartbeat_stop = threading.Event()

            def heartbeat() -> None:
                while not heartbeat_stop.wait(self.progress_seconds):
                    elapsed = int(time.monotonic() - started)
                    progress(
                        f"Qwen3.8-Max alive elapsed={elapsed}s phase={state['phase']} "
                        f"reasoning_chars={state['reasoning_chars']} "
                        f"answer_chars={state['answer_chars']}",
                        self.verbose,
                    )

            progress(
                f"Qwen3.8-Max request attempt {attempt}/{self.max_retries}",
                self.verbose,
            )
            heartbeat_thread = threading.Thread(target=heartbeat, daemon=True)
            heartbeat_thread.start()
            try:
                stream = self._sdk_client.chat.completions.create(
                    model=payload["model"],
                    messages=payload["messages"],
                    temperature=payload.get("temperature", self.temperature),
                    max_tokens=payload.get("max_tokens", self.max_tokens),
                    response_format=payload.get("response_format", {"type": "json_object"}),
                    extra_body={"enable_thinking": True},
                    stream=True,
                    stream_options={"include_usage": True},
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
                        raw_usage = chunk.usage
                        usage = raw_usage.model_dump() if hasattr(raw_usage, "model_dump") else str(raw_usage)
                    choices = getattr(chunk, "choices", None) or []
                    if not choices:
                        continue
                    delta = choices[0].delta
                    reasoning = getattr(delta, "reasoning_content", None)
                    content = getattr(delta, "content", None)
                    if reasoning:
                        state["phase"] = "thinking"
                        reasoning_parts.append(reasoning)
                        state["reasoning_chars"] += len(reasoning)
                    if content:
                        state["phase"] = "answering"
                        answer_parts.append(content)
                        state["answer_chars"] += len(content)

                answer = "".join(answer_parts)
                reasoning = "".join(reasoning_parts)
                if not answer.strip():
                    raise RuntimeError("Qwen3.8-Max returned an empty answer")
                elapsed = int(time.monotonic() - started)
                self.call_metadata.append({
                    "response_id": response_id,
                    "model_requested": payload["model"],
                    "model_returned": returned_model,
                    "enable_thinking": True,
                    "stream": True,
                    "temperature": payload.get("temperature", self.temperature),
                    "max_tokens": payload.get("max_tokens", self.max_tokens),
                    "reasoning_char_count": len(reasoning),
                    "answer_char_count": len(answer),
                    "usage": usage,
                    "elapsed_seconds": elapsed,
                })
                progress("Qwen3.8-Max request succeeded", self.verbose)
                return {
                    "id": response_id,
                    "model": returned_model or payload["model"],
                    "choices": [{"message": {"role": "assistant", "content": answer}}],
                    "usage": usage,
                }
            except Exception as exc:
                last_error = exc
                elapsed = int(time.monotonic() - started)
                progress(
                    f"Qwen3.8-Max request attempt {attempt} failed after {elapsed}s: "
                    f"{type(exc).__name__}: {str(exc)[:500]}",
                    self.verbose,
                )
                if attempt < self.max_retries:
                    delay = min(60.0, (2 ** (attempt - 1)) + random.random())
                    progress(f"Retrying after {delay:.1f}s", self.verbose)
                    time.sleep(delay)
            finally:
                heartbeat_stop.set()
                heartbeat_thread.join(timeout=1.0)
        raise RuntimeError(f"Qwen3.8-Max request failed after retries: {last_error}")
