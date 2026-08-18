from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from paper_mining.io_utils import read_json, stable_hash, write_json
from paper_mining.progress import progress
from paper_mining.qwenvl_client import QwenVLClient, repair_common_json_issues


_INVALID_ESCAPE = re.compile(r'\\(?!["\\/bfnrt]|u[0-9a-fA-F]{4})')


class ResilientQwenVLClient(QwenVLClient):
    """Revision-only Qwen-VL client with restartable JSON recovery.

    Cache keys and request payloads intentionally match the protected original
    client.  The only behavioral extension is that a malformed answer is saved,
    complete JSON objects are conservatively salvaged without inventing missing
    content, and every recovery is audited beside the normal cache file.
    """

    json_repair_attempts: int = 3

    def cache_path_for(
        self,
        stage: str,
        system_prompt: str,
        user_text: str,
        images: list[Path] | None = None,
        extra_cache_key: dict[str, Any] | None = None,
    ) -> Path:
        images = images or []
        cache_payload = {
            "stage": stage,
            "model": self.model,
            "system": system_prompt,
            "user": user_text,
            "images": [str(path.resolve()) for path in images],
            "extra": extra_cache_key or {},
        }
        return self.cache_dir / stage / f"{stable_hash(cache_payload)}.json"

    def chat_json(
        self,
        stage: str,
        system_prompt: str,
        user_text: str,
        images: list[Path] | None = None,
        extra_cache_key: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        images = images or []
        cache_path = self.cache_path_for(
            stage, system_prompt, user_text, images, extra_cache_key,
        )
        if cache_path.exists():
            progress(f"{stage}: cache hit -> {cache_path.name}", self.verbose)
            return read_json(cache_path)
        if self.dry_run:
            progress(f"{stage}: dry-run response generated", self.verbose)
            data = self._dry_run_response(stage, user_text, images)
            write_json(cache_path, data)
            return data

        raw_path = cache_path.with_suffix(".raw.txt")
        if raw_path.exists():
            progress(
                f"{stage}: recovering previously saved malformed answer -> {raw_path.name}",
                self.verbose,
            )
            return self._recover_and_cache(stage, raw_path.read_text(encoding="utf-8", errors="replace"), cache_path)

        progress(f"{stage}: API request, images={len(images)}, cache={cache_path.name}", self.verbose)
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": self._multimodal_content(user_text, images)},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "response_format": {"type": "json_object"},
        }
        raw = self._post_json("/chat/completions", payload)
        content = raw["choices"][0]["message"]["content"]
        try:
            parsed = self._parse_json_content(content)
        except (json.JSONDecodeError, ValueError) as exc:
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            raw_path.write_text(content, encoding="utf-8", errors="replace")
            progress(
                f"{stage}: model returned invalid JSON ({exc}); raw response saved -> {raw_path.name}",
                self.verbose,
            )
            return self._recover_and_cache(stage, content, cache_path)
        write_json(cache_path, parsed)
        return parsed

    def _recover_and_cache(self, stage: str, broken: str, cache_path: Path) -> dict[str, Any]:
        # First try deterministic recovery. It keeps only complete model-emitted
        # objects and never guesses a truncated field or creates a claim.
        parsed = self._local_recovery(stage, broken)
        if parsed is not None:
            self._save_recovery(cache_path, parsed, "local_complete_object_salvage", 0, broken)
            progress(
                f"{stage}: recovered saved answer locally; no extraction request repeated",
                self.verbose,
            )
            return parsed

        # Reuse any completed repair response left by an interrupted rerun.
        for attempt in range(1, self.json_repair_attempts + 1):
            repair_path = cache_path.with_suffix(f".repair{attempt:02d}.raw.txt")
            if not repair_path.exists():
                continue
            repaired = repair_path.read_text(encoding="utf-8", errors="replace")
            parsed = self._parse_or_salvage(stage, repaired)
            if parsed is not None:
                self._save_recovery(cache_path, parsed, "saved_model_syntax_repair", attempt, broken)
                progress(f"{stage}: recovered from saved repair attempt {attempt}", self.verbose)
                return parsed

        for attempt in range(1, self.json_repair_attempts + 1):
            repair_path = cache_path.with_suffix(f".repair{attempt:02d}.raw.txt")
            if repair_path.exists():
                continue
            progress(
                f"{stage}: requesting syntax-only JSON repair {attempt}/{self.json_repair_attempts}",
                self.verbose,
            )
            repair_payload = {
                "model": self.model,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "Repair malformed JSON syntax. Return exactly one minified JSON object. "
                            "Do not add, infer, complete, paraphrase, or summarize any scientific content. "
                            "If the tail is truncated, discard only the incomplete tail item and close its arrays/object."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            "Repair syntax only. Preserve every complete key/value and emit no Markdown.\n\n"
                            + broken
                        ),
                    },
                ],
                "temperature": 0,
                "max_tokens": self.max_tokens,
                "response_format": {"type": "json_object"},
            }
            repaired_raw = self._post_json("/chat/completions", repair_payload)
            repaired = repaired_raw["choices"][0]["message"]["content"]
            repair_path.parent.mkdir(parents=True, exist_ok=True)
            repair_path.write_text(repaired, encoding="utf-8", errors="replace")
            parsed = self._parse_or_salvage(stage, repaired)
            if parsed is not None:
                self._save_recovery(cache_path, parsed, "model_syntax_repair", attempt, broken)
                progress(f"{stage}: JSON repair attempt {attempt} succeeded", self.verbose)
                return parsed

        raise RuntimeError(
            f"{stage}: malformed JSON could not be conservatively recovered after "
            f"{self.json_repair_attempts} syntax-only attempts; raw and repair files were preserved"
        )

    def _parse_or_salvage(self, stage: str, content: str) -> dict[str, Any] | None:
        try:
            return self._parse_json_content(content)
        except (json.JSONDecodeError, ValueError):
            return self._local_recovery(stage, content)

    def _local_recovery(self, stage: str, content: str) -> dict[str, Any] | None:
        try:
            heuristic = repair_common_json_issues(content)
            return self._parse_json_content(heuristic)
        except (json.JSONDecodeError, ValueError):
            pass

        if stage == "l1_chunk_extract":
            keys = ("research_problem_atoms", "method_atoms")
            result: dict[str, Any] = {
                "research_problem_atoms": [],
                "method_atoms": [],
                "cross_modal_notes": [],
            }
        elif "compact_plan" in stage:
            keys = ("problem_groups", "method_groups", "links")
            result = {
                "problem_groups": [],
                "method_groups": [],
                "links": [],
                "unresolved_refs": [],
            }
        elif stage.startswith("l2_paper_merge"):
            keys = ("paper_research_problems", "paper_methods", "problem_method_links")
            result = {
                "paper_research_problems": [],
                "paper_methods": [],
                "problem_method_links": [],
                "unresolved_or_ambiguous": [],
            }
        elif stage == "relation_completion":
            keys = ("inferred_problem_method_links",)
            result = {"inferred_problem_method_links": []}
        else:
            return None

        recognized = 0
        complete_items = 0
        for key in keys:
            found, items = _extract_complete_object_items(content, key)
            if found:
                recognized += 1
                complete_items += len(items)
                result[key] = items
        # A recognized empty array is a legitimate response. If the only
        # recognized array is truncated before its first complete item, require
        # syntax-only model repair instead of silently returning an empty graph.
        if recognized and (complete_items or _has_explicit_empty_array(content, keys)):
            return result
        return None

    @staticmethod
    def _save_recovery(
        cache_path: Path,
        parsed: dict[str, Any],
        method: str,
        attempt: int,
        broken: str,
    ) -> None:
        write_json(cache_path, parsed)
        write_json(cache_path.with_suffix(".recovery.json"), {
            "recovered": True,
            "method": method,
            "repair_attempt": attempt,
            "source_character_count": len(broken),
            "non_inventive_policy": (
                "Only valid parsed content or complete model-emitted array objects were retained; "
                "incomplete tail items were discarded and no scientific field was guessed."
            ),
        })


def _has_explicit_empty_array(content: str, keys: tuple[str, ...]) -> bool:
    return any(re.search(rf'"{re.escape(key)}"\s*:\s*\[\s*\]', content) for key in keys)


def _loads_object(candidate: str) -> dict[str, Any] | None:
    cleaned = _INVALID_ESCAPE.sub(r"\\\\", candidate).replace("\x00", "")
    cleaned = re.sub(r",\s*([}\]])", r"\1", cleaned)
    try:
        value = json.loads(cleaned, strict=False)
    except (json.JSONDecodeError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def _extract_complete_object_items(content: str, key: str) -> tuple[bool, list[dict[str, Any]]]:
    match = re.search(rf'"{re.escape(key)}"\s*:\s*\[', content)
    if not match:
        return False, []
    items: list[dict[str, Any]] = []
    array_depth = 1
    object_depth = 0
    object_start: int | None = None
    in_string = False
    escaped = False
    index = match.end()
    while index < len(content):
        char = content[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            index += 1
            continue
        if char == '"':
            in_string = True
        elif char == "[":
            array_depth += 1
        elif char == "]":
            array_depth -= 1
            if array_depth == 0:
                break
        elif char == "{":
            if object_depth == 0 and array_depth == 1:
                object_start = index
            object_depth += 1
        elif char == "}" and object_depth:
            object_depth -= 1
            if object_depth == 0 and object_start is not None:
                value = _loads_object(content[object_start : index + 1])
                if value is not None:
                    items.append(value)
                object_start = None
        index += 1
    return True, items
