from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable


REF_RE = re.compile(
    r"^(?P<chunk>[A-Za-z]?\d+_C\d+):(?P<kind>RP|M)-(?P<local>[A-Za-z0-9][A-Za-z0-9._-]*)$"
)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def stable_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def numeric_key(path: Path) -> tuple[int, str]:
    return (int(path.name), path.name) if path.name.isdigit() else (10**9, path.name)


def discover_papers(root_dir: Path, start: int = 1, end: int | None = None) -> list[Path]:
    papers = [
        path
        for path in root_dir.iterdir()
        if path.is_dir() and (path / "full.md").is_file() and (path / "images").is_dir()
    ]
    papers.sort(key=numeric_key)
    selected = []
    for path in papers:
        if path.name.isdigit():
            index = int(path.name)
            if index < start or (end is not None and index > end):
                continue
        selected.append(path)
    return selected


def outputs_dir(paper_dir: Path) -> Path:
    return paper_dir / "outputs"


def _normalize_local_atom_id(value: str) -> str:
    """Normalize numeric segments without destroying hierarchical IDs such as 5.1-1."""
    parts = re.split(r"([._-])", value)
    return "".join(str(int(part)) if part.isdigit() else part for part in parts)


def parse_ref(value: Any) -> tuple[str, str, str] | None:
    if not isinstance(value, str):
        return None
    match = REF_RE.fullmatch(value.strip())
    if not match:
        return None
    return match.group("chunk"), match.group("kind"), _normalize_local_atom_id(match.group("local"))


def canonical_ref(value: Any) -> str | None:
    parsed = parse_ref(value)
    if parsed is None:
        return None
    chunk, kind, local = parsed
    return f"{chunk}:{kind}-{local}"


def unique_strings(values: Iterable[Any]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value).strip()
        if text and text not in seen:
            seen.add(text)
            result.append(text)
    return result


def normalize_final(data: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    problems = data.get("final_research_problems") or data.get("paper_research_problems") or data.get("research_problems") or []
    methods = data.get("final_methods") or data.get("paper_methods") or data.get("methods") or []
    links = data.get("problem_method_links") or data.get("links") or []
    return {
        "problems": [item for item in problems if isinstance(item, dict)],
        "methods": [item for item in methods if isinstance(item, dict)],
        "links": [item for item in links if isinstance(item, dict)],
    }


def node_id(item: dict[str, Any], kind: str, index: int) -> str:
    value = item.get("id") or item.get("problem_id") or item.get("method_id")
    return str(value).strip() if value else f"{'RP' if kind == 'problem' else 'M'}{index}"


def node_text(item: dict[str, Any]) -> str:
    return str(item.get("problem") or item.get("method") or item.get("claim") or "").strip()


def evidence_refs(item: dict[str, Any]) -> list[str]:
    refs = item.get("evidence_refs") or []
    if isinstance(refs, str):
        refs = [refs]
    return [str(value).strip() for value in refs if isinstance(value, str) and value.strip()]


def condition_result_path(paper_dir: Path, condition: str) -> Path:
    outputs = outputs_dir(paper_dir)
    protected = {
        "original_main": outputs / "04_final_extraction.json",
        "original_baseline_oneshot_mineru": outputs / "comparison_experiments" / "baseline_oneshot_mineru" / "baseline_oneshot_mineru.json",
        "original_ablation_text_only": outputs / "comparison_experiments" / "ablations" / "ablation_text_only" / "04_final_extraction.json",
        "original_ablation_no_l3": outputs / "comparison_experiments" / "ablations" / "ablation_no_l3" / "ablation_no_l3_result.json",
        "original_ablation_large_chunk": outputs / "comparison_experiments" / "ablations" / "ablation_large_chunk" / "04_final_extraction.json",
    }
    if condition in protected:
        return protected[condition]
    base = outputs / "major_revision_additions" / condition
    candidates = [
        base / "04_final_extraction.json",
        base / "result.json",
        base / "final_result.json",
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]
