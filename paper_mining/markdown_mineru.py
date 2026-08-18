from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
IMAGE_MD_RE = re.compile(r"!\[[^\]]*]\(([^)]+)\)")
IMAGE_HTML_RE = re.compile(r"<img\b[^>]*\bsrc=[\"']([^\"']+)[\"'][^>]*>", re.IGNORECASE)


@dataclass
class Section:
    id: str
    title: str
    level: int
    text: str
    start_line: int
    image_refs: list[str] = field(default_factory=list)


@dataclass
class Chunk:
    id: str
    section_title: str
    section_level: int
    text: str
    start_line: int
    image_paths: list[Path]


def parse_sections(markdown: str) -> list[Section]:
    lines = markdown.splitlines()
    sections: list[Section] = []
    current_title = "Front Matter"
    current_level = 0
    current_start = 1
    buffer: list[str] = []

    def flush() -> None:
        if not buffer and sections:
            return
        text = "\n".join(buffer).strip()
        image_refs = extract_image_refs(text)
        sec_id = f"S{len(sections) + 1:03d}"
        sections.append(
            Section(
                id=sec_id,
                title=current_title,
                level=current_level,
                text=text,
                start_line=current_start,
                image_refs=image_refs,
            )
        )

    for idx, line in enumerate(lines, start=1):
        match = HEADING_RE.match(line)
        if match:
            flush()
            current_level = len(match.group(1))
            current_title = match.group(2).strip()
            current_start = idx
            buffer = [line]
        else:
            buffer.append(line)
    flush()
    return [s for s in sections if s.text]


def extract_image_refs(text: str) -> list[str]:
    refs = IMAGE_MD_RE.findall(text) + IMAGE_HTML_RE.findall(text)
    cleaned: list[str] = []
    for ref in refs:
        ref = ref.strip().strip("<>")
        if ref and ref not in cleaned:
            cleaned.append(ref)
    return cleaned


def make_chunks(
    sections: Iterable[Section],
    image_dir: Path,
    max_chars: int,
    overlap_chars: int,
    max_images: int,
) -> list[Chunk]:
    all_images = discover_images(image_dir)
    chunks: list[Chunk] = []
    for section in sections:
        pieces = split_with_overlap(section.text, max_chars=max_chars, overlap_chars=overlap_chars)
        section_images = resolve_image_refs(section.image_refs, image_dir, all_images)
        if not section_images:
            section_images = infer_images_from_text(section.text, all_images)
        for idx, piece in enumerate(pieces, start=1):
            chunk_id = f"{section.id}_C{idx:02d}"
            chunks.append(
                Chunk(
                    id=chunk_id,
                    section_title=section.title,
                    section_level=section.level,
                    text=piece,
                    start_line=section.start_line,
                    image_paths=section_images[:max_images],
                )
            )
    return chunks


def split_with_overlap(text: str, max_chars: int, overlap_chars: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    paragraphs = re.split(r"\n\s*\n", text)
    chunks: list[str] = []
    current = ""
    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        if not current:
            current = paragraph
        elif len(current) + len(paragraph) + 2 <= max_chars:
            current = f"{current}\n\n{paragraph}"
        else:
            chunks.append(current)
            overlap = current[-overlap_chars:] if overlap_chars > 0 else ""
            current = f"{overlap}\n\n{paragraph}" if overlap else paragraph
            while len(current) > max_chars:
                chunks.append(current[:max_chars])
                start = max(0, max_chars - overlap_chars)
                current = current[start:]
    if current:
        chunks.append(current)
    return chunks


def discover_images(image_dir: Path) -> list[Path]:
    if not image_dir.exists():
        return []
    suffixes = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    return sorted(p for p in image_dir.rglob("*") if p.is_file() and p.suffix.lower() in suffixes)


def resolve_image_refs(refs: list[str], image_dir: Path, all_images: list[Path]) -> list[Path]:
    resolved: list[Path] = []
    by_name = {p.name.lower(): p for p in all_images}
    for ref in refs:
        name = os.path.basename(ref.split("#", 1)[0].split("?", 1)[0]).lower()
        candidates = []
        direct = (image_dir / ref).resolve()
        if direct.exists():
            candidates.append(direct)
        if name in by_name:
            candidates.append(by_name[name])
        for candidate in candidates:
            if candidate not in resolved:
                resolved.append(candidate)
    return resolved


def infer_images_from_text(text: str, all_images: list[Path]) -> list[Path]:
    lower = text.lower()
    hits: list[Path] = []
    for image in all_images:
        stem = image.stem.lower()
        if stem and stem in lower:
            hits.append(image)
    return hits
