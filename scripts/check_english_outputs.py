"""Check whether experiment JSON outputs contain Chinese characters."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")


def contains_cjk(text: str) -> bool:
    return bool(CJK_RE.search(text))


def preview(text: str, width: int = 120) -> str:
    match = CJK_RE.search(text)
    if not match:
        return ""
    start = max(0, match.start() - 40)
    end = min(len(text), match.end() + width)
    return text[start:end].replace("\n", " ")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root-dir", required=True, help="Root directory to scan.")
    parser.add_argument(
        "--pattern",
        default="*.json",
        help="Glob pattern under root-dir. Default: *.json",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=50,
        help="Maximum number of files to print.",
    )
    args = parser.parse_args()

    root = Path(args.root_dir).expanduser().resolve()
    bad: list[tuple[Path, str]] = []
    for path in root.rglob(args.pattern):
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            text = path.read_text(encoding="utf-8", errors="ignore")
        if contains_cjk(text):
            bad.append((path, preview(text)))

    print(f"Scanned root: {root}")
    print(f"Files containing Chinese characters: {len(bad)}")
    for path, snippet in bad[: args.max_files]:
        print(f"- {path}")
        print(f"  preview: {snippet}")
    if len(bad) > args.max_files:
        print(f"... {len(bad) - args.max_files} more files omitted")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
