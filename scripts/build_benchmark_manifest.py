from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from revision.io_utils import discover_papers, file_sha256, write_csv, write_json


def title_from_markdown(path: Path) -> str:
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        title = re.sub(r"^#+\s*", "", line).strip()
        if title and not title.startswith("!"):
            return title
    return ""


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a release-ready benchmark identifier/parsing manifest without inventing missing metadata.")
    parser.add_argument("--root-dir", required=True)
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--end", type=int, default=100)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sampling-seed", help="Exact seed used during original sampling; omit if unknown")
    parser.add_argument("--sampling-frame", default="TO_BE_COMPLETED_BY_AUTHORS")
    parser.add_argument("--inclusion-criteria", default="TO_BE_COMPLETED_BY_AUTHORS")
    parser.add_argument("--exclusion-criteria", default="TO_BE_COMPLETED_BY_AUTHORS")
    args = parser.parse_args()

    root = Path(args.root_dir).resolve()
    rows = []
    for paper in discover_papers(root, args.start, args.end):
        markdown = paper / "full.md"
        pdfs = sorted(paper.glob("*_origin.pdf"))
        images = [path for path in (paper / "images").rglob("*") if path.is_file()]
        rows.append({
            "paper_id": paper.name,
            "title_from_mineru": title_from_markdown(markdown),
            "venue": "TO_BE_COMPLETED_BY_AUTHORS",
            "year": "TO_BE_COMPLETED_BY_AUTHORS",
            "official_paper_url_or_doi": "TO_BE_COMPLETED_BY_AUTHORS",
            "source_pdf_sha256": file_sha256(pdfs[0]) if pdfs else "MISSING",
            "markdown_sha256": file_sha256(markdown),
            "markdown_chars": len(markdown.read_text(encoding="utf-8", errors="replace")),
            "parsed_image_count": len(images),
            "mineru_parse_status": "success" if markdown.stat().st_size and images else "partial_or_failed",
            "exclusion_reason": "",
        })
    output_dir = Path(args.output_dir).resolve()
    write_csv(output_dir / "benchmark_manifest.csv", rows)
    success = sum(row["mineru_parse_status"] == "success" for row in rows)
    protocol = {
        "paper_count": len(rows),
        "mineru_parse_success_count": success,
        "mineru_partial_or_failure_count": len(rows) - success,
        "sampling_seed": args.sampling_seed or "UNKNOWN_MUST_NOT_BE_INVENTED",
        "sampling_frame": args.sampling_frame,
        "inclusion_criteria": args.inclusion_criteria,
        "exclusion_criteria": args.exclusion_criteria,
        "author_action_required": "Fill all TO_BE_COMPLETED_BY_AUTHORS/UNKNOWN fields from the actual sampling records before manuscript submission.",
    }
    write_json(output_dir / "benchmark_protocol.json", protocol)
    print(json.dumps(protocol, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
