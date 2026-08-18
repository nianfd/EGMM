"""Sequentially process multiple MinerU-parsed paper folders.

Example:
    python scripts/batch_process_papers.py --root-dir C:/nfdproject/nlpproject/data/output

Each child directory under --root-dir is expected to contain full.md and images/.
The script calls the existing single-paper CLI:
    python -m paper_mining.cli --paper-dir <child_dir>
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


def log(message: str) -> None:
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{stamp}] {message}", flush=True)


def natural_key(path: Path) -> tuple[int, str]:
    name = path.name
    return (int(name), name) if name.isdigit() else (10**9, name)


def has_required_inputs(paper_dir: Path) -> bool:
    return (paper_dir / "full.md").is_file() and (paper_dir / "images").is_dir()


def final_output_exists(paper_dir: Path) -> bool:
    candidates = [
        paper_dir / "outputs" / "04_final_extraction.json",
        paper_dir / "04_final_extraction.json",
    ]
    return any(path.is_file() for path in candidates)


def discover_papers(root_dir: Path, start: int | None, end: int | None) -> list[Path]:
    dirs = [path for path in root_dir.iterdir() if path.is_dir()]
    dirs.sort(key=natural_key)
    selected: list[Path] = []
    for path in dirs:
        if path.name.isdigit():
            idx = int(path.name)
            if start is not None and idx < start:
                continue
            if end is not None and idx > end:
                continue
        selected.append(path)
    return selected


def run_one(paper_dir: Path, extra_args: list[str]) -> int:
    cmd = [
        sys.executable,
        "-m",
        "paper_mining.cli",
        "--paper-dir",
        str(paper_dir),
        *extra_args,
    ]
    log("Running: " + " ".join(cmd))
    completed = subprocess.run(cmd)
    return completed.returncode


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root-dir",
        required=True,
        help="Directory containing paper subfolders, e.g. C:/nfdproject/nlpproject/data/output",
    )
    parser.add_argument("--start", type=int, default=None, help="First numeric subfolder to process.")
    parser.add_argument("--end", type=int, default=None, help="Last numeric subfolder to process.")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip a paper if outputs/04_final_extraction.json already exists.",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop the whole batch when one paper fails.",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.0,
        help="Seconds to sleep between papers, useful for online API rate limits.",
    )
    parser.add_argument(
        "--extra-arg",
        action="append",
        default=[],
        help="Extra argument passed to paper_mining.cli. Repeat for multiple arguments.",
    )
    args = parser.parse_args()

    root_dir = Path(args.root_dir).expanduser().resolve()
    if not root_dir.is_dir():
        log(f"Root directory does not exist: {root_dir}")
        return 2

    papers = discover_papers(root_dir, args.start, args.end)
    log(f"Discovered {len(papers)} paper folders under {root_dir}")

    ok = 0
    skipped = 0
    failed: list[tuple[str, int]] = []

    for pos, paper_dir in enumerate(papers, 1):
        log(f"========== Paper {pos}/{len(papers)}: {paper_dir.name} ==========")

        if not has_required_inputs(paper_dir):
            log(f"Skipped: missing full.md or images/: {paper_dir}")
            skipped += 1
            continue

        if args.skip_existing and final_output_exists(paper_dir):
            log("Skipped: final output already exists")
            skipped += 1
            continue

        code = run_one(paper_dir, args.extra_arg)
        if code == 0:
            ok += 1
            log(f"Completed: {paper_dir.name}")
        else:
            failed.append((paper_dir.name, code))
            log(f"Failed: {paper_dir.name}, exit_code={code}")
            if args.stop_on_error:
                break

        if args.sleep > 0 and pos < len(papers):
            log(f"Sleeping {args.sleep:.1f}s before next paper")
            time.sleep(args.sleep)

    log("========== Batch summary ==========")
    log(f"Completed: {ok}")
    log(f"Skipped: {skipped}")
    log(f"Failed: {len(failed)}")
    if failed:
        for name, code in failed:
            log(f"  - {name}: exit_code={code}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
