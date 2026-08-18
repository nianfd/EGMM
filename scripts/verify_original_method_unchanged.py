from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


PROTECTED_FILES = [
    "paper_mining/pipeline.py",
    "paper_mining/prompts.py",
    "paper_mining/markdown_mineru.py",
    "paper_mining/qwenvl_client.py",
    "paper_mining/io_utils.py",
    "paper_mining/progress.py",
    "experiments/one_shot_mineru_baseline.py",
    "experiments/baseline_prompts.py",
    "experiments/metrics.py",
    "scripts/batch_process_papers.py",
    "scripts/batch_run_comparison_experiments.py",
    "scripts/batch_run_large_chunk_ablation.py",
]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify that protected original Qwen-VL method/experiment files match the supplied nlpproject archive.")
    parser.add_argument("--project-dir", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--manifest", default=str(Path(__file__).resolve().parents[1] / "ORIGINAL_QWENVL_METHOD_SHA256.json"))
    args = parser.parse_args()
    root = Path(args.project_dir).resolve()
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    failed = []
    for relative in PROTECTED_FILES:
        expected = manifest["protected_files"].get(relative)
        actual = digest(root / relative) if (root / relative).exists() else "MISSING"
        print(f"{'OK' if actual == expected else 'MISMATCH'}  {relative}")
        if actual != expected:
            failed.append(relative)
    if failed:
        raise SystemExit("Protected original Qwen-VL files changed: " + ", ".join(failed))
    print("Original Qwen-VL method and original experiment implementations are unchanged.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
