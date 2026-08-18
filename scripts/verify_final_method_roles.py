from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import build_paper_tables_1_2_3 as tables
from scripts import build_qwen38_gold as reference
from scripts import run_revision_experiments as experiments


def main() -> int:
    checks = {
        "reference_output_name": reference.REFERENCE_OUTPUT_NAME,
        "reference_chunk_chars": reference.REFERENCE_CHUNK_CHARS,
        "revised_main_condition": tables.REVISED_MAIN_CONDITION,
        "default_revision_conditions": experiments.DEFAULT_CONDITIONS,
        "table1_revised_row": [
            row for row in tables.STRUCTURAL_MAIN
            if row[0] == tables.REVISED_MAIN_CONDITION
        ],
        "table2_revised_row": [
            row for row in tables.SEMANTIC_MAIN
            if row[0] == tables.REVISED_MAIN_CONDITION
        ],
        "table3_revised_row": [
            row for row in tables.CHUNK_SWEEP
            if row[0] == tables.REVISED_MAIN_CONDITION
        ],
    }
    errors: list[str] = []
    if checks["reference_output_name"] != "revision_annotations_main_aligned":
        errors.append("reference output directory changed")
    if checks["reference_chunk_chars"] != 9000:
        errors.append("frozen reference chunk size is not 9000")
    if checks["revised_main_condition"] != "added_chunk_6000":
        errors.append("revised main condition is not added_chunk_6000")
    if checks["default_revision_conditions"] != ["added_chunk_6000"]:
        errors.append("bare run_revision_experiments command does not target added_chunk_6000")
    for name in ("table1_revised_row", "table2_revised_row", "table3_revised_row"):
        if len(checks[name]) != 1:
            errors.append(f"{name} is missing or duplicated")
    payload = {"status": "passed" if not errors else "failed", "checks": checks, "errors": errors}
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
