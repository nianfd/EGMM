from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Aggregate redacted Qwen API call/token records from revision experiments.")
    parser.add_argument("--root-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.root_dir).resolve()
    records = []
    for path in root.glob("*/outputs/major_revision_additions/*/api_usage.jsonl"):
        paper_id, condition = path.parts[-5], path.parts[-2]
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            row.update({"paper_id": paper_id, "condition": condition})
            records.append(row)
    grouped = defaultdict(lambda: {"calls": 0, "cache_hits": 0, "input_tokens": 0, "output_tokens": 0, "total_tokens": 0})
    for row in records:
        target = grouped[row["condition"]]
        target["calls"] += int(row.get("request_count", 0))
        target["cache_hits"] += int(bool(row.get("cache_hit")))
        usage = row.get("usage") or {}
        target["input_tokens"] += int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
        target["output_tokens"] += int(usage.get("completion_tokens") or usage.get("output_tokens") or 0)
        target["total_tokens"] += int(usage.get("total_tokens") or 0)
    output = {"conditions": [{"condition": key, **value} for key, value in sorted(grouped.items())], "raw_record_count": len(records)}
    destination = Path(args.output).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
