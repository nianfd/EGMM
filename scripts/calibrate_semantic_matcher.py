from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from revision.io_utils import read_json, write_csv, write_json
from revision.semantic_matching import SentenceEmbeddingMatcher, hungarian_match_from_scores


def _validated_pairs(packet: dict[str, Any], kind: str) -> set[tuple[str, str]]:
    return {
        (str(pair[0]), str(pair[1]))
        for pair in packet.get(f"{kind}_pairs", [])
        if isinstance(pair, list) and len(pair) == 2
    }


def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def main() -> int:
    parser = argparse.ArgumentParser(description="Calibrate semantic node-match thresholds from verified packets")
    parser.add_argument("--packets-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--embedding-model", default="BAAI/bge-m3")
    parser.add_argument("--embedding-device", default="auto")
    parser.add_argument("--embedding-batch-size", type=int, default=32)
    parser.add_argument("--embedding-cache-file")
    parser.add_argument("--threshold-min", type=float, default=0.50)
    parser.add_argument("--threshold-max", type=float, default=0.90)
    parser.add_argument("--threshold-step", type=float, default=0.01)
    args = parser.parse_args()

    packets_dir = Path(args.packets_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    packets = []
    for path in sorted(packets_dir.glob("*.json")):
        packet = read_json(path)
        if packet.get("status") != "verified_human" or not str(packet.get("reviewer_name") or "").strip():
            continue
        packets.append(packet)
    if not packets:
        raise SystemExit("No status=verified_human packets with reviewer_name were found")
    if args.threshold_step <= 0 or args.threshold_max < args.threshold_min:
        raise SystemExit("Invalid threshold range")
    count = int(round((args.threshold_max - args.threshold_min) / args.threshold_step))
    thresholds = [round(args.threshold_min + index * args.threshold_step, 6) for index in range(count + 1)]
    cache_path = (
        Path(args.embedding_cache_file).resolve()
        if args.embedding_cache_file else packets_dir / "semantic_embeddings.sqlite3"
    )
    matcher = SentenceEmbeddingMatcher(
        args.embedding_model,
        device=args.embedding_device,
        batch_size=args.embedding_batch_size,
        cache_path=cache_path,
    )
    prepared: dict[
        str,
        list[tuple[list[dict[str, str]], list[dict[str, str]], Any, set[tuple[str, str]]]],
    ] = {
        "problem": [], "method": [],
    }
    try:
        for packet in packets:
            for kind in ("problem", "method"):
                predicted = packet.get(f"predicted_{kind}s", [])
                gold = packet.get(f"gold_{kind}s", [])
                scores = matcher.similarity_matrix(
                    [str(item.get("text") or "") for item in predicted],
                    [str(item.get("text") or "") for item in gold],
                )
                prepared[kind].append((predicted, gold, scores, _validated_pairs(packet, kind)))

        rows: list[dict[str, Any]] = []
        best: dict[str, dict[str, Any]] = {}
        for kind in ("problem", "method"):
            kind_rows = []
            for threshold in thresholds:
                tp = fp = fn = 0
                for predicted, gold, scores, accepted in prepared[kind]:
                    match = hungarian_match_from_scores(scores, threshold)
                    produced = {
                        (str(predicted[p]["id"]), str(gold[g]["id"]))
                        for p, g, _ in match.pairs
                    }
                    tp += len(produced & accepted)
                    fp += len(produced - accepted)
                    fn += len(accepted - produced)
                precision, recall, f1 = _prf(tp, fp, fn)
                row = {
                    "kind": kind, "threshold": threshold,
                    "tp": tp, "fp": fp, "fn": fn,
                    "precision": precision, "recall": recall, "f1": f1,
                }
                rows.append(row)
                kind_rows.append(row)
            # F1 is primary; higher threshold breaks exact ties conservatively.
            best[kind] = max(kind_rows, key=lambda row: (row["f1"], row["threshold"]))
    finally:
        matcher.close()

    write_csv(output_dir / "semantic_threshold_sweep.csv", rows)
    payload = {
        "status": "calibrated_from_verified_human_packets",
        "embedding_model": args.embedding_model,
        "problem_similarity_threshold": best["problem"]["threshold"],
        "method_similarity_threshold": best["method"]["threshold"],
        "problem_calibration": best["problem"],
        "method_calibration": best["method"],
        "verified_packet_count": len(packets),
        "matcher": "normalized_dense_cosine_plus_hungarian_with_explicit_unmatched_dummies",
    }
    write_json(output_dir / "semantic_thresholds.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
