from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from revision.evaluation import greedy_match
from revision.semantic_matching import SentenceEmbeddingMatcher, hungarian_match_from_scores


class _FixtureEncoder:
    def __init__(self) -> None:
        self.calls = 0

    def encode(self, texts: list[str], **_: object) -> np.ndarray:
        self.calls += 1
        vectors = {
            "spectral corruption degrades hyperspectral observations": np.array([1.0, 0.0]),
            "hyperspectral images are impaired by band-wise noise": np.array([0.98, 0.20]),
            "a transformer unmixing network removes corruption": np.array([0.0, 1.0]),
            "transformer-based unmixing restores clean spectra": np.array([0.20, 0.98]),
        }
        result = np.stack([vectors[text] for text in texts]).astype(np.float32)
        result /= np.linalg.norm(result, axis=1, keepdims=True)
        return result


def main() -> int:
    predictions = [
        "spectral corruption degrades hyperspectral observations",
        "a transformer unmixing network removes corruption",
    ]
    references = [
        "hyperspectral images are impaired by band-wise noise",
        "transformer-based unmixing restores clean spectra",
    ]
    lexical = greedy_match(predictions, references, threshold=0.45)
    encoder = _FixtureEncoder()
    with tempfile.TemporaryDirectory(prefix="papermining_semantic_selftest_") as temp:
        matcher = SentenceEmbeddingMatcher(
            "fixture-model",
            encoder=encoder,
            show_progress=False,
            cache_path=Path(temp) / "embeddings.sqlite3",
        )
        semantic = matcher.match(predictions, references, threshold=0.70)
        matcher.encode(predictions)
        matcher.close()

    global_assignment = hungarian_match_from_scores(
        [[0.90, 0.80], [0.85, 0.10]], threshold=0.70,
    )
    assert lexical.pairs == [], "fixture must expose the legacy lexical false-negative"
    assert {(p, g) for p, g, _ in semantic.pairs} == {(0, 0), (1, 1)}
    assert {(p, g) for p, g, _ in global_assignment.pairs} == {(0, 1), (1, 0)}
    assert encoder.calls == 1, "the combined prediction/reference batch and cache were not reused"
    payload = {
        "status": "passed",
        "legacy_lexical_matches": len(lexical.pairs),
        "semantic_matches": len(semantic.pairs),
        "global_one_to_one_assignment": [
            [p, g, round(score, 6)] for p, g, score in global_assignment.pairs
        ],
        "encoder_calls": encoder.calls,
    }
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
