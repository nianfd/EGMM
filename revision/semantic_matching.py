from __future__ import annotations

import hashlib
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class SemanticMatchResult:
    pairs: list[tuple[int, int, float]]
    unmatched_predictions: list[int]
    unmatched_gold: list[int]


def hungarian_match_from_scores(scores: Any, threshold: float) -> SemanticMatchResult:
    """Maximum-cardinality, maximum-similarity one-to-one matching.

    Real pairs below ``threshold`` receive negative weight.  Explicit dummy
    rows/columns allow every prediction and reference item to remain unmatched,
    so a low-similarity pair is never forced by the assignment solver.
    """
    import numpy as np
    from scipy.optimize import linear_sum_assignment

    matrix = np.asarray(scores, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("scores must be a two-dimensional matrix")
    n_predictions, n_gold = matrix.shape
    if n_predictions == 0 or n_gold == 0:
        return SemanticMatchResult(
            pairs=[],
            unmatched_predictions=list(range(n_predictions)),
            unmatched_gold=list(range(n_gold)),
        )
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("semantic threshold must be within [0, 1]")

    size = n_predictions + n_gold
    weights = np.zeros((size, size), dtype=np.float64)
    valid = matrix >= threshold
    # The +1 term makes cardinality the first objective; cosine similarity is
    # the tie-breaker among assignments with the same number of valid pairs.
    weights[:n_predictions, :n_gold] = np.where(valid, 1.0 + matrix, -1.0)
    row_indices, column_indices = linear_sum_assignment(weights, maximize=True)

    pairs: list[tuple[int, int, float]] = []
    used_predictions: set[int] = set()
    used_gold: set[int] = set()
    for row, column in zip(row_indices.tolist(), column_indices.tolist()):
        if row >= n_predictions or column >= n_gold or not valid[row, column]:
            continue
        score = float(matrix[row, column])
        pairs.append((row, column, score))
        used_predictions.add(row)
        used_gold.add(column)
    pairs.sort(key=lambda item: item[0])
    return SemanticMatchResult(
        pairs=pairs,
        unmatched_predictions=[index for index in range(n_predictions) if index not in used_predictions],
        unmatched_gold=[index for index in range(n_gold) if index not in used_gold],
    )


class SentenceEmbeddingMatcher:
    """Cached dense sentence embeddings plus Hungarian one-to-one matching."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        *,
        device: str = "auto",
        batch_size: int = 32,
        cache_path: Path | None = None,
        model_cache_dir: Path | None = None,
        show_progress: bool = True,
        encoder: Any | None = None,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.cache_path = cache_path
        self.model_cache_dir = model_cache_dir
        self.show_progress = show_progress
        self._encoder = encoder
        self._memory: dict[str, Any] = {}
        self._connection: sqlite3.Connection | None = None
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            self._connection = sqlite3.connect(cache_path)
            self._connection.execute(
                """
                CREATE TABLE IF NOT EXISTS embeddings (
                    model_name TEXT NOT NULL,
                    text_hash TEXT NOT NULL,
                    text_value TEXT NOT NULL,
                    dimension INTEGER NOT NULL,
                    vector BLOB NOT NULL,
                    PRIMARY KEY (model_name, text_hash)
                )
                """
            )
            self._connection.commit()

    def close(self) -> None:
        if self._connection is not None:
            self._connection.close()
            self._connection = None

    def __enter__(self) -> "SentenceEmbeddingMatcher":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    @staticmethod
    def _text_key(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _load_encoder(self) -> Any:
        if self._encoder is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise RuntimeError(
                    "Semantic evaluation requires sentence-transformers. Install with: "
                    "python -m pip install -r requirements-semantic-eval.txt"
                ) from exc
            kwargs: dict[str, Any] = {}
            if self.device != "auto":
                kwargs["device"] = self.device
            if self.model_cache_dir is not None:
                kwargs["cache_folder"] = str(self.model_cache_dir)
            self._encoder = SentenceTransformer(self.model_name, **kwargs)
        return self._encoder

    def _from_disk(self, text: str) -> Any | None:
        import numpy as np

        if self._connection is None:
            return None
        row = self._connection.execute(
            "SELECT text_value, dimension, vector FROM embeddings "
            "WHERE model_name=? AND text_hash=?",
            (self.model_name, self._text_key(text)),
        ).fetchone()
        if row is None or row[0] != text:
            return None
        return np.frombuffer(row[2], dtype=np.float32, count=int(row[1])).copy()

    def _to_disk(self, text: str, vector: Any) -> None:
        import numpy as np

        if self._connection is None:
            return
        array = np.asarray(vector, dtype=np.float32)
        self._connection.execute(
            "INSERT OR REPLACE INTO embeddings "
            "(model_name, text_hash, text_value, dimension, vector) VALUES (?, ?, ?, ?, ?)",
            (self.model_name, self._text_key(text), text, int(array.size), array.tobytes()),
        )

    def encode(self, texts: list[str]) -> Any:
        import numpy as np

        normalized_texts = [str(value or "").strip() for value in texts]
        missing: list[str] = []
        for text in dict.fromkeys(normalized_texts):
            if text in self._memory:
                continue
            cached = self._from_disk(text)
            if cached is not None:
                self._memory[text] = cached
            else:
                missing.append(text)
        if missing:
            encoder = self._load_encoder()
            vectors = encoder.encode(
                missing,
                batch_size=self.batch_size,
                show_progress_bar=self.show_progress,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
            for text, vector in zip(missing, vectors):
                array = np.asarray(vector, dtype=np.float32)
                self._memory[text] = array
                self._to_disk(text, array)
            if self._connection is not None:
                self._connection.commit()
        if not normalized_texts:
            return np.empty((0, 0), dtype=np.float32)
        return np.stack([self._memory[text] for text in normalized_texts])

    def similarity_matrix(self, predictions: list[str], gold: list[str]) -> Any:
        import numpy as np

        if not predictions or not gold:
            return np.empty((len(predictions), len(gold)), dtype=np.float32)
        # Encode both sides in one batch so the first evaluation pass incurs a
        # single model call.  Subsequent paper/condition passes reuse both the
        # in-memory and persistent SQLite caches.
        combined = self.encode([*predictions, *gold])
        left = combined[:len(predictions)]
        right = combined[len(predictions):]
        # encode(..., normalize_embeddings=True) makes dot product equal cosine.
        return np.clip(left @ right.T, -1.0, 1.0)

    def match(
        self,
        predictions: list[str],
        gold: list[str],
        threshold: float,
    ) -> SemanticMatchResult:
        return hungarian_match_from_scores(
            self.similarity_matrix(predictions, gold),
            threshold,
        )

    def provenance(self) -> dict[str, Any]:
        versions: dict[str, str] = {}
        for module_name in ("sentence_transformers", "scipy", "numpy", "torch"):
            try:
                module = __import__(module_name)
                versions[module_name] = str(getattr(module, "__version__", "unknown"))
            except ImportError:
                versions[module_name] = "not_loaded"
        return {
            "matcher": "dense_cosine_hungarian_with_explicit_unmatched_dummies",
            "embedding_model": self.model_name,
            "device": self.device,
            "batch_size": self.batch_size,
            "normalize_embeddings": True,
            "persistent_cache": str(self.cache_path) if self.cache_path else None,
            "library_versions": versions,
        }
