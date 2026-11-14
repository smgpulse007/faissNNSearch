"""Small exact-search adapter around Faiss. No embedding model or network calls."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from numbers import Integral
from typing import Sequence

import faiss
import numpy as np

METRICS = ("l2", "cosine", "inner_product")


@dataclass(frozen=True)
class Neighbor:
    id: str
    score: float

    def to_dict(self) -> dict:
        return asdict(self)


def _matrix(values, name: str) -> np.ndarray:
    try:
        raw = np.asarray(values)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"{name} must be a rectangular numeric matrix") from exc
    if raw.ndim != 2 or not all(raw.shape):
        raise ValueError(f"{name} must have at least one row and one dimension")
    if raw.dtype.kind not in "iuf":
        raise ValueError(f"{name} must contain real numbers, not strings or booleans")
    with np.errstate(over="ignore", invalid="ignore"):
        matrix = np.array(raw, dtype=np.float32, order="C", copy=True)
    if not np.isfinite(matrix).all():
        raise ValueError(f"{name} must contain finite float32 values")
    return matrix


def _normalize(matrix: np.ndarray, name: str) -> np.ndarray:
    # Float64 norms avoid overflow/underflow while normalizing valid float32 rows.
    norms = np.linalg.norm(matrix.astype(np.float64), axis=1)
    if np.any(norms == 0):
        raise ValueError(f"{name} contains a zero vector; cosine similarity is undefined")
    return np.ascontiguousarray(matrix / norms[:, None], dtype=np.float32)


class VectorIndex:
    """In-memory exact CPU search with caller-supplied unique string IDs.

    l2 scores are squared distances (lower is better); cosine and inner_product
    scores are similarities (higher is better). Ties have no ordering guarantee.
    Inputs are copied. Search never normalizes or otherwise changes caller data.
    """

    def __init__(self, vectors, ids: Sequence[str], *, metric: str = "cosine"):
        if metric not in METRICS:
            raise ValueError(f"metric must be one of {', '.join(METRICS)}")
        matrix = _matrix(vectors, "vectors")
        if isinstance(ids, (str, bytes)):
            raise ValueError("ids must be a sequence of unique nonempty strings")
        self._ids = tuple(ids)
        if (len(self._ids) != len(matrix)
                or any(not isinstance(item, str) or not item.strip() for item in self._ids)
                or len(set(self._ids)) != len(self._ids)):
            raise ValueError("ids must be unique nonempty strings, one per vector")
        self.metric = metric
        self.dimension = matrix.shape[1]
        self.count = matrix.shape[0]
        if metric == "cosine":
            matrix = _normalize(matrix, "vectors")
        self._index = (faiss.IndexFlatL2(self.dimension) if metric == "l2"
                       else faiss.IndexFlatIP(self.dimension))
        self._index.add(matrix)

    @property
    def score_kind(self) -> str:
        return {"l2": "squared_l2_distance", "cosine": "cosine_similarity",
                "inner_product": "inner_product"}[self.metric]

    def search(self, queries, k: int = 5) -> list[list[Neighbor]]:
        if isinstance(k, bool) or not isinstance(k, Integral) or k < 1:
            raise ValueError("k must be a positive integer")
        matrix = _matrix(queries, "queries")
        if matrix.shape[1] != self.dimension:
            raise ValueError(f"queries must have dimension {self.dimension}")
        if self.metric == "cosine":
            matrix = _normalize(matrix, "queries")
        # Faiss pads k > ntotal with -1. Clamp before searching so -1 can never
        # accidentally become the last document's ID through Python indexing.
        scores, rows = self._index.search(matrix, min(int(k), self.count))
        if not np.isfinite(scores).all() or np.any(rows < 0):
            raise ValueError("search scores overflowed; rescale vectors or use cosine")
        return [[Neighbor(self._ids[int(row)], float(score))
                 for row, score in zip(row_ids, row_scores)]
                for row_ids, row_scores in zip(rows, scores)]
