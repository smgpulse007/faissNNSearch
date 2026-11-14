"""Search a local JSON vector matrix, printing machine-readable JSON."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

from vector_search import METRICS, VectorIndex

MAX_JSON_BYTES = 8 * 1024 * 1024
MAX_JSON_DEPTH = 64


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON keys are not allowed")
        result[key] = value
    return result


def _constant(_value):
    raise ValueError("JSON numbers must be finite")


def _float(value):
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("JSON numbers must be finite")
    return number


def _check_depth(text: str):
    # Bound structure before parsing; Python versions have different recursion
    # limits. Brackets inside strings (including escaped quotes) are just data.
    depth, quoted, escaped = 0, False, False
    for char in text:
        if quoted:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                quoted = False
        elif char == '"':
            quoted = True
        elif char in "[{":
            depth += 1
            if depth > MAX_JSON_DEPTH:
                raise ValueError("JSON nesting exceeds the 64-level limit")
        elif char in "]}":
            depth -= 1


def read_json(path: Path):
    with path.open("rb") as source:
        data = source.read(MAX_JSON_BYTES + 1)
    if len(data) > MAX_JSON_BYTES:
        raise ValueError("input JSON exceeds the 8 MiB limit")
    try:
        text = data.decode("utf-8-sig")
        _check_depth(text)
        return json.loads(text, object_pairs_hook=_object,
                          parse_constant=_constant, parse_float=_float)
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as exc:
        raise ValueError("input must be valid UTF-8 JSON with bounded nesting") from exc


def run(data: dict, queries, *, metric: str, k: int) -> dict:
    if not isinstance(data, dict) or set(data) != {"ids", "vectors"}:
        raise ValueError("data must contain exactly ids and vectors")
    if not isinstance(data["ids"], list):
        raise ValueError("ids must be a JSON list")
    index = VectorIndex(data["vectors"], data["ids"], metric=metric)
    results = index.search(queries, k)
    return {"metric": metric, "score_kind": index.score_kind,
            "higher_is_better": metric != "l2", "indexed_vectors": index.count,
            "dimension": index.dimension, "requested_k": k,
            "returned_per_query": len(results[0]),
            "results": [[neighbor.to_dict() for neighbor in row] for row in results]}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="JSON with ids and vectors")
    parser.add_argument("--queries", type=Path, required=True, help="JSON matrix, one query per row")
    parser.add_argument("--metric", choices=METRICS, default="cosine")
    parser.add_argument("-k", type=int, default=5)
    args = parser.parse_args(argv)
    try:
        result = run(read_json(args.data), read_json(args.queries), metric=args.metric, k=args.k)
    except (OSError, ValueError, TypeError, OverflowError) as exc:
        # Do not echo vector data or file contents in errors.
        message = str(exc) if isinstance(exc, ValueError) else "unable to read or validate input files"
        print(json.dumps({"error": message}), file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
