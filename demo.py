"""An actual search walkthrough on hand-written synthetic vectors."""
from pathlib import Path

from search import read_json, run
from vector_search import VectorIndex


def main():
    root = Path(__file__).resolve().parent
    data = read_json(root / "examples/vectors.json")
    queries = read_json(root / "examples/queries.json")
    print("FAISS Search Workbench | actual CPU search, synthetic vectors")
    print("Query: [1, 0]. Same direction is not the same as nearest point.")
    for metric in ("cosine", "l2", "inner_product"):
        result = run(data, queries, metric=metric, k=10)
        print(f"\n{metric}: {result['score_kind']}")
        for rank, match in enumerate(result["results"][0], 1):
            print(f"  {rank}. {match['id']:<18} {match['score']:.4f}")
        print(f"  Requested 10; returned {result['returned_per_query']} real IDs (no padding).")
    try:
        VectorIndex(data["vectors"], data["ids"]).search([[0, 0]])
    except ValueError as exc:
        print(f"\nZero-vector check: {exc}")
    print("\nNo model, API key, download or external service used by this demo.")


if __name__ == "__main__":
    main()
