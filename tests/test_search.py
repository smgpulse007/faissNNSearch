import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from search import MAX_JSON_BYTES, MAX_JSON_DEPTH, read_json, run
from vector_search import VectorIndex

ROOT = Path(__file__).resolve().parents[1]


class SearchTests(unittest.TestCase):
    def test_metrics_change_rankings(self):
        vectors = [[10, 0], [1, 0.2], [0, 1]]
        ids = ["same-direction", "nearby-point", "other-direction"]
        self.assertEqual(VectorIndex(vectors, ids).search([[1, 0]], 1)[0][0].id, ids[0])
        nearest = VectorIndex(vectors, ids, metric="l2").search([[1, 0]], 1)[0][0]
        self.assertEqual(nearest.id, ids[1])
        self.assertAlmostEqual(nearest.score, 0.04, places=6)

    def test_l2_is_squared_distance(self):
        hit = VectorIndex([[3, 4]], ["a"], metric="l2").search([[0, 0]])[0][0]
        self.assertEqual(hit.score, 25)

    def test_cosine_normalizes_database_and_queries(self):
        hit = VectorIndex([[10, 10]], ["a"]).search([[2, 2]])[0][0]
        self.assertAlmostEqual(hit.score, 1, places=6)

    def test_inner_product_keeps_magnitude(self):
        index = VectorIndex([[10, 0], [2, 0]], ["large", "small"], metric="inner_product")
        results = index.search([[2, 0]])[0]
        self.assertEqual([(r.id, r.score) for r in results], [("large", 20), ("small", 4)])

    def test_actual_faiss_matches_numpy_for_all_metrics_and_batches(self):
        rng = np.random.default_rng(410)
        vectors = rng.normal(size=(150, 12)).astype(np.float32)
        queries = rng.normal(size=(9, 12)).astype(np.float32)
        ids = [f"document-{i}" for i in range(len(vectors))]
        for metric in ("l2", "cosine", "inner_product"):
            with self.subTest(metric=metric):
                a, b = vectors.astype(np.float64), queries.astype(np.float64)
                if metric == "cosine":
                    a /= np.linalg.norm(a, axis=1)[:, None]
                    b /= np.linalg.norm(b, axis=1)[:, None]
                scores = (np.sum((b[:, None, :] - a[None, :, :]) ** 2, axis=2)
                          if metric == "l2" else b @ a.T)
                rows = np.argsort(scores if metric == "l2" else -scores, axis=1)[:, :7]
                results = VectorIndex(vectors, ids, metric=metric).search(queries, 7)
                for result, expected_rows, expected_scores in zip(results, rows, scores):
                    self.assertEqual([r.id for r in result], [ids[i] for i in expected_rows])
                    np.testing.assert_allclose([r.score for r in result], expected_scores[expected_rows], rtol=2e-5, atol=2e-5)

    def test_k_larger_than_corpus_has_no_phantom_document(self):
        result = VectorIndex([[1, 0], [0, 1]], ["a", "b"]).search([[1, 0]], 999999)
        self.assertEqual(len(result[0]), 2)
        self.assertEqual({r.id for r in result[0]}, {"a", "b"})

    def test_noncontiguous_inputs_and_caller_arrays_unchanged(self):
        vectors = np.arange(24, dtype=np.float64).reshape(4, 6)[:, ::2]
        queries = np.arange(12, dtype=np.float64).reshape(2, 6)[:, ::2]
        before_v, before_q = vectors.copy(), queries.copy()
        index = VectorIndex(vectors, ["a", "b", "c", "d"])
        index.search(queries, 2)
        np.testing.assert_array_equal(vectors, before_v)
        np.testing.assert_array_equal(queries, before_q)
        vectors[:] = 0  # The internal index also owns its values.
        self.assertEqual(len(index.search(before_q, 2)), 2)

    def test_zero_vectors_rejected_for_cosine(self):
        with self.assertRaisesRegex(ValueError, "zero vector"):
            VectorIndex([[0, 0]], ["a"])
        with self.assertRaisesRegex(ValueError, "zero vector"):
            VectorIndex([[1, 0]], ["a"]).search([[0, 0]])
        self.assertEqual(VectorIndex([[0, 0]], ["a"], metric="l2").search([[0, 0]])[0][0].score, 0)

    def test_normalization_handles_extreme_finite_float32(self):
        for value in (1e30, 1e-30):
            with self.subTest(value=value):
                result = VectorIndex([[value, value]], ["a"]).search([[value, value]])
                self.assertAlmostEqual(result[0][0].score, 1, places=6)

    def test_invalid_shapes_and_dtypes(self):
        for matrix in ([], [[]], [1, 2], [[1], [2, 3]], [[[1]]], [["1"]], [[True]], [[1j]], [[None]]):
            with self.subTest(matrix=matrix), self.assertRaises(ValueError):
                VectorIndex(matrix, ["a"])

    def test_nonfinite_and_float32_overflow_rejected(self):
        for value in (float("inf"), float("nan"), -float("inf"), 1e100):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "finite"):
                VectorIndex([[value]], ["a"])

    def test_score_overflow_rejected(self):
        index = VectorIndex([[1e30]], ["a"], metric="inner_product")
        with self.assertRaisesRegex(ValueError, "overflowed"):
            index.search([[1e30]])

    def test_dimension_mismatch(self):
        with self.assertRaisesRegex(ValueError, "dimension 2"):
            VectorIndex([[1, 2]], ["a"]).search([[1]])

    def test_bad_ids_and_metrics(self):
        for ids in (["a", "a"], ["a"], ["", "b"], ["a", 1], ["a", ["b"]], "ab"):
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                VectorIndex([[1], [2]], ids)
        with self.assertRaises(ValueError):
            VectorIndex([[1]], ["a"], metric="euclidean")

    def test_invalid_k_and_numpy_integer(self):
        index = VectorIndex([[1]], ["a"])
        for k in (0, -1, True, False, 1.5, "2", None):
            with self.subTest(k=k), self.assertRaises(ValueError):
                index.search([[1]], k)
        self.assertEqual(len(index.search([[1]], np.int64(2))[0]), 1)

    def test_json_contract(self):
        result = run({"ids": ["a"], "vectors": [[1, 2]]}, [[1, 2]], metric="cosine", k=3)
        self.assertEqual(result["score_kind"], "cosine_similarity")
        self.assertEqual(result["returned_per_query"], 1)
        self.assertTrue(result["higher_is_better"])
        json.dumps(result, allow_nan=False)
        for bad in ([], {"vectors": [[1]]}, {"ids": "a", "vectors": [[1]]}, {"ids": ["a"], "vectors": [[1]], "extra": 1}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                run(bad, [[1]], metric="cosine", k=1)


class CliTests(unittest.TestCase):
    def test_real_cli_fixture(self):
        result = subprocess.run([sys.executable, str(ROOT / "search.py"),
                                 "--data", str(ROOT / "examples/vectors.json"),
                                 "--queries", str(ROOT / "examples/queries.json"), "-k", "10"],
                                capture_output=True, text=True, timeout=30, cwd=tempfile.gettempdir())
        self.assertEqual(result.returncode, 0, result.stderr)
        data = json.loads(result.stdout)
        self.assertEqual(data["results"][0][0]["id"], "same-direction")
        self.assertEqual(data["returned_per_query"], 3)

    def test_real_cli_invalid_input_has_no_data_echo(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "broken.json"
            source.write_text('{"private_fixture_content":', encoding="utf-8")
            result = subprocess.run([sys.executable, str(ROOT / "search.py"), "--data", str(source),
                                     "--queries", str(ROOT / "examples/queries.json")],
                                    capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 2)
            self.assertEqual(result.stdout, "")
            self.assertNotIn("private_fixture_content", result.stderr)
            self.assertIn("error", json.loads(result.stderr))

    def test_invalid_json_variants(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.json"
            for data in (b'{"a":1,"a":2}', b'[NaN]', b'[Infinity]', b'\xff', b'[' * 1500 + b'0' + b']' * 1500):
                path.write_bytes(data)
                with self.subTest(data=data[:25]), self.assertRaises(ValueError):
                    read_json(path)

    def test_bom_and_size_limit(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.json"
            path.write_text("[[1, 2]]", encoding="utf-8-sig")
            self.assertEqual(read_json(path), [[1, 2]])
            path.write_bytes(b" " * (MAX_JSON_BYTES + 1))
            with self.assertRaisesRegex(ValueError, "8 MiB"):
                read_json(path)

    def test_exponent_overflow_rejected_at_json_boundary(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.json"
            for number in ("1e10000", "-1e10000"):
                path.write_text("[[" + number + "]]", encoding="utf-8")
                with self.subTest(number=number), self.assertRaisesRegex(ValueError, "finite"):
                    read_json(path)

    def test_explicit_depth_limit_ignores_brackets_in_strings(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.json"
            value = {"text": '\\"' + '[' * 200 + '\\"' + ']' * 200}
            path.write_text(json.dumps(value), encoding="utf-8")
            self.assertEqual(read_json(path), value)
            path.write_text('[' * MAX_JSON_DEPTH + '0' + ']' * MAX_JSON_DEPTH, encoding="utf-8")
            self.assertIsInstance(read_json(path), list)
            path.write_text('[' * (MAX_JSON_DEPTH + 1) + '0' + ']' * (MAX_JSON_DEPTH + 1), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "64-level"):
                read_json(path)
            for opening, closing, levels in (('{"a":', '}', 1), ('[{"a":', '}]', 2)):
                count = MAX_JSON_DEPTH // levels
                path.write_text(opening * count + '0' + closing * count, encoding="utf-8")
                self.assertIsInstance(read_json(path), (list, dict))
                path.write_text(opening * (count + 1) + '0' + closing * (count + 1), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "64-level"):
                    read_json(path)


if __name__ == "__main__":
    unittest.main()
