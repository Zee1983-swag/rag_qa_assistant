import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path("services").resolve()))

from rank_fusion import reciprocal_rank_fusion


def chunk(chunk_id, text, **scores):
    return {
        "chunk_id": chunk_id,
        "source": "audit_policy.txt",
        "section": "测试章节",
        "text": text,
        **scores,
    }


class ReciprocalRankFusionTests(unittest.TestCase):
    def setUp(self):
        self.vector_results = [
            chunk("A", "向量结果 A", vector_score=0.95),
            chunk("C", "向量结果 C", vector_score=0.85),
            chunk("D", "向量结果 D", vector_score=0.75),
        ]

        self.bm25_results = [
            chunk("B", "BM25 结果 B", bm25_score=12.0),
            chunk(
                "A",
                "BM25 与向量都命中的结果 A",
                bm25_score=11.0,
            ),
            chunk(
                "C",
                "两个检索器都命中的结果 C",
                bm25_score=10.0,
            ),
        ]

    def get_results(self):
        return reciprocal_rank_fusion({
            "vector": self.vector_results,
            "bm25": self.bm25_results,
        })

    def test_result_is_deduplicated(self):
        ids = [item["chunk_id"] for item in self.get_results()]

        self.assertEqual(len(ids), 4)
        self.assertEqual(len(ids), len(set(ids)))

    def test_result_found_by_two_retrievers_is_prioritized(self):
        results = self.get_results()

        self.assertEqual(results[0]["chunk_id"], "A")
        self.assertIn("vector", results[0]["rank_details"])
        self.assertIn("bm25", results[0]["rank_details"])

    def test_rank_details_are_preserved(self):
        by_id = {item["chunk_id"]: item for item in self.get_results()}

        self.assertEqual(by_id["A"]["rank_details"], {
            "vector": 1,
            "bm25": 2,
        })
        self.assertEqual(by_id["B"]["rank_details"], {
            "bm25": 1,
        })
        self.assertGreater(by_id["A"]["rrf_score"], by_id["B"]["rrf_score"])

    def test_scores_from_both_retrievers_are_merged(self):
        by_id = {item["chunk_id"]: item for item in self.get_results()}

        self.assertEqual(by_id["A"]["vector_score"], 0.95)
        self.assertEqual(by_id["A"]["bm25_score"], 11.0)

    def test_empty_input_returns_empty_list(self):
        self.assertEqual(reciprocal_rank_fusion({}), [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
