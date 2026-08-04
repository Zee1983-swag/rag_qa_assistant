import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path("services").resolve()))

from hybrid_retriever import HybridRetriever


class FakeCollection:
    def __init__(self):
        self.last_n_results = None

    def query(self, query_embeddings, n_results):
        self.last_n_results = n_results

        return {
            "ids": [["A", "C", "D"]],
            "documents": [[
                "向量检索第1名",
                "向量检索第2名",
                "向量检索第3名",
            ]],
            "metadatas": [[
                {"source": "audit_policy.txt", "section": "章节A"},
                {"source": "audit_policy.txt", "section": "章节C"},
                {"source": "audit_policy.txt", "section": "章节D"},
            ]],
            "distances": [[0.1, 0.2, 0.3]],
        }


class FakeBM25Retriever:
    def __init__(self):
        self.last_top_k = None

    def search(self, query, top_k):
        self.last_top_k = top_k

        return [
            {
                "chunk_id": "B",
                "source": "audit_policy.txt",
                "section": "章节B",
                "text": "BM25第1名",
                "bm25_score": 10.0,
            },
            {
                "chunk_id": "A",
                "source": "audit_policy.txt",
                "section": "章节A",
                "text": "两个检索器共同命中的 A",
                "bm25_score": 9.0,
            },
            {
                "chunk_id": "C",
                "source": "audit_policy.txt",
                "section": "章节C",
                "text": "两个检索器共同命中的 C",
                "bm25_score": 8.0,
            },
        ]


class HybridRetrieverTests(unittest.TestCase):
    def setUp(self):
        self.collection = FakeCollection()
        self.bm25 = FakeBM25Retriever()

        self.retriever = HybridRetriever(
            collection=self.collection,
            embed_query=lambda query: [0.1, 0.2, 0.3],
            bm25_retriever=self.bm25,
            candidate_k=3,
            rrf_k=60,
        )

    def test_two_retrievers_are_called(self):
        self.retriever.retrieve("重复付款三方核对", top_k=3)

        self.assertEqual(self.collection.last_n_results, 3)
        self.assertEqual(self.bm25.last_top_k, 3)

    def test_common_result_is_ranked_first(self):
        results = self.retriever.retrieve("重复付款三方核对", top_k=3)

        self.assertEqual(results[0]["chunk_id"], "A")
        self.assertIn("vector", results[0]["rank_details"])
        self.assertIn("bm25", results[0]["rank_details"])

    def test_result_contains_all_observability_fields(self):
        result = self.retriever.retrieve("重复付款三方核对", top_k=3)[0]

        self.assertIn("vector_score", result)
        self.assertIn("bm25_score", result)
        self.assertIn("rrf_score", result)
        self.assertIn("rank_details", result)

    def test_top_k_limits_final_result_count(self):
        results = self.retriever.retrieve("重复付款三方核对", top_k=2)

        self.assertEqual(len(results), 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
