import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path("services").resolve()))

from bm25_retriever import BM25Retriever, tokenize


class BM25RetrieverTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # 从项目真实的 chunks.json 建索引，但不调用任何外部 API。
        cls.retriever = BM25Retriever.from_json()

    def test_tokenize_keeps_audit_terms(self):
        tokens = tokenize("同一供应商30天内相同金额重复付款")

        self.assertIn("同一供应商", tokens)
        self.assertIn("重复付款", tokens)
        self.assertIn("30", tokens)

    def test_duplicate_payment_query_finds_policy_section(self):
        results = self.retriever.search(
            "同一供应商30天相同金额重复付款三方核对",
            top_k=3,
        )

        self.assertEqual(len(results), 3)
        self.assertEqual(results[0]["source"], "audit_policy.txt")
        self.assertIn("第四章 重复付款核查", results[0]["section"])
        self.assertGreater(results[0]["bm25_score"], 0)

    def test_amount_query_finds_amount_policy(self):
        results = self.retriever.search(
            "单笔交易金额超过50万元财务总监审批",
            top_k=3,
        )

        self.assertEqual(results[0]["source"], "audit_policy.txt")
        self.assertIn("第三章 金额审批权限", results[0]["section"])
        self.assertGreater(results[0]["bm25_score"], 0)

    def test_empty_query_returns_empty_list(self):
        self.assertEqual(self.retriever.search("", top_k=3), [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
