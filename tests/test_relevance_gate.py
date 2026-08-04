import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path("config").resolve()))

from relevance_gate import get_abstention_reason, has_audit_anchor


class RelevanceGateTests(unittest.TestCase):
    def test_audit_query_has_anchor(self):
        query = "同一供应商30天内相同金额付款如何核查"
        self.assertTrue(has_audit_anchor(query))

    def test_irrelevant_query_has_no_anchor(self):
        query = "量子计算机天气预报火星探测脑外科手术"
        self.assertFalse(has_audit_anchor(query))

    def test_irrelevant_query_is_rejected_even_with_high_vector_score(self):
        query = "量子计算机天气预报火星探测脑外科手术"

        # 即使向量库错误地给出了 0.99 的“最高相似度”，
        # 领域闸门仍必须拒答。
        reason = get_abstention_reason(
            query=query,
            top_similarity=0.99,
            threshold=0.5,
        )

        self.assertIsNotNone(reason)
        self.assertIn("证据不足", reason)

    def test_low_similarity_audit_query_is_rejected(self):
        reason = get_abstention_reason(
            query="供应商付款审批流程",
            top_similarity=0.30,
            threshold=0.5,
        )

        self.assertIsNotNone(reason)
        self.assertIn("低于阈值", reason)

    def test_relevant_query_can_continue(self):
        reason = get_abstention_reason(
            query="同一供应商重复付款的三单匹配核查",
            top_similarity=0.85,
            threshold=0.5,
        )

        self.assertIsNone(reason)


if __name__ == "__main__":
    unittest.main(verbosity=2)
