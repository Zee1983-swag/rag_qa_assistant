import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path("services").resolve()))

from citations import CitationValidationError, attach_citations
from schemas import AnalysisDraft


class CitationTests(unittest.TestCase):
    def setUp(self):
        self.retrieved_chunks = [
            {
                "chunk_id": "audit_policy.txt_015",
                "source": "audit_policy.txt",
                "section": "第三章 金额审批权限",
                "text": "单笔交易金额超过50万元的，需经财务总监审批。",
                "vector_score": 0.91,
                "bm25_score": 9.65,
                "rrf_score": 0.0327,
            },
            {
                "chunk_id": "compliance_manual.txt_009",
                "source": "compliance_manual.txt",
                "section": "第二章 授权管理",
                "text": "所有交易须经适当授权，授权记录须可追溯。",
                "vector_score": 0.85,
                "bm25_score": 0.84,
                "rrf_score": 0.0312,
            },
        ]

    def make_draft(self, evidence_ids):
        return AnalysisDraft.model_validate({
            "evidence": {
                "key_facts": "该交易可能需要核实审批记录。",
            },
            "selected_evidence_ids": evidence_ids,
            "suggested_actions": [
                "核实审批记录。",
            ],
            "needs_human_review": True,
        })

    def test_valid_evidence_id_returns_original_content(self):
        draft = self.make_draft(["audit_policy.txt_015"])

        citations = attach_citations(draft, self.retrieved_chunks)

        self.assertEqual(len(citations), 1)
        self.assertEqual(citations[0]["source"], "audit_policy.txt")
        self.assertEqual(
            citations[0]["content"],
            "单笔交易金额超过50万元的，需经财务总监审批。",
        )
        self.assertEqual(citations[0]["bm25_score"], 9.65)

    def test_unknown_evidence_id_is_rejected(self):
        draft = self.make_draft(["fabricated_policy.txt_999"])

        with self.assertRaises(CitationValidationError):
            attach_citations(draft, self.retrieved_chunks)

    def test_duplicate_evidence_id_is_rejected(self):
        draft = self.make_draft([
            "audit_policy.txt_015",
            "audit_policy.txt_015",
        ])

        with self.assertRaises(CitationValidationError):
            attach_citations(draft, self.retrieved_chunks)

    def test_multiple_selected_ids_keep_selected_order(self):
        draft = self.make_draft([
            "compliance_manual.txt_009",
            "audit_policy.txt_015",
        ])

        citations = attach_citations(draft, self.retrieved_chunks)

        self.assertEqual(
            [item["chunk_id"] for item in citations],
            [
                "compliance_manual.txt_009",
                "audit_policy.txt_015",
            ],
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
