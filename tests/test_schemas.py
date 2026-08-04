import sys
import unittest
from pathlib import Path

from pydantic import ValidationError

sys.path.insert(0, str(Path("services").resolve()))

from schemas import AnalysisDraft


class AnalysisDraftSchemaTests(unittest.TestCase):
    def valid_payload(self):
        return {
            "evidence": {
                "key_facts": "该笔交易金额可能超过制度规定的审批阈值，需核实审批链路。",
            },
            "selected_evidence_ids": [
                "audit_policy.txt_015",
            ],
            "suggested_actions": [
                "核实财务总监审批记录。",
                "比对审批时间与交易时间。",
            ],
            "needs_human_review": True,
        }

    def test_valid_payload_is_accepted(self):
        draft = AnalysisDraft.model_validate(self.valid_payload())

        self.assertEqual(
            draft.selected_evidence_ids,
            ["audit_policy.txt_015"],
        )
        self.assertTrue(draft.needs_human_review)

    def test_human_review_cannot_be_false(self):
        payload = self.valid_payload()
        payload["needs_human_review"] = False

        with self.assertRaises(ValidationError):
            AnalysisDraft.model_validate(payload)

    def test_evidence_id_count_cannot_exceed_top_k(self):
        payload = self.valid_payload()
        payload["selected_evidence_ids"] = [
            "chunk_1",
            "chunk_2",
            "chunk_3",
            "chunk_4",
        ]

        with self.assertRaises(ValidationError):
            AnalysisDraft.model_validate(payload)

    def test_key_facts_cannot_be_empty(self):
        payload = self.valid_payload()
        payload["evidence"]["key_facts"] = ""

        with self.assertRaises(ValidationError):
            AnalysisDraft.model_validate(payload)

    def test_action_list_cannot_be_empty(self):
        payload = self.valid_payload()
        payload["suggested_actions"] = []

        with self.assertRaises(ValidationError):
            AnalysisDraft.model_validate(payload)

    def test_unknown_field_is_rejected(self):
        payload = self.valid_payload()

        # 模型不允许额外伪造制度来源或原文。
        payload["retrieved_policy"] = {
            "source": "模型伪造的来源",
            "content": "模型伪造的制度原文",
        }

        with self.assertRaises(ValidationError):
            AnalysisDraft.model_validate(payload)

if __name__ == "__main__":
    unittest.main(verbosity=2)
