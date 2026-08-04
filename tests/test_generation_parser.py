import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "services"))

from citations import CitationValidationError
from generation_parser import LLMResponseFormatError, build_verified_result


CHUNKS = [
    {
        "chunk_id": "audit_policy.txt_015",
        "source": "audit_policy.txt",
        "section": "第三章 金额审批权限",
        "text": "单笔交易金额超过50万元的，需经财务总监审批后方可执行。",
        "vector_score": 0.92,
        "bm25_score": 8.6,
        "rrf_score": 0.032,
    },
    {
        "chunk_id": "compliance_manual.txt_012",
        "source": "compliance_manual.txt",
        "section": "第五章 合同与采购合规",
        "text": "单笔采购金额达到审批阈值的，须签订正式合同。",
        "vector_score": 0.81,
        "bm25_score": 2.3,
        "rrf_score": 0.031,
    },
]


def valid_response(**overrides):
    payload = {
        "evidence": {"key_facts": "该交易金额较大，可能需要核实审批流程。"},
        "selected_evidence_ids": ["audit_policy.txt_015"],
        "suggested_actions": ["核实该交易的财务总监审批记录。"],
        "needs_human_review": True,
    }
    payload.update(overrides)
    return json.dumps(payload, ensure_ascii=False)


class GenerationParserTests(unittest.TestCase):
    def test_valid_response_is_verified_and_cited(self):
        result = build_verified_result(valid_response(), CHUNKS)

        self.assertTrue(result["needs_human_review"])
        self.assertEqual(result["selected_evidence_ids"], ["audit_policy.txt_015"])
        self.assertEqual(len(result["retrieved_policies"]), 1)

        citation = result["retrieved_policies"][0]
        self.assertEqual(citation["source"], "audit_policy.txt")
        self.assertEqual(citation["section"], "第三章 金额审批权限")
        self.assertEqual(citation["content"], CHUNKS[0]["text"])

    def test_markdown_json_is_accepted(self):
        raw = f"```json\n{valid_response()}\n```"
        result = build_verified_result(raw, CHUNKS)
        self.assertEqual(result["evidence"]["key_facts"], "该交易金额较大，可能需要核实审批流程。")

    def test_unknown_evidence_id_is_rejected(self):
        raw = valid_response(selected_evidence_ids=["invented_policy_999"])
        with self.assertRaises(CitationValidationError):
            build_verified_result(raw, CHUNKS)

    def test_extra_field_is_rejected_by_schema(self):
        raw = valid_response(confidence=0.99)
        with self.assertRaises(LLMResponseFormatError):
            build_verified_result(raw, CHUNKS)

    def test_invalid_json_is_rejected(self):
        with self.assertRaises(LLMResponseFormatError):
            build_verified_result("这不是 JSON", CHUNKS)


if __name__ == "__main__":
    unittest.main(verbosity=2)
