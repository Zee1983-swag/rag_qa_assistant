import importlib.util
import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "services"))
sys.path.insert(0, str(ROOT / "config"))

SPEC = importlib.util.spec_from_file_location(
    "batch_process",
    ROOT / "scripts" / "05_batch_process.py",
)
batch_process = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(batch_process)


RISK = {
    "trans_id": "TX-001",
    "risk_type": "金额异常",
    "risk_level": "高",
    "trigger_reason": "单笔交易金额超过50万元",
    "amount": 600000,
    "supplier": "示例供应商",
    "trans_date": "2026-08-04",
}

CHUNKS = [
    {
        "chunk_id": "audit_policy.txt_015",
        "source": "audit_policy.txt",
        "section": "第三章 金额审批权限",
        "source_display": "audit_policy.txt · 第三章 金额审批权限",
        "text": "单笔交易金额超过50万元的，需经财务总监审批后方可执行。",
        "score": 0.92,
        "vector_score": 0.92,
        "bm25_score": 8.6,
        "rrf_score": 0.032,
    }
]


class FakeCompletions:
    def __init__(self, content):
        self.content = content
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=self.content)
                )
            ],
            usage=None,
        )


def install_fake_llm(content):
    fake_completions = FakeCompletions(content)
    batch_process.client = SimpleNamespace(
        chat=SimpleNamespace(completions=fake_completions)
    )
    return fake_completions


def valid_llm_response(**overrides):
    payload = {
        "evidence": {
            "key_facts": "该交易金额较大，可能需要核实审批流程。"
        },
        "selected_evidence_ids": ["audit_policy.txt_015"],
        "suggested_actions": ["核实财务总监审批记录。"],
        "needs_human_review": True,
    }
    payload.update(overrides)
    return json.dumps(payload, ensure_ascii=False)


class BatchGenerationV2Tests(unittest.TestCase):
    def test_v2_prompt_and_verified_citation_are_used(self):
        fake = install_fake_llm(valid_llm_response())

        result, score = batch_process.generate_one(RISK, CHUNKS)

        self.assertEqual(score, 0.92)
        self.assertEqual(len(fake.calls), 1)

        request = fake.calls[0]
        self.assertEqual(
            request["messages"][0]["content"],
            batch_process.SYSTEM_PROMPT_V2,
        )
        self.assertIn(
            "audit_policy.txt_015",
            request["messages"][1]["content"],
        )

        self.assertEqual(
            result["retrieved_policy"]["content"],
            CHUNKS[0]["text"],
        )
        self.assertEqual(
            result["retrieved_policy"]["source"],
            "audit_policy.txt",
        )
        self.assertTrue(result["needs_human_review"])
        self.assertNotIn("confidence", result)

    def test_forged_evidence_id_becomes_human_review_result(self):
        fake = install_fake_llm(
            valid_llm_response(
                selected_evidence_ids=["forged_policy_999"]
            )
        )

        result, _ = batch_process.generate_one(RISK, CHUNKS)

        self.assertEqual(len(fake.calls), 1)
        self.assertTrue(result["needs_human_review"])
        self.assertTrue(result["failed"])
        self.assertEqual(result["retrieved_policies"], [])
        self.assertIn("本次检索结果之外", result["evidence"]["key_facts"])

    def test_extra_llm_field_becomes_human_review_result(self):
        fake = install_fake_llm(
            valid_llm_response(confidence=0.99)
        )

        result, _ = batch_process.generate_one(RISK, CHUNKS)

        self.assertTrue(result["needs_human_review"])
        self.assertTrue(result["failed"])
        self.assertIn("不符合 AnalysisDraft 约束", result["evidence"]["key_facts"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
