import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path("config").resolve()))

from prompts import SYSTEM_PROMPT_V2, USER_PROMPT_TEMPLATE_V2


class PromptV2Tests(unittest.TestCase):
    def build_prompt(self):
        return USER_PROMPT_TEMPLATE_V2.format(
            trans_id="TXN_TEST_001",
            risk_type="金额异常",
            risk_level="高",
            trigger_reason="金额超过审批阈值",
            amount=626639.12,
            supplier="供应商G公司",
            trans_date="2026-04-23",
            chunk_id1="audit_policy.txt_015",
            score1=0.91,
            source1="audit_policy.txt · 第三章 金额审批权限",
            content1="单笔交易金额超过50万元的，需经财务总监审批。",
            chunk_id2="compliance_manual.txt_009",
            score2=0.85,
            source2="compliance_manual.txt · 第二章 授权管理",
            content2="所有交易须经适当授权。",
            chunk_id3="case_library.txt_006",
            score3=0.80,
            source3="case_library.txt · 案例二",
            content3="历史案例说明。",
        )

    def test_system_prompt_requires_evidence_ids(self):
        self.assertIn("selected_evidence_ids", SYSTEM_PROMPT_V2)
        self.assertIn("chunk_id", SYSTEM_PROMPT_V2)

    def test_system_prompt_forbids_generated_policy_content(self):
        self.assertIn("不得输出制度来源", SYSTEM_PROMPT_V2)
        self.assertIn("不得输出制度原文", SYSTEM_PROMPT_V2)

    def test_system_prompt_requires_human_review(self):
        self.assertIn("needs_human_review 必须为 true", SYSTEM_PROMPT_V2)

    def test_template_renders_real_chunk_ids(self):
        prompt = self.build_prompt()

        self.assertIn("audit_policy.txt_015", prompt)
        self.assertIn("compliance_manual.txt_009", prompt)
        self.assertIn("case_library.txt_006", prompt)

    def test_template_renders_new_json_contract(self):
        prompt = self.build_prompt()

        self.assertIn('"selected_evidence_ids"', prompt)
        self.assertIn('"suggested_actions"', prompt)
        self.assertNotIn('"retrieved_policy"', prompt)


if __name__ == "__main__":
    unittest.main(verbosity=2)
