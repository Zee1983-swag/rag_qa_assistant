import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "services"))
sys.path.insert(0, str(ROOT / "config"))

SPEC = importlib.util.spec_from_file_location(
    "batch_process",
    ROOT / "scripts" / "05_batch_process.py",
)
batch_process = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(batch_process)


class ReportV2Tests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.original_out_md = batch_process.OUT_MD
        batch_process.OUT_MD = str(
            Path(self.temp_dir.name) / "risk_analysis_report.md"
        )

    def tearDown(self):
        batch_process.OUT_MD = self.original_out_md
        self.temp_dir.cleanup()

    def read_report(self):
        return Path(batch_process.OUT_MD).read_text(encoding="utf-8")

    def test_report_shows_verified_multiple_citations(self):
        results = [
            {
                "trans_id": "TX-001",
                "risk_type": "金额异常",
                "trigger_reason": "单笔交易金额超过50万元",
                "evidence": {
                    "amount": 600000,
                    "key_facts": "金额较大，可能需要核实审批流程。",
                },
                "retrieved_policies": [
                    {
                        "chunk_id": "audit_policy.txt_015",
                        "source": "audit_policy.txt",
                        "section": "第三章 金额审批权限",
                        "content": "单笔交易金额超过50万元的，需经财务总监审批。",
                        "vector_score": 0.92,
                        "bm25_score": 8.6,
                        "rrf_score": 0.032,
                    },
                    {
                        "chunk_id": "compliance_manual.txt_012",
                        "source": "compliance_manual.txt",
                        "section": "第五章 合同与采购合规",
                        "content": "采购金额达到审批阈值的，须签订正式合同。",
                        "vector_score": 0.81,
                        "bm25_score": 2.3,
                        "rrf_score": 0.031,
                    },
                ],
                "suggested_actions": [
                    "核实财务总监审批记录。",
                    "调取对应合同材料。",
                ],
                "needs_human_review": True,
                "failed": False,
            }
        ]

        batch_process.write_markdown(
            results=results,
            total=1,
            success=1,
            insufficient=0,
            failed=0,
        )
        report = self.read_report()

        self.assertIn("已验证制度引用", report)
        self.assertIn("audit_policy.txt_015", report)
        self.assertIn("compliance_manual.txt_012", report)
        self.assertIn("向量=0.9200", report)
        self.assertIn("BM25=8.6000", report)
        self.assertIn("RRF=0.032000", report)
        self.assertIn("核实财务总监审批记录。", report)
        self.assertIn("人工复核**：是", report)
        self.assertNotIn("置信度", report)

    def test_report_marks_insufficient_evidence(self):
        results = [
            {
                "trans_id": "TX-002",
                "risk_type": "重复付款",
                "trigger_reason": "输入与审计无关",
                "evidence": {
                    "amount": 100,
                    "key_facts": "查询缺少审计风险领域关键词。",
                },
                "retrieved_policies": [],
                "suggested_actions": ["人工复核该风险记录。"],
                "needs_human_review": True,
                "failed": False,
            }
        ]

        batch_process.write_markdown(
            results=results,
            total=1,
            success=0,
            insufficient=1,
            failed=0,
        )
        report = self.read_report()

        self.assertIn("证据不足（需人工复核，1 条）", report)
        self.assertIn("无（证据不足或处理失败）", report)

    def test_report_does_not_list_failed_item_as_insufficient(self):
        results = [
            {
                "trans_id": "TX-003",
                "risk_type": "高频交易",
                "trigger_reason": "测试生成失败",
                "evidence": {
                    "amount": 200,
                    "key_facts": "LLM 生成或引用校验失败。",
                },
                "retrieved_policies": [],
                "suggested_actions": ["人工复核该风险记录。"],
                "needs_human_review": True,
                "failed": True,
            }
        ]

        batch_process.write_markdown(
            results=results,
            total=1,
            success=0,
            insufficient=0,
            failed=1,
        )
        report = self.read_report()

        self.assertNotIn("证据不足（需人工复核，1 条）", report)


if __name__ == "__main__":
    unittest.main(verbosity=2)
