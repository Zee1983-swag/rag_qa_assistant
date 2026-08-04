import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WEB_APP_PATH = ROOT / "scripts" / "10_web_app.py"


class WebAppV2ContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.code = WEB_APP_PATH.read_text(encoding="utf-8")

    def test_web_uses_v2_prompt_and_verified_result(self):
        self.assertIn("SYSTEM_PROMPT_V2", self.code)
        self.assertIn("USER_PROMPT_TEMPLATE_V2", self.code)
        self.assertIn("build_verified_result", self.code)

    def test_web_uses_hybrid_retrieval(self):
        self.assertIn("BM25Retriever", self.code)
        self.assertIn("HybridRetriever", self.code)
        self.assertIn("rrf_score", self.code)

    def test_domain_gate_runs_before_retrieval(self):
        gate_position = self.code.find("precheck_reason = get_abstention_reason")
        retrieve_position = self.code.find("chunks = retrieve(query)")

        self.assertNotEqual(gate_position, -1)
        self.assertNotEqual(retrieve_position, -1)
        self.assertLess(gate_position, retrieve_position)

    def test_ui_shows_verified_citations_not_llm_confidence(self):
        self.assertIn('result.get("retrieved_policies"', self.code)
        self.assertIn("已验证制度引用", self.code)
        self.assertNotIn('st.metric("置信度"', self.code)
        self.assertNotIn('"置信度": r.get("confidence"', self.code)

    def test_ui_keeps_human_review_visible(self):
        self.assertIn("需人工复核", self.code)
        self.assertIn('result.get("needs_human_review", True)', self.code)


if __name__ == "__main__":
    unittest.main(verbosity=2)
