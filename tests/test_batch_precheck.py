import importlib.util
import os
import unittest
from pathlib import Path

os.environ.setdefault("OPENAI_API_KEY", "test-key")


def load_batch_module():
    script_path = Path("scripts/05_batch_process.py").resolve()

    spec = importlib.util.spec_from_file_location(
        "batch_process_module",
        script_path,
    )

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeCollection:
    """模拟真实 Chroma query 返回格式，包含 ids、正文、元数据和距离。"""

    def __init__(self):
        self.query_called = False

    def query(self, query_embeddings, n_results):
        self.query_called = True

        return {
            "ids": [[
                "audit_policy.txt_015",
                "compliance_manual.txt_009",
                "compliance_manual.txt_012",
            ]],
            "documents": [[
                "第三章 金额审批权限\n单笔交易金额超过50万元，需审批。",
                "第二章 授权管理\n所有交易须经适当授权。",
                "第五章 合同与采购合规\n采购金额达到阈值须走审批流程。",
            ]],
            "metadatas": [[
                {
                    "source": "audit_policy.txt",
                    "section": "第三章 金额审批权限",
                },
                {
                    "source": "compliance_manual.txt",
                    "section": "第二章 授权管理",
                },
                {
                    "source": "compliance_manual.txt",
                    "section": "第五章 合同与采购合规",
                },
            ]],
            "distances": [[0.2, 0.4, 0.6]],
        }


class FakeBM25Retriever:
    """模拟 BM25 结果；第一条与向量检索第一条相同。"""

    def __init__(self):
        self.search_called = False

    def search(self, query, top_k):
        self.search_called = True

        return [
            {
                "chunk_id": "audit_policy.txt_015",
                "source": "audit_policy.txt",
                "section": "第三章 金额审批权限",
                "text": "第三章 金额审批权限\n单笔交易金额超过50万元，需审批。",
                "bm25_score": 10.0,
            },
            {
                "chunk_id": "compliance_manual.txt_012",
                "source": "compliance_manual.txt",
                "section": "第五章 合同与采购合规",
                "text": "第五章 合同与采购合规\n采购金额达到阈值须走审批流程。",
                "bm25_score": 7.0,
            },
            {
                "chunk_id": "compliance_manual.txt_009",
                "source": "compliance_manual.txt",
                "section": "第二章 授权管理",
                "text": "第二章 授权管理\n所有交易须经适当授权。",
                "bm25_score": 5.0,
            },
        ]


class BatchPrecheckTests(unittest.TestCase):
    def test_irrelevant_risk_is_rejected_before_embedding(self):
        module = load_batch_module()

        def embedding_must_not_be_called(text):
            raise AssertionError("预检拒答后不应调用 Embedding API")

        module.get_embedding = embedding_must_not_be_called

        risk = {
            "trans_id": "TEST_IRRELEVANT",
            "risk_type": "未知类型",
            "trigger_reason": "量子计算机天气预报火星探测",
        }

        chunks, reason = module.retrieve_one(
            collection=None,
            risk=risk,
        )

        self.assertEqual(chunks, [])
        self.assertIsNotNone(reason)
        self.assertIn("证据不足", reason)

    def test_normal_risk_passes_precheck_and_uses_hybrid_retrieval(self):
        module = load_batch_module()

        embedding_called = {"value": False}

        def fake_embedding(text):
            embedding_called["value"] = True
            return [0.1, 0.2, 0.3]

        module.get_embedding = fake_embedding

        collection = FakeCollection()
        bm25_retriever = FakeBM25Retriever()
        module.get_bm25_retriever = lambda: bm25_retriever

        risk = {
            "trans_id": "TEST_AMOUNT",
            "risk_type": "金额异常",
            "trigger_reason": "金额 626639.12 超过审批阈值",
        }

        chunks, reason = module.retrieve_one(
            collection=collection,
            risk=risk,
        )

        self.assertIsNone(reason)
        self.assertTrue(embedding_called["value"])
        self.assertTrue(collection.query_called)
        self.assertTrue(bm25_retriever.search_called)

        self.assertEqual(len(chunks), 3)
        self.assertEqual(chunks[0]["chunk_id"], "audit_policy.txt_015")
        # V2 将文件名与章节分开保存，便于程序按 chunk_id 回填真实引用。
        self.assertEqual(chunks[0]["source"], "audit_policy.txt")
        self.assertEqual(chunks[0]["section"], "第三章 金额审批权限")
        self.assertEqual(
            chunks[0]["source_display"],
            "audit_policy.txt · 第三章 金额审批权限",
        )

        # 验证批处理入口确实保留了混合检索可观测字段。
        self.assertEqual(chunks[0]["vector_score"], 0.9)
        self.assertEqual(chunks[0]["bm25_score"], 10.0)
        self.assertIn("vector", chunks[0]["rank_details"])
        self.assertIn("bm25", chunks[0]["rank_details"])


if __name__ == "__main__":
    unittest.main(verbosity=2)

