import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path


# 测试时不使用真实 API Key，也不会发起网络请求。
os.environ.setdefault("OPENAI_API_KEY", "test-key")


def load_incremental_module():
    """加载文件名以数字开头的脚本。普通 import 无法直接导入 08_incremental_update.py。"""
    script_path = Path("scripts/08_incremental_update.py").resolve()

    spec = importlib.util.spec_from_file_location(
        "incremental_update_module",
        script_path,
    )

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeCollection:
    """
    一个内存版的 Chroma Collection。
    只实现本测试需要的 delete、add、count 方法。
    """

    def __init__(self, records):
        self.records = records
        self.delete_calls = []

    def delete(self, ids):
        self.delete_calls.append(list(ids))
        for chunk_id in ids:
            self.records.pop(chunk_id, None)

    def add(self, ids, embeddings, documents, metadatas):
        for chunk_id, embedding, document, metadata in zip(
            ids, embeddings, documents, metadatas
        ):
            # 如果旧向量没有先删除，使用同一个 ID 添加时应该失败。
            if chunk_id in self.records:
                raise AssertionError(
                    f"旧向量未删除，不能重复写入 chunk_id={chunk_id}"
                )

            self.records[chunk_id] = {
                "embedding": embedding,
                "document": document,
                "metadata": metadata,
            }

    def count(self):
        return len(self.records)


class FakeChromaClient:
    """让脚本以为自己连接到了 Chroma，实际返回内存中的 FakeCollection。"""

    def __init__(self, collection):
        self.collection = collection

    def get_collection(self, name):
        return self.collection

    def get_or_create_collection(self, name, metadata=None):
        return self.collection


class IncrementalUpdateTests(unittest.TestCase):
    def test_modified_document_deletes_old_vector_before_adding_new_vector(self):
        module = load_incremental_module()

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            kb_dir = root / "knowledge_base"
            vector_dir = root / "vector_store"
            kb_dir.mkdir()
            vector_dir.mkdir()

            file_name = "audit_policy.txt"
            old_text = "第三章 金额审批权限\n单笔交易金额超过50万元，需审批。"
            new_text = "第三章 金额审批权限\n单笔交易金额超过80万元，需审批。"

            # 模拟：向量库中已有旧制度，且 chunks.json 也记录旧制度。
            old_chunk = {
                "chunk_id": "audit_policy.txt_000",
                "source": file_name,
                "section": "第三章 金额审批权限",
                "text": old_text,
            }

            (kb_dir / file_name).write_text(new_text, encoding="utf-8")

            chunks_path = vector_dir / "chunks.json"
            chunks_path.write_text(
                json.dumps([old_chunk], ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            fingerprints_path = vector_dir / "doc_fingerprints.json"
            fingerprints_path.write_text(
                json.dumps(
                    {file_name: module.compute_fingerprint(old_text)},
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

            collection = FakeCollection({
                old_chunk["chunk_id"]: {
                    "embedding": [0.1, 0.2],
                    "document": old_text,
                    "metadata": {
                        "source": file_name,
                        "section": old_chunk["section"],
                    },
                }
            })

            # 保存原始全局配置，测试结束后恢复。
            original_values = {
                "KB_DIR": module.KB_DIR,
                "DB_PATH": module.DB_PATH,
                "FINGERPRINT_PATH": module.FINGERPRINT_PATH,
                "CHUNKS_PATH": module.CHUNKS_PATH,
                "persistent_client": module.chromadb.PersistentClient,
                "get_embeddings": module.get_embeddings,
            }

            try:
                # 将真实脚本重定向到临时目录，并替换网络/数据库依赖。
                module.KB_DIR = str(kb_dir)
                module.DB_PATH = str(vector_dir / "fake_chroma")
                module.FINGERPRINT_PATH = str(fingerprints_path)
                module.CHUNKS_PATH = str(chunks_path)
                module.chromadb.PersistentClient = (
                    lambda path: FakeChromaClient(collection)
                )
                module.get_embeddings = (
                    lambda texts: [[float(index)] for index, _ in enumerate(texts)]
                )

                module.incremental_update()

            finally:
                module.KB_DIR = original_values["KB_DIR"]
                module.DB_PATH = original_values["DB_PATH"]
                module.FINGERPRINT_PATH = original_values["FINGERPRINT_PATH"]
                module.CHUNKS_PATH = original_values["CHUNKS_PATH"]
                module.chromadb.PersistentClient = original_values["persistent_client"]
                module.get_embeddings = original_values["get_embeddings"]

            # 断言 1：旧 ID 在写入新向量前被显式删除。
            self.assertEqual(
                collection.delete_calls,
                [["audit_policy.txt_000"]],
            )

            # 断言 2：最终向量库只有一条，且内容是新制度。
            self.assertEqual(collection.count(), 1)
            self.assertEqual(
                collection.records["audit_policy.txt_000"]["document"],
                new_text,
            )

            # 断言 3：新的 chunks.json 也只保存新制度。
            final_chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
            self.assertEqual(len(final_chunks), 1)
            self.assertEqual(final_chunks[0]["text"], new_text)
            self.assertNotIn("50万元", final_chunks[0]["text"])
            self.assertIn("80万元", final_chunks[0]["text"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
