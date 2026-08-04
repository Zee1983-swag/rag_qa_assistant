# services/bm25_retriever.py
# 审计制度的关键词检索器。
# 后续会与向量检索做 RRF 融合，而不是替代向量检索。

import json
import re
from pathlib import Path

import jieba
from rank_bm25 import BM25Okapi


# 将业务短语加入中文分词词典，避免被拆成过细的小词。
AUDIT_TERMS = (
    "金额异常",
    "重复付款",
    "异常时间",
    "高频交易",
    "审批阈值",
    "财务总监",
    "三单匹配",
    "拆分采购",
    "非工作时间",
    "设备指纹",
    "操作IP",
    "同一供应商",
)

for term in AUDIT_TERMS:
    jieba.add_word(term)


def tokenize(text):
    """
    将文本切成 BM25 使用的词元。

    例：
    “同一供应商30天内相同金额重复付款”
    → ["同一供应商", "30", "天内", "相同", "金额", "重复付款"]
    """
    normalized = str(text or "").lower()

    return [
        token.strip()
        for token in jieba.lcut(normalized)
        if token.strip() and re.search(r"[\u4e00-\u9fff0-9a-zA-Z]", token)
    ]


class BM25Retriever:
    """从 chunks.json 建立轻量级关键词索引。"""

    def __init__(self, chunks):
        if not chunks:
            raise ValueError("BM25Retriever 需要至少一个知识库切片")

        self.chunks = chunks
        tokenized_corpus = [tokenize(chunk["text"]) for chunk in chunks]
        self.index = BM25Okapi(tokenized_corpus)

    @classmethod
    def from_json(cls, chunks_path="data/vector_store/chunks.json"):
        """读取现有切片清单，建立内存中的 BM25 索引。"""
        path = Path(chunks_path)

        with path.open("r", encoding="utf-8") as f:
            chunks = json.load(f)

        return cls(chunks)

    def search(self, query, top_k=3):
        """
        返回 BM25 得分最高的 top_k 个切片。

        bm25_score 只能用于 BM25 内部排序，
        不能直接与向量相似度比较。
        """
        query_tokens = tokenize(query)

        if not query_tokens:
            return []

        scores = self.index.get_scores(query_tokens)

        ranked_indexes = sorted(
            range(len(self.chunks)),
            key=lambda index: scores[index],
            reverse=True,
        )[:top_k]

        results = []

        for index in ranked_indexes:
            chunk = dict(self.chunks[index])
            chunk["bm25_score"] = round(float(scores[index]), 4)
            results.append(chunk)

        return results
