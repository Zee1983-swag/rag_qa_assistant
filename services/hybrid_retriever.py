# services/hybrid_retriever.py
# 混合检索：向量检索 + BM25 检索 + RRF 排名融合。

from rank_fusion import reciprocal_rank_fusion


class HybridRetriever:
    """
    通过依赖注入接收：
    - collection：Chroma collection；
    - embed_query：把 query 转为向量的函数；
    - bm25_retriever：关键词检索器。

    这样业务代码可以调用真实 API；
    测试代码可以传入 FakeCollection 和 fake embedding。
    """

    def __init__(
        self,
        collection,
        embed_query,
        bm25_retriever,
        candidate_k=6,
        rrf_k=60,
    ):
        self.collection = collection
        self.embed_query = embed_query
        self.bm25_retriever = bm25_retriever
        self.candidate_k = candidate_k
        self.rrf_k = rrf_k

    def retrieve(self, query, top_k=3):
        """
        返回最终融合后的 top_k 切片。

        每条结果包括：
        - chunk_id、source、section、text；
        - vector_score：向量相似度，仅作展示与安全阈值参考；
        - bm25_score：关键词检索得分；
        - rrf_score：最终融合排序得分；
        - rank_details：分别在 vector / bm25 中的排名。
        """
        query_embedding = self.embed_query(query)

        vector_response = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=self.candidate_k,
        )

        vector_ids = vector_response["ids"][0]
        vector_docs = vector_response["documents"][0]
        vector_metas = vector_response["metadatas"][0]
        vector_distances = vector_response["distances"][0]

        vector_results = []

        for chunk_id, doc, meta, distance in zip(
            vector_ids,
            vector_docs,
            vector_metas,
            vector_distances,
        ):
            # Chroma cosine distance 通常位于 0 到 2。
            vector_score = max(0.0, min(1.0, 1 - distance / 2))

            vector_results.append({
                "chunk_id": chunk_id,
                "source": meta.get("source", ""),
                "section": meta.get("section", ""),
                "text": doc,
                "vector_score": round(vector_score, 4),
            })

        bm25_results = self.bm25_retriever.search(
            query,
            top_k=self.candidate_k,
        )

        fused_results = reciprocal_rank_fusion(
            {
                "vector": vector_results,
                "bm25": bm25_results,
            },
            k=self.rrf_k,
        )

        normalized_results = []

        for item in fused_results[:top_k]:
            normalized_results.append({
                "chunk_id": item["chunk_id"],
                "source": item["source"],
                "section": item["section"],
                "text": item["text"],
                "vector_score": item.get("vector_score", 0.0),
                "bm25_score": item.get("bm25_score", 0.0),
                "rrf_score": item["rrf_score"],
                "rank_details": item["rank_details"],
            })

        return normalized_results
