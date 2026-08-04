# services/citations.py
# 验证模型选择的证据 ID，并由程序回填真实制度原文。

from schemas import AnalysisDraft


class CitationValidationError(ValueError):
    """模型选择了本次检索结果之外的证据时抛出。"""


def attach_citations(draft: AnalysisDraft, retrieved_chunks: list[dict]) -> list[dict]:
    """
    根据 selected_evidence_ids 从本次检索 Top-K 中回填引用。

    返回的 content 永远来自 retrieved_chunks，
    而不是来自 LLM 输出。
    """
    chunk_by_id = {
        chunk["chunk_id"]: chunk
        for chunk in retrieved_chunks
    }

    selected_ids = draft.selected_evidence_ids

    if len(selected_ids) != len(set(selected_ids)):
        raise CitationValidationError("模型重复选择了同一条证据")

    citations = []

    for chunk_id in selected_ids:
        if chunk_id not in chunk_by_id:
            raise CitationValidationError(
                f"模型引用了本次检索结果之外的证据：{chunk_id}"
            )

        chunk = chunk_by_id[chunk_id]

        citations.append({
            "chunk_id": chunk["chunk_id"],
            "source": chunk.get("source", ""),
            "section": chunk.get("section", ""),
            "content": chunk["text"],
            "vector_score": chunk.get("vector_score", 0.0),
            "bm25_score": chunk.get("bm25_score", 0.0),
            "rrf_score": chunk.get("rrf_score", 0.0),
        })

    return citations
