# services/rank_fusion.py
# Reciprocal Rank Fusion（RRF，倒数排名融合）
# 用于融合向量检索和 BM25 检索的结果。

from collections import defaultdict


def reciprocal_rank_fusion(rankings, k=60):
    """
    融合多个有序检索列表。

    参数：
    - rankings：多个检索结果列表；
      每个结果必须包含唯一的 chunk_id。
    - k：平滑常数。k 越大，排名差异的影响越平缓。

    返回：
    - 根据 rrf_score 从高到低排序的去重结果；
    - 每条结果包含 rrf_score 和各检索器中的 rank_details。

    公式：
    RRF(d) = Σ 1 / (k + rank_i(d))
    """
    scores = defaultdict(float)
    first_payload = {}
    rank_details = defaultdict(dict)

    for retriever_name, results in rankings.items():
        for rank, item in enumerate(results, start=1):
            chunk_id = item["chunk_id"]

            # 保存第一次见到的原始切片信息。
            if chunk_id not in first_payload:
                # 第一次出现时，保存切片正文、来源和当前检索器分数。
                first_payload[chunk_id] = dict(item)
            else:
                # 同一 chunk 被另一检索器命中时，
                # 保留第一次的正文信息，并合并另一检索器的分数。
                for score_field in ("vector_score", "bm25_score"):
                    if score_field in item:
                        first_payload[chunk_id][score_field] = item[score_field]

            # 记录该切片在当前检索器中的排名。
            rank_details[chunk_id][retriever_name] = rank

            # 倒数排名加分：排名越靠前，加分越高。
            scores[chunk_id] += 1 / (k + rank)

    fused_results = []

    for chunk_id, score in scores.items():
        result = dict(first_payload[chunk_id])
        result["rrf_score"] = round(score, 6)
        result["rank_details"] = rank_details[chunk_id]
        fused_results.append(result)

    # 分数相同时按 chunk_id 排序，使结果稳定、可复现。
    return sorted(
        fused_results,
        key=lambda item: (-item["rrf_score"], item["chunk_id"]),
    )
