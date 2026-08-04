# scripts/05_batch_process.py
# 第五步：串联检索+生成，批量处理，限流，生成 Markdown 可读报告
import os
import time
import json
import csv
import chromadb
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
import sys

sys.path.insert(0, "config")
sys.path.insert(0, "services")
from prompts import (SYSTEM_PROMPT_V2, USER_PROMPT_TEMPLATE_V2, RELEVANCE_THRESHOLD,
                     TOP_K, BATCH_RATE_SECONDS)
from cost_tracker import CostTracker
from relevance_gate import get_abstention_reason
from bm25_retriever import BM25Retriever
from hybrid_retriever import HybridRetriever
from generation_parser import build_verified_result

load_dotenv()

RISK_ITEMS_PATH = "data/input/risk_items.csv"
DB_PATH = "data/vector_store/chroma_db"
COLLECTION_NAME = "audit_knowledge"
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "embedding-3")
LLM_MODEL = os.getenv("LLM_MODEL", "glm-4-flash")

OUT_JSON = "output/structured_risk_report.json"
OUT_MD = "output/risk_analysis_report.md"
OUT_LOG = "output/retrieval_log.csv"
OUT_COST = "output/cost_report.json"
tracker = CostTracker()

QUERY_MAP = {
    "金额异常": "大额交易审批权限 超过阈值 审批流程 财务总监审批",
    "重复付款": "重复付款核查 同供应商同金额 付款冲销 重复发起",
    "异常时间": "非工作时间操作 周末交易 授权管理 值班授权",
    "高频交易": "拆分采购 高频交易 供应商管理 合同审批 拆分规避审批",
}

client = OpenAI()
_bm25_retriever = None


def get_bm25_retriever():
    """
    批处理期间只从 chunks.json 建一次 BM25 索引。
    98 条风险记录复用同一个内存索引，避免重复分词和重复建库。
    """
    global _bm25_retriever

    if _bm25_retriever is None:
        _bm25_retriever = BM25Retriever.from_json()

    return _bm25_retriever

def get_embedding(text):
    resp = client.embeddings.create(input=[text], model=EMBEDDING_MODEL)
    return resp.data[0].embedding

def extract_json(text):
    if "```json" in text:
        text = text.split("```json", 1)[1].split("```", 1)[0]
    elif "```" in text:
        text = text.split("```", 1)[1].split("```", 1)[0]
    return json.loads(text.strip())

def get_collection():
    chroma_client = chromadb.PersistentClient(path=DB_PATH)

    # 持久化复用：优先使用已有向量库，避免每次删库重建
    try:
        collection = chroma_client.get_collection(name=COLLECTION_NAME)
        if collection.count() > 0:
            print(f"复用已有向量库（{collection.count()} 条切片），跳过重建")
            return collection
    except Exception:
        pass

    print("向量库不存在或为空，首次创建中...")
    collection = chroma_client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )
    with open("data/vector_store/chunks.json", "r", encoding="utf-8") as f:
        chunks = json.load(f)
    texts = [c["text"] for c in chunks]
    print(f"正在向量化 {len(texts)} 个切片...")
    batch_size = 16
    all_vectors = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        resp = client.embeddings.create(input=batch, model=EMBEDDING_MODEL)
        all_vectors.extend([d.embedding for d in resp.data])
    collection.add(
        ids=[c["chunk_id"] for c in chunks],
        embeddings=all_vectors,
        documents=texts,
        metadatas=[{"source": c["source"], "section": c["section"]} for c in chunks],
    )
    print("向量库创建完成")
    return collection

def retrieve_one(collection, risk):
    """
    对一条风险进行混合检索。

    流程：
    原始触发原因预检
        → Embedding 向量检索
        → BM25 关键词检索
        → RRF 融合
        → 返回最终 Top-K 制度证据。
    """
    risk_type = str(risk["risk_type"])
    trigger_reason = str(risk.get("trigger_reason", ""))

    query = f"{QUERY_MAP.get(risk_type, risk_type)} {trigger_reason}"

    # 只检查原始触发原因，避免扩写 query 中的审计词让无关输入误通过。
    precheck_reason = get_abstention_reason(
        query=trigger_reason,
        top_similarity=1.0,
        threshold=RELEVANCE_THRESHOLD,
    )

    if precheck_reason:
        print(f"  预检拒答：{precheck_reason}")
        return [], precheck_reason

    def get_embedding_with_tracking(text):
        vector = get_embedding(text)
        tracker.record_embedding(EMBEDDING_MODEL, len(text))
        return vector

    hybrid_retriever = HybridRetriever(
        collection=collection,
        embed_query=get_embedding_with_tracking,
        bm25_retriever=get_bm25_retriever(),
        candidate_k=max(TOP_K * 2, 6),
    )

    fused_results = hybrid_retriever.retrieve(
        query=query,
        top_k=TOP_K,
    )

    chunks = []

    for item in fused_results:
        chunks.append({
            "chunk_id": item["chunk_id"],
            "text": item["text"],
            "source": item["source"],
            "section": item["section"],
            "source_display": f"{item['source']} · {item['section']}",
            "score": item["vector_score"],
            "vector_score": item["vector_score"],
            "bm25_score": item["bm25_score"],
            "rrf_score": item["rrf_score"],
            "rank_details": item["rank_details"],
        })

    return chunks, None
def generate_one(risk, chunks, precheck_reason=None):
    """
    调用 LLM 生成“分析草稿”，随后进行 Schema 校验与引用回填。

    LLM 只能选择 chunk_id，不能直接提供制度正文。
    """
    trans_id = str(risk["trans_id"])

    def human_review_result(reason, failed=False):
        return {
            "trans_id": trans_id,
            "risk_type": str(risk["risk_type"]),
            "risk_level": str(risk.get("risk_level", "")),
            "trigger_reason": str(risk.get("trigger_reason", "")),
            "evidence": {
                "amount": risk.get("amount", 0),
                "supplier": str(risk.get("supplier", "")),
                "trans_dates": [str(risk.get("trans_date", ""))],
                "key_facts": reason,
            },
            "selected_evidence_ids": [],
            "retrieved_policies": [],
            "retrieved_policy": None,
            "suggested_actions": ["人工复核该风险记录。"],
            "suggested_action": "证据不足，建议人工复核",
            "needs_human_review": True,
            "failed": failed,
        }

    if precheck_reason:
        return human_review_result(precheck_reason), 0.0

    top_score = max(
        (chunk.get("vector_score", chunk.get("score", 0.0)) for chunk in chunks),
        default=0.0,
    )

    if not chunks or top_score < RELEVANCE_THRESHOLD:
        return human_review_result("检索到的制度片段与风险类型相关度不足"), top_score

    fields = {}
    for idx, chunk in enumerate(chunks[:3], 1):
        fields[f"chunk_id{idx}"] = chunk["chunk_id"]
        fields[f"score{idx}"] = chunk["score"]
        fields[f"source{idx}"] = chunk["source_display"]
        fields[f"content{idx}"] = chunk["text"]

    for idx in range(len(chunks[:3]) + 1, 4):
        fields[f"chunk_id{idx}"] = "无可用证据"
        fields[f"score{idx}"] = "N/A"
        fields[f"source{idx}"] = "无"
        fields[f"content{idx}"] = "无"

    user_prompt = USER_PROMPT_TEMPLATE_V2.format(
        trans_id=trans_id,
        risk_type=str(risk["risk_type"]),
        risk_level=str(risk.get("risk_level", "")),
        trigger_reason=str(risk.get("trigger_reason", "")),
        amount=risk.get("amount", 0),
        supplier=str(risk.get("supplier", "")),
        trans_date=str(risk.get("trans_date", "")),
        **fields,
    )

    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            temperature=0,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT_V2},
                {"role": "user", "content": user_prompt},
            ],
        )

        if hasattr(response, "usage") and response.usage:
            tracker.record_chat(
                LLM_MODEL,
                {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                },
            )

        verified = build_verified_result(
            response.choices[0].message.content,
            chunks,
        )

        verified["trans_id"] = trans_id
        verified["risk_type"] = str(risk["risk_type"])
        verified["risk_level"] = str(risk.get("risk_level", ""))
        verified["trigger_reason"] = str(risk.get("trigger_reason", ""))

        # 交易事实由程序补充，不依赖 LLM 复述。
        verified["evidence"] = {
            "amount": risk.get("amount", 0),
            "supplier": str(risk.get("supplier", "")),
            "trans_dates": [str(risk.get("trans_date", ""))],
            **verified["evidence"],
        }

        # 兼容旧报告格式；后续会把报告页面升级为展示多条引用。
        verified["retrieved_policy"] = verified["retrieved_policies"][0]
        verified["suggested_action"] = "\n".join(
            f"{index}. {action}"
            for index, action in enumerate(verified["suggested_actions"], 1)
        )

        return verified, top_score

    except Exception as error:
        return human_review_result(
            f"LLM 生成或引用校验失败：{error}",
            failed=True,
        ), top_score

def safe_amount(item):
    """安全提取 amount 字段并转为 float，避免类型冲突"""
    ev = item.get("evidence")
    if not isinstance(ev, dict):
        return 0.0
    amt = ev.get("amount", 0)
    try:
        return float(amt)
    except (ValueError, TypeError):
        return 0.0

def write_markdown(results, total, success, insufficient, failed):
    """生成可读报告：展示程序验证后的多条制度引用。"""
    by_type = {}
    for result in results:
        by_type.setdefault(result["risk_type"], []).append(result)

    lines = [
        "# RAG 审计风险分析报告",
        "",
        "## 一、处理概况",
        f"- 输入风险记录：{total} 条",
        f"- 成功生成：{success} 条",
        f"- 证据不足：{insufficient} 条",
        f"- 失败：{failed} 条",
        "- 所有分析结果均需人工复核。",
        "",
    ]

    order = ["金额异常", "重复付款", "异常时间", "高频交易"]
    idx = 2
    cn_num = ["二", "三", "四", "五", "六", "七"]

    for risk_type in order:
        items = by_type.get(risk_type, [])
        if not items:
            continue

        items.sort(key=safe_amount, reverse=True)
        lines.append(f"## {cn_num[idx - 2]}、{risk_type}（{len(items)} 条）")

        for number, result in enumerate(items, 1):
            evidence = result.get("evidence", {}) or {}
            policies = result.get("retrieved_policies", []) or []
            actions = result.get("suggested_actions", []) or []

            lines.append(
                f"### {number}. {result.get('trans_id')} | "
                f"金额：{evidence.get('amount', '?')} 元"
            )
            lines.append(
                f"- **触发原因**：{result.get('trigger_reason', '')}"
            )
            lines.append(
                f"- **风险摘要**：{evidence.get('key_facts', '')}"
            )

            if policies:
                lines.append("- **已验证制度引用**：")
                for policy in policies:
                    lines.append(
                        f"  - `{policy['chunk_id']}` | "
                        f"{policy.get('source', '')} · "
                        f"{policy.get('section', '')}"
                    )
                    lines.append(f"    - 原文：\"{policy['content']}\"")
                    lines.append(
                        "    - 检索信号："
                        f"向量={policy.get('vector_score', 0):.4f}，"
                        f"BM25={policy.get('bm25_score', 0):.4f}，"
                        f"RRF={policy.get('rrf_score', 0):.6f}"
                    )
            else:
                lines.append("- **已验证制度引用**：无（证据不足或处理失败）")

            if actions:
                lines.append("- **建议行动**：")
                for action in actions:
                    lines.append(f"  - {action}")
            else:
                lines.append(
                    f"- **建议行动**：{result.get('suggested_action', '建议人工复核')}"
                )

            lines.append("- **人工复核**：是")
            lines.append("")

        idx += 1

    insufficient_items = [
        result for result in results
        if not result.get("retrieved_policies") and not result.get("failed")
    ]

    if insufficient_items:
        section_name = cn_num[idx - 2] if idx - 2 < len(cn_num) else "末"
        lines.append(
            f"## {section_name}、证据不足（需人工复核，{len(insufficient_items)} 条）"
        )

        for number, result in enumerate(insufficient_items, 1):
            lines.append(
                f"### {number}. {result.get('trans_id')} | "
                f"{result.get('risk_type', '')}"
            )
            lines.append(
                f"- **触发原因**：{result.get('trigger_reason', '')}"
            )
            lines.append("- **处理建议**：证据不足，建议人工复核")
            lines.append("")

    with open(OUT_MD, "w", encoding="utf-8") as file:
        file.write("\n".join(lines))

def main():
    os.makedirs("output", exist_ok=True)
    df = pd.read_csv(RISK_ITEMS_PATH)
    collection = get_collection()

    total = len(df)
    results = []
    log_rows = []
    success = insufficient = failed = 0
    print(f"批量处理开始，共 {total} 条风险记录")

    for i, (_, risk) in enumerate(df.iterrows(), 1):
        trans_id = str(risk["trans_id"])
        risk_type = str(risk["risk_type"])
        try:
            chunks, precheck_reason = retrieve_one(collection, risk)
            for rank, ck in enumerate(chunks, 1):
                log_rows.append({
    "trans_id": trans_id,
    "risk_type": risk_type,
    "rank": rank,
    "similarity_score": ck["score"],
    "bm25_score": ck.get("bm25_score", 0.0),
    "rrf_score": ck.get("rrf_score", 0.0),
    "rank_details": json.dumps(
        ck.get("rank_details", {}),
        ensure_ascii=False,
    ),
    "source": ck["source"],
    "chunk_text": ck["text"],
})
            result, top_score = generate_one(
    risk,
    chunks,
    precheck_reason=precheck_reason,
)
            results.append(result)
            if result.get("failed"):
                failed += 1
                mark = "失败"
            elif not result.get("retrieved_policies"):
                insufficient += 1
                mark = "证据不足"
            else:
                success += 1
                mark = "✓"
            print(f"[{i}/{total}] {trans_id} {risk_type} {mark}")
        except Exception as e:
            failed += 1
            results.append({"trans_id": trans_id, "risk_type": risk_type,
                            "risk_level": str(risk.get("risk_level", "")),
                            "trigger_reason": str(risk.get("trigger_reason", "")),
                            "retrieved_policy": None,
                            "suggested_action": "处理失败，建议人工复核",
                            "confidence": 0, "needs_human_review": True,
                            "evidence": {"key_facts": f"处理异常：{e}"}})
            print(f"[{i}/{total}] {trans_id} {risk_type} ✗ {e}")

        time.sleep(BATCH_RATE_SECONDS)

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    with open(OUT_LOG, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
    "trans_id",
    "risk_type",
    "rank",
    "similarity_score",
    "bm25_score",
    "rrf_score",
    "rank_details",
    "source",
    "chunk_text",
])
        w.writeheader()
        w.writerows(log_rows)

    write_markdown(results, total, success, insufficient, failed)

    print("\n处理完成：")
    print(f"  成功：{success} 条")
    print(f"  证据不足：{insufficient} 条（已标记需人工复核）")
    print(f"  失败：{failed} 条")
    # 成本报告
    cost_summary = tracker.save_report(OUT_COST)
    print("\n成本报告：")
    print(f"  - API 调用总数：{cost_summary['total_calls']} 次")
    print(f"  - Chat 调用：{cost_summary['chat_calls']} 次 | Embedding 调用：{cost_summary['embedding_calls']} 次")
    print(f"  - 总 Token 消耗：{cost_summary['total_tokens']:,}（输入 {cost_summary['total_input_tokens']:,} + 输出 {cost_summary['total_output_tokens']:,}）")
    print(f"  - 总费用：{cost_summary['total_cost_yuan']} 元")
    print(f"  - 每条记录平均费用：{cost_summary['avg_cost_per_record']} 元")

    print("\n报告已生成：")
    print(f"  - {OUT_JSON}（结构化数据）")
    print(f"  - {OUT_MD}（可读报告）")
    print(f"  - {OUT_LOG}（检索日志）")
    print(f"  - {OUT_COST}（成本报告）")

if __name__ == "__main__":
    main()

