# scripts/10_web_app.py
# Streamlit Web 界面：单条分析 + 批量报告 + 系统看板
import os, json, time, chromadb, pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
import streamlit as st
import sys
sys.path.insert(0, "config")
sys.path.insert(0, "services")
from prompts import (
    SYSTEM_PROMPT_V2,
    USER_PROMPT_TEMPLATE_V2,
    RELEVANCE_THRESHOLD,
    TOP_K,
)
from cost_tracker import CostTracker
from relevance_gate import get_abstention_reason
from bm25_retriever import BM25Retriever
from hybrid_retriever import HybridRetriever
from generation_parser import build_verified_result

load_dotenv()
DB_PATH = "data/vector_store/chroma_db"
COLLECTION_NAME = "audit_knowledge"
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "embedding-3")
LLM_MODEL = os.getenv("LLM_MODEL", "glm-4-flash")
RISK_ITEMS_PATH = "data/input/risk_items.csv"

QUERY_MAP = {
    "金额异常": "大额交易审批权限 超过阈值 审批流程 财务总监审批",
    "重复付款": "重复付款核查 同供应商同金额 付款冲销 重复发起",
    "异常时间": "非工作时间操作 周末交易 授权管理 值班授权",
    "高频交易": "拆分采购 高频交易 供应商管理 合同审批 拆分规避审批",
}

client = OpenAI()

@st.cache_resource
def get_collection():
    chroma_client = chromadb.PersistentClient(path=DB_PATH)
    return chroma_client.get_collection(name=COLLECTION_NAME)

def get_embedding(text):
    resp = client.embeddings.create(input=[text], model=EMBEDDING_MODEL)
    return resp.data[0].embedding


@st.cache_resource
def get_bm25_retriever():
    """网页运行期间只创建一次 BM25 索引。"""
    return BM25Retriever.from_json()


def retrieve(query_text):
    """向量检索、BM25 检索、RRF 融合，返回带可观测字段的证据。"""
    collection = get_collection()

    hybrid_retriever = HybridRetriever(
        collection=collection,
        embed_query=get_embedding,
        bm25_retriever=get_bm25_retriever(),
        candidate_k=max(TOP_K * 2, 6),
    )

    fused_results = hybrid_retriever.retrieve(
        query=query_text,
        top_k=TOP_K,
    )

    return [
        {
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
        }
        for item in fused_results
    ]


def generate(risk_info, chunks):
    """
    网页端生成链路：
    Prompt V2 → JSON Schema 校验 → chunk_id 校验 → 真实原文回填。
    """
    top_score = max(
        (chunk.get("vector_score", 0.0) for chunk in chunks),
        default=0.0,
    )

    def human_review_result(reason, failed=False):
        return {
            "evidence": {
                "amount": risk_info.get("amount", 0),
                "supplier": risk_info.get("supplier", ""),
                "trans_dates": [risk_info.get("trans_date", "")],
                "key_facts": reason,
            },
            "selected_evidence_ids": [],
            "retrieved_policies": [],
            "suggested_actions": ["人工复核该风险记录。"],
            "needs_human_review": True,
            "failed": failed,
        }

    if not chunks or top_score < RELEVANCE_THRESHOLD:
        return human_review_result(
            "检索到的制度片段与风险类型相关度不足"
        ), top_score

    fields = {}
    for index, chunk in enumerate(chunks[:3], 1):
        fields[f"chunk_id{index}"] = chunk["chunk_id"]
        fields[f"score{index}"] = chunk["score"]
        fields[f"source{index}"] = chunk["source_display"]
        fields[f"content{index}"] = chunk["text"]

    for index in range(len(chunks[:3]) + 1, 4):
        fields[f"chunk_id{index}"] = "无可用证据"
        fields[f"score{index}"] = "N/A"
        fields[f"source{index}"] = "无"
        fields[f"content{index}"] = "无"

    user_prompt = USER_PROMPT_TEMPLATE_V2.format(
        trans_id=risk_info.get("trans_id", "N/A"),
        risk_type=risk_info.get("risk_type", ""),
        risk_level=risk_info.get("risk_level", ""),
        trigger_reason=risk_info.get("trigger_reason", ""),
        amount=risk_info.get("amount", 0),
        supplier=risk_info.get("supplier", ""),
        trans_date=risk_info.get("trans_date", ""),
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

        verified = build_verified_result(
            response.choices[0].message.content,
            chunks,
        )

        verified["evidence"] = {
            "amount": risk_info.get("amount", 0),
            "supplier": risk_info.get("supplier", ""),
            "trans_dates": [risk_info.get("trans_date", "")],
            **verified["evidence"],
        }

        return verified, top_score

    except Exception as error:
        return human_review_result(
            f"LLM 生成或引用校验失败：{error}",
            failed=True,
        ), top_score

# ============================================================
# 页面配置
# ============================================================
st.set_page_config(page_title="RAG 审计助手", page_icon="🔍", layout="wide")

st.sidebar.title("RAG 审计助手")
page = st.sidebar.radio("功能导航", ["单条风险分析", "批量报告浏览", "系统看板"])

# ============================================================
# 页面1：单条风险分析
# ============================================================
if page == "单条风险分析":
    st.title("单条风险交易分析")
    st.markdown("输入风险交易信息，实时获取制度检索与 LLM 分析建议。")

    col1, col2 = st.columns(2)
    with col1:
        trans_id = st.text_input("交易编号", value="TXN000001")
        risk_type = st.selectbox("风险类型", list(QUERY_MAP.keys()))
        risk_level = st.selectbox("风险等级", ["高", "中", "低"])
    with col2:
        amount = st.number_input("交易金额（元）", value=0.0, step=1000.0)
        supplier = st.text_input("供应商", value="")
        trans_date = st.text_input("交易日期", value="")

    trigger_reason = st.text_area("触发原因", value=f"金额 {amount} 超过阈值")

    if st.button("开始分析", type="primary"):
        if not supplier:
            st.warning("请填写供应商名称")
        else:
            # 检查原始触发原因，而不是扩写后的 query。
            # 这样无关输入不会因为 QUERY_MAP 自动补充了审计关键词而误通过。
            precheck_reason = get_abstention_reason(
                query=trigger_reason,
                top_similarity=1.0,
                threshold=RELEVANCE_THRESHOLD,
            )

            if precheck_reason:
                st.warning(precheck_reason)
                st.caption(
                    "已在预检阶段停止：未调用 Embedding、ChromaDB 或 LLM。"
                )
                st.stop()

            with st.spinner("正在检索相关制度..."):
                query = f"{QUERY_MAP.get(risk_type, risk_type)} {trigger_reason}"
                chunks = retrieve(query)

            st.subheader("检索结果（混合检索 + RRF 融合）")
            for i, ck in enumerate(chunks, 1):
                score = ck["vector_score"]
                score_color = (
                    "🟢" if score >= 0.7
                    else "🟡" if score >= RELEVANCE_THRESHOLD
                    else "🔴"
                )

                st.markdown(
                    f"**证据 {i}** `{ck['chunk_id']}` "
                    f"{score_color} 向量相似度：`{score:.4f}`"
                )
                st.caption(f"来源：{ck['source_display']}")

                metric_1, metric_2, metric_3 = st.columns(3)
                metric_1.metric("向量相似度", f"{ck['vector_score']:.4f}")
                metric_2.metric("BM25 关键词分数", f"{ck['bm25_score']:.4f}")
                metric_3.metric("RRF 融合分数", f"{ck['rrf_score']:.6f}")

                st.text(
                    ck["text"][:200] + "..."
                    if len(ck["text"]) > 200
                    else ck["text"]
                )
                st.divider()

            with st.spinner("正在生成并校验分析结果..."):
                risk_info = {
                    "trans_id": trans_id,
                    "risk_type": risk_type,
                    "risk_level": risk_level,
                    "trigger_reason": trigger_reason,
                    "amount": amount,
                    "supplier": supplier,
                    "trans_date": trans_date,
                }
                result, top_score = generate(risk_info, chunks)

            st.subheader("分析结果（已校验）")

            if result.get("failed"):
                st.error("生成或引用校验失败，已停止输出结论，建议人工复核。")
            elif not result.get("retrieved_policies"):
                st.warning("证据不足，建议人工复核。")
            else:
                st.success("分析结果已通过结构校验，制度引用已由系统回填。")

            metric_1, metric_2, metric_3 = st.columns(3)
            metric_1.metric("Top 向量相似度", f"{top_score:.4f}")
            metric_2.metric(
                "已验证引用数",
                len(result.get("retrieved_policies", [])),
            )
            metric_3.metric(
                "需人工复核",
                "是" if result.get("needs_human_review", True) else "否",
            )

            evidence = result.get("evidence", {}) or {}
            st.markdown(
                f"**核心事实**：{evidence.get('key_facts', '')}"
            )

            policies = result.get("retrieved_policies", []) or []
            if policies:
                st.markdown("**已验证制度引用**：")
                for policy in policies:
                    with st.expander(
                        f"{policy['chunk_id']} · "
                        f"{policy.get('source', '')} · "
                        f"{policy.get('section', '')}"
                    ):
                        st.write(policy.get("content", ""))
                        st.caption(
                            "检索信号："
                            f"向量={policy.get('vector_score', 0):.4f}，"
                            f"BM25={policy.get('bm25_score', 0):.4f}，"
                            f"RRF={policy.get('rrf_score', 0):.6f}"
                        )

            actions = result.get("suggested_actions", []) or []
            st.markdown("**建议行动**：")
            if actions:
                for action in actions:
                    st.write(f"- {action}")
            else:
                st.write("- 建议人工复核。")
# ============================================================
# 页面2：批量报告浏览
# ============================================================
elif page == "批量报告浏览":
    st.title("批量报告浏览")

    report_path = "output/structured_risk_report.json"
    if not os.path.exists(report_path):
        st.warning("暂无报告数据，请先运行 05_batch_process.py 生成报告。")
    else:
        with open(report_path, "r", encoding="utf-8") as f:
            results = json.load(f)

        st.metric("报告总数", len(results))

        # 按风险类型筛选
        risk_types = ["全部"] + list(set(r.get("risk_type", "") for r in results))
        selected_type = st.selectbox("按风险类型筛选", risk_types)

        filtered = results if selected_type == "全部" else [r for r in results if r.get("risk_type") == selected_type]

        # 表格展示：展示系统校验后的引用数量与处理状态。
        df_data = []
        for r in filtered:
            evidence = r.get("evidence", {}) or {}
            policies = r.get("retrieved_policies", []) or []

            if r.get("failed"):
                status = "失败（需人工复核）"
            elif not policies:
                status = "证据不足（需人工复核）"
            else:
                status = "已验证引用"

            df_data.append({
                "交易编号": r.get("trans_id", ""),
                "风险类型": r.get("risk_type", ""),
                "风险等级": r.get("risk_level", ""),
                "金额": evidence.get("amount", ""),
                "供应商": evidence.get("supplier", ""),
                "处理状态": status,
                "已验证引用数": len(policies),
                "首条引用": (
                    f"{policies[0].get('source', '')} · "
                    f"{policies[0].get('section', '')}"
                    if policies else "无"
                ),
            })

        st.dataframe(pd.DataFrame(df_data), use_container_width=True)

        # 详情查看
        st.subheader("详情查看")
        selected_idx = st.selectbox("选择记录", range(len(filtered)),
                                    format_func=lambda i: f"{filtered[i].get('trans_id','')} - {filtered[i].get('risk_type','')}")
        if selected_idx is not None:
            r = filtered[selected_idx]
            st.json(r)

# ============================================================
# 页面3：系统看板
# ============================================================
elif page == "系统看板":
    st.title("系统看板")

    col1, col2, col3 = st.columns(3)

    # 向量库状态
    try:
        collection = get_collection()
        col1.metric("向量库切片数", collection.count())
    except Exception:
        col1.metric("向量库切片数", "未连接")

    # 报告数量
    report_path = "output/structured_risk_report.json"
    if os.path.exists(report_path):
        with open(report_path, "r", encoding="utf-8") as f:
            report_count = len(json.load(f))
        col2.metric("已处理风险记录", report_count)
    else:
        col2.metric("已处理风险记录", 0)

    # 成本报告
    cost_path = "output/cost_report.json"
    if os.path.exists(cost_path):
        with open(cost_path, "r", encoding="utf-8") as f:
            cost_data = json.load(f)
            summary = cost_data.get("summary", {})
        col3.metric("总 Token 消耗", f"{summary.get('total_tokens', 0):,}")
        st.metric("总费用（元）", f"{summary.get('total_cost_yuan', 0)}")

        st.subheader("成本明细")
        cost_df = pd.DataFrame(cost_data.get("details", []))
        if not cost_df.empty:
            st.dataframe(cost_df, use_container_width=True)
    else:
        col3.metric("总 Token 消耗", "无数据")

    # 评估报告
    eval_path = "output/evaluation_report.json"
    if os.path.exists(eval_path):
        with open(eval_path, "r", encoding="utf-8") as f:
            eval_data = json.load(f)
            eval_summary = eval_data.get("summary", {})
        st.subheader("检索质量评估")
        ecol1, ecol2, ecol3, ecol4 = st.columns(4)
        ecol1.metric("Hit@1 命中率", f"{eval_summary.get('hit_at_1_rate', 0)*100:.0f}%")
        ecol2.metric("Hit@3 命中率", f"{eval_summary.get('hit_at_3_rate', 0)*100:.0f}%")
        ecol3.metric("MRR", f"{eval_summary.get('mrr', 0):.4f}")
        ecol4.metric("章节准确率", f"{eval_summary.get('section_accuracy', 0)*100:.0f}%")

    # 测试报告
    test_path = "output/test_report.json"
    if os.path.exists(test_path):
        with open(test_path, "r", encoding="utf-8") as f:
            test_data = json.load(f)
        st.subheader("自动化测试")
        tcol1, tcol2, tcol3 = st.columns(3)
        tcol1.metric("通过", test_data.get("passed", 0))
        tcol2.metric("失败", test_data.get("failed", 0))
        tcol3.metric("总计", test_data.get("total", 0))
        if test_data.get("errors"):
            st.error("存在失败项：")
            for err in test_data["errors"]:
                st.write(f"- {err}")

    # 知识库状态
    st.subheader("知识库状态")
    kb_dir = "data/knowledge_base"
    if os.path.exists(kb_dir):
        kb_files = [f for f in os.listdir(kb_dir) if f.endswith(".txt")]
        for f in kb_files:
            size = os.path.getsize(os.path.join(kb_dir, f))
            st.write(f"- {f}（{size:,} 字节）")

    # 指纹状态
    fp_path = "data/vector_store/doc_fingerprints.json"
    if os.path.exists(fp_path):
        with open(fp_path, "r", encoding="utf-8") as f:
            fps = json.load(f)
        st.caption(f"文档指纹记录：{len(fps)} 个文档")
    else:
        st.caption("文档指纹记录：无（未运行增量更新）")


