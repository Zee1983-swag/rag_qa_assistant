# scripts/09_test_suite.py
# 自动化测试套件：单元测试 + 集成测试 + 安全边界测试
import os, json, sys, time, argparse, subprocess, chromadb, pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
sys.path.insert(0, "config")
from prompts import (
    SYSTEM_PROMPT,
    SYSTEM_PROMPT_V2,
    USER_PROMPT_TEMPLATE,
    RELEVANCE_THRESHOLD,
    TOP_K,
    BATCH_RATE_SECONDS,
)
from relevance_gate import get_abstention_reason

load_dotenv()
DB_PATH = "data/vector_store/chroma_db"
COLLECTION_NAME = "audit_knowledge"
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "embedding-3")
LLM_MODEL = os.getenv("LLM_MODEL", "glm-4-flash")
RISK_ITEMS_PATH = "data/input/risk_items.csv"

client = OpenAI()
QUERY_MAP = {
    "金额异常": "大额交易审批权限 超过阈值 审批流程 财务总监审批",
    "重复付款": "重复付款核查 同供应商同金额 付款冲销 重复发起",
    "异常时间": "非工作时间操作 周末交易 授权管理 值班授权",
    "高频交易": "拆分采购 高频交易 供应商管理 合同审批 拆分规避审批",
}

# ============================================================
# 测试结果统计
# ============================================================
passed = 0
failed = 0
errors = []

def assert_true(condition, test_name, detail=""):
    global passed, failed
    if condition:
        passed += 1
        print(f"  [PASS] {test_name}")
    else:
        failed += 1
        errors.append(f"{test_name}: {detail}")
        print(f"  [FAIL] {test_name} - {detail}")

# ============================================================
# 工具函数（从 05_batch_process.py 复制，保持独立可测试）
# ============================================================
def extract_json(text):
    if "```json" in text:
        text = text.split("```json", 1)[1].split("```", 1)[0]
    elif "```" in text:
        text = text.split("```", 1)[1].split("```", 1)[0]
    return json.loads(text.strip())

def calc_similarity(dist):
    """余弦距离转相似度"""
    return max(0.0, min(1.0, 1 - dist / 2))

def safe_amount(item):
    ev = item.get("evidence")
    if not isinstance(ev, dict):
        return 0.0
    amt = ev.get("amount", 0)
    try:
        return float(amt)
    except (ValueError, TypeError):
        return 0.0

def get_embedding(text):
    resp = client.embeddings.create(input=[text], model=EMBEDDING_MODEL)
    return resp.data[0].embedding

# ============================================================
# 第一层：单元测试（纯函数，不依赖 API）
# ============================================================
def test_unit():
    print("\n" + "=" * 60)
    print("第一层：单元测试")
    print("=" * 60)

    # 测试 extract_json - 标准 JSON
    assert_true(
        isinstance(extract_json('{"key": "value"}'), dict),
        "extract_json_标准JSON",
        "无法解析标准JSON"
    )

    # 测试 extract_json - 带 markdown 代码块
    result = extract_json('```json\n{"key": "value"}\n```')
    assert_true(
        isinstance(result, dict) and result.get("key") == "value",
        "extract_json_Markdown代码块",
        "无法解析markdown代码块中的JSON"
    )

    # 测试 extract_json - 带文本前缀
    result = extract_json('这是说明文字\n```\n{"key": "value"}\n```\n')
    assert_true(
        isinstance(result, dict),
        "extract_json_带文本前缀",
        "无法解析带前缀的JSON"
    )

    # 测试 calc_similarity - 边界值
    assert_true(abs(calc_similarity(0) - 1.0) < 0.001, "calc_similarity_距离0_相似度1", "距离0应得到相似度1")
    assert_true(abs(calc_similarity(2) - 0.0) < 0.001, "calc_similarity_距离2_相似度0", "距离2应得到相似度0")
    assert_true(abs(calc_similarity(1) - 0.5) < 0.001, "calc_similarity_距离1_相似度0.5", "距离1应得到相似度0.5")
    assert_true(calc_similarity(-1) == 1.0, "calc_similarity_负距离_截断为1", "负距离应截断为1")
    assert_true(calc_similarity(3) == 0.0, "calc_similarity_超范围_截断为0", "超范围应截断为0")

    # 测试 safe_amount - 正常数值
    assert_true(safe_amount({"evidence": {"amount": 123.45}}) == 123.45, "safe_amount_正常数值", "数值提取错误")

    # 测试 safe_amount - 字符串金额
    assert_true(safe_amount({"evidence": {"amount": "123.45"}}) == 123.45, "safe_amount_字符串金额", "字符串转数值失败")

    # 测试 safe_amount - None evidence
    assert_true(safe_amount({"evidence": None}) == 0.0, "safe_amount_None_evidence", "None应返回0")

    # 测试 safe_amount - 缺失 amount 字段
    assert_true(safe_amount({"evidence": {}}) == 0.0, "safe_amount_缺失字段", "缺失字段应返回0")

    # 测试 safe_amount - 无效字符串
    assert_true(safe_amount({"evidence": {"amount": "abc"}}) == 0.0, "safe_amount_无效字符串", "无效字符串应返回0")

    # 测试 prompt 模板 - 关键变量存在
    assert_true("{trans_id}" in USER_PROMPT_TEMPLATE, "prompt模板_包含trans_id", "模板缺少trans_id占位符")
    assert_true("{risk_type}" in USER_PROMPT_TEMPLATE, "prompt模板_包含risk_type", "模板缺少risk_type占位符")
    assert_true("{score1}" in USER_PROMPT_TEMPLATE, "prompt模板_包含score1", "模板缺少score1占位符")
    assert_true("{content1}" in USER_PROMPT_TEMPLATE, "prompt模板_包含content1", "模板缺少content1占位符")

    # 测试安全边界配置
    assert_true(0 < RELEVANCE_THRESHOLD < 1, "阈值_合理范围", "阈值应在0-1之间")
    assert_true(TOP_K >= 1, "TOP_K_合理值", "TOP_K应大于等于1")
    assert_true(BATCH_RATE_SECONDS >= 0, "限流_非负值", "限流时间应为非负数")

# ============================================================
# 第二层：集成测试（依赖向量库 + API，取2条记录验证）
# ============================================================
def test_integration():
    print("\n" + "=" * 60)
    print("第二层：集成测试（取2条记录验证完整链路）")
    print("=" * 60)

    df = pd.read_csv(RISK_ITEMS_PATH).head(2)

    try:
        chroma_client = chromadb.PersistentClient(path=DB_PATH)
        collection = chroma_client.get_collection(name=COLLECTION_NAME)
        assert_true(collection.count() > 0, "向量库_存在且有数据", f"向量库为空，count={collection.count()}")
    except Exception as e:
        assert_true(False, "向量库_连接", str(e))
        return

    for i, (_, risk) in enumerate(df.iterrows(), 1):
        trans_id = str(risk["trans_id"])
        risk_type = str(risk["risk_type"])
        prefix = f"记录{i}({trans_id}_{risk_type})"

        # --- 检索阶段 ---
        try:
            query = f"{QUERY_MAP.get(risk_type, risk_type)} {risk.get('trigger_reason', '')}"
            q_emb = get_embedding(query)
            res = collection.query(query_embeddings=[q_emb], n_results=TOP_K)
            docs = res["documents"][0]
            metas = res["metadatas"][0]
            dists = res["distances"][0]

            assert_true(len(docs) == TOP_K, f"{prefix}_检索返回数量", f"期望{TOP_K}条，实际{len(docs)}条")
            assert_true(len(docs[0]) > 0, f"{prefix}_检索内容非空", "检索结果第一条为空")

            top_sim = calc_similarity(dists[0])
            assert_true(0 <= top_sim <= 1, f"{prefix}_相似度范围", f"相似度{top_sim}超出0-1范围")

        except Exception as e:
            assert_true(False, f"{prefix}_检索", str(e))
            continue

        # --- 生成阶段（仅在高相似度时测试 LLM 生成）---
        if top_sim < RELEVANCE_THRESHOLD:
            assert_true(True, f"{prefix}_安全边界_证据不足跳过LLM", "")
            continue

        fields = {}
        chunks_data = []
        for idx, (doc, meta, dist) in enumerate(zip(docs, metas, dists), 1):
            sim = calc_similarity(dist)
            chunks_data.append({"text": doc, "source": f"{meta.get('source','')} · {meta.get('section','')}", "score": sim})
            fields[f"score{idx}"] = sim
            fields[f"source{idx}"] = f"{meta.get('source','')} · {meta.get('section','')}"
            fields[f"content{idx}"] = doc

        user_prompt = USER_PROMPT_TEMPLATE.format(
            trans_id=trans_id, risk_type=risk_type,
            risk_level=str(risk.get("risk_level", "")),
            trigger_reason=str(risk.get("trigger_reason", "")),
            amount=risk.get("amount", 0), supplier=str(risk.get("supplier", "")),
            trans_date=str(risk.get("trans_date", "")), **fields)

        try:
            resp = client.chat.completions.create(
                model=LLM_MODEL, temperature=0,
                messages=[{"role": "system", "content": SYSTEM_PROMPT},
                          {"role": "user", "content": user_prompt}])
            parsed = extract_json(resp.choices[0].message.content)

            # 验证输出结构完整性
            assert_true("evidence" in parsed, f"{prefix}_输出含evidence", "缺少evidence字段")
            assert_true("retrieved_policy" in parsed, f"{prefix}_输出含retrieved_policy", "缺少retrieved_policy字段")
            assert_true("suggested_action" in parsed, f"{prefix}_输出含suggested_action", "缺少suggested_action字段")
            assert_true("confidence" in parsed, f"{prefix}_输出含confidence", "缺少confidence字段")

            # 验证 evidence 子结构
            ev = parsed.get("evidence", {})
            assert_true(isinstance(ev, dict), f"{prefix}_evidence是字典", "evidence不是字典类型")
            assert_true("key_facts" in ev, f"{prefix}_evidence含key_facts", "evidence缺少key_facts")

            # 验证置信度范围
            conf = parsed.get("confidence", 0)
            try:
                conf_val = float(conf)
                assert_true(0 <= conf_val <= 1, f"{prefix}_confidence范围", f"置信度{conf_val}超出0-1")
            except (ValueError, TypeError):
                assert_true(False, f"{prefix}_confidence类型", f"置信度{conf}无法转为数值")

            time.sleep(1)

        except json.JSONDecodeError as e:
            assert_true(False, f"{prefix}_JSON解析", str(e))
        except Exception as e:
            assert_true(False, f"{prefix}_LLM生成", str(e))

# ============================================================
# 第三层：安全边界测试
# ============================================================
def test_safety():
    print("\n" + "=" * 60)
    print("第三层：安全边界测试")
    print("=" * 60)

    # 测试1：无关查询必须被领域闸门拦截。
    # 不能再假设“无关 query 的向量相似度一定低”，
    # 因为向量库总会返回库中最接近的一条文本。
    unrelated_query = "量子计算机天气预报 火星探测 脑外科手术"

    reason = get_abstention_reason(
        query=unrelated_query,
        top_similarity=0.99,
        threshold=RELEVANCE_THRESHOLD,
    )

    assert_true(
        reason is not None and "证据不足" in reason,
        "安全边界_无关查询被领域闸门拒答",
        f"无关查询未被拒答，reason={reason}",
    )

    # 测试2：验证人工复核约束覆盖正常、异常与 Prompt 三个层面。
    # V2 正常路径由 Pydantic Literal[True] 保证，而不是依赖重复字符串次数。
    try:
        with open("scripts/05_batch_process.py", "r", encoding="utf-8") as f:
            batch_code = f.read()

        with open("services/schemas.py", "r", encoding="utf-8") as f:
            schema_code = f.read()

        fallback_forces_review = '"needs_human_review": True' in batch_code
        schema_forces_review = "Literal[True]" in schema_code
        prompt_forces_review = "needs_human_review 必须为 true" in SYSTEM_PROMPT_V2

        assert_true(
            fallback_forces_review and schema_forces_review and prompt_forces_review,
            "安全边界_needs_human_review强制True",
            "人工复核约束未同时覆盖异常兜底、Pydantic Schema 与 Prompt V2",
        )
    except Exception as e:
        assert_true(False, "安全边界_代码检查", str(e))

    # 测试3：验证 System Prompt 包含安全措辞要求
    assert_true("建议" in SYSTEM_PROMPT, "安全边界_Prompt含建议措辞", "Prompt缺少'建议'措辞要求")
    assert_true("不得" in SYSTEM_PROMPT, "安全边界_Prompt含禁止措辞", "Prompt缺少禁止性表述")
    assert_true("证据不足" in SYSTEM_PROMPT, "安全边界_Prompt含证据不足处理", "Prompt缺少证据不足处理规则")

    # 测试4：验证 Few-shot 示例存在
    assert_true("示例1" in SYSTEM_PROMPT, "安全边界_FewShot示例1存在", "Prompt缺少Few-shot示例1")
    assert_true("示例2" in SYSTEM_PROMPT, "安全边界_FewShot示例2存在", "Prompt缺少Few-shot示例2")

    # 测试5：验证输出 JSON 格式要求
    assert_true("JSON" in SYSTEM_PROMPT, "安全边界_Prompt要求JSON格式", "Prompt未要求JSON格式输出")
# ============================================================
# 第四层：离线回归测试
# 不调用 Embedding / LLM API，验证历史 bug 不会复发。
# ============================================================
def test_regression():
    print("\n" + "=" * 60)
    print("第四层：离线回归测试")
    print("=" * 60)

    offline_tests = [
        (
            "tests/test_incremental_update.py",
            "回归测试_修改制度先删旧向量",
        ),
        (
            "tests/test_relevance_gate.py",
            "回归测试_无关查询领域闸门",
        ),
        (
            "tests/test_batch_precheck.py",
            "回归测试_批处理预检与正常放行",
        ),
        (
            "tests/test_bm25_retriever.py",
            "回归测试_BM25关键词检索",
        ),
        (
            "tests/test_rank_fusion.py",
            "回归测试_RRF排名与分数合并",
        ),
        (
            "tests/test_hybrid_retriever.py",
            "回归测试_混合检索器",
        ),
        (
            "tests/test_schemas.py",
            "回归测试_Pydantic输出结构",
        ),
        (
            "tests/test_citations.py",
            "回归测试_证据ID引用校验",
        ),
        (
            "tests/test_prompt_v2.py",
            "回归测试_PromptV2输出契约",
        ),
        (
            "tests/test_generation_parser.py",
            "回归测试_生成结果解析与回填",
        ),
        (
            "tests/test_batch_generation_v2.py",
            "回归测试_批处理V2生成链路",
        ),
        (
            "tests/test_report_v2.py",
            "回归测试_V2报告展示",
        ),
        (
            "tests/test_web_app_v2_contract.py",
            "回归测试_网页端V2安全契约",
        ),
    ]

    for test_file, test_name in offline_tests:
        result = subprocess.run(
            [sys.executable, "-X", "utf8", test_file],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )

        assert_true(
            result.returncode == 0,
            test_name,
            result.stdout + "\n" + result.stderr,
        )

        if result.returncode == 0:
            print(f"  [INFO] {test_file} 通过")
# ============================================================
# 主入口
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="RAG 审计助手自动化测试套件"
    )

    parser.add_argument(
        "--offline",
        action="store_true",
        help="只运行不依赖 Embedding / LLM API 的测试",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("RAG 审计助手 - 自动化测试套件")
    print(f"运行模式：{'离线' if args.offline else '完整'}")
    print("=" * 60)

    # 第一层：纯函数测试
    test_unit()

    # 第二层：真实 Chroma + Embedding + LLM 测试。
    # API 不可用时，仍可使用 --offline 验证本地逻辑。
    if args.offline:
        print("\n[SKIP] 已跳过在线集成测试（--offline）")
    else:
        test_integration()

    # 第三层：护栏规则与 Prompt 检查
    test_safety()

    # 第四层：所有离线回归测试
    test_regression()

    print("\n" + "=" * 60)
    print(f"测试结果汇总：通过 {passed} | 失败 {failed} | 总计 {passed + failed}")
    print("=" * 60)

    if failed > 0:
        print("\n失败详情：")
        for err in errors:
            print(f"  - {err}")

    result = {
        "mode": "offline" if args.offline else "full",
        "total": passed + failed,
        "passed": passed,
        "failed": failed,
        "errors": errors,
    }

    with open("output/test_report.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()


