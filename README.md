# 基于 RAG 的企业交易风险洞察与审计辅助工具

> 个人项目 · AI / 数据处理 / 金融科技方向
>
> Python · ChromaDB · Embedding · BM25 · RRF · Pydantic · Streamlit

面向已由上游规则识别的交易风险，本项目构建“**风险事实 → 制度检索 → 证据校验 → 人工复核建议**”的 RAG 审计辅助链路。系统不输出审计结论；所有结果均标记为需人工复核。

## 项目亮点

- **混合检索**：结合向量检索、BM25 关键词检索与 RRF（Reciprocal Rank Fusion）排序，兼顾语义匹配与金额、条款等精确命中。
- **可控生成**：LLM 只能从本次 Top-K 证据中选择 `chunk_id`；Pydantic 校验 JSON 结构，程序回填真实制度原文、来源和章节，避免伪造引用。
- **安全边界**：领域闸门在 Embedding、ChromaDB 和 LLM 调用前拒绝明显无关输入；`needs_human_review` 被 Schema 强制为 `true`。
- **工程化能力**：支持文档 MD5 指纹增量更新、Chroma 索引一致性检查、成本追踪、检索日志、Markdown 报告及 Streamlit 可视化。
- **测试保障**：离线自动化测试 **41 / 41 通过**，覆盖索引一致性、领域拒答、混合检索、引用校验、生成安全和网页端契约。

## 系统架构

```text
风险记录 / 触发原因
        │
        ▼
领域闸门 ──无关──> 证据不足，停止调用 API，人工复核
        │
        ▼
向量检索 ─┐
          ├─> RRF 融合排序 ─> Top-K 候选证据（chunk_id）
BM25 检索 ─┘                         │
                                      ▼
                         Prompt V2：LLM 仅选择证据 ID
                                      │
                                      ▼
                  Pydantic Schema + 引用 ID 校验 + 原文回填
                                      │
                                      ▼
               结构化报告 / Markdown / 检索日志 / Streamlit
```

## 核心流程

1. **知识库构建**：按章节将制度、合规手册和案例切片，写入 `chunks.json`，保留 `chunk_id`、`source`、`section`、`text`。
2. **向量存储**：使用 Embedding 模型将切片写入持久化 ChromaDB。
3. **混合检索**：向量检索负责语义相似，BM25 负责关键词精确匹配，RRF 合并两类候选排名。
4. **受控生成**：LLM 仅返回风险摘要、建议动作和 `selected_evidence_ids`。
5. **结果验证**：Pydantic 拒绝额外字段、非法类型和 `needs_human_review=false`；系统拒绝未知或重复证据 ID，并从本地检索结果回填原文。

## 目录结构

```text
rag_qa_assistant/
├── config/
│   ├── prompts.py                 # Prompt V1/V2、阈值与限流配置
│   └── cost_tracker.py            # API 调用与成本记录
├── data/
│   ├── input/                     # 风险记录与清洗后的交易数据
│   ├── knowledge_base/            # 制度、合规手册、案例文本
│   └── vector_store/              # chunks.json、文档指纹、ChromaDB
├── docs/
│   ├── project-deep-dive.html     # 中文项目深挖网页
│   └── UPGRADE_TUTORIAL.md        # 升级过程与学习教程
├── services/
│   ├── bm25_retriever.py          # 中文 BM25 检索
│   ├── rank_fusion.py             # RRF 排名融合
│   ├── hybrid_retriever.py        # 向量 + BM25 混合检索
│   ├── relevance_gate.py          # 领域闸门 / 拒答规则
│   ├── schemas.py                 # Pydantic 输出约束
│   ├── citations.py               # 证据 ID 校验与原文回填
│   └── generation_parser.py       # LLM JSON 解析与验证
├── scripts/
│   ├── 01_build_kb.py             # 文档切片
│   ├── 02_embed_store.py          # 向量化入库
│   ├── 05_batch_process.py        # 批处理主入口
│   ├── 08_incremental_update.py   # 增量更新
│   ├── 09_test_suite.py           # 自动化测试入口
│   └── 10_web_app.py              # Streamlit 网页端
├── tests/                         # 单元、回归、契约测试
├── output/                        # 报告、日志、成本与测试产物
├── requirements.txt
└── README.md
```

## 技术选型

| 模块 | 技术 | 用途 |
|---|---|---|
| 数据处理 | Pandas | 读取并批量处理风险记录 |
| 向量化 | OpenAI 兼容 Embedding API | 把制度文本编码为向量 |
| 向量数据库 | ChromaDB | 持久化存储与近邻检索 |
| 关键词检索 | jieba + rank-bm25 | 中文分词与精确关键词匹配 |
| 排名融合 | RRF | 融合向量与 BM25 排名 |
| 生成模型 | OpenAI 兼容 Chat API | 生成风险摘要与人工复核建议 |
| 输出校验 | Pydantic | 校验 LLM JSON 与人工复核约束 |
| 可视化 | Streamlit | 单条分析、报告浏览和系统看板 |

## 快速开始

### 1. 创建虚拟环境并安装依赖

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

### 2. 配置环境变量

在项目根目录创建 `.env`（不要提交到 Git）：

```dotenv
OPENAI_API_KEY=your_api_key
OPENAI_BASE_URL=https://open.bigmodel.cn/api/paas/v4
LLM_MODEL=glm-4-flash
EMBEDDING_MODEL=embedding-3
```

项目使用 OpenAI 兼容接口；也可替换为其他兼容服务商的地址、模型名和密钥。

### 3. 构建或更新索引

```powershell
python scripts/01_build_kb.py
python scripts/02_embed_store.py

# 知识库文件变化后，优先运行增量更新
python scripts/08_incremental_update.py
```

### 4. 运行离线测试

```powershell
python scripts/09_test_suite.py --offline
```

当前基线：**41 / 41 通过**。离线模式不会调用 Embedding 或 LLM API。

### 5. 启动网页端

```powershell
python -m streamlit run scripts/10_web_app.py
```

访问终端显示的本地地址（通常为 `http://localhost:8501`）。可先输入与审计无关的文本，验证领域闸门会在 API 调用前停止分析。

### 6. 运行批处理（需要有效 API Key）

```powershell
python scripts/05_batch_process.py
```

输出包括：

- `output/structured_risk_report.json`：结构化数据；
- `output/risk_analysis_report.md`：可读报告；
- `output/retrieval_log.csv`：向量、BM25、RRF 检索日志；
- `output/cost_report.json`：调用与成本统计。

## 关键安全设计

### 领域闸门：无关输入不调用 API

仅向量相似度不一定能识别无关问题，因为小型知识库也会返回“最接近的一条”。`relevance_gate.py` 对原始触发原因做领域锚点检查；不通过时直接给出“证据不足，建议人工复核”。

### 证据 ID 模式：模型不能伪造制度原文

模型输出示例：

```json
{
  "evidence": {"key_facts": "该交易金额较大，可能需要核实审批流程。"},
  "selected_evidence_ids": ["audit_policy.txt_015"],
  "suggested_actions": ["核实财务总监审批记录。"],
  "needs_human_review": true
}
```

`citations.py` 会验证 ID 是否属于本次 Top-K，并从检索结果回填真实的文件、章节、原文和检索信号。未知 ID、重复 ID、额外字段或 `needs_human_review=false` 都会被拒绝并降级为人工复核。

### 人在回路

系统输出仅用于辅助风险复核，不能替代审计结论或业务处置。报告和网页端均保留人工复核标记。

## 测试覆盖

`scripts/09_test_suite.py --offline` 当前覆盖：

- 文档增量更新与 Chroma/清单一致性；
- 无关查询领域闸门；
- BM25、RRF 与混合检索；
- Pydantic Schema 和 `needs_human_review` 约束；
- 证据 ID 合法性、重复 ID 和原文回填；
- Prompt V2、批处理 V2、Markdown 报告 V2；
- Streamlit 网页端安全契约。

## 项目边界与后续计划

- 本项目是上游风险规则识别后的**解释与复核辅助层**，不负责风险识别模型训练，也不替代人工审计。
- 41/41 表示离线工程与安全行为通过，不等于线上模型准确率或业务准确率 100%。
- 在线验收需要有效 API Key；后续将以独立评测集、人工复核反馈、引用有效率和拒答准确率评估真实效果。
- 计划补充 JSON 自动修复重试、trace_id、知识库版本、reranker 和人工反馈闭环。

