# RAG 审计助手：从“可运行原型”到“可解释、可评估系统”的升级教程

> 本文是当前项目的改进方案与实施路线图。目标不是一次性引入所有流行框架，而是在保留现有 Python + OpenAI-compatible API + ChromaDB + Streamlit 技术栈的前提下，优先补齐准确性、拒答安全、可维护性和可验证性。

---

## 1. 先理解当前项目：它已经做对了什么

当前项目的主流程是：

```text
上游风控规则筛出的风险记录
        ↓
根据风险类型扩写查询语句
        ↓
Embedding-3 把查询转为向量
        ↓
ChromaDB 召回 Top-3 制度/案例片段
        ↓
相关度阈值判断（低于 0.5 则不调用 LLM）
        ↓
GLM-4-Flash 生成带制度引用的 JSON 建议
        ↓
JSON / Markdown / 检索日志 / 成本报告 / Streamlit 页面
```

它已经具备一套合格 RAG 原型的基础：

- 有独立知识库，并按章节切片；
- 有向量化与 Chroma 持久化；
- 有风险类型查询扩写；
- 有低相关“证据不足”分支；
- 有 Prompt 中的人工复核、引用原文等审计护栏；
- 有检索日志、成本追踪、回归测试和 Streamlit 展示；
- 有 MD5 文件指纹的增量更新尝试。

但这仍是一个 **demo / 原型**，不是能直接给审计部门使用的生产系统。升级的核心不是“换一个更大的模型”，而是让每一次输出都能回答四个问题：

1. 为什么检索到这几段制度？
2. 这几段制度是否真的适用于当前风险？
3. 大模型输出的每句话能否追溯到输入事实或制度原文？
4. 我们如何量化地知道系统本次比上次更好？

---

## 2. 当前问题清单与优先级

| 优先级 | 问题 | 当前证据 | 风险 | 解决方向 |
|---|---|---|---|---|
| P0 | 无关查询仍可能“高相似” | 测试中的无关 query 相似度为 0.66，高于 0.5 阈值 | 系统会把无关制度误当证据并调用 LLM | 混合检索、拒答校准、重排序、负样本测试 |
| P0 | 修改知识库文档时可能保留旧向量 | `08_incremental_update.py` 对删除文件 delete，但修改文件直接 add 新 chunk | 过期制度仍可能被检索、同一制度重复 | 修改文件也先删除旧 chunk；以索引版本校验 |
| P1 | 关键逻辑复制在多个脚本中 | 03/05/06/09/10 重复 query、embedding、JSON 解析逻辑 | 修一次 bug，其他入口依旧错误 | 抽取共享服务层 |
| P1 | JSON 只解析、不校验 | `json.loads()` 成功后不检查字段类型、引用真实性 | 模型可返回结构不完整或伪造引用 | Pydantic Schema + 引用程序化回填 |
| P1 | 100% 指标容易误读 | 10 条测试用例与知识库/查询语言高度同源 | 看起来满分，不代表真实泛化 | 独立盲测集、负样本和端到端评估 |
| P2 | 纯向量检索不擅长条款号、金额、精确字段 | 向量主要表达语义接近，不保证精确匹配 | “第 20 条”“50 万”等关键审计条件可能被稀释 | BM25 + 向量混合检索 |
| P2 | 模型自报 confidence 不可校准 | `confidence` 来自 LLM 输出 | 被误当作 85% 正确概率 | 显示为“模型自评”；另建可校准的风险分数 |
| P3 | 缺少完整运行追踪 | 目前只有最终日志和成本 | 很难定位是检索、Prompt 还是模型导致错误 | 每次请求生成 trace_id，记录全链路 |

---

## 3. 参考项目如何映射到本项目

这不是“抄项目”，而是借鉴经过实践验证的设计思想。

| 参考 | 借鉴点 | 在本项目中的落地 |
|---|---|---|
| [Ragas](https://github.com/vibrantlabsai/ragas) | 将 RAG 的检索质量与生成质量拆开评估，并围绕失败样本迭代 | 为每条评测记录保存 query、contexts、answer、ground_truth、评估结果和失败原因 |
| [rag-fusion](https://github.com/Raudaschl/rag-fusion) | BM25 与向量检索并行，再用 RRF 融合排序；重视对比实验 | 为制度类文本新增关键词检索，特别保护条款号、金额阈值、审批词 |
| [IBM financial LLM output drift](https://github.com/ibm-client-engineering/output-drift-financial-llms) | 金融任务中，即使温度为 0，也要实测重复调用的一致性 | 同一风险重复运行 N 次，检查结论、引用、动作是否稳定；把一致性与正确性分开 |
| [fin-rag-lab](https://github.com/zyziyun/fin-rag-lab) | 将解析、切片、检索、生成、评估作为完整学习链路 | 保持当前轻量技术栈，但把每个阶段的输入/输出契约写清楚 |

**原则：先做 P0/P1，再考虑 Agent、GraphRAG、多模型编排。** 当前知识库仅 13 个切片，最需要的是可靠的基础检索与评估，不是更复杂的流程图。

---

## 4. 目标架构

```text
                    ┌───────────────────────────────────┐
                    │        知识库管理（离线）           │
                    │ 文档 → 章节切片 → 指纹 → 索引版本   │
                    └───────────────┬───────────────────┘
                                    │
                 ┌──────────────────▼──────────────────┐
                 │            检索服务 Retrieval         │
风险记录 ───────→│ query 构造 → 向量检索 + BM25 → RRF     │
                 │ metadata 过滤 → 重排序 → 拒答判断      │
                 └──────────────────┬──────────────────┘
                                    │ Top-K 证据
                 ┌──────────────────▼──────────────────┐
                 │           生成服务 Generation         │
                 │ 交易事实 + 证据 ID → 严格 JSON Schema  │
                 │ 程序回填引用原文 → 人工复核标记         │
                 └──────────────────┬──────────────────┘
                                    │
        ┌───────────────────────────▼───────────────────────────┐
        │ 输出与可观测性：报告 / UI / trace / 成本 / 评估 / 测试 │
        └───────────────────────────────────────────────────────┘
```

### 推荐目录结构

不要继续把同一逻辑复制到 5 个脚本中。可改为：

```text
rag_qa_assistant/
├── config/
│   ├── settings.py              # 模型、路径、阈值等唯一配置源
│   └── prompts.py
├── services/
│   ├── schemas.py               # Pydantic 输入/输出模型
│   ├── knowledge_base.py        # 切片、指纹、索引更新
│   ├── retrieval.py             # vector / BM25 / RRF / rerank / refusal
│   ├── generation.py            # 组装 Prompt、调用模型、解析输出
│   ├── citations.py             # 由程序验证和回填证据引用
│   └── observability.py         # trace 与成本记录
├── scripts/
│   ├── build_index.py
│   ├── batch_process.py
│   ├── evaluate.py
│   ├── test_suite.py
│   └── web_app.py
├── tests/
│   ├── test_retrieval.py
│   ├── test_generation.py
│   └── fixtures/
├── evals/
│   ├── dataset.jsonl
│   └── results/
└── data/
```

`scripts` 是命令行入口；`services` 才是可复用的业务逻辑。Streamlit、批处理、测试都只能调用 service，不应各自实现一套检索规则。

---

## 5. 第一阶段（P0）：让知识库更新正确

### 5.1 问题原理

当前 `08_incremental_update.py` 中：

- 文件被删除：能找到旧 `chunk_id` 并从 Chroma 删除；
- 文件被修改：从 `chunks.json` 去掉旧 chunk，并向 Chroma 添加新 chunk；
- **遗漏：修改文件对应的旧向量没有从 Chroma 删除。**

这会造成 Chroma 中同时存在旧制度和新制度。新旧内容可能有不同 ID，因此系统会随机检索到过期规则。

### 5.2 修复策略

对“新增 / 修改 / 删除”统一处理：

1. 从旧 manifest（旧 `chunks.json`）查出受影响 source 的所有旧 chunk IDs；
2. 在 Chroma 中先删除这些 IDs；
3. 对新增或修改的当前文件重新切片与向量化；
4. 以新的完整 manifest 覆盖保存；
5. 检查 `collection.count() == len(manifest)`；不一致则失败，不发布索引版本。

### 5.3 核心伪代码

```python
affected_sources = set(added + modified + deleted)
old_ids = [
    chunk["chunk_id"]
    for chunk in old_chunks
    if chunk["source"] in affected_sources
]

if old_ids:
    collection.delete(ids=old_ids)  # 修改与删除都要删旧向量

new_chunks = build_chunks_for(added + modified)
collection.add(... new_chunks ...)

save_manifest(all_unaffected_chunks + new_chunks)
assert collection.count() == len(new_manifest)
```

### 5.4 需要新增的测试

准备一个临时知识库：

1. 写入“第 3 条金额超过 50 万需审批”；建索引；
2. 改写为“金额超过 80 万需审批”；运行增量更新；
3. 查询“金额审批阈值”；
4. 断言结果中不再含“50 万”，且 collection 数量与 manifest 一致。

这是审计场景的 P0：制度版本错误比“模型回答不优雅”严重得多。

---

## 6. 第二阶段（P0/P1）：把纯向量检索升级为混合检索与可靠拒答

### 6.1 为什么纯向量检索会出问题

向量检索的工作是：无论 query 是什么，都在库里找“最接近”的文本。它不会天然返回“库里没有答案”。当知识库很小（当前只有 13 块）时，任何无关 query 都会被迫匹配到某段制度，因此 0.5 这种绝对阈值很容易失效。

对于审计制度，还存在精确词：

- “第 20 条”；
- “50 万”；
- “7 天内 6 次”；
- “22:00 至 6:00”。

这些是关键词检索的强项，不应完全交给语义向量。

### 6.2 混合检索是什么

并行运行两种检索：

```text
同一 query
  ├─ 向量检索：找“语义相近”内容
  └─ BM25 检索：找“关键词/条款号/金额”相近内容
                 ↓
           RRF 融合排序
                 ↓
             Top-K 候选
```

**BM25** 是经典文本检索算法，词在某篇文档中越重要、在所有文档中越少见，得分通常越高。它不需要调用模型，适合精确条款。

**RRF（Reciprocal Rank Fusion，倒数排名融合）** 不强行比较 BM25 分数和向量分数的绝对大小，而只看排名。某段文本在两张榜单中都排前面，就更可靠。

公式：

```text
RRF(d) = Σ 1 / (k + rank_i(d))
```

`d` 是某个文档，`rank_i(d)` 是它在第 i 种检索中的排名，`k` 是平滑常数，工程上常从 60 开始试验。

### 6.3 最小可用实现建议

当前数据量很小，可先使用 `rank-bm25`：

```bash
pip install rank-bm25 pydantic
```

建立 BM25 索引时，与 `chunks.json` 保持相同顺序，并写入文件指纹/索引版本：

```python
from rank_bm25 import BM25Okapi

def tokenize(text: str) -> list[str]:
    # 初版：保留业务短语和数字；中文生产环境建议接入 jieba 或自定义词典
    return list(text.replace(" ", ""))

corpus = [tokenize(chunk["text"]) for chunk in chunks]
bm25 = BM25Okapi(corpus)
scores = bm25.get_scores(tokenize(query))
```

> 中文分词提示：逐字切分是最小 demo，能运行但不是最佳方案。第二版应引入审计领域词典，例如“重复付款”“审批阈值”“三单匹配”“设备指纹”，避免关键业务短语被拆散。

RRF 逻辑：

```python
def reciprocal_rank_fusion(rank_lists: list[list[str]], k: int = 60) -> list[str]:
    score_by_id = {}
    for ranked_ids in rank_lists:
        for rank, doc_id in enumerate(ranked_ids, start=1):
            score_by_id[doc_id] = score_by_id.get(doc_id, 0) + 1 / (k + rank)
    return [doc_id for doc_id, _ in sorted(score_by_id.items(), key=lambda x: x[1], reverse=True)]
```

### 6.4 不能只加混合检索：还要增加“拒答”判断

推荐将单一 `top_score < 0.5` 改为多条件决策：

```text
允许生成，需要同时满足：
1. 向量相似度达到校准门槛；
2. BM25 命中至少一个风险类型核心词/条款词；
3. Top-1 与 Top-2 的分差足够，或者重排序模型判定相关；
4. 检索证据来自允许的 source_type（优先制度）。

否则：输出“证据不足，建议人工复核”，且不调用 LLM。
```

请注意：门槛必须在独立验证集上调参，而不是凭直觉设置。

### 6.5 构造拒答校准集

至少准备 3 类 query，每类先做 20 条：

| 类型 | 例子 | 期望 |
|---|---|---|
| 正例 | “同一供应商 30 天内相同金额付款如何核查” | 检索到第四章重复付款 |
| 近似但不适用 | “差旅住宿标准如何审批” | 若知识库无此条，拒答 |
| 完全无关 | “量子计算机天气预报” | 拒答 |

不要只看“系统能否找到正确制度”，还要看 **它是否会在没有制度依据时克制地拒答**。对审计产品，后者同样重要。

---

## 7. 第三阶段（P1）：让引用不可伪造，让输出可校验

### 7.1 当前做法的漏洞

Prompt 要求模型返回：

```json
"retrieved_policy": {
  "source": "...",
  "content": "制度原文",
  "relevance_score": 0.92
}
```

模型通常会照做，但它理论上可以：

- 改写原文；
- 把第 2 条的内容写成第 1 条的来源；
- 虚构不存在的条款；
- 返回字符串形式的分数或漏掉字段。

关键原则是：**模型只能选择证据 ID，不能自由生成证据原文和来源。**

### 7.2 新的输出结构

让模型只输出 `selected_evidence_ids`：

```json
{
  "evidence": {
    "key_facts": "该笔交易金额可能超过制度规定的审批阈值，需核实审批链路。"
  },
  "selected_evidence_ids": ["audit_policy.txt_015"],
  "suggested_action": [
    "核实财务总监审批记录。",
    "比对审批时间与交易时间。"
  ],
  "needs_human_review": true
}
```

然后由 Python 根据 ID 从本次 Top-K 检索结果中回填：

```python
def attach_citations(model_output, retrieved_chunks):
    by_id = {chunk["chunk_id"]: chunk for chunk in retrieved_chunks}
    selected = []
    for chunk_id in model_output.selected_evidence_ids:
        if chunk_id not in by_id:
            raise ValueError(f"模型引用了未检索到的证据：{chunk_id}")
        chunk = by_id[chunk_id]
        selected.append({
            "chunk_id": chunk["chunk_id"],
            "source": chunk["source"],
            "section": chunk["section"],
            "content": chunk["text"],  # 原文只能由程序填充
        })
    return selected
```

这样能保证展示给审计人员的文本一定来自真实检索证据。

### 7.3 使用 Pydantic 校验 Schema

Pydantic 是 Python 的数据校验库。它将“我希望模型返回什么”写成可执行的类型规则。

```python
from pydantic import BaseModel, Field

class Evidence(BaseModel):
    key_facts: str = Field(min_length=1, max_length=500)

class AnalysisDraft(BaseModel):
    evidence: Evidence
    selected_evidence_ids: list[str] = Field(min_length=1, max_length=3)
    suggested_action: list[str] = Field(min_length=1, max_length=5)
    needs_human_review: bool
```

解析时：

```python
draft = AnalysisDraft.model_validate_json(raw_llm_text)
if draft.needs_human_review is not True:
    raise ValueError("安全护栏：必须人工复核")
```

如果模型漏字段、给错类型、选了 4 条证据，程序会立即发现。失败时可进行一次“只修复 JSON 格式”的重试；仍失败则输出“生成失败，建议人工复核”，而不是猜测。

---

## 8. 第四阶段（P1）：建立真正的评估闭环

### 8.1 你现在的指标够衡量什么？

当前的 Hit@1、Hit@3、MRR、关键词覆盖、章节准确率能够评价 **检索排名**。它们不能回答：

- 模型报告是否忠实于检索原文？
- 建议行动是否真的适合风险？
- JSON 是否正确？
- 系统是否对无依据问题正确拒答？

所以要分层评估。

### 8.2 推荐的评估数据格式

使用 JSONL（每行一个 JSON 对象），方便追加：

```json
{
  "case_id": "dup_001",
  "query": "同一供应商 30 天内相同金额付款如何核查",
  "risk_type": "重复付款",
  "risk": {"amount": 24119.44, "supplier": "供应商F公司"},
  "expected_chunk_ids": ["audit_policy.txt_016"],
  "expected_keywords": ["同一供应商", "30天", "三方核对"],
  "should_abstain": false,
  "reference_action_points": ["付款流水", "采购订单", "入库单"]
}
```

对无关/不覆盖场景：

```json
{
  "case_id": "abstain_001",
  "query": "员工差旅住宿标准如何审批",
  "risk_type": "通用合规",
  "expected_chunk_ids": [],
  "expected_keywords": [],
  "should_abstain": true
}
```

### 8.3 指标分四层

| 层 | 指标 | 问题 |
|---|---|---|
| 索引层 | chunk 数、manifest 与 collection 一致性、过期 chunk 数 | 知识库是否正确？ |
| 检索层 | Recall@K、MRR、nDCG、章节准确率、拒答 precision/recall | 找到的证据对不对？不该答时能否拒答？ |
| 生成层 | JSON Schema 通过率、引用有效率、事实忠实度、建议覆盖率 | 回答有没有基于证据？ |
| 稳定性层 | 多次运行的一致性、P95 延迟、单条成本、异常率 | 同样输入会不会漂移？能否稳定使用？ |

### 8.4 引入 Ragas 的正确姿势

Ragas 适合补充生成质量指标，例如：

- **Faithfulness（忠实度）**：回答是否被检索 context 支持；
- **Answer Relevancy（回答相关性）**：是否真正回答了问题；
- **Context Precision / Recall**：检索上下文是否准确、是否覆盖所需事实。

但要清楚：这类指标通常有 LLM 参与评判，结果不是绝对真相，也有成本和波动。推荐先在 30～50 条人工标注案例上运行，并保存每个 case 的原始 context、答案、评分理由，用于人工抽查。

### 8.5 评估纪律

1. 先冻结一个 baseline：当前系统的指标、模型版本、知识库版本；
2. 一次只改一个变量，例如先只加 BM25；
3. 在同一份盲测集上对比；
4. 同时报告提升和退化；
5. 不以 10 条、同源 query 的 100% 作为对外性能承诺；
6. 新增真实失败案例时，先加入回归集，再修复。

---

## 9. 第五阶段（P1/P2）：金融场景的输出稳定性与人工闭环

### 9.1 为什么 temperature=0 仍要测试

`temperature=0` 会降低随机性，但 API 服务、模型版本、推理路径、JSON 格式细节仍可能让结果不同。金融/审计场景不应只检查“这次是否答对”，还要检查“同一输入重复运行是否改变建议或引用”。

### 9.2 一致性测试

对每个高风险评测样本运行 5 次，记录：

```text
一致性率 = 关键字段完全一致的次数 / 总次数
```

关键字段包括：

- 是否拒答；
- 选择的制度 chunk ID；
- `needs_human_review`；
- 建议行动包含的动作类别。

建议不要求自然语言逐字相同，因为这会过于严格；更重要的是决策和引用不变。

### 9.3 人工复核反馈结构

不要只让用户“点一个满意/不满意”。审计人员的反馈应结构化：

```json
{
  "trace_id": "...",
  "retrieval_correct": true,
  "citation_correct": true,
  "suggested_action_usable": false,
  "final_disposition": "误报",
  "reviewer_note": "同金额为分期付款，不属于重复支付"
}
```

这能逐步积累真正的盲测集，也能帮助区分：到底是上游风险规则误报、检索问题，还是生成问题。

---

## 10. Streamlit 页面应如何升级

当前 UI 已有单条分析、批量浏览、系统看板。建议增加以下审计可用信息：

### 单条风险页

- 显示 `trace_id`、知识库版本、模型版本、运行时间；
- 每个证据卡片显示：文档名、章节、chunk ID、vector rank、BM25 rank、RRF 分数；
- 明确区分“制度条款”和“历史案例”；
- 显示“系统拒答原因”：例如“未命中风险类型核心词”；
- 将模型自报 `confidence` 改名为“模型自评（非正确率）”，或先隐藏；
- 增加人工复核表单和导出功能。

### 系统看板

增加：

- 索引版本与最近更新时间；
- `chunks.json` 数量与 Chroma collection 数量的一致性状态；
- 最近 7 天拒答率、生成失败率、引用校验失败率；
- 分风险类型的 Recall@3、拒答准确率和人工采纳率；
- P50/P95 时延与平均单条成本。

---

## 11. 推荐实施排期

### 第 1 周：正确性底座

- [ ] 修复增量更新：修改文件先删旧向量；
- [ ] 加 manifest / collection 一致性检查；
- [ ] 将 03/05/06/09/10 的公共逻辑抽到 `services`；
- [ ] 增加 20 条正例 + 20 条无关/缺失知识的拒答测试；
- [ ] 修复现有测试中“无关查询却不拒答”的失败。

**验收：** 所有测试通过；修改制度后旧条款不再可检索；无关 query 拒答率达到事先设定目标。

### 第 2 周：检索升级

- [ ] 加 BM25；
- [ ] 加向量 + BM25 的 RRF；
- [ ] 为四类风险建立关键词/短语词表；
- [ ] 在冻结数据集上对比 vector-only 与 hybrid；
- [ ] 选择表现更好的策略，保存实验报告。

**验收：** 不只报告平均分，也报告每个风险类型、拒答样本和错误案例。

### 第 3 周：生成可靠性

- [ ] 引入 Pydantic；
- [ ] 模型只选择 evidence IDs，程序回填原文；
- [ ] JSON 错误一次重试、二次失败安全降级；
- [ ] 增加引用合法性、Schema 通过率测试。

**验收：** 输出中不存在非 Top-K 的引用；关键字段缺失不会进入最终报告。

### 第 4 周：评估与产品闭环

- [ ] 引入 Ragas 或等价的端到端评估；
- [ ] 加重复调用一致性测试；
- [ ] UI 展示 trace / 版本 / 拒答原因；
- [ ] 收集人工复核反馈。

**验收：** 每次发布都有版本化评估报告；能定位一次失败发生在哪个阶段。

---

## 12. 可以保留在简历中的升级成果表述

完成上述 P0/P1 后，可以将项目描述升级为：

> 构建面向交易风险复核的 RAG 审计辅助系统：对制度文档实施章节级切片、版本化增量索引与混合检索（BM25 + 向量 + RRF），为上游识别的风险交易生成可追溯的结构化复核建议。通过 Pydantic Schema、证据 ID 程序化回填和强制人工复核降低引用幻觉；建立检索、生成、拒答和重复调用一致性的分层评估闭环，并以 Streamlit 提供证据可视化与人工反馈入口。

不要写“自动完成审计”或“准确率 100%”。更可信的表达是：

- 系统定位为审计辅助与初筛，不替代最终判断；
- 已构建的评估集规模、指标口径和版本；
- 已发现并修复/正在跟踪的失败类型；
- 以“引用有效率、拒答准确率、人工采纳率”等指标表达实际价值。

---

## 13. 阅读顺序

如果你是 RAG 初学者，建议按此顺序学习并动手：

1. 阅读本项目 `docs/project-deep-dive.html` 的“零基础先修课”和“Python 语法读码”；
2. 重新跑一遍现有 01 → 05，观察 `chunks.json`、`retrieval_log.csv`、`structured_risk_report.json`；
3. 先修复增量更新 bug，并为它写测试；
4. 加入 BM25，只比较检索结果，不急于调用 LLM；
5. 加 RRF，做 vector-only vs hybrid 对比；
6. 把模型输出改为 evidence ID + Pydantic；
7. 最后再接入 Ragas 与人工反馈。

每完成一步，都保存：代码版本、知识库版本、评测集版本、实验结果和失败样本。这比堆叠更多框架更能说明你的工程能力。

---

## 参考仓库

- [Ragas：RAG 评估与错误分析](https://github.com/vibrantlabsai/ragas)
- [rag-fusion：混合检索、RRF 与评估](https://github.com/Raudaschl/rag-fusion)
- [IBM output-drift-financial-llms：金融场景的输出漂移与可复现性](https://github.com/ibm-client-engineering/output-drift-financial-llms)
- [fin-rag-lab：金融文档 RAG 学习链路](https://github.com/zyziyun/fin-rag-lab)

