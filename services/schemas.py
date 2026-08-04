# services/schemas.py
# LLM 结构化输出的可执行契约。

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class EvidenceDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")
    """模型对风险事实的简短摘要。"""

    key_facts: str = Field(
        min_length=1,
        max_length=500,
        description="基于输入交易事实的一句话风险摘要",
    )


class AnalysisDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")
    """
    LLM 可生成的分析草稿。

    重要限制：
    - 模型只能选择 selected_evidence_ids；
    - source、section、content 由 Python 程序回填；
    - needs_human_review 只能为 True。
    """

    evidence: EvidenceDraft

    selected_evidence_ids: list[str] = Field(
        min_length=1,
        max_length=3,
        description="只能从本次检索 Top-K 中选择的 chunk_id",
    )

    suggested_actions: list[str] = Field(
        min_length=1,
        max_length=5,
        description="每项以动词开头的人工复核建议",
    )

    needs_human_review: Literal[True] = True
