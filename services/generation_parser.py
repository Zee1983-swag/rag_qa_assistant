import json
import re

from pydantic import ValidationError

from citations import CitationValidationError, attach_citations
from schemas import AnalysisDraft


class LLMResponseFormatError(ValueError):
    """模型输出不是符合约定的 JSON 时抛出。"""


def extract_json_payload(raw_text: str) -> dict:
    """从纯 JSON、Markdown 代码块或夹带说明文字的响应中提取 JSON 对象。"""
    text = (raw_text or "").strip()

    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    if fenced:
        text = fenced.group(1)
    elif not text.startswith("{"):
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end < start:
            raise LLMResponseFormatError("模型输出中没有找到 JSON 对象")
        text = text[start:end + 1]

    try:
        payload = json.loads(text)
    except json.JSONDecodeError as error:
        raise LLMResponseFormatError(f"模型输出不是合法 JSON：{error.msg}") from error

    if not isinstance(payload, dict):
        raise LLMResponseFormatError("模型输出的 JSON 根节点必须是对象")

    return payload


def parse_analysis_draft(raw_text: str) -> AnalysisDraft:
    """解析并按 Pydantic Schema 校验模型草稿。"""
    payload = extract_json_payload(raw_text)

    try:
        return AnalysisDraft.model_validate(payload)
    except ValidationError as error:
        raise LLMResponseFormatError(
            f"模型输出不符合 AnalysisDraft 约束：{error}"
        ) from error


def build_verified_result(raw_text: str, retrieved_chunks: list[dict]) -> dict:
    """
    验证模型草稿，并把模型选择的 chunk_id 回填为系统实际检索到的制度内容。
    """
    draft = parse_analysis_draft(raw_text)

    try:
        citations = attach_citations(draft, retrieved_chunks)
    except CitationValidationError:
        raise

    return {
        "evidence": draft.evidence.model_dump(),
        "selected_evidence_ids": draft.selected_evidence_ids,
        "retrieved_policies": citations,
        "suggested_actions": draft.suggested_actions,
        "needs_human_review": draft.needs_human_review,
    }
