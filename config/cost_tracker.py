# config/cost_tracker.py
# 成本可观测性模块 —— 记录每次 API 调用的 token 消耗与费用

import json
import os
from datetime import datetime

# 智谱 AI 定价（元/百万 token）
PRICING = {
    "glm-4-flash": {"input": 0.0, "output": 0.0},       # Flash 免费档
    "glm-4": {"input": 50.0, "output": 50.0},            # 标准定价参考
    "embedding-3": {"input": 0.5, "output": 0.0},
}

class CostTracker:
    def __init__(self):
        self.records = []
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost = 0.0

    def record_chat(self, model, usage):
        """记录一次 Chat Completion 调用"""
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        total_tokens = usage.get("total_tokens", prompt_tokens + completion_tokens)

        pricing = PRICING.get(model, {"input": 0.0, "output": 0.0})
        cost = (prompt_tokens * pricing["input"] + completion_tokens * pricing["output"]) / 1_000_000

        self.total_input_tokens += prompt_tokens
        self.total_output_tokens += completion_tokens
        self.total_cost += cost

        self.records.append({
            "type": "chat",
            "model": model,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "cost_yuan": round(cost, 6),
        })

    def record_embedding(self, model, text_length, token_count=None):
        """记录一次 Embedding 调用（智谱 embedding 不返回 token 数，按字符估算）"""
        est_tokens = token_count if token_count else text_length // 2
        pricing = PRICING.get(model, {"input": 0.5, "output": 0.0})
        cost = est_tokens * pricing["input"] / 1_000_000

        self.total_input_tokens += est_tokens
        self.total_cost += cost

        self.records.append({
            "type": "embedding",
            "model": model,
            "text_length": text_length,
            "est_tokens": est_tokens,
            "cost_yuan": round(cost, 6),
        })

    def summary(self):
        """生成成本摘要"""
        chat_calls = sum(1 for r in self.records if r["type"] == "chat")
        embed_calls = sum(1 for r in self.records if r["type"] == "embedding")
        return {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "total_calls": len(self.records),
            "chat_calls": chat_calls,
            "embedding_calls": embed_calls,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_input_tokens + self.total_output_tokens,
            "total_cost_yuan": round(self.total_cost, 6),
            "avg_cost_per_record": round(self.total_cost / max(chat_calls, 1), 6),
        }

    def save_report(self, path):
        """保存详细成本报告"""
        report = {
            "summary": self.summary(),
            "details": self.records,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        return report["summary"]
