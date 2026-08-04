import sys
from pathlib import Path

sys.path.insert(0, str(Path("services").resolve()))

from bm25_retriever import tokenize

text = "同一供应商30天内相同金额重复付款，需进行三单匹配核查。"

print("原始文本：")
print(text)
print()
print("分词结果：")
print(tokenize(text))
