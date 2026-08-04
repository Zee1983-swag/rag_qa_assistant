# scripts/01_build_kb.py
# 第一步：读取知识库文档，按章节切片，附加元数据，输出 chunks.json
import os
import re
import json

KB_DIR = "data/knowledge_base"
OUT_PATH = "data/vector_store/chunks.json"

MIN_LEN = 200
MAX_LEN = 1000
OVERLAP = 80

def split_by_chapter(text):
    """按章节（第X章/案例X）切分文档，每章作为一个独立段落块"""
    # 匹配"第一章"、"第1章"、"案例一"、"案例1"等开头的行
    pattern = r"(?=^(?:第[一二三四五六七八九十百零]+[章节]|案例[一二三四五六七八九十百零0-9]+)\s)"
    parts = re.split(pattern, text, flags=re.MULTILINE)
    # 过滤掉空段落，保留有内容的
    parts = [p.strip() for p in parts if p.strip()]
    return parts

def chunk_section(section_text, max_len=MAX_LEN, min_len=MIN_LEN):
    """如果单个章节仍然过长，按段落进一步切分"""
    if len(section_text) <= max_len:
        return [section_text]

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", section_text) if p.strip()]
    chunks = []
    current = ""

    for para in paragraphs:
        if not current:
            current = para
        elif len(current) + len(para) + 2 <= max_len:
            current = current + "\n\n" + para
        else:
            chunks.append(current)
            overlap_text = current[-OVERLAP:] if len(current) > OVERLAP else ""
            current = (overlap_text + "\n\n" + para).strip() if overlap_text else para

    if current.strip():
        chunks.append(current)
    return chunks

def detect_section_title(text):
    """提取章节标题作为元数据"""
    for line in text.splitlines():
        line = line.strip()
        m = re.match(r"^(第[一二三四五六七八九十百零]+[章节]\s*[^\n]*)", line)
        if m:
            return m.group(1).strip()
        m2 = re.match(r"^(案例[一二三四五六七八九十百零0-9]+[：:][^\n]*)", line)
        if m2:
            return m2.group(1).strip()
    return "未标注章节"

def chunk_document(text, source):
    """主切分函数：先按章节切，再按长度切"""
    sections = split_by_chapter(text)
    chunks = []

    for section in sections:
        section_title = detect_section_title(section)
        pieces = chunk_section(section)
        for piece in pieces:
            chunks.append((piece, section_title))

    return chunks

def main():
    os.makedirs("data/vector_store", exist_ok=True)
    all_chunks = []
    chunk_id = 0

    files = sorted(f for f in os.listdir(KB_DIR) if f.endswith(".txt"))
    print(f"已读取 {len(files)} 个文档")

    for fname in files:
        path = os.path.join(KB_DIR, fname)
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        pieces = chunk_document(text, fname)
        for txt, section in pieces:
            all_chunks.append({
                "chunk_id": f"{fname}_{chunk_id:03d}",
                "source": fname,
                "section": section,
                "text": txt,
            })
            chunk_id += 1
        print(f"{fname}: {len(pieces)} 个切片")

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)

    print(f"总计 {len(all_chunks)} 个切片，保存至 {OUT_PATH}")

if __name__ == "__main__":
    main()