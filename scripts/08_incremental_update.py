# scripts/08_incremental_update.py
# 知识库增量更新：文档指纹比对，自动检测新增/修改/删除，只更新变化部分
import os, re, json, hashlib, chromadb
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
KB_DIR = "data/knowledge_base"
DB_PATH = "data/vector_store/chroma_db"
COLLECTION_NAME = "audit_knowledge"
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "embedding-3")
FINGERPRINT_PATH = "data/vector_store/doc_fingerprints.json"
CHUNKS_PATH = "data/vector_store/chunks.json"

MIN_LEN = 200
MAX_LEN = 1000
OVERLAP = 80

client = OpenAI()

# ============================================================
# 切片函数（与 01_build_kb.py 完全一致，保持一致性）
# ============================================================
def split_by_chapter(text):
    pattern = r"(?=^(?:第[一二三四五六七八九十百零]+[章节]|案例[一二三四五六七八九十百零0-9]+)\s)"
    parts = re.split(pattern, text, flags=re.MULTILINE)
    return [p.strip() for p in parts if p.strip()]

def chunk_section(section_text, max_len=MAX_LEN, min_len=MIN_LEN):
    if len(section_text) <= max_len:
        return [section_text]
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", section_text) if p.strip()]
    chunks, current = [], ""
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
    sections = split_by_chapter(text)
    chunks = []
    for section in sections:
        section_title = detect_section_title(section)
        for piece in chunk_section(section):
            chunks.append((piece, section_title))
    return chunks

# ============================================================
# 指纹管理
# ============================================================
def compute_fingerprint(text):
    """计算文档内容的 MD5 指纹"""
    return hashlib.md5(text.encode("utf-8")).hexdigest()

def load_fingerprints():
    """加载已保存的文档指纹"""
    if os.path.exists(FINGERPRINT_PATH):
        with open(FINGERPRINT_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_fingerprints(fingerprints):
    """保存文档指纹"""
    with open(FINGERPRINT_PATH, "w", encoding="utf-8") as f:
        json.dump(fingerprints, f, ensure_ascii=False, indent=2)

def get_embeddings(texts):
    """批量获取向量"""
    all_vectors = []
    batch_size = 16
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        resp = client.embeddings.create(input=batch, model=EMBEDDING_MODEL)
        all_vectors.extend([d.embedding for d in resp.data])
    return all_vectors

# ============================================================
# 增量更新主逻辑
# ============================================================
def incremental_update():
    os.makedirs("data/vector_store", exist_ok=True)

    # 1. 扫描当前知识库目录
    current_files = sorted(f for f in os.listdir(KB_DIR) if f.endswith(".txt"))
    old_fingerprints = load_fingerprints()
    new_fingerprints = {}

    added, modified, deleted = [], [], []

    for fname in current_files:
        path = os.path.join(KB_DIR, fname)
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        fp = compute_fingerprint(text)
        new_fingerprints[fname] = fp

        if fname not in old_fingerprints:
            added.append(fname)
        elif old_fingerprints[fname] != fp:
            modified.append(fname)

    for fname in old_fingerprints:
        if fname not in new_fingerprints:
            deleted.append(fname)

    if not added and not modified and not deleted:
        print("知识库无变更，跳过更新。")
        return

    print(f"增量更新检测：")
    print(f"  新增文档：{len(added)} 个 {added if added else ''}")
    print(f"  修改文档：{len(modified)} 个 {modified if modified else ''}")
    print(f"  删除文档：{len(deleted)} 个 {deleted if deleted else ''}")

    # 2. 连接向量库
    chroma_client = chromadb.PersistentClient(path=DB_PATH)
    try:
        collection = chroma_client.get_collection(name=COLLECTION_NAME)
    except Exception:
        collection = chroma_client.get_or_create_collection(
            name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"})

        # 3. 先加载旧的切片清单。
    # chunks.json 是“当前知识库应该有哪些切片”的唯一事实来源。
    if os.path.exists(CHUNKS_PATH):
        with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
            all_chunks = json.load(f)
    else:
        all_chunks = []

    # 4. 对“修改”和“删除”的文件，先从 Chroma 删除旧向量。
    # 新增文件没有旧向量，因此不需要删除。
    affected_sources = set(modified + deleted)

    if affected_sources:
        old_ids = [
            chunk["chunk_id"]
            for chunk in all_chunks
            if chunk["source"] in affected_sources
        ]

        if old_ids:
            collection.delete(ids=old_ids)
            print(
                f"  已从向量库删除 {len(old_ids)} 个旧切片："
                f"{sorted(affected_sources)}"
            )

        # 同时从清单中删除旧记录。
        # 注意：删除文件和修改文件都必须移除旧清单。
        all_chunks = [
            chunk for chunk in all_chunks
            if chunk["source"] not in affected_sources
        ]

    # 5. 对新增和修改文件，重新切片、向量化并写入 Chroma。
    to_update = added + modified
    new_chunks = []

    if to_update:
        max_id = max(
            (int(chunk["chunk_id"].split("_")[-1]) for chunk in all_chunks),
            default=-1,
        )
        chunk_id = max_id + 1

        for fname in to_update:
            path = os.path.join(KB_DIR, fname)
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()

            pieces = chunk_document(text, fname)

            for txt, section in pieces:
                new_chunks.append({
                    "chunk_id": f"{fname}_{chunk_id:03d}",
                    "source": fname,
                    "section": section,
                    "text": txt,
                })
                chunk_id += 1

            print(f"  {fname}: {len(pieces)} 个新切片")

        if new_chunks:
            texts = [chunk["text"] for chunk in new_chunks]
            vectors = get_embeddings(texts)

            collection.add(
                ids=[chunk["chunk_id"] for chunk in new_chunks],
                embeddings=vectors,
                documents=texts,
                metadatas=[
                    {"source": chunk["source"], "section": chunk["section"]}
                    for chunk in new_chunks
                ],
            )

    # 无论本次是新增、修改还是删除，都重新保存完整清单。
    all_chunks.extend(new_chunks)

    with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)

    # 索引完整性校验：数据库数量必须和清单数量一致。
    if collection.count() != len(all_chunks):
        raise RuntimeError(
            f"索引不一致：Chroma={collection.count()}，"
            f"chunks.json={len(all_chunks)}"
        )
    # 5. 保存新指纹
    save_fingerprints(new_fingerprints)

    print(f"\n增量更新完成：向量库现有 {collection.count()} 条切片")
    print(f"指纹记录已保存至 {FINGERPRINT_PATH}")

if __name__ == "__main__":
    incremental_update()
