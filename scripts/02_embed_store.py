# scripts/02_embed_store.py
# 第二步：读取切片，调用智谱 Embedding 模型向量化，存入 ChromaDB
import os
import json
import chromadb
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

CHUNKS_PATH = "data/vector_store/chunks.json"
DB_PATH = "data/vector_store/chroma_db"
COLLECTION_NAME = "audit_knowledge"
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "embedding-3")

client = OpenAI()

def get_embeddings(texts):
    """批量获取向量（智谱 embedding-3 支持批量，单次最多 64 条）"""
    all_vectors = []
    batch_size = 16
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        resp = client.embeddings.create(input=batch, model=EMBEDDING_MODEL)
        all_vectors.extend([d.embedding for d in resp.data])
    return all_vectors

def main():
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    texts = [c["text"] for c in chunks]
    print(f"正在向量化 {len(texts)} 个切片...")

    chroma_client = chromadb.PersistentClient(path=DB_PATH)

    # 统一使用余弦距离，与 03/05 脚本保持一致
    try:
        collection = chroma_client.get_collection(name=COLLECTION_NAME)
        if collection.count() == len(texts):
            print(f"向量库已存在且数量一致（{collection.count()} 条），跳过创建。")
            print("如需重建，请删除 data/vector_store/chroma_db/ 目录后重跑。")
            return
        else:
            print(f"向量库数量不匹配（已有 {collection.count()}，期望 {len(texts)}），重建中...")
            chroma_client.delete_collection(name=COLLECTION_NAME)
    except Exception:
        print("向量库不存在，首次创建中...")

    collection = chroma_client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )

    vectors = get_embeddings(texts)
    collection.add(
        ids=[c["chunk_id"] for c in chunks],
        embeddings=vectors,
        documents=texts,
        metadatas=[{"source": c["source"], "section": c["section"]} for c in chunks],
    )

    print("向量化完成，已存入 ChromaDB")
    print(f"  - 集合名称：{COLLECTION_NAME}")
    print(f"  - 向量维度：{len(vectors[0])}")
    print(f"  - 存储路径：{DB_PATH}")

if __name__ == "__main__":
    main()