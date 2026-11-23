import chromadb
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")
col = chromadb.PersistentClient(path="chroma_db").get_collection("vtu")

while True:
    q = input("\nQuestion (or 'exit'): ")
    if q.strip().lower() == "exit":
        break
    emb = model.encode([q]).tolist()
    res = col.query(query_embeddings=emb, n_results=3)
    for text, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
        print(f"\n--- {meta['book']} | page {meta['page']} | distance {dist:.3f}")
        print(text[:400])