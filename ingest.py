import os
import pymupdf
import chromadb
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

PDF_DIR = "data/pdfs"

print("Loading embedding model (first run downloads it)...")
model = SentenceTransformer("all-MiniLM-L6-v2")

client = chromadb.PersistentClient(path="chroma_db")
col = client.get_or_create_collection("vtu", metadata={"hnsw:space": "cosine"})

splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)

for f in os.listdir(PDF_DIR):
    if not f.lower().endswith(".pdf"):
        continue

    sem, subject, book = f[:-4].split("_", 2)
    doc = pymupdf.open(os.path.join(PDF_DIR, f))

    ids, texts, metas = [], [], []
    for page_no, page in enumerate(doc, start=1):
        text = page.get_text().strip()
        if len(text) < 50:  # skip blank or diagram-only pages
            continue
        for i, chunk in enumerate(splitter.split_text(text)):
            ids.append(f"{f}-{page_no}-{i}")
            texts.append(chunk)
            metas.append({"semester": sem, "subject": subject,
                          "book": book, "page": page_no})

    print(f"{f}: {len(texts)} chunks, creating embeddings...")
    embeddings = model.encode(texts, batch_size=64, show_progress_bar=True).tolist()

    for start in range(0, len(ids), 500):
        col.upsert(ids=ids[start:start+500],
                   documents=texts[start:start+500],
                   embeddings=embeddings[start:start+500],
                   metadatas=metas[start:start+500])

    print(f"Done: {f}")

print("Total chunks in database:", col.count())