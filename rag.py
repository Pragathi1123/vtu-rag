import os
import time
from dotenv import load_dotenv
import chromadb
from sentence_transformers import SentenceTransformer
from google import genai

load_dotenv()

embedder = SentenceTransformer("all-MiniLM-L6-v2")
col = chromadb.PersistentClient(path="chroma_db").get_collection("vtu")
llm = genai.Client(api_key=os.getenv("GEMINI_API_KEY"),
                   http_options={"timeout": 60000})   # give up after 60 seconds

# Main model first, then backups used automatically if it is busy
MODELS = [
    os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite"),
    "gemini-3.1-flash-lite",
    "gemini-3.5-flash",
]
MAX_DIST = 0.55   # tune using your test results

STYLES = {
    "definition": "Give a short 2-3 line definition.",
    "points": "Answer in concise bullet points for exam revision.",
    "detailed": "Give a clear, detailed explanation suitable for exam answers.",
}

RETRY_CODES = ("503", "504", "429", "500", "DEADLINE", "UNAVAILABLE", "timed out", "Timeout")


def retrieve(question, subject=None, k=5):
    emb = embedder.encode([question]).tolist()
    where = {"subject": subject} if subject else None
    r = col.query(query_embeddings=emb, n_results=k, where=where)
    return list(zip(r["documents"][0], r["metadatas"][0], r["distances"][0]))


def ask_llm(prompt):
    """Try each model in turn, with waits between rounds. Returns text or None."""
    for round_no in range(3):
        for model in MODELS:
            try:
                resp = llm.models.generate_content(model=model, contents=prompt)
                return resp.text
            except Exception as e:
                msg = str(e)
                if any(code in msg for code in RETRY_CODES):
                    print(f"  ({model} busy, trying next...)", flush=True)
                    continue
                print("LLM error:", msg[:300])      # e.g. wrong model name or bad key
                continue
        time.sleep(8 * (round_no + 1))              # wait 8s, 16s, 24s between rounds
    return None


def answer(question, subject=None, style="detailed"):
    hits = [h for h in retrieve(question, subject) if h[2] < MAX_DIST]
    if not hits:
        return {"answer": "This is not available in the selected textbooks.",
                "sources": []}

    context = "\n\n".join(
        f"[{m['book']}, page {m['page']}]\n{t}" for t, m, d in hits
    )
    prompt = f"""You are a VTU textbook assistant. Answer ONLY using the context below.
Do not mention "the context" or "the provided text" in your answer; just answer directly.
If the context does not contain the answer, reply exactly:
"This is not available in the selected textbooks."
{STYLES.get(style, STYLES['detailed'])}

CONTEXT:
{context}

QUESTION: {question}"""

    text = ask_llm(prompt)
    if text is None:
        return {"answer": "The AI service is busy or unavailable right now. "
                          "Please try again in a moment.",
                "sources": []}

    sources = sorted({(m["book"], m["page"]) for _, m, _ in hits})
    return {"answer": text,
            "sources": [{"book": b, "page": p} for b, p in sources]}


if __name__ == "__main__":
    while True:
        q = input("\nQuestion (or 'exit'): ")
        if q.strip().lower() == "exit":
            break
        result = answer(q)
        print("\n" + result["answer"])
        print("Sources:", result["sources"])
