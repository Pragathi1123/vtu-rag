from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from rag import answer

app = FastAPI(title="VTU Textbook Assistant")


class Ask(BaseModel):
    question: str
    subject: str | None = None
    style: str = "detailed"


@app.post("/ask")
def ask(body: Ask):
    if not body.question.strip():
        return {"answer": "Please type a question.", "sources": []}
    try:
        return answer(body.question.strip(), body.subject or None, body.style)
    except Exception as e:
        print("Server error:", e)
        return {"answer": "Something went wrong. Please try again.", "sources": []}


app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def home():
    return FileResponse("static/index.html")