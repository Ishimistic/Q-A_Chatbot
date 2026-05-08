from fastapi import FastAPI
from pydantic import BaseModel
from rag_setup import ask_question

app = FastAPI()

class Question(BaseModel):
    query: str

@app.get("/")
def root():
    return {"message": "RAG API is running!"}

@app.post("/ask")
def ask(question: Question):
    answer = ask_question(question.query)
    return {"question": question.query, "answer": answer}
