from fastapi import APIRouter
from pydantic import BaseModel
from rag_setup import ask_question

router = APIRouter()

class Question(BaseModel):
    query: str

@router.get("/")
def root():
    return {"message": "RAG API is running!"}

@router.post("/ask")
def ask(question: Question):
    answer = ask_question(question.query)
    return {"question": question.query, "answer": answer}
