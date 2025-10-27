from fastapi import FastAPI, Query
from utils.vector_store import get_top_k_similar_contexts
from pydantic import BaseModel

app = FastAPI()

class QueryRequest(BaseModel):
    vector_store_path: str
    query: str
    k: int = 5

@app.get("/")
def home():
    return {"message": "FAISS + Supabase API running!"}

@app.post("/query")
def query_faiss(request: QueryRequest):
    results = get_top_k_similar_contexts(request.vector_store_path, request.query, request.k)
    return {"results": results}