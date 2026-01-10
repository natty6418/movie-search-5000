import sys
import os
import json
import numpy as np
from typing import List, Optional, Literal, Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Add project root to sys.path to allow importing cli.lib
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from cli.lib.hybrid_search import HybridSearch, cross_encoder_rerank_results
from cli.lib.local_llm import rag_answer, summarize, generate_citations, question_answering

app = FastAPI(title="RAG Search Engine API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Models
class SearchQuery(BaseModel):
    query: str
    limit: int = 10

class HybridSearchQuery(SearchQuery):
    mode: Literal["weighted", "rrf"]
    alpha: float = 0.5 # For weighted
    k: int = 60 # For RRF

class RAGQuery(BaseModel):
    query: str
    mode: Literal["rag", "summarize", "citation", "question"]
    limit: int = 5

# Global instances
MOVIES_PATH = os.path.join(os.path.dirname(__file__), "../data/movies.json")
HYBRID_SEARCH_ENGINE = None

def get_engine():
    global HYBRID_SEARCH_ENGINE
    if HYBRID_SEARCH_ENGINE is None:
        if not os.path.exists(MOVIES_PATH):
             raise RuntimeError(f"Movies file not found at {MOVIES_PATH}")
        with open(MOVIES_PATH, "r", encoding="utf-8") as f:
            movies_list = json.load(f)["movies"]
        HYBRID_SEARCH_ENGINE = HybridSearch(movies_list)
    return HYBRID_SEARCH_ENGINE

def make_serializable(obj):
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [make_serializable(i) for i in obj]
    return obj

@app.on_event("startup")
def startup_event():
    # Pre-load the engine on startup
    try:
        get_engine()
        print("Search engine initialized successfully.")
    except Exception as e:
        print(f"Failed to initialize search engine: {e}")

@app.get("/")
def health_check():
    return {"status": "ok", "message": "RAG Search Engine API is running"}

@app.post("/search/keyword")
def keyword_search(request: SearchQuery):
    engine = get_engine()
    results = engine._bm25_search(request.query, request.limit)
    formatted = []
    for doc, score in results:
        formatted.append({
            "id": doc["id"],
            "title": doc["title"],
            "description": doc["description"],
            "score": float(score)
        })
    return {"results": formatted}

@app.post("/search/semantic")
def semantic_search(request: SearchQuery):
    engine = get_engine()
    # chunked_semantic_search returns list of dicts with score, title, etc.
    results = engine._chunked_semantic_search(request.query, request.limit)
    return {"results": make_serializable(results)}

@app.post("/search/hybrid")
def hybrid_search_endpoint(request: HybridSearchQuery):
    engine = get_engine()
    if request.mode == "weighted":
        results = engine.weighted_search(request.query, request.alpha, request.limit)
        formatted = []
        for doc_id, item in results:
             formatted.append({
                "id": doc_id,
                "title": item["doc"]["title"],
                "description": item["doc"]["description"],
                "score": float(item["hybrid_score"]),
                "details": make_serializable({k: v for k, v in item.items() if k != "doc"})
             })
        return {"results": formatted}
    elif request.mode == "rrf":
        results = engine.rrf_search(request.query, request.k, request.limit)
        formatted = []
        for doc_id, item in results:
             formatted.append({
                "id": doc_id,
                "title": item["doc"]["title"],
                "description": item["doc"]["description"],
                "score": float(item["rrf_score"]),
                "details": make_serializable({k: v for k, v in item.items() if k != "doc"})
             })
        return {"results": formatted}

@app.post("/rag")
def rag_endpoint(request: RAGQuery):
    engine = get_engine()
    query = request.query
    limit = request.limit
    
    if request.mode == "rag":
        # Basic RAG uses RRF + direct answer
        results = engine.rrf_search(query, k=60, limit=5)
        docs = [item["doc"] for _, item in results]
        answer = rag_answer(query, docs)
        return {"answer": answer, "docs": make_serializable(docs)}
    
    # Advanced RAG modes use RRF + Re-ranking
    raw_results = engine.rrf_search(query, k=60, limit=limit * 5)
    reranked = cross_encoder_rerank_results(raw_results, query, limit)
    
    # reranked structure: [ ((doc_id, item_dict), score), ... ]
    docs = [r[0][1]["doc"] for r in reranked]
    
    if request.mode == "summarize":
        response = summarize(query, docs)
        return {"summary": response, "docs": make_serializable(docs)}
    elif request.mode == "citation":
        response = generate_citations(query, docs)
        return {"citations": response, "docs": make_serializable(docs)}
    elif request.mode == "question":
        response = question_answering(query, docs)
        return {"answer": response, "docs": make_serializable(docs)}
    
    raise HTTPException(status_code=400, detail="Invalid mode")