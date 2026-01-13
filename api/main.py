import sys
import os
import json
import numpy as np
from typing import List, Optional, Literal, Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import asyncio
from collections import defaultdict

# Add project root to sys.path to allow importing cli.lib
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from cli.lib.hybrid_search import HybridSearch, cross_encoder_rerank_results
from cli.lib.local_llm import rag_answer, summarize, generate_citations, question_answering
from cli.lib.agent import build_graph

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
    enhance_mode: Literal["none", "fix_spelling", "rewrite", "expand"] = "none"
    bm25_type: Literal["unigram", "bigram", "combined"] = "combined"

class HybridSearchQuery(SearchQuery):
    mode: Literal["weighted", "rrf"]
    alpha: float = 0.5 # For weighted
    k: int = 60 # For RRF

class RAGQuery(BaseModel):
    query: str
    mode: Literal["rag", "summarize", "citation", "question"]
    limit: int = 5
    enhance_mode: Literal["none", "fix_spelling", "rewrite", "expand"] = "none"
    bm25_type: Literal["unigram", "bigram", "combined"] = "combined"

class AgentQuery(BaseModel):
    query: str
    chat_history: List[Dict[str, str]] = [] # Optional history
    session_id: Optional[str] = None  # For tracking conversations

# Global instances
MOVIES_PATH = os.path.join(os.path.dirname(__file__), "../data/movies.json")
HYBRID_SEARCH_ENGINE = None
AGENT_GRAPH = None
VISITOR_COUNT = 0
VISITOR_FILE = os.path.join(os.path.dirname(__file__), "visitors.txt")
CHAT_SESSIONS = defaultdict(list)  # session_id -> messages list

if os.path.exists(VISITOR_FILE):
    with open(VISITOR_FILE, "r") as f:
        try:
            VISITOR_COUNT = int(f.read().strip())
        except:
            VISITOR_COUNT = 1337

def get_engine():
    global HYBRID_SEARCH_ENGINE
    if HYBRID_SEARCH_ENGINE is None:
        if not os.path.exists(MOVIES_PATH):
             raise RuntimeError(f"Movies file not found at {MOVIES_PATH}")
        with open(MOVIES_PATH, "r", encoding="utf-8") as f:
            movies_list = json.load(f)["movies"]
        HYBRID_SEARCH_ENGINE = HybridSearch(movies_list)
    return HYBRID_SEARCH_ENGINE

def get_agent():
    global AGENT_GRAPH
    if AGENT_GRAPH is None:
        engine = get_engine()
        AGENT_GRAPH = build_graph(engine)
    return AGENT_GRAPH

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

def apply_query_enhancement(query: str, mode: str):
    from cli.lib.local_llm import enhance_query, rewrite_query, expand_query
    
    if mode == "fix_spelling":
        return enhance_query(query)
    elif mode == "rewrite":
        return rewrite_query(query)
    elif mode == "expand":
        # expand_query returns only the expansion, so we append it
        expansion = expand_query(query)
        if expansion == query: # No expansion happened or error
             return query
        return f"{query} {expansion}"
    return query

@app.on_event("startup")
def startup_event():
    # Pre-load the engine on startup
    try:
        get_engine()
        get_agent() # Pre-load agent
        print("Search engine and Agent initialized successfully.")
    except Exception as e:
        print(f"Failed to initialize search engine: {e}")

@app.get("/")
def health_check():
    return {"status": "ok", "message": "RAG Search Engine API is running"}

@app.post("/agent")
async def agent_endpoint(request: AgentQuery):
    graph = get_agent()

    # Construct initial state for the new agent structure
    # The new agent expects: query, messages, retrieved_docs, limit, action, needs_search
    initial_state = {
        "query": request.query,
        "messages": request.chat_history.copy(),  # Preserve chat history
        "retrieved_docs": [],
        "limit": 5,
        "action": "",
        "needs_search": True  # Will be determined by classify_query_type node
    }

    try:
        # Run the graph - use invoke (not ainvoke) since build_graph returns sync graph
        final_state = graph.invoke(initial_state)
        
        # Extract the final assistant response from messages
        messages = final_state.get("messages", [])
        last_assistant_msg = None
        for msg in reversed(messages):
            if msg.get("role") == "assistant":
                content = msg.get("content", "")
                # Skip the "Enhanced query:" messages
                if not content.startswith("Enhanced query:"):
                    last_assistant_msg = content
                    break
        
        if not last_assistant_msg:
            last_assistant_msg = "No response generated."
        
        # Format retrieved docs for frontend (if needed)
        retrieved_docs = final_state.get("retrieved_docs", [])
        formatted_docs = []
        for doc in retrieved_docs:
            formatted_docs.append({
                "title": doc.get("title", ""),
                "description": doc.get("description", "")
            })
        
        return {
            "answer": last_assistant_msg,
            "messages": make_serializable(messages),  # Return full trace for debugging/UI
            "retrieved_docs": formatted_docs,
            "docs": formatted_docs,  # Alias for consistency with RAG endpoint
            "query_used": final_state.get("query", request.query)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/agent/stream")
async def agent_stream_endpoint(request: AgentQuery):
    """Stream agent execution states in real-time using Server-Sent Events."""
    graph = get_agent()
    
    # Get or create session
    session_id = request.session_id or f"session_{id(request)}"
    chat_history = CHAT_SESSIONS[session_id] if request.session_id else request.chat_history.copy()
    
    # Construct initial state
    initial_state = {
        "query": request.query,
        "messages": chat_history,
        "retrieved_docs": [],
        "limit": 5,
        "action": "",
        "needs_search": True  # Will be determined by classify_query_type node
    }
    
    async def event_generator():
        try:
            # Accumulate state from stream events
            accumulated_state = initial_state.copy()
            
            # Stream the graph execution
            for event in graph.stream(initial_state):
                # event is a dict with node name as key
                node_name = list(event.keys())[0] if event else None
                node_data = event.get(node_name, {}) if node_name else {}
                
                # Update accumulated state with node output
                if node_data:
                    accumulated_state.update(node_data)
                
                # Determine status message based on node
                status_message = ""
                if node_name == "classify_query_type":
                    status_message = "🧠 Understanding your question..."
                elif node_name == "process_query":
                    status_message = "🔍 Processing your query..."
                elif node_name == "search_movies":
                    status_message = "🎬 Searching movie database..."
                elif node_name == "classify":
                    status_message = "🤔 Analyzing results..."
                elif node_name == "draft_response":
                    status_message = "✍️ Drafting response..."
                elif node_name == "direct_response":
                    status_message = "💬 Preparing response..."
                
                # Send status update
                if status_message:
                    yield f"data: {json.dumps({'type': 'status', 'message': status_message, 'node': node_name})}\n\n"
                
                # Send partial state data
                if node_data:
                    # Extract useful info to send
                    update_data = {
                        "type": "update",
                        "node": node_name,
                    }
                    
                    # Add specific data based on node
                    if node_name == "search_movies" and "retrieved_docs" in node_data:
                        docs = node_data["retrieved_docs"]
                        update_data["retrieved_count"] = len(docs)
                        update_data["docs"] = [{"title": d.get("title", ""), "description": d.get("description", "")[:100]} for d in docs[:3]]
                    
                    if node_name == "classify" and "action" in node_data:
                        update_data["action"] = node_data["action"]
                    
                    if node_name == "draft_response" and "messages" in node_data:
                        messages = node_data["messages"]
                        for msg in reversed(messages):
                            if msg.get("role") == "assistant" and not msg.get("content", "").startswith("Enhanced query:"):
                                update_data["answer"] = msg.get("content", "")
                                break
                    
                    yield f"data: {json.dumps(update_data)}\n\n"
                
                # Small delay to prevent overwhelming the client
                await asyncio.sleep(0.1)
            
            # Use accumulated state from stream (no need to invoke again)
            final_state = accumulated_state
            
            # Extract final answer
            messages = final_state.get("messages", [])
            final_answer = None
            for msg in reversed(messages):
                if msg.get("role") == "assistant" and not msg.get("content", "").startswith("Enhanced query:"):
                    final_answer = msg.get("content", "")
                    break
            
            if not final_answer:
                final_answer = "No response generated."
            
            # Format docs
            retrieved_docs = final_state.get("retrieved_docs", [])
            formatted_docs = [{"title": d.get("title", ""), "description": d.get("description", "")} for d in retrieved_docs]
            
            # Update session
            CHAT_SESSIONS[session_id] = messages
            
            # Send final result
            final_data = {
                "type": "complete",
                "answer": final_answer,
                "docs": formatted_docs,
                "query_used": final_state.get("query", request.query),
                "session_id": session_id
            }
            yield f"data: {json.dumps(final_data)}\n\n"
            
        except Exception as e:
            error_data = {"type": "error", "message": str(e)}
            yield f"data: {json.dumps(error_data)}\n\n"
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )

@app.post("/visit")
def visit():
    global VISITOR_COUNT
    VISITOR_COUNT += 1
    with open(VISITOR_FILE, "w") as f:
        f.write(str(VISITOR_COUNT))
    return {"count": VISITOR_COUNT}

@app.post("/search/keyword")
def keyword_search(request: SearchQuery):
    engine = get_engine()
    final_query = apply_query_enhancement(request.query, request.enhance_mode)
    results = engine._bm25_search(final_query, request.limit, mode=request.bm25_type)
    formatted = []
    for doc, score in results:
        formatted.append({
            "id": doc["id"],
            "title": doc["title"],
            "description": doc["description"],
            "score": float(score)
        })
    return {"results": formatted, "query_used": final_query}

@app.post("/search/semantic")
def semantic_search(request: SearchQuery):
    engine = get_engine()
    final_query = apply_query_enhancement(request.query, request.enhance_mode)
    
    # 1. Expand retrieval
    expanded_limit = request.limit * 5
    raw_results = engine._chunked_semantic_search(final_query, expanded_limit)
    
    # 2. Adapt to expected format for reranker: list of (doc_id, item_dict)
    # _chunked_semantic_search returns list of dicts: {'id':..., 'title':..., 'description':..., 'score':...}
    # We need to wrap it so 'doc' key exists
    adapted_results = []
    for item in raw_results:
        # We need the full doc object. 
        # Ideally _chunked_semantic_search should return it, but currently it returns a specific dict.
        # We can reconstruct a minimal doc object or fetch from docmap if available.
        # Let's trust the item has enough info for reranking (title + description)
        doc_id = item["id"]
        # Use docmap to get full doc if possible, else use item
        full_doc = engine.idx.docmap.get(doc_id) or item 
        
        adapted_results.append((doc_id, {"doc": full_doc, "semantic_score": item["score"]}))

    # 3. Rerank
    reranked_results = cross_encoder_rerank_results(adapted_results, final_query, request.limit)

    # 4. Format
    formatted = []
    for (doc_id, item), cross_score in reranked_results:
         formatted.append({
            "id": doc_id,
            "title": item["doc"]["title"],
            "description": item["doc"]["description"],
            "score": float(cross_score),
            "details": make_serializable({
                "original_semantic_score": item.get("semantic_score")
            })
         })

    return {"results": formatted, "query_used": final_query}

@app.post("/search/hybrid")
def hybrid_search_endpoint(request: HybridSearchQuery):
    engine = get_engine()
    final_query = apply_query_enhancement(request.query, request.enhance_mode)
    
    # 1. Expand retrieval
    expanded_limit = request.limit * 5
    
    if request.mode == "weighted":
        initial_results = engine.weighted_search(final_query, request.alpha, expanded_limit, bm25_mode=request.bm25_type)
    elif request.mode == "rrf":
        initial_results = engine.rrf_search(final_query, request.k, expanded_limit, bm25_mode=request.bm25_type)
    else:
        raise HTTPException(status_code=400, detail="Invalid hybrid mode")

    # 2. Rerank
    reranked_results = cross_encoder_rerank_results(initial_results, final_query, request.limit)
    
    # 3. Format
    formatted = []
    for (doc_id, item), cross_score in reranked_results:
         formatted.append({
            "id": doc_id,
            "title": item["doc"]["title"],
            "description": item["doc"]["description"],
            "score": float(cross_score), # The new reranked score
            "details": make_serializable({
                **{k: v for k, v in item.items() if k != "doc"},
                "original_score": item.get("hybrid_score") if request.mode == "weighted" else item.get("rrf_score")
            })
         })
    return {"results": formatted, "query_used": final_query}

@app.post("/rag")
def rag_endpoint(request: RAGQuery):
    engine = get_engine()
    final_query = apply_query_enhancement(request.query, request.enhance_mode)
    limit = request.limit
    
    # Unified retrieval pipeline for ALL RAG modes
    # 1. Retrieve more candidates via RRF
    raw_results = engine.rrf_search(final_query, k=60, limit=limit * 5, bm25_mode=request.bm25_type)
    
    # 2. Rerank using Cross-Encoder
    reranked = cross_encoder_rerank_results(raw_results, final_query, limit)
    
    # 3. Extract the top docs
    docs = [r[0][1]["doc"] for r in reranked]
    
    # Debug logging
    print(f"RAG Retrieved {len(docs)} docs for query '{final_query}':")
    for d in docs[:3]:
        print(f" - {d.get('title')}")

    if request.mode == "rag":
        answer = rag_answer(final_query, docs)
        return {"answer": answer, "docs": make_serializable(docs), "query_used": final_query}
    
    elif request.mode == "summarize":
        response = summarize(final_query, docs)
        return {"summary": response, "docs": make_serializable(docs), "query_used": final_query}
    elif request.mode == "citation":
        response = generate_citations(final_query, docs)
        return {"citations": response, "docs": make_serializable(docs), "query_used": final_query}
    elif request.mode == "question":
        response = question_answering(final_query, docs)
        return {"answer": response, "docs": make_serializable(docs), "query_used": final_query}
    
    raise HTTPException(status_code=400, detail="Invalid mode")