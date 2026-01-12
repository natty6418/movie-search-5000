# RAG Search Engine - Comprehensive Study Notes

## Table of Contents
1. [Project Overview](#project-overview)
2. [Architecture](#architecture)
3. [Search Techniques](#search-techniques)
4. [Core Components](#core-components)
5. [Frontend Features](#frontend-features)
6. [Key Concepts](#key-concepts)
7. [Configuration](#configuration)

---

## Project Overview

A full-stack Retrieval-Augmented Generation (RAG) search engine demonstrating the evolution from traditional keyword search to modern AI-powered semantic search and generation.

**Tech Stack:**
- **Backend**: Python, FastAPI, Uvicorn
- **Frontend**: React, Vite, TailwindCSS
- **ML/AI**: sentence-transformers, Gemini API, Ollama
- **Graph Framework**: LangGraph

---

## Architecture

### High-Level Architecture

```
┌─────────────┐
│   Frontend  │ (React/Vite)
│  (Port 5173)│
└──────┬──────┘
       │ HTTP/REST
       ▼
┌─────────────┐
│   FastAPI   │ (Python)
│  (Port 8000)│
└──────┬──────┘
       │
       ├──► Inverted Index (BM25)
       ├──► Semantic Search (Embeddings)
       ├──► Hybrid Search (RRF)
       ├──► Ollama (Local LLM)
       └──► Gemini (Agent LLM)
```

### Backend Structure

```
api/
  main.py              # FastAPI endpoints
cli/
  lib/
    inverted_index.py  # BM25 keyword search
    semantic_search.py # Vector embeddings
    hybrid_search.py   # Fusion strategies
    local_llm.py       # Ollama interface
    agent.py           # LangGraph agent
  keyword_search_cli.py
  semantic_search_cli.py
  hybrid_search_cli.py
  augmented_generation_cli.py
data/
  movies.json          # Dataset
  stopwords.txt        # Stopword list
cache/
  index.pkl            # Cached BM25 index
  embeddings.npy       # Cached embeddings
```

---

## Search Techniques

### 1. Keyword Search (BM25)

**What it is:**
- Traditional information retrieval using the BM25 (Best Match 25) algorithm
- Scores documents based on term frequency and inverse document frequency

**How it works:**

```python
# BM25 Formula (simplified)
score = Σ(IDF(qi) × (f(qi,D) × (k1 + 1)) / (f(qi,D) + k1 × (1 - b + b × |D|/avgdl)))

Where:
- qi = query term i
- f(qi,D) = frequency of qi in document D
- |D| = document length
- avgdl = average document length
- k1 = 1.5 (term frequency saturation)
- b = 0.75 (length normalization)
```

**Implementation Details:**

```python
# inverted_index.py

class InvertedIndex:
    def __init__(self):
        self.index = {}        # word -> set of doc IDs
        self.docmap = {}       # doc ID -> document
        self.term_frequencies = {}  # doc ID -> term counts
        self.doc_lengths = {}  # doc ID -> length
```

**Three Modes:**

1. **Unigram (Standard)**: Matches individual words
   - Query: "action movies" → matches docs with "action" OR "movies"

2. **Bigram (Phrases)**: Matches word pairs
   - Query: "action movies" → matches "action_movies" as phrase

3. **Combined (Auto)**: Merges both with bigram boost
   ```python
   # Bigram matches weighted 1.5x higher
   if doc_id in combined:
       combined[doc_id]["score"] += score * 1.5
   ```

**Preprocessing Pipeline:**
1. Lowercase text
2. Remove punctuation
3. Tokenize (split into words)
4. Remove stopwords ("the", "a", "is", etc.)
5. Stem words (Porter Stemmer: "running" → "run")

---

### 2. Semantic Search (Vector Embeddings)

**What it is:**
- Converts text into dense numerical vectors (embeddings)
- Finds similar documents using cosine similarity in vector space

**How it works:**

```
Text → Embedding Model → Vector [768 dimensions]

"action movies" → [0.23, -0.15, 0.87, ..., 0.42]
"thriller films" → [0.21, -0.18, 0.84, ..., 0.39]  ← Similar!
"romantic comedy" → [-0.62, 0.73, -0.21, ..., 0.18] ← Different
```

**Model Used:**
- **all-MiniLM-L6-v2** from sentence-transformers
- Fast, lightweight (80MB)
- 384-dimensional embeddings
- Trained on 1B+ sentence pairs

**Chunked Search:**
- Splits long descriptions into smaller chunks
- Embeds each chunk separately
- Better granularity for matching specific concepts

```python
# semantic_search.py

def chunk_text(text, max_chunk_size=200):
    """Split text into ~200 word chunks"""
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = []
    current_length = 0

    for sentence in sentences:
        words = sentence.split()
        if current_length + len(words) > max_chunk_size:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentence]
            current_length = len(words)
        else:
            current_chunk.append(sentence)
            current_length += len(words)

    return chunks
```

**Similarity Calculation:**

```python
# Cosine similarity
similarity = dot(query_embedding, doc_embedding) / (norm(query) * norm(doc))

# Returns value between -1 (opposite) and 1 (identical)
```

---

### 3. Hybrid Search (RRF - Reciprocal Rank Fusion)

**What it is:**
- Combines keyword and semantic search results
- Uses ranking-based fusion (not score-based)

**Why Hybrid?**
- Keyword search: Good at exact matches, entities, specific terms
- Semantic search: Good at concepts, synonyms, paraphrases
- Combined: Best of both worlds!

**RRF Algorithm:**

```python
# For each document in either result set:
RRF_score = Σ(1 / (k + rank_i))

Where:
- k = 60 (constant, controls influence of lower-ranked results)
- rank_i = position in result list i (1, 2, 3, ...)
```

**Example:**

```
Document: "The Matrix"

Keyword results:     Semantic results:
1. Inception         1. The Matrix    ← rank 1
2. Interstellar      2. Blade Runner
3. The Matrix ←rank 3 3. Minority Report

RRF score = 1/(60+3) + 1/(60+1) = 0.0159 + 0.0164 = 0.0323
```

**Alternative: Weighted Search**
- Directly combines normalized scores
- `final_score = (1 - α) * keyword_score + α * semantic_score`
- α controls balance (not currently exposed in UI)

---

### 4. Cross-Encoder Reranking

**What it is:**
- Second-stage ranking using a more powerful model
- Processes query + document together (not separately)

**Pipeline:**

```
Initial Search (RRF)
  ↓ Top 50 candidates
Cross-Encoder Reranking
  ↓ Top 10 results
Return to User
```

**How it differs from Bi-Encoder:**

| Bi-Encoder (Semantic Search) | Cross-Encoder (Reranking) |
|------------------------------|---------------------------|
| Encodes query & docs separately | Encodes query + doc together |
| Fast (pre-compute doc vectors) | Slower (must process pairs) |
| Good for retrieval | Better for scoring |
| Cosine similarity | Classification score |

**Model Used:**
- **ms-marco-MiniLM-L-6-v2** (cross-encoder)
- Trained on MS MARCO passage ranking dataset
- Outputs relevance score 0-1

```python
# hybrid_search.py

def cross_encoder_rerank_results(results, query, limit):
    model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

    # Create query-document pairs
    pairs = [(query, item[1]["doc"]["title"] + " " +
              item[1]["doc"]["description"]) for item in results]

    # Get relevance scores
    scores = model.predict(pairs)

    # Re-sort by cross-encoder score
    reranked = sorted(zip(results, scores),
                      key=lambda x: x[1], reverse=True)

    return reranked[:limit]
```

---

### 5. RAG (Retrieval-Augmented Generation)

**What it is:**
- Combines retrieval (search) with generation (LLM)
- Provides context to LLM from search results
- Reduces hallucination, grounds answers in data

**RAG Pipeline:**

```
User Query
  ↓
Query Enhancement (optional)
  ↓
Hybrid Search (RRF) → Top 25 candidates
  ↓
Cross-Encoder Reranking → Top 5 documents
  ↓
Format as Context
  ↓
LLM Generation (Ollama: qwen2.5:7b-instruct)
  ↓
Return Answer
```

**Query Enhancement Techniques:**

1. **Fix Spelling**: Correct typos
   ```
   "acton movees" → "action movies"
   ```

2. **Rewrite**: Rephrase for better search
   ```
   "scary films" → "horror thriller suspense movies"
   ```

3. **Expand**: Add related terms
   ```
   "space movies" → "space movies science fiction astronaut"
   ```

**LLM Used: Ollama (Local)**
- **Model**: qwen2.5:7b-instruct
- **Why local**: Free, private, no API limits
- **API**: OpenAI-compatible endpoint

**Four RAG Modes:**

1. **Q&A (rag)**: Direct question answering
   ```python
   system = "Answer based solely on the provided movie information."
   prompt = f"Question: {query}\n\nMovies:\n{context}\n\nAnswer:"
   ```

2. **Summarize**: Aggregate information
   ```python
   system = "Summarize the key themes across these movies."
   ```

3. **Citation**: Include source references
   ```python
   system = "Answer and cite specific movies as sources."
   ```

4. **Question (chat)**: Conversational style
   ```python
   system = "You're a friendly movie expert. Provide recommendations."
   ```

---

### 6. Agent Mode (LangGraph + Gemini)

**What it is:**
- Autonomous agent that can use tools (search)
- Decides when to search vs. answer directly
- Multi-step reasoning

**Architecture:**

```
┌─────────────┐
│ User Query  │
└──────┬──────┘
       ▼
┌─────────────────┐
│  Agent (Brain)  │  ← Gemini LLM
│  - Thinks       │
│  - Decides      │
└────────┬────────┘
         │
    ┌────┴─────┐
    │   Need   │
    │  Info?   │
    └────┬─────┘
         │
    Yes  │  No
    ↓    │   ↓
┌────────┴───┐  ┌──────────┐
│ Tool Exec  │  │  Answer  │
│ (Search)   │  │   User   │
└─────┬──────┘  └──────────┘
      │
      └─────► Loop back to Agent
```

**LangGraph State Machine:**

```python
class AgentState(TypedDict):
    messages: List[dict]  # Conversation history

workflow = StateGraph(AgentState)
workflow.add_node("agent", agent.call_llm)      # Think
workflow.add_node("tools", agent.execute_tools) # Act
workflow.add_conditional_edges("agent", should_continue)
workflow.add_edge("tools", "agent")  # Loop back after tool use
```

**Tool: movie_search**

```python
{
    "name": "movie_search",
    "description": "Search for movies based on plot, genre, director...",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string"}
        }
    }
}
```

**Function Calling Flow:**

1. User: "Compare Inception and Interstellar"
2. Agent thinks: "I need to search for both movies"
3. Agent calls: `movie_search("Inception")`
4. System executes search, returns results
5. Agent receives results as context
6. Agent calls: `movie_search("Interstellar")`
7. System executes search
8. Agent synthesizes comparison from both results
9. Returns answer to user

**Gemini API Details:**
- **Model**: gemini-2.5-flash-lite
- **Why Gemini**: Native function calling support, generous free tier
- **Chat API**: Maintains conversation context properly

**Key Implementation Detail:**

When replaying function calls, we must:
1. Send user message (Gemini responds with function call)
2. Execute function
3. Send function response back

```python
# agent.py - Handling tool responses

if pending_tool_responses:
    # First, replay user message so Gemini makes function call
    chat.send_message(user_msg)

    # Then send function responses
    func_response_parts = [
        Part(function_response=FunctionResponse(
            name=tool_msg["name"],
            response={"result": tool_msg["content"]}
        ))
    ]
    response = chat.send_message(func_response_parts)
```

---

## Core Components

### InvertedIndex (`cli/lib/inverted_index.py`)

**Purpose**: Keyword-based search using BM25

**Key Methods:**

```python
def build():
    """Build index from movies.json"""
    # For each movie:
    #   1. Tokenize title + description
    #   2. Add each token to inverted index
    #   3. Track term frequencies and document lengths

def bm25_search(query, limit):
    """Search using BM25 algorithm"""
    # 1. Tokenize query
    # 2. Calculate IDF for each term
    # 3. Score each document
    # 4. Return top K results

def save() / load():
    """Cache index to disk (pickle)"""
```

**Subclass: BigramInvertedIndex**
- Overrides `tokenize_strategy()` to create bigrams
- Example: "action movies" → ["action_movies"]

---

### ChunkedSemanticSearch (`cli/lib/semantic_search.py`)

**Purpose**: Vector-based semantic search

**Key Methods:**

```python
def load_or_create_chunk_embeddings(documents):
    """Generate embeddings for all movie chunks"""
    # 1. Check cache (chunk_embeddings.npy)
    # 2. If missing:
    #    - Chunk each movie description
    #    - Generate embeddings using sentence-transformers
    #    - Save to cache

def search(query, limit):
    """Find semantically similar movies"""
    # 1. Embed query
    # 2. Compute cosine similarity with all chunks
    # 3. Aggregate chunk scores by movie
    # 4. Return top K movies
```

**Caching Strategy:**
- Embeddings: `cache/chunk_embeddings.npy` (NumPy array)
- Chunk map: `cache/chunk_map.json` (chunk → movie ID)
- Rebuilt when movies.json changes

---

### HybridSearch (`cli/lib/hybrid_search.py`)

**Purpose**: Combines keyword + semantic search

**Key Methods:**

```python
def rrf_search(query, k=60, limit=10):
    """Reciprocal Rank Fusion"""
    # 1. Get BM25 results
    # 2. Get semantic results
    # 3. Compute RRF scores
    # 4. Return merged, sorted results

def weighted_search(query, alpha=0.5, limit=10):
    """Weighted score fusion"""
    # 1. Normalize BM25 scores to [0,1]
    # 2. Normalize semantic scores to [0,1]
    # 3. Combine: (1-α)*keyword + α*semantic
    # 4. Return top K

def cross_encoder_rerank_results(results, query, limit):
    """Rerank using cross-encoder"""
    # Implementation shown above
```

---

### LocalLLM (`cli/lib/local_llm.py`)

**Purpose**: Interface to Ollama for RAG

**Key Functions:**

```python
def _call_local_llm(prompt, system_prompt, json_mode):
    """Call Ollama via OpenAI-compatible API"""
    client = OpenAI(base_url="http://localhost:11434/v1")
    response = client.chat.completions.create(
        model="qwen2.5:7b-instruct",
        messages=[...],
        response_format={"type": "json_object"} if json_mode else None
    )
    return response.choices[0].message.content

def rag_answer(query, docs):
    """Generate answer from retrieved documents"""
    context = format_docs(docs)
    prompt = f"Question: {query}\n\nContext:\n{context}"
    return _call_local_llm(prompt, system="Answer based on context")

def enhance_query(query):
    """Fix spelling errors"""

def rewrite_query(query):
    """Rephrase for better search"""

def expand_query(query):
    """Add related terms"""
```

---

### MovieAgent (`cli/lib/agent.py`)

**Purpose**: LangGraph agent with function calling

**Key Components:**

```python
class MovieAgent:
    def call_llm(state):
        """Brain - decides to search or answer"""
        # Use Gemini chat API
        # Handle function calls/responses
        # Return decision

    def execute_tools(state):
        """Action - run movie_search tool"""
        # Parse function arguments
        # Execute hybrid search
        # Format results as text
        # Return tool response

    def should_continue(state):
        """Router - continue to tools or end?"""
        if last_message.has_tool_calls():
            return "tools"
        return END

def build_graph(engine):
    """Construct LangGraph workflow"""
    workflow = StateGraph(AgentState)
    workflow.add_node("agent", agent.call_llm)
    workflow.add_node("tools", agent.execute_tools)
    workflow.set_entry_point("agent")
    workflow.add_conditional_edges("agent", agent.should_continue)
    workflow.add_edge("tools", "agent")
    return workflow.compile()
```

---

## Frontend Features

### Search Modes

**Implemented in `web/src/App.jsx`:**

1. **Keyword Search** (BM25)
   - Standard text matching
   - Options: Unigram/Bigram/Combined

2. **Semantic AI** (Embeddings)
   - Vector similarity search
   - Understands meaning

3. **Hybrid Fusion** (RRF)
   - Best of both worlds
   - Fixed k=60 parameter

4. **RAG Assistant** (Ollama)
   - LLM-powered answers
   - Modes: Q&A, Summarize, Citations, Chat

5. **Auto Agent** (Gemini + LangGraph)
   - Autonomous reasoning
   - Can search when needed

### Query Enhancement

**Three modes:**
- **None**: Use query as-is
- **Fix Spelling**: Correct typos via LLM
- **Rewrite**: Rephrase for better search
- **Expand**: Add related terms

### Keyword Options

**BM25 modes:**
- **Auto (combined)**: Unigram + Bigram (bigram 1.5x boost)
- **Standard (unigram)**: Individual words
- **Phrases (bigram)**: Word pairs

---

## Key Concepts

### 1. BM25 (Best Match 25)

**Intuition:**
- Rewards documents that mention query terms frequently
- Penalizes documents that are too long
- Accounts for how common/rare each term is

**Saturation:**
- After ~5 occurrences, more mentions matter less
- Prevents keyword stuffing from dominating

### 2. TF-IDF (Term Frequency - Inverse Document Frequency)

```
TF(term, doc) = count(term in doc) / total words in doc
IDF(term) = log(total docs / docs containing term)
TF-IDF = TF × IDF
```

**Example:**
- "the" appears in every doc → low IDF → low score
- "cyberpunk" appears in few docs → high IDF → high score

### 3. Cosine Similarity

**Measures angle between vectors:**

```
cos(θ) = A·B / (|A| × |B|)

Where:
- A·B = dot product
- |A| = magnitude of A
- Returns: -1 (opposite) to 1 (identical)
```

**Why cosine?**
- Ignores vector magnitude (length)
- Focuses on direction (semantic content)
- Efficient to compute

### 4. Embeddings

**Dense vector representations:**
- Traditional: Sparse vectors (size = vocabulary)
  - "action movie" → [0, 0, 1, 0, ..., 1, 0, 0] (2 positions = 1)
- Modern: Dense vectors (size = 384)
  - "action movie" → [0.23, -0.15, 0.87, ..., 0.42] (all positions have values)

**Benefits:**
- Capture semantic meaning
- Similar concepts close in vector space
- Handles synonyms naturally

### 5. Cross-Encoder vs Bi-Encoder

**Bi-Encoder (Two-Tower):**
```
Query → Encoder₁ → Vector_q ─┐
                              ├─→ Similarity
Doc → Encoder₂ → Vector_d ────┘
```
- Independent encoding
- Can pre-compute doc vectors
- Fast at scale

**Cross-Encoder (Single-Tower):**
```
Query + Doc → Joint Encoder → Relevance Score
```
- Attends to interaction
- Must compute for each pair
- Slower but more accurate

### 6. Reciprocal Rank Fusion (RRF)

**Why RRF over score fusion?**
- Scores from different systems aren't comparable
- BM25 score of 5.2 vs semantic score of 0.87 - which is better?
- Ranks are universal: 1st place is 1st place

**Formula:**
```
RRF(d) = Σ(1 / (k + rank_i(d)))
```

**Properties:**
- k=60 is standard (from research)
- Lower ranks contribute exponentially less
- Naturally handles missing documents (rank = ∞)

### 7. Retrieval-Augmented Generation (RAG)

**Problem RAG Solves:**
- LLMs hallucinate (make up facts)
- Knowledge cutoff date
- No access to private data

**RAG Solution:**
1. Retrieve relevant information (search)
2. Inject as context in prompt
3. LLM generates answer based on context

**Benefits:**
- Grounded in facts
- Citable sources
- Up-to-date information
- Domain-specific knowledge

### 8. LangGraph State Machines

**Why graphs for agents?**
- Agents need to loop (think → act → think)
- Complex control flow (conditionals, cycles)
- State management

**LangGraph Structure:**
```python
StateGraph(State) {
    nodes: {
        "agent": call_llm,
        "tools": execute_tools
    },
    edges: {
        "agent" → conditional → "tools" or END
        "tools" → "agent"  # Loop back
    }
}
```

---

## Configuration

### Environment Variables (`.env`)

```bash
# Ollama (for RAG endpoints)
LOCAL_LLM_MODEL=qwen2.5:7b-instruct
LOCAL_LLM_URL=http://localhost:11434

# Gemini (for Agent endpoint)
GEMINI_API_KEY=your_api_key_here
GEMINI_MODEL=gemini-2.5-flash-lite

# HuggingFace (not currently used)
HF_TOKEN=your_token_here
```

### API Endpoints (`api/main.py`)

```
POST /search/keyword
  { query, limit, enhance_mode, bm25_type }
  → { results, query_used }

POST /search/semantic
  { query, limit, enhance_mode }
  → { results, query_used }

POST /search/hybrid
  { query, mode, alpha, k, limit, enhance_mode, bm25_type }
  → { results, query_used }

POST /rag
  { query, mode, limit, enhance_mode, bm25_type }
  → { answer, docs, query_used }

POST /agent
  { query, chat_history }
  → { answer, messages }

POST /visit
  → { count }
```

### CLI Commands

```bash
# Keyword search
python cli/keyword_search_cli.py build
python cli/keyword_search_cli.py search "action movies"

# Semantic search
python cli/semantic_search_cli.py verify_model
python cli/semantic_search_cli.py search "space exploration"
python cli/semantic_search_cli.py chunk_search "time travel"

# Hybrid search
python cli/hybrid_search_cli.py weighted_search "thriller"
python cli/hybrid_search_cli.py rrf-search "comedy"

# RAG
python cli/augmented_generation_cli.py rag "Best 90s movies?"
python cli/augmented_generation_cli.py summarize "sci-fi"
python cli/augmented_generation_cli.py citation "Who directed Inception?"

# Evaluation
python cli/evaluation_cli.py evaluate
```

---

## Performance Optimizations

### Caching Strategy

1. **BM25 Index** (`cache/index.pkl`, `cache/bigram_index.pkl`)
   - Pre-computed inverted indexes
   - Loaded on startup
   - Rebuilt when movies.json changes

2. **Embeddings** (`cache/chunk_embeddings.npy`)
   - Pre-computed vectors for all chunks
   - ~400MB for full dataset
   - Loaded into memory on startup

3. **Cross-Encoder**
   - Model cached by HuggingFace (~90MB)
   - Only run on top-K candidates (50 → 10)

### Search Pipeline Optimization

```
User Query
  ↓
Retrieve 5x candidates (cheap)
  ↓
Rerank with cross-encoder (expensive, but fewer items)
  ↓
Return top K
```

**Why this works:**
- Initial search is fast (BM25/cosine)
- Cross-encoder only on promising candidates
- 80% of compute on 20% of data

---

## Common Pitfalls & Solutions

### Issue: Different models for different features

**Problem:** Agent uses Gemini, RAG uses Ollama, confusion over which model is used where.

**Solution:** Clear naming in `.env`:
```bash
GEMINI_MODEL=...      # For Agent only
LOCAL_LLM_MODEL=...   # For RAG only
```

### Issue: Chat history breaks function calling

**Problem:** Gemini expects function responses immediately after function calls.

**Solution:** When replaying history, send user message first, then function responses:
```python
chat.send_message(user_msg)        # Gemini makes function call
chat.send_message(func_responses)   # Now send our results
```

### Issue: Scores from different systems aren't comparable

**Problem:** Can't add BM25 score (0-100) to semantic score (0-1).

**Solution:** Use RRF (rank-based fusion) instead of score fusion.

### Issue: Slow search with large datasets

**Problem:** Computing cosine similarity with every document is O(n).

**Solution:**
1. Use caching (pre-compute embeddings)
2. Use approximate nearest neighbors (not implemented yet)
3. Use two-stage retrieval (implemented: retrieve → rerank)

---

## Future Enhancements

### Not Yet Implemented

1. **Weighted Hybrid Mode**
   - Alpha slider in UI
   - User-controlled keyword/semantic balance

2. **Conversation History for Agent**
   - Multi-turn conversations
   - Context persistence across requests

3. **Approximate Nearest Neighbors (ANN)**
   - FAISS/Annoy for faster semantic search
   - Scales to millions of documents

4. **Evaluation Metrics**
   - Precision@K, Recall@K, nDCG
   - A/B testing different pipelines

5. **Query Understanding**
   - Intent classification
   - Entity extraction
   - Query expansion with knowledge graphs

6. **Multimodal Search**
   - Image embeddings (movie posters)
   - Vision + text fusion

7. **Streaming Responses**
   - Real-time LLM generation
   - Server-Sent Events (SSE)

---

## References & Further Reading

### Papers
- **BM25**: Robertson & Zaragoza (2009) - "The Probabilistic Relevance Framework: BM25 and Beyond"
- **BERT**: Devlin et al. (2018) - "BERT: Pre-training of Deep Bidirectional Transformers"
- **Sentence-BERT**: Reimers & Gurevych (2019) - "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks"
- **RAG**: Lewis et al. (2020) - "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"
- **RRF**: Cormack et al. (2009) - "Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods"

### Libraries
- [sentence-transformers](https://www.sbert.net/)
- [LangGraph](https://langchain-ai.github.io/langgraph/)
- [FastAPI](https://fastapi.tiangolo.com/)
- [Ollama](https://ollama.ai/)

### Datasets
- **MS MARCO**: Microsoft Machine Reading Comprehension
- **BEIR**: Benchmark for Information Retrieval

---

## Glossary

- **BM25**: Best Match 25, a ranking function for keyword search
- **Bi-encoder**: Separate encoders for query and document
- **Chunking**: Splitting long documents into smaller pieces
- **Cosine Similarity**: Measure of vector angle (semantic similarity)
- **Cross-encoder**: Joint encoder for query + document pairs
- **Embedding**: Dense vector representation of text
- **IDF**: Inverse Document Frequency (rarity of term)
- **LangGraph**: Framework for agent state machines
- **RAG**: Retrieval-Augmented Generation
- **RRF**: Reciprocal Rank Fusion (rank-based merging)
- **Stemming**: Reducing words to root form (running → run)
- **TF-IDF**: Term Frequency × Inverse Document Frequency
- **Tokenization**: Splitting text into words/tokens

---

## Summary

This project demonstrates the evolution of search technology:

1. **Traditional IR** (BM25) - Fast, exact matching, good baselines
2. **Neural Search** (Embeddings) - Semantic understanding, handles synonyms
3. **Hybrid** (RRF) - Best of both approaches
4. **RAG** (Retrieval + LLM) - Natural language answers with sources
5. **Agents** (LangGraph) - Autonomous reasoning and tool use

Each technique builds on the previous, showing how modern AI-powered search systems work under the hood.

**Key Takeaway:** There's no single "best" search method. The optimal approach depends on your use case:
- **Keyword**: Known-item search, entity names
- **Semantic**: Conceptual queries, exploratory search
- **Hybrid**: General-purpose, production systems
- **RAG**: Question answering, summarization
- **Agents**: Complex, multi-step information needs
