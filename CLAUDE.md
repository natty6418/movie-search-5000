# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A Retrieval-Augmented Generation (RAG) search engine for movies demonstrating multiple search techniques: traditional keyword search (BM25/TF-IDF), semantic vector search, hybrid search (fusion of both), and LLM-powered augmented generation with agent capabilities. The project consists of a FastAPI backend, React/Vite frontend, and CLI tools.

## Architecture

### Backend Structure

The backend uses a modular architecture with three layers:

1. **API Layer** (`api/main.py`): FastAPI application serving REST endpoints
2. **Core Library** (`cli/lib/`): Reusable search and LLM components
3. **CLI Tools** (`cli/`): Command-line interfaces for development and testing

### Core Components

**InvertedIndex** (`cli/lib/inverted_index.py`):
- Base class for keyword-based search using BM25 algorithm
- Implements tokenization with stemming and stopword removal
- Maintains inverted index, term frequencies, and document lengths
- Supports both unigram and bigram indexes (via `BigramInvertedIndex` subclass)
- Caches built indexes to `cache/*.pkl` for fast loading

**SemanticSearch** (`cli/lib/semantic_search.py`):
- Uses `sentence-transformers` for embedding generation
- Default model: `all-MiniLM-L6-v2` (local, fast)
- Implements chunked search: splits movie descriptions into chunks for better granularity
- Embeddings cached in `cache/` directory

**HybridSearch** (`cli/lib/hybrid_search.py`):
- Combines BM25 and semantic search results
- Two fusion strategies:
  - **Weighted Sum**: Linear combination with alpha parameter
  - **Reciprocal Rank Fusion (RRF)**: Rank-based fusion with k parameter
- Includes cross-encoder reranking (`cross_encoder_rerank_results`) for improved relevance
- Manages both unigram and bigram indexes

**LocalLLM** (`cli/lib/local_llm.py`):
- Interface to Ollama for local LLM inference
- Default model: `qwen2.5:7b-instruct`
- Functions: RAG answering, summarization, citation generation, query enhancement/expansion

**Agent** (`cli/lib/agent.py`):
- LangGraph-based agentic workflow using HuggingFace endpoints
- Default model: `meta-llama/Llama-4-Scout-17B-16E-Instruct`
- Tool-calling agent that can search movies and synthesize multi-step responses
- Uses stateful graph with conditional routing

### Frontend Structure

React + Vite SPA with retro 90s web aesthetic:
- `web/src/App.jsx`: Main application with mode tabs (keyword, semantic, hybrid, RAG, agent)
- `web/src/api.js`: API client functions
- `web/src/components/`: Reusable UI components

## Development Commands

### Backend

```bash
# Install dependencies (uses uv package manager)
uv sync

# Alternative: pip install
pip install .

# Run API server
uvicorn api.main:app --reload --port 8000

# The API will be available at http://localhost:8000
```

### Frontend

```bash
cd web

# Install dependencies
npm install

# Run dev server
npm run dev

# Build for production
npm run build

# Lint code
npm run lint
```

### CLI Tools

All CLI tools run from project root:

```bash
# Keyword search
python cli/keyword_search_cli.py build           # Build BM25 index
python cli/keyword_search_cli.py search "query"  # Search movies
python cli/keyword_search_cli.py bm25search "query"  # Show BM25 scores

# Semantic search
python cli/semantic_search_cli.py verify_model   # Test embedding model
python cli/semantic_search_cli.py search "query"  # Vector search
python cli/semantic_search_cli.py chunk_search "query"  # Chunked search

# Hybrid search
python cli/hybrid_search_cli.py weighted_search "query"  # Weighted fusion
python cli/hybrid_search_cli.py rrf-search "query"  # RRF fusion

# RAG/LLM
python cli/augmented_generation_cli.py rag "question"  # QA
python cli/augmented_generation_cli.py summarize "query"  # Summarize
python cli/augmented_generation_cli.py citation "query"  # Cited answers

# Evaluation
python cli/evaluation_cli.py evaluate  # Run metrics on golden dataset
```

## Configuration

Create `.env` file in project root:

```env
# Ollama LLM (for RAG endpoints)
LOCAL_LLM_MODEL=qwen2.5:7b-instruct
LOCAL_LLM_URL=http://localhost:11434

# HuggingFace (for Agent endpoints)
HF_TOKEN=your_token_here
LLM_MODEL=meta-llama/Llama-4-Scout-17B-16E-Instruct
```

## Data Requirements

- `data/movies.json`: Main dataset with structure `{"movies": [{"id": "...", "title": "...", "description": "..."}]}`
- `data/stopwords.txt`: Stopword list for tokenization (already included)
- `data/golden_dataset.json`: Test queries for evaluation (optional)

## Search Pipeline Architecture

The production search pipeline (used by API endpoints) follows this pattern:

1. **Query Enhancement** (optional): Spelling correction, rewriting, or expansion via LLM
2. **Retrieval**: Fetch 5x more candidates than needed using chosen method:
   - Keyword: BM25 (unigram/bigram/combined)
   - Semantic: Chunked vector search
   - Hybrid: RRF or weighted fusion
3. **Reranking**: Cross-encoder reranking to final limit
4. **Generation** (RAG only): LLM synthesizes answer from top results

This pipeline is implemented in:
- `/search/keyword`, `/search/semantic`, `/search/hybrid` endpoints for search-only
- `/rag` endpoint for full RAG with answer generation
- `/agent` endpoint for agentic multi-step reasoning

## Cache Management

The system caches several artifacts in `cache/`:
- `index.pkl` / `bigram_index.pkl`: BM25 indexes
- `embeddings.npy` / `chunk_embeddings.npy`: Precomputed embeddings
- `chunk_map.json`: Mapping from chunks to documents

Rebuild caches after changing `movies.json` by deleting cache files or using CLI build commands.

## Python Path Considerations

All CLI scripts assume execution from project root. API adds parent directory to `sys.path` to import `cli.lib` modules. When adding new modules, maintain this convention.

## Dependencies

Key dependencies (see `pyproject.toml` and `web/package.json`):
- **Backend**: fastapi, uvicorn, nltk, sentence-transformers, numpy, python-dotenv, google-genai, langgraph
- **Frontend**: react, vite, axios, tailwindcss, react-markdown, lucide-react

## Testing

Use `cli/evaluation_cli.py` with `golden_dataset.json` to measure precision/recall. The evaluation framework expects specific query-result pairs.

## Prerequisites

- Python 3.11+
- Ollama installed locally with `qwen2.5:7b-instruct` model pulled
- Node.js for frontend development
- HuggingFace token for agent functionality
