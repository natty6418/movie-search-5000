# RAG Search Engine 🎬

A comprehensive Retrieval-Augmented Generation (RAG) search engine for movies with a retro-themed web interface. This project demonstrates various search techniques ranging from traditional keyword search to modern agentic RAG systems.

## 🚀 Features

### Search Modes

*   **Keyword Search:** Uses BM25 (unigram, bigram, or combined) and TF-IDF algorithms for exact matching and term frequency analysis.
*   **Semantic Search:** Utilizes local embeddings (`sentence-transformers`) and vector search with chunking to find conceptually similar movies.
*   **Hybrid Search:** Combines keyword and semantic search results using Weighted Sum or Reciprocal Rank Fusion (RRF), followed by Cross-Encoder reranking for optimal retrieval.
*   **RAG Assistant:** Uses a local LLM (via Ollama) to answer questions, summarize movie details, generate citations, and perform question-answering based on search results.
*   **Agentic RAG:** Multi-step reasoning agent powered by LangGraph that:
    - Classifies queries to determine if database search is needed
    - Enhances queries for better search results
    - Searches the movie database intelligently
    - Analyzes results and decides if more searching is needed
    - Drafts comprehensive responses based on findings

### Additional Features

*   **Query Enhancement:** Automatically improves queries using LLM-based techniques (fix spelling, rewrite, expand).
*   **Cross-Encoder Reranking:** Uses `cross-encoder/ms-marco-TinyBERT-L2-v2` to rerank results for better relevance.
*   **Web Interface:** Retro-themed (90s style) React frontend with real-time streaming responses.
*   **REST API:** FastAPI backend with Server-Sent Events for streaming agent execution.
*   **Evaluation:** Tools to measure the precision and recall of the search results.

## 🛠️ Prerequisites

*   **Python:** 3.11 or higher
*   **Node.js:** 18 or higher (for the web frontend)
*   **Ollama:** For the local LLM integration
*   **Package Manager:** `uv` is recommended (lock file provided), but `pip` works too

## 📦 Installation

### Backend Setup

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd rag-search-engine
    ```

2.  **Install Python Dependencies:**
    Using `uv` (recommended):
    ```bash
    uv sync
    ```
    Or using `pip`:
    ```bash
    pip install .
    ```

3.  **Set up the Local LLM:**
    Install [Ollama](https://ollama.com/) and pull the required model:
    ```bash
    ollama pull qwen2.5:7b-instruct
    ```
    *Note: You can change the model or endpoint by setting `LOCAL_LLM_MODEL` and `LOCAL_LLM_URL` in a `.env` file.*

4.  **Prepare Data:**
    Ensure you have a `movies.json` file in the `data/` directory. The expected format is a JSON object with a `"movies"` key containing a list of movie objects.

    ```json
    {
      "movies": [
        {
          "id": "1",
          "title": "Movie Title",
          "description": "Movie description..."
        }
      ]
    }
    ```
    For evaluation, a `golden_dataset.json` in `data/` is also required.

### Frontend Setup

5.  **Install Node.js Dependencies:**
    ```bash
    cd web
    npm install
    ```

## 💻 Usage

Run the CLI tools from the project root using `python`.

### 1. Keyword Search
Traditional search using BM25.

```bash
# Build the index (optional, happens automatically on search if missing)
python cli/keyword_search_cli.py build

# Search for movies
python cli/keyword_search_cli.py search "sci-fi adventure"

# Inspect BM25 scores
python cli/keyword_search_cli.py bm25search "matrix"
```

### 2. Semantic Search
Search using vector embeddings.

```bash
# Verify the embedding model
python cli/semantic_search_cli.py verify_model

# Search using embeddings
python cli/semantic_search_cli.py search "movies about space travel"

# Search using chunked embeddings (for more granular results)
python cli/semantic_search_cli.py chunk_search "time travel paradox"
```

### 3. Hybrid Search
Combines Keyword and Semantic search.

```bash
# Weighted Search
python cli/hybrid_search_cli.py weighted_search "funny action movies"

# Reciprocal Rank Fusion (RRF)
python cli/hybrid_search_cli.py rrf-search "scary bear movie"
```

### 4. RAG (Augmented Generation)
Ask questions and get answers based on the movie dataset.

```bash
# General RAG QA
python cli/augmented_generation_cli.py rag "What are some good 90s action movies?"

# Summarize findings
python cli/augmented_generation_cli.py summarize "movies with plot twists"

# Get answers with citations
python cli/augmented_generation_cli.py citation "Who directed Inception?"
```

## 📂 Project Structure

*   `cli/`: Command-line interface scripts.
    *   `keyword_search_cli.py`: BM25/TF-IDF implementations.
    *   `semantic_search_cli.py`: Vector embedding generation and search.
    *   `hybrid_search_cli.py`: Fusion logic for hybrid search.
    *   `augmented_generation_cli.py`: RAG pipeline (QA, Summarization).
    *   `evaluation_cli.py`: Metrics and evaluation tools.
*   `cli/lib/`: Core libraries and utilities.
    *   `inverted_index.py`: Logic for keyword indexing.
    *   `local_llm.py`: Interface for the local LLM (Ollama).
    *   `semantic_search.py`: SentenceTransformer logic.
*   `data/`: Directory for `movies.json` dataset.
*   `cache/`: Stores generated embeddings and indices to speed up subsequent runs.

## ⚙️ Configuration

Create a `.env` file to customize the setup:

```env
LOCAL_LLM_MODEL=qwen2.5:7b-instruct
LOCAL_LLM_URL=http://localhost:11434
```
