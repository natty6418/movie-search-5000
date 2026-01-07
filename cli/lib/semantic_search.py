from sentence_transformers import SentenceTransformer
import numpy as np
import os
import json
import re

os.makedirs(os.path.join(os.path.dirname(__file__), "../../cache"), exist_ok=True)
embedding_path = os.path.join(
    os.path.dirname(__file__), "../../cache/movie_embeddings.npy"
)
chunk_embedding_path = os.path.join(
    os.path.dirname(__file__), "../../cache/chunk_embeddings.npy"
)
chunk_metadata_path = os.path.join(
    os.path.dirname(__file__), "../../cache/chunk_metadata.json"
)
movies_path = os.path.join(os.path.dirname(__file__), "../../data/movies.json")
MODEL_NAME = "all-MiniLM-L6-v2"


class SemanticSearch:
    def __init__(self, model_name=MODEL_NAME) -> None:
        self.model = SentenceTransformer(model_name)
        self.docs = None
        self.doc_map = {}
        self.embeddings = None

    def generate_embeddign(self, text: str):
        """
        Generate embedding for the given text using the SentenceTransformer model.

        Args:
        text (str): The input text to generate the embedding for.

        Returns:
        embedding (list): The generated embedding as a list of floats.
        """
        if len(text.strip()) == 0:
            raise ValueError("Input text cannot be empty")

        embedding = self.model.encode([text])
        return embedding[0]

    def build_embeddings(self, documents: list[dict[str, str]]):
        self.documents = documents
        docs = []
        for doc in documents:
            self.doc_map[doc["id"]] = doc
            docs.append(doc["title"] + " " + doc["description"])
        self.embeddings = self.model.encode(docs, show_progress_bar=True)
        if self.embeddings is not None:
            np.save(
                embedding_path,
                self.embeddings,
            )
        return self.embeddings

    def load_or_create_embeddigns(self, documents: list[dict[str, str]]):

        self.documents = documents
        docs = []
        for doc in documents:
            self.doc_map[doc["id"]] = doc
            docs.append(doc["title"] + " " + doc["description"])

        if os.path.exists(embedding_path):
            self.embeddings = np.load(embedding_path)
        else:
            self.embeddings = self.model.encode(docs, show_progress_bar=True)
            np.save(
                embedding_path,
                self.embeddings,
            )

        if len(self.embeddings) != len(documents):
            return self.build_embeddings(documents)
        return self.embeddings

    def search(self, query, limit):
        if self.embeddings is None:
            raise ValueError(
                "No embeddings loaded. Call `load_or_create_embeddings` first."
            )
        query_embedding = self.generate_embeddign(query)
        similarity_scores = []

        for idx, embedding in enumerate(self.embeddings):
            if np.isnan(embedding).any():
                raise ValueError("Embeddings contain NaN values.")
            else:
                score = cosine_similarity(query_embedding, embedding)
                similarity_scores.append((score, self.documents[idx]))

        similarity_scores.sort(key=lambda x: x[0], reverse=True)
        return list(
            map(
                lambda x: {
                    "score": x[0],
                    "title": x[1]["title"],
                    "description": x[1]["description"],
                },
                similarity_scores[:limit],
            )
        )


class ChunkedSemanticSearch(SemanticSearch):
    def __init__(self, model_name="all-MiniLM-L6-v2") -> None:
        super().__init__(model_name)
        self.chunk_embeddings = None
        self.chunk_metadata = None

    def build_chunk_embeddings(self, documents):
        self.documents = documents
        docs = []

        chunks = []
        chunks_meta = []
        for idx, doc in enumerate(documents):
            self.doc_map[doc["id"]] = doc
            docs.append(doc["title"] + " " + doc["description"])
            if len(doc["description"].strip() != 0):
                chunks_per_doc = semantic_chunk(doc["description"], 4, 1)
                for chunk_idx, chunk in enumerate(chunks_per_doc):
                    chunks.append(chunk)
                    chunks_meta.append({
                        "movie_idx": idx,
                        "chunk_idx": chunk_idx,
                        "total_chunks": len(chunks_per_doc),
                    })
        self.chunk_embeddings = self.model.encode(chunks, show_progress_bar=True)
        self.chunk_metadata = chunks_meta
        np.save(
            chunk_embedding_path,
            self.chunk_embeddings,
        )
        with open(chunk_metadata_path, "w", encoding="utf-8") as f:
            json.dump({"chunks": chunks_meta, "total_chunks": len(chunks)}, f, indent=2)

        return self.chunk_embeddings

    def load_or_create_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:
        self.documents = documents
        docs = []
        chunks = []
        chunks_meta = []
        for idx, doc in enumerate(documents):
            self.doc_map[doc["id"]] = doc
            docs.append(doc["title"] + " " + doc["description"])
            if len(doc["description"].strip()) != 0:
                chunks_per_doc = semantic_chunk(doc["description"], 4, 1)
                for chunk_idx, chunk in enumerate(chunks_per_doc):
                    chunks.append(chunk)
                    chunks_meta.append({
                        "movie_idx": idx,
                        "chunk_idx": chunk_idx,
                        "total_chunks": len(chunks_per_doc),
                    })
        if os.path.exists(chunk_embedding_path) and os.path.exists(chunk_metadata_path):
            self.chunk_embeddings = np.load(chunk_embedding_path)
            with open(chunk_metadata_path, "r", encoding="utf-8") as f:
                self.chunk_metadata = json.load(f)["chunks"]
        else:
            self.chunk_embeddings = self.model.encode(chunks, show_progress_bar=True)
            self.chunk_metadata = chunks_meta
            np.save(
                chunk_embedding_path,
                self.chunk_embeddings,
            )
            with open(chunk_metadata_path, "w", encoding="utf-8") as f:
                json.dump(
                    {"chunks": chunks_meta, "total_chunks": len(chunks)}, f, indent=2
                )
        if len(self.chunk_embeddings) != len(chunks):
            return self.build_chunk_embeddings(documents)
        return self.chunk_embeddings

    def search_chunks(self, query: str, limit: int = 10) -> list[dict]:
        query_embedding = self.generate_embeddign(query)
        chunk_score = []
        if self.chunk_embeddings is None or self.chunk_metadata is None:
            raise ValueError(
                "No chunk embeddings loaded. Call `load_or_create_chunk_embeddings` first."
            )
        for idx, embedding in enumerate(self.chunk_embeddings):
            score = cosine_similarity(query_embedding, embedding)
            chunk_score.append({
                "chunk_idx": self.chunk_metadata[idx]["chunk_idx"],
                "movie_idx": self.chunk_metadata[idx]["movie_idx"],
                "score": score,
            })
        movie_scores = {}
        for score in chunk_score:
            if score["movie_idx"] not in movie_scores:
                movie_scores[score["movie_idx"]] = score["score"]
            elif score["score"] > movie_scores[score["movie_idx"]]:
                movie_scores[score["movie_idx"]] = score["score"]
        sorted_movies = sorted(movie_scores.items(), key=lambda x: x[1], reverse=True)[
            :limit
        ]
        formated_result = map(
            lambda movie: {
                "id": self.documents[movie[0]]["id"],
                "title": self.documents[movie[0]]["title"],
                "description": self.documents[movie[0]]["description"][:100] + "...",
                "score": round(movie[1], 4),
                "metadata": self.documents[movie[0]].get("metadata", {}),
            },
            sorted_movies,
        )
        return list(formated_result)


def semantic_chunk(text: str, max_chunk_size: int, overlap: int) -> list[str]:
    text = text.strip()
    if len(text) == 0:
        return []
    text_split = re.split(r"(?<=[.!?])\s+", text)
    chunks = []
    max_chunk_size = min(max_chunk_size, len(text_split))
    l, r = 0, max_chunk_size
    while r < len(text_split):
        chunk = " ".join([i.strip() for i in text_split[l:r]])
        if len(chunk) > 0:
            chunks.append(chunk)
        l = r - overlap
        r = l + max_chunk_size

    if r >= len(text_split) and l < len(text_split):
        chunks.append(" ".join(text_split[l : len(text_split)]))

    return chunks


def embed_chunks():
    if not os.path.exists(movies_path):
        raise FileNotFoundError("Movies dataset not found.")

    with open(movies_path, "r", encoding="utf-8") as f:
        movies_list = json.load(f)["movies"]

    model = ChunkedSemanticSearch()
    embeddings = model.load_or_create_chunk_embeddings(movies_list)
    print(f"Generated {len(embeddings)} chunked embeddings")


def verify_model():
    model = SemanticSearch()
    print(f"Model loaded {model.model}")
    print(f"Max Sequence Length: {model.model.max_seq_length}")


def embed_text(text: str) -> None:
    model = SemanticSearch()
    embedding = model.generate_embeddign(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")


def verify_embedding():
    with open(movies_path, "r", encoding="utf-8") as f:
        movies_list = json.load(f)["movies"]
    model = SemanticSearch()
    embeddings = model.load_or_create_embeddigns(movies_list)

    print(f"Number of docs:   {len(movies_list)}")
    print(
        f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions"
    )


def embed_query_text(query):
    model = SemanticSearch()
    embedding = model.generate_embeddign(query)
    print(f"Query: {query}")
    print(f"First 5 dimensions: {embedding[:5]}")
    print(f"Shape: {embedding.shape}")


def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)
