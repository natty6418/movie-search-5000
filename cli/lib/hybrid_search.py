import os
import json

from .inverted_index import InvertedIndex, BigramInvertedIndex
from .semantic_search import ChunkedSemanticSearch

from sentence_transformers import CrossEncoder

movies_path = os.path.join(os.path.dirname(__file__), "../../data/movies.json")


class HybridSearch:
    def __init__(self, documents):
        self.documents = documents
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)

        # Unigram Index
        self.idx = InvertedIndex()
        if not os.path.exists(self.idx.index_path):
            self.idx.build()
            self.idx.save()
        else:
            self.idx.load()

        # Bigram Index
        self.bigram_idx = BigramInvertedIndex()
        if not os.path.exists(self.bigram_idx.index_path):
            self.bigram_idx.build()
            self.bigram_idx.save()
        else:
            self.bigram_idx.load()

    def _bm25_search(self, query, limit, mode="combined"):
        if mode == "unigram":
            return self.idx.bm25_search(query, limit)
        elif mode == "bigram":
            return self.bigram_idx.bm25_search(query, limit)
        
        # Combined logic
        unigram_results = self.idx.bm25_search(query, limit * 2)
        bigram_results = self.bigram_idx.bm25_search(query, limit * 2)
        
        # Combine scores
        combined = {}
        for doc, score in unigram_results:
            combined[doc["id"]] = {"doc": doc, "score": score}
        
        for doc, score in bigram_results:
            if doc["id"] in combined:
                # Bigram matches are weighted higher
                combined[doc["id"]]["score"] += score * 1.5
            else:
                combined[doc["id"]] = {"doc": doc, "score": score}
        
        sorted_results = sorted(combined.values(), key=lambda x: x["score"], reverse=True)
        return [(item["doc"], item["score"]) for item in sorted_results[:limit]]

    def _chunked_semantic_search(self, query, limit):
        return self.semantic_search.search_chunks(query, limit)

    def weighted_search(self, query, alpha, limit=5, bm25_mode="combined"):
        bm25_results = self._bm25_search(query, limit * 500, mode=bm25_mode)
        chunked_semantic_results = self._chunked_semantic_search(query, limit * 500)

        bm25_results_normalized_scores = normalize([i[1] for i in bm25_results])
        semantic_results_normalized_scores = normalize([
            i["score"] for i in chunked_semantic_results
        ])
        
        result = {}

        # Process BM25 results
        for (doc, _), bm25_norm_score in zip(bm25_results, bm25_results_normalized_scores):
            doc_id = doc["id"]
            if doc_id not in result:
                result[doc_id] = {
                    "doc": doc,
                    "keyword_search_score": 0.0,
                    "semantic_search_score": 0.0,
                }
            result[doc_id]["keyword_search_score"] = bm25_norm_score

        # Process Semantic results
        for semantic_doc, semantic_score in zip(chunked_semantic_results, semantic_results_normalized_scores):
            doc_id = semantic_doc["id"]
            if doc_id not in result:
                result[doc_id] = {
                    "doc": self.idx.docmap.get(doc_id) or semantic_doc, # Fallback if docmap missing
                    "keyword_search_score": 0.0,
                    "semantic_search_score": 0.0,
                }
            result[doc_id]["semantic_search_score"] = semantic_score

        # Calculate Hybrid Score
        for doc_id, entry in result.items():
            entry["hybrid_score"] = (
                alpha * entry["keyword_search_score"]
                + (1 - alpha) * entry["semantic_search_score"]
            )

        ranked_results = sorted(
            list(result.items()), key=lambda x: x[1]["hybrid_score"], reverse=True
        )[:limit]
        return ranked_results

    def rrf_search(self, query, k, limit=10, bm25_mode="combined"):

        bm25_results = self._bm25_search(query, limit * 500, mode=bm25_mode)
        chunked_semantic_results = self._chunked_semantic_search(query, limit * 500)

        result = {}
        for rank, (doc, _) in enumerate(bm25_results, start=1):
            doc_id = doc["id"]
            entry = result.setdefault(
                doc_id,
                {
                    "doc": doc,
                    "keyword_search_rank": 0,
                    "semantic_search_rank": 0,
                },
            )
            entry["keyword_search_rank"] = rank

        # semantic ranks
        for rank, s in enumerate(chunked_semantic_results, start=1):
            doc_id = s["id"]
            entry = result.setdefault(
                doc_id,
                {
                    "doc": self.idx.docmap.get(doc_id) or s, # Fallback
                    "keyword_search_rank": 0,
                    "semantic_search_rank": 0,
                },
            )
            entry["semantic_search_rank"] = rank
            
        for doc_id, entry in result.items():
            keyw_score = (
                1 / (k + entry["keyword_search_rank"])
                if entry["keyword_search_rank"] > 0
                else 0
            )
            semantic_score = (
                1 / (k + entry["semantic_search_rank"])
                if entry["semantic_search_rank"] > 0
                else 0
            )
            entry["rrf_score"] = keyw_score + semantic_score

        ranked_results = sorted(
            list(result.items()), key=lambda x: x[1]["rrf_score"], reverse=True
        )[:limit]
        return ranked_results


def normalize(scores: list[float]) -> list[float]:
    min_score = min(scores)
    max_score = max(scores)

    if min_score == max_score:
        return [1.0 for _ in scores]

    normalized_scores = [(i - min_score) / (max_score - min_score) for i in scores]
    return normalized_scores


def hybrid_search(query, alpha, limit=5):
    with open(movies_path, "r", encoding="utf-8") as f:
        movies_list = json.load(f)["movies"]
    if movies_list is None or len(movies_list) == 0:
        print("No movies found in the dataset.")
        return
    model = HybridSearch(movies_list)
    results = model.weighted_search(query, alpha, limit)
    for idx, (doc_id, result) in enumerate(results, start=1):
        doc = result["doc"]
        score = result["hybrid_score"]
        print(
            f"{idx}. {doc['title']} (score: {score:.4f})\n   {doc['description'][:100]}...\r\n"
        )


def rrf_hybrid_search(query: str, k: int, limit: int = 5) -> list[tuple[str, dict]]:
    """
    Reciprocal Rank Fusion Hybrid Search
    Prints the top results from RRF hybrid search for the given query.

        Args:
            query (str): The search query.
            k (int): The RRF parameter.
            limit (int): The number of top results to return.

        Returns:

        list: A list of tuples containing document IDs and their corresponding search results.
        Each tuple is in the format (doc_id, result), where result is a dictionary containing
        the document and its RRF score.

        Output:
        1. Movie Title 1 (score: 0.1234)
           Movie description snippet...
        ...
        Example:
        results = rrf_hybrid_search("science fiction adventure", k=60, limit=5)

        # results = [
        #   ('doc1', {
        #       'doc': {...},
        #       'rrf_score': 0.2345,
        #       'keyword_search_rank': 2,
        #       'semantic_search_rank': 3
        #   }),
    """
    with open(movies_path, "r", encoding="utf-8") as f:
        movies_list = json.load(f)["movies"]
    if movies_list is None or len(movies_list) == 0:
        raise ValueError("No movies found in the dataset.")
    model = HybridSearch(movies_list)
    results = model.rrf_search(query, k, limit)
    for idx, (doc_id, result) in enumerate(results, start=1):
        doc = result["doc"]
        score = result["rrf_score"]
        print(
            f"{idx}. {doc['title']} (score: {score:.4f})\n   {doc['description'][:100]}...\r\n"
        )
    return results


_CROSS_ENCODER_MODEL = None

def cross_encoder_rerank_results(results, query, limit):
    global _CROSS_ENCODER_MODEL
    if results is None:
        raise ValueError("No results returned from hybrid search")
    
    if len(results) == 0:
        return []

    if _CROSS_ENCODER_MODEL is None:
        _CROSS_ENCODER_MODEL = CrossEncoder("cross-encoder/ms-marco-TinyBERT-L2-v2")

    pairs = []
    for doc_id, result in results:
        doc = result["doc"]
        pairs.append([
            query,
            f"{doc.get('title', '')} - {doc.get('description', '')}",
        ])
    
    if not pairs:
        return []

    # scores is a list of numbers, one for each pair
    scores = _CROSS_ENCODER_MODEL.predict(pairs)
    ranked_result = sorted(
        list(zip(results, scores)), key=lambda x: x[1], reverse=True
    )[:limit]

    return ranked_result
