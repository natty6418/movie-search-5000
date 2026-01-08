import os
import json

from .inverted_index import InvertedIndex
from .semantic_search import ChunkedSemanticSearch

movies_path = os.path.join(os.path.dirname(__file__), "../../data/movies.json")


class HybridSearch:
    def __init__(self, documents):
        self.documents = documents
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)

        self.idx = InvertedIndex()
        if not os.path.exists(self.idx.index_path):
            self.idx.build()
            self.idx.save()

    def _bm25_search(self, query, limit):
        self.idx.load()
        return self.idx.bm25_search(query, limit)

    def _chunked_semantic_search(self, query, limit):
        return self.semantic_search.search_chunks(query, limit)

    def weighted_search(self, query, alpha, limit=5):
        bm25_results = self._bm25_search(query, limit * 500)
        chunked_semantic_results = self._chunked_semantic_search(query, limit * 500)

        bm25_results_normalized_scores = normalize([i[1] for i in bm25_results])
        semantic_results_normalized_scores = normalize([
            i["score"] for i in chunked_semantic_results
        ])
        result = {}
        for (doc, _), semantic_doc, bm25_norm_score, semantic_score in zip(
            bm25_results,
            chunked_semantic_results,
            bm25_results_normalized_scores,
            semantic_results_normalized_scores,
        ):
            if doc["id"] not in result:
                result[doc["id"]] = {
                    "doc": doc,
                    "keyword_search_score": 0.0,
                    "semantic_search_score": 0.0,
                }
            result[doc["id"]]["keyword_search_score"] += bm25_norm_score
            if semantic_doc["id"] not in result:
                result[semantic_doc["id"]] = {
                    "doc": self.idx.docmap[semantic_doc["id"]],
                    "semantic_search_score": 0.0,
                    "keyword_search_score": 0.0,
                }
            result[semantic_doc["id"]]["semantic_search_score"] += semantic_score
        for doc_id, entry in result.items():
            entry["hybrid_score"] = (
                alpha * entry["keyword_search_score"]
                + (1 - alpha) * entry["semantic_search_score"]
            )

        ranked_results = sorted(
            list(result.items()), key=lambda x: x[1]["hybrid_score"], reverse=True
        )[:limit]
        return ranked_results

    def rrf_search(self, query, k, limit=10):

        bm25_results = self._bm25_search(query, limit * 500)
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
            doc_id = s["id"]  # must be doc_id; if chunked, use s["doc_id"]
            entry = result.setdefault(
                doc_id,
                {
                    "doc": self.idx.docmap[doc_id],
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


def rrf_hybrid_search(query, k, limit=5):
    with open(movies_path, "r", encoding="utf-8") as f:
        movies_list = json.load(f)["movies"]
    if movies_list is None or len(movies_list) == 0:
        print("No movies found in the dataset.")
        return
    model = HybridSearch(movies_list)
    results = model.rrf_search(query, k, limit)
    for idx, (doc_id, result) in enumerate(results, start=1):
        doc = result["doc"]
        score = result["rrf_score"]
        print(
            f"{idx}. {doc['title']} (score: {score:.4f})\n   {doc['description'][:100]}...\r\n"
        )
    return results
