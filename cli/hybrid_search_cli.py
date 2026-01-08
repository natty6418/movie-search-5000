import argparse


from lib.hybrid_search import normalize, hybrid_search, rrf_hybrid_search
from lib.local_llm import (
    enhance_query,
    rewrite_query,
    expand_query,
    rank_results,
    batch_rank_results,
)
from sentence_transformers import CrossEncoder


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    normalize_parser = subparsers.add_parser(
        "normalize", help="Normalize a list of scores"
    )
    normalize_parser.add_argument(
        "--nums",
        type=float,
        nargs="+",
        required=True,
        help="List of scores to normalize",
    )
    weighted_search_parser = subparsers.add_parser(
        "weighted_search", help="Perform weighted hybrid search"
    )
    weighted_search_parser.add_argument("query", type=str, help="Search query")
    weighted_search_parser.add_argument(
        "--alpha",
        type=float,
        required=True,
        help="Weight for semantic search component",
    )
    weighted_search_parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of top results to return",
    )

    rrf_search_parser = subparsers.add_parser(
        "rrf-search", help="Perform RRF hybrid search"
    )
    rrf_search_parser.add_argument("query", type=str, help="Search query")
    rrf_search_parser.add_argument("--k", type=int, default=60, help="RRF k parameter")
    rrf_search_parser.add_argument(
        "--limit", type=int, default=5, help="Number of top results to return"
    )
    rrf_search_parser.add_argument(
        "--enhance",
        type=str,
        choices=["spell", "rewrite", "expand"],
        help="Query enhancement method",
    )
    rrf_search_parser.add_argument(
        "--rerank-method",
        type=str,
        choices=["individual", "batch", "cross_encoder"],
        help="Reranking method to apply after RRF",
    )
    args = parser.parse_args()

    match args.command:
        case "normalize":
            normalize(args.nums)

        case "weighted_search":
            hybrid_search(args.query, args.alpha, args.limit)

        case "rrf-search":
            query = args.query
            if args.enhance == "spell":
                query = enhance_query(args.query)
                print(f"Enhanced query (spell): '{args.query}' -> '{query}'\n")
            if args.enhance == "rewrite":
                query = rewrite_query(args.query)
                print(f"Enhanced query (rewrite): '{args.query}' -> '{query}'\n")
            if args.enhance == "expand":
                query = expand_query(args.query)
                print(f"Enhanced query (expand): '{args.query}' -> '{query}'\n")

            if args.rerank_method == "individual":
                results = rrf_hybrid_search(
                    query,
                    args.k,
                    args.limit * 5,
                )
                if results is None:
                    raise ValueError("No results returned from RRF hybrid search")
                print(
                    f"Reranking top {args.limit} results using {args.rerank_method} method..."
                )
                for doc_id, result in results:
                    doc = result["doc"]
                    score = float(rank_results(args.query, doc))
                    result["final_score"] = score

                reranked = sorted(
                    results,
                    key=lambda x: x[1]["final_score"],
                    reverse=True,
                )[: args.limit]
                print(f"Reciprocal Rank Fusion Results for '{query}' (k={args.k}):")

                for idx, (doc_id, result) in enumerate(reranked, start=1):
                    doc = result["doc"]
                    bm25_rank = result["keyword_search_rank"]
                    semantic_rank = result["semantic_search_rank"]
                    final_score = result["final_score"]
                    print(
                        f"{idx}. {doc['title']}\n   Rerank score: {final_score:.4f})\n   BM25 Rank: {bm25_rank}, Semantic Rank: {semantic_rank}\n   {doc['description'][:100]}...\r\n"
                    )

                return
            elif args.rerank_method == "batch":
                results = rrf_hybrid_search(
                    query,
                    args.k,
                    args.limit * 5,
                )
                if results is None:
                    raise ValueError("No results returned from RRF hybrid search")
                print(
                    f"Reranking top {args.limit} results using {args.rerank_method} method..."
                )
                reranked_list = batch_rank_results(
                    query, list(map(lambda result: result[1]["doc"], results))
                )
                results = dict(results)

                for idx, doc_id in enumerate(reranked_list, start=1):
                    doc = results[doc_id]["doc"]
                    bm25_rank = results[doc_id]["keyword_search_rank"]
                    semantic_rank = results[doc_id]["semantic_search_rank"]
                    score = results[doc_id]["rrf_score"]
                    print(
                        f"{idx}. {doc['title']}\n   Rerank Rank: {idx}\n   RRF score: {score:.4f}\n   BM25 Rank: {bm25_rank}, Semantic Rank: {semantic_rank}\n   {doc['description'][:100]}...\r\n"
                    )
                    if idx > args.limit:
                        return
            elif args.rerank_method == "cross_encoder":
                results = rrf_hybrid_search(
                    query,
                    args.k,
                    args.limit * 5,
                )
                if results is None:
                    raise ValueError("No results returned from RRF hybrid search")
                print(
                    f"Reranking top {args.limit} results using {args.rerank_method} method..."
                )
                pairs = []
                cross_encoder = CrossEncoder("cross-encoder/ms-marco-TinyBERT-L2-v2")
                for doc_id, result in results:
                    doc = result["doc"]
                    pairs.append([
                        query,
                        f"{doc.get('title', '')} - {doc.get('description', '')}",
                    ])
                # scores is a list of numbers, one for each pair
                scores = cross_encoder.predict(pairs)
                result = sorted(
                    list(zip(results, scores)), key=lambda x: x[1], reverse=True
                )[: args.limit]
                for idx, ((doc_id, result), score) in enumerate(result, start=1):
                    doc = result["doc"]
                    bm25_rank = result["keyword_search_rank"]
                    semantic_rank = result["semantic_search_rank"]
                    rrf_score = result["rrf_score"]
                    print(
                        f"{idx}. {doc['title']}\n   Cross Encoder Score: {score:.4f})\n   RRF Score: {rrf_score}\n   BM25 Rank: {bm25_rank}, Semantic Rank: {semantic_rank}\n   {doc['description'][:100]}...\r\n"
                    )
                return
            rrf_hybrid_search(query, args.k, args.limit)

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
