#!/usr/bin/env python3

import argparse
import math
from lib.inverted_index import (
    InvertedIndex,
    tokenize,
    bm25_idf_command,
    BM25_K1,
    BM25_B,
    bm25_tf_command,
    bm25search_command,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    subparsers.add_parser("build", help="Build inverted index")

    tf_parser = subparsers.add_parser("tf", help="Get term frequency for a document")
    tf_parser.add_argument("doc_id", type=int, help="Document ID")
    tf_parser.add_argument("term", type=str, help="Term to check frequency for")

    idf_parser = subparsers.add_parser(
        "idf", help="Get inverse document frequency for a term"
    )
    idf_parser.add_argument("term", type=str, help="Term to check IDF for")

    tfidf_parser = subparsers.add_parser(
        "tfidf", help="Get TF-IDF for a term in a document"
    )
    tfidf_parser.add_argument("doc_id", type=int, help="Document ID")
    tfidf_parser.add_argument("term", type=str, help="Term to check TF-IDF for")

    bm25_idf_parser = subparsers.add_parser(
        "bm25idf", help="Get BM25 IDF score for a given term"
    )
    bm25_idf_parser.add_argument(
        "term", type=str, help="Term to get BM25 IDF score for"
    )

    bm25_tf_parser = subparsers.add_parser(
        "bm25tf", help="Get BM25 TF score for a given document ID and term"
    )
    bm25_tf_parser.add_argument("doc_id", type=int, help="Document ID")
    bm25_tf_parser.add_argument("term", type=str, help="Term to get BM25 TF score for")
    bm25_tf_parser.add_argument(
        "k1", type=float, nargs="?", default=BM25_K1, help="Tunable BM25 K1 parameter"
    )
    bm25_tf_parser.add_argument(
        "b", type=float, nargs="?", default=BM25_B, help="Tunable BM25 b parameter"
    )
    bm25search_parser = subparsers.add_parser(
        "bm25search", help="Search movies using full BM25 scoring"
    )
    bm25search_parser.add_argument("query", type=str, help="Search query")
    bm25search_parser.add_argument(
        "limit",
        type=int,
        nargs="?",
        default=5,
        help="Maximum number of results to return",
    )

    args = parser.parse_args()

    inverted_index = InvertedIndex()

    match args.command:
        case "search":
            print("Searching for:", args.query)
            try:
                results = []
                inverted_index.load()
                for word in args.query.split():
                    doc_ids = inverted_index.get_documents(word.lower())
                    for doc_id in doc_ids:
                        results.append({
                            "id": doc_id,
                            "title": inverted_index.docmap[doc_id],
                        })

                for result in results[:5]:
                    print(f"{result['id']}. {result['title']}")
            except FileNotFoundError:
                print("Inverted index not found. Please build the index first.")
                return
        case "build":
            inverted_index.build()
            inverted_index.save()
            return

        case "tf":
            inverted_index.load()
            tf = inverted_index.get_tf(args.doc_id, args.term)
            print(tf)

        case "idf":
            tokenized_term = tokenize(args.term)
            if len(tokenized_term) > 1:
                raise ValueError("Term must be a single word")
            token = tokenized_term[0]
            inverted_index.load()
            total_doc_count = len(inverted_index.docmap)
            term_match_doc_count = len(inverted_index.get_documents(token))
            idf = math.log((total_doc_count + 1) / (term_match_doc_count + 1))
            print(f"Inverse document frequency of '{args.term}': {idf:.2f}")
            return
        case "tfidf":
            inverted_index.load()
            tokenized_term = tokenize(args.term)
            if len(tokenized_term) > 1:
                raise ValueError("Term must be a single word")
            token = tokenized_term[0]
            tf = inverted_index.get_tf(args.doc_id, token)
            total_doc_count = len(inverted_index.docmap)
            term_match_doc_count = len(inverted_index.get_documents(token))
            idf = math.log((total_doc_count + 1) / (term_match_doc_count + 1))

            tfidf = tf * idf

            print(
                f"TF-IDF score of '{args.term}' in document '{args.doc_id}': {tfidf:.2f}"
            )

        case "bm25idf":
            bm25_idf = bm25_idf_command(args.term)
            print(f"BM25 IDF score of '{args.term}': {bm25_idf:.2f}")

        case "bm25tf":
            bm25tf = bm25_tf_command(args.doc_id, args.term, args.k1)
            print(
                f"BM25 TF score of '{args.term}' in document '{args.doc_id}': {bm25tf:.2f}"
            )

        case "bm25search":
            results = bm25search_command(args.query, args.limit)
            for idx, (doc, score) in enumerate(results, start=1):
                print(f"{idx}. {doc['title']} - Score: {score:.2f}")

        case _:
            parser.print_help()

    # for result in results[:5]:
    #     print(f"{result['id']}. {result['title']}")


if __name__ == "__main__":
    main()
