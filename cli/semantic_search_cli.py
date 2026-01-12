#!/usr/bin/env python3

import argparse
from lib.semantic_search import (
    SemanticSearch,
    ChunkedSemanticSearch,
    verify_model,
    embed_text,
    verify_embedding,
    embed_query_text,
    embed_chunks,
    semantic_chunk,
)
import os
import json
import re

movies_path = os.path.join(os.path.dirname(__file__), "../data/movies.json")


def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    subparsers.add_parser("verify_model", help="Verify model initalization")

    embed = subparsers.add_parser("embed", help="Generate embedding for input text")
    embed.add_argument("text", type=str, help="Input text to generate embedding for")

    subparsers.add_parser("verify_embedding", help="Verify embedding generation")

    embedquery = subparsers.add_parser(
        "embed_query", help="Generate embedding for input query text"
    )
    embedquery.add_argument(
        "query", type=str, help="Input query text to generate embedding for"
    )

    search = subparsers.add_parser(
        "search", help="Search documents using semantic search"
    )
    search.add_argument("query", type=str, help="Search query")
    search.add_argument(
        "--limit", type=int, default=5, help="Number of top results to return"
    )

    chunk = subparsers.add_parser("chunk", help="Chunk text into smaller pieces")
    chunk.add_argument("text", type=str, help="Input text to chunk")
    chunk.add_argument("--chunk_size", type=int, default=200, help="Size of each chunk")
    chunk.add_argument(
        "--overlap", type=int, default=20, help="Percentage overlap between chunks"
    )

    semantic_chunk_parser = subparsers.add_parser(
        "semantic_chunk", help="Semantic chunking of text"
    )
    semantic_chunk_parser.add_argument("text", type=str, help="Input text to chunk")
    semantic_chunk_parser.add_argument(
        "--max-chunk-size", type=int, default=4, help="Maximum size of each chunk"
    )
    semantic_chunk_parser.add_argument(
        "--overlap", type=int, default=0, help="Overlap between chunks"
    )

    subparsers.add_parser("embed_chunks", help="Embed chunks of documents")

    chunk_search = subparsers.add_parser(
        "chunk_search", help="Search in chunked documents"
    )
    chunk_search.add_argument("query", type=str, help="Search query")
    chunk_search.add_argument(
        "--limit", type=int, default=5, help="Number of top results to return"
    )

    args = parser.parse_args()

    match args.command:
        case "verify_model":
            verify_model()
        case "verify_embedding":
            verify_embedding()
        case "embed":
            embed_text(args.text)
        case "embed_query":
            embed_query_text(args.query)
        case "chunk":
            text = args.text
            chunk_size = args.chunk_size
            chunk_overlap = args.overlap
            overlap_size = int((chunk_size * chunk_overlap / 100))
            overlap_size = min(overlap_size, chunk_size - 1)
            text_split = text.split()

            chunks = []

            l, r = 0, chunk_size
            while r < len(text_split):
                chunks.append(" ".join(text_split[l:r]))
                l = r - overlap_size
                r = l + chunk_size

            if r >= len(text_split) and l < len(text_split):
                chunks.append(" ".join(text_split[l : len(text_split)]))

            print(f"Chunking {chunk_size} characters")
            for idx, chunk in enumerate(chunks):
                print(f"{idx + 1}: {chunk}\n")

        case "semantic_chunk":
            text = args.text
            max_chunk_size = args.max_chunk_size
            overlap = args.overlap

            chunks = semantic_chunk(text, max_chunk_size, overlap)

            print(f"Semantically chunking {len(text)} characters")
            for idx, chunk in enumerate(chunks):
                print(f"{idx + 1}: {chunk}\n")

        case "embed_chunks":
            embed_chunks()

        case "search":
            model = SemanticSearch()
            with open(movies_path, "r", encoding="utf-8") as f:
                movies_list = json.load(f)["movies"]
            if movies_list is None or len(movies_list) == 0:
                print("No movies found in the dataset.")
                return
            if model.embeddings is None:
                model.load_or_create_embeddings(movies_list)
            results = model.search(args.query, args.limit)
            for idx, result in enumerate(results):
                print(
                    f"{idx + 1}. {result['title']} (score: {result['score']:.4f})\n   {result['description']}\r\n"
                )
        case "chunk_search":
            model = ChunkedSemanticSearch()
            with open(movies_path, "r", encoding="utf-8") as f:
                movies_list = json.load(f)["movies"]
            if movies_list is None or len(movies_list) == 0:
                print("No movies found in the dataset.")
                return
            if model.embeddings is None:
                model.load_or_create_chunk_embeddings(movies_list)
            results = model.search_chunks(args.query, args.limit)
            for idx, result in enumerate(results):
                print(f"{idx + 1}. {result['title']} (score: {result['score']:.4f})")
                print(f"   {result['description']}\r\n")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
