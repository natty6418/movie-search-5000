import argparse
import os
from lib.hybrid_search import rrf_hybrid_search, cross_encoder_rerank_results
from lib.local_llm import rag_answer, summarize, generate_citations, question_answering
from colorama import Fore, Style

movies_path = os.path.join(os.path.dirname(__file__), "../data/movies.json")


def main():
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    rag_parser = subparsers.add_parser(
        "rag", help="Perform RAG (search + generate answer)"
    )
    rag_parser.add_argument("query", type=str, help="Search query for RAG")

    sumarize_parser = subparsers.add_parser(
        "summarize", help="Summarize documents using RAG"
    )
    sumarize_parser.add_argument(
        "query", type=str, help="Search query for summarization"
    )
    sumarize_parser.add_argument(
        "--limit", type=int, default=5, help="Number of documents to summarize"
    )

    citation_parser = subparsers.add_parser(
        "citation", help="Generate citations for a given text"
    )
    citation_parser.add_argument(
        "query", type=str, help="Query to anser with citations"
    )
    citation_parser.add_argument(
        "--limit", type=int, default=5, help="Number of citations to generate"
    )

    question_parser = subparsers.add_parser(
        "question", help="Generate answer to a given question"
    )
    question_parser.add_argument(
        "question", type=str, help="Question to answer"
    )
    question_parser.add_argument(
        "--limit", type=int, default=5, help="Number of documents to summarize"
    )

    args = parser.parse_args()

    match args.command:
        case "rag":
            query = args.query
            results = rrf_hybrid_search(query, k=60, limit=5)
            response = rag_answer(query, list(map(lambda x: x[1]["doc"], results)))

            print("Search Results:")
            for doc_id, doc in results:
                print(f"-  {doc['doc']['title']}")
            print(f"\nRAG response:\n{response}")
            # do RAG stuff here
        case "summarize":
            query = args.query
            limit = args.limit
            results = rrf_hybrid_search(query, k=60, limit=limit * 5)
            results = cross_encoder_rerank_results(results, query, limit)
            docs_to_summarize = list(map(lambda x: x[0][1]["doc"], results))
            response = summarize(query, docs_to_summarize)

            print(Fore.YELLOW)
            print("Search Results:")
            for (doc_id, doc), score in results:
                print(f"-  {doc['doc']['title']}")

            print(Fore.GREEN)
            print(f"\nLLM Summary:\n{response}")
            print(Style.RESET_ALL)
        case "citation":
            query = args.query
            limit = args.limit
            results = rrf_hybrid_search(query, k=60, limit=limit * 5)
            results = cross_encoder_rerank_results(results, query, limit)
            docs_to_cite = list(map(lambda x: x[0][1]["doc"], results))
            response = generate_citations(query, docs_to_cite)

            print(Fore.YELLOW)
            print("Search Results:")
            for (doc_id, doc), score in results:
                print(f"-  {doc['doc']['title']}")

            print(Fore.GREEN)
            print(f"\nGenerated citations:\n{response}")
            print(Style.RESET_ALL)

        case "question":
            query = args.question
            limit = args.limit
            results = rrf_hybrid_search(query, k=60, limit=limit * 5)
            results = cross_encoder_rerank_results(results, query, limit)
            docs_to_answer = list(map(lambda x: x[0][1]["doc"], results))
            response = question_answering(query, docs_to_answer)

            print(Fore.YELLOW)
            print("Search Results:")
            for (doc_id, doc), score in results:
                print(f"-  {doc['doc']['title']}")

            print(Fore.GREEN)
            print(f"\nGenerated answer:\n{response}")
            print(Style.RESET_ALL)


        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
