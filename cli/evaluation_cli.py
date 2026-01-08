import argparse
import json
import os
from lib.hybrid_search import rrf_hybrid_search

golden_data_path = os.path.join(
    os.path.dirname(__file__), "../data/golden_dataset.json"
)


def main():
    parser = argparse.ArgumentParser(description="Search Evaluation CLI")
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of results to evaluate (k for precision@k, recall@k)",
    )

    args = parser.parse_args()
    limit = args.limit

    # run evaluation logic here
    with open(golden_data_path, "r", encoding="utf-8") as f:
        golden_data = json.load(f)

    for test_case in golden_data["test_cases"]:
        query = test_case["query"]
        result = rrf_hybrid_search(query, k=60, limit=limit)
        relevant_titles_num = 0
        for _, doc in result:
            if doc["doc"]["title"] in test_case["relevant_docs"]:
                relevant_titles_num += 1
        precision_at_k = relevant_titles_num / limit
        recall_at_k = relevant_titles_num / len(test_case["relevant_docs"])
        f1_at_k = (
            2 * (precision_at_k * recall_at_k) / (precision_at_k + recall_at_k)
            if (precision_at_k + recall_at_k) > 0
            else 0.0
        )
        print(
            f"""- Query: "{query}"
    - Precision@{limit}: {precision_at_k:.2f}
    - Recall@{limit}: {recall_at_k:.2f}
    - F1@{limit}: {f1_at_k:.2f}
    - Retrieved: {[doc["doc"]["title"] for _, doc in result]}
    - Relevant: {test_case["relevant_docs"]}"""
        )


if __name__ == "__main__":
    main()
