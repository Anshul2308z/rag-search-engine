import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Search Evaluation CLI")
    parser.add_argument(
        "--limit",
    type=int,
    default=5,
    help="Number of results to evaluate (k for precision@k, recall@k)",
    )

    args = parser.parse_args()
    limit = args.limit
    import json
    with open("data/golden_dataset.json", "r") as f:
        test_cases= json.load(f)["test_cases"]

    from lib.hybrid_search import HybridSearch
    from lib.search_utils import load_movies

    movies = load_movies()
    hybrid_search = HybridSearch(movies)

    test_case_precision = []
    test_case_recall = []

    for test_case in test_cases:
        results = hybrid_search.rrf_search(test_case["query"], 60, limit)
        relevant_retrieved = []
        total_relevant = test_case["relevant_docs"]
        total_retrived = []
        for result in results: 
            if result["document"]["title"] in test_case["relevant_docs"]:
                relevant_retrieved.append(result["document"]["title"])
            total_retrived.append(result["document"]["title"])
        
        test_case_precision.append({
            "Precision": len(relevant_retrieved)/ len(total_retrived) ,
            "Retrieved": total_retrived,
            "Relevant": relevant_retrieved,
            "query": test_case["query"],
            "Recall": len(relevant_retrieved)/ len(total_relevant)
        })
    


    for p in test_case_precision: 
        f1 = 2 * (p["Recall"] * p["Precision"] ) / (p["Precision"] + p["Recall"])
        print(f"k={limit}")
        print(f"- Query: {p["query"]}")
        print(f"  - Precision@{limit}: {p["Precision"]:.4f}")
        print(f"  - Recall@{limit}: {p["Recall"]:.4f}")
        print(f"  - F1 Score: {f1:.4f}")
        print(f"  - Retrieved: {', '.join(p["Retrieved"])}")
        print(f"  - Relevant: {', '.join(p["Relevant"])}")

 
     # run evaluation logic here

if __name__ == "__main__":
     main()