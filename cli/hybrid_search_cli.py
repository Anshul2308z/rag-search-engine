import argparse
from lib.hybrid_search import normalize

def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")

    sub_parsers = parser.add_subparsers(dest="command", help="Available commands")

    normalize_parser = sub_parsers.add_parser("normalize", help="Normalize documents")
    normalize_parser.add_argument( "scores", nargs="*", help="List of scores to normalize")

    weighted_search = sub_parsers.add_parser("weighted_search", help="weighted hybrid serach")
    weighted_search.add_argument("query", type=str, help="type query")
    weighted_search.add_argument("--alpha", type=float, default=0.5, help="a constant that we can use to dynamically control the weighting")
    weighted_search.add_argument("--limit", type=int, default=5, help="limit results")


    args = parser.parse_args()

    match args.command:
        case "normalize":
            scores = args.scores
            normalized_scores = normalize(scores)

            for score in normalized_scores:
                print(f"{score:.4f}")

        case _:
            parser.print_help()



if __name__ == "__main__":
    main()