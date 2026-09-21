import argparse
from lib.semantic_search import verify_model, embed_text, verify_embeddings, embed_query_text, SementicSearch
import json


def main() -> None:
    parser = argparse.ArgumentParser(description="Sementic Search CLI")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    verify_parser = subparsers.add_parser("verify", help="verifies the model being used for embedding")
    generate_embedding_parser = subparsers.add_parser("embed_text", help="embed a single text using our model")
    generate_embedding_parser.add_argument("text", type=str, help =" a non empty string and not a sequence of whitespaces only!")

    verify_embeddings_parser = subparsers.add_parser("verify_embeddings", help="verifies embeddings cached against the db, automatically embedds the db in case of inconsitency")

    embed_query_parser= subparsers.add_parser("embed_query", help="embedd query to search it against the db")
    embed_query_parser.add_argument("query", type=str, help="Provide query")

    search_parser = subparsers.add_parser("search", help="semantically search query against the vector db")
    search_parser.add_argument("query", type=str, help="a term to query against db")
    search_parser.add_argument("--limit", type=int, default=5, help="limit to X most similar results")

    args = parser.parse_args()
    match args.command: 
        case "verify":
            verify_model()
        case "embed_text":
            embed_text(args.text)
        case "verify_embeddings":
            verify_embeddings()
        case "embed_query":
            embed_query_text(args.query)
        case "search":
            query = args.query
            limit = args.limit

            sementic_search = SementicSearch()

            documents = []
            with open("data/movies.json", "r") as f:
                documents = json.load(f)["movies"]

            embeddings = sementic_search.load_or_create_embeddings(documents)

            results = sementic_search.search(query, limit)

            for i, r in enumerate(results):
                print(f"{i+1}. {r["title"]}")
                print(r["description"])

        case _:
            parser.print_help()

if __name__ == "__main__": 
    main()

