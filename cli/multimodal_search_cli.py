import argparse 
from lib.multimodal_search import verify_image_embedding, image_search_command

def main()-> None : 
    parser = argparse.ArgumentParser("Mutlimodel Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    verify_image_embedding_parser = subparsers.add_parser(
        "verify_image_embedding",
    )
    verify_image_embedding_parser.add_argument("path", type=str, help="image path")

    image_search_parser = subparsers.add_parser(
        "image_search",
        help="search by image"
    )
    image_search_parser.add_argument("path", type=str, help="image path")

    args = parser.parse_args()

    match args.command:
        case "verify_image_embedding":
            verify_image_embedding(args.path)
        case "image_search":
            path = args.path
            results = image_search_command(path)

            for i, result in enumerate(results):
                print(f"{i+1}. {result["title"]} (similarity: {result["similarity_score"]:.3f})")
                print(f"    {result["description"]}")

if __name__ == "__main__":
    main()