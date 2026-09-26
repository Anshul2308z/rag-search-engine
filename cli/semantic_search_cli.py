import argparse
from lib.semantic_search import verify_model, embed_text, verify_embeddings, embed_query_text, SemanticSearch
import json
import math
import re

def Chunks(text, size, overlap):
    words = text.split(" ")

    # if size == 0 :
    #     print("invalid size!")
    #     return 
    
    chunks = [words[i:i+size] for i in range(0, len(words), size)]

    chunks = []

    i = 0 

    while i <= len(words) :  

        if overlap == 0 :
            chunks = [words[i:i+size] for i in range(0, len(words), size)]
            break 

        if i > len(words): 
            break 
        if overlap >= size : 
            raise Exception("Invalid overlap: {overlap} for size: {size}")

        if i == 0 :
            chunks.append(
                words[i: i+size]
            )
            i+= size  # fixed bug
        else: 
            chunks.append(
                words[i-overlap: i+size]
            )
            i+= size - overlap # fixed bug 

    characters = 0
    for word in words: 
        characters += len(word)
    characters += len(words) - 1 

    return characters, chunks 

def semantic_chunk(text, size, overlap):

    sentences = re.split(r"(?<=[.!?])\s+", text)
    characters = len(text)

    i = 0
    chunks= []

    while i <= len(sentences):
    #     if overlap <= size :
    #         raise Exception("Invalid overlap: {overlap} for size: {size}")

            
        if i == 0:
            chunks.append(sentences[i:i+size])
            i+= size
        else: 
            chunks.append(
                sentences[i - overlap: i+ size - overlap]
            )
            i+= size - overlap

    return characters, chunks 
    


def main() -> None:
    parser = argparse.ArgumentParser(description="semantic Search CLI")

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

    chunk_parser = subparsers.add_parser("chunk", help="give a string of text to chunk and optionally chunk size")
    chunk_parser.add_argument("toChunk",type=str, help="This is a positional argument for text to chunk")
    chunk_parser.add_argument("--chunk-size", type=int, default=200, help="define chunk size, default is 200")
    chunk_parser.add_argument("--overlap", type= int, help="define overlap, default is None/ \"0" )

    
    semantic_chunk_parser = subparsers.add_parser("semantic_chunk", help="give a string of text to chunk and optionally max chunk size for semantic chunking")
    semantic_chunk_parser.add_argument("text", type=str, help="Provide text to chunk")
    semantic_chunk_parser.add_argument("--max-chunk-size", default=4, type=int, help="max chunk size, defaulting to 4")
    semantic_chunk_parser.add_argument("--overlap", type=int, default=0, help="overlap this much last words? Defaults to zero")

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
        case "chunk":
            (characters, chunks) = Chunks(args.toChunk, args.chunk_size, args.overlap) #The key detail is that --chunk-size becomes args.chunk_size, not args.chunk-size, because argparse converts hyphens in argument names to underscores.
            print(f"Chunking {characters} characters")
            for i, chunk in enumerate(chunks):
                print(i+1, end=". ")
                for word in chunk: 
                    print(word, end=" ")
                print()


            
        case "search":
            query = args.query
            limit = args.limit

            semantic_search = SemanticSearch()

            documents = []
            with open("data/movies.json", "r") as f:
                documents = json.load(f)["movies"]

            embeddings = semantic_search.load_or_create_embeddings(documents)

            results = semantic_search.search(query, limit)

            for i, r in enumerate(results):
                print(f"{i+1}. {r["title"]}")
                print(r["description"])
        case "semantic_chunk":
            characters, chunks = semantic_chunk(args.text, args.max_chunk_size, args.overlap)
            print(f"Semantically chunking {characters} characters")
            for i, chunk in enumerate(chunks):
                print(i+1, end=". ")
                for s in chunk:
                    print(s, end=" ")
                print()    
            

        case _:
            parser.print_help()

if __name__ == "__main__": 
    main()

