from lib.hybrid_search import HybridSearch
from llm_integration import augmented_generation, query_llm
from lib.search_utils import load_movies
from lib.search_utils import RRF_K

import argparse

def main() -> None:
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    rag_parser = subparsers.add_parser(
        "rag", help="Perform RAG (search + generate answer)"
    )
    rag_parser.add_argument("query", type=str, help="Search query for RAG")

    summary_parser = subparsers.add_parser(
        "summarize", help="summarize resulting documents using an LLM"
    )
    summary_parser.add_argument("query", type=str, help="Search query for RAG: summary")

    citations_parser = subparsers.add_parser(
        "citations", help="Get a citated result"
    )

    question_parser = subparsers.add_parser(
        "question",
        help="Question the LLM based on our serach results"
    )
    question_parser.add_argument("question", type=str, help="Enter question!")
    question_parser.add_argument("--limit", type=int, default=5, help="Limit search results")

    citations_parser.add_argument("query", type=str, help="Search query for a citated response")
    citations_parser.add_argument("--limit", type=int,default=5 ,help="")

    args = parser.parse_args()
    match args.command:
        case "rag":

            movies = load_movies()
            hybrid_search = HybridSearch(movies)
            query = args.query
            # do RAG stuff here
            results = hybrid_search.rrf_search(query, 5)
            response = augmented_generation(query, results)

            print("Search Results")
            for result in results:
                print(f"- {result["document"]["title"]}")
            print("RAG Response:", response.strip())

        case "summarize":
            movies = load_movies()
            hybrid_search = HybridSearch(movies)
            query = args.query
            results = hybrid_search.rrf_search(query,RRF_K, args.limit)

            docs = []
            for r in results:
                docs.append(r["document"])
                
            
            
            request = f"""Provide information useful to the query below by synthesizing data from multiple search results in detail.
                    The goal is to provide comprehensive information so that users know what their options are.
                    Your response should be information-dense and concise, with several key pieces of information about the genre, plot, etc. of each movie.

                    This should be tailored to Webflyx users. Webflyx is a movie streaming service.

                    Query: {query}

                    Search results:
                    {docs}

                    Provide a comprehensive 3-4 sentence answer that combines information from multiple sources:"""

            response = query_llm(request)

            print("Search Results:")
            for result in results:
                print(f"- {result["document"]["title"]}")

            print("LLM Summary:")
            print(response.strip())

        case "citations":
            movies = load_movies()
            hybrid_search = HybridSearch(movies)

            results = hybrid_search.rrf_search(args.query, RRF_K, args.limit)

            request = prompt = f"""Answer the query below and give information based on the provided documents.

                        The answer should be tailored to users of Webflyx, a movie streaming service.
                        If not enough information is available to provide a good answer, say so, but give the best answer possible while citing the sources available.

                        Query: {args.query}

                        Documents:
                        {results}

                        Instructions:
                        - Provide a comprehensive answer that addresses the query
                        - Cite sources in the format [1], [2], etc. when referencing information
                        - If sources disagree, mention the different viewpoints
                        - If the answer isn't in the provided documents, say "I don't have enough information"
                        - Be direct and informative

                        Answer:"""
            response = query_llm(request)
            print("Search Results:")
            for r in results:
                print(f"- {r["document"]["title"]}")
            print("LLM Answer:")
            print(response.strip())
            
        case "question":
            question = args.question 

            movies = load_movies()
            hybrid_search = HybridSearch(movies)
            results=  hybrid_search.rrf_search(args.question, RRF_K, args.limit)

            request = prompt = f"""Answer the user's question based on the provided movies that are available on Webflyx, a streaming service.

                        Question: {question}

                        Documents:
                        {results}

                        Instructions:
                        - Answer questions directly and concisely
                        - Be casual and conversational
                        - Don't be cringe or hype-y
                        - Talk like a normal person would in a chat conversation

                        Answer:"""

            response = query_llm( request )

            print("Search Results:")
            for result in results:
                print(f"- {result["document"]["title"]}")
            print("Answer:", response.strip())


        case _:
            parser.print_help()

if __name__ == "__main__":
    main()