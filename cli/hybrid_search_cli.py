import argparse
from lib.hybrid_search import HybridSearch
from lib.search_utils import load_movies
from llm_integration import enhance_query, enhancement_result, enhance_rewriter, enhance_expand, rerank_score

import time

def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")

    sub_parsers = parser.add_subparsers(dest="command", help="Available commands")

    normalize_parser = sub_parsers.add_parser("normalize", help="Normalize documents")
    normalize_parser.add_argument( "scores", nargs="*", type=float, help="List of scores to normalize")

    weighted_search = sub_parsers.add_parser("weighted-search", help="weighted hybrid serach")

    weighted_search.add_argument("query", type=str, help="type query")
    weighted_search.add_argument("--alpha", type=float, default=0.5, help="a constant that we can use to dynamically control the weighting")
    weighted_search.add_argument("--limit", type=int, default=5, help="limit results")

    rrf_search_parser = sub_parsers.add_parser("rrf-search", help="Reciprocal Rank Fusion search" )
    rrf_search_parser.add_argument("query", type=str, help="type query")
    rrf_search_parser.add_argument("-k", type=int, default=60)
    rrf_search_parser.add_argument("--limit", type=int, default=5)
    rrf_search_parser.add_argument("--enhance", type=str, choices=["spell", "rewrite", "expand"], help="Query enhnacement method")
    rrf_search_parser.add_argument("--rerank-method", type=str, choices=["individual"], help="Reranking method- indivisual,")


    args = parser.parse_args()

    match args.command:
        case "normalize":
            scores = args.scores
            normalized_scores = []

            scores = sorted(scores)

            if len(scores) == 0:
                return

            if scores[0] == scores[-1]: 
                normalized_scores =  [1.0] * len(scores)
            else: 
                min_score = scores[0]
                max_score = scores[-1]

                for score in scores : 
                    normalized_scores.append(
                        (score - min_score) / (max_score- min_score )
                    )
            for score in normalized_scores:
                print(f"{score:.4f}")

        case "weighted-search":
            movies = load_movies()
            hybrid_search = HybridSearch(movies)

            results = hybrid_search.weighted_search(args.query, args.alpha, args.limit )


            for i, result in enumerate(results): 

                d= hybrid_search.semantic_search.document_map[result["id"]]
                
                print(f"{i+1}. {d["title"]}")
                print(f"  Hybrid Score: {result["hybrid_score"]:.3f}")
                print(f"  BM25: {result["bm25_score"]:.3f}, Semantic: {result["semantic_score"]:.3f}")
                print(f"  {d["description"][:30]}...")

        case "rrf-search": 
            movies = load_movies()

            hybrid_search = HybridSearch(movies)

            query = args.query
            k = args.k 
            limit = args.limit

            enhance = args.enhance 
            rerank_method= args.rerank_method

            if ( rerank_method == "indivisual"):
                limit = 5 * limit

            if enhance == "spell": 
                enhanced = enhance_query(query)
            elif enhance == "rewrite":
                enhanced = enhance_rewriter(query)
            elif enhance == "expand": 
                enhanced = enhance_expand(query)
            else: 
                enhanced = ""

            if enhanced == "":
                results = hybrid_search.rrf_search(query, k, limit)
            else: 
                enhancement_result( enhance, query, enhanced)
                results = hybrid_search.rrf_search(enhanced, k, limit)


            if rerank_method == "individual":
                for result in results: 
                    score = rerank_score(query, result["document"])
                    if score : 
                        print("rerank-score for the query was", score)
                        result["rerank_score"] = score 
                    time.sleep(3)
                        

                results = sorted(results, key= lambda x: x["rerank_score"], reverse=True)
            
            
            for i, result in enumerate(results):
                print(f"{i+1}. {result["document"]["title"]}")
                if result.get("rerank_score") != None:
                    print(f"Re-rank Score: {result["rerank_score"]:.3f}")
                print(f"  RRF Score: {result["rrf_score"]:.3f}")
                print(f"  BM25 Rank: {result["bm25_rank"]}, Semantic Rank: {result["semantic_rank"]}")
                print(f"  {result["document"]["description"][:50]}...")


            
        case _:
            parser.print_help()



if __name__ == "__main__":
    main()