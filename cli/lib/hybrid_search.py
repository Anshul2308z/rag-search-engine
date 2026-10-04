import os

from .keyword_search import InvertedIndex
from .semantic_search import ChunkedSemanticSearch
from .search_utils import SearchResult


class HybridSearch:
    def __init__(self, documents: list[dict]) -> None:
        self.documents = documents
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)

        self.idx = InvertedIndex()
        if not os.path.exists(self.idx.index_path):
            self.idx.build()
            self.idx.save()

    def _bm25_search(self, query: str, limit: int) -> list[SearchResult]:
        self.idx.load()
        return self.idx.bm25_search(query, limit)

    def weighted_search(self, query: str, alpha: float, limit: int = 5) -> list[dict]:

        bm25 = self._bm25_search(query, 500 * limit)
        semantic= self.semantic_search.search_chunks(query, 500 * limit)

        normalized_bm25 = normalize(bm25)
        normalized_semantic = normalize(semantic)

        results = {}

        for result in normalized_bm25: 
            results[result["id"]] = {
                "document": self.semantic_search.document_map[result["id"]],
                "bm25_score": result["score"],
                "semantic_score": 0.0
                  }

        for result in normalized_semantic: 
            if result["id"] in results: 
                results[result["id"]]["semantic_score"] = result["score"]
            else: 
                results[result["id"]] = {
                    "document": self.semantic_search.document_map[result["id"]],
                    "semantic_score": result["score"],
                    "bm25_score": 0.0
                    }

        resultList = []

        for doc_id, result in results.items():
            resultList.append({
                "id": doc_id,
                "document" : result["document"],
                "semantic_score": result["semantic_score"],
                "bm25_score": result["bm25_score"],
                "hybrid_score": (alpha * result["bm25_score"]) + ((1 - alpha) * result["semantic_score"])
            })


        resultList = sorted(resultList, key=lambda x: x["hybrid_score"], reverse=True)
        return resultList[:limit]

        
    def rrf_search(self, query: str, k: int, limit: int = 10) -> list[dict]:

        bm25 = self._bm25_search(query, 500 * limit)
        semantic = self.semantic_search.search_chunks(query, limit * 500)

        '''
        id: int,
        title: str,
        document: str,
        score: float,
        '''

        results = {}

        for pos, result in enumerate(bm25): 
            document = self.semantic_search.document_map[result["id"]]
            results[result["id"]] = {
                "document": document,
                "bm25_rank": pos + 1, 
                "semantic_rank": None, 
            }

        for pos, result in enumerate(semantic):
            if result["id"] in results: 
                results[result["id"]]["semantic_rank"] = pos + 1 
            else: 
                document = self.semantic_search.document_map[result["id"]]
                results[result["id"]] = {
                    "document": document,
                    "bm25_rank": None,
                    "semantic_rank": pos + 1 
                }

        resultList = []

        for doc_id, result in results.items():
            bm25_rank = result["bm25_rank"]
            semantic_rank = result["semantic_rank"]

            score =  0

            if type(bm25_rank) == int: 
                score += rrf_score(bm25_rank, k)

            if type(semantic_rank) == int: 
                score += rrf_score(semantic_rank, k)

                # rrf_score is not added to the results dict as it is directly appended to the list

            resultList.append({
                "id": doc_id,
                "document": result["document"],
                "bm25_rank": result["bm25_rank"],
                "semantic_rank": result["semantic_rank"],
                "rrf_score": score
            })

        #sorting by rrf_score
        resultList = sorted(resultList, key=lambda x: x["rrf_score"], reverse=True)

        return resultList[:limit]



def normalize ( results: list[SearchResult]):
    if len(results) == 0 :
        return results

    if results[0]["score"] == results[-1]["score"]:
        for result in results: 
            result["score"] = 1.0
        return results

    else: 
        min_score = results[-1]["score"]
        max_score = results[0]["score"]
        for result in results :
            result["score"] = ( result["score"] - min_score ) / ( max_score - min_score) 
        return results 

def rrf_score( rank: int, k:int = 60):
    return 1/( rank + k )
