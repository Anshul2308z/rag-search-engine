import os

from .keyword_search import InvertedIndex
from .semantic_search import ChunkedSemanticSearch


class HybridSearch:
    def __init__(self, documents: list[dict]) -> None:
        self.documents = documents
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)

        self.idx = InvertedIndex()
        if not os.path.exists(self.idx.index_path):
            self.idx.build()
            self.idx.save()

    def _bm25_search(self, query: str, limit: int) -> list[tuple[str, float]]:
        self.idx.load()
        return self.idx.bm25_search(query, limit)

    def weighted_search(self, query: str, alpha: float, limit: int = 5) -> list[dict]:
        bm25 = self._bm25_search(query, 500 * limit)
        semantic= self.semantic_search.search_chunks(query, 500 * limit)

        normalized_bm25 = normalize(bm25)


        
    def rrf_search(self, query: str, k: int, limit: int = 10) -> list[dict]:
        raise NotImplementedError("RRF hybrid search is not implemented yet.")


def normalize ( scores: list[float]):
    scores = sorted(scores)

    if scores[0]:
        return []

    if scores[0] == scores[-1]:
        normalized_scores = [ 1.0 ] * len(scores)
    else: 
        min_score = scores[0]
        max_score = scores[-1]
        normalized_scores = [(score - min_score)/ (max_score-min_score) for score in scores]

    return normalized_scores