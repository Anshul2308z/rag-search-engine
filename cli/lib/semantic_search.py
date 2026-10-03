from sentence_transformers import SentenceTransformer
import numpy as np 
from pathlib import Path 
import json 
import re
try:
    from cli.lib.search_utils import format_search_result, Movie
except ModuleNotFoundError:
    from lib.search_utils import format_search_result, Movie


def semantic_chunk(text: str, size: int, overlap: int) -> list[str]:
    if size <= 0:
        raise ValueError("Chunk size must be greater than zero")

    if overlap < 0 or overlap >= size:
        raise ValueError("Overlap must be non-negative and smaller than size")

    text = text.strip()

    if text== "":
        return []

    sentences = [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", text)
        if sentence.strip()
    ]


    if len(sentences) == 1 and not sentences[0].endswith(
        (".", "!", "?")
    ): 
        return sentences

    chunks = []
    step = size - overlap
    start = 0

    while start < len(sentences):
        chunk = sentences[start:start + size]

        # Don't keep a final chunk that's only overlap
        if chunks and len(chunk) <= overlap:
            break

        chunks.append(" ".join(chunk))
        start += step

    return chunks

# helpers 
def verify_model():
    semantic_search = SemanticSearch()
    print(f"Model loaded: {semantic_search.model}")
    print(f"Max sequence length: {semantic_search.model.max_seq_length}")

def embed_text(text):
    semantic_search = SemanticSearch()
    embedding=  semantic_search.generate_embedding(text)

    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")


def verify_embeddings():
    semantic_search = SemanticSearch()

    documents = []
    with open("data/movies.json", "r") as f:
        documents = json.load(f)["movies"]

    embeddings = semantic_search.load_or_create_embeddings(documents)

    print(f"Number of docs:   {len(documents)}")
    print(
        f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions"
    )

def embed_query_text(query): 
    semantic_search = SemanticSearch()
    embedding = semantic_search.generate_embedding(query)

    print(f"Query: {query}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Shape: {embedding.shape}")

def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float: # ndarry -> n dimensional array 
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1) # search about this later
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0   

    return dot_product / (norm1 * norm2)

#SemanticSearch class 
class SemanticSearch:
    def __init__(self, model_name = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.embeddings = None 
        self.documents = None 
        self.document_map = {} 

    def generate_embedding(self, text): 
        if not text.strip():
            raise ValueError("Provided text is either an empty string or only contains whitespaces")
        outputs = self.model.encode([text]) # encode expects a list of tokens, so [text] instead of text 
        return outputs[0]

    def build_embeddings(self, documents):
        self.documents = documents  
        movie_rep = []
        for doc in documents :
            self.document_map[doc["id"]] = doc 
            movie_rep.append(f"{doc['title']}: {doc['description']}")

        embeddings = self.model.encode(movie_rep, show_progress_bar=True)
        self.embeddings = embeddings

        np.save("cache/movie_embeddings.npy", self.embeddings)

        return self.embeddings

    def load_or_create_embeddings(self, documents):
        
        self.documents = documents
        
        for doc in documents :
            self.document_map[doc["id"]] = doc 

        if Path("cache/movie_embeddings.npy").exists():
            self.embeddings = np.load("cache/movie_embeddings.npy")
            if len(self.embeddings) == len(documents):
                return self.embeddings
        
        return self.build_embeddings(documents)

    def search(self, query, limit):
        if ( self.embeddings is None or self.documents is None):
            raise ValueError("No embeddings loaded. Call `load_or_create_embeddings` first.")
        embedding = self.generate_embedding(query)
        outputs= []
        for vec, doc in zip(self.embeddings, self.documents): 
            similarity_score = cosine_similarity(vec,embedding)
            outputs.append(
                (similarity_score, doc)
            )
        outputs.sort(key=lambda x: x[0], reverse=True)
        outputs = outputs[:limit]
        result= []
        for output in outputs: 
            result.append(
                {
                    "score": output[0],
                    "title": output[1]["title"],
                    "description": output[1]["description"]
                }
            )
        return result

#ChunkedSemanticSearch class that inherits from SemanticSearch

class ChunkedSemanticSearch(SemanticSearch):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        super().__init__(model_name)
        self.chunk_embeddings = None
        self.chunk_metadata = None

    def build_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:

        self.documents = documents 

        for doc in documents : # thinking about merging it with for loop beneath 
            self.document_map[doc["id"]] = doc 
        
        all_chunks:list[str] = []
        chunk_metadata: list[dict] = []

        for movieIndex,doc in enumerate(documents): 

            if doc.get("description", "") == "":
                continue
            chunks = semantic_chunk(doc["description"], 4, 1)
            totalChunks = len(chunks)

            for chunkIndex,chunk in enumerate(chunks) : 
                
                all_chunks.append(chunk)

                chunk_metadata.append(
                    {
                        "movie_idx": movieIndex,
                        "chunk_idx": chunkIndex,
                        "total_chunks": totalChunks
                    }
                )

        self.chunk_embeddings = self.model.encode(all_chunks, show_progress_bar=True)
        self.chunk_metadata = chunk_metadata


        np.save("cache/chunk_embeddings.npy", self.chunk_embeddings)

        with open("cache/chunk_metadata.json", "w") as f:
            json.dump({"chunks": chunk_metadata, "total_chunks": len(all_chunks)}, f, indent=2)

        return self.chunk_embeddings
        

    def load_or_create_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:
        self.documents = documents 

        for doc in documents:
            self.document_map[doc["id"]] = doc

        if Path("cache/chunk_embeddings.npy").exists() and Path("cache/chunk_metadata.json").exists():

            self.chunk_embeddings = np.load("cache/chunk_embeddings.npy")
            with open("cache/chunk_metadata.json", "r") as f:
                self.chunk_metadata = json.load(f)["chunks"]

            return self.chunk_embeddings

        else: 
            return self.build_chunk_embeddings(documents)

    def search_chunks(self, query: str, limit: int = 10):

        if self.chunk_embeddings is None or self.chunk_metadata is None:
            raise Exception("load the chunks or metadata correctly!")

        embedding = self.generate_embedding(query)
        chunk_scores = [] 

        for chunk, metadata in zip(self.chunk_embeddings, self.chunk_metadata):
            similarity =  cosine_similarity(chunk, embedding)
            chunk_scores.append(
                {
                    "chunk_idx": metadata["chunk_idx"],
                    "movie_idx": metadata["movie_idx"],
                    "score": similarity
                }
            )

        movies_chunk_scores = {}

        for c in chunk_scores:
            movie_idx = c["movie_idx"]

            if movie_idx not in movies_chunk_scores:
                movies_chunk_scores[movie_idx] = c["score"]

            elif c["score"] > movies_chunk_scores[movie_idx]:
                movies_chunk_scores[movie_idx] = c["score"]

        ranked_movies = sorted(
            movies_chunk_scores.items(),
            key=lambda item: item[1],
            reverse=True,
        )[:limit]

        results =[]

        
        for movie_idx, score in ranked_movies:
            doc = self.documents[movie_idx]
            title = doc["title"]
            document = doc["description"][:100]

            results.append(
                format_search_result(doc["id"], title, document, score)
            )
        return results

