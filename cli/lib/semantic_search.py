from sentence_transformers import SentenceTransformer
import numpy as np 
from pathlib import Path 
import json 

# helpers 
def verify_model():
    semantic_search = semanticSearch()
    print(f"Model loaded: {semantic_search.model}")
    print(f"Max sequence length: {semantic_search.model.max_seq_length}")

def embed_text(text):
    semantic_search = semanticSearch()
    embedding=  semantic_search.generate_embedding(text)

    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")


def verify_embeddings():
    semantic_search = semanticSearch()

    documents = []
    with open("data/movies.json", "r") as f:
        documents = json.load(f)["movies"]

    embeddings = semantic_search.load_or_create_embeddings(documents)

    print(f"Number of docs:   {len(documents)}")
    print(
        f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions"
    )

def embed_query_text(query): 
    semantic_search = semanticSearch()
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

#semanticSearch class 
class semanticSearch:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
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

