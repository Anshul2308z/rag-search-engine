from PIL import Image 
from lib.search_utils import load_movies
from sentence_transformers import SentenceTransformer


class Multimodal:
    def __init__(self, docs, model_name="clip-ViT-B-32"):
        self.model = SentenceTransformer(model_name)
        self.docs = docs 
        self.texts = [ f"{doc['title']}: {doc['description']}" for doc in docs]
        self.text_embeddings = self.model.encode(self.texts, show_progress_bar=True)


    def embed_image(self, image_path):
        img = Image.open(image_path).convert("RGB")
        embedding = self.model.encode([img])[0]
        return embedding

    def search_with_image(self, img_path ):
        image_embedding = self.embed_image(image_path=img_path)

        results = [] 
        for v, doc in zip(self.text_embeddings, self.docs):
            similarity = self.model.similarity(v, image_embedding).item()
            results.append(
                {
                    "id": doc["id"],
                    "title": doc["title"],
                    "description": doc["description"],
                    "similarity_score": similarity
                }
            )
        results = sorted( results, key= lambda x : x["similarity_score"], reverse=True)[:5]
        return results



def verify_image_embedding( img_path ):
    #doing it so that it satisfies our constructor conditions
    movies = load_movies()
    multimodal = Multimodal(movies) # here
    embedding = multimodal.embed_image(image_path=img_path)
    print(f"Embedding shape: {embedding.shape[0]} dimensions")

def image_search_command(img_path):

    movies = load_movies()
    multimodal = Multimodal(movies)

    results = multimodal.search_with_image(img_path=img_path)
    return results 
