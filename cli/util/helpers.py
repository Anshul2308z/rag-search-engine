import json, string
from util import InvertedIndex




def load_movies():
    with open("data/movies.json", "r") as f:
        movies = json.load(f)["movies"]  # Load the movies data from the JSON file
        return movies 

def tokenize_text(text: str) -> list:
    text = removePuntuation(text)
    tokens = text.lower().split()
    return tokens
    

def removePuntuation(text: str) -> str:
    translator = str.maketrans("", "", string.punctuation)
    return text.translate(translator)

def build_command():
    inverted_Index = InvertedIndex()

    inverted_Index.build()
    inverted_Index.save()


