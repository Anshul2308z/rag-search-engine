import argparse
import json

from os import read
import string 

from nltk.stem import PorterStemmer

class InvertedIndex:
    def __init__(self) :
        self.index = {} # used to map token -> id (so each token has a set of form where ti came from ), struct-> token -> (doc1, doc2 etc etc)
        self.docmap = {} # doc ids -> full doc obj ?? 

    def __get_documents( self, doc_id, text ):
        tokens = tokenize_text(text)
        for token in tokens:
            if token in self.index: 
                self.index[token].add(doc_id)
            else: 
                self.index[token] = set()
                self.index[token].add(doc_id)
    def get_documents(self, term):
        toReturn = [] 
        if term in self.index: 
            toReturn.extend(list(self.index[term]))
        toReturn.sort()
        return toReturn 
        
            
def load_movies():
    with open("data/movies.json", "r") as f:
        movies = json.load(f)["movies"]  # Load the movies data from the JSON file

def tokenize_text(text: str) -> list:
    text = removePuntuation(text)
    tokens = text.lower().split()
    return tokens
    

def removePuntuation(text: str) -> str:
    translator = str.maketrans("", "", string.punctuation)
    return text.translate(translator)


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using keywords")
    search_parser.add_argument("query", type=str, help="Search query")

    args = parser.parse_args()

    match args.command:
        case "search":
            print(f"Searching for: {args.query}")
    
            # query = removePuntuation(args.query)  # Remove punctuation from the query
            # tokens = query.lower().split()  # Tokenize the query into lowercase words

            tokens = tokenize_text(args.query)  # Tokenize the query using the tokenize_text function
            
            

            with open("data/stopwords.txt", "r") as f:
                stopwords = f.read()
                stopwords = tokenize_text(stopwords)  # Tokenize the stopwords using the tokenize_text function

            stemmer = PorterStemmer()

            finalTokens = []

            for token in tokens: 
                if token in stopwords:
                    continue  # Skip the token if it's a stopword
                else: 
                    finalTokens.append(stemmer.stem(token))  # Stem the token and add it to finalTokens

            search_results = []
            for movie in movies:
               for token in finalTokens:
                   if token in removePuntuation(movie['title']).lower():
                       search_results.append(movie)
                       break  # Stop checking other tokens if a match is found
            search_results = search_results[:5]  # Limit to top 5 results

            for k, result in enumerate(search_results):
                print(f"{k+1}. {result['title']}")

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()

