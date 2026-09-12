import argparse
import json

from os import read
import string 

from nltk.stem import PorterStemmer

import pickle

# def load_cache_files():
#     with open("cache/index.pkl", "rb") as f:
#         index = pickle.load(f)
#     with open("cache/docmap.pkl", "rb") as g:
#         docmap = pickle.load(g)
#     return index, docmap

class InvertedIndex:
    def __init__(self) :
        self.index = {} # used to map token -> id (so each token has a set of form where ti came from ), struct-> token -> (doc1, doc2 etc etc)
        self.docmap = {} # doc ids -> full doc obj ?? 

    def __add_document( self, doc_id, text ):
        tokens = tokenize_text(text)
        stemmer = PorterStemmer()
        for token in tokens:
            token = stemmer.stem(token)
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

    def build(self):
        movies = load_movies()
        for m in movies: 
            self.docmap[m["id"]] = m
            text = f"{m['title']} {m['description']}"
            self.__add_document(m["id"], text )

    def save(self): 
        with open('cache/index.pkl', 'wb') as f: # wb-> write binary, similarly rb-> read binary
            pickle.dump(self.index, f)
        with open('cache/docmap.pkl', 'wb') as g:
            pickle.dump(self.docmap, g)   

    def load (self):

        with open('cache/index.pkl', 'rb') as f :
            self.index = pickle.load(f)
        with open('cache/docmap.pkl', 'rb') as g :
            self.docmap = pickle.load(g)     

    def get_doc_obj(self, docId):
        if docId in self.docmap:
            return self.docmap[docId]
        return None
            

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


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using keywords")
    build_parser = subparsers.add_parser("build", help="do invert indexing of the db and save to cache dir")
    search_parser.add_argument("query", type=str, help="Search query")

    args = parser.parse_args()




    match args.command:
        case "search":
            print(f"Searching for: {args.query}")
    
            # query = removePuntuation(args.query)  # Remove punctuation from the query
            # tokens = query.lower().split()  # Tokenize the query into lowercase words

            tokens = tokenize_text(args.query)  # Tokenize the query using the tokenize_text function
            
            with open("data/stopwords.txt", "r") as f: #r -> read 
                stopwords = f.read()
                stopwords = tokenize_text(stopwords)  # Tokenize the stopwords using the tokenize_text function
            
            stemmer = PorterStemmer()

            finalTokens = []

            for token in tokens: 
                if token in stopwords:
                    continue  # Skip the token if it's a stopword
                else: 
                    finalTokens.append(stemmer.stem(token))  # Stem the token and add it to finalTokens

            inverted_index = InvertedIndex()

            try:
                inverted_index.load()
            except Exception as e :
                print("Cannot load index for inverted indexing")
                exit


            search_results = []

            for token in finalTokens:
                docIds = inverted_index.get_documents(token)
                if  docIds != []:
                    for id in docIds:
                        doc = inverted_index.get_doc_obj(id)
                        if doc == None:
                            continue
                        search_results.append(doc["title"])
                        if len(search_results) == 5:
                            break 
            
            for title in search_results:
                print(title)
            # for k, result in enumerate(search_results):
            #     print(f"{k+1}. {result['title']}")


        case "build": 
            build_command()
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()

