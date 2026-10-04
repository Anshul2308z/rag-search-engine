import string
from nltk.stem import PorterStemmer
import os
import pickle
from collections import Counter 
import math 
from lib.search_utils import SearchResult


from .search_utils import (
    BM25_B,
    BM25_K1,
    CACHE_DIR,
    DEFAULT_SEARCH_LIMIT,
    STOPWORDS_PATH,
    load_movies,
)

def tokenize_text(text: str) -> list[str]:
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


def tokenizeTerm( term ):
    tokenizedTerm =  tokenize_text(term)
    if len(tokenizedTerm) != 1 :
        raise Exception("Term must tokenize to exactly one token")
    return tokenizedTerm[0]

#this class is the most clutch thing in this repo! 

# basically we are using inverted index- in this for all the tokens we have we'll map them to the documents in which they exits.
# So you'll have something like- index1 -> {docid1, docid2...}
# for this we are using self.index 

class InvertedIndex:
    def __init__(self) : # the constructor 
        self.index = {} # used to map token -> id (so each token has a set of form where ti came from ), struct-> token -> (doc1, doc2 etc etc)
        self.docmap = {} # doc ids -> full doc obj ??
        self.term_frequencies = {} # dictionary mapping docIds to counter objs 
        self.doc_lengths = {} 
        self.doc_lengths_path = os.path.join(CACHE_DIR, "doc_lengths.pkl")
        self.index_path = os.path.join(CACHE_DIR, "index.pkl")
        with open(STOPWORDS_PATH, "r") as stopwords_file:
            self.stopwords = set(tokenize_text(stopwords_file.read()))


    def __add_document( self, doc_id, text ): # we are for the docId putting it in the index- basically creatin thisTOken -> it's doc id added to the set. 
        tokens = tokenize_text(text)
        stemmer = PorterStemmer()
        stemmed_tokens = [
            stemmer.stem(token) for token in tokens if token not in self.stopwords
        ] #stemming things here as term_frequencies also need them. 

        self.term_frequencies[doc_id] = Counter(stemmed_tokens) # here is where we are creating term freq of all tokens within the doc Id 

        doc_length = 0 
        for token in stemmed_tokens:
            doc_length +=1 
            if token in self.index: 
                self.index[token].add(doc_id)
            else: 
                self.index[token] = set()
                self.index[token].add(doc_id)
        self.doc_lengths[doc_id] = doc_length

    def get_documents(self, term): # returning all doc ids for the token ( doc ids are literaly in a set that is value to the key token)
        toReturn = [] 
        if term in self.index: 
            toReturn.extend(list(self.index[term]))
        toReturn.sort() # sorting because it was an instruction from the program ( so ids will be in ascending order)
        return toReturn 

    def build(self): # this is a manual labour of looking through a dict called movies in the method ( look at load_movies to find how it is extracted from json file )
        movies = load_movies() # in docmap we are building full doc objects. So all docId key will have it's docObj as it's value 
        for m in movies:  
            self.docmap[m["id"]] = m 
            text = f"{m['title']} {m['description']}"
            self.__add_document(m["id"], text ) # for each movie we are also calling this function to do indexing of all the doc's tokens

    #this is where we do caching, instead of everytime running build we have actually built a cache by running build and then cached it using the save method and now we simply retrieve it everytime we need it using load. 

    def save(self): 
        with open(os.path.join(CACHE_DIR, "index.pkl"), 'wb') as f: # wb-> write binary, similarly rb-> read binary
            pickle.dump(self.index, f)
        with open(os.path.join(CACHE_DIR, "docmap.pkl"), 'wb') as g:
            pickle.dump(self.docmap, g)   
        with open(os.path.join(CACHE_DIR, "term_frequencies.pkl"), 'wb') as t :
            pickle.dump(self.term_frequencies, t)
        with open(os.path.join(CACHE_DIR, "doc_lengths.pkl"), 'wb') as d:
            pickle.dump(self.doc_lengths, d)

    def load (self):

        with open(os.path.join(CACHE_DIR, "index.pkl"), 'rb') as f :
            self.index = pickle.load(f)
        with open(os.path.join(CACHE_DIR, "docmap.pkl"), 'rb') as g:
            self.docmap = pickle.load(g)     
        with open(os.path.join(CACHE_DIR, "term_frequencies.pkl"), 'rb') as t:
            self.term_frequencies = pickle.load(t)
        with open(os.path.join(CACHE_DIR, "doc_lengths.pkl"), 'rb') as d:
            self.doc_lengths= pickle.load(d)

    # a simple method that searches docmap for an id and returns docObj ( value of docId key) if it exists
    def get_doc_obj(self, docId):
        if docId in self.docmap:
            return self.docmap[docId]
        return None

    # term freq is how many times a term appears in our document. 
    def get_tf( self, doc_id, term):

        if term in self.term_frequencies[doc_id]:
            return self.term_frequencies[doc_id][term]
        return 0 #actually that thing above already returns zero if token is not in the dict 

    def  get_bm25_idf(self, term: str) -> float:
        #df = document freq-> how many docs contain this term. 
        df = len(self.index.get(term, [])) # self.index.get(term) -> returns arr associated with the key that is term. Then if it doesn't give anything you return []. then len wrapping. 
        N = len(self.docmap)

        BMI = math.log((N - df + 0.5) / (df + 0.5) + 1)

        return BMI 

    def get_bm25_tf(self, doc_id, term, k1=BM25_K1, b= BM25_B):
        
        tf = self.get_tf(doc_id, term) 

        avg_doc_length = self.__get_avg_doc_length()
        doc_length = self.doc_lengths[doc_id]

        # Length normalization factor
        length_norm = 1 - b + b * (doc_length / avg_doc_length)

        saturated_term_freq = (tf * (k1 + 1)) / (tf + k1* length_norm) #saturated tf using length_norm 

        return saturated_term_freq
    
    def  __get_avg_doc_length(self) -> float: 
        if not self.doc_lengths:
            return 0.0
        average_doc_length = sum(self.doc_lengths.values()) / len(self.doc_lengths)

        return average_doc_length

    def bm25(self, doc_id, term):
        tf = self.get_bm25_tf(doc_id, term)
        idf = self.get_bm25_idf(term)

        return tf * idf  


    def bm25_search(self, query, limit=DEFAULT_SEARCH_LIMIT)-> list[SearchResult]:
        tokens = tokenize_text(query)
        stemmer = PorterStemmer()
        stemmed_tokens = [stemmer.stem(token) for token in tokens]

        scores = {}

        for token in stemmed_tokens:
            if token in self.index:
                for doc_id in self.index[token]:
                    if doc_id in scores:
                        scores[doc_id] += self.bm25(doc_id, token)
                    else:
                        scores[doc_id] = self.bm25(doc_id, token)

        sorted_docs = sorted(
            scores.items(),
            key=lambda item: item[1],
            reverse=True
        )

        return [
            {
                "id": doc_id,
                "title": self.docmap[doc_id]["title"],
                "document": self.docmap[doc_id]["description"][:100],
                "score": round(score, 3),
                "metadata": {},
            }
            for doc_id, score in sorted_docs[:limit]
        ]