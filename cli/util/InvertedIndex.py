from util.helpers import tokenize_text, load_movies
from nltk.stem import PorterStemmer
import pickle

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

