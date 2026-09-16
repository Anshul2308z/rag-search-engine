import argparse
from nltk.stem import PorterStemmer

from util.InvertedIndex import InvertedIndex 
from util.helpers import tokenize_text, build_command, tokenizeTerm

import math

def idf(term):
    inverted_index= InvertedIndex()
    inverted_index.load()
    stemmer = PorterStemmer()
    Tokenizedterm = stemmer.stem(tokenizeTerm(term))

    term_match_doc_count = 0 #also known as df ( document freq ) 

    for docId in inverted_index.docmap:
        if inverted_index.get_tf(docId, Tokenizedterm) > 0 :
            term_match_doc_count += 1

    total_doc_count = len(inverted_index.docmap)
    idf = math.log((total_doc_count + 1) / (term_match_doc_count + 1))

    return idf 

def tf( doc_id, term):
    term = tokenizeTerm(term)
    inverted_index = InvertedIndex()
    inverted_index.load()
    val = inverted_index.get_tf(doc_id, term)
    if val > 0:
        return val
    else: 
        return 0 
    

def tfidf(docId, term):
    inverted_index = InvertedIndex()
    inverted_index.load()
    term_freq = tf(docId, term)
    inverse_doc_freq = idf(term)

    tfidf = term_freq * inverse_doc_freq
    print(f"TF-IDF score of '{term}' in document '{docId}': {tfidf:.2f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using keywords")
    build_parser = subparsers.add_parser("build", help="do invert indexing of the db and save to cache dir")
    search_parser.add_argument("query", type=str, help="Search query")

    tf_parser = subparsers.add_parser("tf", help="Provide doc_id and a term to find it's freq in that doc!")
    tf_parser.add_argument("doc_id", type=int, help="Document ID")
    tf_parser.add_argument("term", type=str, help="Term to find frequency for")

    idf_parser = subparsers.add_parser("idf", help="Enter a term to retrieve it's Invertse document freq")
    idf_parser.add_argument("term", type=str,help="Term to find inverse doc freq" )

    tfidf_parser = subparsers.add_parser("tfidf", help="A value that scores terms based on rarity accross the docs and freq within a single doc")
    tfidf_parser.add_argument("docId", type=int, help="Enter Doc Id")
    tfidf_parser.add_argument("term", type=str, help="Enter a term")

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

            flag = False
            seen = set()
            for token in finalTokens:
                if flag == True: 
                    break 
                docIds = inverted_index.get_documents(token)

                if  docIds != []:
    
                    for id in docIds:
                        if id in seen:
                            continue
                        seen.add(id)
                        doc = inverted_index.get_doc_obj(id)
                        if doc == None:
                            continue
                        search_results.append(doc["title"])
                        if len(search_results) == 5:
                            flag = True
                            break
            
            for title in search_results:
                print(title)
            # for k, result in enumerate(search_results):
            #     print(f"{k+1}. {result['title']}")


        case "build": 
            build_command()
        case "tf":
            term_freq= tf(args.doc_id, args.term)
            print(args.term + " appeared " + term_freq + "times in the provided doc")

        case "idf":
            inverse_term_freq= idf(args.term)
            print(f"Inverse document frequency of '{args.term}': {inverse_term_freq:.2f}")


        case "tfidf":
            tfidf(args.docId, args.term)

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()

