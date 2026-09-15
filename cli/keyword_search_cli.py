import argparse
from nltk.stem import PorterStemmer

from util.InvertedIndex import InvertedIndex 
from util.helpers import tokenize_text, build_command, tokenizeTerm


def tf( doc_id, term):
    term = tokenizeTerm(term)
    inverted_index = InvertedIndex()
    inverted_index.load()
    val = inverted_index.get_tf(doc_id, term)
    if val > 0:
        print(val)
    else: 
        print(0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using keywords")
    build_parser = subparsers.add_parser("build", help="do invert indexing of the db and save to cache dir")
    search_parser.add_argument("query", type=str, help="Search query")

    tf_parser = subparsers.add_parser("tf", help="Provide doc_id and a term to find it's freq in that doc!")
    tf_parser.add_argument("doc_id", type=int, help="Document ID")
    tf_parser.add_argument("term", type=str, help="Term to find frequency for")

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
            tf(args.doc_id, args.term)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()

