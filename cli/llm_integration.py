from dotenv import load_dotenv
import os
from openai import OpenAI
from lib.search_utils import Movie
from openai.types.chat import ChatCompletionMessageParam


def query_llm( query: str) -> str:
    load_dotenv()
    api_key = os.environ.get("API_KEY")
    if not api_key:
        raise RuntimeError("API_KEY env variable not set")

    client = OpenAI(
        base_url= "https://openrouter.ai/api/v1",
        api_key= api_key
    )

    messages: list[ChatCompletionMessageParam] = [
        {
            "role": "user",
            "content": query
        }
    ]
    try: 
        responseObj = client.chat.completions.create(
            model="apodex/apodex-1.1-mini:free",
            messages =messages
        )

        response = responseObj.choices[0].message.content
        if response is None: 
            return ""
        return response 
    
    except Exception as e: 
        print("Error encountered: ", e)
        return ""
    

def enhancement_result( method: str, query: str, enhanced_query: str): 
    if query != enhance_query: 
        print( f"Enhanced query ({method}): '{query}' -> '{enhanced_query}'\n" ) 

def enhance_query(query):

    request = f"""Fix any spelling errors in the user-provided movie search query below.
                Correct only clear, high-confidence typos. Do not rewrite, add, remove, or reorder words.
                Preserve punctuation and capitalization unless a change is required for a typo fix.
                If there are no spelling errors, or if you're unsure, output the original query unchanged.
                Output only the final query text, nothing else.
                User query: \"{query}\"
                """

    response= query_llm( request )

    if response == "":
        return query 

    return response 

def enhance_rewriter( query: str):

    request = f"""Rewrite the user-provided movie search query below to be more specific and searchable.
                    Consider:
                    - Common movie knowledge (famous actors, popular films)
                    - Genre conventions (horror = scary, animation = cartoon)
                    - Keep the rewritten query concise (under 10 words)
                    - It should be a Google-style search query, specific enough to yield relevant results
                    - Don't use boolean logic

                    Examples:
                    - "that bear movie where leo gets attacked" -> "The Revenant Leonardo DiCaprio bear attack"
                    - "movie about bear in london with marmalade" -> "Paddington London marmalade"
                    - "scary movie with bear from few years ago" -> "bear horror movie 2015-2020"

                    If you cannot improve the query, output the original unchanged.
                    Output only the rewritten query text, nothing else.

                    User query: \"{query}\"
                    """

    response = query_llm( request )

    if response == "": 
        return query 

    return response
    
    
def enhance_expand( query: str ):

    request = f"""Expand the user-provided movie search query below with related terms.

                    Add synonyms and related concepts that might appear in movie descriptions.
                    Keep expansions relevant and focused.
                    Output only the additional terms; they will be appended to the original query.

                    Examples:
                    - "scary bear movie" -> "scary horror grizzly bear movie terrifying film"
                    - "action movie with bear" -> "action thriller bear chase fight adventure"
                    - "comedy with bear" -> "comedy funny bear humor lighthearted"

                    User query: \"{query}\"
                    """
    
    response = query_llm ( request )

    if response == "":
        return query

    return response

def rerank_score(query: str, doc: Movie )-> int | None:

    
    request = f"""Rate how well this movie matches the search query.
                Query: "{query}"
                Movie: {doc.get("title", "")} - {doc.get("document", "")}

                Consider:
                - Direct relevance to query
                - User intent (what they're looking for)
                - Content appropriateness

                Rate 0-10 (10 = perfect match).
                Output ONLY the number in your response, no other text or explanation.

                Score:"""
    response = query_llm(request)

    try : 
        score = int(response)
    except ValueError :
        score = None

    return score 