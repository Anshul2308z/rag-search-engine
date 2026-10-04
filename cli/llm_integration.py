from dotenv import load_dotenv
import os
from openai import OpenAI
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
            model="openrouter/free",
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

    enhanced: bool = True 

    request = f"""Fix any spelling errors in the user-provided movie search query below.
                Correct only clear, high-confidence typos. Do not rewrite, add, remove, or reorder words.
                Preserve punctuation and capitalization unless a change is required for a typo fix.
                If there are no spelling errors, or if you're unsure, output the original query unchanged.
                Output only the final query text, nothing else.
                User query: "{query}"
                """

    response= query_llm( request )

    return response 




    