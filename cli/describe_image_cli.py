import argparse
import mimetypes
from dotenv import load_dotenv
import os

def main() -> None: 
    parser = argparse.ArgumentParser(description="Image Search CLI")
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to the image"
    )
    parser.add_argument(
        "--query",
        type=str,
        required=True,
        help="Query alongside image"
        )

    args = parser.parse_args()
    mime, _ = mimetypes.guess_file_type(args.image)
    mime = mime or "image/jpeg"

    

    with open(args.image, "rb") as f:
        img = f.read()

    load_dotenv()

    api_key= os.environ.get("API_KEY")

    from openai import OpenAI
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key= api_key
    )

    import base64

    system_prompt= f"""Given the included image and text query, rewrite the text query to improve search results from a movie database. Make sure to:
                    - Synthesize visual and textual information
                    - Focus on movie-specific details (actors, scenes, style, etc.)
                    - Return only the rewritten query, without any additional commentary"""

    from openai.types.chat import ChatCompletionMessageParam
    data_url = f"data:{mime};base64,{base64.b64encode(img).decode()}"

    messages: list[ChatCompletionMessageParam] = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": system_prompt.strip()},
                {"type": "image_url", "image_url": {"url": data_url}},
                {"type": "text", "text": args.query.strip()},
            ],
        }]
    response = client.chat.completions.create(model="openrouter/free", messages = messages)
    content = response.choices[0].message.content
    if content: 
        print(f"Rewritten query: {content.strip()}")
    if response.usage is not None:
        print(f"Total tokens:    {response.usage.total_tokens}")


if __name__ == "__main__":
    main()
        