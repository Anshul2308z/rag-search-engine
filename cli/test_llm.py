import os 
from dotenv import load_dotenv

load_dotenv()

api_key = os.environ.get("API_KEY")

if not api_key:
    raise RuntimeError("API_KEY environment variable not set")

from openai import OpenAI

client = OpenAI(
    base_url="  ",
    api_key= api_key,
)

from openai.types.chat import ChatCompletionMessageParam
messages: list[ChatCompletionMessageParam] = [
    {
        "role": "user",
        "content": "Why is Lord of the mysteries the best novel ever? Use one paragraph maximum.",
    }
]

response = client.chat.completions.create(model="openrouter/free", messages=messages)

print(f"Response: {response.choices[0].message.content}")

usage = response.usage

if usage:
    print(f"Prompt tokens: {usage.prompt_tokens}")
    print(f"Response tokens: {usage.completion_tokens}")

