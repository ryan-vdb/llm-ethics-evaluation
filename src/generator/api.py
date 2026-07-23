import os
from openai import OpenAI

api_key = os.getenv("OPENROUTER_API_KEY")

if not api_key:
    raise RuntimeError("OPENROUTER_API_KEY is not set.")

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key,
)


def get_response(model: str, messages: list) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=messages,
    )

    return response.choices[0].message.content