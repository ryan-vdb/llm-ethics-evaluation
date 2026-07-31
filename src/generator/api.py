import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI


ENV_PATH = Path(__file__).resolve().parents[2] / ".env"
API_KEY_VARIABLE = "OPENROUTER_GENERATION_API_KEY"

load_dotenv(ENV_PATH, override=False)
api_key = os.getenv(API_KEY_VARIABLE)

if not api_key:
    raise RuntimeError(
        f"{API_KEY_VARIABLE} is not set. Add it to {ENV_PATH} or export it "
        "in your shell."
    )

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
