import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI


ENV_PATH = Path(__file__).resolve().parents[2] / ".env"
API_KEY_VARIABLE = "OPENROUTER_EMBEDDING_API_KEY"

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


def get_embedding(text: str, model: str = "openai/text-embedding-3-large") -> list[float]:
    """
    Generate an embedding for a piece of text.

    Args:
        text: The text to embed.
        model: The embedding model to use.

    Returns:
        A list of floats representing the embedding vector.
    """
    response = client.embeddings.create(
        model=model,
        input=text,
    )

    return response.data[0].embedding
