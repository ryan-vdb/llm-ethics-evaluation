import os
from openai import OpenAI

api_key = os.getenv("OPENROUTER_API_KEY")

if not api_key:
    raise RuntimeError("OPENROUTER_API_KEY is not set.")

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
    if not os.getenv("OPENROUTER_API_KEY"):
        raise ValueError("OPENROUTER_API_KEY is not set.")
    response = client.embeddings.create(
        model=model,
        input=text,
    )

    return response.data[0].embedding