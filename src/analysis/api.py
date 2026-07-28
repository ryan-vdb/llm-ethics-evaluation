import numpy as np


def cosine_similarity(
    embedding1: list[float],
    embedding2: list[float],
) -> float:
    """
    Compute cosine similarity between two embeddings.
    """

    a = np.array(embedding1)
    b = np.array(embedding2)

    return float(
        np.dot(a, b)
        / (np.linalg.norm(a) * np.linalg.norm(b))
    )


def euclidean_distance(
    embedding1: list[float],
    embedding2: list[float],
) -> float:
    """
    Compute Euclidean distance between two embeddings.
    """

    a = np.array(embedding1)
    b = np.array(embedding2)

    return float(np.linalg.norm(a - b))