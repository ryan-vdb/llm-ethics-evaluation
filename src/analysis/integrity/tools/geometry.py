"""Exact projection and cosine-geometry utilities for integrity responses."""

from __future__ import annotations

import numpy as np


def normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    """L2-normalize a finite array along its final (embedding) axis."""

    values = np.asarray(vectors, dtype=np.float64)
    if values.ndim < 1:
        raise ValueError("vectors must have at least one dimension")
    if values.shape[-1] == 0:
        raise ValueError("embedding dimension cannot be empty")
    if not np.isfinite(values).all():
        raise ValueError("vectors contain NaN or infinite values")
    norms = np.linalg.norm(values, axis=-1, keepdims=True)
    if np.any(norms <= 1e-12):
        raise ValueError("vectors contain a zero or near-zero embedding")
    return values / norms


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    """Return a row-wise L2-normalized two-dimensional matrix."""

    values = np.asarray(matrix, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("matrix must be two-dimensional")
    return normalize_vectors(values)


def paired_cosines(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Cosine similarity along the last axis of two aligned arrays."""

    first_values = np.asarray(first, dtype=np.float64)
    second_values = np.asarray(second, dtype=np.float64)
    if first_values.shape != second_values.shape:
        raise ValueError(
            "paired arrays must have the same shape: "
            f"{first_values.shape} vs {second_values.shape}"
        )
    return np.sum(
        normalize_vectors(first_values) * normalize_vectors(second_values),
        axis=-1,
    )


def remove_question_projection(
    response_embeddings: np.ndarray,
    question_embeddings: np.ndarray,
) -> np.ndarray:
    """Exactly remove each paired question vector from each response vector.

    The two inputs must already be aligned and have identical shapes. Requiring
    explicit alignment prevents accidental broadcasting across questions.
    """

    responses = np.asarray(response_embeddings, dtype=np.float64)
    questions = np.asarray(question_embeddings, dtype=np.float64)
    if responses.shape != questions.shape:
        raise ValueError(
            "response and question embeddings must have the same shape: "
            f"{responses.shape} vs {questions.shape}"
        )
    if responses.ndim < 2:
        raise ValueError("embedding arrays must have at least two dimensions")
    if not np.isfinite(responses).all() or not np.isfinite(questions).all():
        raise ValueError("embedding arrays contain NaN or infinite values")

    question_norm_squared = np.sum(questions**2, axis=-1, keepdims=True)
    if np.any(question_norm_squared <= 1e-15):
        raise ValueError("question embeddings cannot contain zero vectors")
    coefficients = (
        np.sum(responses * questions, axis=-1, keepdims=True)
        / question_norm_squared
    )
    return responses - coefficients * questions


def cosine_matrix(matrix: np.ndarray) -> np.ndarray:
    """Return the row-by-row cosine-similarity matrix for a 2D matrix."""

    normalized = normalize_rows(matrix)
    return np.clip(normalized @ normalized.T, -1.0, 1.0)


def initial_similarity(
    views: np.ndarray,
    *,
    initial_index: int,
) -> np.ndarray:
    """Cosine of every condition with the paired initial response.

    ``views`` must have shape ``[model, question, condition, embedding]``.
    The result has shape ``[model, question, condition]``.
    """

    values = np.asarray(views, dtype=np.float64)
    if values.ndim != 4:
        raise ValueError(
            "views must have shape [model, question, condition, embedding]"
        )
    if not 0 <= initial_index < values.shape[2]:
        raise IndexError("initial_index is outside the condition axis")
    initial = np.broadcast_to(
        values[:, :, initial_index : initial_index + 1, :],
        values.shape,
    )
    return paired_cosines(values, initial)


def condition_displacements(
    views: np.ndarray,
    *,
    initial_index: int,
) -> np.ndarray:
    """Subtract each paired initial response from every condition response."""

    values = np.asarray(views, dtype=np.float64)
    if values.ndim != 4:
        raise ValueError(
            "views must have shape [model, question, condition, embedding]"
        )
    if not 0 <= initial_index < values.shape[2]:
        raise IndexError("initial_index is outside the condition axis")
    return values - values[:, :, initial_index : initial_index + 1, :]
