"""Shared exact-projection and cosine-geometry utilities."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .data import FrameworkDataset


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    """Return a row-wise L2-normalized finite matrix."""

    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("matrix must be two-dimensional")
    if not np.isfinite(matrix).all():
        raise ValueError("matrix contains NaN or infinite values")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms <= 1e-12):
        raise ValueError("matrix contains a zero or near-zero row")
    return matrix / norms


def paired_cosines(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Cosine similarity for corresponding rows in two aligned matrices."""

    if first.shape != second.shape:
        raise ValueError(
            f"paired matrices must have the same shape: "
            f"{first.shape} vs {second.shape}"
        )
    return np.sum(normalize_rows(first) * normalize_rows(second), axis=1)


def cosine_matrix(matrix: np.ndarray) -> np.ndarray:
    """Row-by-row cosine-similarity matrix."""

    normalized = normalize_rows(matrix)
    return np.clip(normalized @ normalized.T, -1.0, 1.0)


def remove_question_projection(
    response_embeddings: np.ndarray,
    question_embeddings: np.ndarray,
    *,
    strength: float = 1.0,
) -> np.ndarray:
    """Remove each paired question direction from its answer.

    ``strength=1.0`` is exact orthogonalization. It is the only strength used
    by the canonical framework loader.
    """

    responses = np.asarray(response_embeddings, dtype=np.float64)
    questions = np.asarray(question_embeddings, dtype=np.float64)
    if responses.shape != questions.shape:
        raise ValueError(
            "Response and question embeddings must have the same shape"
        )
    if not 0.0 <= strength <= 1.0:
        raise ValueError("strength must be between 0 and 1")
    question_norm_squared = np.sum(questions**2, axis=1, keepdims=True)
    if np.any(question_norm_squared <= 1e-15):
        raise ValueError("Question embeddings cannot contain zero vectors")
    coefficients = (
        np.sum(responses * questions, axis=1, keepdims=True)
        / question_norm_squared
    )
    return responses - strength * coefficients * questions


def cross_topic_mask(
    dataset: FrameworkDataset,
    *,
    question_similarity_quantile: float = 0.50,
    require_different_source: bool = False,
) -> tuple[np.ndarray, float]:
    """Select low-question-similarity pairs from different declared domains."""

    if not 0.0 < question_similarity_quantile < 1.0:
        raise ValueError("question_similarity_quantile must be between 0 and 1")
    question_similarity = cosine_matrix(dataset.question_embeddings)
    triangle = np.triu_indices_from(question_similarity, k=1)
    cutoff = float(
        np.quantile(
            question_similarity[triangle],
            question_similarity_quantile,
        )
    )
    domains = np.asarray(dataset.domains, dtype=object)
    mask = (
        (domains[:, None] != domains[None, :])
        & (question_similarity <= cutoff)
    )
    if require_different_source:
        sources = np.asarray(dataset.sources, dtype=object)
        mask &= sources[:, None] != sources[None, :]
    np.fill_diagonal(mask, False)
    return mask, cutoff


def upper_triangle_values(
    matrix: np.ndarray,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """Flatten unique matrix pairs, optionally through a symmetric mask."""

    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("matrix must be square")
    triangle = np.triu(np.ones(matrix.shape, dtype=bool), k=1)
    if mask is not None:
        if mask.shape != matrix.shape:
            raise ValueError("mask and matrix must have the same shape")
        triangle &= mask
    return matrix[triangle]


def similarity_matrices(
    dataset: FrameworkDataset,
    *,
    residual: bool = True,
) -> dict[str, np.ndarray]:
    """Cosine matrices for every model in canonical order."""

    from .data import MODELS

    views = dataset.residuals if residual else dataset.raw_responses
    return {model: cosine_matrix(views[model]) for model in MODELS}
