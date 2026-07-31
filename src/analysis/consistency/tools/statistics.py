"""Shared dyadic correlation and multiple-testing utilities."""

from __future__ import annotations

import numpy as np
from scipy.stats import rankdata

from .geometry import upper_triangle_values


def _rank_residual(values: np.ndarray, covariate: np.ndarray) -> np.ndarray:
    """Rank values and residualize them against a ranked covariate."""

    values = np.asarray(values, dtype=np.float64)
    covariate = np.asarray(covariate, dtype=np.float64)
    if values.shape != covariate.shape:
        raise ValueError("values and covariate must have the same shape")
    ranked_values = rankdata(values, method="average")
    ranked_covariate = rankdata(covariate, method="average")
    design = np.column_stack([np.ones(len(values)), ranked_covariate])
    coefficients, *_ = np.linalg.lstsq(design, ranked_values, rcond=None)
    return ranked_values - design @ coefficients


def partial_spearman_rdm(
    first: np.ndarray,
    second: np.ndarray,
    question_similarity: np.ndarray,
    mask: np.ndarray,
) -> float:
    """Partial Spearman correlation between two similarity matrices."""

    first_values = upper_triangle_values(first, mask)
    second_values = upper_triangle_values(second, mask)
    question_values = upper_triangle_values(question_similarity, mask)
    if len(first_values) < 3:
        raise ValueError("At least three matrix pairs are required")
    first_residual = _rank_residual(first_values, question_values)
    second_residual = _rank_residual(second_values, question_values)
    denominator = np.linalg.norm(first_residual) * np.linalg.norm(second_residual)
    if denominator == 0:
        return float("nan")
    return float(np.dot(first_residual, second_residual) / denominator)


def spearman_rdm(
    first: np.ndarray,
    second: np.ndarray,
    mask: np.ndarray,
) -> float:
    """Ordinary Spearman correlation between two similarity matrices."""

    first_rank = rankdata(upper_triangle_values(first, mask), method="average")
    second_rank = rankdata(upper_triangle_values(second, mask), method="average")
    first_rank -= np.mean(first_rank)
    second_rank -= np.mean(second_rank)
    denominator = np.linalg.norm(first_rank) * np.linalg.norm(second_rank)
    return float(np.dot(first_rank, second_rank) / denominator)


def benjamini_hochberg(p_values: list[float]) -> list[float]:
    """Benjamini-Hochberg FDR-adjusted p-values."""

    values = np.asarray(p_values, dtype=np.float64)
    order = np.argsort(values)
    ranked = values[order]
    adjusted = ranked * len(values) / np.arange(1, len(values) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)
    output = np.empty_like(adjusted)
    output[order] = adjusted
    return output.tolist()


def holm_bonferroni(p_values: list[float]) -> list[float]:
    """Holm familywise-error adjusted p-values."""

    values = np.asarray(p_values, dtype=np.float64)
    order = np.argsort(values)
    ranked = values[order]
    adjusted = (len(values) - np.arange(len(values))) * ranked
    adjusted = np.maximum.accumulate(adjusted)
    adjusted = np.clip(adjusted, 0.0, 1.0)
    output = np.empty_like(adjusted)
    output[order] = adjusted
    return output.tolist()
