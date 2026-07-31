"""Debiased kernel alignment across model-specific ethical geometries.

Centered Kernel Alignment (CKA) compares whole representational spaces without
requiring their individual embedding coordinates to match. Here each model is a
separate view of the same 93 scenarios. Linear Gram matrices are built only
after exact question projection removal and row normalization.

The primary statistic normalizes unbiased HSIC estimators into a CKA-like
ratio. The HSIC terms are unbiased, although their ratio is not itself an
unbiased estimator. Node permutations supply the finite-sample null
distribution.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np

from ..tools.data import MODELS, FrameworkDataset
from ..tools.statistics import benjamini_hochberg, holm_bonferroni


def linear_gram(features: np.ndarray) -> np.ndarray:
    """Linear question-by-question Gram matrix."""

    features = np.asarray(features, dtype=np.float64)
    if features.ndim != 2:
        raise ValueError("features must be two-dimensional")
    return features @ features.T


def unbiased_hsic(first_gram: np.ndarray, second_gram: np.ndarray) -> float:
    """Unbiased Hilbert-Schmidt independence criterion.

    Implements the U-statistic estimator from Song et al. (2012). The diagonals
    are excluded explicitly.
    """

    first = np.asarray(first_gram, dtype=np.float64).copy()
    second = np.asarray(second_gram, dtype=np.float64).copy()
    if first.shape != second.shape:
        raise ValueError("Gram matrices must have the same shape")
    if first.ndim != 2 or first.shape[0] != first.shape[1]:
        raise ValueError("Gram matrices must be square")
    n_samples = first.shape[0]
    if n_samples < 4:
        raise ValueError("Unbiased HSIC requires at least four samples")

    np.fill_diagonal(first, 0.0)
    np.fill_diagonal(second, 0.0)
    first_row_sums = np.sum(first, axis=1)
    second_row_sums = np.sum(second, axis=1)
    term_1 = float(np.sum(first * second))
    term_2 = float(
        np.sum(first) * np.sum(second) / ((n_samples - 1) * (n_samples - 2))
    )
    term_3 = float(
        2.0
        * np.dot(first_row_sums, second_row_sums)
        / (n_samples - 2)
    )
    return (term_1 + term_2 - term_3) / (n_samples * (n_samples - 3))


def unbiased_cka(first_gram: np.ndarray, second_gram: np.ndarray) -> float:
    """CKA ratio from unbiased HSIC terms; it may be negative under the null."""

    cross = unbiased_hsic(first_gram, second_gram)
    first_self = unbiased_hsic(first_gram, first_gram)
    second_self = unbiased_hsic(second_gram, second_gram)
    denominator = np.sqrt(max(first_self, 0.0) * max(second_self, 0.0))
    if denominator <= 0:
        return float("nan")
    return float(cross / denominator)


def cka_permutation_test(
    held_out_gram: np.ndarray,
    consensus_gram: np.ndarray,
    *,
    permutations: int,
    random_state: int,
) -> dict[str, float]:
    """One-sided node-permutation test for CKA from unbiased HSIC terms."""

    observed = unbiased_cka(held_out_gram, consensus_gram)
    rng = np.random.default_rng(random_state)
    null = np.empty(permutations, dtype=np.float64)
    for index in range(permutations):
        permutation = rng.permutation(len(held_out_gram))
        permuted = held_out_gram[np.ix_(permutation, permutation)]
        null[index] = unbiased_cka(permuted, consensus_gram)

    p_value = float((1 + np.sum(null >= observed)) / (permutations + 1))
    null_std = float(np.std(null, ddof=1))
    return {
        "cka": observed,
        "p_value": p_value,
        "null_mean": float(np.mean(null)),
        "null_std": null_std,
        "z_score": (
            float((observed - np.mean(null)) / null_std)
            if null_std > 0
            else float("nan")
        ),
    }


def run_kernel_alignment(
    dataset: FrameworkDataset,
    *,
    permutations: int = 999,
    random_state: int = 42,
) -> dict[str, object]:
    """Run held-out, pairwise, and raw-vs-orthogonal CKA analyses."""

    residual_grams = {
        model: linear_gram(dataset.residuals[model]) for model in MODELS
    }
    raw_grams = {
        model: linear_gram(dataset.raw_responses[model]) for model in MODELS
    }

    held_out_rows = []
    raw_rows = []
    for model_index, held_out_model in enumerate(MODELS):
        others = [model for model in MODELS if model != held_out_model]
        residual_consensus = np.mean(
            np.stack([residual_grams[model] for model in others]),
            axis=0,
        )
        raw_consensus = np.mean(
            np.stack([raw_grams[model] for model in others]),
            axis=0,
        )
        held_out_rows.append(
            {
                "model": held_out_model,
                **cka_permutation_test(
                    residual_grams[held_out_model],
                    residual_consensus,
                    permutations=permutations,
                    random_state=random_state + 1000 * model_index,
                ),
            }
        )
        raw_rows.append(
            {
                "model": held_out_model,
                "cka": unbiased_cka(
                    raw_grams[held_out_model],
                    raw_consensus,
                ),
            }
        )

    adjusted = benjamini_hochberg(
        [float(row["p_value"]) for row in held_out_rows]
    )
    holm = holm_bonferroni(
        [float(row["p_value"]) for row in held_out_rows]
    )
    for row, q_value, holm_value in zip(
        held_out_rows,
        adjusted,
        holm,
        strict=True,
    ):
        row["q_value_bh"] = q_value
        row["p_value_holm"] = holm_value

    pairwise_rows = []
    for first, second in combinations(MODELS, 2):
        pairwise_rows.append(
            {
                "model_1": first,
                "model_2": second,
                "unbiased_cka": unbiased_cka(
                    residual_grams[first],
                    residual_grams[second],
                ),
            }
        )

    return {
        "method": "leave-one-model-out linear CKA using unbiased HSIC estimators",
        "held_out_models": held_out_rows,
        "mean_held_out_cka": float(
            np.mean([float(row["cka"]) for row in held_out_rows])
        ),
        "min_held_out_cka": float(
            np.min([float(row["cka"]) for row in held_out_rows])
        ),
        "mean_raw_held_out_cka": float(
            np.mean([float(row["cka"]) for row in raw_rows])
        ),
        "raw_held_out_models": raw_rows,
        "pairwise_models": pairwise_rows,
        "interpretation": (
            "CKA measures agreement between whole geometries and is invariant "
            "to isotropic scaling and orthogonal rotations. This ratio uses "
            "unbiased HSIC terms, but the ratio itself is not unbiased and can "
            "be negative under the null."
        ),
    }
