"""Representational-similarity tests for a shared ethical geometry.

Unlike clustering, RSA does not require choosing a number of groups. It asks
whether models agree about the entire pattern of which scenarios are relatively
near or far from one another. The primary test:

1. fully orthogonalizes every answer against its paired question;
2. keeps only question pairs from different domains whose question embeddings
   are in the bottom quartile of similarity and come from different sources;
3. controls continuously for remaining question-question similarity; and
4. evaluates a held-out model against the average geometry of the other five.

Question-label permutations provide a node-preserving Mantel-style null: each
permuted matrix keeps its internal distance structure but breaks scenario
alignment.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np

from ..tools.data import MODELS, FrameworkDataset
from ..tools.geometry import (
    cosine_matrix,
    cross_topic_mask,
    similarity_matrices,
)
from ..tools.statistics import (
    benjamini_hochberg,
    holm_bonferroni,
    partial_spearman_rdm,
    spearman_rdm,
)


def _permuted_matrix(matrix: np.ndarray, permutation: np.ndarray) -> np.ndarray:
    return matrix[np.ix_(permutation, permutation)]


def permutation_test(
    held_out: np.ndarray,
    consensus: np.ndarray,
    question_similarity: np.ndarray,
    mask: np.ndarray,
    *,
    permutations: int,
    random_state: int,
) -> dict[str, float | list[float]]:
    """One-sided node-permutation test of held-out RSA."""

    observed = partial_spearman_rdm(
        held_out,
        consensus,
        question_similarity,
        mask,
    )
    rng = np.random.default_rng(random_state)
    null = np.empty(permutations, dtype=np.float64)
    for index in range(permutations):
        permutation = rng.permutation(len(held_out))
        null[index] = partial_spearman_rdm(
            _permuted_matrix(held_out, permutation),
            consensus,
            question_similarity,
            mask,
        )
    p_value = float((1 + np.sum(null >= observed)) / (permutations + 1))
    null_std = float(np.std(null, ddof=1))
    z_score = (
        float((observed - np.mean(null)) / null_std)
        if null_std > 0
        else float("nan")
    )
    return {
        "rho": observed,
        "p_value": p_value,
        "null_mean": float(np.mean(null)),
        "null_std": null_std,
        "z_score": z_score,
    }


def node_bootstrap_ci(
    first: np.ndarray,
    second: np.ndarray,
    question_similarity: np.ndarray,
    mask: np.ndarray,
    *,
    samples: int,
    random_state: int,
) -> tuple[float, float]:
    """Question-node bootstrap confidence interval for partial RSA."""

    rng = np.random.default_rng(random_state)
    n_questions = len(first)
    estimates: list[float] = []
    for _ in range(samples):
        sampled = rng.integers(0, n_questions, size=n_questions)
        sub_first = first[np.ix_(sampled, sampled)]
        sub_second = second[np.ix_(sampled, sampled)]
        sub_question = question_similarity[np.ix_(sampled, sampled)]
        sub_mask = mask[np.ix_(sampled, sampled)].copy()
        # Repeated draws of the same original question are not valid pairs.
        sub_mask &= sampled[:, None] != sampled[None, :]
        try:
            estimate = partial_spearman_rdm(
                sub_first,
                sub_second,
                sub_question,
                sub_mask,
            )
        except ValueError:
            continue
        if np.isfinite(estimate):
            estimates.append(float(estimate))

    if len(estimates) < max(30, samples // 2):
        raise ValueError("Too few valid node-bootstrap estimates")
    return (
        float(np.quantile(estimates, 0.025)),
        float(np.quantile(estimates, 0.975)),
    )


def leave_one_model_out_rsa(
    matrices: dict[str, np.ndarray],
    question_similarity: np.ndarray,
    mask: np.ndarray,
    *,
    permutations: int = 999,
    bootstrap_samples: int = 500,
    random_state: int = 42,
) -> list[dict[str, float | str | list[float]]]:
    """Predict each model's geometry from the mean geometry of the other five."""

    results: list[dict[str, float | str | list[float]]] = []
    for model_index, held_out_model in enumerate(MODELS):
        other_models = [model for model in MODELS if model != held_out_model]
        consensus = np.mean(
            np.stack([matrices[model] for model in other_models]),
            axis=0,
        )
        test = permutation_test(
            matrices[held_out_model],
            consensus,
            question_similarity,
            mask,
            permutations=permutations,
            random_state=random_state + 1000 * model_index,
        )
        confidence_interval = node_bootstrap_ci(
            matrices[held_out_model],
            consensus,
            question_similarity,
            mask,
            samples=bootstrap_samples,
            random_state=random_state + 10_000 + model_index,
        )
        results.append(
            {
                "model": held_out_model,
                **test,
                "ci_95_low": confidence_interval[0],
                "ci_95_high": confidence_interval[1],
            }
        )

    adjusted = benjamini_hochberg(
        [float(result["p_value"]) for result in results]
    )
    holm = holm_bonferroni(
        [float(result["p_value"]) for result in results]
    )
    for result, q_value, holm_value in zip(
        results,
        adjusted,
        holm,
        strict=True,
    ):
        result["q_value_bh"] = q_value
        result["p_value_holm"] = holm_value
    return results


def pairwise_rsa(
    matrices: dict[str, np.ndarray],
    question_similarity: np.ndarray,
    mask: np.ndarray,
) -> list[dict[str, float | str]]:
    """All pairwise model correlations over the cross-topic geometry."""

    rows: list[dict[str, float | str]] = []
    for first, second in combinations(MODELS, 2):
        rows.append(
            {
                "model_1": first,
                "model_2": second,
                "partial_spearman_rho": partial_spearman_rdm(
                    matrices[first],
                    matrices[second],
                    question_similarity,
                    mask,
                ),
                "ordinary_spearman_rho": spearman_rdm(
                    matrices[first],
                    matrices[second],
                    mask,
                ),
            }
        )
    return rows


def split_half_reliability(
    matrices: dict[str, np.ndarray],
    question_similarity: np.ndarray,
    mask: np.ndarray,
) -> dict[str, float | list[dict[str, object]]]:
    """Correlation between every unique three-model vs three-model split."""

    rows: list[dict[str, object]] = []
    # Requiring the first model in the first half avoids counting complements
    # twice: C(5, 2) = 10 unique splits.
    first_model = MODELS[0]
    remaining = MODELS[1:]
    for partners in combinations(remaining, 2):
        first_half = (first_model, *partners)
        second_half = tuple(model for model in MODELS if model not in first_half)
        first_mean = np.mean(
            np.stack([matrices[model] for model in first_half]),
            axis=0,
        )
        second_mean = np.mean(
            np.stack([matrices[model] for model in second_half]),
            axis=0,
        )
        rho = partial_spearman_rdm(
            first_mean,
            second_mean,
            question_similarity,
            mask,
        )
        # Spearman-Brown estimates reliability of the full six-model average.
        corrected = (2.0 * rho / (1.0 + rho)) if rho > -1.0 else float("nan")
        rows.append(
            {
                "half_1": list(first_half),
                "half_2": list(second_half),
                "rho": rho,
                "spearman_brown": corrected,
            }
        )
    values = np.asarray([float(row["rho"]) for row in rows])
    corrected_values = np.asarray(
        [float(row["spearman_brown"]) for row in rows]
    )
    return {
        "splits": rows,
        "mean_rho": float(np.mean(values)),
        "min_rho": float(np.min(values)),
        "max_rho": float(np.max(values)),
        "mean_spearman_brown": float(np.mean(corrected_values)),
    }


def topic_leakage(
    matrices: dict[str, np.ndarray],
    question_similarity: np.ndarray,
) -> list[dict[str, float | str]]:
    """How much residual answer geometry still tracks question geometry."""

    domain_mask = np.asarray(
        [
            [first != second for second in range(len(question_similarity))]
            for first in range(len(question_similarity))
        ],
        dtype=bool,
    )
    # The caller replaces this permissive off-diagonal mask when reporting.
    np.fill_diagonal(domain_mask, False)
    rows = []
    for model in MODELS:
        rows.append(
            {
                "model": model,
                "spearman_rho": spearman_rdm(
                    matrices[model],
                    question_similarity,
                    domain_mask,
                ),
            }
        )
    return rows


def run_rsa_analysis(
    dataset: FrameworkDataset,
    *,
    permutations: int = 999,
    bootstrap_samples: int = 500,
    random_state: int = 42,
) -> dict[str, object]:
    """Run primary and sensitivity RSA analyses."""

    question_similarity = cosine_matrix(dataset.question_embeddings)
    primary_mask, primary_cutoff = cross_topic_mask(
        dataset,
        question_similarity_quantile=0.25,
        require_different_source=True,
    )
    residual_matrices = similarity_matrices(dataset, residual=True)
    raw_matrices = similarity_matrices(dataset, residual=False)

    held_out = leave_one_model_out_rsa(
        residual_matrices,
        question_similarity,
        primary_mask,
        permutations=permutations,
        bootstrap_samples=bootstrap_samples,
        random_state=random_state,
    )

    raw_held_out = []
    for held_out_model in MODELS:
        consensus = np.mean(
            np.stack(
                [
                    raw_matrices[model]
                    for model in MODELS
                    if model != held_out_model
                ]
            ),
            axis=0,
        )
        raw_held_out.append(
            {
                "model": held_out_model,
                "rho": partial_spearman_rdm(
                    raw_matrices[held_out_model],
                    consensus,
                    question_similarity,
                    primary_mask,
                ),
            }
        )

    sensitivity = []
    for quantile in (0.25, 0.50, 0.75):
        mask, cutoff = cross_topic_mask(
            dataset,
            question_similarity_quantile=quantile,
            require_different_source=True,
        )
        rhos = []
        for held_out_model in MODELS:
            consensus = np.mean(
                np.stack(
                    [
                        residual_matrices[model]
                        for model in MODELS
                        if model != held_out_model
                    ]
                ),
                axis=0,
            )
            rhos.append(
                partial_spearman_rdm(
                    residual_matrices[held_out_model],
                    consensus,
                    question_similarity,
                    mask,
                )
            )
        sensitivity.append(
            {
                "question_similarity_quantile": quantile,
                "question_similarity_cutoff": cutoff,
                "require_different_source": True,
                "pair_count": int(np.sum(np.triu(mask, k=1))),
                "mean_held_out_rho": float(np.mean(rhos)),
                "min_held_out_rho": float(np.min(rhos)),
                "max_held_out_rho": float(np.max(rhos)),
            }
        )

    domain_only_mask, domain_only_cutoff = cross_topic_mask(
        dataset,
        question_similarity_quantile=0.50,
        require_different_source=False,
    )
    domain_only_rhos = []
    for held_out_model in MODELS:
        consensus = np.mean(
            np.stack(
                [
                    residual_matrices[model]
                    for model in MODELS
                    if model != held_out_model
                ]
            ),
            axis=0,
        )
        domain_only_rhos.append(
            partial_spearman_rdm(
                residual_matrices[held_out_model],
                consensus,
                question_similarity,
                domain_only_mask,
            )
        )
    sensitivity.append(
        {
            "question_similarity_quantile": 0.50,
            "question_similarity_cutoff": domain_only_cutoff,
            "require_different_source": False,
            "pair_count": int(np.sum(np.triu(domain_only_mask, k=1))),
            "mean_held_out_rho": float(np.mean(domain_only_rhos)),
            "min_held_out_rho": float(np.min(domain_only_rhos)),
            "max_held_out_rho": float(np.max(domain_only_rhos)),
        }
    )

    domains = np.asarray(dataset.domains, dtype=object)
    different_domain = domains[:, None] != domains[None, :]
    np.fill_diagonal(different_domain, False)
    leakage = [
        {
            "model": model,
            "spearman_rho": spearman_rdm(
                residual_matrices[model],
                question_similarity,
                different_domain,
            ),
        }
        for model in MODELS
    ]
    raw_leakage = [
        {
            "model": model,
            "spearman_rho": spearman_rdm(
                raw_matrices[model],
                question_similarity,
                different_domain,
            ),
        }
        for model in MODELS
    ]
    primary_leakage = [
        {
            "model": model,
            "residual_spearman_rho": spearman_rdm(
                residual_matrices[model],
                question_similarity,
                primary_mask,
            ),
            "raw_spearman_rho": spearman_rdm(
                raw_matrices[model],
                question_similarity,
                primary_mask,
            ),
        }
        for model in MODELS
    ]

    return {
        "method": "cross-topic partial Spearman RSA with node permutations",
        "primary_definition": {
            "different_exact_domain": True,
            "different_source": True,
            "question_similarity_quantile": 0.25,
            "question_similarity_cutoff": primary_cutoff,
            "pair_count": int(np.sum(np.triu(primary_mask, k=1))),
            "continuous_topic_control": "ranked question-question cosine",
        },
        "held_out_models": held_out,
        "mean_held_out_rho": float(
            np.mean([float(row["rho"]) for row in held_out])
        ),
        "min_held_out_rho": float(
            np.min([float(row["rho"]) for row in held_out])
        ),
        "mean_raw_held_out_rho": float(
            np.mean([float(row["rho"]) for row in raw_held_out])
        ),
        "raw_held_out_models": raw_held_out,
        "pairwise_models": pairwise_rsa(
            residual_matrices,
            question_similarity,
            primary_mask,
        ),
        "split_half_reliability": split_half_reliability(
            residual_matrices,
            question_similarity,
            primary_mask,
        ),
        "topic_leakage_after_projection": leakage,
        "topic_leakage_before_projection": raw_leakage,
        "topic_leakage_primary_cross_topic_mask": primary_leakage,
        "sensitivity": sensitivity,
    }
