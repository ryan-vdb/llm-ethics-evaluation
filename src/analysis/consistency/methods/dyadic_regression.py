"""Topic-controlled multiple-regression QAP (MRQAP) sensitivity analysis.

Representational similarity gives a rank-based effect. MRQAP complements it
with an interpretable regression estimand: how much of a held-out model's
pairwise answer geometry is predicted by the other five models *after* fitting
question similarity (linear and quadratic) and same-source indicators.

The consensus coefficient is standardized. Incremental R² is the extra dyadic
variance explained when consensus geometry is added to a topic-only model.
Question-label permutations preserve network dependence when testing the
coefficient.
"""

from __future__ import annotations

import numpy as np

from ..tools.data import MODELS, FrameworkDataset
from ..tools.geometry import (
    cosine_matrix,
    cross_topic_mask,
    similarity_matrices,
    upper_triangle_values,
)
from ..tools.statistics import (
    benjamini_hochberg,
    holm_bonferroni,
)


def _standardize(column: np.ndarray) -> np.ndarray:
    column = np.asarray(column, dtype=np.float64)
    std = float(np.std(column))
    if std <= 1e-12:
        return np.zeros_like(column)
    return (column - np.mean(column)) / std


def _design_matrices(
    consensus_similarity: np.ndarray,
    question_similarity: np.ndarray,
    same_source: np.ndarray,
    mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    consensus = _standardize(
        upper_triangle_values(consensus_similarity, mask)
    )
    topic = _standardize(upper_triangle_values(question_similarity, mask))
    topic_squared = _standardize(topic**2)
    source = _standardize(
        upper_triangle_values(same_source.astype(float), mask)
    )
    reduced = np.column_stack(
        [
            np.ones(len(consensus)),
            topic,
            topic_squared,
            source,
        ]
    )
    full = np.column_stack([reduced, consensus])
    return reduced, full


def _fit_ols(target: np.ndarray, design: np.ndarray) -> tuple[np.ndarray, float]:
    coefficients, *_ = np.linalg.lstsq(design, target, rcond=None)
    fitted = design @ coefficients
    residual_sum = float(np.sum((target - fitted) ** 2))
    total_sum = float(np.sum((target - np.mean(target)) ** 2))
    r_squared = 1.0 - residual_sum / total_sum if total_sum > 0 else float("nan")
    return coefficients, r_squared


def fit_mrqap_effect(
    held_out_similarity: np.ndarray,
    consensus_similarity: np.ndarray,
    question_similarity: np.ndarray,
    same_source: np.ndarray,
    mask: np.ndarray,
) -> dict[str, float]:
    """Fit topic-only and topic-plus-consensus dyadic regressions."""

    target = _standardize(upper_triangle_values(held_out_similarity, mask))
    reduced_design, full_design = _design_matrices(
        consensus_similarity,
        question_similarity,
        same_source,
        mask,
    )
    _, reduced_r_squared = _fit_ols(target, reduced_design)
    full_coefficients, full_r_squared = _fit_ols(target, full_design)
    return {
        "standardized_consensus_beta": float(full_coefficients[-1]),
        "topic_only_r_squared": reduced_r_squared,
        "full_r_squared": full_r_squared,
        "incremental_r_squared": float(full_r_squared - reduced_r_squared),
    }


def mrqap_permutation_test(
    held_out_similarity: np.ndarray,
    consensus_similarity: np.ndarray,
    question_similarity: np.ndarray,
    same_source: np.ndarray,
    mask: np.ndarray,
    *,
    permutations: int,
    random_state: int,
) -> dict[str, float]:
    """One-sided complete-network nuisance-residual QAP sensitivity test.

    A topic-only model is first fit over the complete off-diagonal network.
    Its symmetric residual matrix is node-permuted, added back to the fitted
    topic matrix, and the primary masked regression is refit. This preserves
    network dependence while testing consensus conditional on topic controls.
    """

    observed = fit_mrqap_effect(
        held_out_similarity,
        consensus_similarity,
        question_similarity,
        same_source,
        mask,
    )
    # Construct a full symmetric nuisance-fit and residual matrix so node
    # permutations remain well-defined even when the primary mask is strict.
    n_questions = len(held_out_similarity)
    complete_mask = np.ones((n_questions, n_questions), dtype=bool)
    np.fill_diagonal(complete_mask, False)
    complete_target = _standardize(
        upper_triangle_values(held_out_similarity, complete_mask)
    )
    complete_reduced, _ = _design_matrices(
        consensus_similarity,
        question_similarity,
        same_source,
        complete_mask,
    )
    reduced_coefficients, _ = _fit_ols(complete_target, complete_reduced)
    fitted_values = complete_reduced @ reduced_coefficients
    residual_values = complete_target - fitted_values

    triangle = np.triu_indices(n_questions, k=1)
    fitted_matrix = np.zeros((n_questions, n_questions), dtype=np.float64)
    residual_matrix = np.zeros((n_questions, n_questions), dtype=np.float64)
    fitted_matrix[triangle] = fitted_values
    residual_matrix[triangle] = residual_values
    fitted_matrix[(triangle[1], triangle[0])] = fitted_values
    residual_matrix[(triangle[1], triangle[0])] = residual_values

    rng = np.random.default_rng(random_state)
    null_betas = np.empty(permutations, dtype=np.float64)
    for index in range(permutations):
        permutation = rng.permutation(len(held_out_similarity))
        permuted_residual = residual_matrix[np.ix_(permutation, permutation)]
        permuted = fitted_matrix + permuted_residual
        null_betas[index] = fit_mrqap_effect(
            permuted,
            consensus_similarity,
            question_similarity,
            same_source,
            mask,
        )["standardized_consensus_beta"]

    beta = observed["standardized_consensus_beta"]
    p_value = float((1 + np.sum(null_betas >= beta)) / (permutations + 1))
    null_std = float(np.std(null_betas, ddof=1))
    return {
        **observed,
        "p_value": p_value,
        "null_beta_mean": float(np.mean(null_betas)),
        "null_beta_std": null_std,
        "z_score": (
            float((beta - np.mean(null_betas)) / null_std)
            if null_std > 0
            else float("nan")
        ),
    }


def mrqap_node_bootstrap_ci(
    held_out_similarity: np.ndarray,
    consensus_similarity: np.ndarray,
    question_similarity: np.ndarray,
    same_source: np.ndarray,
    mask: np.ndarray,
    *,
    samples: int,
    random_state: int,
) -> dict[str, float]:
    """Question-node percentile CIs for beta and incremental R²."""

    rng = np.random.default_rng(random_state)
    n_questions = len(held_out_similarity)
    betas: list[float] = []
    increments: list[float] = []
    for _ in range(samples):
        sampled = rng.integers(0, n_questions, size=n_questions)
        index = np.ix_(sampled, sampled)
        sub_mask = mask[index].copy()
        sub_mask &= sampled[:, None] != sampled[None, :]
        if np.sum(np.triu(sub_mask, k=1)) < 30:
            continue
        effect = fit_mrqap_effect(
            held_out_similarity[index],
            consensus_similarity[index],
            question_similarity[index],
            same_source[index],
            sub_mask,
        )
        if np.isfinite(effect["standardized_consensus_beta"]):
            betas.append(effect["standardized_consensus_beta"])
            increments.append(effect["incremental_r_squared"])

    if len(betas) < max(30, samples // 2):
        raise ValueError("Too few valid MRQAP bootstrap estimates")
    return {
        "beta_ci_95_low": float(np.quantile(betas, 0.025)),
        "beta_ci_95_high": float(np.quantile(betas, 0.975)),
        "incremental_r2_ci_95_low": float(np.quantile(increments, 0.025)),
        "incremental_r2_ci_95_high": float(np.quantile(increments, 0.975)),
    }


def run_mrqap_analysis(
    dataset: FrameworkDataset,
    *,
    permutations: int = 999,
    bootstrap_samples: int = 500,
    random_state: int = 42,
) -> dict[str, object]:
    """Run primary held-out MRQAP and leave-one-source-out sensitivity."""

    residual_matrices = similarity_matrices(dataset, residual=True)
    raw_matrices = similarity_matrices(dataset, residual=False)
    question_similarity = cosine_matrix(dataset.question_embeddings)
    mask, cutoff = cross_topic_mask(
        dataset,
        question_similarity_quantile=0.25,
    )
    sources = np.asarray(dataset.sources, dtype=object)
    same_source = sources[:, None] == sources[None, :]

    held_out_rows = []
    raw_rows = []
    for model_index, held_out_model in enumerate(MODELS):
        other_models = [model for model in MODELS if model != held_out_model]
        residual_consensus = np.mean(
            np.stack([residual_matrices[model] for model in other_models]),
            axis=0,
        )
        raw_consensus = np.mean(
            np.stack([raw_matrices[model] for model in other_models]),
            axis=0,
        )
        test = mrqap_permutation_test(
            residual_matrices[held_out_model],
            residual_consensus,
            question_similarity,
            same_source,
            mask,
            permutations=permutations,
            random_state=random_state + 1000 * model_index,
        )
        confidence_intervals = mrqap_node_bootstrap_ci(
            residual_matrices[held_out_model],
            residual_consensus,
            question_similarity,
            same_source,
            mask,
            samples=bootstrap_samples,
            random_state=random_state + 10_000 + model_index,
        )
        held_out_rows.append(
            {
                "model": held_out_model,
                **test,
                **confidence_intervals,
            }
        )
        raw_rows.append(
            {
                "model": held_out_model,
                **fit_mrqap_effect(
                    raw_matrices[held_out_model],
                    raw_consensus,
                    question_similarity,
                    same_source,
                    mask,
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

    leave_source_out = []
    for omitted_source in sorted(set(dataset.sources)):
        keep_questions = sources != omitted_source
        source_mask = mask & np.outer(keep_questions, keep_questions)
        if np.sum(np.triu(source_mask, k=1)) < 100:
            continue
        effects = []
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
            effects.append(
                fit_mrqap_effect(
                    residual_matrices[held_out_model],
                    consensus,
                    question_similarity,
                    same_source,
                    source_mask,
                )
            )
        leave_source_out.append(
            {
                "omitted_source": omitted_source,
                "remaining_pair_count": int(
                    np.sum(np.triu(source_mask, k=1))
                ),
                "mean_beta": float(
                    np.mean(
                        [
                            effect["standardized_consensus_beta"]
                            for effect in effects
                        ]
                    )
                ),
                "mean_incremental_r_squared": float(
                    np.mean(
                        [effect["incremental_r_squared"] for effect in effects]
                    )
                ),
            }
        )

    return {
        "method": (
            "leave-one-model-out topic-controlled nuisance-residual MRQAP "
            "sensitivity"
        ),
        "primary_definition": {
            "different_exact_domain": True,
            "different_source": False,
            "question_similarity_quantile": 0.25,
            "question_similarity_cutoff": cutoff,
            "pair_count": int(np.sum(np.triu(mask, k=1))),
            "topic_covariates": [
                "question cosine",
                "question cosine squared",
                "same source",
            ],
            "permutation_note": (
                "Nuisance residuals are estimated on the complete dyadic "
                "network so node permutations remain defined, then the "
                "coefficient is refit on the masked estimand. Treat p-values "
                "as exploratory sensitivity evidence."
            ),
        },
        "held_out_models": held_out_rows,
        "mean_standardized_beta": float(
            np.mean(
                [
                    float(row["standardized_consensus_beta"])
                    for row in held_out_rows
                ]
            )
        ),
        "min_standardized_beta": float(
            np.min(
                [
                    float(row["standardized_consensus_beta"])
                    for row in held_out_rows
                ]
            )
        ),
        "mean_incremental_r_squared": float(
            np.mean(
                [
                    float(row["incremental_r_squared"])
                    for row in held_out_rows
                ]
            )
        ),
        "mean_raw_standardized_beta": float(
            np.mean(
                [
                    float(row["standardized_consensus_beta"])
                    for row in raw_rows
                ]
            )
        ),
        "raw_held_out_models": raw_rows,
        "leave_one_source_out": leave_source_out,
    }
