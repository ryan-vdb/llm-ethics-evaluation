"""Cross-domain neighborhood transfer and interpretable scenario pairs.

This module treats each model as an independent geometric view. For every
question, candidate neighbors must come from a different domain and fall below
the bottom quartile of question-question embedding similarity, and they must
come from different sources. A model is held out; the
other five nominate their nearest neighbors; and the test asks how many of
those neighbors the held-out model independently recovers.

The same geometry is also used to surface stable, high-similarity pairs of
topically unrelated dilemmas for qualitative interpretation.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
from scipy.stats import rankdata

from ..tools.data import MODELS, FrameworkDataset
from ..tools.geometry import (
    cosine_matrix,
    cross_topic_mask,
)
from ..tools.statistics import benjamini_hochberg, holm_bonferroni


def top_neighbors(
    similarity: np.ndarray,
    candidate_mask: np.ndarray,
    *,
    k: int,
) -> list[np.ndarray]:
    """Top-k candidate indices for every question."""

    if similarity.shape != candidate_mask.shape:
        raise ValueError("similarity and candidate_mask must have the same shape")
    output: list[np.ndarray] = []
    for question_index in range(len(similarity)):
        candidates = np.flatnonzero(candidate_mask[question_index])
        if len(candidates) == 0:
            output.append(np.asarray([], dtype=int))
            continue
        count = min(k, len(candidates))
        order = np.argsort(-similarity[question_index, candidates], kind="stable")
        output.append(candidates[order[:count]])
    return output


def neighborhood_recovery(
    held_out_similarity: np.ndarray,
    consensus_similarity: np.ndarray,
    candidate_mask: np.ndarray,
    *,
    k: int,
) -> tuple[float, np.ndarray]:
    """Fraction of consensus neighbors recovered by the held-out model."""

    held_out = top_neighbors(held_out_similarity, candidate_mask, k=k)
    consensus = top_neighbors(consensus_similarity, candidate_mask, k=k)
    per_question = np.full(len(held_out), np.nan, dtype=np.float64)
    for index, (held, expected) in enumerate(zip(held_out, consensus, strict=True)):
        if len(expected) == 0:
            continue
        per_question[index] = len(set(held).intersection(expected)) / len(expected)
    return float(np.nanmean(per_question)), per_question


def recovery_permutation_test(
    held_out_similarity: np.ndarray,
    consensus_similarity: np.ndarray,
    candidate_mask: np.ndarray,
    *,
    k: int,
    permutations: int,
    random_state: int,
) -> dict[str, float | list[float]]:
    """Node-permutation test for held-out cross-topic neighbor recovery."""

    observed, per_question = neighborhood_recovery(
        held_out_similarity,
        consensus_similarity,
        candidate_mask,
        k=k,
    )
    rng = np.random.default_rng(random_state)
    null = np.empty(permutations, dtype=np.float64)
    for permutation_index in range(permutations):
        permutation = rng.permutation(len(held_out_similarity))
        permuted = held_out_similarity[np.ix_(permutation, permutation)]
        null[permutation_index], _ = neighborhood_recovery(
            permuted,
            consensus_similarity,
            candidate_mask,
            k=k,
        )

    p_value = float((1 + np.sum(null >= observed)) / (permutations + 1))
    null_std = float(np.std(null, ddof=1))
    return {
        "recovery": observed,
        "p_value": p_value,
        "null_mean": float(np.mean(null)),
        "null_std": null_std,
        "z_score": (
            float((observed - np.mean(null)) / null_std)
            if null_std > 0
            else float("nan")
        ),
        "per_question": per_question.tolist(),
    }


def stable_cross_topic_pairs(
    dataset: FrameworkDataset,
    similarity_matrices: dict[str, np.ndarray],
    candidate_mask: np.ndarray,
    *,
    limit: int = 25,
) -> list[dict[str, object]]:
    """Highest-ranked, lowest-disagreement cross-topic pairs across models."""

    triangle = np.triu(candidate_mask, k=1)
    first_indices, second_indices = np.where(triangle)
    if len(first_indices) == 0:
        return []

    model_values = np.vstack(
        [
            similarity_matrices[model][first_indices, second_indices]
            for model in MODELS
        ]
    )
    percentile_ranks = np.vstack(
        [
            (rankdata(values, method="average") - 1) / (len(values) - 1)
            for values in model_values
        ]
    )
    mean_percentile = np.mean(percentile_ranks, axis=0)
    rank_std = np.std(percentile_ranks, axis=0)
    mean_similarity = np.mean(model_values, axis=0)
    # Stability is part of selection: a pair must be high for most models,
    # rather than extreme for only one.
    selection_score = mean_percentile - 0.50 * rank_std
    order = np.argsort(-selection_score, kind="stable")[:limit]

    rows: list[dict[str, object]] = []
    for rank, position in enumerate(order, start=1):
        first = int(first_indices[position])
        second = int(second_indices[position])
        rows.append(
            {
                "rank": rank,
                "question_id_1": int(dataset.question_ids[first]),
                "question_id_2": int(dataset.question_ids[second]),
                "domain_1": dataset.domains[first],
                "domain_2": dataset.domains[second],
                "conflict_1": dataset.conflicts[first],
                "conflict_2": dataset.conflicts[second],
                "question_1": dataset.question_texts[first],
                "question_2": dataset.question_texts[second],
                "mean_residual_cosine": float(mean_similarity[position]),
                "mean_within_model_percentile": float(mean_percentile[position]),
                "cross_model_rank_std": float(rank_std[position]),
                "selection_score": float(selection_score[position]),
                "model_cosines": {
                    model: float(model_values[model_index, position])
                    for model_index, model in enumerate(MODELS)
                },
            }
        )
    return rows


def split_half_pair_validation(
    similarity_matrices: dict[str, np.ndarray],
    candidate_mask: np.ndarray,
    *,
    selected_pairs: int = 25,
    permutations: int = 999,
    random_state: int = 42,
) -> dict[str, object]:
    """Select high-similarity pairs in three models and validate in the other three."""

    triangle = np.triu(candidate_mask, k=1)
    first_indices, second_indices = np.where(triangle)
    model_values = {
        model: similarity_matrices[model][first_indices, second_indices]
        for model in MODELS
    }
    model_percentiles = {
        model: (rankdata(values, method="average") - 1) / (len(values) - 1)
        for model, values in model_values.items()
    }

    directed_splits = []
    first_model = MODELS[0]
    for partners in combinations(MODELS[1:], 2):
        first_half = (first_model, *partners)
        second_half = tuple(model for model in MODELS if model not in first_half)
        directed_splits.append((first_half, second_half))
        directed_splits.append((second_half, first_half))

    rows = []
    selected_positions_by_split = []
    for discovery, validation in directed_splits:
        discovery_scores = np.mean(
            np.vstack([model_percentiles[model] for model in discovery]),
            axis=0,
        )
        count = min(selected_pairs, len(discovery_scores))
        selected = np.argsort(-discovery_scores, kind="stable")[:count]
        validation_scores = np.mean(
            np.vstack([model_percentiles[model] for model in validation]),
            axis=0,
        )
        rows.append(
            {
                "discovery_models": list(discovery),
                "validation_models": list(validation),
                "selected_pairs": count,
                "mean_discovery_percentile": float(
                    np.mean(discovery_scores[selected])
                ),
                "mean_validation_percentile": float(
                    np.mean(validation_scores[selected])
                ),
            }
        )
        selected_positions_by_split.append(selected)

    observed = float(
        np.mean([float(row["mean_validation_percentile"]) for row in rows])
    )
    rng = np.random.default_rng(random_state)
    null = np.empty(permutations, dtype=np.float64)
    for permutation_index in range(permutations):
        # One coherent node relabeling is shared by all models and all 20
        # overlapping splits. This preserves their dependence and the
        # cross-model agreement structure inside every null draw.
        permutation = rng.permutation(len(candidate_mask))
        permuted_percentiles = {}
        for model in MODELS:
            permuted = similarity_matrices[model][
                np.ix_(permutation, permutation)
            ]
            values = permuted[first_indices, second_indices]
            permuted_percentiles[model] = (
                rankdata(values, method="average") - 1
            ) / (len(values) - 1)

        split_scores = []
        for split_index, (_, validation) in enumerate(directed_splits):
            validation_mean = np.mean(
                np.vstack(
                    [permuted_percentiles[model] for model in validation]
                ),
                axis=0,
            )
            split_scores.append(
                float(
                    np.mean(
                        validation_mean[selected_positions_by_split[split_index]]
                    )
                )
            )
        null[permutation_index] = np.mean(split_scores)

    p_value = float((1 + np.sum(null >= observed)) / (permutations + 1))
    return {
        "method": "directed 3-model discovery / 3-model validation",
        "selected_pairs_per_split": selected_pairs,
        "directed_split_count": len(rows),
        "splits": rows,
        "mean_validation_percentile": observed,
        "minimum_split_validation_percentile": float(
            np.min([float(row["mean_validation_percentile"]) for row in rows])
        ),
        "permutation_null_mean": float(np.mean(null)),
        "permutation_p_value": p_value,
    }


def run_neighborhood_analysis(
    dataset: FrameworkDataset,
    *,
    permutations: int = 999,
    random_state: int = 42,
) -> dict[str, object]:
    """Run held-out neighbor recovery and extract interpretable pair examples."""

    candidate_mask, question_similarity_cutoff = cross_topic_mask(
        dataset,
        question_similarity_quantile=0.25,
        require_different_source=True,
    )
    matrices = {
        model: cosine_matrix(dataset.residuals[model]) for model in MODELS
    }

    primary_rows = []
    for model_index, held_out_model in enumerate(MODELS):
        consensus = np.mean(
            np.stack(
                [
                    matrices[model]
                    for model in MODELS
                    if model != held_out_model
                ]
            ),
            axis=0,
        )
        primary_rows.append(
            {
                "model": held_out_model,
                **recovery_permutation_test(
                    matrices[held_out_model],
                    consensus,
                    candidate_mask,
                    k=5,
                    permutations=permutations,
                    random_state=random_state + 1000 * model_index,
                ),
            }
        )

    adjusted = benjamini_hochberg(
        [float(row["p_value"]) for row in primary_rows]
    )
    holm = holm_bonferroni(
        [float(row["p_value"]) for row in primary_rows]
    )
    for row, q_value, holm_value in zip(
        primary_rows,
        adjusted,
        holm,
        strict=True,
    ):
        row["q_value_bh"] = q_value
        row["p_value_holm"] = holm_value

    sensitivity = []
    for k in (3, 5, 10):
        recoveries = []
        for held_out_model in MODELS:
            consensus = np.mean(
                np.stack(
                    [
                        matrices[model]
                        for model in MODELS
                        if model != held_out_model
                    ]
                ),
                axis=0,
            )
            recovery, _ = neighborhood_recovery(
                matrices[held_out_model],
                consensus,
                candidate_mask,
                k=k,
            )
            recoveries.append(recovery)
        sensitivity.append(
            {
                "k": k,
                "mean_recovery": float(np.mean(recoveries)),
                "min_recovery": float(np.min(recoveries)),
                "max_recovery": float(np.max(recoveries)),
            }
        )

    candidate_counts = np.sum(candidate_mask, axis=1)
    analytic_chance = float(
        np.mean(
            [
                min(5, int(count)) / int(count)
                for count in candidate_counts
                if count > 0
            ]
        )
    )

    return {
        "method": "leave-one-model-out cross-topic nearest-neighbor recovery",
        "primary_k": 5,
        "question_similarity_cutoff": question_similarity_cutoff,
        "candidate_pair_count": int(np.sum(np.triu(candidate_mask, k=1))),
        "candidate_count_min": int(np.min(candidate_counts)),
        "candidate_count_median": float(np.median(candidate_counts)),
        "candidate_count_max": int(np.max(candidate_counts)),
        "analytic_random_recovery": analytic_chance,
        "held_out_models": primary_rows,
        "mean_recovery": float(
            np.mean([float(row["recovery"]) for row in primary_rows])
        ),
        "mean_permutation_null": float(
            np.mean([float(row["null_mean"]) for row in primary_rows])
        ),
        "sensitivity": sensitivity,
        "split_half_pair_validation": split_half_pair_validation(
            matrices,
            candidate_mask,
            selected_pairs=25,
            permutations=permutations,
            random_state=random_state + 30_000,
        ),
        "stable_cross_topic_pairs": stable_cross_topic_pairs(
            dataset,
            matrices,
            candidate_mask,
            limit=25,
        ),
    }
