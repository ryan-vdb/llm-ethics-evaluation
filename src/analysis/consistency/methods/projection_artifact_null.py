"""A conservative null for geometry induced by the projection operation itself.

Every model is transformed with the same question-specific projection operator.
That common operation could induce alignment even if answers did not correspond
to the same ethical scenario. This null breaks scenario-level correspondence
while preserving:

- each model's original answer vectors;
- broad topic blocks learned from question embeddings only;
- the same exact projection and normalization operation; and
- each permuted model's internal geometric dependencies.

Answers are independently shuffled among questions *within* broad topic blocks,
then reprojected against their new target question. The test statistic is the
mean leave-one-model-out, topic-controlled RSA.
"""

from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans

from ..tools.data import MODELS, FrameworkDataset
from ..tools.geometry import (
    cosine_matrix,
    cross_topic_mask,
)
from ..tools.statistics import partial_spearman_rdm


def topic_blocks(
    question_embeddings: np.ndarray,
    *,
    n_blocks: int = 6,
    random_state: int = 42,
) -> np.ndarray:
    """Broad topic blocks learned without reference to model answers."""

    if not 2 <= n_blocks < len(question_embeddings):
        raise ValueError("n_blocks must be between 2 and n_questions - 1")
    return KMeans(
        n_clusters=n_blocks,
        n_init=50,
        random_state=random_state,
    ).fit_predict(question_embeddings)


def within_block_permutation(
    blocks: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Independently shuffle indices inside every topic block."""

    permutation = np.arange(len(blocks))
    for block in np.unique(blocks):
        positions = np.flatnonzero(blocks == block)
        permutation[positions] = rng.permutation(positions)
    return permutation


def reprojected_similarity_fast(
    answers: np.ndarray,
    questions: np.ndarray,
    permutation: np.ndarray,
) -> np.ndarray:
    """Similarity after re-pairing and exact projection, without 3072-D products.

    Both input matrices must already contain unit rows. The algebra expands the
    residual Gram matrix from three 93x93 Gram matrices, reducing each null draw
    from a high-dimensional matrix multiplication to small indexed operations.
    """

    answer_gram = answers @ answers.T
    answer_question = answers @ questions.T
    question_gram = questions @ questions.T

    assigned_answer_gram = answer_gram[np.ix_(permutation, permutation)]
    assigned_cross = answer_question[permutation, :]
    coefficients = assigned_cross[np.arange(len(permutation)), np.arange(len(permutation))]

    numerator = (
        assigned_answer_gram
        - coefficients[:, None] * assigned_cross.T
        - assigned_cross * coefficients[None, :]
        + np.outer(coefficients, coefficients) * question_gram
    )
    residual_norms = np.sqrt(np.clip(1.0 - coefficients**2, 1e-15, None))
    similarity = numerator / np.outer(residual_norms, residual_norms)
    np.fill_diagonal(similarity, 1.0)
    return np.clip(similarity, -1.0, 1.0)


def mean_leave_one_model_out_rsa(
    matrices: dict[str, np.ndarray],
    question_similarity: np.ndarray,
    mask: np.ndarray,
) -> float:
    """Mean partial RSA when each model is predicted from the other five."""

    effects = []
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
        effects.append(
            partial_spearman_rdm(
                matrices[held_out_model],
                consensus,
                question_similarity,
                mask,
            )
        )
    return float(np.mean(effects))


def run_projection_artifact_null(
    dataset: FrameworkDataset,
    *,
    permutations: int = 999,
    n_topic_blocks: int = 6,
    random_state: int = 42,
) -> dict[str, object]:
    """Compare observed shared geometry with re-pair-and-reproject null draws."""

    question_similarity = cosine_matrix(dataset.question_embeddings)
    mask, cutoff = cross_topic_mask(
        dataset,
        question_similarity_quantile=0.25,
        require_different_source=True,
    )
    observed_matrices = {
        model: cosine_matrix(dataset.residuals[model]) for model in MODELS
    }
    observed = mean_leave_one_model_out_rsa(
        observed_matrices,
        question_similarity,
        mask,
    )

    def draw_null(block_count: int) -> dict[str, object]:
        blocks = topic_blocks(
            dataset.question_embeddings,
            n_blocks=block_count,
            random_state=random_state,
        )
        block_sizes = {
            int(block): int(np.sum(blocks == block))
            for block in np.unique(blocks)
        }
        rng = np.random.default_rng(random_state + 50_000 + block_count)
        null = np.empty(permutations, dtype=np.float64)
        for permutation_index in range(permutations):
            null_matrices = {}
            for model in MODELS:
                permutation = within_block_permutation(blocks, rng)
                null_matrices[model] = reprojected_similarity_fast(
                    dataset.raw_responses[model],
                    dataset.question_embeddings,
                    permutation,
                )
            null[permutation_index] = mean_leave_one_model_out_rsa(
                null_matrices,
                question_similarity,
                mask,
            )
        p_value = float((1 + np.sum(null >= observed)) / (permutations + 1))
        null_std = float(np.std(null, ddof=1))
        return {
            "topic_blocks": block_count,
            "topic_block_sizes": block_sizes,
            "p_value": p_value,
            "null_mean": float(np.mean(null)),
            "null_std": null_std,
            "null_95_percentile": float(np.quantile(null, 0.95)),
            "null_99_percentile": float(np.quantile(null, 0.99)),
            "z_score": (
                float((observed - np.mean(null)) / null_std)
                if null_std > 0
                else float("nan")
            ),
        }

    block_counts = tuple(dict.fromkeys((4, 6, 8, 12, n_topic_blocks)))
    block_results = [draw_null(count) for count in block_counts]
    primary = next(
        row for row in block_results if row["topic_blocks"] == n_topic_blocks
    )
    return {
        "method": "independent within-topic re-pairing followed by re-projection",
        "observed_mean_held_out_rho": observed,
        "permutations": permutations,
        **primary,
        "block_count_sensitivity": block_results,
        "question_similarity_cutoff": cutoff,
        "pair_count": int(np.sum(np.triu(mask, k=1))),
        "interpretation": (
            "The null approximately preserves coarse question-topic block "
            "assignment and raw-answer marginals, and retains the shared "
            "projection operation, while destroying scenario-level alignment."
        ),
    }
