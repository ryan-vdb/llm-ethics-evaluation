"""Leave-one-model-out shared latent ethical axes.

PCA is fit to the mean orthogonal residual of five models, never the held-out
sixth model. Both the five-model consensus and held-out answers are projected
onto those axes. Correlation of their question scores tests whether an axis is
recoverable in independently generated reasoning.

Axis names are deliberately post-hoc descriptive summaries of the researcher
annotations at each extreme. The question text and conflicts are emitted so a
human can decide whether an axis is ethically meaningful.
"""

from __future__ import annotations

import re
from collections import Counter

import numpy as np
from scipy.stats import rankdata
from sklearn.decomposition import PCA

from ..tools.data import MODELS, FrameworkDataset
from ..tools.geometry import normalize_rows
from ..tools.statistics import benjamini_hochberg


STOPWORDS = {
    "and",
    "the",
    "versus",
    "vs",
    "of",
    "to",
    "in",
    "individual",
    "public",
    "professional",
    "social",
    "personal",
    "ethical",
    "ethics",
}


def spearman(first: np.ndarray, second: np.ndarray) -> float:
    """Spearman correlation without scipy result-wrapper edge cases."""

    first_rank = rankdata(first, method="average")
    second_rank = rankdata(second, method="average")
    first_rank -= np.mean(first_rank)
    second_rank -= np.mean(second_rank)
    denominator = np.linalg.norm(first_rank) * np.linalg.norm(second_rank)
    if denominator == 0:
        return float("nan")
    return float(np.dot(first_rank, second_rank) / denominator)


def score_permutation_test(
    consensus_scores: np.ndarray,
    held_out_scores: np.ndarray,
    *,
    permutations: int,
    random_state: int,
) -> dict[str, float]:
    """One-sided question-label permutation test for an axis score."""

    observed = spearman(consensus_scores, held_out_scores)
    rng = np.random.default_rng(random_state)
    null = np.empty(permutations, dtype=np.float64)
    for index in range(permutations):
        null[index] = spearman(
            consensus_scores,
            held_out_scores[rng.permutation(len(held_out_scores))],
        )
    p_value = float((1 + np.sum(null >= observed)) / (permutations + 1))
    null_std = float(np.std(null, ddof=1))
    return {
        "rho": observed,
        "p_value": p_value,
        "null_mean": float(np.mean(null)),
        "null_std": null_std,
        "z_score": (
            float((observed - np.mean(null)) / null_std)
            if null_std > 0
            else float("nan")
        ),
    }


def _concepts(
    dataset: FrameworkDataset,
    positions: np.ndarray,
    *,
    limit: int = 3,
) -> list[str]:
    counts: Counter[str] = Counter()
    for position in positions:
        conflict = dataset.conflicts[int(position)]
        for token in re.findall(r"[A-Za-z][A-Za-z'-]+", conflict.lower()):
            if token not in STOPWORDS and len(token) > 2:
                counts[token] += 1
    return [word.title() for word, _ in counts.most_common(limit)]


def _question_records(
    dataset: FrameworkDataset,
    positions: np.ndarray,
    scores: np.ndarray,
) -> list[dict[str, object]]:
    return [
        {
            "question_id": int(dataset.question_ids[position]),
            "score": float(scores[position]),
            "domain": dataset.domains[position],
            "conflict": dataset.conflicts[position],
            "question": dataset.question_texts[position],
        }
        for position in positions
    ]


def run_shared_axes(
    dataset: FrameworkDataset,
    *,
    components: int = 6,
    permutations: int = 999,
    random_state: int = 42,
) -> dict[str, object]:
    """Fit interpretable consensus axes and test held-out-model recovery."""

    if not 1 <= components < len(dataset.question_ids):
        raise ValueError("components must be between 1 and n_questions - 1")

    all_model_mean = normalize_rows(
        np.mean(
            np.stack([dataset.residuals[model] for model in MODELS]),
            axis=0,
        )
    )
    all_pca = PCA(n_components=components, svd_solver="full")
    all_scores = all_pca.fit_transform(all_model_mean)

    axis_profiles = []
    for component_index in range(components):
        scores = all_scores[:, component_index]
        low_positions = np.argsort(scores, kind="stable")[:5]
        high_positions = np.argsort(-scores, kind="stable")[:5]
        low_concepts = _concepts(dataset, low_positions)
        high_concepts = _concepts(dataset, high_positions)
        axis_profiles.append(
            {
                "axis": component_index + 1,
                "explained_variance_ratio": float(
                    all_pca.explained_variance_ratio_[component_index]
                ),
                "descriptive_label": (
                    f"{' / '.join(low_concepts[:2]) or 'mixed'} ↔ "
                    f"{' / '.join(high_concepts[:2]) or 'mixed'}"
                ),
                "label_status": "post-hoc researcher-annotation summary",
                "low_concepts": low_concepts,
                "high_concepts": high_concepts,
                "low_extreme_questions": _question_records(
                    dataset,
                    low_positions,
                    scores,
                ),
                "high_extreme_questions": _question_records(
                    dataset,
                    high_positions,
                    scores,
                ),
            }
        )

    held_out_rows = []
    all_p_values: list[float] = []
    p_value_locations: list[tuple[int, int]] = []
    for model_index, held_out_model in enumerate(MODELS):
        other_models = [model for model in MODELS if model != held_out_model]
        training_consensus = normalize_rows(
            np.mean(
                np.stack([dataset.residuals[model] for model in other_models]),
                axis=0,
            )
        )
        pca = PCA(n_components=components, svd_solver="full")
        consensus_scores = pca.fit_transform(training_consensus)
        held_out_scores = pca.transform(dataset.residuals[held_out_model])

        component_rows = []
        for component_index in range(components):
            test = score_permutation_test(
                consensus_scores[:, component_index],
                held_out_scores[:, component_index],
                permutations=permutations,
                random_state=(
                    random_state
                    + 10_000 * model_index
                    + 100 * component_index
                ),
            )
            component_rows.append(
                {
                    "axis": component_index + 1,
                    "training_explained_variance_ratio": float(
                        pca.explained_variance_ratio_[component_index]
                    ),
                    **test,
                }
            )
            all_p_values.append(float(test["p_value"]))
            p_value_locations.append((model_index, component_index))
        held_out_rows.append(
            {
                "model": held_out_model,
                "axes": component_rows,
            }
        )

    adjusted = benjamini_hochberg(all_p_values)
    for (model_index, component_index), q_value in zip(
        p_value_locations,
        adjusted,
        strict=True,
    ):
        held_out_rows[model_index]["axes"][component_index][
            "q_value_bh_across_all_axes"
        ] = q_value

    tested_axes = [
        axis
        for model_row in held_out_rows
        for axis in model_row["axes"]
    ]
    fold_axis_recovery = {
        "tests": len(tested_axes),
        "mean_rho": float(
            np.mean([float(row["rho"]) for row in tested_axes])
        ),
        "min_rho": float(
            np.min([float(row["rho"]) for row in tested_axes])
        ),
        "max_rho": float(
            np.max([float(row["rho"]) for row in tested_axes])
        ),
        "tests_significant_fdr_05": int(
            np.sum(
                [
                    float(row["q_value_bh_across_all_axes"]) <= 0.05
                    for row in tested_axes
                ]
            )
        ),
        "interpretation": (
            "Each fold tests its own training-defined PCA axes. Because PCA "
            "components can swap or rotate across folds, these tests are "
            "summarized across the six-dimensional set and are not attached "
            "to the post-hoc labels from the all-model descriptive fit."
        ),
    }

    return {
        "method": "leave-one-model-out consensus PCA axis recovery",
        "component_count": components,
        "consensus_explained_variance_total": float(
            np.sum(all_pca.explained_variance_ratio_)
        ),
        "axis_profiles": axis_profiles,
        "held_out_models": held_out_rows,
        "fold_axis_recovery": fold_axis_recovery,
        "interpretation_warning": (
            "This is a secondary whole-geometry analysis, not topic-controlled. "
            "Axes are learned without ethical labels. Descriptive names are "
            "assigned afterward from extreme-question annotations and are not "
            "independent validation or mapped to fold-specific component numbers."
        ),
    }
