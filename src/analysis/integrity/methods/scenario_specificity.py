"""Scenario specificity of follow-up semantics and revision directions.

Two related tests are reported for every feedback condition.  The first asks
whether a follow-up response is semantically closer to its own initial answer
than to initial answers for other questions.  The second asks whether different
models move in aligned embedding directions for the same question, above the
alignment seen for mismatched questions.

The proximity result is partly expected from conversational anchoring and
shared wording.  Revision-direction alignment is therefore reported separately
and neither statistic is interpreted as evidence of a changed conclusion.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np

from ..tools.data import IntegrityDataset
from ..tools.geometry import condition_displacements, normalize_vectors
from ..tools.statistics import holm_bonferroni


def _monte_carlo_p(
    observed: float,
    reference: np.ndarray,
    *,
    alternative: str = "greater",
) -> float:
    tolerance = np.finfo(np.float64).eps * max(1.0, abs(observed)) * 16.0
    if alternative == "greater":
        extreme = reference >= observed - tolerance
    elif alternative == "less":
        extreme = reference <= observed + tolerance
    elif alternative == "two-sided":
        extreme = np.abs(reference) >= abs(observed) - tolerance
    else:
        raise ValueError(f"Unknown alternative: {alternative}")
    return float((1 + np.sum(extreme)) / (len(reference) + 1))


def _question_permutations(
    n_questions: int,
    n_permutations: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw unrestricted question-label permutations."""

    if n_questions < 2:
        raise ValueError("At least two questions are required")
    return np.argsort(
        rng.random((n_permutations, n_questions)),
        axis=1,
    )


def _same_question_proximity(
    initial: np.ndarray,
    follow_up: np.ndarray,
    question_permutations: np.ndarray,
) -> tuple[dict[str, object], np.ndarray]:
    """Compare matched initial/follow-up cosine with coherent mismatches."""

    n_models, n_questions, _ = initial.shape
    row_indices = np.arange(n_questions)[None, :]
    cross_matrices = np.einsum(
        "mqd,mrd->mqr",
        initial,
        follow_up,
        optimize=True,
    )
    observed_by_model = np.diagonal(cross_matrices, axis1=1, axis2=2).mean(axis=1)
    observed = float(np.mean(observed_by_model))

    null_by_model = np.empty(
        (n_models, len(question_permutations)),
        dtype=np.float64,
    )
    mismatch_by_model = np.empty(n_models, dtype=np.float64)
    off_diagonal = ~np.eye(n_questions, dtype=bool)
    for model_index in range(n_models):
        matrix = cross_matrices[model_index]
        null_by_model[model_index] = np.mean(
            matrix[row_indices, question_permutations],
            axis=1,
        )
        mismatch_by_model[model_index] = float(np.mean(matrix[off_diagonal]))
    reference = np.mean(null_by_model, axis=0)
    mismatch_mean = float(np.mean(mismatch_by_model))

    return (
        {
            "same_question_mean_cosine": observed,
            "all_mismatched_questions_mean_cosine": mismatch_mean,
            "same_minus_mismatched": observed - mismatch_mean,
            "per_model": [
                {
                    "model_index": int(index),
                    "same_question_mean_cosine": float(observed_by_model[index]),
                    "mismatched_mean_cosine": float(mismatch_by_model[index]),
                }
                for index in range(n_models)
            ],
            "permutation_p_value": _monte_carlo_p(observed, reference),
            "null_mean": float(np.mean(reference)),
            "null_95th_percentile": float(np.quantile(reference, 0.95)),
            "null_99th_percentile": float(np.quantile(reference, 0.99)),
        },
        reference,
    )


def _revision_direction_alignment(
    directions: np.ndarray,
    model_permutations: np.ndarray,
    models: tuple[str, ...],
) -> tuple[dict[str, object], np.ndarray]:
    """Test cross-model alignment of unit semantic-revision directions."""

    n_models, n_questions, _ = directions.shape
    pair_rows: list[dict[str, object]] = []
    reference_sum = np.zeros(len(model_permutations), dtype=np.float64)
    observed_values: list[float] = []
    mismatch_values: list[float] = []
    off_diagonal = ~np.eye(n_questions, dtype=bool)

    model_pairs = list(combinations(range(n_models), 2))
    for first, second in model_pairs:
        matrix = np.clip(
            directions[first] @ directions[second].T,
            -1.0,
            1.0,
        )
        observed = float(np.mean(np.diag(matrix)))
        mismatch = float(np.mean(matrix[off_diagonal]))
        observed_values.append(observed)
        mismatch_values.append(mismatch)
        pair_rows.append(
            {
                "model_1": models[first],
                "model_2": models[second],
                "same_question_mean_direction_cosine": observed,
                "mismatched_question_mean_direction_cosine": mismatch,
                "same_minus_mismatched": observed - mismatch,
            }
        )
        first_indices = model_permutations[:, first, :]
        second_indices = model_permutations[:, second, :]
        reference_sum += np.mean(
            matrix[first_indices, second_indices],
            axis=1,
        )

    reference = reference_sum / len(model_pairs)
    observed_mean = float(np.mean(observed_values))
    mismatch_mean = float(np.mean(mismatch_values))
    return (
        {
            "same_question_mean_direction_cosine": observed_mean,
            "mismatched_question_mean_direction_cosine": mismatch_mean,
            "same_minus_mismatched": observed_mean - mismatch_mean,
            "permutation_p_value": _monte_carlo_p(observed_mean, reference),
            "null_mean": float(np.mean(reference)),
            "null_95th_percentile": float(np.quantile(reference, 0.95)),
            "null_99th_percentile": float(np.quantile(reference, 0.99)),
            "model_pairs": pair_rows,
        },
        reference,
    )


def _holm_rows(rows: list[dict[str, object]]) -> None:
    adjusted = holm_bonferroni(
        [float(row["permutation_p_value"]) for row in rows]
    )
    for row, value in zip(rows, adjusted, strict=True):
        row["p_value_holm"] = float(value)


def run_scenario_specificity(
    dataset: IntegrityDataset,
    *,
    permutations: int = 9_999,
    random_state: int = 20260801,
) -> dict[str, object]:
    """Run matched-question proximity and revision-direction tests."""

    if permutations < 1:
        raise ValueError("permutations must be positive")

    n_models, n_questions, _, _ = dataset.residuals.shape
    initial_index = dataset.initial_condition_index
    initial = dataset.residuals[:, :, initial_index, :]

    rng = np.random.default_rng(random_state)
    # The same unrestricted permutation is used for every model and condition
    # in a draw, preserving cross-model dependence while breaking the pairing.
    question_permutations = _question_permutations(
        n_questions,
        permutations,
        rng,
    )
    # Each model receives a question-label permutation, and that full set is
    # reused across conditions.  Independent model labels are necessary here:
    # applying one common permutation would preserve cross-model alignment.
    model_permutations = np.argsort(
        rng.random((permutations, n_models, n_questions)),
        axis=2,
    )

    displacements = condition_displacements(
        dataset.residuals,
        initial_index=initial_index,
    )

    proximity_rows: list[dict[str, object]] = []
    proximity_nulls: dict[str, np.ndarray] = {}
    direction_rows: list[dict[str, object]] = []
    direction_nulls: dict[str, np.ndarray] = {}

    for condition in dataset.helper_conditions:
        condition_index = dataset.condition_index(condition)
        proximity, proximity_reference = _same_question_proximity(
            initial,
            dataset.residuals[:, :, condition_index, :],
            question_permutations,
        )
        proximity["condition"] = condition
        for model_index, row in enumerate(proximity["per_model"]):
            row["model"] = dataset.models[model_index]
            del row["model_index"]
        proximity_rows.append(proximity)
        proximity_nulls[condition] = proximity_reference

        condition_displacement = displacements[:, :, condition_index, :]
        norms = np.linalg.norm(condition_displacement, axis=-1)
        if np.any(norms <= 1e-12):
            locations = np.argwhere(norms <= 1e-12).tolist()
            raise ValueError(
                f"Undefined semantic-revision direction for {condition}: "
                f"near-zero displacement at {locations[:5]}"
            )
        directions = normalize_vectors(condition_displacement)
        direction, direction_reference = _revision_direction_alignment(
            directions,
            model_permutations,
            dataset.models,
        )
        direction["condition"] = condition
        direction_rows.append(direction)
        direction_nulls[condition] = direction_reference

    _holm_rows(proximity_rows)
    _holm_rows(direction_rows)

    opposition = tuple(
        condition
        for condition in dataset.helper_conditions
        if condition != "agreement"
    )
    if len(opposition) != 5:
        raise ValueError(
            "The aggregate direction test requires exactly five "
            f"non-agreement conditions, found {len(opposition)}"
        )

    direction_by_condition = {
        str(row["condition"]): row for row in direction_rows
    }
    aggregate_observed = float(
        np.mean(
            [
                float(
                    direction_by_condition[condition][
                        "same_question_mean_direction_cosine"
                    ]
                )
                for condition in opposition
            ]
        )
    )
    aggregate_mismatch = float(
        np.mean(
            [
                float(
                    direction_by_condition[condition][
                        "mismatched_question_mean_direction_cosine"
                    ]
                )
                for condition in opposition
            ]
        )
    )
    aggregate_reference = np.mean(
        np.stack([direction_nulls[condition] for condition in opposition]),
        axis=0,
    )

    return {
        "method": "scenario-specific semantic-revision analysis",
        "embedding_view": (
            "L2-normalized response embeddings after exact removal of the "
            "paired-question projection"
        ),
        "randomization": {
            "permutations": int(permutations),
            "seed": int(random_state),
            "same_question_proximity": (
                "An unrestricted question-label permutation is shared across "
                "all models and conditions within each draw; fixed points are "
                "allowed under the permutation null."
            ),
            "revision_direction_alignment": (
                "Question labels are permuted independently by model, with the "
                "same model-specific permutations reused across conditions."
            ),
            "monte_carlo_p_value_correction": "(extreme + 1) / (draws + 1)",
        },
        "same_question_semantic_proximity": proximity_rows,
        "revision_direction_alignment": {
            "aggregate_non_agreement": {
                "conditions": list(opposition),
                "same_question_mean_direction_cosine": aggregate_observed,
                "mismatched_question_mean_direction_cosine": aggregate_mismatch,
                "same_minus_mismatched": aggregate_observed - aggregate_mismatch,
                "permutation_p_value": _monte_carlo_p(
                    aggregate_observed,
                    aggregate_reference,
                ),
                "null_mean": float(np.mean(aggregate_reference)),
                "null_95th_percentile": float(
                    np.quantile(aggregate_reference, 0.95)
                ),
                "null_99th_percentile": float(
                    np.quantile(aggregate_reference, 0.99)
                ),
            },
            "by_condition": direction_rows,
        },
        "interpretation_limits": [
            (
                "Same-question proximity is partly expected because follow-up "
                "answers were directly conditioned on the initial answer and "
                "share conversational context and wording. It is a descriptive "
                "sanity check, not independent evidence of integrity."
            ),
            (
                "Direction alignment measures shared semantic movement in the "
                "embedding space; it does not identify which claims changed or "
                "whether conclusions changed."
            ),
        ],
    }
