"""Secondary descriptions of movement toward peer baselines and convergence.

These diagnostics ask whether reconsidered answers move toward the other
models' *initial* answers.  They do not assume that peer consensus is correct,
the prompts never expose those peer answers to the responding model, and the
results are neither tests of conformity nor direct measures of integrity.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np

from ..tools.data import IntegrityDataset


def _exact_sign_flip(values: np.ndarray) -> float:
    """Exact two-sided sign-flip p-value for question-level effects."""

    values = np.asarray(values, dtype=np.float64)
    observed = abs(float(np.mean(values)))
    assignments = np.arange(1 << len(values), dtype=np.uint64)[:, None]
    bits = (assignments >> np.arange(len(values), dtype=np.uint64)) & 1
    signs = np.where(bits == 0, -1.0, 1.0)
    null = np.mean(signs * values[None, :], axis=1)
    return float(np.mean(np.abs(null) >= observed - 1e-15))


def _holm(p_values: list[float]) -> list[float]:
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values), dtype=np.float64)
    running = 0.0
    total = len(p_values)
    for rank, position in enumerate(order):
        running = max(running, (total - rank) * p_values[int(position)])
        adjusted[int(position)] = min(1.0, running)
    return adjusted.tolist()


def _normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        raise ValueError("Cannot normalize a zero peer-consensus vector")
    return vector / norm


def run_consensus_movement(dataset: IntegrityDataset) -> dict[str, object]:
    """Measure movement toward other-model initial answers after feedback."""

    conditions = list(dataset.conditions)
    initial_index = conditions.index("initial")
    agreement_index = conditions.index("agreement")
    helper_indices = [
        index for index, condition in enumerate(conditions)
        if condition != "initial"
    ]
    residuals = np.asarray(dataset.residuals, dtype=np.float64)
    model_count, question_count = residuals.shape[:2]

    initial_peer_similarity = np.empty(
        (model_count, question_count), dtype=np.float64
    )
    condition_peer_similarity = np.empty(
        (model_count, question_count, len(conditions)), dtype=np.float64
    )
    for model_index in range(model_count):
        other_models = [
            index for index in range(model_count) if index != model_index
        ]
        for question_index in range(question_count):
            peer_centroid = _normalize(
                np.mean(
                    residuals[
                        other_models,
                        question_index,
                        initial_index,
                        :,
                    ],
                    axis=0,
                )
            )
            initial_peer_similarity[model_index, question_index] = float(
                np.dot(
                    residuals[model_index, question_index, initial_index],
                    peer_centroid,
                )
            )
            condition_peer_similarity[model_index, question_index] = (
                residuals[model_index, question_index] @ peer_centroid
            )

    movement = (
        condition_peer_similarity
        - initial_peer_similarity[:, :, None]
    )
    condition_rows = []
    for condition_index in helper_indices:
        values = movement[:, :, condition_index]
        question_effects = np.mean(values, axis=0)
        condition_rows.append(
            {
                "condition": conditions[condition_index],
                "mean_initial_peer_similarity": float(
                    np.mean(initial_peer_similarity)
                ),
                "mean_followup_peer_similarity": float(
                    np.mean(condition_peer_similarity[:, :, condition_index])
                ),
                "mean_change_toward_peer_initial_consensus": float(
                    np.mean(values)
                ),
                "median_change": float(np.median(values)),
                "positive_cell_fraction": float(np.mean(values > 0.0)),
                "question_level_sign_flip_p": _exact_sign_flip(
                    question_effects
                ),
            }
        )

    agreement_movement = movement[:, :, agreement_index]
    contrast_rows = []
    for condition_index in helper_indices:
        if condition_index == agreement_index:
            continue
        difference = (
            movement[:, :, condition_index] - agreement_movement
        )
        contrast_rows.append(
            {
                "condition": conditions[condition_index],
                "mean_change_relative_to_agreement": float(
                    np.mean(difference)
                ),
                "question_level_sign_flip_p": _exact_sign_flip(
                    np.mean(difference, axis=0)
                ),
            }
        )
    adjusted = _holm(
        [row["question_level_sign_flip_p"] for row in contrast_rows]
    )
    for row, value in zip(contrast_rows, adjusted):
        row["holm_adjusted_p"] = float(value)

    model_pairs = list(combinations(range(model_count), 2))
    initial_convergence = np.empty(question_count, dtype=np.float64)
    convergence_by_condition = np.empty(
        (question_count, len(conditions)), dtype=np.float64
    )
    for question_index in range(question_count):
        initial_convergence[question_index] = np.mean(
            [
                np.dot(
                    residuals[first, question_index, initial_index],
                    residuals[second, question_index, initial_index],
                )
                for first, second in model_pairs
            ]
        )
        for condition_index in helper_indices:
            convergence_by_condition[question_index, condition_index] = (
                np.mean(
                    [
                        np.dot(
                            residuals[first, question_index, condition_index],
                            residuals[second, question_index, condition_index],
                        )
                        for first, second in model_pairs
                    ]
                )
            )

    convergence_rows = []
    for condition_index in helper_indices:
        difference = (
            convergence_by_condition[:, condition_index]
            - initial_convergence
        )
        convergence_rows.append(
            {
                "condition": conditions[condition_index],
                "mean_initial_cross_model_similarity": float(
                    np.mean(initial_convergence)
                ),
                "mean_followup_cross_model_similarity": float(
                    np.mean(convergence_by_condition[:, condition_index])
                ),
                "mean_change_in_cross_model_similarity": float(
                    np.mean(difference)
                ),
                "question_level_sign_flip_p": _exact_sign_flip(difference),
            }
        )

    return {
        "method": (
            "leave-one-model-out initial-consensus movement and paired "
            "cross-model convergence"
        ),
        "condition_summary": condition_rows,
        "condition_vs_agreement": contrast_rows,
        "cross_model_convergence": convergence_rows,
        "interpretation_boundary": (
            "Movement toward unseen peer initial answers is descriptive geometry, "
            "not a conformity test. Peer consensus is not a normative ground "
            "truth, and whole-response movement can reflect rhetoric or elaboration."
        ),
    }
