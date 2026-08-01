"""Robustness and influence checks for semantic revision effects."""

from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr

from ..tools.data import IntegrityDataset


def _normalize_rows(array: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(array, axis=-1, keepdims=True)
    if np.any(norms <= 1e-12):
        raise ValueError("A robustness transform produced a zero vector")
    return array / norms


def _distance_tensor(
    representations: np.ndarray,
    initial_index: int,
    *,
    angular: bool = False,
) -> np.ndarray:
    similarities = np.sum(
        representations
        * representations[:, :, initial_index : initial_index + 1, :],
        axis=-1,
    )
    similarities = np.clip(similarities, -1.0, 1.0)
    if angular:
        return np.degrees(np.arccos(similarities))
    return 1.0 - similarities


def _opposition_effect(
    distances: np.ndarray,
    conditions: list[str],
) -> tuple[float, np.ndarray]:
    agreement_index = conditions.index("agreement")
    opposition_indices = [
        index for index, condition in enumerate(conditions)
        if condition not in {"initial", "agreement"}
    ]
    block_effects = (
        np.mean(distances[:, :, opposition_indices], axis=2)
        - distances[:, :, agreement_index]
    )
    return float(np.mean(block_effects)), block_effects


def _remove_full_question_span(
    responses: np.ndarray,
    questions: np.ndarray,
) -> tuple[np.ndarray, int]:
    _, singular_values, right = np.linalg.svd(
        questions, full_matrices=False
    )
    tolerance = singular_values[0] * max(questions.shape) * np.finfo(float).eps
    rank = int(np.sum(singular_values > tolerance))
    basis = right[:rank]
    flat = responses.reshape(-1, responses.shape[-1])
    projected = flat - (flat @ basis.T) @ basis
    return _normalize_rows(projected).reshape(responses.shape), rank


def run_robustness_checks(dataset: IntegrityDataset) -> dict[str, object]:
    """Evaluate preprocessing, metric, length, and influence sensitivity."""

    conditions = list(dataset.conditions)
    initial_index = conditions.index("initial")
    residuals = np.asarray(dataset.residuals, dtype=np.float64)
    raw = np.asarray(dataset.raw_responses, dtype=np.float64)

    raw_distances = _distance_tensor(raw, initial_index)
    canonical_distances = _distance_tensor(residuals, initial_index)
    angular_distances = _distance_tensor(
        residuals, initial_index, angular=True
    )
    full_span, question_rank = _remove_full_question_span(
        raw, np.asarray(dataset.question_embeddings, dtype=np.float64)
    )
    full_span_distances = _distance_tensor(full_span, initial_index)

    centered = np.empty_like(residuals)
    for model_index in range(residuals.shape[0]):
        for condition_index in range(residuals.shape[2]):
            values = residuals[:, :, :, :][
                model_index, :, condition_index, :
            ]
            centered[model_index, :, condition_index, :] = (
                _normalize_rows(values - np.mean(values, axis=0))
            )
    centered_distances = _distance_tensor(centered, initial_index)

    checks = []
    block_effects_by_name = {}
    for name, distances, scale in (
        ("raw normalized embeddings", raw_distances, "1 - cosine"),
        (
            "exact paired-question projection",
            canonical_distances,
            "1 - cosine",
        ),
        (
            "remove full ten-question row span",
            full_span_distances,
            "1 - cosine",
        ),
        (
            "model-condition centered scenario patterns",
            centered_distances,
            "1 - cosine",
        ),
        (
            "angular distance after paired projection",
            angular_distances,
            "degrees",
        ),
    ):
        effect, block_effects = _opposition_effect(distances, conditions)
        block_effects_by_name[name] = block_effects
        checks.append(
            {
                "check": name,
                "scale": scale,
                "opposition_minus_agreement_effect": effect,
                "positive_question_fraction": float(
                    np.mean(np.mean(block_effects, axis=0) > 0.0)
                ),
                "positive_model_fraction": float(
                    np.mean(np.mean(block_effects, axis=1) > 0.0)
                ),
            }
        )

    canonical_effects = block_effects_by_name[
        "exact paired-question projection"
    ]
    leave_one_model_out = []
    for model_index, model in enumerate(dataset.models):
        keep = np.arange(len(dataset.models)) != model_index
        leave_one_model_out.append(
            {
                "omitted_model": model,
                "effect": float(np.mean(canonical_effects[keep])),
            }
        )
    leave_one_question_out = []
    for question_index, question_id in enumerate(dataset.question_ids):
        keep = np.arange(len(dataset.question_ids)) != question_index
        leave_one_question_out.append(
            {
                "omitted_question_id": int(question_id),
                "effect": float(np.mean(canonical_effects[:, keep])),
            }
        )

    non_claude = np.asarray(
        [not model.startswith("claude_") for model in dataset.models]
    )
    response_word_counts = np.vectorize(
        lambda value: len(str(value).split()), otypes=[float]
    )(dataset.response_texts)
    initial_words = response_word_counts[:, :, initial_index, None]
    absolute_log_word_change = np.abs(
        np.log((response_word_counts + 1.0) / (initial_words + 1.0))
    )
    helper_mask = [
        index for index, condition in enumerate(conditions)
        if condition != "initial"
    ]
    raw_helper = raw_distances[:, :, helper_mask].ravel()
    residual_helper = canonical_distances[:, :, helper_mask].ravel()
    length_helper = absolute_log_word_change[:, :, helper_mask].ravel()

    paired_question_cosines = np.sum(
        residuals
        * dataset.question_embeddings[None, :, None, :],
        axis=-1,
    )
    raw_question_cosines = np.sum(
        raw * dataset.question_embeddings[None, :, None, :], axis=-1
    )
    retained_norms = np.sqrt(
        np.maximum(0.0, 1.0 - raw_question_cosines**2)
    )

    return {
        "method": (
            "raw/projection/span/centering/metric and leave-one-unit "
            "sensitivity checks"
        ),
        "checks": checks,
        "full_question_span_rank": question_rank,
        "raw_vs_projected_cellwise_spearman": float(
            spearmanr(raw_helper, residual_helper).statistic
        ),
        "semantic_revision_vs_length_change_spearman": float(
            spearmanr(residual_helper, length_helper).statistic
        ),
        "max_absolute_residual_question_cosine": float(
            np.max(np.abs(paired_question_cosines))
        ),
        "projection_retained_norm_mean": float(np.mean(retained_norms)),
        "projection_retained_norm_minimum": float(np.min(retained_norms)),
        "leave_one_model_out": leave_one_model_out,
        "leave_one_question_out": leave_one_question_out,
        "omit_both_claude_models_effect": float(
            np.mean(canonical_effects[non_claude])
        ),
        "interpretation_boundary": (
            "Length is measured after treatment and is therefore only a "
            "diagnostic, not a covariate used to identify the primary effect."
        ),
    }
