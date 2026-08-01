"""Paired semantic-revision effects under different feedback prompts.

The outcome throughout this module is embedding displacement from a model's
own initial answer, ``1 - cosine(initial, follow-up)``.  That is a continuous
measure of semantic revision; it does not by itself establish that a model
changed its conclusion.  Agreement is the conversational control,
and the primary contrast averages the five non-agreement feedback conditions
before subtracting that control.

Questions, rather than the 60 model-question cells, are the exchangeable units
in the exact sign-flip sensitivity. A crossed model-by-question bootstrap then
describes uncertainty across both panel axes.
"""

from __future__ import annotations

from collections.abc import Sequence
from math import comb

import numpy as np

from ..tools.data import IntegrityDataset
from ..tools.geometry import initial_similarity
from ..tools.statistics import (
    aggregate_by_question,
    exact_paired_sign_flip,
    holm_bonferroni,
    two_way_bootstrap,
)


AGREEMENT_CONDITION = "agreement"
PLAIN_DISAGREEMENT_CONDITION = "disagreement"
STRONG_DISAGREEMENT_CONDITION = "strong_disagreement"
SOURCE_CUE_CONDITIONS: tuple[str, ...] = (
    "majority_opinion",
    "lived_experience",
    "expert_consensus",
)


def _semantic_revision_distances(
    dataset: IntegrityDataset,
    *,
    residual: bool,
) -> np.ndarray:
    """Return ``1-cos(initial, condition)`` with shape ``[M, Q, C]``."""

    views = dataset.residuals if residual else dataset.raw_responses
    similarities = initial_similarity(
        views,
        initial_index=dataset.initial_condition_index,
    )
    # Dot-product roundoff can put a normalized self-cosine a few ulps above 1.
    return np.clip(1.0 - similarities, 0.0, 2.0)


def _condition_indices(
    dataset: IntegrityDataset,
) -> tuple[int, tuple[str, ...], list[int]]:
    required = {
        AGREEMENT_CONDITION,
        PLAIN_DISAGREEMENT_CONDITION,
        STRONG_DISAGREEMENT_CONDITION,
        *SOURCE_CUE_CONDITIONS,
    }
    missing = sorted(required - set(dataset.helper_conditions))
    if missing:
        raise ValueError(f"Missing required feedback conditions: {missing}")

    opposition = tuple(
        condition
        for condition in dataset.helper_conditions
        if condition != AGREEMENT_CONDITION
    )
    if len(opposition) != 5:
        raise ValueError(
            "The primary contrast requires exactly five non-agreement "
            f"conditions, found {len(opposition)}"
        )
    return (
        dataset.condition_index(AGREEMENT_CONDITION),
        opposition,
        [dataset.condition_index(condition) for condition in opposition],
    )


def _question_sign_flip(
    panel: np.ndarray,
    *,
    alternative: str = "two-sided",
) -> dict[str, object]:
    """Run the exact test after collapsing repeated model observations."""

    question_effects = aggregate_by_question(panel, question_axis=1)
    return exact_paired_sign_flip(
        question_effects,
        alternative=alternative,  # type: ignore[arg-type]
    )


def _model_sign_flip_sensitivity(panel: np.ndarray) -> dict[str, object]:
    """Repeat the exact sign flip with models, rather than questions, as units."""

    model_effects = np.mean(panel, axis=1)
    result = exact_paired_sign_flip(model_effects, alternative="two-sided")
    # The shared utility is agnostic to what the vector entries represent; make
    # the sensitivity output's inferential unit unambiguous to readers.
    result["n_models"] = result.pop("n_questions")
    result["model_effects"] = result.pop("question_effects")
    result["unit"] = "model"
    return result


def _question_direction_sign_test(panel: np.ndarray) -> dict[str, object]:
    """Exact unweighted two-sided test of positive versus negative questions."""

    effects = aggregate_by_question(panel, question_axis=1)
    positive = int(np.sum(effects > 0.0))
    negative = int(np.sum(effects < 0.0))
    zero = int(np.sum(effects == 0.0))
    n_nonzero = positive + negative
    if n_nonzero == 0:
        p_value = 1.0
    else:
        smaller = min(positive, negative)
        lower_tail = sum(comb(n_nonzero, index) for index in range(smaller + 1))
        p_value = min(1.0, 2.0 * lower_tail / (2**n_nonzero))
    return {
        "positive_questions": positive,
        "negative_questions": negative,
        "zero_questions": zero,
        "n_nonzero_questions": n_nonzero,
        "p_value": float(p_value),
        "alternative": "two-sided",
        "exact": True,
        "unit": "question",
        "weighting": "sign only; question-effect magnitudes ignored",
    }


def _helper_label_exchangeability(
    dataset: IntegrityDataset,
    distances: np.ndarray,
    *,
    permutations: int,
    random_state: int,
) -> dict[str, object]:
    """Use each helper label as a possible pseudo-control within questions.

    Each randomization picks one pseudo-control label per question and uses that
    same label for all six models.  This preserves the model dependence within
    a scenario.  Because the prompts have deliberately different meanings,
    helper-label exchangeability is a finite-panel sensitivity analysis rather
    than the main inferential assumption.
    """

    if permutations < 1:
        raise ValueError("helper_label_permutations must be positive")
    helper_indices = [
        dataset.condition_index(condition)
        for condition in dataset.helper_conditions
    ]
    helper_distances = distances[:, :, helper_indices]
    n_helpers = helper_distances.shape[2]
    if n_helpers < 2:
        raise ValueError("At least two helper conditions are required")

    # For every question and candidate pseudo-control, calculate
    # mean(other labels) - pseudo-control after averaging over models.
    total = np.sum(helper_distances, axis=2, keepdims=True)
    candidate_effects = (
        (total - helper_distances) / (n_helpers - 1)
        - helper_distances
    )
    question_by_label = np.mean(candidate_effects, axis=0)
    agreement_label = dataset.helper_conditions.index(AGREEMENT_CONDITION)
    observed = float(np.mean(question_by_label[:, agreement_label]))

    rng = np.random.default_rng(random_state)
    assignments = rng.integers(
        0,
        n_helpers,
        size=(permutations, len(dataset.question_ids)),
    )
    question_indices = np.arange(len(dataset.question_ids))[None, :]
    reference = np.mean(
        question_by_label[question_indices, assignments],
        axis=1,
    )
    tolerance = np.finfo(np.float64).eps * max(1.0, abs(observed)) * 16.0
    extreme = np.abs(reference) >= abs(observed) - tolerance
    return {
        "observed_mean": observed,
        "p_value_two_sided": float(
            (1 + np.sum(extreme)) / (permutations + 1)
        ),
        "n_randomizations": int(permutations),
        "seed": int(random_state),
        "helper_labels": list(dataset.helper_conditions),
        "label_assignments_per_randomization": int(len(dataset.question_ids)),
        "same_label_used_for_all_models_within_question": True,
        "null_mean": float(np.mean(reference)),
        "null_2_5th_percentile": float(np.quantile(reference, 0.025)),
        "null_97_5th_percentile": float(np.quantile(reference, 0.975)),
        "assumption_warning": (
            "The feedback prompts have different intended meanings, so their "
            "labels are not literally exchangeable; this is a sensitivity null."
        ),
    }


def _descriptive_condition_row(
    dataset: IntegrityDataset,
    distances: np.ndarray,
    condition: str,
) -> dict[str, object]:
    index = dataset.condition_index(condition)
    values = distances[:, :, index]
    return {
        "condition": condition,
        "mean_semantic_revision": float(np.mean(values)),
        "standard_deviation_cells": float(np.std(values, ddof=1)),
        "minimum": float(np.min(values)),
        "maximum": float(np.max(values)),
        "model_means": {
            model: float(np.mean(values[model_index]))
            for model_index, model in enumerate(dataset.models)
        },
        "n_model_question_cells": int(values.size),
    }


def _contrast_row(
    dataset: IntegrityDataset,
    distances: np.ndarray,
    *,
    numerator: str,
    denominator: str,
    label: str,
) -> tuple[dict[str, object], np.ndarray]:
    panel = (
        distances[:, :, dataset.condition_index(numerator)]
        - distances[:, :, dataset.condition_index(denominator)]
    )
    test = _question_sign_flip(panel, alternative="two-sided")
    return (
        {
            "contrast": label,
            "numerator_condition": numerator,
            "denominator_condition": denominator,
            "mean_difference": float(np.mean(panel)),
            "question_sign_flip": test,
        },
        panel,
    )


def _apply_holm(
    rows: Sequence[dict[str, object]],
) -> None:
    raw = [
        float(dict(row["question_sign_flip"])["p_value"])
        for row in rows
    ]
    adjusted = holm_bonferroni(raw)
    for row, value in zip(rows, adjusted, strict=True):
        row["p_value_holm"] = float(value)


def run_revision_effects(
    dataset: IntegrityDataset,
    *,
    bootstrap_samples: int = 5_000,
    helper_label_permutations: int = 9_999,
    random_state: int = 20260801,
) -> dict[str, object]:
    """Estimate the primary opposition-minus-agreement revision contrast."""

    if bootstrap_samples < 1:
        raise ValueError("bootstrap_samples must be positive")

    distances = _semantic_revision_distances(dataset, residual=True)
    raw_distances = _semantic_revision_distances(dataset, residual=False)
    agreement_index, opposition, opposition_indices = _condition_indices(dataset)

    agreement = distances[:, :, agreement_index]
    mean_opposition = np.mean(distances[:, :, opposition_indices], axis=2)
    primary_panel = mean_opposition - agreement
    primary_sign_flip = _question_sign_flip(
        primary_panel,
        alternative="two-sided",
    )
    direction_sign_test = _question_direction_sign_test(primary_panel)
    primary_bootstrap = two_way_bootstrap(
        primary_panel,
        n_bootstrap=bootstrap_samples,
        seed=random_state,
    )
    model_sign_flip = _model_sign_flip_sensitivity(primary_panel)
    helper_label_null = _helper_label_exchangeability(
        dataset,
        distances,
        permutations=helper_label_permutations,
        random_state=random_state + 17,
    )

    condition_rows = [
        _descriptive_condition_row(dataset, distances, condition)
        for condition in dataset.helper_conditions
    ]

    helper_rows: list[dict[str, object]] = []
    for condition in opposition:
        row, _ = _contrast_row(
            dataset,
            distances,
            numerator=condition,
            denominator=AGREEMENT_CONDITION,
            label=f"{condition} minus agreement",
        )
        helper_rows.append(row)
    _apply_holm(helper_rows)

    dose_row, _ = _contrast_row(
        dataset,
        distances,
        numerator=STRONG_DISAGREEMENT_CONDITION,
        denominator=PLAIN_DISAGREEMENT_CONDITION,
        label="strong disagreement minus plain disagreement",
    )

    source_rows: list[dict[str, object]] = []
    source_panels: list[np.ndarray] = []
    for condition in SOURCE_CUE_CONDITIONS:
        row, panel = _contrast_row(
            dataset,
            distances,
            numerator=condition,
            denominator=PLAIN_DISAGREEMENT_CONDITION,
            label=f"{condition} cue minus plain disagreement",
        )
        source_rows.append(row)
        source_panels.append(panel)
    _apply_holm(source_rows)

    combined_source_panel = np.mean(np.stack(source_panels), axis=0)
    combined_source = {
        "contrast": "mean source cue minus plain disagreement",
        "numerator_conditions": list(SOURCE_CUE_CONDITIONS),
        "denominator_condition": PLAIN_DISAGREEMENT_CONDITION,
        "mean_difference": float(np.mean(combined_source_panel)),
        "question_sign_flip": _question_sign_flip(
            combined_source_panel,
            alternative="two-sided",
        ),
    }

    model_effects = []
    for model_index, model in enumerate(dataset.models):
        model_effects.append(
            {
                "model": model,
                "opposition_minus_agreement": float(
                    np.mean(primary_panel[model_index])
                ),
                "mean_opposition_semantic_revision": float(
                    np.mean(mean_opposition[model_index])
                ),
                "agreement_semantic_revision": float(
                    np.mean(agreement[model_index])
                ),
            }
        )

    question_effects = []
    for question_index, question in enumerate(dataset.questions):
        question_effects.append(
            {
                "question_id": int(question.question_id),
                "domain": question.domain,
                "conflict": question.conflict,
                "source": question.source,
                "opposition_minus_agreement": float(
                    np.mean(primary_panel[:, question_index])
                ),
                "mean_opposition_semantic_revision": float(
                    np.mean(mean_opposition[:, question_index])
                ),
                "agreement_semantic_revision": float(
                    np.mean(agreement[:, question_index])
                ),
            }
        )

    observed_primary = float(np.mean(primary_panel))
    leave_one_model_out = []
    for model_index, model in enumerate(dataset.models):
        keep = np.arange(len(dataset.models)) != model_index
        estimate = float(np.mean(primary_panel[keep]))
        leave_one_model_out.append(
            {
                "omitted_model": model,
                "estimate": estimate,
                "change_from_full": estimate - observed_primary,
            }
        )
    leave_one_question_out = []
    for question_index, question_id in enumerate(dataset.question_ids):
        keep = np.arange(len(dataset.question_ids)) != question_index
        estimate = float(np.mean(primary_panel[:, keep]))
        leave_one_question_out.append(
            {
                "omitted_question_id": int(question_id),
                "estimate": estimate,
                "change_from_full": estimate - observed_primary,
            }
        )

    raw_agreement = raw_distances[:, :, agreement_index]
    raw_opposition = np.mean(
        raw_distances[:, :, opposition_indices],
        axis=2,
    )
    raw_panel = raw_opposition - raw_agreement

    return {
        "method": "paired semantic-revision displacement analysis",
        "outcome": {
            "name": "semantic revision distance",
            "formula": "1 - cosine(initial response, follow-up response)",
            "embedding_view": (
                "L2-normalized response embeddings after exact removal of the "
                "paired-question projection"
            ),
            "scope_warning": (
                "Embedding displacement measures semantic revision; it does not "
                "by itself establish a changed conclusion."
            ),
        },
        "design": {
            "models": list(dataset.models),
            "question_count": int(len(dataset.question_ids)),
            "feedback_conditions": list(dataset.helper_conditions),
            "agreement_control": AGREEMENT_CONDITION,
            "opposition_conditions": list(opposition),
            "primary_sign_flip_unit": "question",
            "sign_flip_assumption": (
                "scenario-level effects are sign-exchangeable under a symmetric null"
            ),
            "planned_contrast_tests": "two-sided",
        },
        "primary": {
            "contrast": "mean non-agreement feedback minus agreement",
            "mean_opposition_semantic_revision": float(np.mean(mean_opposition)),
            "agreement_semantic_revision": float(np.mean(agreement)),
            "mean_difference": observed_primary,
            "question_sign_flip": primary_sign_flip,
            "question_direction_sign_test": direction_sign_test,
            "crossed_model_question_bootstrap": primary_bootstrap,
            "model_sign_flip_sensitivity": model_sign_flip,
            "helper_label_exchangeability_sensitivity": helper_label_null,
        },
        "condition_summaries": condition_rows,
        "per_helper_vs_agreement": helper_rows,
        "secondary_contrasts": {
            "disagreement_dose": dose_row,
            "source_cues_vs_plain_disagreement": source_rows,
            "combined_source_cue": combined_source,
        },
        "heterogeneity": {
            "by_model": model_effects,
            "by_question": question_effects,
        },
        "leave_one_out": {
            "models": leave_one_model_out,
            "questions": leave_one_question_out,
            "minimum_leave_one_model_out_estimate": float(
                min(row["estimate"] for row in leave_one_model_out)
            ),
            "minimum_leave_one_question_out_estimate": float(
                min(row["estimate"] for row in leave_one_question_out)
            ),
        },
        "raw_embedding_sensitivity": {
            "embedding_view": "normalized responses before question projection",
            "mean_opposition_semantic_revision": float(np.mean(raw_opposition)),
            "agreement_semantic_revision": float(np.mean(raw_agreement)),
            "mean_difference": float(np.mean(raw_panel)),
            "question_sign_flip": _question_sign_flip(
                raw_panel,
                alternative="two-sided",
            ),
        },
    }
