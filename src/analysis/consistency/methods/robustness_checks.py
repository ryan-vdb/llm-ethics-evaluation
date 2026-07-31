"""Sensitivity checks for the shared-geometry result.

The primary operation removes only the direction of each paired question. This
module deliberately applies stronger or alternative removals and asks how much
of the held-out cross-topic RSA effect survives:

- remove the full 93-question linear row-span from every answer;
- remove each model's generic answer centroid, then paired-question projection;
- omit the unusually long Gemini generation at question 37;
- leave every source out in turn; and
- vary projection strength for comparison with older exploratory code.
"""

from __future__ import annotations

import numpy as np

from ..tools.data import MODELS, FrameworkDataset
from ..tools.geometry import (
    cosine_matrix,
    cross_topic_mask,
    normalize_rows,
)
from ..tools.statistics import partial_spearman_rdm


def _held_out_effects(
    views: dict[str, np.ndarray],
    question_similarity: np.ndarray,
    mask: np.ndarray,
) -> list[float]:
    matrices = {model: cosine_matrix(views[model]) for model in MODELS}
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
    return effects


def _summarize(
    name: str,
    effects: list[float],
    baseline_mean: float,
    **metadata,
) -> dict[str, object]:
    mean_effect = float(np.mean(effects))
    return {
        "check": name,
        "mean_held_out_rho": mean_effect,
        "min_held_out_rho": float(np.min(effects)),
        "max_held_out_rho": float(np.max(effects)),
        "rho_relative_to_primary": (
            mean_effect / baseline_mean if baseline_mean != 0 else float("nan")
        ),
        "held_out_rhos": {
            model: float(effect)
            for model, effect in zip(MODELS, effects, strict=True)
        },
        **metadata,
    }


def run_robustness_checks(dataset: FrameworkDataset) -> dict[str, object]:
    """Run projection, outlier, and source sensitivity analyses."""

    question_similarity = cosine_matrix(dataset.question_embeddings)
    primary_mask, cutoff = cross_topic_mask(
        dataset,
        question_similarity_quantile=0.25,
        require_different_source=True,
    )
    baseline_effects = _held_out_effects(
        dataset.residuals,
        question_similarity,
        primary_mask,
    )
    baseline_mean = float(np.mean(baseline_effects))
    rows = [
        _summarize(
            "exact paired-question projection",
            baseline_effects,
            baseline_mean,
            description="Primary transformation, strength=1.0.",
        )
    ]

    # Remove the complete linear span of all question embeddings. This is much
    # more aggressive than paired projection and can also remove genuine shared
    # ethical semantics that happen to use the same embedding directions.
    _, singular_values, right_vectors = np.linalg.svd(
        dataset.question_embeddings,
        full_matrices=False,
    )
    tolerance = singular_values[0] * 1e-10
    rank = int(np.sum(singular_values > tolerance))
    question_basis = right_vectors[:rank]
    full_span_views = {}
    for model in MODELS:
        answers = dataset.raw_responses[model]
        full_span_views[model] = normalize_rows(
            answers - (answers @ question_basis.T) @ question_basis
        )
    rows.append(
        _summarize(
            "remove full question row-span",
            _held_out_effects(
                full_span_views,
                question_similarity,
                primary_mask,
            ),
            baseline_mean,
            question_subspace_rank=rank,
            description=(
                "Projects every answer away from the complete 93-question "
                "linear span before normalization."
            ),
        )
    )

    centered_views = {}
    questions = dataset.question_embeddings
    for model in MODELS:
        centered = (
            dataset.raw_responses[model]
            - np.mean(dataset.raw_responses[model], axis=0, keepdims=True)
        )
        paired_coefficients = np.sum(centered * questions, axis=1, keepdims=True)
        centered_views[model] = normalize_rows(
            centered - paired_coefficients * questions
        )
    rows.append(
        _summarize(
            "remove model answer centroid then paired projection",
            _held_out_effects(
                centered_views,
                question_similarity,
                primary_mask,
            ),
            baseline_mean,
            description=(
                "Removes each model's common answer-centroid direction before "
                "exact paired-question projection."
            ),
        )
    )

    keep_without_37 = dataset.question_ids != 37
    index_without_37 = np.flatnonzero(keep_without_37)
    reduced_mask = primary_mask[np.ix_(index_without_37, index_without_37)]
    reduced_question_similarity = question_similarity[
        np.ix_(index_without_37, index_without_37)
    ]
    reduced_views = {
        model: dataset.residuals[model][index_without_37] for model in MODELS
    }
    rows.append(
        _summarize(
            "omit question 37 generation outlier",
            _held_out_effects(
                reduced_views,
                reduced_question_similarity,
                reduced_mask,
            ),
            baseline_mean,
            omitted_question_id=37,
            description=(
                "Drops the 21,106-character Gemini generation and the matching "
                "question from all model views."
            ),
        )
    )

    strength_rows = []
    for strength in (0.75, 0.85, 1.0):
        views = {}
        post_cosines = []
        for model in MODELS:
            answers = dataset.raw_responses[model]
            coefficients = np.sum(
                answers * dataset.question_embeddings,
                axis=1,
                keepdims=True,
            )
            residual = answers - strength * coefficients * dataset.question_embeddings
            views[model] = normalize_rows(residual)
            post_cosines.extend(
                np.sum(
                    views[model] * dataset.question_embeddings,
                    axis=1,
                ).tolist()
            )
        effects = _held_out_effects(
            views,
            question_similarity,
            primary_mask,
        )
        strength_rows.append(
            {
                "strength": strength,
                "mean_held_out_rho": float(np.mean(effects)),
                "min_held_out_rho": float(np.min(effects)),
                "max_abs_paired_question_cosine": float(
                    np.max(np.abs(post_cosines))
                ),
            }
        )

    source_rows = []
    sources = np.asarray(dataset.sources, dtype=object)
    for source in sorted(set(dataset.sources)):
        keep = sources != source
        source_mask = primary_mask & np.outer(keep, keep)
        pair_count = int(np.sum(np.triu(source_mask, k=1)))
        if pair_count < 100:
            continue
        effects = _held_out_effects(
            dataset.residuals,
            question_similarity,
            source_mask,
        )
        source_rows.append(
            {
                "omitted_source": source,
                "remaining_pair_count": pair_count,
                "mean_held_out_rho": float(np.mean(effects)),
                "min_held_out_rho": float(np.min(effects)),
            }
        )

    return {
        "method": "geometric sensitivity checks specified for this analysis",
        "primary_question_similarity_cutoff": cutoff,
        "primary_pair_count": int(np.sum(np.triu(primary_mask, k=1))),
        "checks": rows,
        "projection_strength_sensitivity": strength_rows,
        "leave_one_source_out": source_rows,
        "all_checks_retain_half_primary_rho": bool(
            all(
                float(row["rho_relative_to_primary"]) >= 0.50
                for row in rows
            )
        ),
        "all_leave_one_source_out_effects_positive": bool(
            source_rows
            and all(float(row["min_held_out_rho"]) > 0 for row in source_rows)
        ),
    }
