"""Question-level sign-flip and repeated-measures uncertainty utilities."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from itertools import product
from typing import Literal

import numpy as np


Alternative = Literal["two-sided", "greater", "less"]


def _validate_p_values(p_values: Sequence[float]) -> np.ndarray:
    values = np.asarray(p_values, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("p_values must be one-dimensional")
    if not np.isfinite(values).all() or np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("p_values must be finite values between zero and one")
    return values


def benjamini_hochberg(p_values: Sequence[float]) -> list[float]:
    """Return Benjamini-Hochberg FDR-adjusted p-values."""

    values = _validate_p_values(p_values)
    if values.size == 0:
        return []
    order = np.argsort(values, kind="stable")
    ranked = values[order]
    adjusted = ranked * len(values) / np.arange(1, len(values) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    output = np.empty_like(adjusted)
    output[order] = np.clip(adjusted, 0.0, 1.0)
    return output.tolist()


def holm_bonferroni(p_values: Sequence[float]) -> list[float]:
    """Return Holm familywise-error adjusted p-values."""

    values = _validate_p_values(p_values)
    if values.size == 0:
        return []
    order = np.argsort(values, kind="stable")
    ranked = values[order]
    adjusted = (len(values) - np.arange(len(values))) * ranked
    adjusted = np.maximum.accumulate(adjusted)
    output = np.empty_like(adjusted)
    output[order] = np.clip(adjusted, 0.0, 1.0)
    return output.tolist()


def aggregate_by_question(
    paired_values: np.ndarray,
    *,
    question_axis: int = -1,
) -> np.ndarray:
    """Average all repeated observations within each question.

    This helper makes the inferential unit explicit: models are repeated views,
    while the ten scenarios are the independently sign-flipped units.
    """

    values = np.asarray(paired_values, dtype=np.float64)
    if values.ndim < 1 or not np.isfinite(values).all():
        raise ValueError("paired_values must be a finite non-empty array")
    axis = question_axis if question_axis >= 0 else values.ndim + question_axis
    if not 0 <= axis < values.ndim:
        raise ValueError(
            f"question_axis {question_axis} is invalid for {values.ndim} dimensions"
        )
    moved = np.moveaxis(values, axis, -1)
    if moved.shape[-1] == 0:
        raise ValueError("question axis cannot be empty")
    if moved.ndim == 1:
        return moved.copy()
    return np.mean(moved, axis=tuple(range(moved.ndim - 1)))


def exact_paired_sign_flip(
    question_effects: Sequence[float],
    *,
    alternative: Alternative = "two-sided",
) -> dict[str, object]:
    """Exact paired sign-flip test using questions as exchangeable sign units.

    Pass one paired contrast per question, normally obtained by averaging the
    model-level paired differences with :func:`aggregate_by_question`. With ten
    questions the reference distribution contains all 2**10 sign assignments.
    """

    effects = np.asarray(question_effects, dtype=np.float64)
    if effects.ndim != 1 or effects.size == 0:
        raise ValueError("question_effects must be a non-empty 1D sequence")
    if effects.size > 20:
        raise ValueError(
            "Exact sign-flip enumeration is limited to 20 questions; "
            "use a Monte Carlo procedure for larger panels"
        )
    if not np.isfinite(effects).all():
        raise ValueError("question_effects contain NaN or infinite values")
    if alternative not in {"two-sided", "greater", "less"}:
        raise ValueError("alternative must be 'two-sided', 'greater', or 'less'")

    observed = float(np.mean(effects))
    signs = np.asarray(list(product((-1.0, 1.0), repeat=effects.size)))
    reference = np.mean(signs * effects[None, :], axis=1)
    tolerance = np.finfo(np.float64).eps * max(1.0, abs(observed)) * 16.0
    if alternative == "two-sided":
        extreme = np.abs(reference) >= abs(observed) - tolerance
    elif alternative == "greater":
        extreme = reference >= observed - tolerance
    else:
        extreme = reference <= observed + tolerance

    return {
        "observed_mean": observed,
        "p_value": float(np.mean(extreme)),
        "alternative": alternative,
        "n_questions": int(effects.size),
        "n_assignments": int(reference.size),
        "exact": True,
        "question_effects": effects.tolist(),
    }


def two_way_bootstrap(
    values: np.ndarray,
    *,
    statistic: Callable[[np.ndarray], float] = np.mean,
    n_bootstrap: int = 2_000,
    confidence_level: float = 0.95,
    seed: int = 20260801,
) -> dict[str, float | int]:
    """Independently resample model and question axes of a crossed panel.

    ``values`` must be a finite ``[model, question]`` matrix. This crossed
    bootstrap reflects uncertainty across both observed model and scenario
    samples without pretending their individual cells are independent.
    """

    panel = np.asarray(values, dtype=np.float64)
    if panel.ndim != 2 or min(panel.shape) == 0:
        raise ValueError("values must be a non-empty [model, question] matrix")
    if not np.isfinite(panel).all():
        raise ValueError("values contain NaN or infinite values")
    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be positive")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be between zero and one")

    estimate = float(statistic(panel))
    if not np.isfinite(estimate):
        raise ValueError("statistic returned a non-finite estimate")
    rng = np.random.default_rng(seed)
    draws = np.empty(n_bootstrap, dtype=np.float64)
    n_models, n_questions = panel.shape
    for draw in range(n_bootstrap):
        model_indices = rng.integers(0, n_models, size=n_models)
        question_indices = rng.integers(0, n_questions, size=n_questions)
        draws[draw] = float(statistic(panel[np.ix_(model_indices, question_indices)]))
    if not np.isfinite(draws).all():
        raise ValueError("statistic returned a non-finite bootstrap value")

    alpha = 1.0 - confidence_level
    lower, upper = np.quantile(draws, [alpha / 2.0, 1.0 - alpha / 2.0])
    return {
        "estimate": estimate,
        "ci_lower": float(lower),
        "ci_upper": float(upper),
        "standard_error": float(np.std(draws, ddof=1)) if n_bootstrap > 1 else 0.0,
        "confidence_level": float(confidence_level),
        "n_bootstrap": int(n_bootstrap),
        "n_models": int(n_models),
        "n_questions": int(n_questions),
        "seed": int(seed),
    }
