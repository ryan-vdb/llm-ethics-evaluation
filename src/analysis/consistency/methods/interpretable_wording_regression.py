"""Cross-fitted regression of residual geometry on interpretable wording.

This secondary method turns the answer-only NMF interpretation layer into a
dyadic regression.  For each held-out model, TF-IDF and NMF are fit only to the
other five models' question-token-removed answers.  Their mean topic profile
for each scenario defines a wording-similarity matrix that predicts the held-
out model's exactly question-orthogonalized answer geometry.

The primary coefficient and incremental R-squared therefore transfer across
models.  A second, explicitly descriptive and more flexible equation uses one
common all-answer NMF basis to estimate named topic co-activations.  It is a
separate post-hoc model, not a decomposition of the primary cross-fitted gain.
"""

from __future__ import annotations

from collections import Counter

import numpy as np

from ..tools.data import MODELS, FrameworkDataset
from ..tools.geometry import (
    cosine_matrix,
    cross_topic_mask,
    normalize_rows,
    similarity_matrices,
    upper_triangle_values,
)
from ..tools.statistics import (
    benjamini_hochberg,
    holm_bonferroni,
    spearman_rdm,
)
from .dyadic_regression import (
    fit_mrqap_effect,
    mrqap_node_bootstrap_ci,
    mrqap_permutation_test,
)
from .interpretable_reasoning_topics import (
    fit_answer_only_nmf,
    load_answer_only_documents,
)


def _standardize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    standard_deviation = float(np.std(values))
    if standard_deviation <= 1e-12:
        return np.zeros_like(values)
    return (values - np.mean(values)) / standard_deviation


def _fit_ols(target: np.ndarray, design: np.ndarray) -> tuple[np.ndarray, float]:
    coefficients, *_ = np.linalg.lstsq(design, target, rcond=None)
    fitted = design @ coefficients
    residual_sum = float(np.sum((target - fitted) ** 2))
    total_sum = float(np.sum((target - np.mean(target)) ** 2))
    r_squared = 1.0 - residual_sum / total_sum if total_sum > 0 else float("nan")
    return coefficients, r_squared


def _question_profiles(
    documents: list[dict[str, object]],
    document_topics: np.ndarray,
    *,
    question_count: int,
    expected_documents_per_question: int,
) -> np.ndarray:
    profiles = np.zeros(
        (question_count, document_topics.shape[1]),
        dtype=np.float64,
    )
    counts = np.zeros(question_count, dtype=int)
    for document, topic_weights in zip(documents, document_topics, strict=True):
        question_id = int(document["question_id"])
        profiles[question_id] += topic_weights
        counts[question_id] += 1
    if not np.all(counts == expected_documents_per_question):
        raise ValueError(
            "Answer-only NMF documents are not balanced across questions: "
            f"expected {expected_documents_per_question}, found "
            f"{sorted(set(counts.tolist()))}"
        )
    return profiles / counts[:, None]


def _fit_fold_wording_predictor(
    documents: list[dict[str, object]],
    *,
    held_out_model: str,
    question_count: int,
    components: int,
    random_state: int,
) -> tuple[np.ndarray, dict[str, object]]:
    training_documents = [
        document
        for document in documents
        if str(document["model"]) != held_out_model
    ]
    expected = question_count * (len(MODELS) - 1)
    if len(training_documents) != expected:
        raise ValueError(
            f"Expected {expected} training documents for {held_out_model}, "
            f"found {len(training_documents)}"
        )
    vectorizer, model, document_topics, tfidf = fit_answer_only_nmf(
        training_documents,
        components=components,
        random_state=random_state,
    )
    profiles = _question_profiles(
        training_documents,
        document_topics,
        question_count=question_count,
        expected_documents_per_question=len(MODELS) - 1,
    )
    normalized_profiles = normalize_rows(profiles + 1e-12)
    vocabulary = np.asarray(vectorizer.get_feature_names_out())
    component_terms = []
    for component_index in range(components):
        term_indices = np.argsort(
            -model.components_[component_index],
            kind="stable",
        )[:8]
        component_terms.append(vocabulary[term_indices].tolist())
    return cosine_matrix(normalized_profiles), {
        "held_out_model": held_out_model,
        "training_documents": len(training_documents),
        "vocabulary_size": int(tfidf.shape[1]),
        "reconstruction_error": float(model.reconstruction_err_),
        "iterations": int(model.n_iter_),
        "component_top_terms": component_terms,
    }


def _map_effect(
    model_name: str,
    test: dict[str, float | str],
    confidence_intervals: dict[str, float],
    predictor_question_rho: float,
    fold_basis: dict[str, object],
) -> dict[str, object]:
    return {
        "model": model_name,
        "standardized_wording_beta": test["standardized_consensus_beta"],
        "controls_only_r_squared": test["topic_only_r_squared"],
        "wording_full_r_squared": test["full_r_squared"],
        "wording_incremental_r_squared": test["incremental_r_squared"],
        "p_value": test["p_value"],
        "alternative": test["alternative"],
        "null_beta_mean": test["null_beta_mean"],
        "null_beta_std": test["null_beta_std"],
        "z_score": test["z_score"],
        "beta_ci_95_low": confidence_intervals["beta_ci_95_low"],
        "beta_ci_95_high": confidence_intervals["beta_ci_95_high"],
        "incremental_r2_ci_95_low": confidence_intervals[
            "incremental_r2_ci_95_low"
        ],
        "incremental_r2_ci_95_high": confidence_intervals[
            "incremental_r2_ci_95_high"
        ],
        "wording_vs_question_spearman_rho": predictor_question_rho,
        "fold_basis": fold_basis,
    }


def _descriptive_topic_attribution(
    dataset: FrameworkDataset,
    documents: list[dict[str, object]],
    residual_matrices: dict[str, np.ndarray],
    question_similarity: np.ndarray,
    mask: np.ndarray,
    *,
    components: int,
    random_state: int,
) -> dict[str, object]:
    """Describe which common NMF topic co-activations carry the association."""

    vectorizer, model, document_topics, tfidf = fit_answer_only_nmf(
        documents,
        components=components,
        random_state=random_state,
    )
    by_model: dict[str, np.ndarray] = {}
    for model_name in MODELS:
        model_indices = [
            index
            for index, document in enumerate(documents)
            if str(document["model"]) == model_name
        ]
        model_documents = [documents[index] for index in model_indices]
        by_model[model_name] = _question_profiles(
            model_documents,
            document_topics[model_indices],
            question_count=len(dataset.question_ids),
            expected_documents_per_question=1,
        )

    question_values = _standardize(
        upper_triangle_values(question_similarity, mask)
    )
    question_squared = _standardize(question_values**2)
    reduced_design = np.column_stack(
        [np.ones(len(question_values)), question_values, question_squared]
    )
    held_out_rows = []
    coefficient_rows: dict[str, list[float]] = {
        model_name: [] for model_name in MODELS
    }
    condition_numbers = []
    for held_out_model in MODELS:
        training_profile = np.mean(
            np.stack(
                [
                    by_model[model_name]
                    for model_name in MODELS
                    if model_name != held_out_model
                ]
            ),
            axis=0,
        )
        topic_columns = []
        for component_index in range(components):
            coactivation = np.outer(
                training_profile[:, component_index],
                training_profile[:, component_index],
            )
            topic_columns.append(
                _standardize(upper_triangle_values(coactivation, mask))
            )
        topic_design = np.column_stack(topic_columns)
        full_design = np.column_stack([reduced_design, topic_design])
        target = _standardize(
            upper_triangle_values(residual_matrices[held_out_model], mask)
        )
        _, reduced_r_squared = _fit_ols(target, reduced_design)
        coefficients, full_r_squared = _fit_ols(target, full_design)
        topic_coefficients = coefficients[-components:]
        coefficient_rows[held_out_model] = topic_coefficients.tolist()
        condition_numbers.append(float(np.linalg.cond(full_design[:, 1:])))
        held_out_rows.append(
            {
                "model": held_out_model,
                "controls_only_r_squared": reduced_r_squared,
                "full_topic_equation_r_squared": full_r_squared,
                "topic_equation_incremental_r_squared": float(
                    full_r_squared - reduced_r_squared
                ),
                "topic_coefficients": topic_coefficients.tolist(),
            }
        )

    vocabulary = np.asarray(vectorizer.get_feature_names_out())
    topic_rows = []
    for component_index in range(components):
        top_indices = np.argsort(
            -model.components_[component_index],
            kind="stable",
        )[:12]
        values = np.asarray(
            [
                coefficient_rows[model_name][component_index]
                for model_name in MODELS
            ],
            dtype=np.float64,
        )
        terms = vocabulary[top_indices].tolist()
        topic_rows.append(
            {
                "topic": component_index + 1,
                "label": " · ".join(terms[:3]),
                "top_terms": terms,
                "mean_standardized_beta": float(np.mean(values)),
                "minimum_standardized_beta": float(np.min(values)),
                "maximum_standardized_beta": float(np.max(values)),
                "positive_models": int(np.sum(values > 0)),
                "by_model": {
                    model_name: float(coefficient_rows[model_name][component_index])
                    for model_name in MODELS
                },
            }
        )

    return {
        "basis_fit_scope": "all response texts; descriptive attribution only",
        "vocabulary_size": int(tfidf.shape[1]),
        "reconstruction_error": float(model.reconstruction_err_),
        "iterations": int(model.n_iter_),
        "mean_design_condition_number": float(np.mean(condition_numbers)),
        "maximum_design_condition_number": float(np.max(condition_numbers)),
        "mean_controls_only_r_squared": float(
            np.mean([row["controls_only_r_squared"] for row in held_out_rows])
        ),
        "mean_full_topic_equation_r_squared": float(
            np.mean(
                [row["full_topic_equation_r_squared"] for row in held_out_rows]
            )
        ),
        "mean_topic_equation_incremental_r_squared": float(
            np.mean(
                [
                    row["topic_equation_incremental_r_squared"]
                    for row in held_out_rows
                ]
            )
        ),
        "held_out_models": held_out_rows,
        "topic_coefficients": topic_rows,
    }


def run_interpretable_wording_regression(
    dataset: FrameworkDataset,
    *,
    components: int = 10,
    permutations: int = 999,
    bootstrap_samples: int = 500,
    random_state: int = 42,
) -> dict[str, object]:
    """Run cross-fitted NMF-similarity regression and lexical attribution."""

    documents = load_answer_only_documents()
    residual_matrices = similarity_matrices(dataset, residual=True)
    question_similarity = cosine_matrix(dataset.question_embeddings)
    mask, cutoff = cross_topic_mask(
        dataset,
        question_similarity_quantile=0.25,
        require_different_source=True,
    )
    sources = np.asarray(dataset.sources, dtype=object)
    same_source = sources[:, None] == sources[None, :]

    predictors: dict[str, np.ndarray] = {}
    fold_bases: dict[str, dict[str, object]] = {}
    held_out_rows = []
    term_fold_counts: Counter[str] = Counter()
    term_rank_scores: Counter[str] = Counter()
    for model_index, held_out_model in enumerate(MODELS):
        predictor, fold_basis = _fit_fold_wording_predictor(
            documents,
            held_out_model=held_out_model,
            question_count=len(dataset.question_ids),
            components=components,
            random_state=random_state + model_index,
        )
        predictors[held_out_model] = predictor
        fold_bases[held_out_model] = fold_basis
        fold_terms: set[str] = set()
        for terms in fold_basis["component_top_terms"]:
            for rank, term in enumerate(terms):
                fold_terms.add(str(term))
                term_rank_scores[str(term)] += len(terms) - rank
        term_fold_counts.update(fold_terms)

        test = mrqap_permutation_test(
            residual_matrices[held_out_model],
            predictor,
            question_similarity,
            same_source,
            mask,
            permutations=permutations,
            random_state=random_state + 1000 * model_index,
            alternative="two-sided",
        )
        confidence_intervals = mrqap_node_bootstrap_ci(
            residual_matrices[held_out_model],
            predictor,
            question_similarity,
            same_source,
            mask,
            samples=bootstrap_samples,
            random_state=random_state + 10_000 + model_index,
        )
        held_out_rows.append(
            _map_effect(
                held_out_model,
                test,
                confidence_intervals,
                spearman_rdm(predictor, question_similarity, mask),
                fold_basis,
            )
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

    component_sensitivity = []
    for component_count in (6, 8, 10, 12, 14):
        effects = []
        for model_index, held_out_model in enumerate(MODELS):
            if component_count == components:
                predictor = predictors[held_out_model]
            else:
                predictor, _ = _fit_fold_wording_predictor(
                    documents,
                    held_out_model=held_out_model,
                    question_count=len(dataset.question_ids),
                    components=component_count,
                    random_state=(
                        random_state + component_count * 100 + model_index
                    ),
                )
            effects.append(
                fit_mrqap_effect(
                    residual_matrices[held_out_model],
                    predictor,
                    question_similarity,
                    same_source,
                    mask,
                )
            )
        component_sensitivity.append(
            {
                "components": component_count,
                "mean_standardized_wording_beta": float(
                    np.mean(
                        [effect["standardized_consensus_beta"] for effect in effects]
                    )
                ),
                "minimum_standardized_wording_beta": float(
                    np.min(
                        [effect["standardized_consensus_beta"] for effect in effects]
                    )
                ),
                "mean_incremental_r_squared": float(
                    np.mean([effect["incremental_r_squared"] for effect in effects])
                ),
                "minimum_incremental_r_squared": float(
                    np.min([effect["incremental_r_squared"] for effect in effects])
                ),
            }
        )

    leave_one_source_out = []
    for omitted_source in sorted(set(dataset.sources)):
        keep = sources != omitted_source
        source_mask = mask & np.outer(keep, keep)
        if np.sum(np.triu(source_mask, k=1)) < 100:
            continue
        effects = [
            fit_mrqap_effect(
                residual_matrices[held_out_model],
                predictors[held_out_model],
                question_similarity,
                same_source,
                source_mask,
            )
            for held_out_model in MODELS
        ]
        leave_one_source_out.append(
            {
                "omitted_source": omitted_source,
                "remaining_pair_count": int(np.sum(np.triu(source_mask, k=1))),
                "mean_standardized_wording_beta": float(
                    np.mean(
                        [effect["standardized_consensus_beta"] for effect in effects]
                    )
                ),
                "mean_incremental_r_squared": float(
                    np.mean([effect["incremental_r_squared"] for effect in effects])
                ),
            }
        )

    recurring_terms = sorted(
        term_fold_counts,
        key=lambda term: (
            -term_fold_counts[term],
            -term_rank_scores[term],
            term,
        ),
    )[:24]
    descriptive_attribution = _descriptive_topic_attribution(
        dataset,
        documents,
        residual_matrices,
        question_similarity,
        mask,
        components=components,
        random_state=random_state,
    )

    mean_controls = float(
        np.mean([row["controls_only_r_squared"] for row in held_out_rows])
    )
    mean_full = float(
        np.mean([row["wording_full_r_squared"] for row in held_out_rows])
    )
    mean_increment = float(
        np.mean(
            [row["wording_incremental_r_squared"] for row in held_out_rows]
        )
    )
    return {
        "method": (
            "leave-one-model-out answer-only NMF wording-similarity "
            "nuisance-residual QAP regression"
        ),
        "equation": (
            "z(Y^m_ij) = beta_0 + beta_1 z(Q_ij) + beta_2 "
            "z([z(Q_ij)]^2) + beta_W z(W^-m_ij) + error_ij"
        ),
        "primary_definition": {
            "components": components,
            "pair_count": int(np.sum(np.triu(mask, k=1))),
            "different_exact_domain": True,
            "different_source": True,
            "question_similarity_quantile": 0.25,
            "question_similarity_cutoff": cutoff,
            "controls": ["question cosine", "question cosine squared"],
            "held_out_profile_rule": (
                "For held-out model m, fit TF-IDF/NMF only on the other five "
                "models and average their topic mixtures by question."
            ),
            "question_token_removal": True,
            "permutation_note": (
                "Complete-network nuisance residuals are node-permuted before "
                "refitting the strict masked coefficient; two-sided p-values "
                "are exploratory and Holm-adjusted across held-out models."
            ),
        },
        "held_out_models": held_out_rows,
        "mean_standardized_wording_beta": float(
            np.mean([row["standardized_wording_beta"] for row in held_out_rows])
        ),
        "minimum_standardized_wording_beta": float(
            np.min([row["standardized_wording_beta"] for row in held_out_rows])
        ),
        "mean_controls_only_r_squared": mean_controls,
        "mean_wording_full_r_squared": mean_full,
        "mean_wording_incremental_r_squared": mean_increment,
        "holm_significant_models": int(
            np.sum([float(row["p_value_holm"]) <= 0.05 for row in held_out_rows])
        ),
        "component_count_sensitivity": component_sensitivity,
        "leave_one_source_out": leave_one_source_out,
        "recurring_fold_basis_terms": [
            {
                "term": term,
                "folds": int(term_fold_counts[term]),
                "rank_score": int(term_rank_scores[term]),
            }
            for term in recurring_terms
        ],
        "descriptive_topic_attribution": descriptive_attribution,
        "interpretation_warning": (
            "The held-out model's response text is excluded from its primary "
            "NMF wording predictor, making the aggregate effect cross-model. "
            "The scenarios, response corpus, and embedding encoder remain "
            "shared, so this is fixed-panel interpretation rather than causal "
            "or unseen-scenario validation. The named topic coefficients use "
            "one all-response NMF basis in a separate, more flexible descriptive "
            "model. Its R-squared does not decompose the primary cross-fitted "
            "gain, and its topic coefficients receive no inferential p-values."
        ),
    }
