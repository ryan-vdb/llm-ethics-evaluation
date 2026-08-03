"""Interpretable NMF topics in answer-only reasoning language.

The geometric tests establish reproducibility but do not name the dimensions.
This secondary analysis offers a sparse textual surrogate:

1. remove English stopwords and every token appearing in the paired question;
2. TF–IDF weight the remaining answer language across all 558 responses;
3. fit non-negative matrix factorization (NMF); and
4. relate question-level topic profiles to the exact-orthogonal embedding
   geometry on the same strict cross-topic pairs.

NMF topic labels are post-hoc interpretation aids. Because the same responses
feed both the embeddings and TF–IDF, their association is not independent
validation of the geometric result.
"""

from __future__ import annotations

import re
from pathlib import Path

import duckdb
import numpy as np
from scipy.stats import rankdata
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer

from ..tools.data import DB_PATH, MODELS, FrameworkDataset
from ..tools.geometry import (
    cosine_matrix,
    cross_topic_mask,
    normalize_rows,
)
from ..tools.statistics import partial_spearman_rdm


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z][a-z'-]+", text.lower())


def load_answer_only_documents(
    database: Path = DB_PATH,
) -> list[dict[str, object]]:
    """Load answers and remove tokens copied directly from each question."""

    connection = duckdb.connect(str(database), read_only=True)
    try:
        rows = connection.execute(
            """
            SELECT
                r.question_id,
                r.model,
                r.response_text,
                q.question_text
            FROM consistency_responses AS r
            JOIN consistency_questions AS q
                ON r.question_id = q.question_id
            ORDER BY r.question_id, r.model
            """
        ).fetchall()
    finally:
        connection.close()

    if len(rows) != 93 * len(MODELS):
        raise ValueError(f"Expected 558 responses, found {len(rows)}")

    documents = []
    for question_id, model, answer, question in rows:
        question_tokens = set(_tokens(question))
        remaining = [
            token
            for token in _tokens(answer)
            if token not in question_tokens
            and token not in ENGLISH_STOP_WORDS
            and len(token) > 2
        ]
        documents.append(
            {
                "question_id": int(question_id),
                "model": model,
                "text": " ".join(remaining),
            }
        )
    return documents


def _spearman(first: np.ndarray, second: np.ndarray) -> float:
    first_rank = rankdata(first, method="average")
    second_rank = rankdata(second, method="average")
    first_rank -= np.mean(first_rank)
    second_rank -= np.mean(second_rank)
    denominator = np.linalg.norm(first_rank) * np.linalg.norm(second_rank)
    if denominator == 0:
        return float("nan")
    return float(np.dot(first_rank, second_rank) / denominator)


def fit_answer_only_nmf(
    documents: list[dict[str, object]],
    *,
    components: int,
    random_state: int,
) -> tuple[TfidfVectorizer, NMF, np.ndarray, object]:
    """Fit the canonical answer-only TF-IDF/NMF representation.

    The helper is shared with the wording-regression method so its descriptive
    basis uses exactly the same token filtering and factorization settings.
    Returned document-topic rows are L1-normalized topic mixtures.
    """

    vectorizer = TfidfVectorizer(
        min_df=4,
        max_df=0.90,
        max_features=5000,
        ngram_range=(1, 2),
        sublinear_tf=True,
    )
    tfidf = vectorizer.fit_transform(
        [str(document["text"]) for document in documents]
    )
    model = NMF(
        n_components=components,
        init="nndsvda",
        max_iter=2000,
        l1_ratio=0.10,
        random_state=random_state,
    )
    document_topics = model.fit_transform(tfidf)
    topic_sums = np.sum(document_topics, axis=1, keepdims=True)
    document_topics = np.divide(
        document_topics,
        topic_sums,
        out=np.zeros_like(document_topics),
        where=topic_sums > 0,
    )
    return vectorizer, model, document_topics, tfidf


def run_reasoning_topics(
    dataset: FrameworkDataset,
    *,
    components: int = 10,
    permutations: int = 999,
    random_state: int = 42,
) -> dict[str, object]:
    """Fit answer-only NMF topics and compare them with residual geometry."""

    documents = load_answer_only_documents()
    vectorizer, model, document_topics, tfidf = fit_answer_only_nmf(
        documents,
        components=components,
        random_state=random_state,
    )

    by_model = {
        model_name: np.zeros((93, components), dtype=np.float64)
        for model_name in MODELS
    }
    for row_index, document in enumerate(documents):
        by_model[str(document["model"])][int(document["question_id"])] = (
            document_topics[row_index]
        )

    question_topics = np.mean(
        np.stack([by_model[model_name] for model_name in MODELS]),
        axis=0,
    )
    question_topics = normalize_rows(question_topics + 1e-12)
    topic_similarity = cosine_matrix(question_topics)
    residual_similarity = np.mean(
        np.stack(
            [
                cosine_matrix(dataset.residuals[model_name])
                for model_name in MODELS
            ]
        ),
        axis=0,
    )
    question_similarity = cosine_matrix(dataset.question_embeddings)
    mask, cutoff = cross_topic_mask(
        dataset,
        question_similarity_quantile=0.25,
        require_different_source=True,
    )
    observed = partial_spearman_rdm(
        topic_similarity,
        residual_similarity,
        question_similarity,
        mask,
    )

    rng = np.random.default_rng(random_state)
    null = np.empty(permutations, dtype=np.float64)
    for permutation_index in range(permutations):
        permutation = rng.permutation(len(topic_similarity))
        permuted = topic_similarity[np.ix_(permutation, permutation)]
        null[permutation_index] = partial_spearman_rdm(
            permuted,
            residual_similarity,
            question_similarity,
            mask,
        )
    p_value = float((1 + np.sum(null >= observed)) / (permutations + 1))

    vocabulary = np.asarray(vectorizer.get_feature_names_out())
    topic_profiles = []
    for component_index in range(components):
        top_term_indices = np.argsort(
            -model.components_[component_index],
            kind="stable",
        )[:12]
        top_question_indices = np.argsort(
            -question_topics[:, component_index],
            kind="stable",
        )[:6]
        held_out_correlations = []
        for held_out_model in MODELS:
            other_mean = np.mean(
                np.stack(
                    [
                        by_model[model_name][:, component_index]
                        for model_name in MODELS
                        if model_name != held_out_model
                    ]
                ),
                axis=0,
            )
            held_out_correlations.append(
                _spearman(
                    by_model[held_out_model][:, component_index],
                    other_mean,
                )
            )
        topic_profiles.append(
            {
                "topic": component_index + 1,
                "descriptive_label": " · ".join(
                    vocabulary[top_term_indices[:3]].tolist()
                ),
                "top_terms": vocabulary[top_term_indices].tolist(),
                "mean_cross_model_profile_rho": float(
                    np.nanmean(held_out_correlations)
                ),
                "minimum_cross_model_profile_rho": float(
                    np.nanmin(held_out_correlations)
                ),
                "top_questions": [
                    {
                        "question_id": int(
                            dataset.question_ids[question_index]
                        ),
                        "weight": float(
                            question_topics[question_index, component_index]
                        ),
                        "domain": dataset.domains[question_index],
                        "conflict": dataset.conflicts[question_index],
                        "question": dataset.question_texts[question_index],
                    }
                    for question_index in top_question_indices
                ],
            }
        )

    return {
        "method": "answer-only TF-IDF plus non-negative matrix factorization",
        "documents": len(documents),
        "vocabulary_size": int(tfidf.shape[1]),
        "components": components,
        "reconstruction_error": float(model.reconstruction_err_),
        "iterations": int(model.n_iter_),
        "question_similarity_cutoff": cutoff,
        "cross_topic_pair_count": int(np.sum(np.triu(mask, k=1))),
        "topic_vs_residual_partial_rho": observed,
        "permutation_p_value": p_value,
        "permutation_null_mean": float(np.mean(null)),
        "topic_profiles": topic_profiles,
        "interpretation_warning": (
            "The TF-IDF vocabulary and NMF basis are jointly fit to all response "
            "texts. Topics are sparse post-hoc descriptions of the same texts "
            "used for embedding geometry, so their cross-model concordance aids "
            "interpretation but is not held-out or independent validation."
        ),
    }
