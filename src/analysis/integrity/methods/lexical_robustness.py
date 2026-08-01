"""Question-token-removed lexical check of semantic revision patterns."""

from __future__ import annotations

import re

import numpy as np
from scipy.stats import rankdata, spearmanr
from sklearn.feature_extraction.text import TfidfVectorizer

from ..tools.data import IntegrityDataset


TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z'-]+")


def _exact_sign_flip(values: np.ndarray) -> float:
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
    for rank, position in enumerate(order):
        running = max(
            running,
            (len(p_values) - rank) * p_values[int(position)],
        )
        adjusted[int(position)] = min(1.0, running)
    return adjusted.tolist()


def _remove_question_tokens(response: str, question: str) -> str:
    question_tokens = {
        token.lower() for token in TOKEN_RE.findall(question)
        if len(token) >= 3
    }
    retained = [
        token.lower() for token in TOKEN_RE.findall(response)
        if token.lower() not in question_tokens
    ]
    return " ".join(retained) or "emptydocument"


def _partial_spearman(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> float:
    ranked_x = rankdata(x)
    ranked_y = rankdata(y)
    ranked_z = rankdata(z)
    design = np.column_stack([np.ones(len(z)), ranked_z])
    residual_x = ranked_x - design @ np.linalg.lstsq(
        design, ranked_x, rcond=None
    )[0]
    residual_y = ranked_y - design @ np.linalg.lstsq(
        design, ranked_y, rcond=None
    )[0]
    return float(np.corrcoef(residual_x, residual_y)[0, 1])


def run_lexical_robustness(dataset: IntegrityDataset) -> dict[str, object]:
    """Compare embedding revision with an answer-text-only TF-IDF view."""

    conditions = list(dataset.conditions)
    initial_index = conditions.index("initial")
    agreement_index = conditions.index("agreement")
    model_count = len(dataset.models)
    question_count = len(dataset.question_ids)
    condition_count = len(conditions)

    documents = []
    positions = []
    for model_index in range(model_count):
        for question_index in range(question_count):
            for condition_index in range(condition_count):
                documents.append(
                    _remove_question_tokens(
                        str(
                            dataset.response_texts[
                                model_index,
                                question_index,
                                condition_index,
                            ]
                        ),
                        dataset.question_texts[question_index],
                    )
                )
                positions.append(
                    (model_index, question_index, condition_index)
                )

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
        norm="l2",
    )
    tfidf = vectorizer.fit_transform(documents)
    row_for = {
        position: row_index for row_index, position in enumerate(positions)
    }
    lexical_retention = np.full(
        (model_count, question_count, condition_count),
        np.nan,
        dtype=np.float64,
    )
    word_change = np.full_like(lexical_retention, np.nan)
    cell_rows = []
    for model_index, model in enumerate(dataset.models):
        for question_index, question_id in enumerate(dataset.question_ids):
            initial_row = row_for[
                (model_index, question_index, initial_index)
            ]
            initial_words = len(
                str(
                    dataset.response_texts[
                        model_index, question_index, initial_index
                    ]
                ).split()
            )
            for condition_index, condition in enumerate(conditions):
                if condition == "initial":
                    continue
                followup_row = row_for[
                    (model_index, question_index, condition_index)
                ]
                similarity = float(
                    tfidf[initial_row].multiply(tfidf[followup_row]).sum()
                )
                followup_words = len(
                    str(
                        dataset.response_texts[
                            model_index,
                            question_index,
                            condition_index,
                        ]
                    ).split()
                )
                length_change = abs(
                    np.log((followup_words + 1) / (initial_words + 1))
                )
                lexical_retention[
                    model_index, question_index, condition_index
                ] = similarity
                word_change[
                    model_index, question_index, condition_index
                ] = length_change
                semantic_retention = float(
                    np.dot(
                        dataset.residuals[
                            model_index, question_index, initial_index
                        ],
                        dataset.residuals[
                            model_index, question_index, condition_index
                        ],
                    )
                )
                cell_rows.append(
                    {
                        "model": model,
                        "question_id": int(question_id),
                        "condition": condition,
                        "lexical_retention": similarity,
                        "lexical_revision": 1.0 - similarity,
                        "semantic_retention": semantic_retention,
                        "semantic_revision": 1.0 - semantic_retention,
                        "absolute_log_word_count_change": float(
                            length_change
                        ),
                    }
                )

    helper_indices = [
        index for index, condition in enumerate(conditions)
        if condition != "initial"
    ]
    condition_rows = []
    for condition_index in helper_indices:
        values = 1.0 - lexical_retention[:, :, condition_index]
        condition_rows.append(
            {
                "condition": conditions[condition_index],
                "mean_lexical_revision": float(np.mean(values)),
                "mean_lexical_retention": float(1.0 - np.mean(values)),
                "mean_absolute_log_word_count_change": float(
                    np.mean(word_change[:, :, condition_index])
                ),
            }
        )

    agreement_revision = 1.0 - lexical_retention[:, :, agreement_index]
    contrast_rows = []
    for condition_index in helper_indices:
        if condition_index == agreement_index:
            continue
        difference = (
            1.0
            - lexical_retention[:, :, condition_index]
            - agreement_revision
        )
        contrast_rows.append(
            {
                "condition": conditions[condition_index],
                "mean_lexical_revision_difference_vs_agreement": float(
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

    semantic_revision = np.asarray(
        [row["semantic_revision"] for row in cell_rows],
        dtype=np.float64,
    )
    lexical_revision_values = np.asarray(
        [row["lexical_revision"] for row in cell_rows],
        dtype=np.float64,
    )
    length_values = np.asarray(
        [row["absolute_log_word_count_change"] for row in cell_rows],
        dtype=np.float64,
    )
    return {
        "method": (
            "question-token-removed unigram/bigram TF-IDF paired-answer "
            "similarity"
        ),
        "vocabulary_size": int(len(vectorizer.vocabulary_)),
        "condition_summary": condition_rows,
        "condition_vs_agreement": contrast_rows,
        "cellwise_semantic_lexical_spearman": float(
            spearmanr(semantic_revision, lexical_revision_values).statistic
        ),
        "cellwise_semantic_length_spearman": float(
            spearmanr(semantic_revision, length_values).statistic
        ),
        "semantic_lexical_partial_spearman_controlling_length": (
            _partial_spearman(
                semantic_revision,
                lexical_revision_values,
                length_values,
            )
        ),
        "cell_metrics": cell_rows,
        "interpretation_boundary": (
            "TF-IDF is an encoder-independent but not data-independent lexical "
            "sensitivity check. Follow-ups may echo helper-prompt vocabulary; "
            "this is not a validated detector of retained or reversed conclusions."
        ),
    }
