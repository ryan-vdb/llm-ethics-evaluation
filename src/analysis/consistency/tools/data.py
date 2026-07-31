"""Canonical loader for the six aligned consistency-model views.

The loader owns all database access for this analysis. Every answer is exactly
orthogonalized against its paired question and L2-normalized before a method
can access it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import duckdb
import numpy as np


MODELS: tuple[str, ...] = (
    "claude_sonnet",
    "claude_opus",
    "gemini_flash",
    "gpt_55",
    "grok",
    "deepseek",
)

DB_PATH = (
    Path(__file__).resolve().parents[3]
    / "data"
    / "llm_ethics_data.duckdb"
)


@dataclass(frozen=True)
class FrameworkDataset:
    """Aligned metadata, raw answers, and exact residual views."""

    question_ids: np.ndarray
    domains: tuple[str, ...]
    conflicts: tuple[str, ...]
    sources: tuple[str, ...]
    question_texts: tuple[str, ...]
    question_embeddings: np.ndarray
    raw_responses: dict[str, np.ndarray]
    residuals: dict[str, np.ndarray]
    raw_question_cosines: dict[str, np.ndarray]
    residual_question_cosines: dict[str, np.ndarray]


def _load_model_embeddings(
    connection: duckdb.DuckDBPyConnection,
    model: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load one model's question-aligned embedding matrices."""

    rows = connection.execute(
        """
        SELECT
            r.question_id,
            q.embedding AS question_embedding,
            e.embedding AS response_embedding
        FROM consistency_responses AS r
        JOIN question_embeddings AS q
            ON r.question_id = q.question_id
        JOIN consistency_embeddings AS e
            ON r.response_id = e.response_id
        WHERE r.model = ?
        ORDER BY r.question_id
        """,
        [model],
    ).fetchall()
    if len(rows) != 93:
        raise ValueError(
            f"Expected 93 aligned embeddings for {model}, found {len(rows)}"
        )

    question_ids = np.asarray([row[0] for row in rows], dtype=int)
    question_embeddings = np.asarray([row[1] for row in rows], dtype=np.float64)
    response_embeddings = np.asarray([row[2] for row in rows], dtype=np.float64)
    if question_embeddings.shape != response_embeddings.shape:
        raise ValueError(
            f"Embedding-shape mismatch for {model}: "
            f"{question_embeddings.shape} vs {response_embeddings.shape}"
        )
    return question_ids, question_embeddings, response_embeddings


def load_framework_dataset() -> FrameworkDataset:
    """Load, exactly orthogonalize, normalize, align, and audit all model views."""

    from .geometry import (
        normalize_rows,
        paired_cosines,
        remove_question_projection,
    )

    connection = duckdb.connect(str(DB_PATH), read_only=True)
    try:
        metadata_rows = connection.execute(
            """
            SELECT
                question_id,
                domain,
                hidden_conflict,
                source,
                question_text
            FROM consistency_questions
            ORDER BY question_id
            """
        ).fetchall()
        if len(metadata_rows) != 93:
            raise ValueError(
                f"Expected 93 consistency questions, found {len(metadata_rows)}"
            )

        metadata_ids = np.asarray([row[0] for row in metadata_rows], dtype=int)
        if not np.array_equal(metadata_ids, np.arange(93)):
            raise ValueError(
                "Consistency question IDs must be contiguous from 0 to 92"
            )

        reference_questions: np.ndarray | None = None
        raw_responses: dict[str, np.ndarray] = {}
        residuals: dict[str, np.ndarray] = {}
        raw_question_cosines: dict[str, np.ndarray] = {}
        residual_question_cosines: dict[str, np.ndarray] = {}

        for model in MODELS:
            question_ids, question_embeddings, response_embeddings = (
                _load_model_embeddings(connection, model)
            )
            if not np.array_equal(question_ids, metadata_ids):
                raise ValueError(f"Question alignment failed for model {model}")
            if reference_questions is None:
                reference_questions = question_embeddings
            elif not np.allclose(reference_questions, question_embeddings):
                raise ValueError(f"Question embeddings differ for model {model}")

            orthogonal = remove_question_projection(
                response_embeddings,
                question_embeddings,
                strength=1.0,
            )
            normalized_raw = normalize_rows(response_embeddings)
            normalized_residual = normalize_rows(orthogonal)
            raw_cosines = paired_cosines(
                response_embeddings,
                question_embeddings,
            )
            post_cosines = paired_cosines(
                orthogonal,
                question_embeddings,
            )
            max_post_cosine = float(np.max(np.abs(post_cosines)))
            if max_post_cosine >= 1e-10:
                raise ValueError(
                    f"Orthogonalization audit failed for {model}: "
                    f"max |cos|={max_post_cosine:.3e}"
                )

            raw_responses[model] = normalized_raw
            residuals[model] = normalized_residual
            raw_question_cosines[model] = raw_cosines
            residual_question_cosines[model] = post_cosines
    finally:
        connection.close()

    assert reference_questions is not None
    return FrameworkDataset(
        question_ids=metadata_ids,
        domains=tuple(row[1] for row in metadata_rows),
        conflicts=tuple(row[2] for row in metadata_rows),
        sources=tuple(row[3] for row in metadata_rows),
        question_texts=tuple(row[4] for row in metadata_rows),
        question_embeddings=normalize_rows(reference_questions),
        raw_responses=raw_responses,
        residuals=residuals,
        raw_question_cosines=raw_question_cosines,
        residual_question_cosines=residual_question_cosines,
    )
