"""Strict loader for the balanced integrity repeated-measures panel.

Model names, question IDs, and helper labels are discovered from the database.
The current experiment is nevertheless validated as a complete 6 x 10 x 7
panel: six models, ten integrity questions, and the initial response plus six
helper conditions. Every response is exactly orthogonalized against its paired
question before it is exposed to analysis methods.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import duckdb
import numpy as np

from .geometry import (
    normalize_vectors,
    paired_cosines,
    remove_question_projection,
)


EXPECTED_MODEL_COUNT = 6
EXPECTED_QUESTION_COUNT = 10
EXPECTED_CONDITION_COUNT = 7
INITIAL_CONDITION = "initial"
PREFERRED_HELPER_ORDER: tuple[str, ...] = (
    "agreement",
    "disagreement",
    "strong_disagreement",
    "majority_opinion",
    "lived_experience",
    "expert_consensus",
)

DB_PATH = Path(__file__).resolve().parents[3] / "data" / "llm_ethics_data.duckdb"


@dataclass(frozen=True)
class IntegrityQuestion:
    """Human-readable metadata for one integrity scenario."""

    question_id: int
    domain: str
    conflict: str
    source: str
    text: str


@dataclass(frozen=True)
class IntegrityDataset:
    """Aligned text and embedding tensors for the complete integrity panel.

    Tensor axes are always ``[model, question, condition, embedding]``. The
    response-ID, response-text, and cosine tensors omit the final embedding
    axis. ``raw_responses`` and ``residuals`` are both L2-normalized; the
    former retain paired-question signal and the latter remove it exactly.
    """

    models: tuple[str, ...]
    question_ids: np.ndarray
    conditions: tuple[str, ...]
    questions: tuple[IntegrityQuestion, ...]
    helper_prompts: dict[str, str]
    question_embeddings: np.ndarray
    response_ids: np.ndarray
    response_texts: np.ndarray
    raw_responses: np.ndarray
    residuals: np.ndarray
    raw_question_cosines: np.ndarray
    residual_question_cosines: np.ndarray

    @property
    def domains(self) -> tuple[str, ...]:
        return tuple(question.domain for question in self.questions)

    @property
    def conflicts(self) -> tuple[str, ...]:
        return tuple(question.conflict for question in self.questions)

    @property
    def sources(self) -> tuple[str, ...]:
        return tuple(question.source for question in self.questions)

    @property
    def question_texts(self) -> tuple[str, ...]:
        return tuple(question.text for question in self.questions)

    @property
    def helper_conditions(self) -> tuple[str, ...]:
        return tuple(
            condition
            for condition in self.conditions
            if condition != INITIAL_CONDITION
        )

    @property
    def initial_condition_index(self) -> int:
        return self.condition_index(INITIAL_CONDITION)

    @property
    def embedding_dimension(self) -> int:
        return int(self.raw_responses.shape[-1])

    @property
    def response_character_counts(self) -> np.ndarray:
        """Character counts aligned to ``[model, question, condition]``."""

        lengths = np.fromiter(
            (len(text) for text in self.response_texts.flat),
            dtype=np.int64,
            count=self.response_texts.size,
        )
        return lengths.reshape(self.response_texts.shape)

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.response_ids.shape

    def model_index(self, model: str) -> int:
        try:
            return self.models.index(model)
        except ValueError as error:
            raise KeyError(f"Unknown integrity model: {model}") from error

    def condition_index(self, condition: str) -> int:
        try:
            return self.conditions.index(condition)
        except ValueError as error:
            raise KeyError(f"Unknown integrity condition: {condition}") from error

    def question_index(self, question_id: int) -> int:
        matches = np.flatnonzero(self.question_ids == question_id)
        if len(matches) != 1:
            raise KeyError(f"Unknown integrity question ID: {question_id}")
        return int(matches[0])

    def view(
        self,
        *,
        model: str | None = None,
        condition: str | None = None,
        residual: bool = True,
    ) -> np.ndarray:
        """Select an embedding view without relying on numeric database IDs."""

        values = self.residuals if residual else self.raw_responses
        if model is not None:
            values = values[self.model_index(model)]
        if condition is not None:
            condition_axis = 1 if model is not None else 2
            values = np.take(
                values,
                self.condition_index(condition),
                axis=condition_axis,
            )
        return values


def _require_nonempty_text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _stack_embeddings(
    embeddings: Sequence[object],
    *,
    label: str,
    expected_dimension: int | None = None,
) -> np.ndarray:
    """Validate and stack database vectors without accepting ragged arrays."""

    if not embeddings:
        raise ValueError(f"No {label} embeddings were found")
    vectors: list[np.ndarray] = []
    dimensions: set[int] = set()
    for index, embedding in enumerate(embeddings):
        try:
            vector = np.asarray(embedding, dtype=np.float64)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{label} embedding {index} is not numeric") from error
        if vector.ndim != 1 or vector.size == 0:
            raise ValueError(f"{label} embedding {index} is not a non-empty vector")
        if not np.isfinite(vector).all():
            raise ValueError(f"{label} embedding {index} contains non-finite values")
        vectors.append(vector)
        dimensions.add(int(vector.size))
    if len(dimensions) != 1:
        raise ValueError(
            f"{label} embeddings have inconsistent dimensions: {sorted(dimensions)}"
        )
    dimension = next(iter(dimensions))
    if expected_dimension is not None and dimension != expected_dimension:
        raise ValueError(
            f"{label} embedding dimension {dimension} does not match "
            f"question dimension {expected_dimension}"
        )
    return np.stack(vectors)


def _validate_unique(values: Sequence[object], *, label: str) -> None:
    counts = Counter(values)
    duplicates = [value for value, count in counts.items() if count != 1]
    if duplicates:
        preview = ", ".join(repr(value) for value in duplicates[:5])
        raise ValueError(f"Duplicate {label}: {preview}")


def load_integrity_dataset(
    database_path: str | Path = DB_PATH,
) -> IntegrityDataset:
    """Load and strictly audit the current complete integrity experiment."""

    path = Path(database_path)
    if not path.is_file():
        raise FileNotFoundError(f"Integrity database not found: {path}")

    connection = duckdb.connect(str(path), read_only=True)
    try:
        question_rows = connection.execute(
            """
            SELECT question_id, domain, hidden_conflict, source, question_text
            FROM integrity_questions
            ORDER BY question_id
            """
        ).fetchall()
        if len(question_rows) != EXPECTED_QUESTION_COUNT:
            raise ValueError(
                f"Expected {EXPECTED_QUESTION_COUNT} integrity questions, "
                f"found {len(question_rows)}"
            )
        _validate_unique(
            [row[0] for row in question_rows],
            label="integrity question IDs",
        )
        questions = tuple(
            IntegrityQuestion(
                question_id=int(row[0]),
                domain=_require_nonempty_text(row[1], "question domain"),
                conflict=_require_nonempty_text(row[2], "question conflict"),
                source=_require_nonempty_text(row[3], "question source"),
                text=_require_nonempty_text(row[4], "question text"),
            )
            for row in question_rows
        )
        question_ids = np.asarray(
            [question.question_id for question in questions],
            dtype=np.int64,
        )

        helper_rows = connection.execute(
            """
            SELECT helper_type, helper_text
            FROM helpers
            ORDER BY helper_type
            """
        ).fetchall()
        if len(helper_rows) != EXPECTED_CONDITION_COUNT - 1:
            raise ValueError(
                f"Expected {EXPECTED_CONDITION_COUNT - 1} helper prompts, "
                f"found {len(helper_rows)}"
            )
        discovered_helper_types = tuple(
            _require_nonempty_text(row[0], "helper type") for row in helper_rows
        )
        _validate_unique(discovered_helper_types, label="helper types")
        if INITIAL_CONDITION in discovered_helper_types:
            raise ValueError(
                f"Helper type {INITIAL_CONDITION!r} is reserved for baseline responses"
            )
        helper_prompts = {
            helper_type: _require_nonempty_text(row[1], "helper prompt")
            for helper_type, row in zip(discovered_helper_types, helper_rows)
        }
        preferred_positions = {
            condition: index
            for index, condition in enumerate(PREFERRED_HELPER_ORDER)
        }
        helper_types = tuple(
            sorted(
                discovered_helper_types,
                key=lambda condition: (
                    preferred_positions.get(condition, len(preferred_positions)),
                    condition,
                ),
            )
        )
        conditions = (INITIAL_CONDITION, *helper_types)

        response_rows = connection.execute(
            """
            SELECT response_id, model, question_id, helper_type, response_text
            FROM integrity_responses
            ORDER BY model, question_id, helper_type, response_id
            """
        ).fetchall()
        expected_responses = (
            EXPECTED_MODEL_COUNT
            * EXPECTED_QUESTION_COUNT
            * EXPECTED_CONDITION_COUNT
        )
        if len(response_rows) != expected_responses:
            raise ValueError(
                f"Expected {expected_responses} integrity responses, "
                f"found {len(response_rows)}"
            )
        _validate_unique(
            [row[0] for row in response_rows],
            label="integrity response IDs",
        )

        models = tuple(
            sorted(
                {
                    _require_nonempty_text(row[1], "response model")
                    for row in response_rows
                }
            )
        )
        if len(models) != EXPECTED_MODEL_COUNT:
            raise ValueError(
                f"Expected {EXPECTED_MODEL_COUNT} integrity models, "
                f"found {len(models)}: {models}"
            )

        expected_cells = {
            (model, int(question_id), condition)
            for model in models
            for question_id in question_ids
            for condition in conditions
        }
        observed_cells: list[tuple[str, int, str]] = []
        for response_id, model, question_id, condition, response_text in response_rows:
            model = _require_nonempty_text(model, "response model")
            condition = _require_nonempty_text(condition, "response condition")
            _require_nonempty_text(response_text, f"response {response_id} text")
            observed_cells.append((model, int(question_id), condition))
        cell_counts = Counter(observed_cells)
        duplicate_cells = [cell for cell, count in cell_counts.items() if count > 1]
        missing_cells = expected_cells - set(cell_counts)
        unexpected_cells = set(cell_counts) - expected_cells
        if duplicate_cells or missing_cells or unexpected_cells:
            details: list[str] = []
            if duplicate_cells:
                details.append(f"duplicate={duplicate_cells[:3]}")
            if missing_cells:
                details.append(f"missing={sorted(missing_cells)[:3]}")
            if unexpected_cells:
                details.append(f"unexpected={sorted(unexpected_cells)[:3]}")
            raise ValueError("Integrity panel is not complete: " + "; ".join(details))

        # Enforce an exact one-to-one embedding join before constructing tensors.
        embedding_rows = connection.execute(
            """
            SELECT response_id, embedding
            FROM integrity_embeddings
            ORDER BY response_id
            """
        ).fetchall()
        _validate_unique(
            [row[0] for row in embedding_rows],
            label="integrity embedding response IDs",
        )
        response_id_set = {int(row[0]) for row in response_rows}
        embedding_id_set = {int(row[0]) for row in embedding_rows}
        missing_embedding_ids = response_id_set - embedding_id_set
        orphan_embedding_ids = embedding_id_set - response_id_set
        if missing_embedding_ids or orphan_embedding_ids:
            raise ValueError(
                "Integrity response/embedding alignment failed: "
                f"missing={sorted(missing_embedding_ids)[:5]}, "
                f"orphan={sorted(orphan_embedding_ids)[:5]}"
            )
        embedding_by_response_id = {
            int(response_id): embedding for response_id, embedding in embedding_rows
        }

        question_embedding_rows = connection.execute(
            """
            SELECT q.question_id, e.embedding
            FROM integrity_questions AS q
            LEFT JOIN question_embeddings AS e
                ON q.question_id = e.question_id
            ORDER BY q.question_id
            """
        ).fetchall()
        if len(question_embedding_rows) != EXPECTED_QUESTION_COUNT:
            raise ValueError("Question-embedding join changed integrity question count")
        if [int(row[0]) for row in question_embedding_rows] != question_ids.tolist():
            raise ValueError("Question embeddings are not aligned to integrity questions")
        if any(row[1] is None for row in question_embedding_rows):
            missing = [int(row[0]) for row in question_embedding_rows if row[1] is None]
            raise ValueError(f"Missing integrity question embeddings: {missing}")
        raw_question_embeddings = _stack_embeddings(
            [row[1] for row in question_embedding_rows],
            label="question",
        )
        dimension = int(raw_question_embeddings.shape[1])

        model_indices = {model: index for index, model in enumerate(models)}
        question_indices = {
            int(question_id): index
            for index, question_id in enumerate(question_ids)
        }
        condition_indices = {
            condition: index for index, condition in enumerate(conditions)
        }
        tensor_shape = (
            len(models),
            len(question_ids),
            len(conditions),
        )
        response_ids = np.empty(tensor_shape, dtype=np.int64)
        response_texts = np.empty(tensor_shape, dtype=object)
        ordered_embedding_objects: list[object | None] = [None] * expected_responses

        for response_id, model, question_id, condition, response_text in response_rows:
            position = (
                model_indices[str(model)],
                question_indices[int(question_id)],
                condition_indices[str(condition)],
            )
            response_ids[position] = int(response_id)
            response_texts[position] = str(response_text)
            flat_position = np.ravel_multi_index(position, tensor_shape)
            ordered_embedding_objects[flat_position] = embedding_by_response_id[
                int(response_id)
            ]
        if any(value is None for value in ordered_embedding_objects):
            raise ValueError("Internal error while aligning integrity embeddings")
        raw_response_embeddings = _stack_embeddings(
            ordered_embedding_objects,
            label="response",
            expected_dimension=dimension,
        ).reshape((*tensor_shape, dimension))

        paired_questions = np.broadcast_to(
            raw_question_embeddings[None, :, None, :],
            raw_response_embeddings.shape,
        )
        raw_question_cosines = paired_cosines(
            raw_response_embeddings,
            paired_questions,
        )
        orthogonal = remove_question_projection(
            raw_response_embeddings,
            paired_questions,
        )
        residual_question_cosines = paired_cosines(orthogonal, paired_questions)
        max_post_cosine = float(np.max(np.abs(residual_question_cosines)))
        if max_post_cosine >= 1e-10:
            raise ValueError(
                "Orthogonalization audit failed: "
                f"max paired |cosine|={max_post_cosine:.3e}"
            )

        return IntegrityDataset(
            models=models,
            question_ids=question_ids,
            conditions=conditions,
            questions=questions,
            helper_prompts=helper_prompts,
            question_embeddings=normalize_vectors(raw_question_embeddings),
            response_ids=response_ids,
            response_texts=response_texts,
            raw_responses=normalize_vectors(raw_response_embeddings),
            residuals=normalize_vectors(orthogonal),
            raw_question_cosines=raw_question_cosines,
            residual_question_cosines=residual_question_cosines,
        )
    finally:
        connection.close()
