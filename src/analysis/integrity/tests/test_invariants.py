"""Fast checks for panel alignment, orthogonalization, and inference helpers."""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from ..methods.revision_effects import run_revision_effects
from ..runner import (
    DEFAULT_OUTPUT,
    MANIFEST_SCHEMA_VERSION,
    PROJECT_ROOT,
    RESULTS_PRODUCER,
    validate_output_target,
)
from ..tools.data import (
    EXPECTED_CONDITION_COUNT,
    EXPECTED_MODEL_COUNT,
    EXPECTED_QUESTION_COUNT,
    INITIAL_CONDITION,
    PREFERRED_HELPER_ORDER,
    _stack_embeddings,
    load_integrity_dataset,
)
from ..tools.geometry import (
    initial_similarity,
    remove_question_projection,
)
from ..tools.output import sha256_file, write_rows
from ..tools.statistics import (
    aggregate_by_question,
    benjamini_hochberg,
    exact_paired_sign_flip,
    holm_bonferroni,
    two_way_bootstrap,
)


class GeometryTests(unittest.TestCase):
    def test_projection_is_exact_for_tensor(self) -> None:
        questions = np.asarray(
            [
                [[[1.0, 0.0], [1.0, 0.0]]],
                [[[1.0, 1.0], [1.0, 1.0]]],
            ]
        )
        responses = np.asarray(
            [
                [[[3.0, 4.0], [-2.0, 5.0]]],
                [[[2.0, -1.0], [4.0, 3.0]]],
            ]
        )
        residuals = remove_question_projection(responses, questions)
        paired_dot = np.sum(residuals * questions, axis=-1)
        np.testing.assert_allclose(paired_dot, 0.0, atol=1e-12)

    def test_database_panel_is_complete_and_does_not_assume_zero_ids(self) -> None:
        dataset = load_integrity_dataset()
        self.assertEqual(dataset.shape, (6, 10, 7))
        self.assertEqual(len(dataset.models), EXPECTED_MODEL_COUNT)
        self.assertEqual(len(dataset.question_ids), EXPECTED_QUESTION_COUNT)
        self.assertEqual(len(dataset.conditions), EXPECTED_CONDITION_COUNT)
        self.assertEqual(dataset.conditions[0], INITIAL_CONDITION)
        self.assertEqual(dataset.helper_conditions, PREFERRED_HELPER_ORDER)
        self.assertNotEqual(int(dataset.question_ids[0]), 0)
        self.assertEqual(len(np.unique(dataset.response_ids)), 420)
        self.assertEqual(dataset.raw_responses.shape, (6, 10, 7, 3072))
        self.assertEqual(dataset.response_texts.shape, (6, 10, 7))
        self.assertEqual(dataset.response_character_counts.shape, (6, 10, 7))
        self.assertTrue(np.all(dataset.response_character_counts > 0))
        self.assertLess(
            float(np.max(np.abs(dataset.residual_question_cosines))),
            1e-10,
        )
        similarities = initial_similarity(
            dataset.residuals,
            initial_index=dataset.initial_condition_index,
        )
        np.testing.assert_allclose(
            similarities[:, :, dataset.initial_condition_index],
            1.0,
            atol=1e-12,
        )

    def test_embedding_validator_rejects_bad_vectors(self) -> None:
        with self.assertRaisesRegex(ValueError, "inconsistent dimensions"):
            _stack_embeddings([[1.0, 2.0], [1.0]], label="test")
        with self.assertRaisesRegex(ValueError, "non-finite"):
            _stack_embeddings([[1.0, np.nan]], label="test")


class InferenceTests(unittest.TestCase):
    def test_exact_question_sign_flip(self) -> None:
        result = exact_paired_sign_flip([1.0] * 10, alternative="greater")
        self.assertEqual(result["n_assignments"], 1024)
        self.assertEqual(result["n_questions"], 10)
        self.assertAlmostEqual(float(result["p_value"]), 1.0 / 1024.0)

    def test_question_aggregation_and_two_way_bootstrap_are_deterministic(self) -> None:
        panel = np.arange(60, dtype=np.float64).reshape(6, 10)
        np.testing.assert_allclose(
            aggregate_by_question(panel, question_axis=1),
            np.mean(panel, axis=0),
        )
        first = two_way_bootstrap(panel, n_bootstrap=50, seed=7)
        second = two_way_bootstrap(panel, n_bootstrap=50, seed=7)
        self.assertEqual(first, second)
        self.assertAlmostEqual(float(first["estimate"]), float(np.mean(panel)))
        with self.assertRaisesRegex(ValueError, "question_axis"):
            aggregate_by_question(panel, question_axis=3)

    def test_adjustments_are_monotonic_for_sorted_p_values(self) -> None:
        p_values = [0.001, 0.01, 0.04]
        bh = benjamini_hochberg(p_values)
        holm = holm_bonferroni(p_values)
        self.assertTrue(all(a <= b for a, b in zip(bh, bh[1:])))
        self.assertTrue(all(a <= b for a, b in zip(holm, holm[1:])))

    def test_primary_result_uses_two_sided_question_units(self) -> None:
        result = run_revision_effects(
            load_integrity_dataset(),
            bootstrap_samples=100,
            helper_label_permutations=99,
            random_state=7,
        )
        primary = result["primary"]
        question_test = primary["question_sign_flip"]
        self.assertEqual(question_test["alternative"], "two-sided")
        self.assertEqual(question_test["n_questions"], 10)
        self.assertEqual(question_test["n_assignments"], 1024)
        self.assertAlmostEqual(primary["mean_difference"], 0.02553, places=5)
        direction_test = primary["question_direction_sign_test"]
        self.assertEqual(direction_test["positive_questions"], 8)
        self.assertAlmostEqual(direction_test["p_value"], 0.109375)
        model_test = primary["model_sign_flip_sensitivity"]
        self.assertEqual(model_test["unit"], "model")
        self.assertEqual(model_test["n_models"], 6)


class RunnerAndOutputTests(unittest.TestCase):
    def test_default_output_is_allowed_and_project_root_is_rejected(self) -> None:
        validate_output_target(DEFAULT_OUTPUT)
        with self.assertRaises(ValueError):
            validate_output_target(PROJECT_ROOT)

    def test_custom_output_requires_an_owned_snapshot_or_empty_directory(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            empty = root / "empty"
            empty.mkdir()
            validate_output_target(empty)

            unowned = root / "unowned"
            unowned.mkdir()
            (unowned / "notes.txt").write_text("keep", encoding="utf-8")
            with self.assertRaises(ValueError):
                validate_output_target(unowned)

            owned = root / "owned"
            owned.mkdir()
            (owned / "manifest.json").write_text(
                json.dumps(
                    {
                        "producer": RESULTS_PRODUCER,
                        "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
                        "artifacts": [],
                    }
                ),
                encoding="utf-8",
            )
            validate_output_target(owned)

    def test_owned_snapshot_edits_are_never_silently_replaced(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "owned"
            output.mkdir()
            artifact = output / "human-work.csv"
            artifact.write_text("code,notes\nretained,reviewed\n", encoding="utf-8")
            (output / "manifest.json").write_text(
                json.dumps(
                    {
                        "producer": RESULTS_PRODUCER,
                        "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
                        "artifacts": [
                            {
                                "path": artifact.name,
                                "bytes": artifact.stat().st_size,
                                "sha256": sha256_file(artifact),
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            validate_output_target(output)
            artifact.write_text("code,notes\nreversed,edited\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "edited files"):
                validate_output_target(output)

    def test_nested_csv_values_are_standards_compliant_json(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "rows.csv"
            write_rows(
                destination,
                [{"id": 1, "payload": {"b": 2, "a": [1, 3]}}],
            )
            with destination.open(encoding="utf-8", newline="") as handle:
                row = next(csv.DictReader(handle))
            self.assertEqual(
                json.loads(row["payload"]),
                {"a": [1, 3], "b": 2},
            )


if __name__ == "__main__":
    unittest.main()
