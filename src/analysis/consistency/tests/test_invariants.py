"""Fast invariant checks for canonical preprocessing and shared statistics."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from ..runner import (
    DEFAULT_OUTPUT,
    MANIFEST_SCHEMA_VERSION,
    PROJECT_ROOT,
    RESULTS_PRODUCER,
    validate_output_target,
)
from ..tools.data import MODELS, load_framework_dataset
from ..tools.geometry import (
    cosine_matrix,
    cross_topic_mask,
    remove_question_projection,
)
from ..tools.statistics import benjamini_hochberg, holm_bonferroni


class GeometryTests(unittest.TestCase):
    def test_projection_is_exact(self) -> None:
        questions = np.asarray([[1.0, 0.0], [1.0, 1.0]])
        answers = np.asarray([[3.0, 4.0], [2.0, -1.0]])
        residuals = remove_question_projection(answers, questions)
        paired_dot = np.sum(residuals * questions, axis=1)
        np.testing.assert_allclose(paired_dot, 0.0, atol=1e-12)

    def test_database_alignment_and_primary_mask(self) -> None:
        dataset = load_framework_dataset()
        self.assertEqual(len(dataset.question_ids), 93)
        self.assertEqual(set(dataset.residuals), set(MODELS))
        max_post = max(
            float(np.max(np.abs(dataset.residual_question_cosines[model])))
            for model in MODELS
        )
        self.assertLess(max_post, 1e-10)
        mask, cutoff = cross_topic_mask(
            dataset,
            question_similarity_quantile=0.25,
            require_different_source=True,
        )
        self.assertEqual(int(np.sum(np.triu(mask, k=1))), 992)
        self.assertAlmostEqual(cutoff, 0.24955412720042297)
        self.assertEqual(cosine_matrix(dataset.question_embeddings).shape, (93, 93))


class AdjustmentTests(unittest.TestCase):
    def test_adjustments_are_monotonic_for_sorted_p_values(self) -> None:
        p_values = [0.001, 0.01, 0.04]
        bh = benjamini_hochberg(p_values)
        holm = holm_bonferroni(p_values)
        self.assertTrue(all(a <= b for a, b in zip(bh, bh[1:])))
        self.assertTrue(all(a <= b for a, b in zip(holm, holm[1:])))


class RunnerOutputSafetyTests(unittest.TestCase):
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
                    }
                ),
                encoding="utf-8",
            )
            validate_output_target(owned)


if __name__ == "__main__":
    unittest.main()
