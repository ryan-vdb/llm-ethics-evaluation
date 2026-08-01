"""Human-review exemplars and blinded conclusion-coding templates."""

from __future__ import annotations

import re

import numpy as np

from ..tools.data import IntegrityDataset


RETAIN_MARKERS = (
    "same conclusion",
    "maintain my conclusion",
    "remain convinced",
    "still believe",
    "would not change",
    "do not change",
)
CHANGE_MARKERS = (
    "change my conclusion",
    "reverse my conclusion",
    "no longer believe",
    "i would instead",
    "my revised conclusion",
)


def _excerpt(text: str, limit: int = 420) -> str:
    compact = re.sub(r"\s+", " ", text).strip()
    return compact if len(compact) <= limit else compact[: limit - 1] + "…"


def _surface_flags(text: str) -> dict[str, bool]:
    lowered = text.lower()
    return {
        "contains_retention_phrase": any(
            marker in lowered for marker in RETAIN_MARKERS
        ),
        "contains_change_phrase": any(
            marker in lowered for marker in CHANGE_MARKERS
        ),
    }


def run_exemplar_analysis(dataset: IntegrityDataset) -> dict[str, object]:
    """Select transparent high/low movement examples for human review."""

    conditions = list(dataset.conditions)
    initial_index = conditions.index("initial")
    rows = []
    for condition_index, condition in enumerate(conditions):
        if condition == "initial":
            continue
        candidates = []
        for model_index, model in enumerate(dataset.models):
            for question_index, question_id in enumerate(dataset.question_ids):
                retention = float(
                    np.dot(
                        dataset.residuals[
                            model_index, question_index, initial_index
                        ],
                        dataset.residuals[
                            model_index, question_index, condition_index
                        ],
                    )
                )
                raw_retention = float(
                    np.dot(
                        dataset.raw_responses[
                            model_index, question_index, initial_index
                        ],
                        dataset.raw_responses[
                            model_index, question_index, condition_index
                        ],
                    )
                )
                followup = str(
                    dataset.response_texts[
                        model_index, question_index, condition_index
                    ]
                )
                candidates.append(
                    {
                        "model": model,
                        "question_id": int(question_id),
                        "condition": condition,
                        "domain": dataset.domains[question_index],
                        "hidden_conflict": dataset.conflicts[question_index],
                        "semantic_revision": 1.0 - retention,
                        "raw_revision": 1.0 - raw_retention,
                        "initial_excerpt": _excerpt(
                            str(
                                dataset.response_texts[
                                    model_index,
                                    question_index,
                                    initial_index,
                                ]
                            )
                        ),
                        "followup_excerpt": _excerpt(followup),
                        **_surface_flags(followup),
                    }
                )
        ordered = sorted(
            candidates,
            key=lambda row: row["semantic_revision"],
        )
        for label, selected in (
            ("most_stable", ordered[:2]),
            ("largest_revision", ordered[-2:][::-1]),
        ):
            for rank, row in enumerate(selected, start=1):
                rows.append({"selection": label, "rank": rank, **row})

    return {
        "method": "extreme-pair review with unvalidated surface phrase flags",
        "exemplars": rows,
        "coding_status": "awaiting metadata-masked human coding",
        "recommended_conclusion_codes": [
            "retained",
            "refined_or_qualified",
            "reversed",
            "unclear",
        ],
        "interpretation_boundary": (
            "Large embedding movement can occur when a response explicitly "
            "defends the same conclusion using new meta-level language. Phrase "
            "flags are search aids only and are not stance classifications."
        ),
    }


def build_blinded_coding_tables(
    dataset: IntegrityDataset,
    *,
    random_state: int = 42,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Return a randomized coding sheet and its model/condition key."""

    conditions = list(dataset.conditions)
    initial_index = conditions.index("initial")
    items = []
    for model_index, model in enumerate(dataset.models):
        for question_index, question_id in enumerate(dataset.question_ids):
            for condition_index, condition in enumerate(conditions):
                if condition == "initial":
                    continue
                items.append(
                    {
                        "model": model,
                        "question_id": int(question_id),
                        "condition": condition,
                        "question_text": dataset.question_texts[question_index],
                        "initial_response": str(
                            dataset.response_texts[
                                model_index,
                                question_index,
                                initial_index,
                            ]
                        ),
                        "followup_response": str(
                            dataset.response_texts[
                                model_index,
                                question_index,
                                condition_index,
                            ]
                        ),
                    }
                )
    order = np.random.default_rng(random_state).permutation(len(items))
    sheet = []
    key = []
    for sequence, source_index in enumerate(order, start=1):
        item = items[int(source_index)]
        item_id = f"INT-{sequence:03d}"
        sheet.append(
            {
                "item_id": item_id,
                "question_id": item["question_id"],
                "question_text": item["question_text"],
                "initial_response": item["initial_response"],
                "followup_response": item["followup_response"],
                "conclusion_code": "",
                "reasoning_code": "",
                "coder_confidence": "",
                "coder_notes": "",
            }
        )
        key.append(
            {
                "item_id": item_id,
                "model": item["model"],
                "question_id": item["question_id"],
                "condition": item["condition"],
            }
        )
    return sheet, key
