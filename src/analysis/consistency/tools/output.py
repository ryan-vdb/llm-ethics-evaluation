"""Serialization helpers for generated analysis artifacts."""

from __future__ import annotations

import numpy as np


def serializable(value):
    """Convert numpy-heavy results to standards-compliant JSON values."""

    if isinstance(value, np.ndarray):
        return [serializable(item) for item in value.tolist()]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, dict):
        return {str(key): serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [serializable(item) for item in value]
    return value
