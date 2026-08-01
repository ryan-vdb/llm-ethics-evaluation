"""Serialization helpers for generated integrity-analysis artifacts."""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np


def serializable(value):
    """Convert numpy-heavy results to standards-compliant JSON values."""

    if is_dataclass(value) and not isinstance(value, type):
        return serializable(asdict(value))
    if isinstance(value, np.ndarray):
        return [serializable(item) for item in value.tolist()]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [serializable(item) for item in value]
    return value


def write_json(path: str | Path, value: object) -> Path:
    """Write stable, readable JSON and create its parent directory."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(serializable(value), indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    return destination


def write_rows(path: str | Path, rows: Iterable[Mapping[str, object]]) -> Path | None:
    """Write dictionaries to CSV, returning ``None`` for an empty table."""

    materialized = [dict(row) for row in rows]
    if not materialized:
        return None
    fields = list(materialized[0])
    if any(set(row) != set(fields) for row in materialized):
        raise ValueError("all CSV rows must have the same fields")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(
            {
                key: (
                    json.dumps(converted, ensure_ascii=False, sort_keys=True)
                    if isinstance(converted := serializable(value), (dict, list))
                    else converted
                )
                for key, value in row.items()
            }
            for row in materialized
        )
    return destination


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 digest of one artifact."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
