"""Batch CSV export shared by the napari widgets (no Qt dependency, so unit-testable)."""
from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import pandas as pd


def append_batch_csv(csv_path, row: Mapping[str, object], columns: Sequence[str]) -> Path:
    """Append ``row`` to the batch CSV at ``csv_path`` and return the path written.

    The file gets a ``.csv`` suffix if it has none. ``columns`` come first, in that
    order; columns that an existing file already has are kept after them. An
    existing file that cannot be read raises instead of being overwritten.
    """
    csv_path = Path(csv_path)
    if csv_path.suffix.lower() != ".csv":
        csv_path = csv_path.with_suffix(".csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    table = pd.DataFrame([{k: "" if v is None else str(v) for k, v in row.items()}])
    if csv_path.exists() and csv_path.stat().st_size > 0:
        existing = pd.read_csv(csv_path, dtype=str).fillna("")
        table = pd.concat([existing, table], ignore_index=True)
    extra = [c for c in table.columns if c not in columns]
    table.reindex(columns=list(columns) + extra, fill_value="").fillna("").to_csv(csv_path, index=False)
    return csv_path
