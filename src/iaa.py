"""Inter-annotator agreement — Krippendorff's alpha with interval distance.

Used for RQ3 annotation validation (year-valid-from / year-valid-to annotations
on 100 passages by 2 annotators). Target: alpha >= 0.70. See tasks/plan.md §5 T4.

The heavy math lives in the `krippendorff` package; this module is a typed
wrapper that accepts Python-native `list[list[int | None]]` and provides a CLI
for CSV-based annotation files.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import krippendorff


def krippendorff_alpha_interval(ratings: Sequence[Sequence[int | float | None]]) -> float:
    """Alpha with interval distance.

    `ratings` is shape (n_annotators, n_units). Use None or np.nan for missing.
    """
    if not ratings:
        raise ValueError("ratings is empty")
    widths = {len(row) for row in ratings}
    if len(widths) != 1:
        raise ValueError(f"all annotator rows must have the same length, got {widths}")

    data = np.array(
        [[np.nan if v is None else float(v) for v in row] for row in ratings],
        dtype=float,
    )
    return float(
        krippendorff.alpha(reliability_data=data, level_of_measurement="interval")
    )


def _read_csv(path: Path) -> list[list[float | None]]:
    rows: list[list[float | None]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        for raw_row in csv.reader(f):
            row: list[float | None] = []
            for cell in raw_row:
                cell = cell.strip()
                row.append(None if cell == "" else float(cell))
            rows.append(row)
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compute Krippendorff's alpha (interval) from a CSV.",
    )
    parser.add_argument(
        "csv_path",
        type=Path,
        help="CSV where each row is one annotator and each column is one unit. Empty cells = missing.",
    )
    args = parser.parse_args(argv)
    rows = _read_csv(args.csv_path)
    alpha = krippendorff_alpha_interval(rows)
    print(f"{alpha:.6f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
