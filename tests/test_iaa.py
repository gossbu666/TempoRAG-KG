"""Tests for src.iaa — canonical example, None handling, CSV CLI."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from src.iaa import krippendorff_alpha_interval


# Krippendorff canonical reliability data (from the fast-krippendorff reference
# implementation). Interval alpha ≈ 0.8747.
CANONICAL_DATA = [
    [None, None, None, None, None, 3, 4, 1, 2, 1, 1, 3, 3, None, 3],
    [1, None, 2, 1, 3, 3, 4, 3, None, None, None, None, None, None, None],
    [None, None, 2, 1, 3, 4, 4, None, 2, 1, 1, 3, 3, None, 4],
    [1, 1, None, None, None, None, 4, 1, 2, 1, 1, 3, 3, None, 4],
]
CANONICAL_ALPHA = 0.874702


def test_canonical_example_matches_within_tolerance() -> None:
    alpha = krippendorff_alpha_interval(CANONICAL_DATA)
    assert abs(alpha - CANONICAL_ALPHA) < 0.005


def test_perfect_agreement_is_one() -> None:
    data = [[1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5]]
    assert krippendorff_alpha_interval(data) == pytest.approx(1.0)


def test_none_and_nan_are_equivalent() -> None:
    with_none = CANONICAL_DATA
    with_nan = [[np.nan if v is None else v for v in row] for row in CANONICAL_DATA]
    a1 = krippendorff_alpha_interval(with_none)
    a2 = krippendorff_alpha_interval(with_nan)
    assert a1 == pytest.approx(a2, abs=1e-9)


def test_uneven_rows_raises() -> None:
    with pytest.raises(ValueError):
        krippendorff_alpha_interval([[1, 2, 3], [1, 2]])


def test_empty_raises() -> None:
    with pytest.raises(ValueError):
        krippendorff_alpha_interval([])


def test_cli_reads_csv_and_prints_alpha(tmp_path: Path) -> None:
    csv_path = tmp_path / "ann.csv"
    lines = []
    for row in CANONICAL_DATA:
        lines.append(",".join("" if v is None else str(v) for v in row))
    csv_path.write_text("\n".join(lines), encoding="utf-8")

    proc = subprocess.run(
        [sys.executable, "-m", "src.iaa", str(csv_path)],
        capture_output=True,
        text=True,
        check=True,
    )
    printed = float(proc.stdout.strip())
    assert abs(printed - CANONICAL_ALPHA) < 0.005
