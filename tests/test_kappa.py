"""Test the Cohen's κ helper. Interactive loop is covered manually."""
from scripts.kappa_sample import cohen_kappa


def test_kappa_perfect_agreement():
    k = cohen_kappa(["A", "B", "A", "C"], ["A", "B", "A", "C"])
    assert abs(k - 1.0) < 1e-9


def test_kappa_zero_on_independent():
    # 50/50 splits with random pairing: κ around 0.
    k = cohen_kappa(["A", "B"] * 50, ["A", "B"] * 25 + ["B", "A"] * 25)
    assert -0.1 < k < 0.1


def test_kappa_worse_than_chance_is_negative():
    k = cohen_kappa(["A", "B", "A", "B"], ["B", "A", "B", "A"])
    assert k < 0
