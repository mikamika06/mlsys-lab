import sys

sys.path.insert(0, ".")
import numpy as np

from scoremod import alibi_slopes, softcap_backward, softcap_forward

N_HEADS_CASES = [1, 2, 3, 5, 8, 12, 16, 24]
POWER_OF_2_CASES = [1, 2, 4, 8, 16]
CAPS = [1.0, 5.0, 20.0]


def test_alibi_slopes_shape_and_bounds():
    for n in N_HEADS_CASES:
        slopes = np.asarray(alibi_slopes(n), dtype=np.float64)
        assert slopes.shape == (n,), f"n_heads={n}: got shape {slopes.shape}"
        assert np.all(slopes > 0.0), f"n_heads={n}: found non-positive slope {slopes}"
        assert np.all(slopes <= 1.0 + 1e-9), f"n_heads={n}: found slope above 1: {slopes}"


def test_alibi_slopes_are_distinct_per_head():
    for n in N_HEADS_CASES:
        slopes = np.asarray(alibi_slopes(n), dtype=np.float64)
        rounded = [round(float(v), 9) for v in slopes]
        assert len(set(rounded)) == n, f"n_heads={n}: duplicate slopes assigned to different heads: {slopes}"


def test_alibi_slopes_strictly_decrease_for_power_of_two_heads():
    for n in POWER_OF_2_CASES:
        slopes = np.asarray(alibi_slopes(n), dtype=np.float64)
        if slopes.size < 2:
            continue
        assert np.all(np.diff(slopes) < 0.0), f"n_heads={n}: slopes not strictly decreasing: {slopes}"


def test_softcap_forward_never_exceeds_cap():
    x = np.linspace(-200.0, 200.0, 401)
    for cap in CAPS:
        y = np.asarray(softcap_forward(x, cap), dtype=np.float64)
        assert np.all(np.abs(y) <= cap + 1e-9), f"cap={cap}: softcap exceeded cap, max={np.max(np.abs(y))}"


def test_softcap_backward_factor_stays_in_unit_interval():
    x = np.linspace(-10.0, 10.0, 101)
    for cap in CAPS:
        grad = np.asarray(softcap_backward(np.ones_like(x), x, cap), dtype=np.float64)
        assert np.all(grad > 0.0), f"cap={cap}: non-positive gradient factor {grad.min()}"
        assert np.all(grad <= 1.0 + 1e-9), f"cap={cap}: gradient factor exceeds 1: {grad.max()}"
