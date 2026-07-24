"""Grader for `num-nextafter-x-inf-by-integer-bit-increment`.

Oracle: `np.nextafter(x, +inf)` (real IEEE-754 libm), compared bit-for-bit
via the uint32 view of the float32 output. Fixture mixes hand-picked
corner cases (±0, subnormals, boundary max value) with random finite
float32 bit patterns.
"""
from __future__ import annotations

import numpy as np


def _random_finite_f32(rng, n):
    """Random float32 bit patterns, excluding inf/NaN (exponent == 0xFF)."""
    out = []
    while len(out) < n:
        raw = rng.integers(0, 2 ** 32, size=n, dtype=np.int64).astype(np.uint32)
        exp = (raw >> np.uint32(23)) & np.uint32(0xFF)
        raw = raw[exp != 0xFF]
        out.extend(raw.tolist())
    return np.array(out[:n], dtype=np.uint32).view(np.float32)


def _fixture_values():
    rng = np.random.default_rng(0)
    specials = np.array([
        0.0, -0.0, 1.0, -1.0, 0.5, -0.5,
        np.finfo(np.float32).max, -np.finfo(np.float32).max,
        np.finfo(np.float32).tiny, -np.finfo(np.float32).tiny,
        np.finfo(np.float32).smallest_subnormal, -np.finfo(np.float32).smallest_subnormal,
    ], dtype=np.float32)
    random_vals = _random_finite_f32(rng, 500)
    return np.concatenate([specials, random_vals]).astype(np.float32)


def grade(sol, fx) -> dict:
    x = _fixture_values()
    expected = np.nextafter(x, np.float32(np.inf)).astype(np.float32)

    try:
        got = np.asarray(sol.next_up(x), dtype=np.float32)
    except Exception:
        return {"exact_match": 0.0}

    if got.shape != expected.shape:
        return {"exact_match": 0.0}

    got_bits = got.view(np.uint32)
    exp_bits = expected.view(np.uint32)
    return {"exact_match": float(np.mean(got_bits == exp_bits))}
