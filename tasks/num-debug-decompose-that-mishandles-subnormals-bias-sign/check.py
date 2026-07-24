"""Grader for `num-debug-decompose-that-mishandles-subnormals-bias-sign`.

The oracle is real IEEE-754 binary32: the fixture is built from raw 32-bit
patterns, and the reference (sign, exponent, significand) triple is re-derived
here from those same bits with NumPy. Nothing is hardcoded.
"""
from __future__ import annotations

import numpy as np

from mlsys import scorers

_MANT_BITS = 23
_BIAS = 127


def _fixture() -> np.ndarray:
    """Subnormal-heavy finite float32 vector (no inf, no NaN)."""
    rng = np.random.default_rng(0)
    n = 4096

    kind = rng.integers(0, 10, size=n)
    raw_exp = np.where(kind < 5,
                       np.uint32(0),                                   # subnormal / zero
                       rng.integers(1, 255, size=n, dtype=np.uint32))  # normal
    mant = rng.integers(0, 1 << _MANT_BITS, size=n, dtype=np.uint32)
    mant = np.where(kind < 1, np.uint32(0), mant)                      # signed zeros
    sign = rng.integers(0, 2, size=n, dtype=np.uint32)

    bits = (sign << 31) | (raw_exp << _MANT_BITS) | mant
    return bits.astype(np.uint32).view(np.float32)


def _oracle(x: np.ndarray):
    """Reference decomposition straight from the bit pattern."""
    b = np.asarray(x, dtype=np.float32).view(np.uint32)
    sign = (b >> 31).astype(np.int64)
    raw_exp = ((b >> _MANT_BITS) & np.uint32(0xFF)).astype(np.int64)
    mant = (b & np.uint32((1 << _MANT_BITS) - 1)).astype(np.int64)

    is_sub = raw_exp == 0
    exponent = np.where(is_sub, np.int64(1 - _BIAS), raw_exp - _BIAS)
    lead = np.where(is_sub, 0.0, 1.0)
    significand = lead + mant.astype(np.float64) / float(1 << _MANT_BITS)
    return sign, exponent, significand


def _oracle_pack(sign, exponent, significand) -> np.ndarray:
    v = (-1.0) ** np.asarray(sign, dtype=np.float64)
    v = v * np.asarray(significand, dtype=np.float64) * np.exp2(
        np.asarray(exponent, dtype=np.float64))
    return v.astype(np.float32)


def grade(sol, fx) -> dict:
    x = _fixture()
    o_sign, o_exp, o_sig = _oracle(x)

    out = {
        "byte_exact_fraction": 0.0,
        "field_exact_fraction": 0.0,
        "recompose_byte_exact_fraction": 0.0,
    }

    # --- 1. the candidate's own decompose must match the bit-level oracle -----
    try:
        s, e, m = sol.decompose(x.copy())
        s = np.asarray(s).astype(np.int64)
        e = np.asarray(e).astype(np.int64)
        m = np.asarray(m).astype(np.float64)
    except Exception:
        return out

    if s.shape == o_sign.shape and e.shape == o_exp.shape and m.shape == o_sig.shape:
        hits = int(np.sum(s == o_sign) + np.sum(e == o_exp) + np.sum(m == o_sig))
        out["field_exact_fraction"] = hits / float(3 * x.size)

    # --- 2. the candidate's recompose fed with ORACLE fields ------------------
    try:
        r = sol.recompose(o_sign.copy(), o_exp.copy(), o_sig.copy())
        r = np.asarray(r)
        if r.dtype == np.float32:
            out["recompose_byte_exact_fraction"] = scorers.byte_exact_fraction(x, r)
    except Exception:
        pass

    # --- 3. full round trip through the candidate's own pair ------------------
    try:
        rs, re_, rm = sol.decompose(x.copy())
        y = np.asarray(sol.recompose(rs, re_, rm))
        if y.dtype == np.float32:
            out["byte_exact_fraction"] = scorers.byte_exact_fraction(x, y)
    except Exception:
        pass

    # sanity: the oracle round trip must itself be byte exact
    assert scorers.byte_exact_fraction(x, _oracle_pack(o_sign, o_exp, o_sig)) == 1.0

    return out
