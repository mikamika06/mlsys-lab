"""Deterministic fixtures for num-naive-vs-stable-lse-on-overflow-underflow-fixture.

Two small 1-D arrays chosen so that the NAIVE formula
``log(sum(exp(x)))`` breaks in float64:

* ``x_overflow`` — entries around 1000, so ``exp(x)`` overflows to ``inf``
  and the naive result becomes ``log(inf) = inf``.
* ``x_underflow`` — entries around -1000, so ``exp(x)`` underflows to
  ``0.0`` for every entry and the naive result becomes ``log(0) = -inf``.

The numerically stable max-shift formula handles both without difficulty.

Run with:

    python3 tasks/num-naive-vs-stable-lse-on-overflow-underflow-fixture/gen_fixtures.py
"""
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent / "fixtures"


def build():
    x_overflow = np.array([1000.0, 1000.5, 999.0, 998.2, 1000.1], dtype=np.float64)
    x_underflow = np.array([-1000.0, -1000.5, -999.0, -1001.3], dtype=np.float64)
    return x_overflow, x_underflow


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    x_overflow, x_underflow = build()
    np.save(OUT / "x_overflow.npy", x_overflow)
    np.save(OUT / "x_underflow.npy", x_underflow)
    print("wrote", x_overflow.shape, x_underflow.shape)
