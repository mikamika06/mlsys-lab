"""Deterministic fixture for num-off-diagonal-norm-decreases-jacobi.

A random symmetric 6x6 matrix. The classical cyclic Jacobi eigenvalue
algorithm should drive its off-diagonal Frobenius norm monotonically toward
zero, sweep after sweep.

Run with:

    python3 tasks/num-off-diagonal-norm-decreases-jacobi/gen_fixtures.py
"""
from pathlib import Path

import numpy as np

N = 6
OUT = Path(__file__).resolve().parent / "fixtures"


def build():
    rng = np.random.default_rng(0)
    B = rng.standard_normal((N, N))
    A = 0.5 * (B + B.T)
    return A


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    A = build()
    np.save(OUT / "A.npy", A)
    print("wrote", A.shape)
