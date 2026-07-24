"""Deterministic SPD fixture for num-cholesky-lower-factor-reconstruct-a-ll-t.

    python3 tasks/num-cholesky-lower-factor-reconstruct-a-ll-t/gen_fixtures.py
"""
from pathlib import Path

import numpy as np

N = 48
OUT = Path(__file__).resolve().parent / "fixtures"


def build():
    rng = np.random.default_rng(0)
    B = rng.standard_normal((N, N))
    A = (B @ B.T) / N + 2.0 * np.eye(N)   # symmetric positive definite, well conditioned
    return 0.5 * (A + A.T)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    A = build()
    np.save(OUT / "A.npy", A)
    print("wrote", A.shape, "cond", np.linalg.cond(A))
