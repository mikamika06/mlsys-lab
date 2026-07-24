"""Deterministic fixtures for alg-subspace-block-iteration-for-top-k.

Builds a symmetric matrix ``A = U diag(lambda) U^T`` whose spectrum has a clean
gap after the top-k block, plus a fixed starting block ``Q0``. Run with:

    python3 tasks/alg-subspace-block-iteration-for-top-k/gen_fixtures.py
"""
from pathlib import Path

import numpy as np

N = 120
K = 16

OUT = Path(__file__).resolve().parent / "fixtures"


def build():
    rng = np.random.default_rng(0)
    U, _ = np.linalg.qr(rng.standard_normal((N, N)))
    lam = np.empty(N, dtype=np.float64)
    lam[:K] = 2.0 - 0.05 * np.arange(K)          # 2.00 .. 1.25, all distinct
    lam[K:] = rng.uniform(0.05, 0.25, size=N - K)  # clear spectral gap
    lam[K:] = np.sort(lam[K:])[::-1]
    A = (U * lam) @ U.T
    A = 0.5 * (A + A.T)
    Q0 = rng.standard_normal((N, K))
    return A, Q0


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    A, Q0 = build()
    np.save(OUT / "A.npy", A)
    np.save(OUT / "Q0.npy", Q0)
    print("wrote", A.shape, Q0.shape)
