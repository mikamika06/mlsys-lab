"""Deterministic fixture for rws-pca-slice-error-equals-tail-eigenvalue-sum.

A data matrix X (80 rows, 12 columns) built from a fixed random orthogonal
basis with a designed, unequal eigenvalue spectrum (so the top-k / tail
split is meaningful, not an arbitrary random matrix), plus a keep-count k.

Run with:
    python3 tasks/rws-pca-slice-error-equals-tail-eigenvalue-sum/gen_fixtures.py
"""
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent / "fixtures"

N, D = 80, 12
K = 5


def build():
    rng = np.random.default_rng(4)
    Q, _ = np.linalg.qr(rng.standard_normal((D, D)))
    scales = np.array([8.0, 6.0, 4.5, 3.0, 2.0, 1.2, 0.7, 0.4, 0.25, 0.15, 0.08, 0.03])
    Z = rng.standard_normal((N, D))
    X = (Z * scales) @ Q.T
    return X.astype(np.float64)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    X = build()
    np.save(OUT / "pca_x.npy", X)
    np.save(OUT / "pca_k.npy", np.array(K, dtype=np.int64))
    print("wrote X", X.shape, "k", K)
