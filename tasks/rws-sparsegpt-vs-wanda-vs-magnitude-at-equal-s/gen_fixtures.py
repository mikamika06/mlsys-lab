"""Deterministic layer weight + correlated calibration-activation fixture
for the SparseGPT vs Wanda vs magnitude reconstruction-MSE comparison.

Run from the task directory: python3 gen_fixtures.py
"""
import pathlib

import numpy as np


def main() -> None:
    rng = np.random.default_rng(3)

    m, d, n = 10, 16, 64
    k = 5
    Z = rng.standard_normal((k, n))
    A = rng.standard_normal((k, d))
    X = A.T @ Z
    feat_scale = rng.uniform(0.2, 5.0, size=(d, 1))
    X = X * feat_scale
    X += 0.05 * rng.standard_normal((d, n))

    W = rng.standard_normal((m, d)) * rng.uniform(0.3, 1.5, size=(m, 1))

    out = pathlib.Path(__file__).resolve().parent / "fixtures"
    out.mkdir(exist_ok=True)
    np.save(out / "layer_w.npy", W)
    np.save(out / "layer_x.npy", X)


if __name__ == "__main__":
    main()
