"""A deterministic weight matrix that already satisfies 2:4 structured
sparsity exactly (2 nonzeros per group of 4 columns, every row) plus an
input matrix, for the dense-vs-compressed-matmul numeric-equality task.

Run from the task directory: python3 gen_fixtures.py
"""
import pathlib

import numpy as np


def main() -> None:
    rng = np.random.default_rng(31)

    m, n, p = 20, 32, 5
    W = np.zeros((m, n), dtype=np.float64)
    for row in range(m):
        for g in range(n // 4):
            pos = rng.choice(4, size=2, replace=False)
            vals = rng.standard_normal(2) * rng.uniform(0.3, 3.0)
            W[row, g * 4 + pos] = vals

    X = rng.standard_normal((n, p))

    out = pathlib.Path(__file__).resolve().parent / "fixtures"
    out.mkdir(exist_ok=True)
    np.save(out / "w24.npy", W)
    np.save(out / "x.npy", X)


if __name__ == "__main__":
    main()
