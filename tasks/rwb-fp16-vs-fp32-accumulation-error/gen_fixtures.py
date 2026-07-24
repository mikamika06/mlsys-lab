"""Deterministic Q/K/V fixture for the fp16-vs-fp32 accumulation-error task.

Run from the task directory: python3 gen_fixtures.py
"""
import pathlib

import numpy as np


def main() -> None:
    rng = np.random.default_rng(0)

    n, m, d, dv = 5, 6, 16, 4  # queries, keys, head_dim, value_dim
    Q = rng.normal(scale=1.5, size=(n, d))
    K = rng.normal(scale=1.5, size=(m, d))
    V = rng.normal(scale=2.0, size=(m, dv))

    out = pathlib.Path(__file__).resolve().parent / "fixtures"
    out.mkdir(exist_ok=True)
    np.save(out / "q.npy", Q)
    np.save(out / "k.npy", K)
    np.save(out / "v.npy", V)


if __name__ == "__main__":
    main()
