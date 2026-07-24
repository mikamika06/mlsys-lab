"""Deterministic fixture for a single decode query attending over a long
KV cache, for the two-chunk split-KV log-sum-exp merge task.

Run from the task directory: python3 gen_fixtures.py
"""
import pathlib

import numpy as np


def main() -> None:
    rng = np.random.default_rng(4242)

    d, dv, N = 32, 16, 200  # head_dim, value_dim, KV cache length

    q = rng.normal(size=(d,))
    k = rng.normal(size=(N, d))
    v = rng.normal(size=(N, dv))

    out = pathlib.Path(__file__).resolve().parent / "fixtures"
    out.mkdir(exist_ok=True)
    np.save(out / "q.npy", q)
    np.save(out / "k.npy", k)
    np.save(out / "v.npy", v)


if __name__ == "__main__":
    main()
