"""Deterministic Q/K/V fixture with one outlier-magnitude head, for the
per-tensor vs per-head FP8 KV scale comparison task.

Run from the task directory: python3 gen_fixtures.py
"""
import pathlib

import numpy as np


def main() -> None:
    rng = np.random.default_rng(2026)

    H, N, M, D = 6, 12, 8, 16  # heads, keys, queries, head_dim

    K = rng.normal(size=(H, N, D))
    V = rng.normal(size=(H, N, D))
    Q = rng.normal(size=(H, M, D))

    # One head with a much larger value range than the rest: per-tensor
    # scaling must set its scale from this head's amax, crushing the
    # precision available to every other head.
    outlier_head = 4
    K[outlier_head] *= 20000.0
    V[outlier_head] *= 20000.0

    out = pathlib.Path(__file__).resolve().parent / "fixtures"
    out.mkdir(exist_ok=True)
    np.save(out / "q.npy", Q)
    np.save(out / "k.npy", K)
    np.save(out / "v.npy", V)


if __name__ == "__main__":
    main()
