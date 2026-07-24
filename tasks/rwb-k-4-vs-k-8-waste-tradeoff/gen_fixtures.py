"""Deterministic fixtures for rwb-k-4-vs-k-8-waste-tradeoff.

A synthetic "observed request size" histogram: 20 distinct sizes with
skewed counts (mimicking a real sequence-length / batch-size
distribution), stored as two parallel arrays. 20 distinct sizes is
comfortably more than both K=4 and K=8, so both bucket budgets leave real
(and different) padding waste to compare.

Run with:

    python3 tasks/rwb-k-4-vs-k-8-waste-tradeoff/gen_fixtures.py
"""
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent / "fixtures"


def build():
    rng = np.random.default_rng(11)
    sizes = sorted(set(rng.integers(1, 600, size=40).tolist()))[:20]
    counts = rng.integers(1, 50, size=len(sizes)).tolist()
    return np.array(sizes, dtype=np.int64), np.array(counts, dtype=np.int64)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    sizes, counts = build()
    np.save(OUT / "sizes.npy", sizes)
    np.save(OUT / "counts.npy", counts)
    print("wrote", sizes.shape, counts.shape)
