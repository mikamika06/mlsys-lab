"""Deterministic fixture for
rws-expected-sparsity-retained-magnitude-of-a-pattern-distribution.

A batch of groups-of-4 weight magnitudes, each with a probability
distribution over the 6 canonical 2-of-4 "N:M sparsity" keep patterns
(the way a probabilistic / learned N:M mask-selection scheme scores which
pattern to commit to for each group before rounding to a hard decision).

Run with:
    python3 tasks/rws-expected-sparsity-retained-magnitude-of-a-pattern-distribution/gen_fixtures.py
"""
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent / "fixtures"

N_GROUPS = 50


def build():
    rng = np.random.default_rng(9)
    # Dirichlet gives valid probability rows (each sums to 1) with varied
    # concentration, so some groups are near-deterministic about their
    # pattern and others spread probability broadly.
    alpha = rng.uniform(0.3, 3.0, size=6)
    p = rng.dirichlet(alpha, size=N_GROUPS)
    w = np.abs(rng.standard_normal((N_GROUPS, 4))) * rng.uniform(0.1, 2.0, size=(N_GROUPS, 1))
    return p.astype(np.float64), w.astype(np.float64)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    p, w = build()
    np.save(OUT / "pat_p.npy", p)
    np.save(OUT / "pat_w.npy", w)
    print("wrote p", p.shape, "w", w.shape)
