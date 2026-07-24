"""Deterministic fixture for num-fix-bad-pivot-selection.

Builds an "ill-scaled" matrix ``A`` whose (0, 0) entry is tiny (~1e-12) while
every other entry has magnitude in [0.5, 2.0]. Gaussian elimination that keeps
row 0 as the pivot for column 0 (because it is merely nonzero, not because it
is the largest-magnitude candidate) is forced to divide by that tiny number,
producing huge multipliers and catastrophic cancellation in float64. Partial
pivoting (swap in the largest-magnitude row first) avoids this entirely.

Run with:

    python3 tasks/num-fix-bad-pivot-selection/gen_fixtures.py
"""
from pathlib import Path

import numpy as np

N = 7
TINY = 1e-12

OUT = Path(__file__).resolve().parent / "fixtures"


def build():
    rng = np.random.default_rng(0)
    signs = rng.choice([-1.0, 1.0], size=(N, N))
    A = rng.uniform(0.5, 2.0, size=(N, N)) * signs
    A[0, 0] = TINY
    return A


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    A = build()
    np.save(OUT / "A.npy", A)
    print("wrote", A.shape)
