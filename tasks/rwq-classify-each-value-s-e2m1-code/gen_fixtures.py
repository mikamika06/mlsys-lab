"""Deterministic fixture for rwq-classify-each-value-s-e2m1-code.

Builds `fp4_x.npy`: a 1D float64 array mixing exact E2M1 grid points, exact
midpoints between adjacent grid points (round-to-nearest tie cases),
saturating out-of-range magnitudes, zero, and random continuous values --
each with both signs. Run with:

    python3 tasks/rwq-classify-each-value-s-e2m1-code/gen_fixtures.py
"""
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent / "fixtures"

GRID = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


def build():
    rng = np.random.default_rng(0)

    exact = GRID.copy()
    midpoints = (GRID[:-1] + GRID[1:]) / 2.0  # 0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0
    saturating = np.array([6.5, 9.0, 100.0, 1e6])
    tiny = np.array([1e-9, 0.01, 0.24])
    random_vals = rng.uniform(0.0, 8.0, size=200)

    magnitudes = np.concatenate([exact, midpoints, saturating, tiny, random_vals])
    signs = rng.choice([-1.0, 1.0], size=magnitudes.shape[0])
    x = (magnitudes * signs).astype(np.float64)
    rng.shuffle(x)
    return x


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    x = build()
    np.save(OUT / "fp4_x.npy", x)
    print("wrote", x.shape)
