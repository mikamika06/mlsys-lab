"""Deterministic fixture for rws-1d-wasserstein-via-cdf-integral.

Two empirical 1-D distributions with unequal support sizes: `w1_u.npy` (a
bimodal mixture, e.g. a weight tensor's distribution before some transform)
and `w1_v.npy` (a shifted unimodal distribution, e.g. after it) — the kind
of "did this transform shift the distribution" comparison W1 is used for,
with sample counts that deliberately do not match.

Run with:
    python3 tasks/rws-1d-wasserstein-via-cdf-integral/gen_fixtures.py
"""
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent / "fixtures"


def build():
    rng = np.random.default_rng(21)

    n_u = 137
    comp = rng.integers(0, 2, size=n_u)
    u = np.where(comp == 0, rng.normal(-1.5, 0.4, size=n_u), rng.normal(1.5, 0.5, size=n_u))

    n_v = 219
    v = rng.normal(0.3, 1.1, size=n_v)

    return u.astype(np.float64), v.astype(np.float64)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    u, v = build()
    np.save(OUT / "w1_u.npy", u)
    np.save(OUT / "w1_v.npy", v)
    print("wrote u", u.shape, "v", v.shape)
