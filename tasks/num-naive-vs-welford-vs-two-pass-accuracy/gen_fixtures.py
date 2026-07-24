"""Deterministic fixture: unit-variance data offset by a large mean (1e8).

Run with:
    python3 tasks/num-naive-vs-welford-vs-two-pass-accuracy/gen_fixtures.py
"""
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent / "fixtures"


def build():
    rng = np.random.default_rng(42)
    n = 20000
    x = (rng.standard_normal(n) + 1e8).astype(np.float64)
    return x


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    x = build()
    np.save(OUT / "x.npy", x)
    print("wrote", x.shape)
