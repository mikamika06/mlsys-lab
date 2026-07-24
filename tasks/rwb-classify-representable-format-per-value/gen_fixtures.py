"""Deterministic fixture: raw scalar values spanning several magnitude bands
(typical, beyond-E4M3-range, beyond-both-range, and subnormal-adjacent) plus
a handful of exact boundary values, so both formats "win" on a meaningful
share of the values and genuine round-trip-error ties occur.

Run with:
    python3 tasks/rwb-classify-representable-format-per-value/gen_fixtures.py
"""
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent / "fixtures"


def build():
    rng = np.random.default_rng(9)
    vals = []
    vals += list(rng.uniform(-2, 2, 20))
    vals += list(rng.uniform(-500, 500, 20))
    vals += list(rng.uniform(-50000, 50000, 20))
    vals += list(rng.uniform(-0.005, 0.005, 20))
    vals += [448.0, -448.0, 500.0, 0.0, 57344.0, 100000.0,
             1e-5, -1e-5, 0.0009, 0.0001]
    return np.array(vals, dtype=np.float64)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    values = build()
    np.save(OUT / "values.npy", values)
    print("wrote", values.shape)
