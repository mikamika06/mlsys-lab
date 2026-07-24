"""Deterministic fixture for rwb-preferred-batch-size-a-vs-b.

A 2000-request Poisson-ish arrival stream (cumulative exponential
inter-arrival gaps), the input the dynamic-batcher simulator consumes.

Run with:

    python3 tasks/rwb-preferred-batch-size-a-vs-b/gen_fixtures.py
"""
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent / "fixtures"


def build():
    rng = np.random.default_rng(0)
    n = 2000
    lam = 400.0  # mean arrival rate, requests/second
    inter_arrival = rng.exponential(1.0 / lam, size=n)
    arrivals = np.cumsum(inter_arrival)
    return arrivals.astype(np.float64)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    arrivals = build()
    np.save(OUT / "arrivals.npy", arrivals)
    print("wrote", arrivals.shape)
