"""Deterministic fixture for rwb-mean-added-queue-latency.

A 40-request arrival stream (non-decreasing integer timestamps, some
ties), the same shape of input the dynamic-batcher event simulator uses.

Run with:

    python3 tasks/rwb-mean-added-queue-latency/gen_fixtures.py
"""
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent / "fixtures"


def build():
    rng = np.random.default_rng(55)
    n = 40
    gaps = rng.integers(0, 7, size=n)
    arrivals = np.cumsum(gaps)
    return arrivals.astype(np.int64)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    arrivals = build()
    np.save(OUT / "arrivals.npy", arrivals)
    print("wrote", arrivals.shape)
