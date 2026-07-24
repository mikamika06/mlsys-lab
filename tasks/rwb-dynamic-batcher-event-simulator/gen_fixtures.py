"""Deterministic arrival-timestamp stream fixture.

Run from the task directory: python3 gen_fixtures.py
"""
import pathlib

import numpy as np


def main() -> None:
    rng = np.random.default_rng(777)

    n = 40
    gaps = rng.integers(0, 8, size=n)  # inter-arrival gaps, some zero (bursts)
    arrivals = np.cumsum(gaps).astype(np.int64)

    out = pathlib.Path(__file__).resolve().parent / "fixtures"
    out.mkdir(exist_ok=True)
    np.save(out / "arrivals.npy", arrivals)


if __name__ == "__main__":
    main()
