"""Deterministic arrival stream: a mix of bursts (fill batches by size)
and sparse tails (only flush via the delay cap), so p99 latency is
genuinely sensitive to the cap.

Run from the task directory:  python3 gen_fixtures.py
"""
import pathlib

import numpy as np


def main() -> None:
    arrivals = np.array(
        [0, 1, 1, 2, 10, 11, 12, 13, 30, 45, 46, 70, 71, 72, 73, 74, 100, 140, 141, 200],
        dtype=np.int64,
    )

    out = pathlib.Path(__file__).resolve().parent / "fixtures"
    out.mkdir(exist_ok=True)
    np.save(out / "arrivals.npy", arrivals)


if __name__ == "__main__":
    main()
