"""Arrival stream with a mix of bursts and sparse tails, so the achieved
batch size (and therefore throughput) is genuinely sensitive to the
delay cap.

Run from the task directory:  python3 gen_fixtures.py
"""
import pathlib

import numpy as np


def main() -> None:
    arrivals = np.array(
        [0, 0, 1, 2, 3, 20, 21, 40, 41, 42, 60, 100, 101, 102, 103, 104, 105, 150],
        dtype=np.int64,
    )

    out = pathlib.Path(__file__).resolve().parent / "fixtures"
    out.mkdir(exist_ok=True)
    np.save(out / "arrivals.npy", arrivals)


if __name__ == "__main__":
    main()
