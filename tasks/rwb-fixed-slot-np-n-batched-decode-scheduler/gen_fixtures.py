"""Deterministic request stream for the fixed-slot scheduler: staggered
arrivals with a mix of short and long generations, so slots free up and
get re-admitted at different times.

Run from the task directory:  python3 gen_fixtures.py
"""
import pathlib

import numpy as np


def main() -> None:
    arrivals = np.array([0, 0, 0, 1, 2, 2, 5, 6, 6, 6], dtype=np.int64)
    gen_lens = np.array([3, 5, 1, 2, 4, 1, 2, 3, 1, 6], dtype=np.int64)

    out = pathlib.Path(__file__).resolve().parent / "fixtures"
    out.mkdir(exist_ok=True)
    np.save(out / "arrivals.npy", arrivals)
    np.save(out / "gen_lens.npy", gen_lens)


if __name__ == "__main__":
    main()
