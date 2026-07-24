"""Deterministic arrival stream that exercises both triggers: several
bursts big enough to hit the preferred batch size immediately, and a
sparse trailing tail that can only ever flush via the delay cap.

Run from the task directory:  python3 gen_fixtures.py
"""
import pathlib

import numpy as np


def main() -> None:
    arrivals = np.array(
        [0, 0, 0, 5, 9, 20, 21, 22, 23, 50, 51, 90],
        dtype=np.int64,
    )

    out = pathlib.Path(__file__).resolve().parent / "fixtures"
    out.mkdir(exist_ok=True)
    np.save(out / "arrivals.npy", arrivals)


if __name__ == "__main__":
    main()
