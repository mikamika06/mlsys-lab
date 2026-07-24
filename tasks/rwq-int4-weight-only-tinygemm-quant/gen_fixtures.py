"""Deterministic weight matrix for the tinygemm-style int4 weight-only quant task.
Shape (16, 512) so that with group_size=128 every row has 4 groups. One group
is forced constant to exercise the max==min (scale==0) edge case.

Run from the task directory:  python3 gen_fixtures.py
"""
import pathlib

import numpy as np

ROWS, COLS = 16, 512


def main() -> None:
    rng = np.random.default_rng(0)
    W = (rng.standard_normal((ROWS, COLS)) * 0.05).astype(np.float64)

    # Force one group to a constant value: exercises the max == min edge case.
    W[3, 128:256] = 0.007

    # A row with a wider, asymmetric range.
    W[7, :] = rng.uniform(-0.3, 0.9, size=COLS)

    out = pathlib.Path(__file__).resolve().parent / "fixtures"
    out.mkdir(exist_ok=True)
    np.save(out / "tao_w.npy", W)


if __name__ == "__main__":
    main()
