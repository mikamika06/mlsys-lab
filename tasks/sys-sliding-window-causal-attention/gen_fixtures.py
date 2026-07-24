"""Deterministic fixture: a full-matrix sliding-window causal attention
mask (n=12, window=4), used as the "materialize the whole mask" reference
that the tiled implementation must match numerically without ever
building an (n, n) mask itself.

Run from the task directory: python3 gen_fixtures.py
"""
import pathlib

import numpy as np


def main() -> None:
    n = 12
    window = 4

    rows = np.arange(n).reshape(-1, 1)
    cols = np.arange(n).reshape(1, -1)
    mask = (cols <= rows) & (rows - cols < window)  # i - w < j <= i

    out = pathlib.Path(__file__).resolve().parent / "fixtures"
    out.mkdir(exist_ok=True)
    np.save(out / "mask.npy", mask)


if __name__ == "__main__":
    main()
