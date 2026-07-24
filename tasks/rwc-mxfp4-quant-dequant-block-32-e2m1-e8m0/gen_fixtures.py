"""Deterministic 'real weight'-like blocks for the MXFP4 (block=32,
e2m1 + shared E8M0 scale) quant/dequant task.

Run from the task directory: python3 gen_fixtures.py
"""
import pathlib

import numpy as np


def main() -> None:
    rng = np.random.default_rng(555)

    B = 50  # number of 32-element blocks
    block_size = 32

    # Per-block magnitude varies a lot (some near-zero blocks, some with
    # large outlier weights) so the shared exponent differs across blocks.
    block_scale = rng.uniform(0.05, 40.0, size=(B, 1))
    x = rng.normal(size=(B, block_size)) * block_scale

    out = pathlib.Path(__file__).resolve().parent / "fixtures"
    out.mkdir(exist_ok=True)
    np.save(out / "weights.npy", x)


if __name__ == "__main__":
    main()
