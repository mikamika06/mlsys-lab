"""Deterministic 'real weight'-like blocks for the MXFP4 full-pipeline task
(block=32, shared E8M0 power-of-two scale, E2M1 elements).

Run from the task directory: python3 gen_fixtures.py
"""
import pathlib

import numpy as np


def main() -> None:
    rng = np.random.default_rng(777)

    B = 48  # number of 32-element blocks
    block_size = 32

    # Per-block magnitude varies widely (near-zero blocks, mid-range blocks,
    # large-outlier blocks) so the shared exponent differs block to block.
    block_scale = rng.uniform(0.02, 60.0, size=(B, 1))
    x = rng.normal(size=(B, block_size)) * block_scale

    # A handful of hand-placed rows exercising edge cases: an all-zero block,
    # a block whose max magnitude lands exactly on a power-of-two boundary
    # (12.0 -> ratio 2.0 -> ceil(log2)=1 exactly, no float fuzz), and a
    # block with a single huge outlier next to otherwise-tiny values.
    x[0, :] = 0.0
    x[1, :] = 0.0
    x[1, 3] = 12.0
    x[2, :] = 0.01
    x[2, 5] = 59.0

    out = pathlib.Path(__file__).resolve().parent / "fixtures"
    out.mkdir(exist_ok=True)
    np.save(out / "mx_w.npy", x)


if __name__ == "__main__":
    main()
