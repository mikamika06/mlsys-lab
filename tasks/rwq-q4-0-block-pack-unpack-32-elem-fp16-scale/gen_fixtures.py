"""Deterministic 'real GGUF weight'-like 1-D stream for the Q4_0 (32-elem
block, fp16 scale) task.

Run from the task directory: python3 gen_fixtures.py
"""
import pathlib

import numpy as np


def main() -> None:
    rng = np.random.default_rng(2024)

    n_blocks = 80
    block_size = 32

    # Per-block magnitude varies (some near-flat blocks, some with a sharp
    # outlier), and the sign of the extreme element alternates naturally
    # from the random draw, exercising both signs of d = max_signed / -8.
    block_scale = rng.uniform(0.02, 5.0, size=(n_blocks, 1))
    x = (rng.standard_normal((n_blocks, block_size)) * block_scale).reshape(-1)

    out = pathlib.Path(__file__).resolve().parent / "fixtures"
    out.mkdir(exist_ok=True)
    np.save(out / "gguf_w.npy", x.astype(np.float64))


if __name__ == "__main__":
    main()
