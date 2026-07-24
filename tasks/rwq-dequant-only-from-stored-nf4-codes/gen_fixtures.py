"""Deterministic packed NF4 (4-bit index) tensor for the dequant-only task.

Run from the task directory: python3 gen_fixtures.py
"""
import pathlib

import numpy as np


def main() -> None:
    rng = np.random.default_rng(1234)

    block_size = 64
    n_blocks = 12
    n = block_size * n_blocks

    idx = rng.integers(0, 16, size=n).astype(np.uint8)
    absmax = rng.uniform(0.01, 3.0, size=n_blocks).astype(np.float32)

    out = pathlib.Path(__file__).resolve().parent / "fixtures"
    out.mkdir(exist_ok=True)
    np.save(out / "nf4_idx.npy", idx)
    np.save(out / "nf4_absmax.npy", absmax)


if __name__ == "__main__":
    main()
