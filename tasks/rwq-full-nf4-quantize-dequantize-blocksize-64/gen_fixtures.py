"""Deterministic 1D weight vector for the full NF4 quantize/dequantize task,
covering typical small-magnitude weight blocks, a larger-magnitude block, a
uniform block, and one all-zero block (the absmax==0 edge case).

Run from the task directory:  python3 gen_fixtures.py
"""
import pathlib

import numpy as np

BLOCK = 64


def main() -> None:
    rng = np.random.default_rng(0)

    parts = [
        rng.standard_normal(BLOCK * 40) * 0.02,   # typical small NN weights
        np.zeros(BLOCK),                           # all-zero block (absmax == 0)
        rng.standard_normal(BLOCK * 10) * 0.5,      # larger-magnitude block
        rng.uniform(-1.0, 1.0, size=BLOCK * 5),     # near-full-range block
    ]
    w = np.concatenate(parts).astype(np.float64)

    out = pathlib.Path(__file__).resolve().parent / "fixtures"
    out.mkdir(exist_ok=True)
    np.save(out / "nf4_w.npy", w)


if __name__ == "__main__":
    main()
