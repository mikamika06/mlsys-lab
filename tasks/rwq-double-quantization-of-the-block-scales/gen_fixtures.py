"""Deterministic fixture for rwq-double-quantization-of-the-block-scales.

Builds `nf4_absmax.npy`: the first-level per-64-weight-block absmax array
you get from NF4 block quantization of a realistic weight tensor. Run with:

    python3 tasks/rwq-double-quantization-of-the-block-scales/gen_fixtures.py
"""
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent / "fixtures"

N_BLOCKS = 8192
BLOCK1 = 64


def build():
    rng = np.random.default_rng(0)
    w = (rng.standard_normal(N_BLOCKS * BLOCK1) * 0.02).astype(np.float32)
    absmax = np.max(np.abs(w.reshape(N_BLOCKS, BLOCK1)), axis=1).astype(np.float32)
    return absmax


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    absmax = build()
    np.save(OUT / "nf4_absmax.npy", absmax)
    print("wrote", absmax.shape)
